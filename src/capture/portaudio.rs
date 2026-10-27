use super::{CaptureGate, INPUT_BLOCK_FRAMES, INPUT_SAMPLE_RATE};
use anyhow::{Context, Result, bail};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_int, c_ulong, c_void},
    path::{Path, PathBuf},
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[repr(C)]
struct DeviceInfo {
    version: c_int,
    name: *const c_char,
    host_api: c_int,
    max_input_channels: c_int,
    max_output_channels: c_int,
    default_low_input_latency: f64,
    default_low_output_latency: f64,
    default_high_input_latency: f64,
    default_high_output_latency: f64,
    default_sample_rate: f64,
}

#[repr(C)]
struct StreamParameters {
    device: c_int,
    channel_count: c_int,
    sample_format: c_ulong,
    suggested_latency: f64,
    host_api_specific_stream_info: *mut c_void,
}

type Callback = unsafe extern "C" fn(
    *const c_void,
    *mut c_void,
    c_ulong,
    *const c_void,
    c_ulong,
    *mut c_void,
) -> c_int;
type Initialize = unsafe extern "C" fn() -> c_int;
type Terminate = unsafe extern "C" fn() -> c_int;
type DefaultInput = unsafe extern "C" fn() -> c_int;
type GetDeviceInfo = unsafe extern "C" fn(c_int) -> *const DeviceInfo;
type ErrorText = unsafe extern "C" fn(c_int) -> *const c_char;
type OpenStream = unsafe extern "C" fn(
    *mut *mut c_void,
    *const StreamParameters,
    *const StreamParameters,
    f64,
    c_ulong,
    c_ulong,
    Option<Callback>,
    *mut c_void,
) -> c_int;
type StreamFunction = unsafe extern "C" fn(*mut c_void) -> c_int;
type SetFinished =
    unsafe extern "C" fn(*mut c_void, Option<unsafe extern "C" fn(*mut c_void)>) -> c_int;

struct Runtime {
    _library: Library,
    terminate: Terminate,
    close: StreamFunction,
    abort: StreamFunction,
    error_text: ErrorText,
}

impl Runtime {
    fn error(&self, code: c_int) -> anyhow::Error {
        // Pa_GetErrorText returns a static C string owned by the loaded DLL.
        let message = unsafe {
            let text = (self.error_text)(code);
            if text.is_null() {
                format!("PortAudio error {code}")
            } else {
                CStr::from_ptr(text).to_string_lossy().into_owned()
            }
        };
        anyhow::anyhow!("{message} ({code})")
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe {
            (self.terminate)();
        }
    }
}

struct CallbackContext {
    gate: Arc<Mutex<CaptureGate>>,
    channels: usize,
    commands: Sender<Command>,
}

unsafe extern "C" fn finished(userdata: *mut c_void) {
    let context = unsafe { &*(userdata as *const CallbackContext) };
    let _ = context.commands.send(Command::Finished);
}

unsafe extern "C" fn callback(
    input: *const c_void,
    _output: *mut c_void,
    frames: c_ulong,
    _time: *const c_void,
    flags: c_ulong,
    userdata: *mut c_void,
) -> c_int {
    // The boxed context outlives Pa_CloseStream, which joins its callbacks.
    let context = unsafe { &*(userdata as *const CallbackContext) };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if let Ok(mut gate) = context.gate.lock() {
            if flags & 2 != 0 {
                gate.fail("麦克风输入缓冲溢出，已取消本次录音，请检查设备或系统负载".into());
            }
            if !input.is_null() {
                let samples = unsafe {
                    std::slice::from_raw_parts(
                        input.cast::<f32>(),
                        frames as usize * context.channels,
                    )
                };
                gate.accept(samples, context.channels);
            }
        }
    }));
    if outcome.is_ok() { 0 } else { 2 } // paContinue / paAbort
}

pub(super) struct Stream {
    runtime: Runtime,
    stream: *mut c_void,
    _context: Box<CallbackContext>,
    active: StreamFunction,
}

impl Stream {
    fn open(dll: &Path, gate: Arc<Mutex<CaptureGate>>, commands: Sender<Command>) -> Result<Self> {
        if !dll.is_absolute() || !dll.is_file() {
            bail!("PortAudio DLL is missing: {}", dll.display());
        }
        // Absolute-path loading avoids searching the working directory.
        unsafe {
            let library = Library::new(dll).with_context(|| format!("Load {}", dll.display()))?;
            let initialize: Initialize = *library.get(b"Pa_Initialize\0")?;
            let terminate: Terminate = *library.get(b"Pa_Terminate\0")?;
            let default_input: DefaultInput = *library.get(b"Pa_GetDefaultInputDevice\0")?;
            let device_info: GetDeviceInfo = *library.get(b"Pa_GetDeviceInfo\0")?;
            let open: OpenStream = *library.get(b"Pa_OpenStream\0")?;
            let start: StreamFunction = *library.get(b"Pa_StartStream\0")?;
            let close: StreamFunction = *library.get(b"Pa_CloseStream\0")?;
            let abort: StreamFunction = *library.get(b"Pa_AbortStream\0")?;
            let error_text: ErrorText = *library.get(b"Pa_GetErrorText\0")?;
            let active: StreamFunction = *library.get(b"Pa_IsStreamActive\0")?;
            let set_finished: SetFinished = *library.get(b"Pa_SetStreamFinishedCallback\0")?;
            let status = initialize();
            if status != 0 {
                bail!("Pa_Initialize failed ({status})");
            }
            let runtime = Runtime {
                _library: library,
                terminate,
                close,
                abort,
                error_text,
            };
            let device = default_input();
            if device < 0 {
                bail!("未找到默认麦克风设备");
            }
            let info = device_info(device);
            if info.is_null() {
                bail!("Pa_GetDeviceInfo returned a null device");
            }
            let channels = (*info).max_input_channels.clamp(0, 2) as usize;
            if channels == 0 {
                bail!("默认设备没有输入声道");
            }
            let parameters = StreamParameters {
                device,
                channel_count: channels as c_int,
                sample_format: 1,
                suggested_latency: (*info).default_high_input_latency,
                host_api_specific_stream_info: ptr::null_mut(),
            };
            let mut context = Box::new(CallbackContext {
                gate,
                channels,
                commands,
            });
            let mut stream = ptr::null_mut();
            let status = open(
                &mut stream,
                &parameters,
                ptr::null(),
                INPUT_SAMPLE_RATE as f64,
                INPUT_BLOCK_FRAMES as c_ulong,
                0,
                Some(callback),
                (&mut *context as *mut CallbackContext).cast(),
            );
            if status != 0 {
                return Err(runtime.error(status));
            }
            if stream.is_null() {
                bail!("Pa_OpenStream returned a null stream");
            }
            let status = set_finished(stream, Some(finished));
            if status != 0 {
                close(stream);
                return Err(runtime.error(status));
            }
            let status = start(stream);
            if status != 0 {
                close(stream);
                return Err(runtime.error(status));
            }
            Ok(Self {
                runtime,
                stream,
                _context: context,
                active,
            })
        }
    }
}

trait ManagedStream {
    fn is_active(&self) -> bool;
}
impl ManagedStream for Stream {
    fn is_active(&self) -> bool {
        unsafe { (self.active)(self.stream) == 1 }
    }
}

enum Command {
    Reopen,
    Finished,
    Shutdown,
}

/// All initialize/open/close/reopen calls belong to this thread, never to the
/// Windows low-level hook/message thread. Missing/unplugged devices retry.
pub(super) struct Owner {
    commands: Sender<Command>,
    shutdown: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

fn supervise<S: ManagedStream>(
    mut open: impl FnMut() -> Result<S>,
    gate: Arc<Mutex<CaptureGate>>,
    commands: Receiver<Command>,
    ready: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    retry: Duration,
) {
    let mut stream: Option<S> = None;
    let mut recovered = false;
    while !shutdown.load(Ordering::Acquire) {
        if stream.is_none() {
            match open() {
                Ok(opened) => {
                    stream = Some(opened);
                    ready.store(true, Ordering::Release);
                    if recovered {
                        if let Ok(mut gate) = gate.lock() {
                            gate.error = Some("麦克风已恢复".into());
                        }
                    }
                    recovered = true;
                }
                Err(error) => {
                    if let Ok(mut gate) = gate.lock() {
                        gate.error = Some(format!("麦克风不可用，正在重试: {error}"));
                    }
                }
            }
        }
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        match commands.recv_timeout(retry) {
            Ok(Command::Shutdown) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Ok(Command::Reopen) => {
                ready.store(false, Ordering::Release);
                stream.take();
            }
            Ok(Command::Finished) | Err(mpsc::RecvTimeoutError::Timeout) => {
                if stream.as_ref().is_some_and(|stream| !stream.is_active()) {
                    ready.store(false, Ordering::Release);
                    stream.take();
                    if let Ok(mut gate) = gate.lock() {
                        gate.error = Some("麦克风流已停止，正在重新打开设备".into());
                    }
                }
            }
        }
    }
    ready.store(false, Ordering::Release);
    drop(stream);
}

impl Owner {
    pub(super) fn open(dll: &Path, gate: Arc<Mutex<CaptureGate>>) -> Result<Self> {
        if !dll.is_absolute() || !dll.is_file() {
            bail!("PortAudio DLL is missing: {}", dll.display());
        }
        let dll: PathBuf = dll.into();
        let (commands, receiver) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_ready = Arc::clone(&ready);
        let worker_shutdown = Arc::clone(&shutdown);
        let callback_commands = commands.clone();
        let callback_gate = Arc::clone(&gate);
        let worker = thread::Builder::new()
            .name("microphone-owner".into())
            .spawn(move || {
                supervise(
                    || Stream::open(&dll, Arc::clone(&callback_gate), callback_commands.clone()),
                    gate,
                    receiver,
                    worker_ready,
                    worker_shutdown,
                    Duration::from_secs(1),
                );
            })?;
        Ok(Self {
            commands,
            shutdown,
            ready,
            worker: Some(worker),
        })
    }
    pub(super) fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
    pub(super) fn reopen(&self) {
        self.ready.store(false, Ordering::Release);
        let _ = self.commands.send(Command::Reopen);
    }
}

impl Drop for Owner {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        let _ = self.commands.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    struct FakeStream {
        generation: usize,
        drops: Arc<AtomicUsize>,
    }
    impl ManagedStream for FakeStream {
        fn is_active(&self) -> bool {
            self.generation > 1
        }
    }
    impl Drop for FakeStream {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn stopped_stream_reopens_and_closes_each_generation_without_native_calls() {
        let (events, _rx) = crate::desktop::event_channel(1024 * 1024);
        let gate = Arc::new(Mutex::new(CaptureGate::new(events)));
        let (commands, receiver) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let shutdown = Arc::new(AtomicBool::new(false));
        let opens = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let worker_opens = Arc::clone(&opens);
        let worker_drops = Arc::clone(&drops);
        let worker = thread::spawn(move || {
            supervise(
                || {
                    let generation = worker_opens.fetch_add(1, Ordering::SeqCst) + 1;
                    Ok(FakeStream {
                        generation,
                        drops: Arc::clone(&worker_drops),
                    })
                },
                gate,
                receiver,
                ready,
                shutdown,
                Duration::from_millis(5),
            )
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while opens.load(Ordering::SeqCst) < 2 {
            assert!(std::time::Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        commands.send(Command::Shutdown).unwrap();
        worker.join().unwrap();
        assert_eq!(opens.load(Ordering::SeqCst), 2);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn initially_missing_device_recovers_on_retry_without_native_calls() {
        let (events, _rx) = crate::desktop::event_channel(1024 * 1024);
        let gate = Arc::new(Mutex::new(CaptureGate::new(events)));
        let (commands, receiver) = mpsc::channel();
        let ready = Arc::new(AtomicBool::new(false));
        let worker_ready = Arc::clone(&ready);
        let shutdown = Arc::new(AtomicBool::new(false));
        let opens = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let worker_opens = Arc::clone(&opens);
        let worker_drops = Arc::clone(&drops);
        let worker = thread::spawn(move || {
            supervise(
                || {
                    let generation = worker_opens.fetch_add(1, Ordering::SeqCst) + 1;
                    if generation < 3 {
                        bail!("fake unplugged input");
                    }
                    Ok(FakeStream {
                        generation,
                        drops: Arc::clone(&worker_drops),
                    })
                },
                gate,
                receiver,
                worker_ready,
                shutdown,
                Duration::from_millis(5),
            )
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !ready.load(Ordering::Acquire) {
            assert!(std::time::Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        commands.send(Command::Shutdown).unwrap();
        worker.join().unwrap();
        assert_eq!(opens.load(Ordering::SeqCst), 3);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(!ready.load(Ordering::Acquire));
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            (self.runtime.abort)(self.stream);
            (self.runtime.close)(self.stream);
        }
    }
}

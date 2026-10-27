use crate::{
    config::Settings,
    pipeline::Pipeline,
    protocol::{Reply, Request, read_packet, read_reply, write_packet, write_reply},
};
use anyhow::{Context, Result, bail, ensure};
use std::os::windows::process::CommandExt;
use std::{
    fs::OpenOptions,
    io::{BufReader, BufWriter},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

const CREATE_NO_WINDOW: u32 = 0x08000000;

fn send_reply(stream: &mut impl std::io::Write, reply: &Reply) -> Result<()> {
    match write_reply(stream, reply, &[]) {
        // Header serialization happens before frame output, so only this error
        // permits a safe replacement frame. I/O failures may have written bytes.
        Err(error) if error.is::<crate::protocol::HeaderLimitExceeded>() => write_reply(
            stream,
            &Reply::error(
                reply.id,
                format!("{error:#}; transcription result exceeded the reply limit"),
            ),
            &[],
        ),
        other => other,
    }
}

pub fn run(settings: Settings) -> Result<()> {
    let (send, recv) = mpsc::sync_channel(4);
    thread::Builder::new()
        .name("ipc-input".into())
        .spawn(move || {
            let input = std::io::stdin();
            let mut input = BufReader::new(input.lock());
            loop {
                match read_packet::<Request>(&mut input) {
                    Ok(Some(frame)) => {
                        if send.send(Ok(frame)).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(error) => {
                        let _ = send.send(Err(error));
                        break;
                    }
                }
            }
        })?;
    let output = std::io::stdout();
    let mut output = BufWriter::new(output.lock());
    write_reply(&mut output, &Reply::ok(0, "boot"), &[])?;
    // The parent's single model-dispatch thread is the sole ASR idle owner.
    // A second independent worker timeout could retire a process while a new
    // request was already in its pipe.
    let mut pipeline = Pipeline::new(settings);
    loop {
        match recv.recv_timeout(Duration::from_secs(1)) {
            Ok(Ok((request, audio))) => {
                let shutdown = matches!(request, Request::Shutdown { .. });
                let reply = pipeline.dispatch(request, &audio);
                send_reply(&mut output, &reply)?;
                if shutdown {
                    break;
                }
            }
            Ok(Err(error)) => return Err(error),
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                pipeline.cleanup_idle();
            }
        }
    }
    Ok(())
}

struct Connection {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
    closing: Arc<AtomicBool>,
    timeout: Duration,
}
impl Connection {
    fn spawn(settings: &Settings, closing: Arc<AtomicBool>) -> Result<Self> {
        let log_dir = std::env::var_os("YYUR_LOG_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or(
                std::env::current_exe()?
                    .parent()
                    .context("Executable folder")?
                    .join("logs"),
            );
        std::fs::create_dir_all(&log_dir)?;
        let log_path = log_dir.join("rust-worker.log");
        if log_path
            .metadata()
            .is_ok_and(|metadata| metadata.len() > 2 * 1024 * 1024)
        {
            let previous = log_dir.join("rust-worker.previous.log");
            if previous.exists() {
                std::fs::remove_file(&previous)?;
            }
            std::fs::rename(&log_path, &previous)?;
        }
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;
        let mut command = Command::new(std::env::current_exe()?);
        // Keep cache policy inside the owned worker's environment. The native
        // backend validates device/driver identity and treats cache I/O failure
        // as a cache miss, so read-only application folders still run normally.
        if settings.gpu && settings.vulkan_pipeline_cache {
            command.env(
                "GGML_VK_PIPELINE_CACHE_DIR",
                settings.app_dir.join("cache/vulkan"),
            );
        } else {
            command.env_remove("GGML_VK_PIPELINE_CACHE_DIR");
        }
        let mut child = command
            .arg("worker")
            .arg("--settings-json")
            .arg(serde_json::to_string(settings)?)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log))
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()?;
        let input = BufWriter::new(child.stdin.take().context("Worker stdin")?);
        let mut output = BufReader::new(child.stdout.take().context("Worker stdout")?);
        let boot = crate::process_guard::ProcessGuard::watch(
            &child,
            closing.clone(),
            Duration::from_secs(30),
        )?;
        let result = (|| -> Result<()> {
            let (ready, audio) = read_reply(&mut output)?.context("Worker exited during boot")?;
            ensure!(
                ready.kind == "boot" && ready.id == 0 && audio.is_empty(),
                "Worker boot handshake failed"
            );
            Ok(())
        })();
        drop(boot);
        if let Err(error) = result {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        Ok(Self {
            child,
            input,
            output,
            closing,
            timeout: Duration::from_secs(settings.worker_timeout_seconds),
        })
    }
    fn call(&mut self, request: &Request, pcm: &[f32]) -> Result<Reply> {
        let timeout = if matches!(request, Request::Shutdown { .. }) {
            Duration::from_secs(5)
        } else {
            self.timeout
        };
        let guard =
            crate::process_guard::ProcessGuard::watch(&self.child, self.closing.clone(), timeout)?;
        let packet = (|| -> Result<(Reply, Vec<f32>)> {
            write_packet(&mut self.input, request, pcm)?;
            read_reply(&mut self.output)?
                .context("Inference worker exited unexpectedly; task was not replayed")
        })();
        if guard.cancelled() {
            bail!("Inference cancelled because the application is stopping")
        }
        if guard.expired() {
            bail!("Inference worker exceeded the configured timeout; current task was cancelled")
        }
        let (reply, audio) = packet?;
        ensure!(
            reply.id == request.id() && audio.is_empty(),
            "Worker reply order violated"
        );
        Ok(reply)
    }
}
impl Drop for Connection {
    fn drop(&mut self) {
        // This handle owns exactly the process created here. No PID/name enumeration or external kills.
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}
enum Job {
    Request(Request, Vec<f32>, mpsc::Sender<Result<Reply>>),
    Stop,
}
#[derive(Clone)]
pub struct WorkerClient {
    jobs: mpsc::SyncSender<Job>,
    closing: Arc<AtomicBool>,
}
pub struct WorkerService {
    client: WorkerClient,
    thread: Option<thread::JoinHandle<()>>,
}
impl WorkerService {
    pub fn start(settings: Settings) -> Result<Self> {
        let (send, recv) = mpsc::sync_channel(4);
        let closing = Arc::new(AtomicBool::new(false));
        let client = WorkerClient {
            jobs: send,
            closing: closing.clone(),
        };
        let thread = thread::Builder::new()
            .name("model-dispatch".into())
            .spawn(move || {
                let mut connection: Option<Connection> = None;
                let mut last_complete = Instant::now();
                loop {
                    let job = match recv.recv_timeout(Duration::from_secs(1)) {
                        Ok(job) => job,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if settings.asr_idle_seconds > 0
                                && last_complete.elapsed().as_secs() >= settings.asr_idle_seconds
                            {
                                if let Some(mut retired) = connection.take() {
                                    let _ = retired.call(&Request::Shutdown { id: u64::MAX }, &[]);
                                }
                            }
                            continue;
                        }
                    };
                    let Job::Request(request, audio, response) = job else {
                        break;
                    };
                    if closing.load(Ordering::Acquire) {
                        let _ = response.send(Err(anyhow::anyhow!("Service is stopping")));
                        continue;
                    }
                    let mut retry = 1u64;
                    let result = loop {
                        if closing.load(Ordering::Acquire) {
                            break Err(anyhow::anyhow!("Service is stopping"));
                        }
                        let result = (|| -> Result<Reply> {
                            if connection
                                .as_mut()
                                .is_some_and(|v| v.child.try_wait().ok().flatten().is_some())
                            {
                                connection = None;
                            }
                            if connection.is_none() {
                                connection = Some(Connection::spawn(&settings, closing.clone())?);
                            }
                            connection.as_mut().unwrap().call(&request, &audio)
                        })();
                        match result {
                            Ok(reply)
                                if reply.kind == "load_error"
                                    && matches!(request, Request::Segment { .. }) =>
                            {
                                eprintln!(
                                    "Model load failed; retain PCM and retry in {retry}s: {}",
                                    reply.error.as_deref().unwrap_or("unknown")
                                );
                                for _ in 0..retry * 10 {
                                    if closing.load(Ordering::Acquire) {
                                        break;
                                    }
                                    thread::sleep(Duration::from_millis(100));
                                }
                                retry = (retry * 2).min(10);
                            }
                            Err(error) => {
                                connection = None;
                                break Err(error);
                            }
                            other => break other,
                        }
                    };
                    last_complete = Instant::now();
                    let _ = response.send(result);
                }
                if let Some(mut connection) = connection {
                    if !closing.load(Ordering::Acquire)
                        && connection.child.try_wait().ok().flatten().is_none()
                    {
                        let _ = connection.call(&Request::Shutdown { id: u64::MAX }, &[]);
                    }
                }
            })?;
        Ok(Self {
            client,
            thread: Some(thread),
        })
    }
    pub fn client(&self) -> WorkerClient {
        self.client.clone()
    }
    pub fn request_stop(&self) {
        self.client.closing.store(true, Ordering::Release);
    }
    pub fn shutdown(&mut self) {
        self.request_stop();
        let _ = self.client.jobs.send(Job::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Drop for WorkerService {
    fn drop(&mut self) {
        self.shutdown();
    }
}
impl WorkerClient {
    pub fn cancellation_token(&self) -> Arc<AtomicBool> {
        self.closing.clone()
    }
    pub fn submit(
        &self,
        request: Request,
        audio: Vec<f32>,
    ) -> Result<mpsc::Receiver<Result<Reply>>> {
        if self.closing.load(Ordering::Acquire) {
            bail!("Service is stopping")
        }
        let (send, recv) = mpsc::channel();
        self.jobs
            .send(Job::Request(request, audio, send))
            .context("Model dispatcher stopped")?;
        Ok(recv)
    }
    pub fn call(&self, request: Request, audio: Vec<f32>) -> Result<Reply> {
        let reply = self
            .submit(request, audio)?
            .recv()
            .context("Model dispatcher lost reply")??;
        if let Some(error) = &reply.error {
            bail!("{error}")
        }
        Ok(reply)
    }
}

pub fn serve(settings: Settings) -> Result<()> {
    let service = WorkerService::start(settings)?;
    let client = service.client();
    let input = std::io::stdin();
    let mut input = BufReader::new(input.lock());
    let output = std::io::stdout();
    let mut output = BufWriter::new(output.lock());
    write_reply(&mut output, &Reply::ok(0, "boot"), &[])?;
    while let Some((request, audio)) = read_packet::<Request>(&mut input)? {
        let stop = matches!(request, Request::Shutdown { .. });
        let id = request.id();
        let reply = match client.call(request, audio) {
            Ok(v) => v,
            Err(e) => Reply::error(id, format!("{e:#}")),
        };
        send_reply(&mut output, &reply)?;
        if stop {
            break;
        }
    }
    drop(service);
    Ok(())
}

use crate::{
    config::Settings,
    desktop::{self, DesktopEvent, DesktopOutput},
    media,
    pipeline::{AudioSegment, SegmentBuffer, now},
    protocol::{Recognition, Request},
    random::SamplerSeeds,
    text::{
        HotwordEngine, SaveOptions, Transcript, TranscriptWriter, strip_trailing_punctuation,
        sync_tokens_from_text,
    },
    worker::{WorkerClient, WorkerService},
};
use anyhow::{Context, Result, ensure};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
};

struct InstanceGuard(windows_sys::Win32::Foundation::HANDLE);
impl InstanceGuard {
    fn acquire(app_dir: &Path) -> Result<Option<Self>> {
        use sha2::{Digest, Sha256};
        use windows_sys::Win32::{
            Foundation::{CloseHandle, ERROR_ALREADY_EXISTS, GetLastError},
            System::Threading::CreateMutexW,
        };
        let key = format!(
            "Local\\YuYbUuRu.{:x}",
            Sha256::digest(app_dir.to_string_lossy().to_ascii_lowercase().as_bytes())
        );
        let name: Vec<u16> = key.encode_utf16().chain(Some(0)).collect();
        let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
        ensure!(
            !handle.is_null(),
            "Cannot create the application instance mutex"
        );
        if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
            unsafe { CloseHandle(handle) };
            return Ok(None);
        }
        Ok(Some(Self(handle)))
    }
}
impl Drop for InstanceGuard {
    fn drop(&mut self) {
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0) };
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}
pub struct TextProcessor {
    settings: Settings,
    engine: HotwordEngine,
    signature: Vec<(u64, Option<std::time::SystemTime>)>,
}
impl TextProcessor {
    pub fn new(settings: &Settings) -> Result<Self> {
        let mut value = Self {
            settings: settings.clone(),
            engine: HotwordEngine::from_contents(
                "",
                "",
                settings.hot_threshold,
                settings.hot_similar_threshold,
            )?,
            signature: Vec::new(),
        };
        value.refresh()?;
        Ok(value)
    }
    fn refresh(&mut self) -> Result<()> {
        let paths = [
            self.settings.app_dir.join("hot.txt"),
            self.settings.app_dir.join("hot-rule.txt"),
        ];
        let signature = paths
            .iter()
            .map(|p| {
                p.metadata()
                    .ok()
                    .map(|m| (m.len(), m.modified().ok()))
                    .unwrap_or((0, None))
            })
            .collect::<Vec<_>>();
        if signature == self.signature {
            return Ok(());
        }
        let hot = if self.settings.hotwords {
            std::fs::read_to_string(&paths[0]).unwrap_or_default()
        } else {
            String::new()
        };
        let rules = if self.settings.hotword_rules {
            std::fs::read_to_string(&paths[1]).unwrap_or_default()
        } else {
            String::new()
        };
        self.engine = HotwordEngine::from_contents(
            &hot,
            &rules,
            self.settings.hot_threshold,
            self.settings.hot_similar_threshold,
        )?;
        self.signature = signature;
        Ok(())
    }
    pub fn mic(&mut self, text: &str, force_strip: bool) -> Result<String> {
        self.refresh()?;
        let corrected = self.engine.correct_with_k(text, 0).text;
        let stripped = strip_trailing_punctuation(&corrected, "，。,.", 8, force_strip);
        Ok(self.engine.substitute_rules(&stripped))
    }
    pub fn file(&mut self, result: &Recognition) -> Result<Transcript> {
        self.refresh()?;
        let text = if result.text_accu.is_empty() {
            &result.text
        } else {
            &result.text_accu
        };
        let corrected = self
            .engine
            .substitute_rules(&self.engine.correct_with_k(text, 0).text);
        let mut out = Transcript {
            text: result.text.clone(),
            text_accu: result.text_accu.clone(),
            tokens: result.tokens.clone(),
            timestamps: result.timestamps.clone(),
        };
        if corrected != *text && !out.tokens.is_empty() {
            let merged = sync_tokens_from_text(&out.tokens, &out.timestamps, &corrected);
            out.tokens = merged.tokens;
            out.timestamps = merged.timestamps;
            out.text = corrected.clone();
            out.text_accu = corrected;
        }
        Ok(out)
    }
}

#[derive(Clone)]
pub struct RuntimeClient {
    worker: WorkerClient,
    text: Arc<Mutex<TextProcessor>>,
    seeds: Arc<Mutex<Option<SamplerSeeds>>>,
    settings: Settings,
}
impl RuntimeClient {
    pub fn new(worker: WorkerClient, settings: Settings, seed: Option<u32>) -> Result<Self> {
        Ok(Self {
            worker,
            text: Arc::new(Mutex::new(TextProcessor::new(&settings)?)),
            seeds: Arc::new(Mutex::new(seed.map(SamplerSeeds::new))),
            settings,
        })
    }
    pub fn warmup(&self) -> Result<()> {
        let _ = self
            .worker
            .submit(Request::Warmup { id: next_id() }, Vec::new())?;
        Ok(())
    }
    fn segment(
        &self,
        task_id: &str,
        source: &str,
        part: AudioSegment,
        started: f64,
    ) -> Result<Recognition> {
        let seed = self
            .seeds
            .lock()
            .unwrap()
            .as_mut()
            .map(SamplerSeeds::next_seed);
        let result = self
            .worker
            .call(
                Request::Segment {
                    id: next_id(),
                    task_id: task_id.into(),
                    source: source.into(),
                    language: self.settings.language.clone(),
                    context: String::new(),
                    offset: part.offset,
                    overlap: part.overlap,
                    is_final: part.is_final,
                    time_start: started,
                    time_submit: now(),
                    seed,
                },
                part.audio,
            )
            .and_then(|reply| reply.result.context("Worker returned no recognition"));
        if result.is_err() {
            let _ = self.worker.submit(
                Request::DropSession {
                    id: next_id(),
                    task_id: task_id.into(),
                },
                Vec::new(),
            );
        }
        result
    }
    pub fn transcribe(
        &self,
        file: &Path,
        output_dir: Option<&Path>,
        overwrite: bool,
    ) -> Result<(Transcript, Vec<PathBuf>)> {
        let target = if let Some(dir) = output_dir {
            std::fs::create_dir_all(dir)?;
            dir.join(file.file_name().context("Input filename")?)
        } else {
            file.to_owned()
        };
        if !overwrite {
            for extension in ["txt", "json", "srt"] {
                ensure!(
                    !target.with_extension(extension).exists(),
                    "Output already exists: {} (use --overwrite explicitly)",
                    target.with_extension(extension).display()
                );
            }
        }
        let id = format!("file-{}", next_id());
        let mut cache =
            SegmentBuffer::new(self.settings.segment_seconds, self.settings.overlap_seconds);
        let started = now();
        let process = (|| -> Result<Recognition> {
            media::stream_audio_cancellable(
                &self.settings.app_dir,
                file,
                self.worker.cancellation_token(),
                |audio| {
                    for part in cache.append(audio) {
                        self.segment(&id, "file", part, started)?;
                    }
                    Ok(())
                },
            )?;
            self.segment(&id, "file", cache.finish(), started)
        })();
        if process.is_err() {
            let _ = self.worker.call(
                Request::DropSession {
                    id: next_id(),
                    task_id: id,
                },
                Vec::new(),
            );
        }
        let transcript = self.text.lock().unwrap().file(&process?)?;
        let files = TranscriptWriter::write(
            &target,
            &transcript,
            &SaveOptions {
                txt: true,
                json: true,
                srt: true,
                merge_txt: false,
            },
        )?;
        Ok((transcript, files))
    }
}

pub fn desktop(settings: Settings, seed: Option<u32>, passive: bool) -> Result<()> {
    let Some(_instance) = InstanceGuard::acquire(&settings.app_dir)? else {
        return Ok(());
    };
    let service = WorkerService::start(settings.clone())?;
    let client = RuntimeClient::new(service.client(), settings.clone(), seed)?;
    let (events_tx, events_rx) = desktop::event_channel(1024 * 1024);
    let (output_tx, output_rx) = mpsc::channel();
    let dispatch_output = output_tx.clone();
    let dispatch_client = client.clone();
    // Only one file decoder exists. Pending entries hold paths, never full PCM segments.
    let (files_tx, files_rx) = mpsc::sync_channel::<PathBuf>(4);
    let file_client = client.clone();
    let file_output = output_tx.clone();
    let file_thread = thread::Builder::new()
        .name("file-transcription".into())
        .spawn(move || {
            while let Ok(file) = files_rx.recv() {
                if file_client
                    .worker
                    .cancellation_token()
                    .load(Ordering::Acquire)
                {
                    break;
                }
                let _ = file_output.send(DesktopOutput::State("正在转录文件".into()));
                match file_client.transcribe(&file, None, false) {
                    Ok((_, files)) => {
                        let _ = file_output.send(DesktopOutput::State(format!(
                            "转录完成：{} 个输出文件",
                            files.len()
                        )));
                    }
                    Err(error) => {
                        let _ = file_output.send(DesktopOutput::Error(format!("{error:#}")));
                    }
                }
            }
        })?;
    let dispatcher = thread::Builder::new()
        .name("desktop-dispatch".into())
        .spawn(move || {
            let mut session: Option<(String, f64, SegmentBuffer)> = None;
            while let Ok(event) = events_rx.recv() {
                let quitting = matches!(event, DesktopEvent::Quit);
                let voice_event = matches!(
                    &event,
                    DesktopEvent::Start
                        | DesktopEvent::Audio(_)
                        | DesktopEvent::Stop
                        | DesktopEvent::CaptureFailed(_)
                );
                let result = (|| -> Result<()> {
                    match event {
                        DesktopEvent::Start => {
                            session = Some((
                                format!("mic-{}", next_id()),
                                now(),
                                SegmentBuffer::new(
                                    dispatch_client.settings.segment_seconds,
                                    dispatch_client.settings.overlap_seconds,
                                ),
                            ));
                            dispatch_client.warmup()?;
                            let _ = dispatch_output.send(DesktopOutput::State("正在听写".into()));
                        }
                        DesktopEvent::Audio(audio) => {
                            if let Some((id, start, cache)) = session.as_mut() {
                                // Inference is much faster than recording, but this path remains FIFO and bounded by segments.
                                for part in cache.append(&audio) {
                                    dispatch_client.segment(id, "mic", part, *start)?;
                                }
                            }
                        }
                        DesktopEvent::Stop => {
                            if let Some((id, start, mut cache)) = session.take() {
                                let _ =
                                    dispatch_output.send(DesktopOutput::State("正在识别".into()));
                                let result =
                                    dispatch_client.segment(&id, "mic", cache.finish(), start)?;
                                let force = desktop::foreground_process_name()
                                    .eq_ignore_ascii_case("weixin.exe");
                                let text = dispatch_client
                                    .text
                                    .lock()
                                    .unwrap()
                                    .mic(&result.text, force)?;
                                if !text.is_empty() {
                                    let _ = dispatch_output.send(DesktopOutput::Text(text));
                                }
                                let _ = dispatch_output
                                    .send(DesktopOutput::State("离线听写已就绪".into()));
                            }
                        }
                        DesktopEvent::Transcribe(file) => {
                            files_tx.try_send(file).map_err(|_| {
                                anyhow::anyhow!("文件转录队列已满（最多4个待处理文件）")
                            })?;
                        }
                        DesktopEvent::CaptureFailed(message) => anyhow::bail!("{message}"),
                        DesktopEvent::Quit => {}
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    if voice_event {
                        if let Some((task_id, _, _)) = session.take() {
                            let _ = dispatch_client.worker.submit(
                                Request::DropSession {
                                    id: next_id(),
                                    task_id,
                                },
                                Vec::new(),
                            );
                        }
                    }
                    let _ = dispatch_output.send(DesktopOutput::Error(format!("{error:#}")));
                }
                if quitting {
                    break;
                }
            }
            drop(files_tx);
            let _ = file_thread.join();
        })?;
    let mut options = desktop::DesktopOptions::new(settings.app_dir);
    if passive {
        options.enable_hooks = false;
        options.enable_capture = false;
    }
    let result = desktop::run(options, events_tx.clone(), output_rx);
    service.request_stop();
    let _ = events_tx.send(DesktopEvent::Quit);
    drop(events_tx);
    let _ = dispatcher.join();
    drop(service);
    result
}

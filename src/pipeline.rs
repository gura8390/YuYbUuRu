use crate::{
    config::Settings,
    engine::{Aligner, Engine, EngineOptions},
    protocol::{Recognition, Reply, Request},
    text::{TextFormatter, merge_by_text, merge_segments, sync_tokens_from_text},
};
use anyhow::{Context, Result};
use std::{
    collections::HashMap,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

pub fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}
pub fn engine_options(settings: &Settings) -> EngineOptions {
    EngineOptions {
        onnx_dll: settings.onnx_dll.clone(),
        encoder_provider: settings.encoder_provider,
        encoder_cuda_mem_limit_mb: settings.encoder_cuda_mem_limit_mb,
        encoder_frontend_threads: settings.encoder_frontend_threads,
        encoder_frontend_arena: settings.encoder_frontend_arena,
        encoder_backend_arena: settings.encoder_backend_arena,
        encoder_parallel_execution: settings.encoder_parallel_execution,
        encoder_spin_wait: settings.encoder_spin_wait,
        encoder_dynamic_block_base: settings.encoder_dynamic_block_base,
        use_gpu: settings.gpu,
        gpu_backend: settings.gpu_backend,
        onnx_intra_threads: settings.onnx_intra_threads,
        onnx_inter_threads: settings.onnx_inter_threads,
        llm_threads: settings.llm_threads,
        llm_batch_threads: settings.llm_batch_threads,
        onnx_memory_pattern: settings.onnx_memory_pattern,
        onnx_arena_shrink: settings.onnx_arena_shrink,
        onnx_device_initializers: settings.onnx_device_initializers,
        onnx_prepacking: settings.onnx_prepacking,
        asr_dynamic_context: settings.asr_dynamic_context,
        asr_single_output: settings.asr_single_output,
        aligner_cpu_mem_arena: settings.aligner_cpu_mem_arena,
        aligner_dynamic_context: settings.aligner_dynamic_context,
        aligner_logits_limit: settings.aligner_logits_limit,
        ..Default::default()
    }
}
pub struct Pipeline {
    settings: Settings,
    options: EngineOptions,
    engine: Option<Engine>,
    aligner: Option<Aligner>,
    aligner_last_used: Instant,
    sessions: HashMap<String, Recognition>,
}
impl Pipeline {
    pub fn new(settings: Settings) -> Self {
        let options = engine_options(&settings);
        Self {
            settings,
            options,
            engine: None,
            aligner: None,
            aligner_last_used: Instant::now(),
            sessions: HashMap::new(),
        }
    }
    pub fn cleanup_idle(&mut self) {
        if self.settings.aligner_idle_seconds > 0
            && self.aligner_last_used.elapsed().as_secs() >= self.settings.aligner_idle_seconds
        {
            self.aligner = None;
        }
    }
    fn warmup(&mut self) -> Result<()> {
        if self.engine.is_none() {
            self.engine = Some(Engine::load(&self.settings.app_dir, &self.options)?);
            if self.settings.trim_after_load {
                // Match the existing application's post-initialization policy.
                // GPU upload leaves model-file pages resident even though the
                // CPU no longer needs most of them. This changes residency,
                // not private commitment, model data or inference parameters.
                unsafe {
                    windows_sys::Win32::System::ProcessStatus::EmptyWorkingSet(
                        windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    );
                }
            }
        }
        Ok(())
    }
    pub fn dispatch(&mut self, request: Request, pcm: &[f32]) -> Reply {
        let id = request.id();
        match &request {
            Request::Warmup { .. } | Request::Segment { .. } => {
                if let Err(error) = self.warmup() {
                    let mut r = Reply::error(id, format!("{error:#}"));
                    r.kind = "load_error".into();
                    return r;
                }
            }
            _ => {}
        }
        match self.process(request, pcm) {
            Ok(reply) => reply,
            Err(error) => Reply::error(id, format!("{error:#}")),
        }
    }
    fn process(&mut self, request: Request, pcm: &[f32]) -> Result<Reply> {
        let id = request.id();
        let Request::Segment {
            task_id,
            source,
            language,
            context,
            offset,
            overlap,
            is_final,
            time_start,
            time_submit,
            seed,
            ..
        } = request
        else {
            match request {
                Request::DropSession { task_id, .. } => {
                    self.sessions.remove(&task_id);
                }
                Request::Shutdown { .. } => {
                    self.sessions.clear();
                    self.aligner = None;
                    self.engine = None;
                }
                _ => {}
            }
            return Ok(Reply::ok(id, "ready"));
        };
        let first = !self.sessions.contains_key(&task_id);
        // Commit a session only after the whole segment succeeds; load errors never consume PCM.
        let mut result = self
            .sessions
            .get(&task_id)
            .cloned()
            .unwrap_or_else(|| Recognition {
                task_id: task_id.clone(),
                ..Default::default()
            });
        result.time_start = time_start;
        result.time_submit = time_submit;
        result.time_complete = now();
        result.is_final = is_final;
        let mut metrics = serde_json::Value::Null;
        if pcm.len() >= 1600 {
            result.duration += pcm.len() as f64 / 16000.0 - overlap;
            if is_final {
                result.duration += overlap;
            }
            let handoff = source == "file" && self.settings.file_context_handoff;
            if handoff && let Some(aligner) = self.aligner.as_mut() {
                aligner.release_context();
            }
            let engine = self.engine.as_mut().context("ASR was not initialized")?;
            engine.set_seed(seed);
            let transcribed = engine.transcribe(pcm, &language, &context);
            if handoff {
                engine.release_context();
            }
            let raw = transcribed?;
            let segment = raw
                .text
                .replace("@@", "")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            result.text = merge_by_text(&result.text, &segment);
            let mut tokens = Vec::new();
            let mut timestamps = Vec::new();
            if source == "file" && !raw.text.trim().is_empty() {
                if self.aligner.is_none() {
                    self.aligner = Some(Aligner::load(&self.settings.app_dir, &self.options)?);
                }
                let aligner = self.aligner.as_mut().unwrap();
                let aligned = aligner.align(pcm, &raw.text, &language, 0.0);
                if handoff {
                    aligner.release_context();
                }
                let items = aligned?;
                for item in items {
                    tokens.push(item.text);
                    timestamps.push(item.start_time);
                }
                self.aligner_last_used = Instant::now();
            }
            let merged = merge_segments(
                &result.tokens,
                &result.timestamps,
                &tokens,
                &timestamps,
                offset,
                overlap,
                first,
            );
            result.tokens = merged.tokens;
            result.timestamps = merged.timestamps;
            result.text_accu = crate::text::tokens_to_text(&result.tokens);
            if is_final {
                let formatter = TextFormatter::new(crate::text::TextOptions {
                    format_num: self.settings.format_num,
                    format_spell: self.settings.format_spell,
                });
                result.text = formatter.format(&result.text);
                result.text_accu = formatter.format(&result.text_accu);
                if !result.tokens.is_empty() && !result.text_accu.is_empty() {
                    let merged = sync_tokens_from_text(
                        &result.tokens,
                        &result.timestamps,
                        &result.text_accu,
                    );
                    result.tokens = merged.tokens;
                    result.timestamps = merged.timestamps;
                }
                if result.tokens.is_empty() && !result.text.is_empty() {
                    result.text_accu = result.text.clone();
                    result.tokens = result
                        .text_accu
                        .chars()
                        .filter(|c| *c != ' ')
                        .map(|c| c.to_string())
                        .collect();
                    if !result.tokens.is_empty() && result.duration > 0.0 {
                        let dt = result.duration / result.tokens.len() as f64;
                        result.timestamps =
                            (0..result.tokens.len()).map(|i| i as f64 * dt).collect();
                    }
                }
            }
            result.time_complete = now();
            metrics = serde_json::to_value(raw)?;
        }
        let mut reply = Reply::ok(id, "result");
        if is_final {
            self.sessions.remove(&task_id);
            reply.result = Some(result);
        } else if source == "file" {
            // File callers discard intermediate results. Keep the accumulated
            // transcript in the worker and acknowledge only segment metadata.
            reply.result = Some(Recognition {
                task_id: result.task_id.clone(),
                duration: result.duration,
                is_final: result.is_final,
                time_start: result.time_start,
                time_submit: result.time_submit,
                time_complete: result.time_complete,
                ..Default::default()
            });
            self.sessions.insert(task_id, result);
        } else {
            self.sessions.insert(task_id, result.clone());
            reply.result = Some(result);
        }
        reply.metrics = metrics;
        Ok(reply)
    }
}

pub struct SegmentBuffer {
    data: Vec<f32>,
    offset: f64,
    segment: usize,
    overlap: usize,
}
pub struct AudioSegment {
    pub audio: Vec<f32>,
    pub offset: f64,
    pub overlap: f64,
    pub is_final: bool,
}
impl SegmentBuffer {
    pub fn new(segment_seconds: usize, overlap_seconds: usize) -> Self {
        Self {
            data: Vec::new(),
            offset: 0.0,
            segment: segment_seconds * 16000,
            overlap: overlap_seconds * 16000,
        }
    }
    pub fn append(&mut self, pcm: &[f32]) -> Vec<AudioSegment> {
        self.data.extend_from_slice(pcm);
        let mut out = Vec::new();
        while self.data.len() >= self.segment + 2 * self.overlap {
            out.push(AudioSegment {
                audio: self.data[..self.segment + self.overlap].to_vec(),
                offset: self.offset,
                overlap: self.overlap as f64 / 16000.0,
                is_final: false,
            });
            self.data.drain(..self.segment);
            self.offset += self.segment as f64 / 16000.0;
        }
        out
    }
    pub fn finish(&mut self) -> AudioSegment {
        let out = AudioSegment {
            audio: std::mem::take(&mut self.data),
            offset: self.offset,
            overlap: self.overlap as f64 / 16000.0,
            is_final: true,
        };
        self.offset = 0.0;
        out
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_options_preserves_encoder_cpu_controls_and_auto_threads() {
        let settings: Settings = serde_json::from_str(
            r#"{"encoder_frontend_threads":0,"encoder_frontend_arena":true,"encoder_backend_arena":true,"encoder_parallel_execution":true,"encoder_spin_wait":true,"encoder_dynamic_block_base":4}"#,
        ).unwrap();
        let options = engine_options(&settings);
        assert_eq!(options.encoder_frontend_threads, Some(0));
        assert!(options.encoder_frontend_arena);
        assert!(options.encoder_backend_arena);
        assert!(options.encoder_parallel_execution);
        assert!(options.encoder_spin_wait);
        assert_eq!(options.encoder_dynamic_block_base, 4);
        let defaults = engine_options(&Settings::default());
        assert_eq!(defaults.encoder_frontend_threads, None);
        assert!(!defaults.encoder_frontend_arena);
        assert!(!defaults.encoder_backend_arena);
        assert!(!defaults.encoder_parallel_execution);
        assert!(!defaults.encoder_spin_wait);
        assert_eq!(defaults.encoder_dynamic_block_base, 0);
    }

    #[test]
    fn engine_options_maps_asr_dynamic_context() {
        let mut settings = Settings::default();
        assert!(!engine_options(&settings).asr_dynamic_context);
        settings.asr_dynamic_context = true;
        assert!(engine_options(&settings).asr_dynamic_context);
    }

    fn session_request(id: u64, source: &str, is_final: bool) -> Request {
        Request::Segment {
            id,
            task_id: "test-session".into(),
            source: source.into(),
            language: "auto".into(),
            context: String::new(),
            offset: 0.0,
            overlap: 4.0,
            is_final,
            time_start: 10.0,
            time_submit: 20.0,
            seed: None,
        }
    }

    #[test]
    fn file_ack_size_is_independent_of_accumulated_transcript_and_final_is_complete() {
        let mut pipeline = Pipeline::new(Settings::default());
        let state = Recognition {
            task_id: "test-session".into(),
            text: "长".repeat(100_000),
            text_accu: "长".repeat(100_000),
            tokens: vec!["长".into(); 100_000],
            timestamps: (0..100_000).map(|i| i as f64 * 0.2).collect(),
            duration: 20_000.0,
            ..Default::default()
        };
        assert!(serde_json::to_vec(&state).unwrap().len() > 1024 * 1024);
        pipeline
            .sessions
            .insert(state.task_id.clone(), state.clone());
        for id in 1..=3 {
            // Empty PCM exercises commit/reply ownership without loading a model.
            let reply = pipeline
                .process(session_request(id, "file", false), &[])
                .unwrap();
            assert!(serde_json::to_vec(&reply).unwrap().len() < 512);
            let ack = reply.result.unwrap();
            assert_eq!(ack.task_id, state.task_id);
            assert_eq!(ack.duration, state.duration);
            assert_eq!(ack.time_start, 10.0);
            assert_eq!(ack.time_submit, 20.0);
            assert!(!ack.is_final);
            assert!(ack.text.is_empty() && ack.text_accu.is_empty());
            assert!(ack.tokens.is_empty() && ack.timestamps.is_empty());
            let retained = &pipeline.sessions["test-session"];
            assert_eq!(retained.text, state.text);
            assert_eq!(retained.text_accu, state.text_accu);
            assert_eq!(retained.tokens, state.tokens);
            assert_eq!(retained.timestamps, state.timestamps);
        }
        let reply = pipeline
            .process(session_request(4, "file", true), &[])
            .unwrap();
        let final_result = reply.result.unwrap();
        assert!(final_result.is_final);
        assert_eq!(final_result.text, state.text);
        assert_eq!(final_result.text_accu, state.text_accu);
        assert_eq!(final_result.tokens, state.tokens);
        assert_eq!(final_result.timestamps, state.timestamps);
        assert_eq!(final_result.duration, state.duration);
        assert!(pipeline.sessions.is_empty());
    }

    #[test]
    fn microphone_intermediate_reply_still_contains_the_complete_result() {
        let mut pipeline = Pipeline::new(Settings::default());
        let state = Recognition {
            task_id: "test-session".into(),
            text: "原文。".into(),
            text_accu: "原文。".into(),
            tokens: vec!["原".into(), "文".into(), "。".into()],
            timestamps: vec![0.0, 0.2, 0.4],
            duration: 1.0,
            ..Default::default()
        };
        pipeline
            .sessions
            .insert(state.task_id.clone(), state.clone());
        let reply = pipeline
            .process(session_request(1, "mic", false), &[])
            .unwrap();
        let result = reply.result.unwrap();
        assert_eq!(result.text, state.text);
        assert_eq!(result.text_accu, state.text_accu);
        assert_eq!(result.tokens, state.tokens);
        assert_eq!(result.timestamps, state.timestamps);
        assert_eq!(pipeline.sessions["test-session"].text, state.text);
    }

    #[test]
    fn preserves_overlap_and_last_partial() {
        let mut cache = SegmentBuffer::new(60, 4);
        let mut parts = Vec::new();
        for _ in 0..1600 {
            parts.extend(cache.append(&vec![0.2; 800]));
        }
        parts.push(cache.finish());
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].audio.len(), 64 * 16000);
        assert_eq!(parts[1].audio.len(), 20 * 16000);
        assert_eq!(parts[1].offset, 60.0);
        assert!(!parts[0].is_final && parts[1].is_final);
    }
}

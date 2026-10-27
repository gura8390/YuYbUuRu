//! Native Qwen3 inference. Model files and DLLs are read from the supplied app directory.
//! No Python interpreter or subprocess participates in inference.
#[path = "engine/aligner.rs"]
mod aligner;
#[path = "engine/encoder.rs"]
mod encoder;
#[path = "engine/gguf.rs"]
mod gguf;
#[path = "engine/llama.rs"]
mod llama;
#[path = "engine/mel.rs"]
mod mel;

pub use crate::config::{EncoderProvider, GpuBackend};
pub use aligner::{AlignedToken, Aligner, AlignmentMetrics};
use anyhow::{Context, Result, bail, ensure};
use encoder::AudioEncoder;
use gguf::EmbeddingTable;
use llama::{NativeContext, NativeModel};
pub use mel::MelExtractor;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashSet, VecDeque},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

/// Opt-in diagnostics use stderr so the binary protocol remains unchanged.
pub(super) fn load_timing(stage: &'static str, started: Instant) {
    if std::env::var_os("YYUR_PROFILE_LOAD").is_some() {
        eprintln!(
            "YYUR_LOAD_PROFILE {{\"stage\":\"{stage}\",\"seconds\":{},\"pid\":{}}}",
            started.elapsed().as_secs_f64(),
            std::process::id()
        );
    }
}

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub asr_model_dir: Option<PathBuf>,
    pub aligner_model_dir: Option<PathBuf>,
    pub onnx_dll: Option<PathBuf>,
    pub encoder_provider: EncoderProvider,
    pub encoder_cuda_mem_limit_mb: usize,
    pub llama_dir: Option<PathBuf>,
    pub use_gpu: bool,
    pub gpu_backend: GpuBackend,
    pub onnx_intra_threads: usize,
    pub onnx_inter_threads: usize,
    pub llm_threads: i32,
    pub llm_batch_threads: i32,
    pub n_ctx: u32,
    pub aligner_n_ctx: u32,
    pub seed: Option<u32>,
    pub temperature: f64,
    pub cpu_mem_arena: bool,
    pub onnx_memory_pattern: bool,
    pub onnx_arena_shrink: bool,
    pub onnx_device_initializers: bool,
    pub onnx_prepacking: bool,
    pub asr_dynamic_context: bool,
    pub asr_single_output: bool,
    pub aligner_cpu_mem_arena: bool,
    pub aligner_dynamic_context: bool,
    pub aligner_logits_limit: u32,
}
impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            asr_model_dir: None,
            aligner_model_dir: None,
            onnx_dll: None,
            encoder_provider: EncoderProvider::Cpu,
            encoder_cuda_mem_limit_mb: 512,
            llama_dir: None,
            use_gpu: true,
            gpu_backend: GpuBackend::Cuda,
            onnx_intra_threads: 16,
            onnx_inter_threads: 1,
            llm_threads: 1,
            llm_batch_threads: 1,
            n_ctx: 2048,
            aligner_n_ctx: 3072,
            seed: None,
            temperature: 0.4,
            cpu_mem_arena: false,
            onnx_memory_pattern: true,
            onnx_arena_shrink: false,
            onnx_device_initializers: false,
            onnx_prepacking: true,
            asr_dynamic_context: false,
            asr_single_output: false,
            aligner_cpu_mem_arena: false,
            aligner_dynamic_context: false,
            aligner_logits_limit: 0,
        }
    }
}
impl EngineOptions {
    fn llama_dir(&self, app: &Path) -> PathBuf {
        self.llama_dir
            .clone()
            .unwrap_or_else(|| app.join("core/server/engines/llama/bin"))
    }
    fn onnx_dll(&self, app: &Path) -> PathBuf {
        match self.onnx_dll.as_ref() {
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => app.join(path),
            None => app.join("internal/onnxruntime/capi/onnxruntime.dll"),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawRecognition {
    pub text: String,
    pub token_ids: Vec<i32>,
    pub t_encode: f64,
    pub t_prefill: f64,
    pub t_generate: f64,
    #[serde(default)]
    pub t_context: f64,
    #[serde(default)]
    pub context_capacity: usize,
    pub n_prefill: usize,
    pub n_generate: usize,
    pub is_aborted: bool,
}

/// A mutable context is intentionally used by exactly one inference worker.
pub struct Engine {
    // Context must be destroyed before the model and DLL references.
    ctx: Option<NativeContext>,
    context_capacity: usize,
    model: NativeModel,
    embeddings: EmbeddingTable,
    encoder: AudioEncoder,
    options: EngineOptions,
    special: [i32; 5],
    next_seeds: VecDeque<u32>,
}

impl Engine {
    pub fn load(app_dir: &Path, options: &EngineOptions) -> Result<Self> {
        let load_started = Instant::now();
        let dir = options
            .asr_model_dir
            .clone()
            .unwrap_or_else(|| app_dir.join("models/Qwen3-ASR/Qwen3-ASR-0.6B"));
        // The encoder and decoder own independent runtimes and immutable model
        // inputs. Overlap their initialization without changing inference order.
        // Always join the encoder, including when decoder initialization fails.
        let (encoder, model, embeddings, ctx) = std::thread::scope(|scope| -> Result<_> {
            let encoder = std::thread::Builder::new()
                .name("encoder-load".into())
                .spawn_scoped(scope, || {
                    let stage = Instant::now();
                    let result = AudioEncoder::load(
                        &options.onnx_dll(app_dir),
                        &dir,
                        "qwen3_asr_encoder_frontend.int4.onnx",
                        "qwen3_asr_encoder_backend.int4.onnx",
                        options,
                    );
                    load_timing("asr.encoder", stage);
                    result
                })?;
            let decoder = (|| -> Result<_> {
                let stage = Instant::now();
                let model = NativeModel::load_with_backend(
                    &options.llama_dir(app_dir),
                    &dir.join("qwen3_asr_llm.q4_k.gguf"),
                    options.use_gpu,
                    options.gpu_backend,
                )?;
                load_timing("asr.native_model", stage);
                let stage = Instant::now();
                let embeddings =
                    EmbeddingTable::open(&dir.join("qwen3_asr_llm.q4_k.gguf"), model.api.clone())?;
                ensure!(
                    embeddings.dim == model.dim,
                    "Embedding dimension differs from native model"
                );
                // Dynamic mode waits for the prompt length; otherwise an
                // initial full context would inflate the first-request peak.
                let ctx = if options.asr_dynamic_context {
                    None
                } else {
                    Some(NativeContext::new_asr(
                        &model,
                        options.n_ctx,
                        4096,
                        options.llm_threads,
                        options.llm_batch_threads,
                        options.asr_single_output,
                    )?)
                };
                load_timing("asr.embeddings_and_context", stage);
                Ok((model, embeddings, ctx))
            })();
            let encoder = encoder
                .join()
                .map_err(|_| anyhow::anyhow!("Encoder loader thread panicked"))?;
            let (model, embeddings, ctx) = decoder?;
            Ok((encoder?, model, embeddings, ctx))
        })?;
        let mut special = [0; 5];
        for (i, s) in [
            "<|im_start|>",
            "<|im_end|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<asr_text>",
        ]
        .iter()
        .enumerate()
        {
            special[i] = model.token_id(s)?;
        }
        load_timing("asr.total", load_started);
        let context_capacity = if ctx.is_some() {
            options.n_ctx as usize
        } else {
            0
        };
        Ok(Self {
            ctx,
            context_capacity,
            model,
            embeddings,
            encoder,
            options: options.clone(),
            special,
            next_seeds: VecDeque::new(),
        })
    }

    pub fn set_seed(&mut self, seed: Option<u32>) {
        self.options.seed = seed;
        self.next_seeds.clear();
    }
    /// Native sampler seeds, one for each possible repetition-recovery attempt.
    pub fn set_native_seeds(&mut self, seeds: &[u32]) {
        self.next_seeds = seeds.iter().copied().collect();
    }

    /// Release decoder state while retaining model weights and the encoder.
    /// The next decode recreates the same configured context on demand.
    pub fn release_context(&mut self) -> bool {
        self.context_capacity = 0;
        let Some(ctx) = self.ctx.take() else {
            return false;
        };
        drop(ctx);
        true
    }

    fn ensure_context(&mut self, prefill_tokens: usize) -> Result<bool> {
        let capacity = asr_context_capacity(
            prefill_tokens,
            self.options.n_ctx as usize,
            self.options.asr_dynamic_context,
        );
        if self.ctx.is_some() && self.context_capacity == capacity {
            return Ok(false);
        }
        // Never overlap the old KV/compute buffers with their replacement.
        drop(self.ctx.take());
        self.context_capacity = 0;
        // Publish only a completely initialized context. A failed native
        // allocation leaves None so a later request can retry safely.
        let ctx = NativeContext::new_asr(
            &self.model,
            capacity.try_into()?,
            4096,
            self.options.llm_threads,
            self.options.llm_batch_threads,
            self.options.asr_single_output,
        )?;
        self.ctx = Some(ctx);
        self.context_capacity = capacity;
        Ok(true)
    }

    pub fn transcribe(
        &mut self,
        audio: &[f32],
        language: &str,
        context: &str,
    ) -> Result<RawRecognition> {
        Ok(self.transcribe_with_embedding(audio, language, context)?.0)
    }

    /// Diagnostic API: returns the exact ONNX output alongside recognition.
    pub fn transcribe_with_embedding(
        &mut self,
        audio: &[f32],
        language: &str,
        context: &str,
    ) -> Result<(RawRecognition, Vec<f32>, usize)> {
        ensure!(!audio.is_empty(), "Cannot transcribe empty audio");
        let audio = &audio[..audio.len().min(80 * 16000)];
        let started = Instant::now();
        let feature = self.encoder.encode(audio)?;
        ensure!(
            feature.dim == self.model.dim,
            "ASR encoder dimension mismatch"
        );
        let encode_time = started.elapsed().as_secs_f64();
        let language = normalize_language(language)?;
        let mut pre = vec![self.special[0]];
        pre.extend(self.model.tokenize(&format!(
            "system\n{}",
            if context.is_empty() {
                "You are a helpful assistant."
            } else {
                context
            }
        ))?);
        pre.push(self.special[1]);
        pre.push(self.special[0]);
        pre.extend(self.model.tokenize("user\n")?);
        pre.push(self.special[2]);
        let mut post = vec![self.special[3], self.special[1], self.special[0]];
        let mut assistant = "assistant\n".to_string();
        if let Some(lang) = language {
            assistant.push_str("language ");
            assistant.push_str(&lang);
        }
        post.extend(self.model.tokenize(&assistant)?);
        post.push(self.special[4]);
        let mut full = self.embeddings.rows(&pre)?;
        full.extend_from_slice(&feature.data);
        full.extend(self.embeddings.rows(&post)?);
        let n = full.len() / self.model.dim;
        ensure!(
            n + 1 <= self.options.n_ctx as usize,
            "ASR prompt has {n} tokens, exceeding context {}",
            self.options.n_ctx
        );
        let mut temperature = self.options.temperature;
        let mut out = RawRecognition::default();
        let mut context_time = 0.0;
        for attempt in 0..4 {
            let seed = self
                .next_seeds
                .pop_front()
                .or(self.options.seed)
                .unwrap_or_else(|| {
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .subsec_nanos()
                        & 0x7fff_ffff
                });
            out = self.decode(&full, n, temperature as f32, seed)?;
            context_time += out.t_context;
            if !out.is_aborted {
                break;
            }
            temperature += 0.3;
            out.text.push_str("====解码有误，强制熔断====");
            if attempt == 3 {
                break;
            }
        }
        out.t_encode = encode_time;
        out.t_context = context_time;
        Ok((out, feature.data, feature.dim))
    }

    fn decode(
        &mut self,
        full: &[f32],
        n: usize,
        temperature: f32,
        seed: u32,
    ) -> Result<RawRecognition> {
        let context_started = Instant::now();
        self.ensure_context(n)?;
        let context_time = context_started.elapsed().as_secs_f64();
        let ctx = self
            .ctx
            .as_mut()
            .context("ASR context was not initialized")?;
        let mut out = RawRecognition {
            n_prefill: n,
            t_context: context_time,
            context_capacity: self.context_capacity,
            ..Default::default()
        };
        ctx.clear()?;
        let started = Instant::now();
        ctx.prefill(full, n, self.model.dim, &[])?;
        out.t_prefill = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut sampler = self.model.sampler(temperature, seed)?;
        let mut token = sampler.sample(ctx)?;
        let mut pending = VecDeque::new();
        let mut bytes = Vec::new();
        for _ in 0..ASR_MAX_GENERATE_TOKENS {
            if token == self.model.eos || token == self.special[1] {
                break;
            }
            // Match the shipped engine: only retain a sampled token after a successful decode.
            if ctx.decode_token(token)? != 0 {
                break;
            }
            pending.push_back(token);
            if pending.len() > 5 {
                let ready = pending.pop_front().unwrap();
                out.token_ids.push(ready);
                bytes.extend(self.model.token_bytes(ready)?);
            }
            if out.token_ids.len() > 15
                && out.token_ids[out.token_ids.len() - 15..]
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>()
                    .len()
                    <= 3
            {
                out.is_aborted = true;
                break;
            }
            token = sampler.sample(ctx)?;
            out.n_generate += 1;
        }
        if !out.is_aborted {
            for ready in pending {
                out.token_ids.push(ready);
                bytes.extend(self.model.token_bytes(ready)?);
            }
        }
        // Python's incremental decoder is only flushed on a non-aborted final
        // decode. Preserve an incomplete final UTF-8 fragment on abort.
        out.text = decode_utf8(&bytes, !out.is_aborted);
        out.t_generate = started.elapsed().as_secs_f64();
        Ok(out)
    }
}

const ASR_MAX_GENERATE_TOKENS: usize = 512;

fn asr_context_capacity(prefill_tokens: usize, maximum: usize, dynamic: bool) -> usize {
    if !dynamic {
        return maximum;
    }
    // Reserve every decode step the original full context could accept. Near
    // the logical cap, keep that same cap rather than rejecting valid prompts.
    prefill_tokens
        .saturating_add(ASR_MAX_GENERATE_TOKENS)
        .div_ceil(256)
        .saturating_mul(256)
        .max(768)
        .min(maximum)
}

fn decode_utf8(bytes: &[u8], final_chunk: bool) -> String {
    if final_chunk {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    let mut cursor = 0;
    while cursor < bytes.len() {
        match std::str::from_utf8(&bytes[cursor..]) {
            Ok(_) => break,
            Err(error) => {
                cursor += error.valid_up_to();
                if let Some(n) = error.error_len() {
                    cursor += n;
                } else {
                    return String::from_utf8_lossy(&bytes[..cursor]).into_owned();
                }
            }
        }
    }
    String::from_utf8_lossy(bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn asr_dynamic_context_reserves_all_original_generation_steps() {
        for maximum in [512, 768, 1024, 2048] {
            for prefill in 1..maximum {
                let capacity = asr_context_capacity(prefill, maximum, true);
                assert!(capacity <= maximum);
                assert!(capacity >= prefill);
                assert!(capacity - prefill >= ASR_MAX_GENERATE_TOKENS.min(maximum - prefill));
                assert_eq!(capacity % 256, 0);
            }
        }
    }

    #[test]
    fn asr_dynamic_context_bucket_edges_and_reuse_target() {
        for (prefill, expected) in [
            (1, 768),
            (256, 768),
            (257, 1024),
            (512, 1024),
            (513, 1280),
            (1024, 1536),
            (1025, 1792),
            (1280, 1792),
            (1281, 2048),
            (1536, 2048),
            (1537, 2048),
            (2047, 2048),
        ] {
            assert_eq!(asr_context_capacity(prefill, 2048, true), expected);
        }
        assert_eq!(asr_context_capacity(usize::MAX, 2048, true), 2048);
    }

    #[test]
    fn asr_fixed_context_retains_original_capacity() {
        assert!(!EngineOptions::default().asr_dynamic_context);
        for prefill in [1, 256, 513, 1536, 2047] {
            assert_eq!(asr_context_capacity(prefill, 2048, false), 2048);
        }
    }

    #[test]
    fn abort_preserves_unflushed_utf8_suffix() {
        assert_eq!(decode_utf8(&[b'a', 0xe4, 0xb8], false), "a");
        assert_eq!(decode_utf8(&[b'a', 0xe4, 0xb8], true), "a�");
        assert_eq!(decode_utf8(&[0xff, b'a', 0xe4, 0xb8], false), "�a");
    }
}

fn normalize_language(language: &str) -> Result<Option<String>> {
    let value = language.trim().to_lowercase();
    if value.is_empty() || value == "auto" {
        return Ok(None);
    }
    let supported = [
        "chinese",
        "english",
        "cantonese",
        "arabic",
        "german",
        "french",
        "spanish",
        "portuguese",
        "indonesian",
        "italian",
        "korean",
        "russian",
        "thai",
        "vietnamese",
        "japanese",
        "turkish",
        "hindi",
        "malay",
        "dutch",
        "swedish",
        "danish",
        "finnish",
        "polish",
        "czech",
        "filipino",
        "persian",
        "greek",
        "romanian",
        "hungarian",
        "macedonian",
    ];
    if !supported.contains(&value.as_str()) {
        bail!("Unsupported language: {language}");
    }
    let mut result = value;
    result[..1].make_ascii_uppercase();
    Ok(Some(result))
}

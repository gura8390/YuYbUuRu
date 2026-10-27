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

pub use aligner::{AlignedToken, Aligner};
use anyhow::{Result, bail, ensure};
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

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub asr_model_dir: Option<PathBuf>,
    pub aligner_model_dir: Option<PathBuf>,
    pub onnx_dll: Option<PathBuf>,
    pub llama_dir: Option<PathBuf>,
    pub use_gpu: bool,
    pub onnx_intra_threads: usize,
    pub onnx_inter_threads: usize,
    pub llm_threads: i32,
    pub llm_batch_threads: i32,
    pub n_ctx: u32,
    pub aligner_n_ctx: u32,
    pub seed: Option<u32>,
    pub temperature: f64,
    pub cpu_mem_arena: bool,
}
impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            asr_model_dir: None,
            aligner_model_dir: None,
            onnx_dll: None,
            llama_dir: None,
            use_gpu: true,
            onnx_intra_threads: 16,
            onnx_inter_threads: 1,
            llm_threads: 1,
            llm_batch_threads: 1,
            n_ctx: 2048,
            aligner_n_ctx: 3072,
            seed: None,
            temperature: 0.4,
            cpu_mem_arena: false,
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
        self.onnx_dll
            .clone()
            .unwrap_or_else(|| app.join("internal/onnxruntime/capi/onnxruntime.dll"))
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawRecognition {
    pub text: String,
    pub token_ids: Vec<i32>,
    pub t_encode: f64,
    pub t_prefill: f64,
    pub t_generate: f64,
    pub n_prefill: usize,
    pub n_generate: usize,
    pub is_aborted: bool,
}

/// A mutable context is intentionally used by exactly one inference worker.
pub struct Engine {
    // Context must be destroyed before the model and DLL references.
    ctx: NativeContext,
    model: NativeModel,
    embeddings: EmbeddingTable,
    encoder: AudioEncoder,
    options: EngineOptions,
    special: [i32; 5],
    next_seeds: VecDeque<u32>,
}

impl Engine {
    pub fn load(app_dir: &Path, options: &EngineOptions) -> Result<Self> {
        let dir = options
            .asr_model_dir
            .clone()
            .unwrap_or_else(|| app_dir.join("models/Qwen3-ASR/Qwen3-ASR-0.6B"));
        let encoder = AudioEncoder::load(
            &options.onnx_dll(app_dir),
            &dir,
            "qwen3_asr_encoder_frontend.int4.onnx",
            "qwen3_asr_encoder_backend.int4.onnx",
            options,
        )?;
        let model = NativeModel::load(
            &options.llama_dir(app_dir),
            &dir.join("qwen3_asr_llm.q4_k.gguf"),
            options.use_gpu,
        )?;
        let embeddings =
            EmbeddingTable::open(&dir.join("qwen3_asr_llm.q4_k.gguf"), model.api.clone())?;
        ensure!(
            embeddings.dim == model.dim,
            "Embedding dimension differs from native model"
        );
        let ctx = NativeContext::new(
            &model,
            options.n_ctx,
            4096,
            options.llm_threads,
            options.llm_batch_threads,
        )?;
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
        Ok(Self {
            ctx,
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
        Ok((out, feature.data, feature.dim))
    }

    fn decode(
        &mut self,
        full: &[f32],
        n: usize,
        temperature: f32,
        seed: u32,
    ) -> Result<RawRecognition> {
        let mut out = RawRecognition {
            n_prefill: n,
            ..Default::default()
        };
        self.ctx.clear()?;
        let started = Instant::now();
        self.ctx.prefill(full, n, self.model.dim, &[])?;
        out.t_prefill = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let mut sampler = self.model.sampler(temperature, seed)?;
        let mut token = sampler.sample(&self.ctx)?;
        let mut pending = VecDeque::new();
        let mut bytes = Vec::new();
        for _ in 0..512 {
            if token == self.model.eos || token == self.special[1] {
                break;
            }
            // Match the shipped engine: only retain a sampled token after a successful decode.
            if self.ctx.decode_token(token)? != 0 {
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
            token = sampler.sample(&self.ctx)?;
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

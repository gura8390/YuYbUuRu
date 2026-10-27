use super::{
    EngineOptions,
    encoder::AudioEncoder,
    gguf::EmbeddingTable,
    llama::{NativeContext, NativeModel},
    normalize_language,
};
use anyhow::{Context, Result, ensure};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::OnceLock, time::Instant};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlignedToken {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}
/// Optional diagnostics; this does not change the public transcription result.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AlignmentMetrics {
    pub audio_tokens: usize,
    pub prompt_tokens: usize,
    pub timestamp_outputs: usize,
    pub logits_limit: u32,
    pub context_capacity: usize,
    pub context_rebuilt: bool,
    pub context_build_count: u64,
    pub requested_batch_tokens: usize,
    pub microbatch_tokens: usize,
    pub feature_bytes: usize,
    pub prompt_bytes: usize,
    pub encode_seconds: f64,
    pub prompt_seconds: f64,
    pub context_seconds: f64,
    pub prefill_seconds: f64,
    pub argmax_seconds: f64,
}
pub struct Aligner {
    ctx: Option<NativeContext>,
    model: NativeModel,
    embeddings: EmbeddingTable,
    encoder: AudioEncoder,
    audio_start: i32,
    audio_end: i32,
    timestamp: i32,
    n_ctx: usize,
    dynamic_context: bool,
    context_capacity: usize,
    context_build_count: u64,
    threads: i32,
    batch_threads: i32,
    last_metrics: AlignmentMetrics,
    logits_limit: u32,
    last_timestamp_positions: Vec<usize>,
}
impl Aligner {
    pub fn load(app_dir: &Path, options: &EngineOptions) -> Result<Self> {
        let load_started = Instant::now();
        let dir = options
            .aligner_model_dir
            .clone()
            .unwrap_or_else(|| app_dir.join("models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B"));
        // Retain the aligner's automatic ORT threading and CPU llama.
        // Allocator policy is independent of the ASR thread configuration.
        let mut align_options = options.clone();
        // The ASR CUDA experiment must not change the alignment model's EP.
        align_options.encoder_provider = super::EncoderProvider::Cpu;
        align_options.encoder_frontend_threads = None;
        align_options.encoder_frontend_arena = false;
        align_options.encoder_backend_arena = false;
        align_options.encoder_parallel_execution = false;
        align_options.encoder_spin_wait = false;
        align_options.encoder_dynamic_block_base = 0;
        align_options.onnx_intra_threads = 0;
        align_options.onnx_inter_threads = 0;
        align_options.cpu_mem_arena = options.aligner_cpu_mem_arena;
        let logical = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(4) as i32;
        let threads = (logical / 2).max(1);
        // Keep CPU llama and its context on this thread while the independent
        // ORT encoder initializes. Join the encoder before propagating errors.
        let (encoder, model, embeddings, ctx) = std::thread::scope(|scope| -> Result<_> {
            let encoder = std::thread::Builder::new()
                .name("aligner-encoder-load".into())
                .spawn_scoped(scope, || {
                    let stage = Instant::now();
                    let result = AudioEncoder::load(
                        &options.onnx_dll(app_dir),
                        &dir,
                        "qwen3_aligner_encoder_frontend.int4.onnx",
                        "qwen3_aligner_encoder_backend.int4.onnx",
                        &align_options,
                    );
                    super::load_timing("aligner.encoder", stage);
                    result
                })?;
            let decoder = (|| -> Result<_> {
                let stage = Instant::now();
                let model = NativeModel::load(
                    &options.llama_dir(app_dir),
                    &dir.join("qwen3_aligner_llm.q4_k.gguf"),
                    false,
                )?;
                super::load_timing("aligner.native_model", stage);
                let stage = Instant::now();
                let embeddings = EmbeddingTable::open(
                    &dir.join("qwen3_aligner_llm.q4_k.gguf"),
                    model.api.clone(),
                )?;
                ensure!(
                    embeddings.dim == model.dim,
                    "Aligner embedding dimension mismatch"
                );
                // Dynamic mode waits until the actual prompt length is known.
                // Allocating the full context here first would hide a double
                // peak during the first right-sizing operation.
                let ctx = if options.aligner_dynamic_context {
                    None
                } else {
                    Some(NativeContext::new_with_logits_limit(
                        &model,
                        options.aligner_n_ctx,
                        options.aligner_n_ctx,
                        threads,
                        logical,
                        options.aligner_logits_limit,
                    )?)
                };
                super::load_timing("aligner.embeddings_and_context", stage);
                Ok((model, embeddings, ctx))
            })();
            let encoder = encoder
                .join()
                .map_err(|_| anyhow::anyhow!("Aligner encoder loader thread panicked"))?;
            let (model, embeddings, ctx) = decoder?;
            Ok((encoder?, model, embeddings, ctx))
        })?;
        let audio_start = model.token_id("<|audio_start|>")?;
        let audio_end = model.token_id("<|audio_end|>")?;
        let timestamp = model.token_id("<timestamp>")?;
        super::load_timing("aligner.total", load_started);
        Ok(Self {
            ctx,
            model,
            embeddings,
            encoder,
            audio_start,
            audio_end,
            timestamp,
            n_ctx: options.aligner_n_ctx as usize,
            dynamic_context: options.aligner_dynamic_context,
            context_capacity: if options.aligner_dynamic_context {
                0
            } else {
                options.aligner_n_ctx as usize
            },
            context_build_count: if options.aligner_dynamic_context {
                0
            } else {
                1
            },
            threads,
            batch_threads: logical,
            last_metrics: AlignmentMetrics::default(),
            logits_limit: options.aligner_logits_limit,
            last_timestamp_positions: Vec::new(),
        })
    }
    pub fn last_metrics(&self) -> &AlignmentMetrics {
        &self.last_metrics
    }

    /// Release decoder state while retaining model weights and the encoder.
    /// Captured timestamp logits must be read before calling this method.
    pub fn release_context(&mut self) -> bool {
        self.context_capacity = 0;
        self.last_timestamp_positions.clear();
        let Some(ctx) = self.ctx.take() else {
            return false;
        };
        drop(ctx);
        true
    }

    /// Explicit diagnostic copy only. Normal alignment never materializes a
    /// second logits table on the Rust heap.
    pub fn last_timestamp_logits(&self) -> Result<Vec<f32>> {
        let ctx = self.ctx.as_ref().context("No completed aligner context")?;
        let count = self
            .last_timestamp_positions
            .len()
            .checked_mul(4000)
            .context("Timestamp diagnostic size overflow")?;
        let mut values = Vec::with_capacity(count);
        for &index in &self.last_timestamp_positions {
            values.extend(ctx.timestamp_logits_copy(index, 4000)?);
        }
        Ok(values)
    }

    fn ensure_context(&mut self, n: usize) -> Result<bool> {
        let capacity = if self.dynamic_context {
            // The native KV length is padded to 256. Keep at least one full
            // existing 512-token microbatch and preserve the configured cap.
            n.div_ceil(256).saturating_mul(256).max(512).min(self.n_ctx)
        } else {
            self.n_ctx
        };
        if self.ctx.is_some() && self.context_capacity == capacity {
            return Ok(false);
        }
        // Drop before allocating so old logits/KV/compute buffers do not
        // overlap the replacement, including when native creation fails.
        drop(self.ctx.take());
        self.context_capacity = 0;
        let ctx = NativeContext::new_with_logits_limit(
            &self.model,
            capacity.try_into()?,
            self.n_ctx.try_into()?,
            self.threads,
            self.batch_threads,
            self.logits_limit,
        )?;
        self.ctx = Some(ctx);
        self.context_capacity = capacity;
        self.context_build_count += 1;
        Ok(true)
    }
    pub fn align(
        &mut self,
        audio: &[f32],
        text: &str,
        language: &str,
        offset_sec: f64,
    ) -> Result<Vec<AlignedToken>> {
        Ok(self
            .align_with_embedding(audio, text, language, offset_sec)?
            .0)
    }
    pub fn align_with_embedding(
        &mut self,
        audio: &[f32],
        text: &str,
        language: &str,
        offset_sec: f64,
    ) -> Result<(Vec<AlignedToken>, Vec<f32>, usize)> {
        self.align_with_embedding_profile(audio, text, language, offset_sec, |_, _| Ok(()))
    }

    /// Checkpoints allow a separate probe to read process commitment without
    /// adding platform-specific memory APIs to the inference implementation.
    pub fn align_with_embedding_profile(
        &mut self,
        audio: &[f32],
        text: &str,
        language: &str,
        offset_sec: f64,
        mut checkpoint: impl FnMut(&'static str, &AlignmentMetrics) -> Result<()>,
    ) -> Result<(Vec<AlignedToken>, Vec<f32>, usize)> {
        self.last_metrics = AlignmentMetrics {
            context_capacity: self.context_capacity,
            context_build_count: self.context_build_count,
            logits_limit: self.logits_limit,
            requested_batch_tokens: self.n_ctx,
            microbatch_tokens: 512,
            ..Default::default()
        };
        self.last_timestamp_positions.clear();
        if text.is_empty() {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        ensure!(offset_sec.is_finite(), "Non-finite alignment offset");
        let lang = normalize_language(language)?.unwrap_or_else(|| "Chinese".to_owned());
        checkpoint("before_encoder", &self.last_metrics)?;
        let stage = Instant::now();
        let feature = self.encoder.encode(audio)?;
        self.last_metrics.encode_seconds = stage.elapsed().as_secs_f64();
        ensure!(
            feature.dim == self.model.dim,
            "Aligner encoder dimension mismatch"
        );
        self.last_metrics.audio_tokens = feature.data.len() / feature.dim;
        self.last_metrics.feature_bytes = std::mem::size_of_val(feature.data.as_slice());
        checkpoint("after_encoder", &self.last_metrics)?;
        let stage = Instant::now();
        let words = tokenize(text, &lang);
        let mut post = vec![self.audio_end];
        let mut ts_positions = Vec::with_capacity(words.len() * 2);
        let prefix = 1 + feature.data.len() / feature.dim + 1;
        let mut post_len = 0;
        for word in &words {
            let tokens = self.model.tokenize(word)?;
            post_len += tokens.len();
            post.extend(tokens);
            ts_positions.push(prefix + post_len);
            post.push(self.timestamp);
            post_len += 1;
            ts_positions.push(prefix + post_len);
            post.push(self.timestamp);
            post_len += 1;
        }
        let mut full = self.embeddings.rows(&[self.audio_start])?;
        full.extend_from_slice(&feature.data);
        full.extend(self.embeddings.rows(&post)?);
        let n = full.len() / self.model.dim;
        self.last_metrics.prompt_tokens = n;
        self.last_metrics.timestamp_outputs = ts_positions.len();
        self.last_metrics.prompt_bytes = std::mem::size_of_val(full.as_slice());
        self.last_metrics.prompt_seconds = stage.elapsed().as_secs_f64();
        ensure!(
            n <= self.n_ctx,
            "Aligner prompt has {n} tokens, exceeding context {}",
            self.n_ctx
        );
        checkpoint("after_prompt", &self.last_metrics)?;
        let stage = Instant::now();
        self.last_metrics.context_rebuilt = self.ensure_context(n)?;
        self.last_metrics.context_seconds = stage.elapsed().as_secs_f64();
        self.last_metrics.context_capacity = self.context_capacity;
        self.last_metrics.context_build_count = self.context_build_count;
        checkpoint("after_context", &self.last_metrics)?;
        let ctx = self
            .ctx
            .as_mut()
            .context("Aligner context was not initialized")?;
        ctx.clear()?;
        let stage = Instant::now();
        ctx.prefill(&full, n, self.model.dim, &ts_positions)?;
        self.last_metrics.prefill_seconds = stage.elapsed().as_secs_f64();
        checkpoint("after_prefill", &self.last_metrics)?;
        let stage = Instant::now();
        let mut raw = Vec::with_capacity(ts_positions.len());
        for &index in &ts_positions {
            raw.push(ctx.timestamp_argmax(index, 4000)?);
        }
        self.last_metrics.argmax_seconds = stage.elapsed().as_secs_f64();
        checkpoint("after_argmax", &self.last_metrics)?;
        self.last_timestamp_positions = ts_positions;
        let fixed = fix_timestamps(&raw);
        let items = words
            .into_iter()
            .enumerate()
            .map(|(i, text)| AlignedToken {
                text,
                start_time: fixed[i * 2] as f64 * 80.0 / 1000.0 + offset_sec,
                end_time: fixed[i * 2 + 1] as f64 * 80.0 / 1000.0 + offset_sec,
            })
            .collect::<Vec<_>>();
        Ok((reconcile(text, &items), feature.data, feature.dim))
    }
}

fn kept(ch: char) -> bool {
    static MATCH: OnceLock<Regex> = OnceLock::new();
    let regex = MATCH.get_or_init(|| Regex::new(r"^[\p{L}\p{N}']$").unwrap());
    regex.is_match(ch.encode_utf8(&mut [0; 4]))
}
fn cjk(ch: char) -> bool {
    let c = ch as u32;
    matches!(c,0x4e00..=0x9fff|0x3400..=0x4dbf|0x20000..=0x2a6df|0x2a700..=0x2b73f|0x2b740..=0x2b81f|0x2b820..=0x2ceaf|0xf900..=0xfaff)
}
fn tokenize(text: &str, language: &str) -> Vec<String> {
    // Neither optional nagisa nor soynlp exists in the deployed slim runtime;
    // its documented ImportError path returns list(text), including punctuation.
    if language == "Japanese" || language == "Korean" {
        return text.chars().map(|ch| ch.to_string()).collect();
    }
    let mut words = Vec::new();
    for seg in text.split_whitespace() {
        let mut buffer = String::new();
        for ch in seg.chars().filter(|ch| kept(*ch)) {
            if cjk(ch) {
                if !buffer.is_empty() {
                    words.push(std::mem::take(&mut buffer));
                }
                words.push(ch.to_string());
            } else {
                buffer.push(ch);
            }
        }
        if !buffer.is_empty() {
            words.push(buffer);
        }
    }
    words
}
fn fix_timestamps(data: &[i32]) -> Vec<i32> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let mut dp = vec![1; n];
    let mut parent = vec![None; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = Some(j);
            }
        }
    }
    let mut maximum = 0;
    for i in 1..n {
        if dp[i] > dp[maximum] {
            maximum = i;
        }
    }
    let mut normal = vec![false; n];
    let mut current = Some(maximum);
    while let Some(i) = current {
        normal[i] = true;
        current = parent[i];
    }
    let mut result = data.to_vec();
    let mut i = 0;
    while i < n {
        if normal[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !normal[j] {
            j += 1;
        }
        let count = j - i;
        let left = (0..i).rev().find(|&k| normal[k]).map(|k| result[k]);
        let right = (j..n).find(|&k| normal[k]).map(|k| result[k]);
        for (k, value) in result.iter_mut().enumerate().take(j).skip(i) {
            *value = match (left, right) {
                (None, Some(r)) => r,
                (Some(l), None) => l,
                (Some(l), Some(r)) => {
                    if count <= 2 {
                        if k - i + 1 <= j - k { l } else { r }
                    } else {
                        (l as f64 + (r - l) as f64 / (count + 1) as f64 * (k - i + 1) as f64) as i32
                    }
                }
                _ => *value,
            };
        }
        i = j;
    }
    result
}
fn find_token(text: &[char], target: &[char], start: usize) -> Option<(usize, usize)> {
    if target.is_empty() {
        return None;
    }
    let mut t = 0;
    let mut first = None;
    let mut i = start;
    while i < text.len() {
        if text[i] == target[t] {
            if t == 0 {
                first = Some(i);
            }
            t += 1;
            if t == target.len() {
                return Some((first.unwrap(), i + 1));
            }
        } else if kept(text[i]) {
            if let Some(back) = first {
                i = back;
                first = None;
                t = 0;
            }
        }
        i += 1;
    }
    None
}
fn reconcile(original: &str, items: &[AlignedToken]) -> Vec<AlignedToken> {
    if items.is_empty() {
        return if original.is_empty() {
            Vec::new()
        } else {
            vec![AlignedToken {
                text: original.to_owned(),
                start_time: 0.0,
                end_time: 0.0,
            }]
        };
    }
    let chars = original.chars().collect::<Vec<_>>();
    let mut pointer = 0;
    let mut last = items[0].start_time;
    let mut output = Vec::new();
    for item in items {
        if let Some((start, end)) =
            find_token(&chars, &item.text.chars().collect::<Vec<_>>(), pointer)
        {
            if start > pointer {
                output.push(AlignedToken {
                    text: chars[pointer..start].iter().collect(),
                    start_time: last,
                    end_time: last,
                });
            }
            output.push(AlignedToken {
                text: chars[start..end].iter().collect(),
                start_time: item.start_time,
                end_time: item.end_time,
            });
            pointer = end;
        } else {
            output.push(item.clone());
        }
        last = item.end_time;
    }
    if pointer < chars.len() {
        output.push(AlignedToken {
            text: chars[pointer..].iter().collect(),
            start_time: last,
            end_time: last,
        });
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn general_tokens_keep_unicode_categories() {
        assert_eq!(
            tokenize("你好， GPT's 3.14 AB测试!", "Chinese"),
            ["你", "好", "GPT's", "314", "AB", "测", "试"]
        );
    }
    #[test]
    fn lis_ties_match_first_python_max() {
        assert_eq!(
            fix_timestamps(&[10, 1, 2, 3, 2, 2, 4]),
            vec![1, 1, 2, 2, 2, 2, 4]
        );
    }
    #[test]
    fn punctuation_uses_left_endpoint() {
        let input = vec![
            AlignedToken {
                text: "你好".into(),
                start_time: 1.0,
                end_time: 2.0,
            },
            AlignedToken {
                text: "GPT".into(),
                start_time: 2.5,
                end_time: 3.0,
            },
        ];
        let out = reconcile("你好， GPT！", &input);
        assert_eq!(
            out.iter().map(|x| x.text.as_str()).collect::<Vec<_>>(),
            ["你好", "， ", "GPT", "！"]
        );
        assert_eq!((out[1].start_time, out[1].end_time), (2.0, 2.0));
    }
}

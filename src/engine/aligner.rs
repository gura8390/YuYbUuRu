use super::{
    EngineOptions,
    encoder::AudioEncoder,
    gguf::EmbeddingTable,
    llama::{NativeContext, NativeModel},
    normalize_language,
};
use anyhow::{Result, ensure};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::OnceLock, time::Instant};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AlignedToken {
    pub text: String,
    pub start_time: f64,
    pub end_time: f64,
}
pub struct Aligner {
    ctx: NativeContext,
    model: NativeModel,
    embeddings: EmbeddingTable,
    encoder: AudioEncoder,
    audio_start: i32,
    audio_end: i32,
    timestamp: i32,
    n_ctx: usize,
}
impl Aligner {
    pub fn load(app_dir: &Path, options: &EngineOptions) -> Result<Self> {
        let load_started = Instant::now();
        let dir = options
            .aligner_model_dir
            .clone()
            .unwrap_or_else(|| app_dir.join("models/Qwen3-ForcedAligner/Qwen3-ForcedAligner-0.6B"));
        // The deployed aligner uses ORT automatic threads/arena and CPU llama.
        // ASR thread tuning must not silently change its numerical execution.
        let mut align_options = options.clone();
        align_options.onnx_intra_threads = 0;
        align_options.onnx_inter_threads = 0;
        align_options.cpu_mem_arena = true;
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
                let logical = std::thread::available_parallelism()
                    .map(usize::from)
                    .unwrap_or(4) as i32;
                let ctx = NativeContext::new(
                    &model,
                    options.aligner_n_ctx,
                    options.aligner_n_ctx,
                    (logical / 2).max(1),
                    logical,
                )?;
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
        })
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
        if text.is_empty() {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        ensure!(offset_sec.is_finite(), "Non-finite alignment offset");
        let lang = normalize_language(language)?.unwrap_or_else(|| "Chinese".to_owned());
        let feature = self.encoder.encode(audio)?;
        ensure!(
            feature.dim == self.model.dim,
            "Aligner encoder dimension mismatch"
        );
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
        ensure!(
            n <= self.n_ctx,
            "Aligner prompt has {n} tokens, exceeding context {}",
            self.n_ctx
        );
        self.ctx.clear()?;
        self.ctx.prefill(&full, n, self.model.dim, &ts_positions)?;
        let mut raw = Vec::with_capacity(ts_positions.len());
        for index in ts_positions {
            raw.push(self.ctx.timestamp_argmax(index, 4000)?);
        }
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

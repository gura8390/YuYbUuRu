use super::{PUNCTUATION, sequence_matching_blocks};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Transcript {
    pub text: String,
    pub text_accu: String,
    pub tokens: Vec<String>,
    pub timestamps: Vec<f64>,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SaveOptions {
    pub txt: bool,
    pub json: bool,
    pub srt: bool,
    pub merge_txt: bool,
}
impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            txt: true,
            json: true,
            srt: true,
            merge_txt: false,
        }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RenderedTranscript {
    pub display_text: String,
    pub split_text: String,
    pub merge_text: String,
    pub json: String,
    pub srt: String,
}
pub struct TranscriptWriter;

pub fn smart_split(text: &str, min_chars: usize) -> String {
    static RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"[，。？]|[.,?!](?:\s+|$)").unwrap());
    let (mut cursor, mut buffer) = (0, String::new());
    let mut lines = Vec::new();
    for m in RE.find_iter(text) {
        buffer.push_str(&text[cursor..m.start()]);
        let part = m.as_str();
        let clean = part.trim();
        buffer.push_str(part);
        if clean.chars().count() == 1
            && ("。？.?!".contains(clean) || buffer.chars().count() > min_chars)
        {
            lines.push(std::mem::take(&mut buffer));
        }
        cursor = m.end();
    }
    buffer.push_str(&text[cursor..]);
    if !buffer.is_empty() {
        lines.push(buffer);
    }
    lines.join("\n")
}

fn cleaned(text: &str) -> Vec<char> {
    static DIGIT: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"^\d$").unwrap());
    text.to_lowercase()
        .chars()
        .filter(|c| {
            !PUNCTUATION.contains(*c) && !c.is_whitespace() && !DIGIT.is_match(&c.to_string())
        })
        .collect()
}
fn stamp(seconds: f64) -> String {
    // datetime.timedelta rounds to the nearest microsecond before srt truncates milliseconds.
    let ms = ((seconds * 1_000_000.).round_ties_even() as i64) / 1000;
    let h = ms / 3_600_000;
    let m = (ms / 60_000) % 60;
    let s = (ms / 1000) % 60;
    let millis = ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{millis:03}")
}

pub fn render_srt(tokens: &[String], timestamps: &[f64], text_lines: &[String]) -> Result<String> {
    if tokens.len() != timestamps.len() {
        bail!("Subtitle tokens and timestamps must have equal lengths");
    }
    if timestamps.iter().any(|t| !t.is_finite()) {
        bail!("Subtitle timestamps must be finite");
    }
    let lines: Vec<String> = text_lines
        .iter()
        .map(|l| {
            l.trim_end_matches(|c| "，。？！,.?!\r\n ".contains(c))
                .to_owned()
        })
        .collect();
    if tokens.is_empty() {
        if lines.iter().all(|l| cleaned(l).is_empty()) {
            return Ok(String::new());
        }
        bail!("Cannot align nonempty subtitle text without timed tokens");
    }
    let words: Vec<String> = tokens.iter().map(|t| t.replace('@', "")).collect();
    let mut token_chars = Vec::new();
    let mut indices = Vec::new();
    for (i, t) in words.iter().enumerate() {
        for c in cleaned(t) {
            token_chars.push(c);
            indices.push(i);
        }
    }
    let joined = lines.iter().map(|l| l.trim()).collect::<String>();
    let target = cleaned(&joined);
    let mut mapping = std::collections::HashMap::new();
    for (a, b, n) in sequence_matching_blocks(&token_chars, &target, true) {
        for k in 0..n {
            mapping.insert(b + k, indices[a + k]);
        }
    }
    let (mut offset, mut last) = (0, 0);
    let mut subtitles = Vec::new();
    for line in lines {
        let len = cleaned(&line).len();
        if len == 0 {
            continue;
        }
        let hits: Vec<usize> = (offset..offset + len)
            .filter_map(|i| mapping.get(&i).copied())
            .collect();
        let (start, end) = if !hits.is_empty() {
            let lo = *hits.iter().min().unwrap();
            let hi = *hits.iter().max().unwrap();
            last = hi;
            let end = (timestamps[hi] + 0.2)
                .min(timestamps.get(hi + 1).copied().unwrap_or(f64::INFINITY));
            (timestamps[lo], end)
        } else {
            let t = timestamps[(last + 1).min(timestamps.len() - 1)];
            (t, t + 0.5)
        };
        if start >= 0. && end > start && !line.trim().is_empty() {
            subtitles.push((start, end, line.trim().to_string()));
        }
        offset += len;
    }
    subtitles.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut out = String::new();
    for (i, (start, end, line)) in subtitles.into_iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            i + 1,
            stamp(start),
            stamp(end),
            line
        ));
    }
    Ok(out)
}

fn python_json(tokens: &[String], timestamps: &[f64]) -> Result<String> {
    #[derive(Serialize)]
    struct Output<'a> {
        timestamps: &'a [f64],
        tokens: &'a [String],
    }
    let compact = serde_json::to_string(&Output { timestamps, tokens })?;
    let (mut quote, mut escaped) = (false, false);
    let mut out = String::new();
    for c in compact.chars() {
        out.push(c);
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' && quote {
            escaped = true;
            continue;
        }
        if c == '"' {
            quote = !quote;
        }
        if !quote && (c == ',' || c == ':') {
            out.push(' ');
        }
    }
    Ok(out)
}

fn windows_text(text: &str) -> Vec<u8> {
    text.replace("\r\n", "\n")
        .replace('\n', "\r\n")
        .into_bytes()
}
impl TranscriptWriter {
    fn render_parts(message: &Transcript, include_srt: bool) -> Result<RenderedTranscript> {
        let text = if message.text_accu.is_empty() {
            &message.text
        } else {
            &message.text_accu
        };
        let split = smart_split(text, 2);
        let lines = split.lines().map(str::to_owned).collect::<Vec<_>>();
        Ok(RenderedTranscript {
            display_text: message.text.clone(),
            split_text: split,
            merge_text: text.clone(),
            json: python_json(&message.tokens, &message.timestamps)?,
            srt: if include_srt {
                render_srt(&message.tokens, &message.timestamps, &lines)?
            } else {
                String::new()
            },
        })
    }
    pub fn render(message: &Transcript) -> Result<RenderedTranscript> {
        Self::render_parts(message, true)
    }
    pub fn write(file: &Path, message: &Transcript, options: &SaveOptions) -> Result<Vec<PathBuf>> {
        let rendered = Self::render_parts(message, options.srt)?;
        let mut written = Vec::new();
        for (enabled, extension, data) in [
            (options.merge_txt, "merge.txt", &rendered.merge_text),
            (options.txt, "txt", &rendered.split_text),
            (options.json, "json", &rendered.json),
            (options.srt, "srt", &rendered.srt),
        ] {
            if enabled {
                let output = file.with_extension(extension);
                std::fs::write(&output, windows_text(data))?;
                written.push(output);
            }
        }
        let txt = file.with_extension("txt");
        if !options.txt && txt.exists() {
            std::fs::remove_file(txt)?;
        }
        Ok(written)
    }
}

/// Rebuild only SRT after an edited TXT file; preserve the original token/timestamp JSON.
pub fn adjust_srt(media: &Path) -> Result<PathBuf> {
    let text = std::fs::read_to_string(media.with_extension("txt"))?;
    #[derive(Deserialize)]
    struct Timed {
        tokens: Vec<String>,
        timestamps: Vec<f64>,
    }
    let timed: Timed =
        serde_json::from_str(&std::fs::read_to_string(media.with_extension("json"))?)?;
    let output = media.with_extension("srt");
    let lines = text.lines().map(str::to_owned).collect::<Vec<_>>();
    std::fs::write(
        &output,
        windows_text(&render_srt(&timed.tokens, &timed.timestamps, &lines)?),
    )?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_writer_fixtures() {
        use sha2::{Digest, Sha256};
        let rows: serde_json::Value =
            serde_json::from_str(include_str!("../../assets/text/writer_fixtures.json")).unwrap();
        for row in rows.as_array().unwrap() {
            let t = Transcript {
                text: row["text"].as_str().unwrap().into(),
                text_accu: row["text_accu"].as_str().unwrap().into(),
                tokens: serde_json::from_value(row["tokens"].clone()).unwrap(),
                timestamps: serde_json::from_value(row["timestamps"].clone()).unwrap(),
            };
            let got = TranscriptWriter::render(&t).unwrap();
            assert_eq!(got.split_text, row["rendered_txt"].as_str().unwrap());
            assert_eq!(got.split_text, row["txt"].as_str().unwrap());
            assert_eq!(got.json, row["json"].as_str().unwrap());
            assert_eq!(
                got.srt,
                row["rendered_srt"].as_str().unwrap(),
                "{}",
                row["id"]
            );
            assert_eq!(got.srt, row["srt"].as_str().unwrap());
            for (suffix, contents) in [
                (".txt", &got.split_text),
                (".json", &got.json),
                (".srt", &got.srt),
            ] {
                assert_eq!(
                    format!("{:x}", Sha256::digest(windows_text(contents))),
                    row["sha256"][suffix].as_str().unwrap(),
                    "{} {suffix}",
                    row["id"]
                );
            }
        }
    }
    #[test]
    fn decimal_does_not_split() {
        assert_eq!(
            smart_split("值3.14，正常。结束", 2),
            "值3.14，\n正常。\n结束"
        );
    }
    #[test]
    fn disabled_subtitles_do_not_require_timed_tokens() {
        let text = Transcript {
            text: "文本。".into(),
            ..Default::default()
        };
        assert!(TranscriptWriter::render_parts(&text, false).is_ok());
        assert!(TranscriptWriter::render_parts(&text, true).is_err());
    }
}

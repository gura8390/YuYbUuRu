//! Native equivalents of the installed CapsWriter text, hotword and subtitle pipeline.
#[path = "text/hotword.rs"]
mod hotword;
#[path = "text/itn.rs"]
mod itn;
#[path = "text/merge.rs"]
mod merge;
#[path = "text/writer.rs"]
mod writer;

pub use hotword::{CorrectionMatch, CorrectionResult, HotwordEngine};
pub use itn::{TextFormatter, TextOptions, adjust_space, chinese_to_num};
pub use merge::{
    MergedTokens, merge_by_text, merge_segments, sequence_matching_blocks, sync_tokens_from_text,
    tokens_to_text,
};
pub use writer::{
    RenderedTranscript, SaveOptions, Transcript, TranscriptWriter, adjust_srt, render_srt,
    smart_split,
};

pub const PUNCTUATION: &str = "，。！？；：、「」『』（）《》【】[]{},.!?;:\"'";

pub fn count_semantic_units(text: &str) -> usize {
    static RE: std::sync::LazyLock<regex::Regex> =
        std::sync::LazyLock::new(|| regex::Regex::new(r"[一-鿿㐀-䶿豈-﫿]|[a-zA-Z]+|\d+").unwrap());
    RE.find_iter(text).count()
}

/// Strip exactly the last configured mark, retaining the installed short-text/force behavior.
pub fn strip_trailing_punctuation(
    text: &str,
    marks: &str,
    threshold: usize,
    force: bool,
) -> String {
    if text.is_empty()
        || marks.is_empty()
        || (!force && threshold > 0 && count_semantic_units(text) > threshold)
    {
        return text.to_owned();
    }
    let pattern = format!(r"(?<=.)[{}]$", regex::escape(marks));
    match fancy_regex::Regex::new(&pattern) {
        Ok(re) => re.replace(text, "").into_owned(),
        Err(_) => text.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_mark_threshold_and_force() {
        assert_eq!(count_semantic_units("测试hello 123"), 4);
        assert_eq!(
            strip_trailing_punctuation("你好。", "，。,.", 8, false),
            "你好"
        );
        assert_eq!(
            strip_trailing_punctuation("一二三四五六七八九。", "，。,.", 8, false),
            "一二三四五六七八九。"
        );
        assert_eq!(
            strip_trailing_punctuation("一二三四五六七八九。", "，。,.", 8, true),
            "一二三四五六七八九"
        );
        assert_eq!(strip_trailing_punctuation("。", "，。,.", 0, false), "。");
    }
}

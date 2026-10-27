//! Weighted phoneme matching, installed RAG coarse filter, and Python-style rule replacement.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[path = "compact_dictionary.rs"]
mod compact_dictionary;
#[cfg(test)]
#[path = "../../assets/text/compact_dictionary_tests.rs"]
mod compact_dictionary_tests;
#[cfg(test)]
#[path = "../../assets/text/hotword_v02_reference.rs"]
mod legacy_reference;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorrectionMatch {
    pub original: String,
    pub hotword: String,
    pub score: f64,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrectionResult {
    pub text: String,
    pub matches: Vec<CorrectionMatch>,
    pub similars: Vec<CorrectionMatch>,
}
#[derive(Debug, Clone)]
struct Phoneme {
    value: String,
    lang: u8,
    start: bool,
    end: bool,
    tone: bool,
    char_start: usize,
    char_end: usize,
}
const SIMILAR: [(&str, &str); 17] = [
    ("an", "ang"),
    ("en", "eng"),
    ("in", "ing"),
    ("ian", "iang"),
    ("uan", "uang"),
    ("z", "zh"),
    ("c", "ch"),
    ("s", "sh"),
    ("l", "n"),
    ("f", "h"),
    ("ai", "ei"),
    ("o", "uo"),
    ("e", "ie"),
    ("p", "t"),
    ("p", "b"),
    ("t", "d"),
    ("k", "g"),
];
fn similar(a: &str, b: &str) -> bool {
    SIMILAR
        .iter()
        .any(|(x, y)| (a == *x && b == *y) || (a == *y && b == *x))
}
fn is_zh(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

fn phonemes(text: &str) -> Vec<Phoneme> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < chars.len() {
        let c = chars[i];
        if is_zh(c) {
            let mut end = i + 1;
            while end < chars.len() && is_zh(chars[end]) {
                end += 1;
            }
            while i < end {
                let (pron, n) = if let Some((pron, len)) =
                    compact_dictionary::matching_phrase(&chars[i..end])
                {
                    (Some(pron), len)
                } else {
                    (compact_dictionary::char_pronunciation(chars[i]), 1)
                };
                if let Some(pron) = pron {
                    for (k, (init, fin, tone)) in pron.syllables().enumerate().take(n) {
                        if !init.is_empty() {
                            out.push(Phoneme {
                                value: init.to_owned(),
                                lang: 0,
                                start: true,
                                end: false,
                                tone: false,
                                char_start: i + k,
                                char_end: i + k + 1,
                            });
                        }
                        if !fin.is_empty() {
                            out.push(Phoneme {
                                value: fin.to_owned(),
                                lang: 0,
                                start: init.is_empty(),
                                end: false,
                                tone: false,
                                char_start: i + k,
                                char_end: i + k + 1,
                            });
                        }
                        if !tone.is_empty() {
                            out.push(Phoneme {
                                value: tone.to_owned(),
                                lang: 0,
                                start: false,
                                end: true,
                                tone: true,
                                char_start: i + k,
                                char_end: i + k + 1,
                            });
                        }
                        if init.is_empty() && fin.is_empty() && tone.is_empty() {
                            out.push(Phoneme {
                                value: chars[i + k].to_string(),
                                lang: 0,
                                start: true,
                                end: true,
                                tone: false,
                                char_start: i + k,
                                char_end: i + k + 1,
                            });
                        }
                    }
                }
                i += n;
            }
            continue;
        }
        if c.is_ascii_alphanumeric() {
            let start = i;
            i += 1;
            while i < chars.len() && chars[i].is_ascii_alphanumeric() {
                let (p, c) = (chars[i - 1], chars[i]);
                if (p.is_ascii_lowercase() && c.is_ascii_uppercase())
                    || (p.is_ascii_alphabetic() && c.is_ascii_digit())
                    || (p.is_ascii_digit() && c.is_ascii_alphabetic())
                {
                    break;
                }
                i += 1;
            }
            let lang = if chars[start..i].iter().all(|c| c.is_ascii_digit()) {
                2
            } else {
                1
            };
            for j in start..i {
                out.push(Phoneme {
                    value: chars[j].to_ascii_lowercase().to_string(),
                    lang,
                    start: j == start,
                    end: j + 1 == i,
                    tone: chars[j].is_ascii_digit(),
                    char_start: j,
                    char_end: j + 1,
                });
            }
            continue;
        }
        i += 1;
    }
    out
}

fn lcs<T: Eq>(a: &[T], b: &[T]) -> usize {
    let mut prev = vec![0; b.len() + 1];
    for x in a {
        let mut row = vec![0; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            row[j + 1] = if x == y {
                prev[j] + 1
            } else {
                prev[j + 1].max(row[j])
            };
        }
        prev = row;
    }
    prev[b.len()]
}
fn cost(a: &Phoneme, b: &Phoneme) -> f64 {
    if a.lang != b.lang {
        return 1.;
    }
    if a.value == b.value {
        return 0.;
    }
    if a.lang == 0 && (a.tone || similar(&a.value, &b.value)) {
        return 0.5;
    }
    if a.lang == 1 {
        let x: Vec<char> = a.value.chars().collect();
        let y: Vec<char> = b.value.chars().collect();
        return 1. - lcs(&x, &y) as f64 / x.len().max(y.len()) as f64;
    }
    1.
}

fn constrained(hw: &[Phoneme], input: &[Phoneme], threshold: f64) -> Vec<(f64, usize, usize)> {
    let (n, m) = (hw.len(), input.len());
    if n == 0 || m == 0 {
        return Vec::new();
    }
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    let mut path = vec![vec![0; m + 1]; n + 1];
    for j in 0..=m {
        if j == 0 || (j < m && input[j].start) {
            dp[0][j] = 0.;
            path[0][j] = j;
        }
    }
    for i in 1..=n {
        let mut row_min = f64::INFINITY;
        for j in 1..=m {
            let match_d = dp[i - 1][j - 1] + cost(&hw[i - 1], &input[j - 1]);
            let del = dp[i - 1][j] + 1.;
            let ins = dp[i][j - 1] + 1.;
            let (v, p) = if match_d <= del {
                if match_d <= ins {
                    (match_d, path[i - 1][j - 1])
                } else {
                    (ins, path[i][j - 1])
                }
            } else if del <= ins {
                (del, path[i - 1][j])
            } else {
                (ins, path[i][j - 1])
            };
            dp[i][j] = v;
            path[i][j] = p;
            row_min = row_min.min(v);
        }
        if row_min > n as f64 * (1. - threshold) + 2. {
            break;
        }
    }
    let mut out = Vec::new();
    for j in 1..=m {
        if !input[j - 1].end {
            continue;
        }
        let dist = dp[n][j];
        if dist >= n as f64 * 0.8 {
            continue;
        }
        let score = 1. - dist / n as f64;
        if score >= threshold {
            out.push((score, path[n][j], j));
        }
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out
}

fn osa(a: &[&str], b: &[&str]) -> usize {
    let mut dp = vec![vec![0; b.len() + 1]; a.len() + 1];
    for i in 0..=a.len() {
        dp[i][0] = i;
    }
    for j in 0..=b.len() {
        dp[0][j] = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + usize::from(a[i - 1] != b[j - 1]));
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                dp[i][j] = dp[i][j].min(dp[i - 2][j - 2] + 1);
            }
        }
    }
    dp[a.len()][b.len()]
}

// RapidFuzz's MIT reference partial_ratio alignment: prefix, complete windows, suffix;
// Indel normalization is 2*LCS/(len1+len2). Reference: rapidfuzz/fuzz_py.py.
fn partial_impl<'a>(needle: &[&'a str], hay: &[&'a str]) -> (f64, usize, usize) {
    let (n, m) = (needle.len(), hay.len());
    let mut best = (0., 0, n);
    let mut consider = |start: usize, end: usize| {
        let slice = &hay[start..end];
        let score = 2. * lcs(needle, slice) as f64 / (n + slice.len()) as f64;
        if score > best.0 {
            best = (score, start, end);
        }
    };
    for i in 1..n {
        if needle.contains(&hay[i - 1]) {
            consider(0, i);
        }
    }
    for i in 0..m - n {
        if needle.contains(&hay[i + n - 1]) {
            consider(i, i + n);
        }
    }
    for i in m - n..m {
        if needle.contains(&hay[i]) {
            consider(i, m);
        }
    }
    best
}
fn partial_alignment<'a>(input: &[&'a str], hw: &[&'a str], cutoff: f64) -> Option<(usize, usize)> {
    if hw.is_empty() || input.len() < hw.len() {
        return None;
    }
    let score;
    let (mut src_start, mut src_end) = (0, input.len());
    if input.len() > hw.len() {
        let result = partial_impl(hw, input);
        score = result.0;
        src_start = result.1;
        src_end = result.2;
    } else {
        let result = partial_impl(input, hw);
        let mut current = result.0;
        if current < 1. {
            let reverse = partial_impl(hw, input);
            if reverse.0 > current {
                current = reverse.0;
                src_start = reverse.1;
                src_end = reverse.2;
            }
        }
        score = current;
    }
    (score >= cutoff).then_some((src_start, src_end))
}

#[derive(Debug, Clone)]
struct Hotword {
    target: String,
    aliases: Vec<Vec<Phoneme>>,
}
#[derive(Debug, Clone)]
struct Rule {
    pattern: String,
    regex: Option<fancy_regex::Regex>,
    replacement: String,
}
#[derive(Debug, Clone)]
pub struct HotwordEngine {
    hotwords: Vec<Hotword>,
    rules: Vec<Rule>,
    pub threshold: f64,
    pub similar_threshold: f64,
}
#[derive(Clone)]
struct Match {
    start: usize,
    end: usize,
    score: f64,
    target: String,
}

fn replacement_python(text: &str) -> String {
    let chars: Vec<char> = text.replace(r"\s", " ").chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '$' {
            out.push_str("$$");
            i += 1;
            continue;
        }
        if c != '\\' || i + 1 >= chars.len() {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        let c = chars[i];
        if c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            out.push_str("${");
            out.extend(chars[start..i].iter());
            out.push('}');
            continue;
        }
        if c == 'g' && chars.get(i + 1) == Some(&'<') {
            i += 2;
            out.push_str("${");
            while i < chars.len() && chars[i] != '>' {
                out.push(chars[i]);
                i += 1;
            }
            out.push('}');
            i += 1;
            continue;
        }
        match c {
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            '\\' => out.push('\\'),
            _ => {
                out.push('\\');
                out.push(c);
            }
        }
        i += 1;
    }
    out
}

impl HotwordEngine {
    pub fn from_contents(
        hot: &str,
        rules: &str,
        threshold: f64,
        similar_threshold: f64,
    ) -> Result<Self> {
        let mut engine = Self {
            hotwords: Vec::new(),
            rules: Vec::new(),
            threshold,
            similar_threshold,
        };
        engine.update_hotwords(hot);
        engine.update_rules(rules);
        Ok(engine)
    }
    pub fn installed_defaults() -> Result<Self> {
        Self::from_contents(
            include_str!("../../assets/text/hot_default.txt"),
            include_str!("../../assets/text/rules_default.txt"),
            0.85,
            0.6,
        )
    }
    pub fn update_hotwords(&mut self, text: &str) -> usize {
        self.hotwords.clear();
        for line in text
            .trim_start_matches('\u{feff}')
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let parts: Vec<&str> = line.split('|').map(str::trim).collect();
            let aliases: Vec<_> = parts
                .iter()
                .map(|p| phonemes(p))
                .filter(|p| !p.is_empty())
                .collect();
            if aliases.is_empty() {
                continue;
            }
            if let Some(old) = self.hotwords.iter_mut().find(|w| w.target == parts[0]) {
                old.aliases = aliases;
            } else {
                self.hotwords.push(Hotword {
                    target: parts[0].into(),
                    aliases,
                });
            }
        }
        self.hotwords.len()
    }
    pub fn update_rules(&mut self, text: &str) -> usize {
        self.rules.clear();
        for line in text.trim_start_matches('\u{feff}').lines() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<_> = line.split(" = ").collect();
            if parts.len() != 2 {
                continue;
            }
            let pattern = parts[0].trim().to_string();
            let rule = Rule {
                regex: fancy_regex::Regex::new(&pattern).ok(),
                pattern: pattern.clone(),
                replacement: replacement_python(parts[1].trim()),
            };
            if let Some(i) = self.rules.iter().position(|r| r.pattern == pattern) {
                self.rules[i] = rule;
            } else {
                self.rules.push(rule);
            }
        }
        self.rules.len()
    }
    pub fn substitute_rules(&self, text: &str) -> String {
        let mut out = text.to_owned();
        for rule in &self.rules {
            if let Some(re) = &rule.regex {
                if let Ok(next) = re.try_replacen(&out, 0, rule.replacement.as_str()) {
                    out = next.into_owned();
                }
            }
        }
        out
    }
    pub fn correct(&self, text: &str) -> CorrectionResult {
        self.correct_with_k(text, 10)
    }
    pub fn apply(&self, text: &str) -> String {
        self.substitute_rules(&self.correct(text).text)
    }
    pub fn correct_with_k(&self, text: &str, k: usize) -> CorrectionResult {
        if text.is_empty() || self.hotwords.is_empty() {
            return CorrectionResult {
                text: text.into(),
                ..Default::default()
            };
        }
        let input = phonemes(text);
        if input.is_empty() {
            return CorrectionResult {
                text: text.into(),
                ..Default::default()
            };
        }
        let chars: Vec<char> = text.chars().collect();
        let search = self.threshold.min(self.similar_threshold) - 0.1;
        let vals: Vec<&str> = input.iter().map(|p| p.value.as_str()).collect();
        let mut coarse = Vec::new();
        for word in &self.hotwords {
            for alias in &word.aliases {
                let hw: Vec<&str> = alias.iter().map(|p| p.value.as_str()).collect();
                let anchors: Vec<_> = input
                    .iter()
                    .enumerate()
                    .filter_map(|(i, p)| {
                        hw.iter()
                            .take(2)
                            .any(|v| p.value == *v || similar(&p.value, v))
                            .then_some(i)
                    })
                    .collect();
                let mut kept = Vec::new();
                for a in anchors {
                    if kept.last().is_none_or(|last| a - last > 2) {
                        kept.push(a);
                    }
                }
                for anchor in kept {
                    let start = anchor.saturating_sub(2);
                    let end = input.len().min(anchor + hw.len() + 3);
                    let local = &vals[start..end];
                    if local.len() < hw.len() {
                        continue;
                    }
                    if let Some((s, e)) = partial_alignment(local, &hw, search) {
                        let distance = osa(&local[s..e], &hw);
                        let cutoff = (hw.len() as f64 * (1. - search)) as usize;
                        if distance > cutoff {
                            continue;
                        }
                        let score = ((1. - distance as f64 / hw.len() as f64) * 1000.)
                            .round_ties_even()
                            / 1000.;
                        let endpoint = start + s + hw.len();
                        if !coarse
                            .iter()
                            .any(|(target, _, pos): &(String, f64, usize)| {
                                target == &word.target && *pos == endpoint
                            })
                        {
                            coarse.push((word.target.clone(), score, endpoint));
                        }
                    }
                }
            }
        }
        coarse.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut endpoints: Vec<(String, Vec<usize>)> = Vec::new();
        for (target, _, end) in coarse {
            if let Some((_, positions)) = endpoints.iter_mut().find(|(t, _)| t == &target) {
                if !positions.iter().any(|p| p.abs_diff(end) < 5) {
                    positions.push(end);
                }
            } else {
                endpoints.push((target, vec![end]));
            }
        }
        let mut matches = Vec::new();
        let mut similars = Vec::new();
        for (target, ends) in endpoints {
            let word = self.hotwords.iter().find(|w| w.target == target).unwrap();
            for end in ends {
                for alias in &word.aliases {
                    let start = end.saturating_sub(alias.len() + 10);
                    let stop = input.len().min(end + 5);
                    for (score, s, e) in constrained(alias, &input[start..stop], search) {
                        if e <= s {
                            continue;
                        }
                        let cs = input[start + s].char_start;
                        let ce = input[start + e - 1].char_end;
                        let original = chars[cs..ce].iter().collect::<String>();
                        if score >= self.threshold {
                            matches.push(Match {
                                start: cs,
                                end: ce,
                                score,
                                target: target.clone(),
                            });
                        }
                        if score >= self.similar_threshold {
                            similars.push(CorrectionMatch {
                                original,
                                hotword: target.clone(),
                                score,
                            });
                        }
                    }
                }
            }
        }
        similars.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(b.hotword.chars().count().cmp(&a.hotword.chars().count()))
        });
        let mut seen = HashSet::new();
        similars.retain(|m| seen.insert(m.hotword.clone()));
        similars.truncate(k);
        matches.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then((b.end - b.start).cmp(&(a.end - a.start)))
        });
        let mut occupied: Vec<(usize, usize)> = Vec::new();
        let mut final_matches = Vec::new();
        for m in matches {
            if occupied
                .iter()
                .any(|(s, e)| !(m.end <= *s || m.start >= *e))
            {
                continue;
            }
            if chars[m.start..m.end].iter().collect::<String>() != m.target {
                final_matches.push(m.clone());
            }
            occupied.push((m.start, m.end));
        }
        final_matches.sort_by(|a, b| b.start.cmp(&a.start));
        let info = final_matches
            .iter()
            .map(|m| CorrectionMatch {
                original: chars[m.start..m.end].iter().collect(),
                hotword: m.target.clone(),
                score: m.score,
            })
            .collect();
        let mut result = chars;
        for m in final_matches {
            result.splice(m.start..m.end, m.target.chars());
        }
        CorrectionResult {
            text: result.iter().collect(),
            matches: info,
            similars,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generic_fixture_hotword_cases() {
        let engine = HotwordEngine::installed_defaults().unwrap();
        let rows: serde_json::Value =
            serde_json::from_str(include_str!("../../assets/text/hotword_fixtures.json")).unwrap();
        let mut failures = Vec::new();
        for r in rows.as_array().unwrap() {
            let input = r["input"].as_str().unwrap();
            let correction = engine.correct(input);
            let got = engine.substitute_rules(&correction.text);
            if got != r["expected"].as_str().unwrap() {
                failures.push(format!("{input:?}: {got:?} != {:?}", r["expected"]));
            }
            if correction.text != r["phoneme_text"].as_str().unwrap() {
                failures.push(format!(
                    "phoneme {input:?}: {:?} != {:?}",
                    correction.text, r["phoneme_text"]
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} failures:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
    #[test]
    fn regex_backrefs_and_lookarounds() {
        let e =
            HotwordEngine::from_contents("", r"(艾特)\s*(\w+)\s*(点)\s*(\w+) = @\2.\4", 0.85, 0.6)
                .unwrap();
        assert_eq!(e.apply("艾特 foo 点 com"), "@foo.com");
    }
    #[test]
    fn configured_rules_compile_and_voice_commands_keep_newlines() {
        let engine = HotwordEngine::installed_defaults().unwrap();
        assert!(
            engine.rules.iter().all(|r| r.regex.is_some()),
            "{:?}",
            engine
                .rules
                .iter()
                .filter(|r| r.regex.is_none())
                .map(|r| &r.pattern)
                .collect::<Vec<_>>()
        );
        assert_eq!(engine.substitute_rules("回车"), "\n");
        assert_eq!(engine.substitute_rules("回车，回车"), "\n\n");
        assert_eq!(engine.substitute_rules("请回车"), "请\n");
        assert_eq!(engine.substitute_rules("你好回车测试"), "你好回车测试");
        assert_eq!(
            engine.substitute_rules("c plus plus 和 c sharp 与 dot net"),
            "C++ 和 C# 与 .NET"
        );
    }
}

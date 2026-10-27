//! Port of the installed Chinese ITN lexer, numeric parser, range/date/time reducers.
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TextOptions {
    pub format_num: bool,
    pub format_spell: bool,
}
impl Default for TextOptions {
    fn default() -> Self {
        Self {
            format_num: true,
            format_spell: true,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct TextFormatter {
    pub options: TextOptions,
}
impl TextFormatter {
    pub fn new(options: TextOptions) -> Self {
        Self { options }
    }
    pub fn format(&self, text: &str) -> String {
        let out = if self.options.format_num {
            chinese_to_num(text)
        } else {
            text.to_owned()
        };
        if self.options.format_spell {
            adjust_space(&out)
        } else {
            out
        }
    }
}

static UNITS: LazyLock<Vec<(String, Option<String>)>> = LazyLock::new(|| {
    let map: std::collections::BTreeMap<String, Option<String>> =
        serde_json::from_str(include_str!("../../assets/text/units.json")).unwrap();
    let mut rows: Vec<_> = map.into_iter().collect();
    rows.sort_by_key(|(k, _)| std::cmp::Reverse(k.chars().count()));
    rows
});
static IDIOMS: LazyLock<Vec<String>> =
    LazyLock::new(|| serde_json::from_str(include_str!("../../assets/text/idioms.json")).unwrap());
static PATTERN: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    let value: serde_json::Value =
        serde_json::from_str(include_str!("../../assets/text/itn_pattern.json")).unwrap();
    fancy_regex::Regex::new(value["pattern"].as_str().unwrap()).expect("installed ITN regex")
});

#[derive(Debug, Clone, PartialEq, Eq)]
enum Kind {
    Digit,
    Zero,
    Ten,
    Hundred,
    Thousand,
    TenThousand,
    HundredMillion,
    Dot,
    Percent,
    Fraction,
    Ratio,
    Year,
    Month,
    Day,
    Minute,
    Second,
    Other,
}
#[derive(Debug, Clone)]
struct Token {
    kind: Kind,
    value: i64,
    text: String,
}
fn numeric(t: &Token) -> bool {
    matches!(
        t.kind,
        Kind::Digit
            | Kind::Zero
            | Kind::Ten
            | Kind::Hundred
            | Kind::Thousand
            | Kind::TenThousand
            | Kind::HundredMillion
            | Kind::Dot
    )
}
fn all_numeric(t: &[Token]) -> bool {
    t.iter().all(numeric)
}
fn value(c: char) -> i64 {
    match c {
        '零' => 0,
        '一' | '幺' => 1,
        '二' | '两' => 2,
        '三' => 3,
        '四' => 4,
        '五' => 5,
        '六' => 6,
        '七' => 7,
        '八' => 8,
        '九' => 9,
        '十' => 10,
        '百' => 100,
        '千' => 1000,
        '万' => 10000,
        '亿' => 100000000,
        _ => 0,
    }
}
fn tokenize(text: &str) -> Vec<Token> {
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    while i < chars.len() {
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        let rest: String = chars[i..].iter().collect();
        let multi = if rest.starts_with("百分之") {
            Some((Kind::Percent, 3))
        } else if rest.starts_with("千分之") {
            Some((Kind::Percent, 3))
        } else if rest.starts_with("分之") {
            Some((Kind::Fraction, 2))
        } else {
            None
        };
        if let Some((kind, n)) = multi {
            out.push(Token {
                kind,
                value: 0,
                text: chars[i..i + n].iter().collect(),
            });
            i += n;
            continue;
        }
        let c = chars[i];
        let kind = match c {
            '零' => Kind::Zero,
            '一' | '二' | '两' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '幺' => {
                Kind::Digit
            }
            '十' => Kind::Ten,
            '百' => Kind::Hundred,
            '千' => Kind::Thousand,
            '万' => Kind::TenThousand,
            '亿' => Kind::HundredMillion,
            '点' => Kind::Dot,
            '比' => Kind::Ratio,
            '年' => Kind::Year,
            '月' => Kind::Month,
            '日' | '号' => Kind::Day,
            '秒' => Kind::Second,
            '分' if i + 1 == chars.len() || "零幺一二三四五六七八九十秒".contains(chars[i + 1]) => {
                Kind::Minute
            }
            _ => Kind::Other,
        };
        out.push(Token {
            kind,
            value: value(c),
            text: c.to_string(),
        });
        i += 1;
    }
    out
}

fn atomic(t: &[Token], i: usize) -> Option<(i64, usize)> {
    if i >= t.len() {
        return None;
    }
    let n = t.len();
    let kind = |j: usize, k: Kind| j < n && t[j].kind == k;
    if t[i].kind == Kind::Digit {
        let d = t[i].value;
        for (k, m) in [
            (Kind::HundredMillion, 100000000),
            (Kind::TenThousand, 10000),
            (Kind::Thousand, 1000),
        ] {
            if kind(i + 1, k) {
                return Some((d * m, 2));
            }
        }
        if kind(i + 1, Kind::Hundred) {
            let (mut base, mut consumed, mut j) = (d * 100, 2, i + 2);
            if kind(j, Kind::Zero) {
                if kind(j + 1, Kind::Digit) {
                    base += t[j + 1].value;
                    consumed += 2;
                }
            } else if kind(j, Kind::Digit) {
                let tens = t[j].value;
                if kind(j + 1, Kind::Ten) {
                    base += tens * 10;
                    consumed += 2;
                    j += 2;
                    if kind(j, Kind::Digit) {
                        base += t[j].value;
                        consumed += 1;
                    }
                } else {
                    base += tens * 10;
                    consumed += 1;
                }
            }
            return Some((base, consumed));
        }
        if kind(i + 1, Kind::Ten) && kind(i + 2, Kind::Digit) && !kind(i + 3, Kind::Ten) {
            return Some((d * 10 + t[i + 2].value, 3));
        }
        if kind(i + 1, Kind::Ten) && !(i > 0 && kind(i - 1, Kind::Digit)) {
            return Some((d * 10, 2));
        }
        return Some((d, 1));
    }
    if t[i].kind == Kind::Ten {
        if kind(i + 1, Kind::Digit) && kind(i + 2, Kind::HundredMillion) {
            return Some(((10 + t[i + 1].value) * 100000000, 3));
        }
        if kind(i + 1, Kind::HundredMillion) {
            return Some((1000000000, 2));
        }
        if kind(i + 1, Kind::Digit) {
            return Some((10 + t[i + 1].value, 2));
        }
        return Some((10, 1));
    }
    if matches!(
        t[i].kind,
        Kind::Zero | Kind::Hundred | Kind::Thousand | Kind::TenThousand | Kind::HundredMillion
    ) {
        return Some((t[i].value, 1));
    }
    None
}

fn build_number(t: &[Token], i: usize) -> Option<(i64, usize)> {
    let (mut v, mut consumed) = atomic(t, i)?;
    let mut j = i + consumed;
    if let Some(next) = t.get(j) {
        if next.kind == Kind::TenThousand && v > 0 && v < 10000 {
            v *= 10000;
            consumed += 1;
            j += 1;
        } else if next.kind == Kind::HundredMillion && v > 0 && v < 100000000 {
            v *= 100000000;
            consumed += 1;
            j += 1;
        }
    }
    let limit = if v >= 10000 {
        Some(10000)
    } else if v >= 1000 {
        Some(1000)
    } else {
        None
    };
    if let Some(limit) = limit {
        while j < t.len() {
            if t[j].kind == Kind::Zero {
                consumed += 1;
                j += 1;
                continue;
            }
            let Some((chunk, n)) = atomic(t, j) else {
                break;
            };
            if chunk >= limit {
                break;
            }
            v += chunk;
            consumed += n;
            j += n;
        }
    }
    if let Some(next) = t.get(j) {
        if next.kind == Kind::TenThousand && v > 0 && v < 10000 {
            v *= 10000;
            consumed += 1;
        } else if next.kind == Kind::HundredMillion && v > 0 && v < 100000000 {
            v *= 100000000;
            consumed += 1;
        }
    }
    Some((v, consumed))
}

fn parse_tokens(t: &[Token]) -> Option<Vec<String>> {
    if t.is_empty() || !all_numeric(t) {
        return None;
    }
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let (v, mut n) = build_number(t, i)?;
        let mut s = v.to_string();
        if t.get(i + n).is_some_and(|x| x.kind == Kind::Dot) {
            let mut k = i + n + 1;
            let mut digits = String::new();
            while k < t.len() && matches!(t[k].kind, Kind::Digit | Kind::Zero) {
                digits.push_str(&t[k].value.to_string());
                k += 1;
            }
            s.push('.');
            s.push_str(&digits);
            n = if digits.is_empty() { n + 1 } else { k - i };
        }
        out.push(s);
        i += n;
    }
    Some(out)
}
fn single(t: &[Token]) -> Option<String> {
    let v = parse_tokens(t)?;
    if v.len() == 1 {
        Some(v[0].clone())
    } else {
        None
    }
}
fn strip_unit(text: &str) -> (String, String) {
    let (mut stripped, mut unit) = (text.to_owned(), String::new());
    for (k, v) in UNITS.iter() {
        if text.ends_with(k) {
            stripped = text[..text.len() - k.len()].to_owned();
            unit = v.clone().unwrap_or_else(|| k.clone());
            break;
        }
    }
    if unit.is_empty() {
        let n = stripped
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_alphabetic())
            .count();
        if n > 0 {
            let start = stripped.len() - n;
            unit = stripped[start..].to_owned();
            stripped.truncate(start);
        }
    }
    (stripped.trim().to_owned(), unit)
}
fn pure(text: &str, strict: bool) -> Option<String> {
    let (s, u) = strip_unit(text);
    if s == "一" && !strict {
        return Some(text.into());
    }
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '零' | '一' | '幺' | '二' | '两' | '三' | '四' | '五' | '六' | '七' | '八' | '九' => {
                out.push_str(&value(c).to_string())
            }
            '点' => out.push('.'),
            _ => return None,
        }
    }
    out.push_str(&u);
    Some(out)
}
fn sequence(text: &str) -> Option<String> {
    let (mut s, mut u) = strip_unit(text);
    if u == "万" || u == "亿" {
        s = text.into();
        u.clear();
    }
    if s.is_empty() {
        return None;
    }
    let mut t = tokenize(&s);
    let mut extra = String::new();
    while t.last().is_some_and(|x| x.kind == Kind::Other) {
        extra = t.pop()?.text + &extra;
    }
    u = extra + &u;
    if t.is_empty() {
        return None;
    }
    if matches!(t.last()?.kind, Kind::TenThousand | Kind::HundredMillion) {
        if let Some(v) = parse_tokens(&t[..t.len() - 1]) {
            return Some(v.join(" ") + &t.last()?.text + &u);
        }
    }
    Some(parse_tokens(&t)?.join(" ") + &u)
}

fn range(text: &str) -> Option<String> {
    if text.contains('点') {
        return None;
    }
    let (mut s, mut physical) = (text.to_owned(), String::new());
    for (k, v) in UNITS.iter() {
        if ["万", "亿", "千", "百", "十"].contains(&k.as_str()) {
            continue;
        }
        if text.ends_with(k) {
            s = text[..text.len() - k.len()].into();
            physical = v.clone().unwrap_or_else(|| k.clone());
            break;
        }
    }
    let t = tokenize(&s);
    if t.is_empty() || t.iter().any(|x| !numeric(x) || x.kind == Kind::Dot) {
        return None;
    }
    let mut runs = Vec::new();
    let mut i = 0;
    while i < t.len() {
        if t[i].kind == Kind::Digit {
            let start = i;
            while i < t.len() && t[i].kind == Kind::Digit {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    if runs.iter().any(|(a, b)| b - a > 2) {
        return None;
    }
    let pairs: Vec<_> = runs.into_iter().filter(|(a, b)| b - a == 2).collect();
    if pairs.len() != 1 {
        return None;
    }
    let (a, b) = pairs[0];
    let (v1, v2) = (t[a].value, t[a + 1].value);
    if !(v1 < v2 && (v2 - v1 == 1 || (v1 == 3 && v2 == 5))) {
        return None;
    }
    let suffix = &t[b..];
    let big = |x: &Token| matches!(x.kind, Kind::TenThousand | Kind::HundredMillion);
    if a == 0 {
        if suffix.is_empty() {
            return Some(format!("{v1}~{v2}{physical}"));
        }
        if !matches!(
            suffix[0].kind,
            Kind::Ten | Kind::Hundred | Kind::Thousand | Kind::TenThousand | Kind::HundredMillion
        ) || !suffix[1..].iter().all(big)
        {
            return None;
        }
        let unit = &suffix[0];
        let tail = suffix[1..]
            .iter()
            .map(|t| t.text.as_str())
            .collect::<String>();
        if unit.kind == Kind::Ten {
            return Some(format!("{}~{}{tail}{physical}", v1 * 10, v2 * 10));
        }
        if big(unit) || (unit.kind == Kind::Thousand && !tail.is_empty()) {
            return Some(format!("{v1}~{v2}{}{tail}{physical}", unit.text));
        }
        return Some(format!(
            "{}~{}{tail}{physical}",
            v1 * unit.value,
            v2 * unit.value
        ));
    }
    let base = single(&t[..a])?.parse::<i64>().ok()?;
    let (mult, tail) = if suffix.first().is_some_and(|x| x.kind == Kind::Ten) {
        if !suffix[1..].iter().all(big) {
            return None;
        }
        (
            10,
            suffix[1..]
                .iter()
                .map(|t| t.text.as_str())
                .collect::<String>(),
        )
    } else {
        if !suffix.iter().all(big) {
            return None;
        }
        (
            t[a - 1].value / 10,
            suffix.iter().map(|t| t.text.as_str()).collect::<String>(),
        )
    };
    Some(format!(
        "{}~{}{tail}{physical}",
        base + v1 * mult,
        base + v2 * mult
    ))
}

fn date(t: &[Token]) -> Option<String> {
    let mut suffixes = Vec::new();
    for kind in [Kind::Year, Kind::Month, Kind::Day] {
        let ix: Vec<_> = t
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (x.kind == kind).then_some(i))
            .collect();
        if ix.len() > 1 {
            return None;
        }
        if let Some(&i) = ix.first() {
            suffixes.push(i);
        }
    }
    if suffixes.is_empty() || suffixes.windows(2).any(|p| p[0] > p[1]) {
        return None;
    }
    let mut last = 0;
    let mut out = String::new();
    for i in suffixes {
        let part = &t[last..i];
        if part.is_empty() || !all_numeric(part) {
            return None;
        }
        let s = if t[i].kind == Kind::Year
            && part
                .iter()
                .all(|x| matches!(x.kind, Kind::Digit | Kind::Zero))
        {
            pure(
                &part.iter().map(|t| t.text.as_str()).collect::<String>(),
                true,
            )?
        } else {
            single(part)?
        };
        out.push_str(&s);
        out.push_str(&t[i].text);
        last = i + 1;
    }
    if last != t.len() {
        return None;
    }
    Some(out)
}
fn zfill(s: &str) -> String {
    if s.len() < 2 {
        format!("0{s}")
    } else {
        s.into()
    }
}
fn minute_value(t: &[Token]) -> Option<String> {
    let mut v = parse_tokens(t)?;
    if v.len() == 2 && v[0] == "0" {
        v.remove(0);
    }
    if v.len() == 1 {
        Some(v[0].clone())
    } else {
        None
    }
}
fn clock(t: &[Token]) -> Option<String> {
    let dots: Vec<_> = t
        .iter()
        .enumerate()
        .filter_map(|(i, x)| (x.kind == Kind::Dot).then_some(i))
        .collect();
    let mins: Vec<_> = t
        .iter()
        .enumerate()
        .filter_map(|(i, x)| (x.kind == Kind::Minute).then_some(i))
        .collect();
    if dots.len() != 1 || mins.len() != 1 || dots[0] >= mins[0] {
        return None;
    }
    let (d, m) = (dots[0], mins[0]);
    let h = single(&t[..d])?;
    let minute = minute_value(&t[d + 1..m])?;
    let tail = &t[m + 1..];
    let mut out = format!("{}:{}", zfill(&h), zfill(&minute));
    if !tail.is_empty() {
        if tail.last()?.kind != Kind::Second {
            return None;
        }
        out.push(':');
        out.push_str(&zfill(&minute_value(&tail[..tail.len() - 1])?));
    }
    Some(out)
}
fn date_time(t: &[Token]) -> Option<String> {
    let split = t
        .iter()
        .enumerate()
        .filter_map(|(i, x)| matches!(x.kind, Kind::Day | Kind::Month | Kind::Year).then_some(i))
        .last();
    if let Some(i) = split {
        let d = date(&t[..i + 1])?;
        let time = if i + 1 < t.len() {
            clock(&t[i + 1..])?
        } else {
            String::new()
        };
        Some(d + &time)
    } else {
        clock(t)
    }
}
fn binary(t: &[Token], kind: Kind, reverse: bool, sep: &str) -> Option<String> {
    let ix: Vec<_> = t
        .iter()
        .enumerate()
        .filter_map(|(i, x)| (x.kind == kind).then_some(i))
        .collect();
    if ix.len() != 1 {
        return None;
    }
    let i = ix[0];
    let a = single(&t[..i])?;
    let b = single(&t[i + 1..])?;
    Some(if reverse {
        format!("{b}{sep}{a}")
    } else {
        format!("{a}{sep}{b}")
    })
}
fn reduce(original: &str) -> Option<String> {
    let t = tokenize(original);
    if t.first().is_some_and(|x| x.kind == Kind::Percent) {
        if let Some(v) = single(&t[1..]) {
            return Some(format!(
                "{v}{}",
                if t[0].text == "百分之" { "%" } else { "‰" }
            ));
        }
    }
    if let Some(v) = binary(&t, Kind::Fraction, true, "/") {
        return Some(v);
    }
    if let Some(v) = binary(&t, Kind::Ratio, false, ":") {
        return Some(v);
    }
    if let Some(v) = date_time(&t) {
        return Some(v);
    }
    if let Some(v) = range(original) {
        return Some(v);
    }
    let (mut s, u) = strip_unit(original);
    if u == "万" || u == "亿" {
        s = original.into();
    }
    if s.is_empty() || s.ends_with('点') {
        return None;
    }
    let stripped = tokenize(&s);
    if stripped.is_empty() || !all_numeric(&stripped) {
        return None;
    }
    if stripped
        .iter()
        .all(|t| matches!(t.kind, Kind::Digit | Kind::Zero | Kind::Dot))
    {
        pure(original, false)
    } else {
        sequence(original)
    }
}

pub fn chinese_to_num(text: &str) -> String {
    PATTERN
        .replace_all(text, |caps: &fancy_regex::Captures| {
            let original = caps.get(2).map(|x| x.as_str()).unwrap_or("");
            let head = caps.get(1).map(|x| x.as_str()).unwrap_or("");
            let span = caps.get(2).unwrap();
            let left = text[..span.start()].chars().count().saturating_sub(2);
            let right = text[..span.end()].chars().count();
            let blocked = IDIOMS.iter().any(|idiom| {
                text.find(idiom).is_some_and(|p| {
                    let pos = text[..p].chars().count();
                    pos >= left && pos < right && original.chars().count() <= idiom.chars().count()
                })
            });
            let result = if blocked || original.contains('几') {
                original.to_owned()
            } else {
                reduce(original).unwrap_or_else(|| original.to_owned())
            };
            format!("{head}{result}")
        })
        .into_owned()
}

pub fn adjust_space(text: &str) -> String {
    static RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(r"(?ix)([\u4e00-\u9fa5])?([a-z0-9+\#][a-z0-9\s'.,!+\#/_:@%&?\-]*[a-z0-9+\#%]|[a-z0-9])([\u4e00-\u9fa5])?").unwrap()
    });
    let mut out = text.to_owned();
    for _ in 0..3 {
        let next = RE
            .replace_all(&out, |caps: &regex::Captures| {
                let left = caps.get(1).map(|m| m.as_str()).unwrap_or("");
                let right = caps.get(3).map(|m| m.as_str()).unwrap_or("");
                let words: Vec<_> = caps[2].split_whitespace().collect();
                let center = if words.len() <= 1 {
                    words.first().copied().unwrap_or("").to_owned()
                } else if words
                    .iter()
                    .all(|s| s.chars().count() == 1 || s.chars().all(|c| c.is_numeric()))
                    || (words.len() == 2
                        && words[0].chars().count() == 1
                        && words[0].chars().all(|c| c.is_alphabetic())
                        && words[1].chars().next().is_some_and(|c| c.is_alphabetic()))
                {
                    words.concat()
                } else {
                    words.join(" ")
                };
                let alpha = center.chars().any(|c| c.is_ascii_alphabetic());
                format!(
                    "{left}{}{center}{}{right}",
                    if !left.is_empty() && alpha { " " } else { "" },
                    if !right.is_empty() && alpha { " " } else { "" }
                )
            })
            .into_owned();
        if next == out {
            break;
        }
        out = next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn installed_format_fixtures() {
        let rows: serde_json::Value =
            serde_json::from_str(include_str!("../../assets/text/format_fixtures.json")).unwrap();
        for r in rows.as_array().unwrap() {
            let s = r["input"].as_str().unwrap();
            assert_eq!(chinese_to_num(s), r["itn"].as_str().unwrap(), "ITN {s}");
            assert_eq!(adjust_space(s), r["space"].as_str().unwrap(), "space {s}");
            assert_eq!(
                TextFormatter::default().format(s),
                r["formatted"].as_str().unwrap(),
                "format {s}"
            );
        }
    }
}

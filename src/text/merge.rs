use super::PUNCTUATION;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MergedTokens {
    pub tokens: Vec<String>,
    pub timestamps: Vec<f64>,
}

/// difflib.SequenceMatcher matching blocks, including its final zero-sized sentinel.
/// Ties favor the earliest a index, then earliest b index, as in CPython.
pub fn sequence_matching_blocks(
    a: &[char],
    b: &[char],
    autojunk: bool,
) -> Vec<(usize, usize, usize)> {
    let mut positions: HashMap<char, Vec<usize>> = HashMap::new();
    for (i, c) in b.iter().enumerate() {
        positions.entry(*c).or_default().push(i);
    }
    if autojunk && b.len() >= 200 {
        let ntest = b.len() / 100 + 1;
        positions.retain(|_, v| v.len() <= ntest);
    }
    let longest = |alo: usize, ahi: usize, blo: usize, bhi: usize| {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0);
        let mut prev: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut row = HashMap::new();
            if let Some(js) = positions.get(&a[i]) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = if j == 0 {
                        1
                    } else {
                        prev.get(&(j - 1)).copied().unwrap_or(0) + 1
                    };
                    row.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            prev = row;
        }
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    };
    let mut queue = vec![(0, a.len(), 0, b.len())];
    let mut blocks = Vec::new();
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest(alo, ahi, blo, bhi);
        if k > 0 {
            blocks.push((i, j, k));
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    blocks.sort_unstable();
    let mut result: Vec<(usize, usize, usize)> = Vec::new();
    for (i, j, k) in blocks {
        if let Some(last) = result.last_mut() {
            if last.0 + last.2 == i && last.1 + last.2 == j {
                last.2 += k;
                continue;
            }
        }
        result.push((i, j, k));
    }
    result.push((a.len(), b.len(), 0));
    result
}

fn best_overlap(a: &[char], b: &[char]) -> Option<(usize, usize, usize)> {
    let mut best = None;
    let mut score = i64::MIN;
    for (i, j, k) in sequence_matching_blocks(a, b, false) {
        if k >= 2 && i + k > a.len() / 4 * 3 && j <= b.len() / 4 {
            let value = (k * k + i) as i64 - j as i64;
            if value > score {
                score = value;
                best = Some((i, j, k));
            }
        }
    }
    best
}

pub fn merge_by_text(previous: &str, next: &str) -> String {
    if previous.is_empty() {
        return next.into();
    }
    if next.is_empty() {
        return previous.into();
    }
    let a: Vec<char> = previous
        .trim_end_matches(|c| PUNCTUATION.contains(c))
        .chars()
        .collect();
    let b0: Vec<char> = next.chars().collect();
    let start = b0.iter().take_while(|c| PUNCTUATION.contains(**c)).count();
    let b = &b0[start..];
    if a.is_empty() || b.is_empty() {
        return format!("{previous}{next}");
    }
    let tail_start = a.len().saturating_sub(100);
    let tail = &a[tail_start..];
    let head = &b[..b.len().min(100)];
    if let Some((i, j, k)) = best_overlap(tail, head) {
        a[..tail_start + i + k]
            .iter()
            .chain(b0[start + j + k..].iter())
            .collect()
    } else {
        format!("{previous}{next}")
    }
}

fn char_to_token(tokens: &[String], base: usize, position: usize) -> usize {
    let mut count = 0;
    for (i, t) in tokens.iter().enumerate().skip(base) {
        if count >= position {
            return i;
        }
        count += t.chars().count();
    }
    tokens.len()
}

pub fn merge_segments(
    previous: &[String],
    previous_ts: &[f64],
    next: &[String],
    next_ts: &[f64],
    offset: f64,
    overlap: f64,
    first: bool,
) -> MergedTokens {
    let global: Vec<f64> = next_ts.iter().map(|t| t + offset).collect();
    if first || previous.is_empty() {
        return MergedTokens {
            tokens: next.to_vec(),
            timestamps: global,
        };
    }
    if next.is_empty() {
        return MergedTokens {
            tokens: previous.to_vec(),
            timestamps: previous_ts.to_vec(),
        };
    }
    let estimate = ((overlap * 5.) as usize).max(20);
    let tail_n = previous.len().min(estimate * 3);
    let head_n = next.len().min(estimate * 3);
    let a: Vec<char> = previous[previous.len() - tail_n..]
        .concat()
        .chars()
        .collect();
    let b: Vec<char> = next[..head_n].concat().chars().collect();
    let (cut, start, clean) = if let Some((i, j, k)) = best_overlap(&a, &b) {
        (
            char_to_token(previous, previous.len() - tail_n, i + k),
            char_to_token(next, 0, j + k),
            true,
        )
    } else {
        let last = previous_ts.last().copied().unwrap_or(offset);
        (
            previous.len(),
            global
                .iter()
                .position(|t| *t > last + 0.1)
                .unwrap_or(next.len()),
            false,
        )
    };
    let tokens: Vec<String> = previous[..cut]
        .iter()
        .chain(next[start..].iter())
        .cloned()
        .collect();
    let timestamps: Vec<f64> = previous_ts[..cut.min(previous_ts.len())]
        .iter()
        .chain(global[start.min(global.len())..].iter())
        .copied()
        .collect();
    if !clean {
        return MergedTokens { tokens, timestamps };
    }
    let mut out = MergedTokens::default();
    for (t, ts) in tokens.into_iter().zip(timestamps) {
        if out.tokens.last() == Some(&t)
            && t.chars().count() == 1
            && (PUNCTUATION.contains(&t) || t == " ")
        {
            continue;
        }
        out.tokens.push(t);
        out.timestamps.push(ts);
    }
    out
}

pub fn tokens_to_text(tokens: &[String]) -> String {
    tokens.concat().replace("@@", "")
}

fn replacement_tokens(chars: &[char]) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for &c in chars {
        if c.is_ascii_alphanumeric() {
            buf.push(c);
        } else {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push(c.to_string());
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

pub fn sync_tokens_from_text(raw: &[String], raw_ts: &[f64], formatted: &str) -> MergedTokens {
    let need_merge = raw.iter().any(|t| t.chars().count() > 1);
    let mut work = Vec::new();
    let mut times = Vec::new();
    if need_merge {
        for (t, ts) in raw.iter().zip(raw_ts) {
            for c in t.chars() {
                work.push(c.to_string());
                times.push(*ts);
            }
        }
    } else {
        work = raw.to_vec();
        times = raw_ts.to_vec();
    }
    let raw_text = work.concat();
    if raw_text == formatted || work.is_empty() {
        return MergedTokens {
            tokens: raw.to_vec(),
            timestamps: raw_ts.to_vec(),
        };
    }
    let a: Vec<char> = raw_text.chars().collect();
    let b: Vec<char> = formatted.chars().collect();
    let mut map = Vec::new();
    for (i, t) in work.iter().enumerate() {
        map.extend(std::iter::repeat_n(i, t.chars().count()));
    }
    let mut out = MergedTokens::default();
    let mut emitted = HashSet::new();
    let (mut ai, mut bi) = (0, 0);
    for (i, j, k) in sequence_matching_blocks(&a, &b, true) {
        if ai < i || bi < j {
            if ai < i {
                for r in ai..i {
                    emitted.insert(map[r]);
                }
            }
            if bi < j {
                let ts = if ai < i {
                    times.get(map[ai]).copied().unwrap_or(0.)
                } else {
                    out.timestamps
                        .last()
                        .copied()
                        .or_else(|| times.first().copied())
                        .unwrap_or(0.)
                };
                for t in replacement_tokens(&b[bi..j]) {
                    out.tokens.push(t);
                    out.timestamps.push(ts);
                }
            }
        }
        for r in i..i + k {
            let ti = map[r];
            if emitted.insert(ti) {
                out.tokens.push(work[ti].clone());
                out.timestamps.push(times.get(ti).copied().unwrap_or(0.));
            }
        }
        ai = i + k;
        bi = j + k;
    }
    if need_merge {
        let mut merged = MergedTokens::default();
        for (t, ts) in out.tokens.into_iter().zip(out.timestamps) {
            if t.is_ascii() && !t.is_empty() && t.chars().all(|c| c.is_ascii_alphanumeric()) {
                if let Some(last) = merged.tokens.last_mut() {
                    if last.is_ascii()
                        && !last.is_empty()
                        && last.chars().all(|c| c.is_ascii_alphanumeric())
                    {
                        last.push_str(&t);
                        continue;
                    }
                }
            }
            merged.tokens.push(t);
            merged.timestamps.push(ts);
        }
        merged
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn installed_merge_fixtures() {
        let rows: serde_json::Value =
            serde_json::from_str(include_str!("../../assets/text/merge_fixtures.json")).unwrap();
        for row in rows.as_array().unwrap() {
            let a = row["previous"].as_str().unwrap();
            let b = row["next"].as_str().unwrap();
            assert_eq!(
                merge_by_text(a, b),
                row["text"].as_str().unwrap(),
                "{a} / {b}"
            );
            let ss = |name: &str| serde_json::from_value::<Vec<String>>(row[name].clone()).unwrap();
            let ts = |name: &str| serde_json::from_value::<Vec<f64>>(row[name].clone()).unwrap();
            let got = merge_segments(
                &ss("previous_tokens"),
                &ts("previous_timestamps"),
                &ss("next_tokens"),
                &ts("next_timestamps"),
                row["offset"].as_f64().unwrap(),
                2.,
                row["first"].as_bool().unwrap(),
            );
            assert_eq!(got.tokens, ss("tokens"));
            assert_eq!(got.timestamps, ts("timestamps"));
        }
    }
    #[test]
    fn sync_multichar() {
        let got = sync_tokens_from_text(&["cloud".into()], &[1.], "Claude");
        assert_eq!(got.tokens.concat(), "Claude");
    }
    #[test]
    fn sync_preserves_insert_replace_and_timestamp_anchors() {
        let got = sync_tokens_from_text(
            &["三".into(), "百".into(), "元".into()],
            &[0., 0.2, 0.4],
            "300元。",
        );
        assert_eq!(got.tokens, vec!["300", "元", "。"]);
        assert_eq!(got.timestamps, vec![0., 0.4, 0.4]);
    }
}

//! Test-only frozen 0.2 oracle. None of this module or its JSON tables is in release runtime.
use super::*;
use sha2::{Digest, Sha256};

fn signature(text: &str) -> Vec<(String, u8, bool, bool, bool, usize, usize)> {
    phonemes(text)
        .into_iter()
        .map(|p| {
            (
                p.value,
                p.lang,
                p.start,
                p.end,
                p.tone,
                p.char_start,
                p.char_end,
            )
        })
        .collect()
}

fn decoded(pron: compact_dictionary::Pronunciation) -> Vec<Vec<String>> {
    pron.syllables()
        .map(|(a, b, c)| vec![a.into(), b.into(), c.into()])
        .collect()
}

#[test]
fn every_dictionary_syllable_and_phrase_order_matches_v02() {
    let chars = legacy_reference::fixture_chars();
    let groups = legacy_reference::fixture_phrases();
    assert_eq!(chars.len(), 20992);
    assert_eq!(groups.values().map(Vec::len).sum::<usize>(), 47111);
    let data = compact_dictionary::static_data();
    assert_eq!(&data[..8], b"CWPDIC1\0");
    assert_eq!(data.len(), 1481268);
    let mut syllables = 0;
    let mut empty = 0;
    let mut phrase_count = 0;
    for cp in 0x4e00..=0x9fff {
        let c = char::from_u32(cp).unwrap();
        let expected = &chars[&c.to_string()];
        let got = decoded(compact_dictionary::char_pronunciation(c).unwrap());
        assert_eq!(&got, expected, "char U+{cp:04X}");
        assert_eq!(
            signature(&c.to_string()),
            legacy_reference::fixture_phonemes(&c.to_string()),
            "single char including Some(empty) U+{cp:04X}"
        );
        syllables += got.len();
        empty += usize::from(got.is_empty());
        let rows: Vec<_> = compact_dictionary::phrase_entries(c).collect();
        let old = groups.get(&c).map(Vec::as_slice).unwrap_or(&[]);
        assert_eq!(rows.len(), old.len(), "group {c}");
        for ((word, pron), (expected_word, expected_pron)) in rows.into_iter().zip(old) {
            assert_eq!(
                word.chars().collect::<Vec<_>>(),
                *expected_word,
                "ordered group {c}"
            );
            let got = decoded(pron);
            assert_eq!(&got, expected_pron, "phrase {word}");
            assert!(
                word.as_ptr() >= data.as_ptr()
                    && word.as_ptr() < data.as_ptr().wrapping_add(data.len())
            );
            assert_eq!(
                signature(word),
                legacy_reference::fixture_phonemes(word),
                "phonemes {word}"
            );
            let prefix: String = word.chars().take(word.chars().count() - 1).collect();
            let boundary = format!("{prefix}A{word}😀{prefix}");
            assert_eq!(
                signature(&boundary),
                legacy_reference::fixture_phonemes(&boundary),
                "truncated phrase and non-Han boundaries {word}"
            );
            let word_chars: Vec<char> = word.chars().collect();
            assert_eq!(
                compact_dictionary::matching_phrase(&word_chars).unwrap().1,
                word_chars.len(),
                "longest exact word takes priority over its prefixes {word}"
            );
            syllables += got.len();
            phrase_count += 1;
        }
    }
    assert_eq!((phrase_count, syllables, empty), (47111, 164785, 68));
    assert!(compact_dictionary::char_pronunciation('A').is_none());
    assert!(compact_dictionary::char_pronunciation('\u{20000}').is_none());
}

fn next(seed: &mut u64) -> usize {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed as usize
}

#[test]
fn long_random_mixed_and_polyphonic_v02_regression() {
    let old = legacy_reference::HotwordEngine::installed_defaults().unwrap();
    let new = HotwordEngine::installed_defaults().unwrap();
    let mut words: Vec<String> = legacy_reference::fixture_phrases()
        .values()
        .flat_map(|rows| rows.iter().map(|(w, _)| w.iter().collect()))
        .collect();
    words.sort();
    let specials = [
        "银行行长重新重载重庆长大音乐快乐单于解读",
        "长长长长重重重重行行行行乐乐乐乐",
        "一模一样地久天长着重着手薄荷薄饼参差不齐",
        "𠀀〇㐀\u{9fff}\u{4e00}",
        "CapsWriter Qwen3ASR v0.2 HTTPServer UTF8 20261027",
        "C++ C# .NET 艾特foo点com 回车，回车",
        "你好\nhelloWorld123中文ABC99done\t再见",
        "春天的公园里有鲜花和绿色树木",
        "\0\u{301}😀éÜß\r\n！？，。[]()",
        "微软小米新建文件夹谷歌浏览器ChatGPT",
    ];
    let mut seed = 0x20261027c0ffee_u64;
    let mut cases: Vec<String> = specials.iter().map(|s| s.to_string()).collect();
    for n in 0..128 {
        let target = if n < 96 {
            64 + next(&mut seed) % 960
        } else {
            2048 + next(&mut seed) % 2048
        };
        let mut text = String::new();
        while text.chars().count() < target {
            match next(&mut seed) % 5 {
                0 => text.push_str(specials[next(&mut seed) % specials.len()]),
                1 => text.push(char::from_u32(0x4e00 + (next(&mut seed) % 20992) as u32).unwrap()),
                _ => text.push_str(&words[next(&mut seed) % words.len()]),
            }
        }
        cases.push(text);
    }
    let mut digest = Sha256::new();
    for (index, text) in cases.iter().enumerate() {
        let got = signature(text);
        let expected = legacy_reference::fixture_phonemes(text);
        assert_eq!(got, expected, "phoneme case {index}");
        digest.update(text.as_bytes());
        // Long 2k-4k strings check complete phoneme segmentation; shorter mixed strings
        // additionally run the unchanged scorer/tie/conflict/rule path against frozen 0.2.
        if index < 58 {
            let got = new.correct_with_k(text, 8);
            let expected = old.correct_with_k(text, 8);
            assert_eq!(
                serde_json::to_value(&got).unwrap(),
                serde_json::to_value(&expected).unwrap(),
                "complete correction including scores/match order case {index}"
            );
            assert_eq!(
                new.substitute_rules(&got.text),
                old.substitute_rules(&expected.text),
                "final rules case {index}"
            );
        }
    }
    println!(
        "compact dictionary regression: 138 mixed/polyphonic cases; 58 exact correction/score/match-order cases; corpus_sha256={:x}",
        digest.finalize()
    );
}

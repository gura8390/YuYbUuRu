//! Borrowed pronunciation tables: no runtime JSON parsing or dictionary heap allocation.
//! Fixed-width little-endian byte decoding avoids alignment, layout, and pointer assumptions.
#[path = "../../assets/text/pinyin_compact_meta.rs"]
mod parts;

const DATA: &[u8] = include_bytes!("../../assets/text/pinyin_compact.bin");
const BASE: u32 = 0x4e00;
const CHAR_COUNT: usize = 20992;
const CHAR_OFFSET: usize = 44;
const RANGE_OFFSET: usize = CHAR_OFFSET + CHAR_COUNT * 3;
const RECORD_OFFSET: usize = RANGE_OFFSET + (CHAR_COUNT + 1) * 4;
const RECORD_SIZE: usize = 10;

fn u32_at(offset: usize) -> usize {
    u32::from_le_bytes(DATA[offset..offset + 4].try_into().unwrap()) as usize
}

#[derive(Clone, Copy)]
pub(super) struct Pronunciation(&'static [u8]);
impl Pronunciation {
    pub(super) fn syllables(self) -> impl Iterator<Item = (&'static str, &'static str, &'static str)> {
        self.0.chunks_exact(3).map(|p| {
            (parts::INITIALS[p[0] as usize], parts::FINALS[p[1] as usize], parts::TONES[p[2] as usize])
        })
    }
}

pub(super) fn char_pronunciation(c: char) -> Option<Pronunciation> {
    let index = (c as u32).checked_sub(BASE)? as usize;
    if index >= CHAR_COUNT {
        return None;
    }
    let offset = CHAR_OFFSET + index * 3;
    let bytes = &DATA[offset..offset + 3];
    // The legacy table has 68 present but empty pronunciations; retain those entries.
    Some(Pronunciation(if bytes[0] == 255 { &[] } else { bytes }))
}

fn phrase_range(first: char) -> std::ops::Range<usize> {
    let Some(index) = (first as u32).checked_sub(BASE).map(|i| i as usize) else {
        return 0..0;
    };
    if index >= CHAR_COUNT {
        return 0..0;
    }
    u32_at(RANGE_OFFSET + index * 4)..u32_at(RANGE_OFFSET + (index + 1) * 4)
}

fn phrase(row: usize) -> (&'static str, usize, Pronunciation) {
    let record = RECORD_OFFSET + row * RECORD_SIZE;
    let word_start = u32_at(32) + u32_at(record);
    let byte_len = DATA[record + 4] as usize;
    let char_len = DATA[record + 5] as usize;
    let pron_start = u32_at(36) + u32_at(record + 6);
    let word = std::str::from_utf8(&DATA[word_start..word_start + byte_len]).unwrap();
    (word, char_len, Pronunciation(&DATA[pron_start..pron_start + char_len * 3]))
}

pub(super) fn matching_phrase(chars: &[char]) -> Option<(Pronunciation, usize)> {
    let first = *chars.first()?;
    for row in phrase_range(first) {
        let (word, len, pron) = phrase(row);
        if len <= chars.len() && word.chars().eq(chars[..len].iter().copied()) {
            return Some((pron, len));
        }
    }
    None
}

#[cfg(test)]
pub(super) fn phrase_entries(first: char) -> impl Iterator<Item = (&'static str, Pronunciation)> {
    phrase_range(first).map(|row| {
        let (word, _, pron) = phrase(row);
        (word, pron)
    })
}

#[cfg(test)]
pub(super) fn static_data() -> &'static [u8] {
    DATA
}

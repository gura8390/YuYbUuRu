Native text pipeline data

The application implements hotword correction, number formatting, token merging,
and TXT/JSON/SRT rendering in Rust. The running application does not execute
Python. Personal hotwords and replacement rules are read from the external
application directory; hot_default.txt contains public example terms only.

Data and provenance

The pronunciation JSON tables were exported from pypinyin behavior and retain
20,992 basic Han character entries and 47,111 phrase pronunciations. The original
JSON tables remain the complete data source and the independent dictionary test
oracle. Character data originated in pinyin-data, and phrase data originated in
phrase-pinyin-data. Their upstream projects and MIT notices are retained with
the third-party license material:

  https://github.com/mozillazg/python-pinyin
  https://github.com/mozillazg/pinyin-data
  https://github.com/mozillazg/phrase-pinyin-data

The text pipeline and generic ITN unit, idiom, and pattern data were ported from
CapsWriter-Offline. The upstream MIT notice is also retained in
upstream-CapsWriter-LICENSE.txt:

  https://github.com/HaujetZhao/CapsWriter-Offline

The partial-alignment algorithm refers to RapidFuzz's MIT-licensed implementation:

  https://github.com/rapidfuzz/RapidFuzz

Dictionary representation

The runtime embeds pinyin_compact.bin and pinyin_compact_meta.rs as read-only
program data. It does not parse the JSON pronunciation tables into a permanent
heap dictionary. The packed dictionary retains all 164,785 syllables and all
68 character entries whose pronunciation list is empty.

The 1,481,268-byte binary has a 44-byte little-endian header, dense character
pronunciations, first-character phrase ranges, phrase records, a UTF-8 word blob,
and interned pronunciation parts. Phrases are ordered by first character, then
descending character length, then Unicode word order. Three 255 sentinel bytes
represent an existing character with an empty pronunciation list.

Regenerate or check the deterministic assets from the project root:

  python assets/text/generate_compact_dictionary.py
  python assets/text/generate_compact_dictionary.py --check

compact_dictionary_receipt.json records source and output hashes and the full
round-trip checks. It contains only general dictionary statistics and hashes.

Tests

Run the project tests with:

  .\build.ps1 -Tests

The four JSON fixture files contain hand-authored synthetic examples only:

  hotword_fixtures.json: 17 generic terms, aliases, and replacement-rule cases.
  format_fixtures.json: 17 numeric, unit, spacing, spelling, and idiom cases.
  merge_fixtures.json: 8 empty, overlap, offset, time-gate, punctuation, and
    multi-character-token cases.
  writer_fixtures.json: 4 single-line, multi-line, decimal, and empty transcripts.

The expected outputs are specified independently of the Rust implementation.
Writer fixtures include exact TXT/JSON/SRT strings and SHA256 digests calculated
over UTF-8 bytes after Windows CRLF conversion. They contain no recorded speech,
personal contact details, application installation inventory, or user hotwords.

compact_dictionary_tests.rs compares every character and phrase, syllable order,
phrase pronunciation, and selected mixed-text behavior against the test-only
hotword_v02_reference.rs and the complete JSON tables. The legacy dictionary
implementation and JSON tables are compiled only for tests. Additional unit
tests cover timestamp anchors, punctuation policies, decimal splitting, and
disabled subtitle output.

These tests establish the specified data and text contracts. They do not test
microphones, native model inference, physical shortcuts, real text insertion,
device reconnection, or application memory consumption. Tests that access
devices, models, or foreground input are explicitly ignored by default.

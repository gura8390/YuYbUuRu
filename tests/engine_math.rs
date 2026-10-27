use std::path::PathBuf;
use yuybuuru::engine::MelExtractor;

fn floats(path: PathBuf) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
#[test]
fn mel_matches_shipped_numpy_fixture() {
    let root = std::env::var_os("CAPS_ENGINE_FIXTURE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let dir = root.join("assets/fixtures/engine-static");
    assert!(dir.join("manifest.json").is_file(), "Synthetic math references must be present");
    let mel = MelExtractor::new();
    for id in ["silence", "sine", "noise", "short", "end_impulse"] {
        let audio = floats(dir.join(format!("{id}.pcm.f32")));
        let expected = floats(dir.join(format!("{id}.mel.f32")));
        let (actual, _) = mel.extract(&audio).unwrap();
        assert_eq!(actual.len(), expected.len());
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        eprintln!("mel fixture {id}: max_abs_error={error:e}");
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{id} mel output differs at the float32 bit level; max error {error}"
        );
    }
}

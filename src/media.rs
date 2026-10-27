use anyhow::{Context, Result, ensure};
use std::os::windows::process::CommandExt;
use std::{
    io::{BufReader, Read},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

const CREATE_NO_WINDOW: u32 = 0x08000000;
pub fn ffmpeg_path(app: &Path) -> PathBuf {
    app.join("ffmpeg.exe")
}
pub fn stream_audio(
    app: &Path,
    file: &Path,
    mut consume: impl FnMut(&[f32]) -> Result<()>,
) -> Result<f64> {
    stream_audio_cancellable(app, file, Arc::new(AtomicBool::new(false)), &mut consume)
}
pub fn stream_audio_cancellable(
    app: &Path,
    file: &Path,
    cancel: Arc<AtomicBool>,
    mut consume: impl FnMut(&[f32]) -> Result<()>,
) -> Result<f64> {
    ensure!(file.is_file(), "Input does not exist: {}", file.display());
    let mut child = Command::new(ffmpeg_path(app))
        .arg("-nostdin")
        .arg("-v")
        .arg("error")
        .arg("-i")
        .arg(file)
        .args(["-f", "f32le", "-ac", "1", "-ar", "16000", "-"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
        .context("Start bundled FFmpeg")?;
    let _guard = crate::process_guard::ProcessGuard::watch(&child, cancel, Duration::ZERO)?;
    let result = (|| {
        let mut stream = BufReader::new(child.stdout.take().context("FFmpeg stdout")?);
        let mut bytes = [0u8; 16384];
        let mut carried = Vec::new();
        let mut count = 0usize;
        loop {
            let n = stream.read(&mut bytes)?;
            if n == 0 {
                break;
            }
            carried.extend_from_slice(&bytes[..n]);
            let full = carried.len() / 4 * 4;
            let pcm: Vec<f32> = carried[..full]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            count += pcm.len();
            consume(&pcm)?;
            carried.drain(..full);
        }
        ensure!(
            carried.is_empty(),
            "FFmpeg returned an incomplete PCM sample"
        );
        let status = child.wait()?;
        ensure!(status.success(), "FFmpeg failed with {status}");
        Ok(count as f64 / 16000.0)
    })();
    if child.try_wait()?.is_none() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}
pub fn read_audio(app: &Path, file: &Path) -> Result<Vec<f32>> {
    if file.extension().is_some_and(|v| v == "f32") {
        let data = std::fs::read(file)?;
        ensure!(data.len() % 4 == 0, "Invalid raw f32 audio");
        return Ok(data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect());
    }
    let mut out = Vec::new();
    stream_audio(app, file, |chunk| {
        out.extend_from_slice(chunk);
        Ok(())
    })?;
    Ok(out)
}

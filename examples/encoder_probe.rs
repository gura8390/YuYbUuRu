//! CPU encoder-only workload. Run through an owned-process benchmark launcher.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
pub use yuybuuru::engine::{EncoderProvider, EngineOptions};
fn load_timing(stage: &'static str, started: Instant) {
    if std::env::var_os("YYUR_PROFILE_LOAD").is_some() {
        eprintln!(
            "YYUR_ENCODER_LOAD {}",
            json!({"stage":stage,"seconds":started.elapsed().as_secs_f64()})
        );
    }
}
#[path = "../src/engine/encoder.rs"]
mod encoder;
#[path = "../src/engine/mel.rs"]
#[allow(dead_code)]
mod mel;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    app_dir: PathBuf,
    /// JSON array of {id,pcm,pcm_sha256}; order and repetition are explicit.
    #[arg(long)]
    manifest: PathBuf,
    /// Optional JSON Settings, passed through the normal engine-options path.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Existing probe report for full output SHA/shape regression.
    #[arg(long)]
    reference: Option<PathBuf>,
    #[arg(long, default_value_t=1, value_parser=clap::value_parser!(u32).range(1..=20))]
    repeats: u32,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Deserialize)]
struct Input {
    id: String,
    pcm: PathBuf,
    pcm_sha256: String,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.output.exists(),
        "Refusing to overwrite an existing report"
    );
    let app = args.app_dir.canonicalize()?;
    let config = args.config.unwrap_or_else(|| app.join("settings.json"));
    let settings = yuybuuru::config::Settings::load(Some(&config), Some(&app))?;
    ensure!(
        settings.encoder_provider == EncoderProvider::Cpu,
        "This probe accepts CPU encoders only"
    );
    let options = yuybuuru::pipeline::engine_options(&settings);
    let dll = match &options.onnx_dll {
        Some(p) if p.is_absolute() => p.clone(),
        Some(p) => app.join(p),
        None => app.join("internal/onnxruntime/capi/onnxruntime.dll"),
    };
    let model = options
        .asr_model_dir
        .clone()
        .unwrap_or_else(|| app.join("models/Qwen3-ASR/Qwen3-ASR-0.6B"));
    let inputs: Vec<Input> = serde_json::from_slice(&fs::read(&args.manifest)?)?;
    ensure!(!inputs.is_empty(), "Empty manifest");
    let mut audio = Vec::new();
    for item in &inputs {
        let bytes = fs::read(&item.pcm).with_context(|| format!("Read local input {}", item.id))?;
        ensure!(
            hash(&bytes) == item.pcm_sha256,
            "PCM hash mismatch for {}",
            item.id
        );
        ensure!(
            !bytes.is_empty() && bytes.len() % 4 == 0,
            "Invalid float32 PCM"
        );
        let values = bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect::<Vec<_>>();
        ensure!(values.iter().all(|x| x.is_finite()), "Nonfinite PCM");
        audio.push(values);
    }
    let settings_hash = hash_file(&config)?;
    let ort_hash = hash_file(&dll)?;
    let frontend_hash = hash_file(&model.join("qwen3_asr_encoder_frontend.int4.onnx"))?;
    let backend_hash = hash_file(&model.join("qwen3_asr_encoder_backend.int4.onnx"))?;
    let manifest_hash = hash_file(&args.manifest)?;
    let tick = Instant::now();
    let mut encoder = encoder::AudioEncoder::load(
        &dll,
        &model,
        "qwen3_asr_encoder_frontend.int4.onnx",
        "qwen3_asr_encoder_backend.int4.onnx",
        &options,
    )?;
    let load_seconds = tick.elapsed().as_secs_f64();
    let warmup_metrics = encoder.last_metrics().clone();
    let expected = args
        .reference
        .as_ref()
        .map(|path| -> Result<serde_json::Value> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
        .transpose()?;
    if let Some(old) = &expected {
        ensure!(
            old["complete"] == true && old["cpu_only"] == true,
            "Reference must be a completed CPU encoder run"
        );
        ensure!(
            old["manifest_sha256"] == manifest_hash,
            "Reference manifest identity mismatch"
        );
        ensure!(
            old["samples"]
                .as_array()
                .context("Reference samples")?
                .len()
                == inputs.len() * args.repeats as usize,
            "Reference workload length mismatch"
        );
    }
    let mut rows = Vec::new();
    let mut exact = true;
    for repeat in 0..args.repeats {
        for (index, (item, audio)) in inputs.iter().zip(&audio).enumerate() {
            let tick = Instant::now();
            let feature = encoder.encode(audio)?;
            let seconds = tick.elapsed().as_secs_f64();
            let mut digest = Sha256::new();
            for value in &feature.data {
                digest.update(value.to_le_bytes());
            }
            let sha = format!("{:x}", digest.finalize());
            let mut matched = None;
            if let Some(old) = &expected {
                let before = &old["samples"][repeat as usize * inputs.len() + index];
                ensure!(
                    before["id"] == item.id && before["pcm_sha256"] == item.pcm_sha256,
                    "Reference input order/hash mismatch"
                );
                let equal = before["embedding_sha256"] == sha
                    && before["embedding_values"] == feature.data.len()
                    && before["embedding_dimension"] == feature.dim;
                exact &= equal;
                matched = Some(equal);
            }
            ensure!(
                feature.data.iter().all(|v| v.is_finite()),
                "Nonfinite encoder result"
            );
            rows.push(json!({"id":item.id,"repeat":repeat,"pcm_sha256":item.pcm_sha256,
                "embedding_sha256":sha,"embedding_values":feature.data.len(),"embedding_dimension":feature.dim,
                "encode_seconds":seconds,"stages":encoder.last_metrics(),"output_exact":matched}));
        }
    }
    drop(encoder);
    let result = json!({"complete":true,"outputs_exact":expected.as_ref().map(|_|exact),"cpu_only":true,
        "settings":settings,"settings_sha256":settings_hash,"ort_sha256":ort_hash,
        "frontend_sha256":frontend_hash,"backend_sha256":backend_hash,
        "manifest_sha256":manifest_hash,"load_and_warmup_seconds":load_seconds,
        "load_warmup_metrics":warmup_metrics,"samples":rows,"no_decoder_or_sampling":true});
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?
        .write_all(serde_json::to_string_pretty(&result)?.as_bytes())?;
    ensure!(exact, "Encoder float32 output changed; report retained");
    println!(
        "{}",
        json!({"complete":true,"output_exact":result["outputs_exact"],"samples":rows.len()})
    );
    Ok(())
}

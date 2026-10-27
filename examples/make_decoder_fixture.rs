//! Build the production auto-language, empty-context prompt from an existing
//! encoder matrix. Tokens are an external workload, never generated here.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy, Debug)]
pub enum GpuBackend {
    Cuda,
    Vulkan,
}
fn load_timing(stage: &'static str, started: Instant) {
    eprintln!(
        "YYUR_FIXTURE_LOAD {}",
        json!({"stage":stage,"seconds":started.elapsed().as_secs_f64()})
    );
}
#[path = "../src/engine/gguf.rs"]
#[allow(dead_code)]
mod gguf;
#[path = "../src/engine/llama.rs"]
#[allow(dead_code)]
mod llama;
use gguf::EmbeddingTable;
use llama::NativeModel;

#[derive(Parser)]
struct Args {
    /// Runtime containing the original Vulkan/CPU DLLs and ASR GGUF.
    #[arg(long)]
    app_dir: PathBuf,
    /// Bare encoder output, little-endian float32 [N,1024], without a NPY header.
    #[arg(long)]
    embedding_f32: PathBuf,
    /// Existing accepted Vulkan sample's token_ids array; no sampling occurs.
    #[arg(long)]
    tokens_json: PathBuf,
    /// Must not already exist. Contains prefill.f32, tokens.json, metadata.json.
    #[arg(long)]
    output_dir: PathBuf,
}
fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}
fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
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
fn save_new(path: &Path, bytes: &[u8]) -> Result<()> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    let app = args.app_dir.canonicalize()?;
    let embedding_path = args.embedding_f32.canonicalize()?;
    let token_path = args.tokens_json.canonicalize()?;
    ensure!(
        fs::metadata(&embedding_path)?.len() <= 4096 * 1024 * 4,
        "Encoder matrix exceeds the supported context window"
    );
    ensure!(
        fs::metadata(&token_path)?.len() <= 1024 * 1024,
        "Token JSON is unexpectedly large"
    );
    let feature_bytes = fs::read(&embedding_path)?;
    ensure!(
        !feature_bytes.starts_with(b"\x93NUMPY"),
        "Use a bare encoder .f32 matrix, not a NPY container"
    );
    ensure!(
        !feature_bytes.is_empty() && feature_bytes.len() % (1024 * 4) == 0,
        "Encoder input must have complete nonempty 1024-float rows"
    );
    let feature = feature_bytes
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(
        feature.iter().all(|v| v.is_finite()),
        "Encoder matrix contains nonfinite values"
    );
    let token_bytes = fs::read(&token_path)?;
    let tokens: Vec<i32> = serde_json::from_slice(&token_bytes)
        .context("Expected external JSON array of i32 tokens")?;
    ensure!(
        !tokens.is_empty() && tokens.iter().all(|&t| t >= 0),
        "Token workload must be nonempty and nonnegative"
    );
    ensure!(
        !args.output_dir.exists(),
        "Refusing to replace an existing fixture directory"
    );
    let weights = app.join("models/Qwen3-ASR/Qwen3-ASR-0.6B/qwen3_asr_llm.q4_k.gguf");
    let weights_sha = hash_file(&weights)?;
    // CPU-only weight loading supplies tokenization and the original GGML
    // dequantizer. No NativeContext, encoder, sampler or decode is constructed.
    let model = NativeModel::load_with_backend(
        &app.join("core/server/engines/llama/bin"),
        &weights,
        false,
        GpuBackend::Vulkan,
    )?;
    let vocab = model.vocab_size()?;
    ensure!(
        model.dim == 1024 && tokens.iter().all(|&t| t < vocab),
        "Dimension/token vocabulary mismatch"
    );
    let table = EmbeddingTable::open(&weights, model.api.clone())?;
    ensure!(table.dim == model.dim, "GGUF embedding dimension mismatch");
    let im_start = model.token_id("<|im_start|>")?;
    let im_end = model.token_id("<|im_end|>")?;
    let audio_start = model.token_id("<|audio_start|>")?;
    let audio_end = model.token_id("<|audio_end|>")?;
    let asr_text = model.token_id("<asr_text>")?;
    // Keep this order identical to Engine::transcribe_with_embedding for
    // language="auto", context="". MRoPE positions are added by replay.
    let mut pre = vec![im_start];
    pre.extend(model.tokenize("system\nYou are a helpful assistant.")?);
    pre.extend([im_end, im_start]);
    pre.extend(model.tokenize("user\n")?);
    pre.push(audio_start);
    let mut post = vec![audio_end, im_end, im_start];
    post.extend(model.tokenize("assistant\n")?);
    post.push(asr_text);
    let mut full = table.rows(&pre)?;
    full.extend_from_slice(&feature);
    full.extend(table.rows(&post)?);
    ensure!(
        full.iter().all(|v| v.is_finite()),
        "Full prompt contains nonfinite values"
    );
    let n_prefill = full.len() / model.dim;
    ensure!(
        n_prefill + tokens.len() <= 4096,
        "Prompt plus external workload exceeds context4096"
    );
    let full_bytes = full
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let canonical_tokens = serde_json::to_vec(&tokens)?;
    let metadata = json!({
        "schema":1,"prompt_policy":{"language":"auto","context":"","production_order":true},
        "input":{"embedding_path":embedding_path,"embedding_sha256":hash(&feature_bytes),
            "embedding_rows":feature.len()/1024,"tokens_path":token_path,
            "tokens_file_sha256":hash(&token_bytes),"token_ids_sha256":hash(&canonical_tokens)},
        "model":{"path":weights,"sha256":weights_sha,"dimension":model.dim,"vocabulary":vocab,
            "load_gpu":false,"backend_selection":"CPU model in Vulkan runtime","decode_calls":0,"encoder_calls":0},
        "prompt":{"n_prefill":n_prefill,"prefix_tokens":pre,"suffix_tokens":post,
            "prefix_rows":pre.len(),"suffix_rows":post.len(),"audio_rows":feature.len()/1024,
            "format":"raw-le-f32","shape":[n_prefill,1024],"sha256":hash(&full_bytes)},
        "external_workload":{"n_tokens":tokens.len(),"token_ids_sha256":hash(&canonical_tokens),
            "origin":"caller supplied; no expected tokens generated by fixture or replay"},
        "quality_equivalence":"not evaluated by this fixture builder"
    });
    if let Some(parent) = args
        .output_dir
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&args.output_dir)?;
    save_new(&args.output_dir.join("prefill.f32"), &full_bytes)?;
    save_new(&args.output_dir.join("tokens.json"), &canonical_tokens)?;
    save_new(
        &args.output_dir.join("metadata.json"),
        &serde_json::to_vec_pretty(&metadata)?,
    )?;
    println!("{}", args.output_dir.join("metadata.json").display());
    Ok(())
}

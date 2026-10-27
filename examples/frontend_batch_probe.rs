//! CPU-only frontend batching experiment. No decoder, backend encoder,
//! sampling or ASR expected-output generation is involved.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use ort::{
    logging::LogLevel,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};
#[path = "../src/engine/mel.rs"]
#[allow(dead_code)]
mod mel;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    onnx_dll: PathBuf,
    #[arg(long)]
    frontend_source: PathBuf,
    #[arg(long)]
    frontend_candidate: PathBuf,
    /// Defaults to candidate filename with .receipt.json extension.
    #[arg(long)]
    receipt: Option<PathBuf>,
    /// Raw little-endian float32 mono PCM at 16000Hz, at most 80 seconds.
    #[arg(long)]
    pcm_f32: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "2,4,8")]
    batch_sizes: Vec<usize>,
    /// Hot whole-input passes, in addition to one first-session pass.
    #[arg(long, default_value_t=3, value_parser=clap::value_parser!(u32).range(1..=50))]
    repeats: u32,
    #[arg(long, default_value_t = 8)]
    intra_threads: usize,
    #[arg(long)]
    profile_dir: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}

fn builder_error<R>(error: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("{error}")
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn float_bytes(values: &[f32]) -> &[u8] {
    // Windows x64 is little endian. A live f32 slice has initialized bytes and
    // no padding; the returned byte slice cannot outlive the original data.
    assert!(cfg!(target_endian = "little"));
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
fn save(path: &Path, value: &Value) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn session(path: &Path, args: &Args, tag: &str) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_intra_threads(args.intra_threads)
        .map_err(builder_error)?
        .with_inter_threads(1)
        .map_err(builder_error)?
        .with_parallel_execution(false)
        .map_err(builder_error)?
        .with_log_level(LogLevel::Error)
        .map_err(builder_error)?
        .with_config_entry("session.intra_op.allow_spinning", "0")
        .map_err(builder_error)?
        .with_config_entry("session.inter_op.allow_spinning", "0")
        .map_err(builder_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(builder_error)?
        .with_memory_pattern(true)
        .map_err(builder_error)?
        .with_prepacking(true)
        .map_err(builder_error)?
        .with_execution_providers(vec![
            ort::ep::CPU::default().with_arena_allocator(false).build(),
        ])
        .map_err(builder_error)?;
    if let Some(directory) = args.profile_dir.as_ref() {
        fs::create_dir_all(directory)?;
        builder = builder
            .with_profiling(directory.join(format!("{}-{tag}", std::process::id())))
            .map_err(builder_error)?;
    }
    builder
        .commit_from_file(path)
        .with_context(|| format!("Load CPU frontend {}", path.display()))
}

#[derive(Serialize)]
struct Comparison {
    compared_values: usize,
    differing_float_bits: usize,
    max_absolute_error_finite: f64,
    all_finite: bool,
    first_difference: Option<Value>,
}
#[derive(Serialize)]
struct Pass {
    kind: String,
    calls: usize,
    ort_run_seconds: f64,
    wall_seconds_including_validation: f64,
    call_seconds: Vec<f64>,
    input_shapes: Vec<[usize; 3]>,
    output_shapes: Vec<Vec<i64>>,
    output_values: usize,
    output_sha256: String,
    all_finite: bool,
    comparison: Option<Comparison>,
}
fn run_pass(
    session: &mut Session,
    packed: &[f32],
    chunks: usize,
    batch_size: usize,
    expected: Option<&[f32]>,
    kind: String,
) -> Result<(Pass, Vec<f32>)> {
    const INPUT_CHUNK: usize = 128 * 100;
    const OUTPUT_CHUNK: usize = 13 * 896;
    let started = Instant::now();
    let mut report = Pass {
        kind,
        calls: 0,
        ort_run_seconds: 0.0,
        wall_seconds_including_validation: 0.0,
        call_seconds: Vec::new(),
        input_shapes: Vec::new(),
        output_shapes: Vec::new(),
        output_values: 0,
        output_sha256: String::new(),
        all_finite: true,
        comparison: expected.map(|_| Comparison {
            compared_values: 0,
            differing_float_bits: 0,
            max_absolute_error_finite: 0.0,
            all_finite: true,
            first_difference: None,
        }),
    };
    let mut reference = if expected.is_none() {
        Vec::with_capacity(chunks * OUTPUT_CHUNK)
    } else {
        Vec::new()
    };
    let mut digest = Sha256::new();
    for first in (0..chunks).step_by(batch_size) {
        let actual_batch = batch_size.min(chunks - first);
        let input_shape = [actual_batch, 128, 100];
        let input = &packed[first * INPUT_CHUNK..(first + actual_batch) * INPUT_CHUNK];
        let tensor = TensorRef::from_array_view((input_shape, input))?;
        let run_started = Instant::now();
        let outputs = session.run(ort::inputs!["chunk_mel"=>tensor])?;
        let seconds = run_started.elapsed().as_secs_f64();
        report.calls += 1;
        report.ort_run_seconds += seconds;
        report.call_seconds.push(seconds);
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        ensure!(
            shape.len() == 3
                && shape[0] == actual_batch as i64
                && shape[1] == 13
                && shape[2] == 896,
            "Unexpected frontend output shape {shape:?}, batch {actual_batch}"
        );
        ensure!(
            values.len() == actual_batch * OUTPUT_CHUNK,
            "Unexpected frontend data length"
        );
        report.input_shapes.push(input_shape);
        report.output_shapes.push(shape.iter().copied().collect());
        report.output_values += values.len();
        digest.update(float_bytes(values));
        report.all_finite &= values.iter().all(|v| v.is_finite());
        if let Some(expected) = expected {
            let want = &expected[first * OUTPUT_CHUNK..(first + actual_batch) * OUTPUT_CHUNK];
            let comparison = report.comparison.as_mut().unwrap();
            comparison.compared_values += values.len();
            for (index, (&observed, &wanted)) in values.iter().zip(want).enumerate() {
                comparison.all_finite &= observed.is_finite() && wanted.is_finite();
                if observed.is_finite() && wanted.is_finite() {
                    comparison.max_absolute_error_finite = comparison
                        .max_absolute_error_finite
                        .max((f64::from(observed) - f64::from(wanted)).abs());
                }
                if observed.to_bits() != wanted.to_bits() {
                    comparison.differing_float_bits += 1;
                    if comparison.first_difference.is_none() {
                        comparison.first_difference =
                            Some(json!({"index":first*OUTPUT_CHUNK+index,
                            "chunk":first+index/OUTPUT_CHUNK,"within_chunk":index%OUTPUT_CHUNK,
                            "source_bits":wanted.to_bits(),"candidate_bits":observed.to_bits()}));
                    }
                }
            }
        } else {
            reference.extend_from_slice(values);
        }
    }
    report.output_sha256 = format!("{:x}", digest.finalize());
    report.wall_seconds_including_validation = started.elapsed().as_secs_f64();
    Ok((report, reference))
}
fn pass_exact(pass: &Pass) -> bool {
    pass.all_finite
        && pass
            .comparison
            .as_ref()
            .is_none_or(|c| c.all_finite && c.differing_float_bits == 0)
}
fn finish_profile(session: &mut Session, enabled: bool) -> Result<Option<String>> {
    if enabled {
        Ok(Some(session.end_profiling()?))
    } else {
        Ok(None)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=32).contains(&args.intra_threads),
        "--intra-threads must be1..32"
    );
    ensure!(
        !args.batch_sizes.is_empty() && args.batch_sizes.iter().all(|n| (1..=16).contains(n)),
        "--batch-sizes must be nonempty values1..16"
    );
    ensure!(
        fs::metadata(&args.pcm_f32)?.len() <= 80 * 16000 * 4,
        "PCM input exceeds80s"
    );
    let pcm_bytes = fs::read(&args.pcm_f32)?;
    ensure!(
        !pcm_bytes.is_empty() && pcm_bytes.len() % 4 == 0,
        "Expected bare float32 PCM"
    );
    let pcm = pcm_bytes
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(
        pcm.iter().all(|v| v.is_finite()),
        "PCM contains nonfinite samples"
    );
    let manifest_path = args
        .receipt
        .clone()
        .unwrap_or_else(|| args.frontend_candidate.with_extension("receipt.json"));
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    let source_sha = hash_file(&args.frontend_source)?;
    let candidate_sha = hash_file(&args.frontend_candidate)?;
    ensure!(
        manifest["prepared"] == true
            && manifest["source_sha256"] == source_sha
            && manifest["candidate_sha256"] == candidate_sha,
        "Preparation receipt is not bound to these model files"
    );
    ensure!(
        manifest["source_weights_sha256"].as_str().is_some()
            && manifest["source_weights_sha256"] == manifest["candidate_weights_sha256"]
            && manifest["all_nodes_and_attributes_byte_identical"] == true,
        "Preparation receipt does not establish unchanged weights and operators"
    );
    let mel_started = Instant::now();
    let (mel, frames) = mel::MelExtractor::new().extract(&pcm)?;
    let mel_seconds = mel_started.elapsed().as_secs_f64();
    let chunks = frames.div_ceil(100);
    let pack_started = Instant::now();
    let mut packed = vec![0.0f32; chunks * 128 * 100];
    for chunk in 0..chunks {
        let offset = chunk * 100;
        let count = 100.min(frames - offset);
        for band in 0..128 {
            packed[chunk * 12800 + band * 100..chunk * 12800 + band * 100 + count]
                .copy_from_slice(&mel[band * frames + offset..band * frames + offset + count]);
        }
    }
    let pack_seconds = pack_started.elapsed().as_secs_f64();
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .context("Refusing to overwrite an existing probe report")?;
    let mut report = json!({"schema":1,"complete":false,"equivalence_passed":false,
        "scope":"CPU frontend only; no backend encoder, decoder or sampling",
        "models":{"source_sha256":source_sha,"candidate_sha256":candidate_sha,
            "source_weights_sha256":manifest["source_weights_sha256"],"candidate_weights_sha256":manifest["candidate_weights_sha256"],
            "weight_provenance":"Preparation receipt bound to both complete model file hashes","receipt":manifest_path},
        "input":{"pcm_sha256":hash(&pcm_bytes),"sample_count":pcm.len(),"sample_rate":16000,
            "mel_frames":frames,"chunks":chunks,"valid_frontend_rows":mel::output_length(frames),
            "mel_sha256":hash(float_bytes(&mel)),"packed_chunks_sha256":hash(float_bytes(&packed)),
            "mel_seconds":mel_seconds,"pack_seconds":pack_seconds},
        "parameters":{"intra_threads":args.intra_threads,"inter_threads":1,"cpu_mem_arena":false,
            "parallel_execution":false,"allow_spinning":false,"memory_pattern":true,"prepacking":true,
            "providers":["CPUExecutionProvider"],"hot_repeats":args.repeats},
        "timing_boundary":"ort_run_seconds measures blocking Session.run only; excludes Mel, packing, session load and output comparison. First-session pass is not an OS/file-cache cold start.",
        "candidate_batches":[]});
    save(&args.output, &report)?;
    ort::init_from(&args.onnx_dll)?
        .with_name("FrontendBatchProbe")
        .commit();
    let load = Instant::now();
    let mut original = session(&args.frontend_source, &args, "source")?;
    let load_seconds = load.elapsed().as_secs_f64();
    let (first, reference) = run_pass(
        &mut original,
        &packed,
        chunks,
        1,
        None,
        "first_session_pass".into(),
    )?;
    ensure!(
        first.all_finite,
        "Original frontend output contains nonfinite floats"
    );
    let mut baseline_exact = true;
    let mut hot = Vec::new();
    for index in 0..args.repeats {
        let (pass, _) = run_pass(
            &mut original,
            &packed,
            chunks,
            1,
            Some(&reference),
            format!("hot_{}", index + 1),
        )?;
        baseline_exact &= pass_exact(&pass);
        hot.push(pass);
    }
    let profile = finish_profile(&mut original, args.profile_dir.is_some())?;
    drop(original);
    report["baseline"] = json!({"batch_size":1,"session_load_seconds":load_seconds,
        "first":first,"hot":hot,"self_exact":baseline_exact,"profile":profile});
    save(&args.output, &report)?;
    let mut all_exact = baseline_exact;
    for &batch in &args.batch_sizes {
        let load = Instant::now();
        let mut candidate = session(&args.frontend_candidate, &args, &format!("batch{batch}"))?;
        let load_seconds = load.elapsed().as_secs_f64();
        let (first, _) = run_pass(
            &mut candidate,
            &packed,
            chunks,
            batch,
            Some(&reference),
            "first_session_pass".into(),
        )?;
        let mut exact = pass_exact(&first);
        let mut hot = Vec::new();
        for index in 0..args.repeats {
            let (pass, _) = run_pass(
                &mut candidate,
                &packed,
                chunks,
                batch,
                Some(&reference),
                format!("hot_{}", index + 1),
            )?;
            exact &= pass_exact(&pass);
            hot.push(pass);
        }
        let profile = finish_profile(&mut candidate, args.profile_dir.is_some())?;
        drop(candidate);
        report["candidate_batches"].as_array_mut().unwrap().push(json!({"batch_size":batch,
            "session_load_seconds":load_seconds,"first":first,"hot":hot,"all_float_bits_exact":exact,"profile":profile}));
        all_exact &= exact;
        save(&args.output, &report)?;
    }
    report["complete"] = json!(true);
    report["equivalence_passed"] = json!(all_exact);
    save(&args.output, &report)?;
    println!("{}", args.output.display());
    ensure!(
        all_exact,
        "Frontend batch output differs in float bits; full diagnostic report preserved"
    );
    Ok(())
}

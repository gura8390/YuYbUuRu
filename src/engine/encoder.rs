use super::{
    EncoderProvider, EngineOptions,
    mel::{MelExtractor, output_length},
};
use anyhow::{Context, Result, ensure};
use ort::{
    logging::LogLevel,
    session::{RunOptions, Session, builder::GraphOptimizationLevel},
    value::{Tensor, TensorElementType, TensorRef, ValueType},
};
use std::{
    path::{Path, PathBuf},
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

static ORT_PATH: OnceLock<PathBuf> = OnceLock::new();
static ORT_INIT: Mutex<()> = Mutex::new(());
static PROFILE_ID: AtomicU64 = AtomicU64::new(0);
fn builder_error<R>(error: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("{error}")
}
fn init_ort(path: &Path) -> Result<()> {
    let path = path
        .canonicalize()
        .with_context(|| format!("ONNX Runtime DLL {}", path.display()))?;
    let _guard = ORT_INIT
        .lock()
        .map_err(|_| anyhow::anyhow!("ORT loader poisoned"))?;
    if let Some(active) = ORT_PATH.get() {
        ensure!(
            active == &path,
            "A different ONNX Runtime DLL is already active"
        );
        return Ok(());
    }
    ort::init_from(&path)?.with_name("CapsWriterRust").commit();
    let _ = ORT_PATH.set(path);
    Ok(())
}
pub(super) struct Features {
    pub data: Vec<f32>,
    pub dim: usize,
}
/// Opt-in diagnostics. No tensors, audio or recognized text are recorded.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub(super) struct EncoderMetrics {
    pub audio_samples: usize,
    pub frames: usize,
    pub chunks: usize,
    pub valid_rows: usize,
    pub frontend_dim: usize,
    pub output_dim: usize,
    pub fp16: bool,
    pub mel_seconds: f64,
    pub frontend_pack_seconds: f64,
    pub frontend_run_seconds: f64,
    pub frontend_collect_seconds: f64,
    pub backend_prepare_seconds: f64,
    pub backend_run_seconds: f64,
    pub output_copy_seconds: f64,
    pub total_seconds: f64,
}
fn start(enabled: bool) -> Option<std::time::Instant> {
    enabled.then(std::time::Instant::now)
}
fn elapsed(started: Option<std::time::Instant>) -> f64 {
    started.map_or(0.0, |tick| tick.elapsed().as_secs_f64())
}
pub(super) struct AudioEncoder {
    fe: Session,
    be: Session,
    mel: MelExtractor,
    fp16: bool,
    run_options: Option<RunOptions>,
    profiling: bool,
    timing: bool,
    last_metrics: EncoderMetrics,
}
impl AudioEncoder {
    pub fn load(
        dll: &Path,
        dir: &Path,
        frontend: &str,
        backend: &str,
        options: &EngineOptions,
    ) -> Result<Self> {
        ensure!(
            options
                .encoder_frontend_threads
                .is_none_or(|threads| threads <= 64)
                && options.encoder_dynamic_block_base <= 64,
            "Encoder CPU tuning exceeds the supported resource bounds"
        );
        let stage = std::time::Instant::now();
        init_ort(dll)?;
        super::load_timing("encoder.ort_init", stage);
        let profile_dir = std::env::var_os("YYUR_ORT_PROFILE_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        if let Some(directory) = profile_dir.as_ref() {
            std::fs::create_dir_all(directory)
                .with_context(|| format!("Create ORT profile directory {}", directory.display()))?;
        }
        fn session(
            path: &Path,
            options: &EngineOptions,
            cuda: bool,
            frontend: bool,
            profile_dir: Option<&Path>,
        ) -> Result<Session> {
            let intra_threads = if frontend {
                options
                    .encoder_frontend_threads
                    .unwrap_or(options.onnx_intra_threads)
            } else {
                options.onnx_intra_threads
            };
            let cpu_arena = options.cpu_mem_arena
                || if frontend {
                    options.encoder_frontend_arena
                } else {
                    options.encoder_backend_arena
                };
            let parallel = !frontend && options.encoder_parallel_execution;
            let spinning = if options.encoder_spin_wait { "1" } else { "0" };
            let mut builder = Session::builder()?
                .with_intra_threads(intra_threads)
                .map_err(builder_error)?
                .with_inter_threads(options.onnx_inter_threads)
                .map_err(builder_error)?
                .with_parallel_execution(parallel)
                .map_err(builder_error)?
                .with_log_level(LogLevel::Error)
                .map_err(builder_error)?
                .with_config_entry("session.intra_op.allow_spinning", spinning)
                .map_err(builder_error)?
                .with_config_entry("session.inter_op.allow_spinning", spinning)
                .map_err(builder_error)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(builder_error)?
                // ORT memory patterns require sequential graph execution.
                .with_memory_pattern(options.onnx_memory_pattern && !parallel)
                .map_err(builder_error)?
                .with_prepacking(options.onnx_prepacking)
                .map_err(builder_error)?;
            if options.encoder_spin_wait {
                // Keep workers awake within a Run, stop them at its final exit.
                builder = builder
                    .with_config_entry("session.force_spinning_stop", "1")
                    .map_err(builder_error)?;
            }
            if options.encoder_dynamic_block_base > 0 {
                builder = builder
                    .with_config_entry(
                        "session.dynamic_block_base",
                        &options.encoder_dynamic_block_base.to_string(),
                    )
                    .map_err(builder_error)?;
            }
            let cpu = ort::ep::CPU::default()
                .with_arena_allocator(cpu_arena)
                .build();
            let providers = if cuda {
                let bytes = options
                    .encoder_cuda_mem_limit_mb
                    .checked_mul(1024 * 1024)
                    .filter(|bytes| *bytes > 0)
                    .context("Encoder CUDA arena limit must be a positive MiB value")?;
                vec![
                    ort::ep::CUDA::default()
                        .with_device_id(0)
                        .with_memory_limit(bytes)
                        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
                        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                        .with_conv_max_workspace(false)
                        .with_tf32(false)
                        .build()
                        .error_on_failure(),
                    cpu,
                ]
            } else {
                vec![cpu]
            };
            builder = builder
                .with_execution_providers(providers)
                .map_err(builder_error)?;
            if options.onnx_device_initializers {
                builder = builder
                    .with_device_allocated_initializers()
                    .map_err(builder_error)?;
            }
            if let Some(directory) = profile_dir {
                let id = PROFILE_ID.fetch_add(1, Ordering::Relaxed);
                let model = path
                    .file_name()
                    .context("ONNX model filename")?
                    .to_string_lossy();
                let prefix = directory.join(format!("{}-{id}-{model}", std::process::id()));
                builder = builder.with_profiling(prefix).map_err(builder_error)?;
            }
            let session = builder
                .commit_from_file(path)
                .with_context(|| format!("Load ONNX model {}", path.display()))?;
            if profile_dir.is_some() {
                // This is the requested EP; actual node assignment is in the
                // profile. In particular, CPU shape nodes can be intentional.
                eprintln!(
                    "YYUR_ORT_SESSION {}",
                    serde_json::json!({
                        "model": path.file_name().map(|name| name.to_string_lossy()), "requested_cuda": cuda,
                        "cpu_arena": cpu_arena,
                        "intra_threads": intra_threads,
                        "spin_wait": options.encoder_spin_wait,
                        "dynamic_block_base": options.encoder_dynamic_block_base,
                        "parallel_execution": parallel,
                        "memory_pattern": options.onnx_memory_pattern && !parallel,
                        "cuda_arena_limit_mb": if cuda { Some(options.encoder_cuda_mem_limit_mb) } else { None },
                    })
                );
            }
            Ok(session)
        }
        let frontend_cuda = options.encoder_provider == EncoderProvider::CudaAll;
        let backend_cuda = options.encoder_provider != EncoderProvider::Cpu;
        let stage = std::time::Instant::now();
        let fe = session(
            &dir.join(frontend),
            options,
            frontend_cuda,
            true,
            profile_dir.as_deref(),
        )?;
        super::load_timing("encoder.frontend_session", stage);
        let stage = std::time::Instant::now();
        let be = session(
            &dir.join(backend),
            options,
            backend_cuda,
            false,
            profile_dir.as_deref(),
        )?;
        super::load_timing("encoder.backend_session", stage);
        let fp16 = matches!(
            fe.inputs().first().map(|i| i.dtype()),
            Some(ValueType::Tensor {
                ty: TensorElementType::Float16,
                ..
            })
        );
        // CPU arena shrinkage is invalid when the CPU EP arena is disabled.
        // Apply it only to the dynamic backend. Frontend chunks always have
        // the same 100-frame shape, so scanning after every chunk adds work
        // without retiring the backend's long-utterance allocations.
        // Retain the old null RunOptions path unless explicitly requested.
        let backend_cpu_arena = options.cpu_mem_arena || options.encoder_backend_arena;
        let run_options = if options.onnx_arena_shrink && (backend_cpu_arena || backend_cuda) {
            let mut value = RunOptions::new()?;
            let arenas = match (backend_cpu_arena, backend_cuda) {
                (true, true) => "cpu:0;gpu:0",
                (true, false) => "cpu:0",
                (false, true) => "gpu:0",
                (false, false) => unreachable!(),
            };
            value.add_config_entry("memory.enable_memory_arena_shrinkage", arenas)?;
            Some(value)
        } else {
            None
        };
        let mut encoder = Self {
            fe,
            be,
            mel: MelExtractor::new(),
            fp16,
            run_options,
            profiling: profile_dir.is_some(),
            timing: std::env::var_os("YYUR_ENCODER_TIMING").is_some(),
            last_metrics: EncoderMetrics::default(),
        };
        // Match the existing non-DML two-second warmup.
        let stage = std::time::Instant::now();
        let _ = encoder.encode(&vec![0.0; 32000])?;
        super::load_timing("encoder.warmup", stage);
        Ok(encoder)
    }
    pub fn encode(&mut self, audio: &[f32]) -> Result<Features> {
        let total_started = start(self.timing);
        let mut metrics = EncoderMetrics {
            audio_samples: audio.len(),
            fp16: self.fp16,
            ..Default::default()
        };
        let mel_started = start(self.timing);
        let (mel, frames) = self.mel.extract(audio)?;
        metrics.mel_seconds = elapsed(mel_started);
        let chunks = frames.div_ceil(100);
        let valid = output_length(frames);
        metrics.frames = frames;
        metrics.chunks = chunks;
        metrics.valid_rows = valid;
        let mut hidden = Vec::new();
        let mut dim = 0;
        for chunk in 0..chunks {
            let pack_started = start(self.timing);
            let mut input = vec![0.0f32; 128 * 100];
            let offset = chunk * 100;
            let n = (frames - offset).min(100);
            for band in 0..128 {
                input[band * 100..band * 100 + n]
                    .copy_from_slice(&mel[band * frames + offset..band * frames + offset + n]);
            }
            if self.fp16 {
                let data = input
                    .into_iter()
                    .map(half::f16::from_f32)
                    .collect::<Vec<_>>();
                let tensor = Tensor::from_array(([1usize, 128, 100], data))?;
                metrics.frontend_pack_seconds += elapsed(pack_started);
                let run_started = start(self.timing);
                let output = self.fe.run(ort::inputs!["chunk_mel"=>tensor])?;
                metrics.frontend_run_seconds += elapsed(run_started);
                let collect_started = start(self.timing);
                let (shape, data) = output[0].try_extract_tensor::<half::f16>()?;
                ensure!(
                    shape.len() == 3 && shape[0] == 1 && shape[1] == 13,
                    "Unexpected frontend shape {shape:?}"
                );
                let d = usize::try_from(shape[2])?;
                if dim == 0 {
                    dim = d;
                }
                ensure!(dim == d && dim > 0, "Frontend embedding dimension changed");
                hidden.extend(data.iter().map(|v| v.to_f32()));
                metrics.frontend_collect_seconds += elapsed(collect_started);
            } else {
                let tensor = Tensor::from_array(([1usize, 128, 100], input))?;
                metrics.frontend_pack_seconds += elapsed(pack_started);
                let run_started = start(self.timing);
                let output = self.fe.run(ort::inputs!["chunk_mel"=>tensor])?;
                metrics.frontend_run_seconds += elapsed(run_started);
                let collect_started = start(self.timing);
                let (shape, data) = output[0].try_extract_tensor::<f32>()?;
                ensure!(
                    shape.len() == 3 && shape[0] == 1 && shape[1] == 13,
                    "Unexpected frontend shape {shape:?}"
                );
                let d = usize::try_from(shape[2])?;
                if dim == 0 {
                    dim = d;
                }
                ensure!(dim == d && dim > 0, "Frontend embedding dimension changed");
                hidden.extend_from_slice(data);
                metrics.frontend_collect_seconds += elapsed(collect_started);
            }
        }
        ensure!(
            valid > 0 && valid * dim <= hidden.len(),
            "Invalid frontend effective length"
        );
        hidden.truncate(valid * dim);
        metrics.frontend_dim = dim;
        let prepare_started = start(self.timing);
        let mask_len = valid
            .checked_mul(valid)
            .context("ONNX attention mask size overflow")?;
        if self.fp16 {
            let hidden = hidden
                .into_iter()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>();
            let mask = vec![half::f16::ZERO; mask_len];
            let values = TensorRef::from_array_view(([1usize, valid, dim], hidden.as_slice()))?;
            let mask = TensorRef::from_array_view(([1usize, 1, valid, valid], mask.as_slice()))?;
            metrics.backend_prepare_seconds = elapsed(prepare_started);
            let run_started = start(self.timing);
            let outputs = if let Some(options) = self.run_options.as_ref() {
                self.be.run_with_options(
                    ort::inputs!["hidden_states"=>values,"attention_mask"=>mask],
                    options,
                )?
            } else {
                self.be
                    .run(ort::inputs!["hidden_states"=>values,"attention_mask"=>mask])?
            };
            metrics.backend_run_seconds = elapsed(run_started);
            let copy_started = start(self.timing);
            let (shape, result) = outputs[0].try_extract_tensor::<half::f16>()?;
            ensure!(
                shape.len() == 3 && shape[0] == 1 && shape[1] >= valid as i64,
                "Unexpected backend shape {shape:?}"
            );
            let output_dim = usize::try_from(shape[2])?;
            let features = Features {
                data: result[..valid * output_dim]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect(),
                dim: output_dim,
            };
            metrics.output_dim = output_dim;
            metrics.output_copy_seconds = elapsed(copy_started);
            metrics.total_seconds = elapsed(total_started);
            self.last_metrics = metrics;
            return Ok(features);
        } else {
            let mask = vec![0f32; mask_len];
            let values = TensorRef::from_array_view(([1usize, valid, dim], hidden.as_slice()))?;
            let mask = TensorRef::from_array_view(([1usize, 1, valid, valid], mask.as_slice()))?;
            metrics.backend_prepare_seconds = elapsed(prepare_started);
            let run_started = start(self.timing);
            let outputs = if let Some(options) = self.run_options.as_ref() {
                self.be.run_with_options(
                    ort::inputs!["hidden_states"=>values,"attention_mask"=>mask],
                    options,
                )?
            } else {
                self.be
                    .run(ort::inputs!["hidden_states"=>values,"attention_mask"=>mask])?
            };
            metrics.backend_run_seconds = elapsed(run_started);
            let copy_started = start(self.timing);
            let (shape, result) = outputs[0].try_extract_tensor::<f32>()?;
            ensure!(
                shape.len() == 3 && shape[0] == 1 && shape[1] >= valid as i64,
                "Unexpected backend shape {shape:?}"
            );
            let output_dim = usize::try_from(shape[2])?;
            let features = Features {
                data: result[..valid * output_dim].to_vec(),
                dim: output_dim,
            };
            metrics.output_dim = output_dim;
            metrics.output_copy_seconds = elapsed(copy_started);
            metrics.total_seconds = elapsed(total_started);
            self.last_metrics = metrics;
            return Ok(features);
        }
    }
    #[allow(dead_code)] // Read by the standalone encoder diagnostic.
    pub fn last_metrics(&self) -> &EncoderMetrics {
        &self.last_metrics
    }
}

impl Drop for AudioEncoder {
    fn drop(&mut self) {
        if self.profiling {
            // ORT requires explicit finalization. A killed worker cannot flush
            // this diagnostic; its absent/empty profile is not valid evidence.
            for (stage, session) in [("frontend", &mut self.fe), ("backend", &mut self.be)] {
                match session.end_profiling() {
                    Ok(path) => eprintln!(
                        "YYUR_ORT_PROFILE {}",
                        serde_json::json!({"stage": stage, "path": path})
                    ),
                    Err(error) => eprintln!("YYUR_ORT_PROFILE_ERROR {stage}: {error}"),
                }
            }
        }
    }
}

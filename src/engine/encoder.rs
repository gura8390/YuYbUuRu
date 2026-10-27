use super::{
    EngineOptions,
    mel::{MelExtractor, output_length},
};
use anyhow::{Context, Result, ensure};
use ort::{
    logging::LogLevel,
    session::{Session, builder::GraphOptimizationLevel},
    value::{Tensor, TensorElementType, TensorRef, ValueType},
};
use std::{
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

static ORT_PATH: OnceLock<PathBuf> = OnceLock::new();
static ORT_INIT: Mutex<()> = Mutex::new(());
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
pub(super) struct AudioEncoder {
    fe: Session,
    be: Session,
    mel: MelExtractor,
    fp16: bool,
}
impl AudioEncoder {
    pub fn load(
        dll: &Path,
        dir: &Path,
        frontend: &str,
        backend: &str,
        options: &EngineOptions,
    ) -> Result<Self> {
        let stage = std::time::Instant::now();
        init_ort(dll)?;
        super::load_timing("encoder.ort_init", stage);
        fn session(path: &Path, options: &EngineOptions) -> Result<Session> {
            Ok(Session::builder()?
                .with_intra_threads(options.onnx_intra_threads)
                .map_err(builder_error)?
                .with_inter_threads(options.onnx_inter_threads)
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
                .with_execution_providers([ort::ep::CPU::default()
                    .with_arena_allocator(options.cpu_mem_arena)
                    .build()])
                .map_err(builder_error)?
                .commit_from_file(path)
                .with_context(|| format!("Load ONNX model {}", path.display()))?)
        }
        let stage = std::time::Instant::now();
        let fe = session(&dir.join(frontend), options)?;
        super::load_timing("encoder.frontend_session", stage);
        let stage = std::time::Instant::now();
        let be = session(&dir.join(backend), options)?;
        super::load_timing("encoder.backend_session", stage);
        let fp16 = matches!(
            fe.inputs().first().map(|i| i.dtype()),
            Some(ValueType::Tensor {
                ty: TensorElementType::Float16,
                ..
            })
        );
        let mut encoder = Self {
            fe,
            be,
            mel: MelExtractor::new(),
            fp16,
        };
        // Match the existing non-DML two-second warmup.
        let stage = std::time::Instant::now();
        let _ = encoder.encode(&vec![0.0; 32000])?;
        super::load_timing("encoder.warmup", stage);
        Ok(encoder)
    }
    pub fn encode(&mut self, audio: &[f32]) -> Result<Features> {
        let (mel, frames) = self.mel.extract(audio)?;
        let chunks = frames.div_ceil(100);
        let valid = output_length(frames);
        let mut hidden = Vec::new();
        let mut dim = 0;
        for chunk in 0..chunks {
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
                let output = self.fe.run(ort::inputs!["chunk_mel"=>tensor])?;
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
            } else {
                let tensor = Tensor::from_array(([1usize, 128, 100], input))?;
                let output = self.fe.run(ort::inputs!["chunk_mel"=>tensor])?;
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
            }
        }
        ensure!(
            valid > 0 && valid * dim <= hidden.len(),
            "Invalid frontend effective length"
        );
        hidden.truncate(valid * dim);
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
            let outputs = self
                .be
                .run(ort::inputs!["hidden_states"=>values,"attention_mask"=>mask])?;
            let (shape, result) = outputs[0].try_extract_tensor::<half::f16>()?;
            ensure!(
                shape.len() == 3 && shape[0] == 1 && shape[1] >= valid as i64,
                "Unexpected backend shape {shape:?}"
            );
            let output_dim = usize::try_from(shape[2])?;
            return Ok(Features {
                data: result[..valid * output_dim]
                    .iter()
                    .map(|v| v.to_f32())
                    .collect(),
                dim: output_dim,
            });
        } else {
            let mask = vec![0f32; mask_len];
            let values = TensorRef::from_array_view(([1usize, valid, dim], hidden.as_slice()))?;
            let mask = TensorRef::from_array_view(([1usize, 1, valid, valid], mask.as_slice()))?;
            let outputs = self
                .be
                .run(ort::inputs!["hidden_states"=>values,"attention_mask"=>mask])?;
            let (shape, result) = outputs[0].try_extract_tensor::<f32>()?;
            ensure!(
                shape.len() == 3 && shape[0] == 1 && shape[1] >= valid as i64,
                "Unexpected backend shape {shape:?}"
            );
            let output_dim = usize::try_from(shape[2])?;
            return Ok(Features {
                data: result[..valid * output_dim].to_vec(),
                dim: output_dim,
            });
        }
    }
}

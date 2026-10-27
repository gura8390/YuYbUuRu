use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuBackend {
    #[default]
    Cuda,
    Vulkan,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncoderProvider {
    #[default]
    Cpu,
    CudaBackend,
    CudaAll,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub app_dir: PathBuf,
    pub onnx_dll: Option<PathBuf>,
    pub encoder_provider: EncoderProvider,
    /// Per-session CUDA EP arena limit; excludes libraries and other sessions.
    pub encoder_cuda_mem_limit_mb: usize,
    pub language: String,
    pub asr_idle_seconds: u64,
    pub aligner_idle_seconds: u64,
    pub worker_timeout_seconds: u64,
    pub trim_after_load: bool,
    pub onnx_intra_threads: usize,
    pub onnx_inter_threads: usize,
    pub llm_threads: i32,
    pub llm_batch_threads: i32,
    pub onnx_memory_pattern: bool,
    pub onnx_arena_shrink: bool,
    pub onnx_device_initializers: bool,
    pub onnx_prepacking: bool,
    pub asr_dynamic_context: bool,
    pub asr_single_output: bool,
    pub aligner_cpu_mem_arena: bool,
    pub aligner_dynamic_context: bool,
    pub aligner_logits_limit: u32,
    pub file_context_handoff: bool,
    pub vulkan_pipeline_cache: bool,
    pub gpu: bool,
    pub gpu_backend: GpuBackend,
    /// Parallelize independent Q/K/V branches in the CUDA graph only.
    pub cuda_graph_optimization: bool,
    pub hotwords: bool,
    pub hotword_rules: bool,
    pub hot_threshold: f64,
    pub hot_similar_threshold: f64,
    pub format_num: bool,
    pub format_spell: bool,
    pub segment_seconds: usize,
    pub overlap_seconds: usize,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            app_dir: PathBuf::from("."),
            onnx_dll: None,
            encoder_provider: EncoderProvider::Cpu,
            encoder_cuda_mem_limit_mb: 512,
            language: "auto".into(),
            asr_idle_seconds: 120,
            aligner_idle_seconds: 10,
            worker_timeout_seconds: 120,
            trim_after_load: true,
            onnx_intra_threads: 16,
            onnx_inter_threads: 1,
            llm_threads: 1,
            llm_batch_threads: 1,
            onnx_memory_pattern: true,
            onnx_arena_shrink: false,
            onnx_device_initializers: false,
            onnx_prepacking: true,
            asr_dynamic_context: false,
            asr_single_output: true,
            aligner_cpu_mem_arena: false,
            aligner_dynamic_context: true,
            aligner_logits_limit: 4000,
            file_context_handoff: true,
            vulkan_pipeline_cache: false,
            gpu: true,
            gpu_backend: GpuBackend::Cuda,
            cuda_graph_optimization: false,
            hotwords: true,
            hotword_rules: true,
            hot_threshold: 0.85,
            hot_similar_threshold: 0.6,
            format_num: true,
            format_spell: true,
            segment_seconds: 60,
            overlap_seconds: 4,
        }
    }
}
impl Settings {
    pub fn load(path: Option<&Path>, app_override: Option<&Path>) -> Result<Self> {
        let mut value = if let Some(path) = path {
            serde_json::from_slice::<Self>(
                &std::fs::read(path).with_context(|| format!("Read {}", path.display()))?,
            )?
        } else {
            Self::default()
        };
        if let Some(app) = app_override {
            value.app_dir = app.to_owned();
        } else if value.app_dir.is_relative() {
            let base = path.and_then(Path::parent).map(Path::to_owned).unwrap_or(
                std::env::current_exe()?
                    .parent()
                    .context("Executable directory")?
                    .to_owned(),
            );
            value.app_dir = base.join(&value.app_dir);
        }
        value.app_dir = value
            .app_dir
            .canonicalize()
            .with_context(|| format!("Application directory {}", value.app_dir.display()))?;
        ensure!(
            value.segment_seconds > 0 && value.segment_seconds + 2 * value.overlap_seconds <= 80,
            "Segments must fit the 80-second model window"
        );
        ensure!(
            (0.0..=1.0).contains(&value.hot_threshold),
            "Invalid hotword threshold"
        );
        ensure!(
            matches!(value.aligner_logits_limit, 0 | 4000),
            "Aligner logits limit must be 0 (full vocabulary) or 4000 (timestamps)"
        );
        ensure!(
            value.encoder_cuda_mem_limit_mb > 0
                && value
                    .encoder_cuda_mem_limit_mb
                    .checked_mul(1024 * 1024)
                    .is_some(),
            "Encoder CUDA arena limit must be a positive MiB value"
        );
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_provider_is_explicit_and_cpu_by_default() {
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(old.encoder_provider, EncoderProvider::Cpu);
        assert!(old.onnx_dll.is_none());
        assert_eq!(old.encoder_cuda_mem_limit_mb, 512);
        for (name, expected) in [
            ("cpu", EncoderProvider::Cpu),
            ("cuda_backend", EncoderProvider::CudaBackend),
            ("cuda_all", EncoderProvider::CudaAll),
        ] {
            let settings: Settings = serde_json::from_value(serde_json::json!({
                "encoder_provider": name,
                "onnx_dll": "runtime-ort-gpu/capi/onnxruntime.dll"
            }))
            .unwrap();
            assert_eq!(settings.encoder_provider, expected);
        }
        assert!(serde_json::from_str::<Settings>(r#"{"encoder_provider":"cuda"}"#).is_err());
    }

    #[test]
    fn asr_dynamic_context_is_opt_in_for_old_settings() {
        assert!(!Settings::default().asr_dynamic_context);
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert!(!old.asr_dynamic_context);
        let enabled: Settings = serde_json::from_str(r#"{"asr_dynamic_context":true}"#).unwrap();
        assert!(enabled.asr_dynamic_context);
    }
}

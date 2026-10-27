use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub app_dir: PathBuf,
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
    pub aligner_cpu_mem_arena: bool,
    pub aligner_dynamic_context: bool,
    pub aligner_logits_limit: u32,
    pub file_context_handoff: bool,
    pub vulkan_pipeline_cache: bool,
    pub gpu: bool,
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
            aligner_cpu_mem_arena: false,
            aligner_dynamic_context: false,
            aligner_logits_limit: 0,
            file_context_handoff: false,
            vulkan_pipeline_cache: true,
            gpu: true,
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
        Ok(value)
    }
}

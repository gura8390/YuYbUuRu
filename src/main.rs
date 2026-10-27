#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::{io::Write, path::PathBuf};
use yuybuuru::{
    application,
    config::Settings,
    engine::{Engine, EngineOptions},
    media, worker,
};

#[derive(Parser)]
#[command(
    version,
    about = "Offline Qwen3-ASR dictation, hotwords and file transcription"
)]
struct Cli {
    #[arg(long, global = true)]
    app_dir: Option<PathBuf>,
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[arg(long, global = true)]
    seed: Option<u32>,
    #[arg(long, global = true)]
    asr_idle_seconds: Option<u64>,
    #[arg(long, global = true)]
    aligner_idle_seconds: Option<u64>,
    #[command(subcommand)]
    command: Option<Mode>,
}
#[derive(Subcommand)]
enum Mode {
    Desktop {
        #[arg(long)]
        passive: bool,
    },
    Transcribe {
        #[arg(required = true)]
        files: Vec<PathBuf>,
        #[arg(long)]
        output_dir: Option<PathBuf>,
        #[arg(long)]
        overwrite: bool,
    },
    /// Internal local protocol endpoint; no TCP listener.
    Serve,
    #[command(hide = true)]
    Worker {
        #[arg(long)]
        settings_json: String,
    },
    /// Offline native diagnostic, bypassing the desktop and input hooks.
    Infer {
        file: PathBuf,
        #[arg(long)]
        native_seed: Option<u32>,
        #[arg(long, default_value = "auto")]
        language: String,
        #[arg(long, default_value = "")]
        context: String,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error:#}");
        std::process::exit(1)
    }
}
fn run() -> Result<()> {
    let cli = Cli::parse();
    if let Some(Mode::Worker { settings_json }) = &cli.command {
        return worker::run(serde_json::from_str(settings_json)?);
    }
    let default_config = std::env::current_exe()?
        .parent()
        .map(|dir| dir.join("settings.json"));
    let config_path = cli
        .config
        .as_deref()
        .or_else(|| default_config.as_deref().filter(|path| path.is_file()));
    let mut settings = Settings::load(config_path, cli.app_dir.as_deref())?;
    if let Some(value) = cli.asr_idle_seconds {
        settings.asr_idle_seconds = value;
    }
    if let Some(value) = cli.aligner_idle_seconds {
        settings.aligner_idle_seconds = value;
    }
    match cli.command.unwrap_or(Mode::Desktop { passive: false }) {
        Mode::Desktop { passive } => application::desktop(settings, cli.seed, passive),
        Mode::Serve => worker::serve(settings),
        Mode::Worker { .. } => unreachable!(),
        Mode::Transcribe {
            files,
            output_dir,
            overwrite,
        } => {
            let service = worker::WorkerService::start(settings.clone())?;
            let client = application::RuntimeClient::new(service.client(), settings, cli.seed)?;
            for file in files {
                let (result, outputs) =
                    client.transcribe(&file, output_dir.as_deref(), overwrite)?;
                println!(
                    "{}",
                    serde_json::to_string(
                        &serde_json::json!({"file":file,"text":result.text_accu,"outputs":outputs})
                    )?
                );
            }
            drop(service);
            Ok(())
        }
        Mode::Infer {
            file,
            native_seed,
            language,
            context,
            output,
        } => {
            let mut options: EngineOptions = application_options(&settings);
            options.seed = native_seed;
            let audio = media::read_audio(&settings.app_dir, &file)?;
            let mut engine = Engine::load(&settings.app_dir, &options)?;
            let result = engine.transcribe(&audio, &language, &context)?;
            let data = serde_json::to_vec_pretty(&result)?;
            if let Some(file) = output {
                std::fs::write(file, data)?;
            } else {
                std::io::stdout().write_all(&data)?;
            }
            Ok(())
        }
    }
}
fn application_options(settings: &Settings) -> EngineOptions {
    yuybuuru::pipeline::engine_options(settings)
}

//! `lethe-codex` — Codex CLI adapter binary.
//!
//! Helpers the plugin under `plugins/codex/` needs that don't belong in
//! `lethe-core`. Today: parsing Codex's rollout JSONL transcripts. Mirrors
//! `lethe-claude-code` so the plugin pipeline stays uniform across agents.

#![allow(clippy::print_stdout)]

use clap::{Parser, Subcommand};

mod transcript;

#[derive(Parser, Debug)]
#[command(
    name = "lethe-codex",
    version,
    about = "Codex CLI adapter for the lethe memory store."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Extract a user/assistant turn from a Codex rollout JSONL transcript.
    Transcript {
        path: String,
        /// Surface a specific turn UUID (default: last pair).
        #[arg(long)]
        turn: Option<String>,
    },
}

fn main() -> std::process::ExitCode {
    let cli = Cli::parse();
    let rc = match dispatch(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    };
    std::process::ExitCode::from(u8::try_from(rc).unwrap_or(1))
}

fn dispatch(cli: Cli) -> anyhow::Result<i32> {
    match cli.cmd {
        Cmd::Transcript { path, turn } => transcript::run(&path, turn.as_deref()),
    }
}

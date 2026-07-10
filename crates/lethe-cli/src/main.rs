//! `lethe` — CLI for the lethe memory store.
//!
//! Subcommands match `research_playground/lethe_reference/lethe/cli.py` byte-for-byte on their JSON
//! outputs; the Claude Code plugin scripts parse stdout, so any drift
//! breaks them. See the porting plan
//! (`.claude/plans/functional-herding-boot.md`) for the contract.
//!
//! With no subcommand the TUI launches when stdout is a terminal;
//! otherwise `--help` is printed and the binary exits non-zero.

#![allow(clippy::print_stdout)] // CLI: stdout is the interface.

use std::io::IsTerminal;

use clap::{CommandFactory, Parser, Subcommand};

mod commands;
mod paths;

#[derive(Parser, Debug)]
#[command(
    name = "lethe",
    version,
    about = "Persistent memory store for LLM agents.",
    long_about = "Persistent memory store for LLM agents — hybrid retrieval, RIF, optional enrichment.\n\
                  Run with no arguments to launch the interactive TUI."
)]
struct Cli {
    /// Project root. Default: git root of CWD.
    #[arg(long, global = true)]
    root: Option<String>,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Index this project's agent transcripts into the store.
    Index {
        /// Reindex every registered project, not just the current one.
        #[arg(long)]
        all: bool,
        #[arg(long)]
        json_output: bool,
        #[arg(long)]
        no_register: bool,
    },
    /// Retrieve top-k memories for a query.
    Search {
        query: String,
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        #[arg(long)]
        json_output: bool,
        /// Search across all registered projects (~/.lethe/projects.json).
        #[arg(long)]
        all: bool,
        /// Comma-separated project slugs or paths.
        #[arg(long)]
        projects: Option<String>,
        /// Open the index read-only and skip the post-retrieve save.
        /// Use this when several lethe processes need to share the
        /// index concurrently (DuckDB allows many readers xor one
        /// writer per file). Trade-off: retrieval-driven RIF state
        /// (suppression scores, tier transitions, retrieval counts)
        /// won't evolve for this query.
        /// No effect with --all (cross-project search is already read-only).
        #[arg(long)]
        read_only: bool,
    },
    /// Print the full markdown section for one or more chunk ids.
    Expand {
        #[arg(num_args = 1.., required = true)]
        chunk_ids: Vec<String>,
    },
    /// Print diagnostic JSON for the store.
    Status {
        #[arg(long)]
        json_output: bool,
    },
    /// Read or write config values.
    Config {
        #[arg(value_parser = ["get", "set"])]
        action: String,
        key: Option<String>,
        value: Option<String>,
    },
    /// Delete this project's global index (transcripts are untouched).
    Reset {
        #[arg(long)]
        yes: bool,
    },
    /// Manage the global project registry.
    Projects {
        #[command(subcommand)]
        action: ProjectsCmd,
    },
    /// Delete dead/empty Claude Code & Codex transcripts from disk.
    /// Dry-run unless `--yes`.
    Cleanup {
        /// Actually delete (default is a preview).
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        json_output: bool,
    },
    /// Interactive TUI. Implicit when `lethe` is run with no args in a terminal.
    Tui,
}

#[derive(Subcommand, Debug)]
enum ProjectsCmd {
    /// List registered projects.
    List {
        #[arg(long)]
        json_output: bool,
    },
    /// Register a project by root path (default: cwd).
    Add { path: Option<String> },
    /// Unregister a project by path or slug.
    Remove { name: String },
    /// Drop registry entries whose roots no longer exist.
    Prune,
    /// Remove registered projects that have no memories (indexed, empty).
    /// Dry-run unless `--yes`.
    PruneEmpty {
        /// Also delete each project's Claude/Codex transcripts on disk.
        #[arg(long)]
        transcripts: bool,
        /// Actually delete (default is a preview).
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        json_output: bool,
    },
}

fn main() -> std::process::ExitCode {
    // Initialize tracing only when LETHE_LOG is set; CLI stdout is the
    // contract, so noise on stderr is opt-in only.
    if std::env::var_os("LETHE_LOG").is_some() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_env("LETHE_LOG")
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_writer(std::io::stderr)
            .try_init();
    }
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
    let root = cli.root.as_deref();
    let cmd = match cli.cmd {
        Some(c) => c,
        None => {
            // No subcommand: open the TUI when interactive, else print
            // help so scripted callers see the contract instead of a
            // hung process.
            if std::io::stdout().is_terminal() {
                Cmd::Tui
            } else {
                Cli::command().print_help()?;
                return Ok(2);
            }
        }
    };
    match cmd {
        Cmd::Index {
            all,
            json_output,
            no_register,
        } => commands::index::run(root, all, json_output, no_register),
        Cmd::Search {
            query,
            top_k,
            json_output,
            all,
            projects,
            read_only,
        } => {
            if all || projects.is_some() {
                commands::search::run_union(&query, top_k, json_output, projects.as_deref())
            } else {
                commands::search::run_local(root, &query, top_k, json_output, read_only)
            }
        }
        Cmd::Expand { chunk_ids } => commands::expand::run(root, &chunk_ids),
        Cmd::Status { json_output: _ } => commands::status::run(root),
        Cmd::Config { action, key, value } => {
            commands::config::run(root, &action, key.as_deref(), value.as_deref())
        }
        Cmd::Reset { yes } => commands::reset::run(root, yes),
        Cmd::Projects { action } => commands::projects::run(action),
        Cmd::Cleanup { yes, json_output } => commands::cleanup::run(yes, json_output),
        Cmd::Tui => commands::tui::run(),
    }
}

// Make the `ProjectsCmd` reachable from the commands module.
pub(crate) use ProjectsCmd as ProjectsCmdExport;

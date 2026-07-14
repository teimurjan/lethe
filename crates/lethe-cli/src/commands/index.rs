//! `lethe index` — index this project's agent transcripts into the store.
//!
//! Full scan: reparses every changed transcript (Claude Code, Codex, and Oh
//! My Pi) and add-only-syncs its turns. Idempotent — unchanged transcripts are
//! skipped via the manifest. Also registers the project so
//! `lethe search --all` / recall-global can find its global index.
//!
//! `--all` reindexes every registered project in one pass — useful after
//! upgrading, since projects are otherwise only indexed when you search
//! (or run `lethe index`) inside them.

use anyhow::Result;
use lethe_core::registry;
use serde::Serialize;

use crate::paths::{resolve, Paths};

use lethe_core::transcript_index;

use super::store_helpers::{ensure_index_format, load_config, open_store, CliConfig};

#[derive(Serialize)]
struct IndexCounts {
    added: usize,
    unchanged: usize,
    total: usize,
}

pub fn run(root: Option<&str>, all: bool, json_output: bool, no_register: bool) -> Result<i32> {
    if all {
        return run_all(json_output);
    }
    let paths = resolve(root);
    let counts = index_project(&paths, None)?;

    if !no_register && !registry::is_disabled() {
        // Best-effort: don't fail the command if the registry write
        // raced with another process.
        let _ = registry::register(&paths.root);
    }

    let payload = IndexCounts {
        added: counts.added,
        unchanged: counts.unchanged,
        total: counts.total,
    };
    if json_output {
        println!("{}", serde_json::to_string(&payload)?);
    } else {
        println!(
            "indexed: added={} unchanged={} total={}",
            payload.added, payload.unchanged, payload.total
        );
    }
    Ok(0)
}

/// Sync one project's transcripts into its global index. `cfg` is loaded
/// once by the caller when reindexing many projects.
fn index_project(
    paths: &Paths,
    cfg: Option<&CliConfig>,
) -> Result<lethe_core::transcript_store::SyncCounts> {
    ensure_index_format(&paths.index())?;
    std::fs::create_dir_all(paths.index())?;
    let loaded = if cfg.is_none() {
        Some(load_config(&paths.config_path())?)
    } else {
        None
    };
    let cfg = cfg.unwrap_or_else(|| loaded.as_ref().expect("loaded when cfg is None"));
    let store = open_store(&paths.index(), cfg, true, false)?;
    let counts = transcript_index::ensure_fresh(&store, &paths.root)?;
    store.save()?;
    // Index now reflects the current format (canonical ids).
    store.mark_index_format()?;
    Ok(counts)
}

fn run_all(json_output: bool) -> Result<i32> {
    let entries = registry::load();
    if entries.is_empty() {
        eprintln!("no registered projects — run `lethe index` in a repo first");
        return Ok(1);
    }
    let cfg = load_config(&registry::registry_dir().join("config.toml"))?;

    #[derive(Serialize)]
    struct ProjectResult {
        slug: String,
        root: String,
        added: usize,
        total: usize,
    }
    let mut results: Vec<ProjectResult> = Vec::new();
    let (mut grand_added, mut grand_total) = (0usize, 0usize);

    for (i, entry) in entries.iter().enumerate() {
        let paths = Paths {
            root: entry.root.clone(),
        };
        if !json_output {
            eprintln!("[{}/{}] {} …", i + 1, entries.len(), paths.root.display());
        }
        match index_project(&paths, Some(&cfg)) {
            Ok(counts) => {
                grand_added += counts.added;
                grand_total += counts.total;
                results.push(ProjectResult {
                    slug: entry.slug.clone(),
                    root: entry.root.to_string_lossy().into_owned(),
                    added: counts.added,
                    total: counts.total,
                });
            }
            Err(e) => eprintln!("  skip {}: {e}", entry.root.display()),
        }
    }

    if json_output {
        println!("{}", serde_json::to_string(&results)?);
    } else {
        println!(
            "reindexed {} projects: added={} total={}",
            results.len(),
            grand_added,
            grand_total
        );
    }
    Ok(0)
}

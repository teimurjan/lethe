//! `lethe index` — index this project's agent transcripts into the store.
//!
//! Full scan: reparses every changed transcript (Claude Code + Codex) and
//! add-only-syncs its turns. Idempotent — unchanged transcripts are
//! skipped via the manifest. Also registers the project so
//! `lethe search --all` / recall-global can find its global index.

use anyhow::Result;
use lethe_core::registry;
use serde::Serialize;

use crate::paths::resolve;

use super::store_helpers::{load_config, open_store};
use super::transcript_index;

#[derive(Serialize)]
struct IndexCounts {
    added: usize,
    unchanged: usize,
    total: usize,
}

pub fn run(root: Option<&str>, json_output: bool, no_register: bool) -> Result<i32> {
    let paths = resolve(root);
    std::fs::create_dir_all(paths.index())?;
    let cfg = load_config(&paths.config_path())?;
    let store = open_store(&paths.index(), &cfg, true, false)?;
    let counts = transcript_index::ensure_fresh(&store, &paths.root)?;
    store.save()?;
    drop(store);

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

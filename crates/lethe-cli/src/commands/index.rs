//! `lethe index` — reindex `.lethe/memory/*.md` into the store.

use std::path::PathBuf;

use anyhow::Result;
use lethe_core::markdown_store;
use lethe_core::registry;
use serde::Serialize;

use crate::paths::resolve;

use super::store_helpers::{load_config, open_store};

#[derive(Serialize)]
struct IndexCounts {
    added: usize,
    removed: usize,
    unchanged: usize,
    total: usize,
}

pub fn run(
    root: Option<&str>,
    dir: Option<&str>,
    json_output: bool,
    no_register: bool,
) -> Result<i32> {
    let paths = resolve(root);
    std::fs::create_dir_all(paths.memory())?;
    std::fs::create_dir_all(paths.index())?;
    let cfg = load_config(&paths.config_path())?;
    let store = open_store(&paths.index(), &cfg, true, false)?;
    let memory_dir = dir.map_or(paths.memory(), PathBuf::from);
    let counts = markdown_store::reindex(&memory_dir, &store)?;
    store.save()?;
    drop(store);

    if !no_register && !registry::is_disabled() {
        // Best-effort: don't fail the command if the registry write
        // raced with another process.
        let _ = registry::register(&paths.root);
    }

    let payload = IndexCounts {
        added: counts.added,
        removed: counts.removed,
        unchanged: counts.unchanged,
        total: counts.total,
    };
    if json_output {
        println!("{}", serde_json::to_string(&payload)?);
    } else {
        println!(
            "indexed: added={} removed={} unchanged={} total={}",
            payload.added, payload.removed, payload.unchanged, payload.total
        );
    }
    Ok(0)
}

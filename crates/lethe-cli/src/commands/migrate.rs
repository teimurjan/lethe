//! `lethe migrate` — convert a legacy Python `embeddings.npz` into the
//! Rust-native `entry_embeddings` DuckDB table. One-shot per project.
//!
//! Behavior:
//!   * No-op (exit 0) when the npz is absent or already migrated.
//!   * Reads ids + f32 vectors from `<index>/embeddings.npz` and writes
//!     them in one transactional batch to the DuckDB store.
//!   * Renames the npz to `embeddings.npz.bak` so subsequent opens see
//!     a clean Rust-native store with no fallback path.
//!   * `--all` iterates every registered project in `~/.lethe/projects.json`.

use std::path::Path;

use anyhow::{Context, Result};
use lethe_core::db::MemoryDb;
use lethe_core::npz;
use lethe_core::registry;
use serde::Serialize;

use crate::paths::resolve;

#[derive(Serialize, Default)]
struct Migrated {
    projects: Vec<ProjectResult>,
}

#[derive(Serialize)]
struct ProjectResult {
    root: String,
    status: String,
    entries: usize,
}

pub fn run(root: Option<&str>, all: bool, json_output: bool) -> Result<i32> {
    let mut out = Migrated::default();

    if all {
        let entries = registry::load();
        if entries.is_empty() {
            eprintln!("no projects registered");
            return Ok(1);
        }
        for entry in entries {
            let res = migrate_one(&entry.root);
            out.projects.push(res);
        }
    } else {
        let paths = resolve(root);
        let res = migrate_one(&paths.root);
        out.projects.push(res);
    }

    if json_output {
        println!("{}", serde_json::to_string(&out)?);
    } else {
        for p in &out.projects {
            println!("{} — {} ({} entries)", p.root, p.status, p.entries);
        }
    }
    let any_failed = out.projects.iter().any(|p| p.status == "error");
    Ok(i32::from(any_failed))
}

fn migrate_one(project_root: &Path) -> ProjectResult {
    let root_str = project_root.to_string_lossy().into_owned();
    match try_migrate(project_root) {
        Ok((status, entries)) => ProjectResult {
            root: root_str,
            status,
            entries,
        },
        Err(e) => {
            eprintln!("[lethe] {}: {e}", project_root.display());
            ProjectResult {
                root: root_str,
                status: "error".into(),
                entries: 0,
            }
        }
    }
}

fn try_migrate(project_root: &Path) -> Result<(String, usize)> {
    let index = project_root.join(".lethe").join("index");
    let duckdb_path = index.join("lethe.duckdb");
    let npz_path = index.join("embeddings.npz");
    let bak_path = index.join("embeddings.npz.bak");

    if !duckdb_path.exists() {
        return Ok(("no-store".into(), 0));
    }
    if !npz_path.exists() {
        if bak_path.exists() {
            return Ok(("already-migrated".into(), 0));
        }
        return Ok(("no-npz".into(), 0));
    }

    let db = MemoryDb::open(&duckdb_path).context("open lethe.duckdb")?;
    if !db.embeddings_empty()? {
        // Rust store already has embeddings; rename the orphan npz so
        // future opens stop nudging the user.
        std::fs::rename(&npz_path, &bak_path).context("rename npz to .bak")?;
        return Ok(("already-migrated".into(), 0));
    }

    let map = npz::load_embeddings(&npz_path).context("read embeddings.npz")?;
    let count = map.len();
    let items: Vec<_> = map.into_iter().collect();
    db.save_embeddings_bulk(&items)
        .context("write entry_embeddings")?;
    std::fs::rename(&npz_path, &bak_path).context("rename npz to .bak")?;
    Ok(("migrated".into(), count))
}

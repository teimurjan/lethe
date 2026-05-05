//! `lethe status` — diagnostic JSON.

use anyhow::Result;
use lethe_core::db::MemoryDb;
use lethe_core::entry::Tier;
use serde::Serialize;
use std::collections::BTreeMap;

use crate::paths::resolve;

#[derive(Serialize)]
struct StatusEmpty<'a> {
    root: &'a str,
    initialized: bool,
    total_entries: i64,
}

#[derive(Serialize)]
struct StatusFull {
    root: String,
    memory_dir: String,
    initialized: bool,
    total_entries: i64,
    tiers: BTreeMap<String, i64>,
}

pub fn run(root: Option<&str>) -> Result<i32> {
    let paths = resolve(root);
    let db_path = paths.index().join("lethe.duckdb");
    // The directory may exist (e.g. partially-initialized after a
    // half-finished `lethe index`) without the DuckDB file — treat
    // that as uninitialized rather than erroring.
    if !paths.index().exists() || !db_path.exists() {
        let payload = StatusEmpty {
            root: paths.root.to_str().unwrap_or(""),
            initialized: false,
            total_entries: 0,
        };
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(0);
    }
    let db = MemoryDb::open_with_mode(db_path, true)?;
    let rows = db.load_all_entries()?;
    let total_entries = rows.len() as i64;
    let mut tiers: BTreeMap<String, i64> = BTreeMap::new();
    for t in [Tier::Naive, Tier::Gc, Tier::Memory, Tier::Apoptotic] {
        tiers.insert(t.as_str().to_owned(), 0);
    }
    for r in &rows {
        *tiers.entry(r.tier.as_str().to_owned()).or_insert(0) += 1;
    }
    drop(db);

    let payload = StatusFull {
        root: paths.root.to_string_lossy().into_owned(),
        memory_dir: paths.memory().to_string_lossy().into_owned(),
        initialized: true,
        total_entries,
        tiers,
    };
    println!("{}", serde_json::to_string_pretty(&payload)?);
    Ok(0)
}

//! `lethe expand <id1> [id2 id3 ...]` — print the full chunk body for each id.
//!
//! Multiple ids share a single DB open so the agent doesn't pay ONNX/DuckDB
//! load cost per chunk. Output is plain with `=== <id> ===` separators so
//! callers can split the stream.

use anyhow::Result;
use lethe_core::db::MemoryDb;

use crate::paths::resolve;

pub fn run(root: Option<&str>, chunk_ids: &[String]) -> Result<i32> {
    let paths = resolve(root);
    let db_path = paths.index().join("lethe.duckdb");
    if !db_path.exists() {
        for id in chunk_ids {
            eprintln!("chunk {id:?} not found");
        }
        return Ok(1);
    }
    let db = MemoryDb::open_with_mode(&db_path, true)?;
    let mut missing = false;
    for (idx, id) in chunk_ids.iter().enumerate() {
        if let Some(content) = db.get_content(id)? {
            if idx > 0 {
                println!();
            }
            println!("=== {id} ===");
            println!("{content}");
        } else {
            eprintln!("chunk {id:?} not found");
            missing = true;
        }
    }
    drop(db);
    Ok(i32::from(missing))
}

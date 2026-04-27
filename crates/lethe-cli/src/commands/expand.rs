//! `lethe expand <chunk_id>` — print the full chunk body.

use anyhow::Result;
use lethe_core::db::MemoryDb;

use crate::paths::resolve;

pub fn run(root: Option<&str>, chunk_id: &str) -> Result<i32> {
    let paths = resolve(root);
    let db_path = paths.index().join("lethe.duckdb");
    if !db_path.exists() {
        eprintln!("chunk {chunk_id:?} not found");
        return Ok(1);
    }
    let db = MemoryDb::open(&db_path)?;
    let content = db.get_content(chunk_id)?;
    drop(db);
    let Some(content) = content else {
        eprintln!("chunk {chunk_id:?} not found");
        return Ok(1);
    };
    println!("{content}");
    Ok(0)
}

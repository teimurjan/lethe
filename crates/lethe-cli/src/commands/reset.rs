//! `lethe reset` — wipe this project's global index directory.
//!
//! Transcripts under `~/.claude` / `~/.codex` are untouched, so the next
//! `lethe index` / `lethe search` rebuilds from them. This is also the
//! way to prune turns left behind by transcript compaction (a full
//! rebuild only re-adds turns still present in the current transcripts).

use anyhow::Result;

use crate::paths::resolve;

pub fn run(root: Option<&str>, yes: bool) -> Result<i32> {
    let paths = resolve(root);
    if !yes {
        eprintln!(
            "Would delete {} (transcripts are untouched; rebuild with `lethe index`). \
             Pass --yes to confirm.",
            paths.index().display()
        );
        return Ok(1);
    }
    if paths.index().exists() {
        std::fs::remove_dir_all(paths.index())?;
        println!("removed {}", paths.index().display());
    } else {
        println!("nothing to remove");
    }
    Ok(0)
}

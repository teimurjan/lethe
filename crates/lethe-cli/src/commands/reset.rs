//! `lethe reset` — wipe `.lethe/index/` (markdown stays).

use anyhow::Result;

use crate::paths::resolve;

pub fn run(root: Option<&str>, yes: bool) -> Result<i32> {
    let paths = resolve(root);
    if !yes {
        eprintln!(
            "Would delete {} (markdown in {} is preserved). Pass --yes to confirm.",
            paths.index().display(),
            paths.memory().display()
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

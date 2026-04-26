//! `Paths` helper — port of `legacy/lethe/cli.py::resolve_paths` and
//! `Paths`. Walks up to a git root if `--root` isn't given, falls
//! back to CWD.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Paths {
    pub root: PathBuf,
}

impl Paths {
    pub fn base(&self) -> PathBuf {
        self.root.join(".lethe")
    }
    pub fn memory(&self) -> PathBuf {
        self.base().join("memory")
    }
    pub fn index(&self) -> PathBuf {
        self.base().join("index")
    }
    pub fn config_path(&self) -> PathBuf {
        self.base().join("config.toml")
    }
    /// Default lock-file path used when the CLI takes the global mutex.
    /// Currently unused (Rust port doesn't take the lock yet) but kept
    /// for the upcoming wrapper that mirrors Python's `requires_lock`.
    #[allow(dead_code)]
    pub fn lock_path(&self) -> PathBuf {
        self.base().join("lethe.lock")
    }
}

pub fn resolve(root: Option<&str>) -> Paths {
    if let Some(r) = root {
        return Paths {
            root: PathBuf::from(r)
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from(r)),
        };
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if let Some(r) = walk_up_for_git(&cwd) {
        return Paths { root: r };
    }
    Paths { root: cwd }
}

fn walk_up_for_git(start: &Path) -> Option<PathBuf> {
    let mut current: Option<&Path> = Some(start);
    while let Some(c) = current {
        if c.join(".git").exists() {
            return Some(c.to_path_buf());
        }
        current = c.parent();
    }
    None
}

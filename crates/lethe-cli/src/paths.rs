//! `Paths` helper — port of `research_playground/lethe_reference/lethe/cli.py::resolve_paths` and
//! `Paths`. Walks up to a git root if `--root` isn't given, falls
//! back to CWD.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Paths {
    pub root: PathBuf,
}

impl Paths {
    /// Stable per-project identifier (`p_<base>_<sha1[:8]>`), matching the
    /// slug the registry records. Also names the global index directory.
    pub fn slug(&self) -> String {
        lethe_core::registry::slugify(&self.root)
    }
    /// Per-project index directory, global under `~/.lethe/index/<slug>/`.
    /// Nothing is written into the user's repo.
    pub fn index(&self) -> PathBuf {
        lethe_core::registry::registry_dir()
            .join("index")
            .join(self.slug())
    }
    /// Single global config shared by every project — encoder choice is
    /// rarely per-project. Takes `&self` for call-site ergonomics
    /// (`paths.config_path()`) though the path is project-independent.
    #[allow(clippy::unused_self)]
    pub fn config_path(&self) -> PathBuf {
        lethe_core::registry::registry_dir().join("config.toml")
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
        let unified = main_worktree(&r).unwrap_or(r);
        return Paths { root: unified };
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

/// Resolve the main worktree for a git repo so every worktree of the same
/// repo maps to a single `.lethe/` location. Returns `None` when git is
/// unavailable, the lookup fails, or the repo is bare with no checkout.
pub(crate) fn main_worktree(start: &Path) -> Option<PathBuf> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(start)
        .args(["worktree", "list", "--porcelain"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The first record is always the main worktree. Records are
    // separated by blank lines; a `bare` flag inside the first record
    // means the repo has no checkout to unify into.
    let mut wt: Option<PathBuf> = None;
    let mut bare = false;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("worktree ") {
            wt = Some(PathBuf::from(rest.trim()));
            bare = false;
        } else if line == "bare" {
            bare = true;
        } else if line.is_empty() && wt.is_some() {
            return if bare { None } else { wt };
        }
    }
    if bare {
        None
    } else {
        wt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn run(dir: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .status()
            .expect("git missing");
        assert!(status.success(), "git {args:?} failed");
    }

    fn init_repo(dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        run(dir, &["init", "-q", "-b", "main"]);
        run(dir, &["config", "user.email", "t@t"]);
        run(dir, &["config", "user.name", "t"]);
        run(dir, &["commit", "--allow-empty", "-q", "-m", "init"]);
    }

    fn canon(p: &Path) -> PathBuf {
        p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
    }

    #[test]
    fn main_worktree_from_main_tree_returns_self() {
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("repo");
        init_repo(&main);

        let got = main_worktree(&main).expect("expected main worktree");
        assert_eq!(canon(&got), canon(&main));
    }

    #[test]
    fn main_worktree_from_linked_worktree_returns_main() {
        let tmp = tempfile::tempdir().unwrap();
        let main = tmp.path().join("repo");
        let wt = tmp.path().join("wt");
        init_repo(&main);
        run(
            &main,
            &["worktree", "add", "-q", wt.to_str().unwrap(), "-b", "feat"],
        );

        let got = main_worktree(&wt).expect("expected main worktree");
        assert_eq!(canon(&got), canon(&main));
    }

    #[test]
    fn main_worktree_returns_none_for_non_git_dir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(main_worktree(tmp.path()).is_none());
    }
}

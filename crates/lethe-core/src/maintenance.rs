//! Cleanup operations shared by the CLI and TUI.
//!
//! Two axes of "cleanup":
//! * **Empty projects** — registered projects whose index exists but holds
//!   zero memories (indexed, produced nothing). [`empty_projects`] finds
//!   them; [`delete_project_data`] removes lethe's data (and, optionally,
//!   the project's on-disk transcripts).
//! * **Stale transcripts** — Claude/Codex transcripts on disk whose repo is
//!   gone or that yield no memories. [`scan_stale_transcripts`] flags them;
//!   [`delete_transcripts`] reclaims the disk.
//!
//! All transcript deletion is irreversible, so callers must preview +
//! confirm. Detection never decodes the Claude folder slug (lossy) — it
//! reads the recorded `cwd` and tests it with [`Path::exists`].

use std::path::{Path, PathBuf};

use crate::db::MemoryDb;
use crate::registry::{self, ProjectEntry};
use crate::transcript_index::{
    self, claude_config_dir, claude_project_dirs, codex_session_files, jsonl_recursive,
    path_to_claude_slug, read_cwd, Source,
};

/// Human-readable byte size (e.g. `3.4 MB`).
#[must_use]
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// `~/.lethe/index/<slug>/`.
#[must_use]
pub fn index_dir(slug: &str) -> PathBuf {
    registry::registry_dir().join("index").join(slug)
}

/// Live memory count for a slug. `None` when the project has no index dir
/// (never indexed) — distinct from `Some(0)` (indexed, empty).
#[must_use]
pub fn count_memories(slug: &str) -> Option<i64> {
    let db_path = index_dir(slug).join("lethe.duckdb");
    if !db_path.exists() {
        return None;
    }
    MemoryDb::open_with_mode(&db_path, true).ok()?.count().ok()
}

/// Registered projects whose index exists and holds zero memories.
/// Never-indexed projects (no index dir) are deliberately excluded.
#[must_use]
pub fn empty_projects() -> Vec<ProjectEntry> {
    registry::load()
        .into_iter()
        .filter(|e| count_memories(&e.slug) == Some(0))
        .collect()
}

/// What a delete/cleanup reclaimed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Reclaimed {
    pub projects: usize,
    pub transcripts: usize,
    /// Individual memory rows deleted (age-based "delete state records").
    pub records: usize,
    pub bytes: u64,
}

impl Reclaimed {
    fn add(&mut self, other: Reclaimed) {
        self.projects += other.projects;
        self.transcripts += other.transcripts;
        self.records += other.records;
        self.bytes += other.bytes;
    }
}

/// Delete stored memory records from the given projects. `older_than_days
/// == 0` wipes each project's whole index (via [`delete_project_data`],
/// keeping transcripts on disk); a positive value deletes only rows first
/// indexed before the cutoff. Callers must confirm — this is irreversible.
#[must_use]
pub fn delete_records(entries: &[ProjectEntry], older_than_days: u32) -> Reclaimed {
    let mut r = Reclaimed::default();
    if older_than_days == 0 {
        for e in entries {
            r.add(delete_project_data(e, false));
        }
        return r;
    }
    let cutoff = now_epoch().saturating_sub(i64::from(older_than_days) * 86_400);
    for e in entries {
        let db_path = index_dir(&e.slug).join("lethe.duckdb");
        if !db_path.exists() {
            continue;
        }
        if let Ok(db) = MemoryDb::open_with_mode(&db_path, false) {
            if let Ok(n) = db.delete_entries_older_than(cutoff) {
                r.records += n;
            }
        }
    }
    r
}

/// Current unix time in whole seconds.
fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Remove lethe's data for a project (unregister + drop its index dir).
/// When `delete_transcripts`, also delete its Claude folder and matching
/// Codex session files from disk. Transcripts are always safe to leave;
/// deleting them is irreversible.
pub fn delete_project_data(entry: &ProjectEntry, delete_transcripts: bool) -> Reclaimed {
    let mut r = Reclaimed::default();
    let _ = registry::unregister(&entry.slug);
    let dir = index_dir(&entry.slug);
    let bytes = dir_size(&dir);
    if std::fs::remove_dir_all(&dir).is_ok() {
        r.projects += 1;
        r.bytes += bytes;
    }
    if delete_transcripts {
        r.add(delete_project_transcripts(entry));
    }
    r
}

/// Delete a project's transcript files: its Claude project folder (direct
/// slug mapping from the known root) and every Codex session whose `cwd`
/// matches the root.
fn delete_project_transcripts(entry: &ProjectEntry) -> Reclaimed {
    let mut r = Reclaimed::default();
    let target = canon(&entry.root);

    let claude_dir = claude_config_dir()
        .join("projects")
        .join(path_to_claude_slug(&entry.root));
    if claude_dir.is_dir() {
        let bytes = dir_size(&claude_dir);
        if std::fs::remove_dir_all(&claude_dir).is_ok() {
            r.transcripts += 1;
            r.bytes += bytes;
        }
    }

    for path in codex_session_files() {
        let matches =
            read_cwd(&path, Source::Codex).is_some_and(|c| canon(Path::new(&c)) == target);
        if matches {
            r.add(remove_file(&path));
        }
    }
    prune_empty_dirs(&transcript_index::codex_home().join("sessions"));
    r
}

/// Why a transcript is flagged for deletion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaleReason {
    /// The recorded `cwd` no longer exists on disk.
    RepoGone,
    /// The transcript yields no indexable memories.
    NoMemories,
}

impl StaleReason {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            StaleReason::RepoGone => "repo gone",
            StaleReason::NoMemories => "no memories",
        }
    }
}

/// A transcript path flagged for deletion. `path` is a Claude project
/// **folder** or a single Codex session **file**.
#[derive(Debug, Clone)]
pub struct StaleTranscript {
    pub path: PathBuf,
    pub source: Source,
    pub cwd: Option<String>,
    pub reason: StaleReason,
    pub bytes: u64,
}

/// Scan Claude + Codex storage for transcripts whose repo is gone or that
/// yield no memories. Parse-only (no encoders); can be slow on large
/// histories, so callers should run it off the UI thread.
#[must_use]
pub fn scan_stale_transcripts() -> Vec<StaleTranscript> {
    let mut out = Vec::new();

    // Claude: one candidate per project folder (transcripts may nest in
    // <session-id>/subagents/, so scan recursively; also read the session
    // index, which retains pruned sessions' first prompts + project path).
    for dir in claude_project_dirs() {
        let mut files = jsonl_recursive(&dir);
        let idx = dir.join("sessions-index.json");
        if idx.is_file() {
            files.push(idx);
        }
        let mut cwd: Option<String> = None;
        let mut has_memories = false;
        for f in &files {
            let parsed = transcript_index::parse_file(f, Source::ClaudeCode);
            if cwd.is_none() {
                cwd = parsed.cwd;
            }
            if !parsed.chunks.is_empty() {
                has_memories = true;
            }
        }
        if let Some(reason) = stale_reason(cwd.as_deref(), has_memories) {
            out.push(StaleTranscript {
                bytes: dir_size(&dir),
                path: dir,
                source: Source::ClaudeCode,
                cwd,
                reason,
            });
        }
    }

    // Codex: one candidate per session file.
    for path in codex_session_files() {
        let parsed = transcript_index::parse_file(&path, Source::Codex);
        if let Some(reason) = stale_reason(parsed.cwd.as_deref(), !parsed.chunks.is_empty()) {
            out.push(StaleTranscript {
                bytes: file_size(&path),
                path,
                source: Source::Codex,
                cwd: parsed.cwd,
                reason,
            });
        }
    }

    out
}

/// Flag rule: repo-gone (cwd recorded but missing) OR yields no memories.
/// An unknown cwd is not treated as "gone" (we can't verify), so such a
/// transcript is stale only if it also has no memories.
fn stale_reason(cwd: Option<&str>, has_memories: bool) -> Option<StaleReason> {
    let gone = cwd.is_some_and(|c| !Path::new(c).exists());
    if gone {
        Some(StaleReason::RepoGone)
    } else if !has_memories {
        Some(StaleReason::NoMemories)
    } else {
        None
    }
}

/// Delete the given stale transcripts (folders or files). Best-effort;
/// counts only what was actually removed.
pub fn delete_transcripts(items: &[StaleTranscript]) -> Reclaimed {
    let mut r = Reclaimed::default();
    for it in items {
        let removed = if it.path.is_dir() {
            std::fs::remove_dir_all(&it.path).is_ok()
        } else {
            std::fs::remove_file(&it.path).is_ok()
        };
        if removed {
            r.transcripts += 1;
            r.bytes += it.bytes;
        }
    }
    prune_empty_dirs(&transcript_index::codex_home().join("sessions"));
    r
}

fn remove_file(path: &Path) -> Reclaimed {
    let bytes = file_size(path);
    if std::fs::remove_file(path).is_ok() {
        Reclaimed {
            transcripts: 1,
            bytes,
            ..Default::default()
        }
    } else {
        Reclaimed::default()
    }
}

fn canon(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

fn file_size(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut total = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.metadata() {
            Ok(m) if m.is_dir() => total += dir_size(&path),
            Ok(m) => total += m.len(),
            Err(_) => {}
        }
    }
    total
}

/// Recursively remove now-empty directories under `root` (but not `root`
/// itself). Best-effort cleanup after deleting Codex session files.
fn prune_empty_dirs(root: &Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            prune_empty_dirs(&path);
            if std::fs::read_dir(&path)
                .map(|mut d| d.next().is_none())
                .unwrap_or(false)
            {
                let _ = std::fs::remove_dir(&path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::MutexGuard;
    use tempfile::tempdir;

    struct EnvGuard {
        _lock: MutexGuard<'static, ()>,
        keys: Vec<(&'static str, Option<String>)>,
    }
    impl EnvGuard {
        // Serialized against every other HOME-mutating test via the
        // shared crate lock (these also set CLAUDE_CONFIG_DIR/CODEX_HOME).
        fn set(keys: &[(&'static str, &Path)]) -> Self {
            let lock = crate::TEST_HOME_LOCK
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            let mut saved = Vec::new();
            for (k, v) in keys {
                saved.push((*k, std::env::var(k).ok()));
                std::env::set_var(k, v);
            }
            EnvGuard {
                _lock: lock,
                keys: saved,
            }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.keys {
                match v {
                    Some(val) => std::env::set_var(k, val),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
    }

    #[test]
    fn scan_flags_repo_gone_and_no_memories() {
        let home = tempdir().unwrap();
        let cfg = tempdir().unwrap();
        let codex = tempdir().unwrap();
        let _g = EnvGuard::set(&[
            ("HOME", home.path()),
            ("CLAUDE_CONFIG_DIR", cfg.path()),
            ("CODEX_HOME", codex.path()),
        ]);

        // A live repo the transcript points at (exists on disk).
        let live_repo = tempdir().unwrap();
        let live = live_repo.path().to_string_lossy();

        let projects = cfg.path().join("projects");
        // Alive + has memories → NOT flagged.
        write(
            &projects.join("alive").join("s.jsonl"),
            &format!(
                "{{\"type\":\"attachment\",\"cwd\":\"{live}\",\"sessionId\":\"a\"}}\n\
                 {{\"type\":\"user\",\"uuid\":\"u1\",\"sessionId\":\"a\",\"message\":{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"hi\"}}]}}}}\n\
                 {{\"type\":\"assistant\",\"sessionId\":\"a\",\"message\":{{\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"yo\"}}]}}}}\n"
            ),
        );
        // Repo gone → flagged RepoGone.
        write(
            &projects.join("dead").join("s.jsonl"),
            "{\"type\":\"attachment\",\"cwd\":\"/no/such/path/xyz\",\"sessionId\":\"b\"}\n\
             {\"type\":\"user\",\"uuid\":\"u1\",\"sessionId\":\"b\",\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}\n\
             {\"type\":\"assistant\",\"sessionId\":\"b\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"yo\"}]}}\n",
        );

        let stale = scan_stale_transcripts();
        let dead = stale
            .iter()
            .find(|s| s.path.ends_with("dead"))
            .expect("dead flagged");
        assert_eq!(dead.reason, StaleReason::RepoGone);
        assert!(
            !stale.iter().any(|s| s.path.ends_with("alive")),
            "alive project must not be flagged"
        );
    }

    #[test]
    fn scan_counts_subagent_transcripts_in_subdirs() {
        let home = tempdir().unwrap();
        let cfg = tempdir().unwrap();
        let _g = EnvGuard::set(&[("HOME", home.path()), ("CLAUDE_CONFIG_DIR", cfg.path())]);
        let live_repo = tempdir().unwrap();
        let live = live_repo.path().to_string_lossy();

        // A project folder with NO top-level jsonl, only a subagent
        // transcript nested in <session-id>/subagents/. Recursion must
        // find it → the folder has memories → not flagged.
        let sub = cfg
            .path()
            .join("projects")
            .join("nested")
            .join("sess-1")
            .join("subagents");
        write(
            &sub.join("agent-x.jsonl"),
            &format!(
                "{{\"type\":\"user\",\"uuid\":\"u1\",\"sessionId\":\"a\",\"cwd\":\"{live}\",\"message\":{{\"role\":\"user\",\"content\":[{{\"type\":\"text\",\"text\":\"hi\"}}]}}}}\n\
                 {{\"type\":\"assistant\",\"sessionId\":\"a\",\"message\":{{\"role\":\"assistant\",\"content\":[{{\"type\":\"text\",\"text\":\"yo\"}}]}}}}\n"
            ),
        );

        let stale = scan_stale_transcripts();
        assert!(
            !stale.iter().any(|s| s.path.ends_with("nested")),
            "folder with subagent memories must not be flagged: {stale:?}"
        );
    }

    #[test]
    fn empty_projects_skips_never_indexed() {
        let home = tempdir().unwrap();
        let _g = EnvGuard::set(&[("HOME", home.path())]);
        // No index dirs, no registry → nothing counted as empty.
        assert!(empty_projects().is_empty());
        assert_eq!(count_memories("p_missing_00000000"), None);
    }

    #[test]
    fn delete_project_data_removes_index_registry_and_transcripts() {
        let home = tempdir().unwrap();
        let cfg = tempdir().unwrap();
        let repo = tempdir().unwrap();
        let _g = EnvGuard::set(&[("HOME", home.path()), ("CLAUDE_CONFIG_DIR", cfg.path())]);

        // Register the project and give it an (empty) index so it counts
        // as an empty project.
        let entry = registry::register(repo.path()).unwrap();
        let idx = index_dir(&entry.slug);
        MemoryDb::open(idx.join("lethe.duckdb")).unwrap(); // writes schema, 0 rows
        assert_eq!(count_memories(&entry.slug), Some(0));
        assert!(empty_projects().iter().any(|e| e.slug == entry.slug));

        // Its Claude transcript folder (direct slug mapping from the root).
        let claude_dir = cfg
            .path()
            .join("projects")
            .join(path_to_claude_slug(&entry.root));
        write(&claude_dir.join("s.jsonl"), "{}\n");

        let r = delete_project_data(&entry, true);
        assert_eq!(r.projects, 1);
        assert_eq!(r.transcripts, 1);
        assert!(!idx.exists(), "index dir removed");
        assert!(!claude_dir.exists(), "claude transcripts removed");
        assert!(registry::find(&entry.slug).is_none(), "unregistered");
    }
}

//! Transcript discovery + incremental indexing.
//!
//! lethe indexes the agent transcripts (Claude Code / Codex JSONL) directly,
//! at turn granularity, into the per-project store.
//!
//! Claude Code writes one JSONL per session under
//! `$CLAUDE_CONFIG_DIR/projects/<slug>/` where `<slug>` is the launch cwd
//! with every `/` replaced by `-`. Because a session launched from a git
//! worktree records the worktree path (a different slug than the unified
//! main worktree), discovery enumerates every worktree of the repo and
//! scans all matching slugs, filtering by the transcript's recorded `cwd`.
//!
//! Codex writes JSONL under `$CODEX_HOME/sessions/**` with no per-project
//! directory, so its whole tree is walked and filtered by `cwd`.
//!
//! [`ensure_fresh`] is the freshness hook run before a search: it stats
//! every candidate transcript, reparses only those whose (mtime, size)
//! diverge from the manifest, and add-only-syncs their turns.
//!
//! Parsing ([`parse_claude_file`] / [`parse_codex_file`]) returns the
//! recorded `cwd` alongside the turn chunks and does **not** filter by
//! cwd — the caller decides. `ensure_fresh` applies the cwd filter; the
//! `maintenance` module reuses the raw parse to scan for stale transcripts.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde_json::Value;

use crate::memory_store::MemoryStore;
use crate::transcript_store::{build_chunk, sync, SyncCounts, TurnChunk};

/// Which agent sources to scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    ClaudeCode,
    Codex,
}

#[must_use]
pub fn all_sources() -> Vec<Source> {
    vec![Source::ClaudeCode, Source::Codex]
}

/// A parsed transcript: its recorded working directory (if any) plus the
/// turn chunks it produced. Unfiltered — the caller applies cwd rules.
#[derive(Debug, Default)]
pub struct ParsedFile {
    pub cwd: Option<String>,
    pub chunks: Vec<TurnChunk>,
}

/// Freshen the store from any changed transcripts, then return the sync
/// counts. Reads the manifest through the (already open, read-write)
/// store, reparses only dirty files, and records their new stat.
pub fn ensure_fresh(store: &MemoryStore, project_root: &Path) -> crate::Result<SyncCounts> {
    let files = discover_files(project_root, &all_sources());
    if files.is_empty() {
        return Ok(SyncCounts::default());
    }
    let manifest = store.with_db(crate::db::MemoryDb::get_manifest)?;
    let cwds = expected_cwds(project_root);

    let mut total = SyncCounts::default();
    for f in files {
        let path_str = f.path.to_string_lossy().into_owned();
        if let Some(&(mtime, size)) = manifest.get(&path_str) {
            if mtime == f.mtime_ns && size == f.size_bytes {
                continue; // unchanged — append-only guarantee
            }
        }
        let parsed = parse_file(&f.path, f.source);
        // Skip turns whose recorded cwd belongs to a different project;
        // an absent cwd is treated as a match (same as before).
        let chunks = if parsed.cwd.as_deref().is_some_and(|c| !cwds.contains(c)) {
            Vec::new()
        } else {
            parsed.chunks
        };
        let counts = sync(&chunks, store)?;
        store.with_db(|db| {
            db.upsert_manifest_row(&path_str, f.mtime_ns, f.size_bytes, chunks.len() as i64)
        })?;
        total.added += counts.added;
        total.unchanged += counts.unchanged;
        total.total += counts.total;
    }
    Ok(total)
}

/// Parse a transcript file by source into `(cwd, chunks)`.
#[must_use]
pub fn parse_file(path: &Path, source: Source) -> ParsedFile {
    match source {
        Source::ClaudeCode => parse_claude_file(path),
        Source::Codex => parse_codex_file(path),
    }
}

/// Cheaply read a transcript's recorded `cwd` without materializing turns
/// (stops at the first `cwd` field). Used by cleanup to match/verify a
/// project without a full parse.
#[must_use]
pub fn read_cwd(path: &Path, source: Source) -> Option<String> {
    let f = fs::File::open(path).ok()?;
    for line in BufReader::new(f).lines() {
        let Ok(line) = line else { continue };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        let holder = match source {
            Source::ClaudeCode => &rec,
            Source::Codex => rec.get("payload").unwrap_or(&rec),
        };
        if let Some(c) = holder.get("cwd").and_then(|v| v.as_str()) {
            if !c.is_empty() {
                return Some(c.to_owned());
            }
        }
    }
    None
}

/// A discovered transcript file with its freshness stat.
struct FileEntry {
    source: Source,
    path: PathBuf,
    mtime_ns: i64,
    size_bytes: i64,
}

fn stat_ns(meta: &fs::Metadata) -> (i64, i64) {
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos() as i64);
    (mtime, meta.len() as i64)
}

/// All candidate transcript files across the requested sources.
fn discover_files(project_root: &Path, sources: &[Source]) -> Vec<FileEntry> {
    let mut out = Vec::new();
    if sources.contains(&Source::ClaudeCode) {
        for path in claude_files_for(project_root) {
            if let Ok(meta) = fs::metadata(&path) {
                let (mtime_ns, size_bytes) = stat_ns(&meta);
                out.push(FileEntry {
                    source: Source::ClaudeCode,
                    path,
                    mtime_ns,
                    size_bytes,
                });
            }
        }
    }
    if sources.contains(&Source::Codex) {
        for path in codex_session_files() {
            if let Ok(meta) = fs::metadata(&path) {
                let (mtime_ns, size_bytes) = stat_ns(&meta);
                out.push(FileEntry {
                    source: Source::Codex,
                    path,
                    mtime_ns,
                    size_bytes,
                });
            }
        }
    }
    out
}

/// `$CLAUDE_CONFIG_DIR` (default `~/.claude`).
#[must_use]
pub fn claude_config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    home().join(".claude")
}

/// `$CODEX_HOME` (default `~/.codex`).
#[must_use]
pub fn codex_home() -> PathBuf {
    if let Some(dir) = std::env::var_os("CODEX_HOME") {
        return PathBuf::from(dir);
    }
    home().join(".codex")
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// Claude Code's project-dir encoding: `/` → `-` on the absolute path.
/// One-way (a `-` in a path segment is indistinguishable from a separator),
/// so never decode a folder name — read the transcript's `cwd` instead.
#[must_use]
pub fn path_to_claude_slug(p: &Path) -> String {
    p.to_string_lossy().replace('/', "-")
}

/// Every immediate subdirectory of `$CLAUDE_CONFIG_DIR/projects` (one per
/// launch cwd Claude Code has ever seen).
#[must_use]
pub fn claude_project_dirs() -> Vec<PathBuf> {
    let projects = claude_config_dir().join("projects");
    let Ok(entries) = fs::read_dir(&projects) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect()
}

/// Every `*.jsonl` under `$CODEX_HOME/sessions` (recursively).
#[must_use]
pub fn codex_session_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let sessions = codex_home().join("sessions");
    if sessions.exists() {
        walk_jsonl(&sessions, &mut out);
    }
    out
}

/// Every `*.jsonl` under a directory, recursively. Claude Code nests
/// subagent transcripts in `<session-id>/subagents/*.jsonl`, so a
/// non-recursive scan would miss those memories.
#[must_use]
pub fn jsonl_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_jsonl(dir, &mut out);
    out
}

fn walk_jsonl(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            walk_jsonl(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            out.push(path);
        }
    }
}

/// Claude transcript files for a project: the union over every worktree
/// root of the repo (deduped).
fn claude_files_for(project_root: &Path) -> Vec<PathBuf> {
    let projects = claude_config_dir().join("projects");
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut out = Vec::new();
    for root in worktree_roots(project_root) {
        for path in jsonl_recursive(&projects.join(path_to_claude_slug(&root))) {
            if seen.insert(path.clone()) {
                out.push(path);
            }
        }
    }
    out
}

/// Every worktree root of the repo containing `project_root`, so sessions
/// launched from linked worktrees are discovered too. Falls back to just
/// `project_root` when git is unavailable.
fn worktree_roots(project_root: &Path) -> Vec<PathBuf> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(project_root)
        .args(["worktree", "list", "--porcelain"])
        .output();
    let mut roots = Vec::new();
    if let Ok(o) = out {
        if o.status.success() {
            for line in String::from_utf8_lossy(&o.stdout).lines() {
                if let Some(rest) = line.strip_prefix("worktree ") {
                    roots.push(PathBuf::from(rest.trim()));
                }
            }
        }
    }
    if roots.is_empty() {
        roots.push(project_root.to_path_buf());
    }
    roots
}

/// Canonicalized string form of every worktree root — the set a
/// transcript's recorded `cwd` must belong to.
fn expected_cwds(project_root: &Path) -> HashSet<String> {
    worktree_roots(project_root)
        .into_iter()
        .map(|r| r.canonicalize().unwrap_or(r).to_string_lossy().into_owned())
        .collect()
}

// -------- Claude Code parsing (per-turn) --------

fn parse_claude_file(path: &Path) -> ParsedFile {
    let Ok(f) = fs::File::open(path) else {
        return ParsedFile::default();
    };
    let reader = BufReader::new(f);
    let mut session_id: Option<String> = None;
    let mut cwd: Option<String> = None;
    let mut pending_user: Option<(String, String)> = None; // (uuid, text)
    let mut pairs: Vec<(String, String, String)> = Vec::new(); // (uuid, user, assistant)

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        if session_id.is_none() {
            if let Some(id) = rec.get("sessionId").and_then(|v| v.as_str()) {
                session_id = Some(id.to_owned());
            }
        }
        if cwd.is_none() {
            if let Some(c) = rec.get("cwd").and_then(|v| v.as_str()) {
                if !c.is_empty() {
                    cwd = Some(c.to_owned());
                }
            }
        }
        let kind = role_of_claude(&rec);
        let msg = rec.get("message").unwrap_or(&rec);
        match kind.as_deref() {
            Some("user") => {
                if claude_is_tool_result_only(msg) {
                    continue;
                }
                let text = claude_message_text(msg);
                if text.is_empty() {
                    continue;
                }
                let uid = rec
                    .get("uuid")
                    .or_else(|| rec.get("id"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                pending_user = Some((uid, text));
            }
            Some("assistant") => {
                let text = claude_message_text(msg);
                if text.is_empty() {
                    continue;
                }
                if let Some((uid, user_text)) = pending_user.take() {
                    pairs.push((uid, user_text, text));
                } else if let Some(prev) = pairs.last_mut() {
                    prev.2.push('\n');
                    prev.2.push_str(&text);
                }
            }
            _ => {}
        }
    }

    let Some(session_id) = session_id else {
        return ParsedFile {
            cwd,
            chunks: Vec::new(),
        };
    };
    let path_str = path.to_string_lossy();
    let chunks = pairs
        .into_iter()
        .enumerate()
        .map(|(i, (uid, user, assistant))| {
            let turn = if uid.is_empty() { "LAST" } else { &uid };
            build_chunk(&session_id, turn, &path_str, &user, &assistant, i as i64)
        })
        .collect();
    ParsedFile { cwd, chunks }
}

fn role_of_claude(rec: &Value) -> Option<String> {
    rec.get("message")
        .and_then(|m| m.get("role"))
        .and_then(|v| v.as_str())
        .or_else(|| rec.get("type").and_then(|v| v.as_str()))
        .map(str::to_owned)
}

fn claude_is_tool_result_only(msg: &Value) -> bool {
    let Some(content) = msg.get("content").and_then(|c| c.as_array()) else {
        return false;
    };
    if content.is_empty() {
        return false;
    }
    content.iter().all(|b| {
        b.get("type")
            .and_then(|t| t.as_str())
            .is_some_and(|t| t == "tool_result")
    })
}

fn claude_message_text(msg: &Value) -> String {
    if let Some(s) = msg.get("content").and_then(|c| c.as_str()) {
        return s.to_owned();
    }
    let Some(blocks) = msg.get("content").and_then(|c| c.as_array()) else {
        return String::new();
    };
    let mut parts: Vec<String> = Vec::new();
    for b in blocks {
        let Some(btype) = b.get("type").and_then(|t| t.as_str()) else {
            continue;
        };
        match btype {
            "text" => {
                if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                    parts.push(t.to_owned());
                }
            }
            "tool_use" => {
                let name = b.get("name").and_then(|n| n.as_str()).unwrap_or("?");
                parts.push(format!("[tool_use: {name}]"));
            }
            _ => {}
        }
    }
    parts.join("\n")
}

// -------- Codex parsing (per-turn) --------

fn parse_codex_file(path: &Path) -> ParsedFile {
    let Ok(f) = fs::File::open(path) else {
        return ParsedFile::default();
    };
    let reader = BufReader::new(f);
    let mut session_id: Option<String> = None;
    let mut cwd: Option<String> = None;
    let mut pending_user: Option<String> = None;
    let mut pending_turn: Option<String> = None;
    let mut pairs: Vec<(String, String, String)> = Vec::new(); // (turn_id, user, assistant)

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        let outer = rec.get("type").and_then(|v| v.as_str());
        let payload = rec.get("payload").unwrap_or(&rec);
        match outer {
            Some("session_meta") => {
                if let Some(id) = payload.get("id").and_then(|v| v.as_str()) {
                    session_id = Some(id.to_owned());
                }
                if let Some(c) = payload.get("cwd").and_then(|v| v.as_str()) {
                    if !c.is_empty() {
                        cwd = Some(c.to_owned());
                    }
                }
            }
            Some("event_msg") => {
                let inner = payload.get("type").and_then(|v| v.as_str());
                match inner {
                    Some("user_message") => {
                        let text = payload
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        if !text.is_empty() {
                            pending_user = Some(text);
                            pending_turn = None;
                        }
                    }
                    Some("turn_started") => {
                        if let Some(t) = payload.get("turn_id").and_then(|v| v.as_str()) {
                            pending_turn = Some(t.to_owned());
                        }
                    }
                    Some("agent_message") => {
                        let text = payload
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        if text.is_empty() {
                            continue;
                        }
                        if let Some(u) = pending_user.take() {
                            let turn = pending_turn.take().unwrap_or_default();
                            pairs.push((turn, u, text));
                        } else if let Some(prev) = pairs.last_mut() {
                            prev.2.push('\n');
                            prev.2.push_str(&text);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let Some(session_id) = session_id else {
        return ParsedFile {
            cwd,
            chunks: Vec::new(),
        };
    };
    let path_str = path.to_string_lossy();
    let chunks = pairs
        .into_iter()
        .enumerate()
        .map(|(i, (turn_id, user, assistant))| {
            let turn = if turn_id.is_empty() { "LAST" } else { &turn_id };
            build_chunk(&session_id, turn, &path_str, &user, &assistant, i as i64)
        })
        .collect();
    ParsedFile { cwd, chunks }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_jsonl(name: &str, lines: &[&str]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "lethe-tidx-test-{}-{name}.jsonl",
            std::process::id()
        ));
        let mut f = fs::File::create(&path).unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        path
    }

    #[test]
    fn slug_encodes_path() {
        assert_eq!(
            path_to_claude_slug(Path::new("/Users/me/Projects/foo")),
            "-Users-me-Projects-foo"
        );
    }

    #[test]
    fn parse_claude_emits_one_chunk_per_pair_with_cwd() {
        let p = write_jsonl(
            "claude",
            &[
                r#"{"type":"attachment","cwd":"/repo","sessionId":"sess-1"}"#,
                r#"{"type":"user","uuid":"u1","sessionId":"sess-1","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}"#,
                r#"{"type":"assistant","sessionId":"sess-1","message":{"role":"assistant","content":[{"type":"text","text":"hi back"}]}}"#,
                r#"{"type":"user","uuid":"u2","sessionId":"sess-1","message":{"role":"user","content":[{"type":"text","text":"again"}]}}"#,
                r#"{"type":"assistant","sessionId":"sess-1","message":{"role":"assistant","content":[{"type":"text","text":"sure"}]}}"#,
            ],
        );
        let parsed = parse_claude_file(&p);
        assert_eq!(parsed.cwd.as_deref(), Some("/repo"));
        assert_eq!(parsed.chunks.len(), 2);
        assert!(parsed.chunks[0].content.contains("turn:u1"));
        assert!(parsed.chunks[1].content.contains("again"));
        fs::remove_file(&p).ok();
    }

    #[test]
    fn parse_codex_emits_pairs_with_turn_ids_and_cwd() {
        let p = write_jsonl(
            "codex",
            &[
                r#"{"type":"session_meta","payload":{"id":"sess-c","cwd":"/repo"}}"#,
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t1"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"hello"}}"#,
            ],
        );
        let parsed = parse_codex_file(&p);
        assert_eq!(parsed.cwd.as_deref(), Some("/repo"));
        assert_eq!(parsed.chunks.len(), 1);
        assert!(parsed.chunks[0].content.contains("turn:t1"));
        fs::remove_file(&p).ok();
    }
}

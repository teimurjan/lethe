//! Transcript discovery + incremental indexing.
//!
//! Replaces the old `seed` command and the Haiku summarization pipeline.
//! Instead of summarizing turns into `.lethe/memory/*.md`, lethe indexes
//! the agent transcripts (Claude Code / Codex JSONL) directly, at turn
//! granularity, into the per-project store.
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

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use lethe_core::memory_store::MemoryStore;
use lethe_core::transcript_store::{build_chunk, sync, SyncCounts, TurnChunk};
use serde_json::Value;

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

/// Freshen the store from any changed transcripts, then return the sync
/// counts. Reads the manifest through the (already open, read-write)
/// store, reparses only dirty files, and records their new stat.
pub fn ensure_fresh(store: &MemoryStore, project_root: &Path) -> Result<SyncCounts> {
    let files = discover_files(project_root, &all_sources());
    if files.is_empty() {
        return Ok(SyncCounts::default());
    }
    let manifest = store.with_db(lethe_core::db::MemoryDb::get_manifest)?;
    let cwds = expected_cwds(project_root);

    let mut total = SyncCounts::default();
    for f in files {
        let path_str = f.path.to_string_lossy().into_owned();
        if let Some(&(mtime, size)) = manifest.get(&path_str) {
            if mtime == f.mtime_ns && size == f.size_bytes {
                continue; // unchanged — append-only guarantee
            }
        }
        let chunks = match f.source {
            Source::ClaudeCode => parse_claude_file(&f.path, &cwds),
            Source::Codex => parse_codex_file(&f.path, &cwds),
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
        discover_claude_files(project_root, &mut out);
    }
    if sources.contains(&Source::Codex) {
        discover_codex_files(&mut out);
    }
    out
}

fn claude_config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    home().join(".claude")
}

fn codex_home() -> PathBuf {
    if let Some(dir) = std::env::var_os("CODEX_HOME") {
        return PathBuf::from(dir);
    }
    home().join(".codex")
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// Claude Code's project-dir encoding: `/` → `-` on the absolute path.
fn path_to_claude_slug(p: &Path) -> String {
    p.to_string_lossy().replace('/', "-")
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

fn discover_claude_files(project_root: &Path, out: &mut Vec<FileEntry>) {
    let projects = claude_config_dir().join("projects");
    let mut seen: HashSet<PathBuf> = HashSet::new();
    for root in worktree_roots(project_root) {
        let dir = projects.join(path_to_claude_slug(&root));
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if !seen.insert(path.clone()) {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
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

fn discover_codex_files(out: &mut Vec<FileEntry>) {
    let sessions = codex_home().join("sessions");
    if sessions.exists() {
        walk_codex(&sessions, out);
    }
}

fn walk_codex(dir: &Path, out: &mut Vec<FileEntry>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            walk_codex(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let (mtime_ns, size_bytes) = stat_ns(&meta);
        out.push(FileEntry {
            source: Source::Codex,
            path,
            mtime_ns,
            size_bytes,
        });
    }
}

// -------- Claude Code parsing (per-turn) --------

fn parse_claude_file(path: &Path, expected_cwds: &HashSet<String>) -> Vec<TurnChunk> {
    let Ok(f) = fs::File::open(path) else {
        return Vec::new();
    };
    let reader = BufReader::new(f);
    let mut session_id: Option<String> = None;
    let mut cwd_match: Option<bool> = None;
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
        if cwd_match.is_none() {
            if let Some(cwd) = rec.get("cwd").and_then(|v| v.as_str()) {
                if !cwd.is_empty() {
                    cwd_match = Some(expected_cwds.contains(cwd));
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

    if matches!(cwd_match, Some(false)) {
        return Vec::new();
    }
    let Some(session_id) = session_id else {
        return Vec::new();
    };
    let path_str = path.to_string_lossy();
    pairs
        .into_iter()
        .enumerate()
        .map(|(i, (uid, user, assistant))| {
            let turn = if uid.is_empty() { "LAST" } else { &uid };
            build_chunk(&session_id, turn, &path_str, &user, &assistant, i as i64)
        })
        .collect()
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

fn parse_codex_file(path: &Path, expected_cwds: &HashSet<String>) -> Vec<TurnChunk> {
    let Ok(f) = fs::File::open(path) else {
        return Vec::new();
    };
    let reader = BufReader::new(f);
    let mut session_id: Option<String> = None;
    let mut cwd_match: Option<bool> = None;
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
                if let Some(cwd) = payload.get("cwd").and_then(|v| v.as_str()) {
                    cwd_match = Some(expected_cwds.contains(cwd));
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

    if matches!(cwd_match, Some(false)) {
        return Vec::new();
    }
    let Some(session_id) = session_id else {
        return Vec::new();
    };
    let path_str = path.to_string_lossy();
    pairs
        .into_iter()
        .enumerate()
        .map(|(i, (turn_id, user, assistant))| {
            let turn = if turn_id.is_empty() { "LAST" } else { &turn_id };
            build_chunk(&session_id, turn, &path_str, &user, &assistant, i as i64)
        })
        .collect()
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

    fn cwds(paths: &[&str]) -> HashSet<String> {
        paths.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn slug_encodes_path() {
        assert_eq!(
            path_to_claude_slug(Path::new("/Users/me/Projects/foo")),
            "-Users-me-Projects-foo"
        );
    }

    #[test]
    fn parse_claude_emits_one_chunk_per_pair() {
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
        let chunks = parse_claude_file(&p, &cwds(&["/repo"]));
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].content.contains("turn:u1"));
        assert!(chunks[0].content.contains("hello"));
        assert_eq!(chunks[0].turn_idx, 0);
        assert!(chunks[1].content.contains("turn:u2"));
        assert!(chunks[1].content.contains("again"));
        assert_eq!(chunks[1].turn_idx, 1);
        fs::remove_file(&p).ok();
    }

    #[test]
    fn parse_claude_skips_cwd_mismatch() {
        let p = write_jsonl(
            "claude-mismatch",
            &[
                r#"{"type":"attachment","cwd":"/elsewhere","sessionId":"sess-2"}"#,
                r#"{"type":"user","uuid":"u1","sessionId":"sess-2","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#,
                r#"{"type":"assistant","sessionId":"sess-2","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}"#,
            ],
        );
        assert!(parse_claude_file(&p, &cwds(&["/repo"])).is_empty());
        fs::remove_file(&p).ok();
    }

    #[test]
    fn parse_codex_emits_pairs_with_turn_ids() {
        let p = write_jsonl(
            "codex",
            &[
                r#"{"type":"session_meta","payload":{"id":"sess-c","cwd":"/repo"}}"#,
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t1"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"hello"}}"#,
            ],
        );
        let chunks = parse_codex_file(&p, &cwds(&["/repo"]));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("turn:t1"));
        assert!(chunks[0].content.contains("hello"));
        fs::remove_file(&p).ok();
    }
}

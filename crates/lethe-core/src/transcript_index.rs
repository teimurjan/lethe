//! Transcript discovery + incremental indexing.
//!
//! lethe indexes agent transcripts (Claude Code, Codex, and Oh My Pi JSONL)
//! directly, at turn granularity, into the per-project store.
//!
//! Claude Code writes one JSONL per session under
//! `$CLAUDE_CONFIG_DIR/projects/<slug>/` where `<slug>` is the launch cwd
//! with every `/` replaced by `-`. Because a session launched from a git
//! worktree records the worktree path (a different slug than the unified
//! main worktree), discovery enumerates every worktree of the repo and
//! scans all matching slugs, filtering by the transcript's recorded `cwd`.
//!
//! Codex and Oh My Pi write JSONL under their global session directories
//! without per-project lookup metadata, so those trees are walked and filtered
//! by each transcript's recorded `cwd`.
//!
//! [`ensure_fresh`] is the freshness hook run before a search: it stats
//! every candidate transcript, reparses only those whose (mtime, size)
//! diverge from the manifest, and add-only-syncs their turns.
//!
//! Parsing ([`parse_claude_file`], [`parse_codex_file`], and
//! [`parse_omp_file`]) returns the
//! recorded `cwd` alongside the turn chunks and does **not** filter by
//! cwd — the caller decides. `ensure_fresh` applies the cwd filter; the
//! `maintenance` module reuses the raw parse to scan for stale transcripts.

use std::collections::{HashMap, HashSet};
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
    OhMyPi,
}

#[must_use]
pub fn all_sources() -> Vec<Source> {
    vec![Source::ClaudeCode, Source::Codex, Source::OhMyPi]
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

/// Filename of Claude Code's per-project session index.
const CLAUDE_SESSIONS_INDEX: &str = "sessions-index.json";

/// Parse a transcript file by source into `(cwd, chunks)`. Claude's
/// `sessions-index.json` is handled specially (metadata, not a transcript).
#[must_use]
pub fn parse_file(path: &Path, source: Source) -> ParsedFile {
    if source == Source::ClaudeCode
        && path.file_name().and_then(|n| n.to_str()) == Some(CLAUDE_SESSIONS_INDEX)
    {
        return parse_claude_index(path);
    }
    match source {
        Source::ClaudeCode => parse_claude_file(path),
        Source::Codex => parse_codex_file(path),
        Source::OhMyPi => parse_omp_file(path),
    }
}

/// Parse Claude Code's `sessions-index.json`. This survives after the
/// actual `.jsonl` transcripts are pruned, so for any session whose
/// transcript is gone we recover a lightweight memory from its recorded
/// `firstPrompt` (sessions whose transcript still exists are skipped —
/// the full transcript is indexed instead). `cwd` comes from the entries'
/// `projectPath`, so a folder with only a pruned index still resolves to
/// its (possibly still-present) repo.
fn parse_claude_index(path: &Path) -> ParsedFile {
    let Ok(text) = fs::read_to_string(path) else {
        return ParsedFile::default();
    };
    let Ok(json) = serde_json::from_str::<Value>(&text) else {
        return ParsedFile::default();
    };
    let Some(entries) = json.get("entries").and_then(|v| v.as_array()) else {
        return ParsedFile::default();
    };
    let mut cwd: Option<String> = None;
    let mut chunks: Vec<TurnChunk> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        if cwd.is_none() {
            cwd = e
                .get("projectPath")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .map(str::to_owned);
        }
        // Skip sessions whose transcript still exists — indexed in full
        // elsewhere; don't duplicate as a stub.
        let full = e.get("fullPath").and_then(|v| v.as_str()).unwrap_or("");
        if !full.is_empty() && Path::new(full).exists() {
            continue;
        }
        let first = e
            .get("firstPrompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if first.is_empty() {
            continue;
        }
        let session = e.get("sessionId").and_then(|v| v.as_str()).unwrap_or("");
        let count = e
            .get("messageCount")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(0);
        let assistant = format!("(transcript pruned — {count} messages)");
        chunks.push(build_chunk(
            session, "index", full, first, &assistant, i as i64,
        ));
    }
    ParsedFile { cwd, chunks }
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
            Source::ClaudeCode | Source::OhMyPi => &rec,
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
    if sources.contains(&Source::OhMyPi) {
        for path in omp_session_files() {
            if let Ok(meta) = fs::metadata(&path) {
                let (mtime_ns, size_bytes) = stat_ns(&meta);
                out.push(FileEntry {
                    source: Source::OhMyPi,
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

/// Oh My Pi's agent data directory. A named `OMP_PROFILE` / `PI_PROFILE`
/// selects its profile directory; otherwise `PI_CODING_AGENT_DIR` overrides
/// the default below `PI_CONFIG_DIR` (normally `~/.omp/agent`).
#[must_use]
pub fn omp_agent_dir() -> PathBuf {
    let config = std::env::var_os("PI_CONFIG_DIR").unwrap_or_else(|| ".omp".into());
    let root = home().join(config);
    if let Some(profile) = omp_profile() {
        return root.join("profiles").join(profile).join("agent");
    }
    std::env::var_os("PI_CODING_AGENT_DIR")
        .map_or_else(|| root.join("agent"), PathBuf::from)
}

fn omp_profile() -> Option<String> {
    std::env::var("OMP_PROFILE")
        .ok()
        .or_else(|| std::env::var("PI_PROFILE").ok())
        .map(|p| p.trim().to_owned())
        .filter(|p| !p.is_empty() && p != "default")
}

fn omp_sessions_dir() -> PathBuf {
    let profile = omp_profile();
    if profile.is_none() {
        if let Some(dir) = std::env::var_os("PI_CODING_AGENT_DIR") {
            return PathBuf::from(dir).join("sessions");
        }
    }
    if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
        let mut root = PathBuf::from(xdg).join("omp");
        if let Some(profile) = profile {
            root = root.join("profiles").join(profile);
        }
        if root.is_dir() {
            return root.join("sessions");
        }
    }
    omp_agent_dir().join("sessions")
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

/// Every `*.jsonl` under Oh My Pi's agent session directory (recursively).
#[must_use]
pub fn omp_session_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let sessions = omp_sessions_dir();
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
        let dir = projects.join(path_to_claude_slug(&root));
        for path in jsonl_recursive(&dir) {
            if seen.insert(path.clone()) {
                out.push(path);
            }
        }
        // The session index lets us recover pruned sessions' first prompts.
        let idx = dir.join(CLAUDE_SESSIONS_INDEX);
        if idx.is_file() && seen.insert(idx.clone()) {
            out.push(idx);
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

// -------- Oh My Pi parsing (per-turn) --------

fn parse_omp_file(path: &Path) -> ParsedFile {
    let Ok(f) = fs::File::open(path) else {
        return ParsedFile::default();
    };
    let reader = BufReader::new(f);
    let mut session_id: Option<String> = None;
    let mut cwd: Option<String> = None;
    let mut entry_turn: HashMap<String, usize> = HashMap::new();
    let mut turns: Vec<(String, String, String)> = Vec::new(); // (turn_id, user, assistant)

    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        let kind = rec.get("type").and_then(|v| v.as_str());
        if kind == Some("session") {
            if let Some(id) = rec.get("id").and_then(|v| v.as_str()) {
                session_id = Some(id.to_owned());
            }
            if let Some(value) = rec.get("cwd").and_then(|v| v.as_str()) {
                if !value.is_empty() {
                    cwd = Some(value.to_owned());
                }
            }
            continue;
        }

        let entry_id = rec.get("id").and_then(|v| v.as_str());
        let parent_turn = rec
            .get("parentId")
            .and_then(|v| v.as_str())
            .and_then(|id| entry_turn.get(id).copied());
        if kind != Some("message") {
            if let (Some(id), Some(turn)) = (entry_id, parent_turn) {
                entry_turn.insert(id.to_owned(), turn);
            }
            continue;
        }

        let msg = rec.get("message").unwrap_or(&rec);
        match msg.get("role").and_then(|v| v.as_str()) {
            Some("user") => {
                let text = omp_message_text(msg);
                if text.is_empty() {
                    continue;
                }
                let Some(id) = entry_id else { continue };
                let turn = turns.len();
                turns.push((id.to_owned(), text, String::new()));
                entry_turn.insert(id.to_owned(), turn);
            }
            Some("assistant") => {
                let Some(turn) = parent_turn else { continue };
                let text = omp_message_text(msg);
                if !text.is_empty() {
                    let assistant = &mut turns[turn].2;
                    if !assistant.is_empty() {
                        assistant.push('\n');
                    }
                    assistant.push_str(&text);
                }
                if let Some(id) = entry_id {
                    entry_turn.insert(id.to_owned(), turn);
                }
            }
            _ => {
                if let (Some(id), Some(turn)) = (entry_id, parent_turn) {
                    entry_turn.insert(id.to_owned(), turn);
                }
            }
        }
    }

    let Some(session_id) = session_id else {
        return ParsedFile {
            cwd,
            chunks: Vec::new(),
        };
    };
    let path_str = path.to_string_lossy();
    let chunks = turns
        .into_iter()
        .filter(|(_, _, assistant)| !assistant.is_empty())
        .enumerate()
        .map(|(i, (turn_id, user, assistant))| {
            build_chunk(
                &session_id,
                &turn_id,
                &path_str,
                &user,
                &assistant,
                i as i64,
            )
        })
        .collect();
    ParsedFile { cwd, chunks }
}

fn omp_message_text(msg: &Value) -> String {
    if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
        return text.to_owned();
    }
    let Some(blocks) = msg.get("content").and_then(|v| v.as_array()) else {
        return String::new();
    };
    let mut parts = Vec::new();
    for block in blocks {
        match block.get("type").and_then(|v| v.as_str()) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    parts.push(text.to_owned());
                }
            }
            Some("toolCall") => {
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                parts.push(format!("[tool_use: {name}]"));
            }
            _ => {}
        }
    }
    parts.join("\n")
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
    fn parse_claude_index_recovers_pruned_sessions() {
        // fullPath points at a jsonl that doesn't exist → recover the
        // firstPrompt as a memory; cwd comes from projectPath.
        let p = write_jsonl(
            "index",
            &[r#"{"version":1,"entries":[
                {"sessionId":"s1","fullPath":"/gone/s1.jsonl","firstPrompt":"fix the anonymization worker","messageCount":15,"projectPath":"/repo"},
                {"sessionId":"s2","fullPath":"/gone/s2.jsonl","firstPrompt":"add dark mode","messageCount":3,"projectPath":"/repo"}
            ]}"#],
        );
        // Rename to the exact filename parse_file dispatches on.
        let idx = p.with_file_name("sessions-index.json");
        fs::rename(&p, &idx).unwrap();

        let parsed = parse_file(&idx, Source::ClaudeCode);
        assert_eq!(parsed.cwd.as_deref(), Some("/repo"));
        assert_eq!(parsed.chunks.len(), 2);
        assert!(parsed.chunks[0].content.contains("anonymization worker"));
        assert!(parsed.chunks[0].content.contains("session:s1"));
        fs::remove_file(&idx).ok();
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

    #[test]
    fn parse_omp_emits_tree_aware_pairs_with_cwd() {
        let p = write_jsonl(
            "omp",
            &[
                r#"{"type":"title","title":"ignored preamble"}"#,
                r#"{"type":"session","version":3,"id":"sess-o","cwd":"/repo"}"#,
                r#"{"type":"model_change","id":"m1","parentId":null,"model":"openai/gpt"}"#,
                r#"{"type":"message","id":"u1","parentId":"m1","message":{"role":"user","content":[{"type":"text","text":"first"}]}}"#,
                r#"{"type":"message","id":"a1","parentId":"u1","message":{"role":"assistant","content":[{"type":"thinking","thinking":"hidden"},{"type":"toolCall","name":"read"}]}}"#,
                r#"{"type":"message","id":"r1","parentId":"a1","message":{"role":"toolResult","content":[{"type":"text","text":"tool output"}]}}"#,
                r#"{"type":"message","id":"a2","parentId":"r1","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}"#,
                r#"{"type":"message","id":"u2","parentId":"a2","message":{"role":"user","content":[{"type":"text","text":"second"}]}}"#,
                r#"{"type":"message","id":"a3","parentId":"u2","message":{"role":"assistant","content":[{"type":"text","text":"second answer"}]}}"#,
                r#"{"type":"message","id":"u3","parentId":"a2","message":{"role":"user","content":[{"type":"text","text":"alternate"}]}}"#,
                r#"{"type":"message","id":"a4","parentId":"u3","message":{"role":"assistant","content":[{"type":"text","text":"alternate answer"}]}}"#,
            ],
        );
        let parsed = parse_omp_file(&p);
        assert_eq!(parsed.cwd.as_deref(), Some("/repo"));
        assert_eq!(parsed.chunks.len(), 3);
        assert!(parsed.chunks[0].content.contains("turn:u1"));
        assert!(parsed.chunks[0].content.contains("[tool_use: read]\ndone"));
        assert!(parsed.chunks[1].content.contains("second answer"));
        assert!(parsed.chunks[2].content.contains("alternate answer"));
        fs::remove_file(&p).ok();
    }
}

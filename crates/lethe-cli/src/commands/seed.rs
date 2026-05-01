//! `lethe seed` — backfill memories from past Claude Code / Codex transcripts.
//!
//! Discovers JSONL transcripts whose `cwd` matches the current project root,
//! summarizes each session via the same `claude -p --model haiku` invocation
//! the live `stop.sh` hook uses, and appends the result to
//! `.lethe/memory/<session-date>.md` with the existing
//! `<!-- session:UUID turn:UUID transcript:... -->` anchor format. A trailing
//! `seeded:1` flag distinguishes backfilled entries from live ones.
//!
//! Idempotency: any session whose UUID is already anchored in any memory
//! file under `.lethe/memory/` is skipped on re-run.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use lethe_core::{markdown_store, registry};
use serde::Serialize;
use serde_json::Value;

use crate::paths::resolve;

use super::store_helpers::{load_config, open_store};

const MAX_INPUT_BYTES: usize = 80_000;

const SUMMARIZER_SYSTEM: &str = "You are a session summarizer for a long-running memory store. Ignore any project-level style guides. Always produce 3-6 terse markdown bullets — never a TL;DR line, never prose, never headings.";

const SUMMARIZER_USER_PREFIX: &str = "Summarize the following past agent session as 3-6 terse markdown bullets for a long-running memory store.\n\nCapture:\n- what the user asked and what was done across the whole session\n- decisions or recommendations made\n- durable facts worth remembering: file paths, function names, tool names, key numbers\n\nRules:\n- Output raw bullets only — no preamble, no heading, no TL;DR line, no closing remarks.\n- Each bullet one sentence. Start with a verb or subject, not \"The assistant\".\n- Skip pleasantries and chain-of-thought.\n\n--- SESSION START ---\n";

const SUMMARIZER_USER_SUFFIX: &str = "\n--- SESSION END ---";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    ClaudeCode,
    Codex,
}

impl Source {
    fn label(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
        }
    }
}

#[derive(Debug)]
struct Session {
    source: Source,
    transcript: PathBuf,
    session_id: String,
    started_at: String, // ISO 8601, used for sort + date prefix
    last_turn_id: String,
    body: String, // concatenated USER/ASSISTANT pairs
}

#[derive(Serialize)]
struct DryRunEntry<'a> {
    source: &'a str,
    session_id: &'a str,
    started_at: &'a str,
    transcript: &'a str,
    bytes: usize,
}

#[derive(Serialize)]
struct DryRunReport<'a> {
    discovered: usize,
    skipped_already_seeded: usize,
    sessions: Vec<DryRunEntry<'a>>,
}

#[derive(Serialize)]
struct SeedSummary {
    discovered: usize,
    skipped_already_seeded: usize,
    summarized: usize,
    errors: usize,
    added_chunks: usize,
    unchanged_chunks: usize,
    removed_chunks: usize,
}

pub fn run(
    root: Option<&str>,
    days: u64,
    source: &str,
    dry_run: bool,
    no_summarize: bool,
    max_sessions: Option<usize>,
    json_output: bool,
) -> Result<i32> {
    let paths = resolve(root);
    fs::create_dir_all(paths.memory())?;
    fs::create_dir_all(paths.index())?;

    let want = parse_source(source)?;
    let project_root = paths
        .root
        .canonicalize()
        .unwrap_or_else(|_| paths.root.clone());
    let cutoff = SystemTime::now()
        .checked_sub(Duration::from_secs(days.saturating_mul(86_400)))
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let mut discovered: Vec<Session> = Vec::new();
    if want.contains(&Source::ClaudeCode) {
        discovered.extend(discover_claude_code(&project_root, cutoff));
    }
    if want.contains(&Source::Codex) {
        discovered.extend(discover_codex(&project_root, cutoff));
    }
    discovered.sort_by(|a, b| a.started_at.cmp(&b.started_at));

    let already = load_seeded_session_ids(&paths.memory());
    let total_discovered = discovered.len();
    discovered.retain(|s| !already.contains(&s.session_id));
    let skipped_already_seeded = total_discovered - discovered.len();

    if let Some(cap) = max_sessions {
        discovered.truncate(cap);
    }

    if dry_run {
        emit_dry_run(total_discovered, skipped_already_seeded, &discovered, json_output)?;
        return Ok(0);
    }

    let summarizers = if no_summarize {
        Summarizers::none()
    } else {
        Summarizers::detect()
    };
    if !no_summarize && !summarizers.any() {
        eprintln!(
            "warning: neither `claude` nor `codex` CLI on PATH — falling back to --no-summarize (raw last-prompt snippets)"
        );
    }

    let outcome = summarize_all(&discovered, summarizers, json_output);
    write_seeded_entries(&paths.memory(), outcome.by_date)?;
    let counts = reindex(&paths)?;
    let summarized = outcome.summarized;
    let errors = outcome.errors;

    let payload = SeedSummary {
        discovered: total_discovered,
        skipped_already_seeded,
        summarized,
        errors,
        added_chunks: counts.added,
        unchanged_chunks: counts.unchanged,
        removed_chunks: counts.removed,
    };
    if json_output {
        println!("{}", serde_json::to_string(&payload)?);
    } else {
        println!(
            "seeded: discovered={} skipped={} summarized={} errors={} added_chunks={}",
            total_discovered, skipped_already_seeded, summarized, errors, counts.added
        );
    }
    Ok(0)
}

struct SummarizeOutcome {
    summarized: usize,
    errors: usize,
    by_date: BTreeMap<String, Vec<(Session, String)>>,
}

#[derive(Clone, Copy)]
struct Summarizers {
    claude: bool,
    codex: bool,
}

impl Summarizers {
    fn detect() -> Self {
        Self {
            claude: which("claude").is_some(),
            codex: which("codex").is_some(),
        }
    }
    fn none() -> Self {
        Self {
            claude: false,
            codex: false,
        }
    }
    fn any(self) -> bool {
        self.claude || self.codex
    }
    /// Pick the summarizer for a source: prefer the agent's native CLI, fall
    /// back to the other if missing. Returns `None` only when neither is on
    /// PATH (caller writes the raw fallback snippet instead).
    fn pick(self, source: Source) -> Option<fn(&str) -> Result<String>> {
        match source {
            Source::ClaudeCode => {
                if self.claude {
                    Some(summarize_with_claude)
                } else if self.codex {
                    Some(summarize_with_codex)
                } else {
                    None
                }
            }
            Source::Codex => {
                if self.codex {
                    Some(summarize_with_codex)
                } else if self.claude {
                    Some(summarize_with_claude)
                } else {
                    None
                }
            }
        }
    }
}

fn summarize_all(
    discovered: &[Session],
    summarizers: Summarizers,
    json_output: bool,
) -> SummarizeOutcome {
    let mut summarized = 0usize;
    let mut errors = 0usize;
    let mut by_date: BTreeMap<String, Vec<(Session, String)>> = BTreeMap::new();

    for (i, session) in discovered.iter().enumerate() {
        let label = format!(
            "[{}/{}] {} {}",
            i + 1,
            discovered.len(),
            session.source.label(),
            short_id(&session.session_id),
        );

        let summary_result = match summarizers.pick(session.source) {
            Some(f) => f(&session.body),
            None => Ok(fallback_snippet(&session.body)),
        };

        match summary_result {
            Ok(s) if !s.trim().is_empty() => {
                let date = date_prefix(&session.started_at);
                by_date
                    .entry(date)
                    .or_default()
                    .push((clone_session(session), s));
                summarized += 1;
                if !json_output {
                    eprintln!("{label} ok");
                }
            }
            Ok(_) => {
                errors += 1;
                if !json_output {
                    eprintln!("{label} skip (empty summary)");
                }
            }
            Err(e) => {
                errors += 1;
                if !json_output {
                    eprintln!("{label} fail: {e}");
                }
            }
        }
    }
    SummarizeOutcome {
        summarized,
        errors,
        by_date,
    }
}

fn write_seeded_entries(
    memory_dir: &Path,
    by_date: BTreeMap<String, Vec<(Session, String)>>,
) -> Result<()> {
    for (date, mut entries) in by_date {
        entries.sort_by(|a, b| a.0.started_at.cmp(&b.0.started_at));
        let file = memory_dir.join(format!("{date}.md"));
        ensure_date_header(&file, &date)?;
        for (session, summary) in entries {
            append_seeded_entry(&file, &session, &summary)?;
        }
    }
    Ok(())
}

fn reindex(paths: &crate::paths::Paths) -> Result<markdown_store::ReindexCounts> {
    let cfg = load_config(&paths.config_path())?;
    let store = open_store(&paths.index(), &cfg, true)?;
    let counts = markdown_store::reindex(&paths.memory(), &store)?;
    store.save()?;
    drop(store);

    if !registry::is_disabled() {
        let _ = registry::register(&paths.root);
    }
    Ok(counts)
}

fn parse_source(s: &str) -> Result<Vec<Source>> {
    Ok(match s {
        "all" => vec![Source::ClaudeCode, Source::Codex],
        "claude-code" => vec![Source::ClaudeCode],
        "codex" => vec![Source::Codex],
        _ => anyhow::bail!("unknown --source: {s}"),
    })
}

fn date_prefix(iso: &str) -> String {
    if iso.len() >= 10 {
        iso[..10].to_owned()
    } else {
        // Final fallback so we never panic; date-less sessions land in today's file.
        let now = SystemTime::now();
        let secs = now
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Crude UTC date — only hit when transcript lacks any timestamp.
        let days = secs / 86_400;
        let (y, m, d) = days_to_ymd(days as i64);
        format!("{y:04}-{m:02}-{d:02}")
    }
}

fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Civil-from-days algorithm (Howard Hinnant). Days since 1970-01-01.
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let doe = (days - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + era as i32 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn short_id(s: &str) -> String {
    let n = s.len().min(8);
    s[..n].to_owned()
}

fn clone_session(s: &Session) -> Session {
    Session {
        source: s.source,
        transcript: s.transcript.clone(),
        session_id: s.session_id.clone(),
        started_at: s.started_at.clone(),
        last_turn_id: s.last_turn_id.clone(),
        body: String::new(), // body no longer needed once summary is computed
    }
}

fn ensure_date_header(file: &Path, date: &str) -> Result<()> {
    if file.exists() {
        return Ok(());
    }
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(file)?;
    writeln!(f, "# {date}")?;
    writeln!(f)?;
    Ok(())
}

fn append_seeded_entry(file: &Path, session: &Session, summary: &str) -> Result<()> {
    let mut f = fs::OpenOptions::new().create(true).append(true).open(file)?;
    let time = if session.started_at.len() >= 16 {
        &session.started_at[11..16]
    } else {
        "00:00"
    };
    writeln!(f)?;
    writeln!(f, "### {time}")?;
    writeln!(
        f,
        "<!-- session:{} turn:{} transcript:{} seeded:1 -->",
        session.session_id,
        if session.last_turn_id.is_empty() {
            "LAST"
        } else {
            &session.last_turn_id
        },
        session.transcript.display(),
    )?;
    let trimmed = summary.trim_end();
    writeln!(f, "{trimmed}")?;
    Ok(())
}

fn emit_dry_run(
    discovered: usize,
    skipped: usize,
    sessions: &[Session],
    json_output: bool,
) -> Result<()> {
    if json_output {
        let entries: Vec<DryRunEntry<'_>> = sessions
            .iter()
            .map(|s| DryRunEntry {
                source: s.source.label(),
                session_id: &s.session_id,
                started_at: &s.started_at,
                transcript: s.transcript.to_str().unwrap_or(""),
                bytes: s.body.len(),
            })
            .collect();
        let report = DryRunReport {
            discovered,
            skipped_already_seeded: skipped,
            sessions: entries,
        };
        println!("{}", serde_json::to_string(&report)?);
        return Ok(());
    }
    println!(
        "would seed {} session(s) (discovered={}, skipped_already_seeded={}):",
        sessions.len(),
        discovered,
        skipped
    );
    for s in sessions {
        println!(
            "  {:<11} {} {} ({} bytes) {}",
            s.source.label(),
            short_id(&s.session_id),
            &s.started_at,
            s.body.len(),
            s.transcript.display(),
        );
    }
    Ok(())
}

fn which(cmd: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(cmd);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

fn build_summarizer_input(body: &str, include_system_prompt: bool) -> String {
    let mut input = String::with_capacity(body.len() + 512);
    if include_system_prompt {
        // Codex has no `--append-system-prompt`; bundle the rules into the
        // user message so the agent treats them as instructions.
        input.push_str(SUMMARIZER_SYSTEM);
        input.push_str("\n\n");
    }
    input.push_str(SUMMARIZER_USER_PREFIX);
    if body.len() > MAX_INPUT_BYTES {
        // Truncate from the front: the tail of a session is usually most
        // relevant (final decisions, last assistant output).
        let cut = body.len() - MAX_INPUT_BYTES;
        input.push_str("[earlier turns truncated]\n");
        input.push_str(safe_slice(body, cut));
    } else {
        input.push_str(body);
    }
    input.push_str(SUMMARIZER_USER_SUFFIX);
    input
}

fn summarize_with_claude(body: &str) -> Result<String> {
    let input = build_summarizer_input(body, false);

    // Spawn the summarizer outside any git repo so the lethe stop-hook
    // running inside the spawned `claude -p` session resolves
    // LETHE_GIT_ROOT to a temp dir (per common.sh) and any incidental
    // memory writes land there instead of in the user's real project.
    let mut child = Command::new("claude")
        .current_dir(std::env::temp_dir())
        .args([
            "-p",
            "--model",
            "haiku",
            "--append-system-prompt",
            SUMMARIZER_SYSTEM,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("spawn claude")?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(input.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        anyhow::bail!("claude exited with {}", output.status);
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn summarize_with_codex(body: &str) -> Result<String> {
    let input = build_summarizer_input(body, true);

    // Codex exec has no equivalent of --output to pure stdout, so
    // capture the final assistant message via a temp file. PID + nanos
    // keep concurrent invocations from clobbering each other.
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let out_path = std::env::temp_dir().join(format!(
        "lethe-seed-codex-{}-{nanos}.txt",
        std::process::id()
    ));
    let _ = fs::remove_file(&out_path);

    // Spawn outside any git repo (same reason as summarize_with_claude:
    // routes any nested lethe stop-hook writes to $TMPDIR/.lethe).
    let mut child = Command::new("codex")
        .current_dir(std::env::temp_dir())
        .args([
            "exec",
            "--skip-git-repo-check",
            "--output-last-message",
            out_path.to_str().unwrap_or("/tmp/lethe-seed-codex.txt"),
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("spawn codex")?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(input.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        let _ = fs::remove_file(&out_path);
        anyhow::bail!("codex exec exited with {status}");
    }
    let summary = fs::read_to_string(&out_path).unwrap_or_default();
    let _ = fs::remove_file(&out_path);
    Ok(summary)
}

fn fallback_snippet(body: &str) -> String {
    // Find last USER block and the assistant tail so the memory still
    // carries something searchable.
    let last_user = body.rfind("USER:\n").map_or("", |i| &body[i + 6..]);
    let user_first_line = last_user.lines().next().unwrap_or("").trim();
    let assistant_tail = body.rfind("ASSISTANT:\n").map_or("", |i| &body[i + 11..]);
    let mut tail = assistant_tail.trim().to_owned();
    if tail.len() > 200 {
        tail.truncate(safe_truncate(&tail, 200));
        tail.push('…');
    }
    let mut out = String::new();
    if !user_first_line.is_empty() {
        out.push_str("- last user prompt: ");
        out.push_str(user_first_line);
        out.push('\n');
    }
    if !tail.is_empty() {
        out.push_str("- last assistant reply: ");
        out.push_str(&tail.replace('\n', " "));
        out.push('\n');
    }
    if out.is_empty() {
        out.push_str("- (no usable content extracted)\n");
    }
    out
}

fn safe_slice(s: &str, mut byte_start: usize) -> &str {
    while byte_start < s.len() && !s.is_char_boundary(byte_start) {
        byte_start += 1;
    }
    &s[byte_start..]
}

fn safe_truncate(s: &str, mut byte_end: usize) -> usize {
    if byte_end >= s.len() {
        return s.len();
    }
    while byte_end > 0 && !s.is_char_boundary(byte_end) {
        byte_end -= 1;
    }
    byte_end
}

fn load_seeded_session_ids(memory_dir: &Path) -> HashSet<String> {
    let mut out = HashSet::new();
    let Ok(entries) = fs::read_dir(memory_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Ok(file) = fs::File::open(&path) else {
            continue;
        };
        for line in BufReader::new(file).lines().map_while(Result::ok) {
            let trimmed = line.trim();
            // Match `<!-- session:UUID ` (live and seeded both share this prefix).
            if let Some(rest) = trimmed.strip_prefix("<!-- session:") {
                if let Some(id) = rest.split_whitespace().next() {
                    if !id.is_empty() {
                        out.insert(id.to_owned());
                    }
                }
            }
        }
    }
    out
}

fn discover_claude_code(project_root: &Path, cutoff: SystemTime) -> Vec<Session> {
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let slug = path_to_claude_slug(project_root);
    let dir = PathBuf::from(home).join(".claude").join("projects").join(&slug);
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let canonical_root = project_root.to_string_lossy().to_string();
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let mtime = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        if mtime < cutoff {
            continue;
        }
        match parse_claude_session(&path, &canonical_root) {
            Ok(Some(s)) => out.push(s),
            Ok(None) => {}
            Err(e) => eprintln!(
                "warning: failed to read {}: {e}",
                path.display()
            ),
        }
    }
    out
}

fn path_to_claude_slug(p: &Path) -> String {
    // Claude Code's project-dir encoding: each `/` becomes `-`, with a
    // leading `-` (so absolute paths starting with `/` produce a slug
    // starting with `-`). Other characters are preserved as-is.
    p.to_string_lossy().replace('/', "-")
}

fn parse_claude_session(path: &Path, expected_cwd: &str) -> Result<Option<Session>> {
    let f = fs::File::open(path)?;
    let reader = BufReader::new(f);
    let mut session_id: Option<String> = None;
    let mut started_at: Option<String> = None;
    let mut cwd_match: Option<bool> = None;
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut last_turn_id = String::new();
    let mut pending_user: Option<(String, String)> = None; // (uuid, text)
    let mut latest_ts: Option<String> = None;

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
        if let Some(ts) = rec.get("timestamp").and_then(|v| v.as_str()) {
            if started_at.is_none() {
                started_at = Some(ts.to_owned());
            }
            latest_ts = Some(ts.to_owned());
        }
        // cwd may live on `attachment` records or on user/assistant lines —
        // grab the first non-empty one we see.
        if cwd_match.is_none() {
            if let Some(cwd) = rec.get("cwd").and_then(|v| v.as_str()) {
                if !cwd.is_empty() {
                    cwd_match = Some(cwd == expected_cwd);
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
                    if !uid.is_empty() {
                        last_turn_id = uid;
                    }
                    pairs.push((user_text, text));
                } else if let Some(prev) = pairs.last_mut() {
                    // Multiple assistant continuations under the same user turn —
                    // append to the previous assistant block instead of dropping.
                    prev.1.push('\n');
                    prev.1.push_str(&text);
                }
            }
            _ => {}
        }
    }
    let Some(session_id) = session_id else {
        return Ok(None);
    };
    if pairs.is_empty() {
        return Ok(None);
    }
    if matches!(cwd_match, Some(false)) {
        // Slug matched but cwd disagrees → skip rather than misattribute.
        return Ok(None);
    }
    let started_at = started_at
        .or(latest_ts)
        .unwrap_or_else(|| "1970-01-01T00:00:00.000Z".to_owned());
    let body = render_pairs(&pairs);
    Ok(Some(Session {
        source: Source::ClaudeCode,
        transcript: path.to_path_buf(),
        session_id,
        started_at,
        last_turn_id,
        body,
    }))
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

fn discover_codex(project_root: &Path, cutoff: SystemTime) -> Vec<Session> {
    let Some(home) = std::env::var_os("HOME") else {
        return Vec::new();
    };
    let sessions_root = PathBuf::from(home).join(".codex").join("sessions");
    if !sessions_root.exists() {
        return Vec::new();
    }
    let canonical_root = project_root.to_string_lossy().to_string();
    let mut out = Vec::new();
    walk_codex_dir(&sessions_root, cutoff, &canonical_root, &mut out);
    out
}

fn walk_codex_dir(
    dir: &Path,
    cutoff: SystemTime,
    expected_cwd: &str,
    out: &mut Vec<Session>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            walk_codex_dir(&path, cutoff, expected_cwd, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if mtime < cutoff {
            continue;
        }
        match parse_codex_session(&path, expected_cwd) {
            Ok(Some(s)) => out.push(s),
            Ok(None) => {}
            Err(e) => eprintln!("warning: failed to read {}: {e}", path.display()),
        }
    }
}

fn parse_codex_session(path: &Path, expected_cwd: &str) -> Result<Option<Session>> {
    let f = fs::File::open(path)?;
    let reader = BufReader::new(f);
    let mut session_id: Option<String> = None;
    let mut started_at: Option<String> = None;
    let mut cwd_match: Option<bool> = None;
    let mut pending_user: Option<String> = None;
    let mut pending_turn: Option<String> = None;
    let mut last_turn_id = String::new();
    let mut pairs: Vec<(String, String)> = Vec::new();

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
                if let Some(ts) = payload
                    .get("timestamp")
                    .and_then(|v| v.as_str())
                    .or_else(|| rec.get("timestamp").and_then(|v| v.as_str()))
                {
                    started_at = Some(ts.to_owned());
                }
                if let Some(cwd) = payload.get("cwd").and_then(|v| v.as_str()) {
                    cwd_match = Some(cwd == expected_cwd);
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
                            if let Some(t) = pending_turn.take() {
                                last_turn_id = t;
                            }
                            pairs.push((u, text));
                        } else if let Some(prev) = pairs.last_mut() {
                            prev.1.push('\n');
                            prev.1.push_str(&text);
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    let Some(session_id) = session_id else {
        return Ok(None);
    };
    if matches!(cwd_match, Some(false)) {
        return Ok(None);
    }
    if pairs.is_empty() {
        return Ok(None);
    }
    let started_at = started_at.unwrap_or_else(|| "1970-01-01T00:00:00.000Z".to_owned());
    let body = render_pairs(&pairs);
    Ok(Some(Session {
        source: Source::Codex,
        transcript: path.to_path_buf(),
        session_id,
        started_at,
        last_turn_id,
        body,
    }))
}

fn render_pairs(pairs: &[(String, String)]) -> String {
    let mut out = String::new();
    for (i, (u, a)) in pairs.iter().enumerate() {
        if i > 0 {
            out.push_str("\n---\n");
        }
        out.push_str("USER:\n");
        out.push_str(u.trim());
        out.push_str("\n\nASSISTANT:\n");
        out.push_str(a.trim());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_jsonl(name: &str, lines: &[&str]) -> PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("lethe-seed-test-{}-{name}.jsonl", std::process::id()));
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
    fn dates_round_trip() {
        // Sanity: 2024-01-01 corresponds to days_since_epoch 19_723.
        assert_eq!(days_to_ymd(19_723), (2024, 1, 1));
        assert_eq!(date_prefix("2026-04-29T10:11:12.000Z"), "2026-04-29");
    }

    #[test]
    fn parse_claude_collects_pairs() {
        let p = write_jsonl(
            "claude",
            &[
                r#"{"type":"attachment","cwd":"/repo","timestamp":"2026-04-29T10:00:00.000Z","sessionId":"sess-1"}"#,
                r#"{"type":"user","uuid":"u1","sessionId":"sess-1","timestamp":"2026-04-29T10:00:01.000Z","message":{"role":"user","content":[{"type":"text","text":"hello"}]}}"#,
                r#"{"type":"assistant","sessionId":"sess-1","timestamp":"2026-04-29T10:00:02.000Z","message":{"role":"assistant","content":[{"type":"text","text":"hi back"}]}}"#,
                r#"{"type":"user","uuid":"u2","sessionId":"sess-1","timestamp":"2026-04-29T10:00:03.000Z","message":{"role":"user","content":[{"type":"text","text":"again"}]}}"#,
                r#"{"type":"assistant","sessionId":"sess-1","timestamp":"2026-04-29T10:00:04.000Z","message":{"role":"assistant","content":[{"type":"text","text":"sure"}]}}"#,
            ],
        );
        let s = parse_claude_session(&p, "/repo").unwrap().unwrap();
        assert_eq!(s.session_id, "sess-1");
        assert_eq!(s.last_turn_id, "u2");
        assert!(s.body.contains("hello"));
        assert!(s.body.contains("again"));
        assert!(s.body.contains("sure"));
        fs::remove_file(&p).ok();
    }

    #[test]
    fn parse_claude_skips_when_cwd_mismatch() {
        let p = write_jsonl(
            "claude-mismatch",
            &[
                r#"{"type":"attachment","cwd":"/somewhere/else","timestamp":"2026-04-29T10:00:00.000Z","sessionId":"sess-2"}"#,
                r#"{"type":"user","uuid":"u1","sessionId":"sess-2","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#,
                r#"{"type":"assistant","sessionId":"sess-2","message":{"role":"assistant","content":[{"type":"text","text":"hi"}]}}"#,
            ],
        );
        assert!(parse_claude_session(&p, "/repo").unwrap().is_none());
        fs::remove_file(&p).ok();
    }

    #[test]
    fn parse_codex_collects_pairs() {
        let p = write_jsonl(
            "codex",
            &[
                r#"{"type":"session_meta","payload":{"id":"sess-c","timestamp":"2026-04-29T10:00:00.000Z","cwd":"/repo"}}"#,
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t1"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"hello"}}"#,
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"again"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t2"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"second reply"}}"#,
            ],
        );
        let s = parse_codex_session(&p, "/repo").unwrap().unwrap();
        assert_eq!(s.session_id, "sess-c");
        assert_eq!(s.last_turn_id, "t2");
        assert!(s.body.contains("again"));
        assert!(s.body.contains("second reply"));
        fs::remove_file(&p).ok();
    }

    #[test]
    fn idempotency_anchor_scan() {
        let dir = std::env::temp_dir().join(format!(
            "lethe-seed-idem-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let f = dir.join("2026-04-29.md");
        fs::write(
            &f,
            "# 2026-04-29\n\n### 10:00\n<!-- session:abc-123 turn:t1 transcript:/x seeded:1 -->\n- bullet\n",
        )
        .unwrap();
        let ids = load_seeded_session_ids(&dir);
        assert!(ids.contains("abc-123"));
        fs::remove_dir_all(&dir).ok();
    }
}

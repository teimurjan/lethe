//! `lethe-codex transcript <path>` — extract a user/assistant turn from a
//! Codex CLI rollout JSONL.
//!
//! Codex rollouts (`$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl`) are
//! tagged-union JSONL where each line is a `RolloutItem`:
//!
//! ```text
//! {"type":"session_meta","payload":{"id":"<thread-id>", ...}}
//! {"type":"event_msg","payload":{"type":"user_message","message":"<text>"}}
//! {"type":"event_msg","payload":{"type":"turn_started","turn_id":"<uuid>", ...}}
//! {"type":"event_msg","payload":{"type":"agent_message","message":"<text>"}}
//! ```
//!
//! Output format mirrors `lethe-claude-code` byte-for-byte so the plugin's
//! `parse-transcript.sh` works unchanged across agents:
//!
//! Default (no `--turn`):
//!     SESSION_ID: <id>
//!     TURN_ID: <id>
//!     ---
//!     USER:
//!     <text>
//!     ---
//!     ASSISTANT:
//!     <text>
//!
//! With `--turn <uuid>`:
//!     USER:
//!     <text>
//!
//!     ASSISTANT:
//!     <text>

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

pub fn run(path: &str, turn: Option<&str>) -> Result<i32> {
    let p = Path::new(path);
    if !p.exists() {
        eprintln!("transcript not found: {path}");
        return Ok(1);
    }
    if let Some(turn_id) = turn {
        let pair = pair_for_turn(p, turn_id).with_context(|| format!("read {path}"))?;
        if pair.user.is_none() && pair.assistant.is_none() {
            eprintln!("(no matching turn)");
            return Ok(1);
        }
        println!("USER:");
        println!("{}", pair.user.unwrap_or_default().trim());
        println!();
        println!("ASSISTANT:");
        println!("{}", pair.assistant.unwrap_or_default().trim());
        return Ok(0);
    }

    let parsed = parse_last(p).with_context(|| format!("read {path}"))?;
    if parsed.user.is_none() && parsed.assistant.is_none() {
        eprintln!("(no matching turn)");
        return Ok(1);
    }
    println!("SESSION_ID: {}", parsed.session_id.unwrap_or_default());
    println!("TURN_ID: {}", parsed.turn_id.unwrap_or_default());
    println!("---");
    println!("USER:");
    println!("{}", parsed.user.unwrap_or_default().trim());
    println!("---");
    println!("ASSISTANT:");
    println!("{}", parsed.assistant.unwrap_or_default().trim());
    Ok(0)
}

#[derive(Default)]
struct Parsed {
    session_id: Option<String>,
    turn_id: Option<String>,
    user: Option<String>,
    assistant: Option<String>,
}

/// Walk the rollout once tracking the last user message and the assistant
/// reply that followed it, plus the turn id active at the time.
///
/// Codex emits events in the order: `user_message`, `turn_started`,
/// `agent_message`. We attribute the most recent `user_message` to whatever
/// `turn_started` arrives next, and capture the next `agent_message` that
/// follows.
fn parse_last(path: &Path) -> Result<Parsed> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Parsed::default();
    let mut pending_user: Option<String> = None;
    let mut pending_turn_id: Option<String> = None;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match rollout_kind(&rec) {
            RolloutKind::SessionMeta => {
                if let Some(id) = rec
                    .pointer("/payload/id")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned)
                {
                    out.session_id = Some(id);
                }
            }
            RolloutKind::EventUserMessage(text) => {
                if !text.is_empty() {
                    pending_user = Some(text);
                    pending_turn_id = None;
                    out.assistant = None;
                }
            }
            RolloutKind::EventTurnStarted(turn_id) => {
                pending_turn_id = Some(turn_id);
            }
            RolloutKind::EventAgentMessage(text) => {
                if text.is_empty() {
                    continue;
                }
                if let Some(u) = pending_user.take() {
                    out.user = Some(u);
                    out.turn_id = pending_turn_id.take();
                }
                out.assistant = Some(text);
            }
            RolloutKind::Other => {}
        }
    }
    Ok(out)
}

/// Find a turn by `turn_id`, then capture the surrounding user message
/// (the `user_message` immediately preceding the `turn_started`) and the
/// next `agent_message`.
fn pair_for_turn(path: &Path, target: &str) -> Result<Parsed> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Parsed::default();
    let mut pending_user: Option<String> = None;
    let mut captured = false;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match rollout_kind(&rec) {
            RolloutKind::EventUserMessage(text) => {
                if !text.is_empty() {
                    pending_user = Some(text);
                }
            }
            RolloutKind::EventTurnStarted(turn_id) => {
                if turn_id == target {
                    out.user = pending_user.take();
                    out.turn_id = Some(turn_id);
                    captured = true;
                } else {
                    // Different turn — drop the pending user message so it
                    // isn't mis-attributed if the matching turn comes later.
                    pending_user = None;
                }
            }
            RolloutKind::EventAgentMessage(text) if captured && out.assistant.is_none() => {
                if !text.is_empty() {
                    out.assistant = Some(text);
                    return Ok(out);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

enum RolloutKind {
    SessionMeta,
    EventUserMessage(String),
    EventAgentMessage(String),
    EventTurnStarted(String),
    Other,
}

fn rollout_kind(rec: &Value) -> RolloutKind {
    let outer = rec.get("type").and_then(|v| v.as_str());
    let payload = rec.get("payload").unwrap_or(rec);
    match outer {
        Some("session_meta") => RolloutKind::SessionMeta,
        Some("event_msg") => {
            let inner = payload.get("type").and_then(|v| v.as_str());
            match inner {
                Some("user_message") => RolloutKind::EventUserMessage(
                    payload
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned(),
                ),
                Some("agent_message") => RolloutKind::EventAgentMessage(
                    payload
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned(),
                ),
                Some("turn_started") => RolloutKind::EventTurnStarted(
                    payload
                        .get("turn_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned(),
                ),
                _ => RolloutKind::Other,
            }
        }
        _ => RolloutKind::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, lines: &[&str]) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "lethe-codex-test-{}-{name}.jsonl",
            std::process::id()
        ));
        let mut f = File::create(&path).unwrap();
        for line in lines {
            writeln!(f, "{line}").unwrap();
        }
        path
    }

    #[test]
    fn parses_last_user_assistant_pair() {
        let p = write_temp(
            "last",
            &[
                r#"{"type":"session_meta","payload":{"id":"sess-1"}}"#,
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"hello"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t1"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"hi"}}"#,
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"again"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t2"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"second reply"}}"#,
            ],
        );
        let parsed = parse_last(&p).unwrap();
        assert_eq!(parsed.session_id.as_deref(), Some("sess-1"));
        assert_eq!(parsed.turn_id.as_deref(), Some("t2"));
        assert_eq!(parsed.user.as_deref(), Some("again"));
        assert_eq!(parsed.assistant.as_deref(), Some("second reply"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn finds_specific_turn() {
        let p = write_temp(
            "turn",
            &[
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"first"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t1"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"reply 1"}}"#,
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"second"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t2"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"reply 2"}}"#,
            ],
        );
        let pair = pair_for_turn(&p, "t1").unwrap();
        assert_eq!(pair.user.as_deref(), Some("first"));
        assert_eq!(pair.assistant.as_deref(), Some("reply 1"));
        let pair2 = pair_for_turn(&p, "t2").unwrap();
        assert_eq!(pair2.user.as_deref(), Some("second"));
        assert_eq!(pair2.assistant.as_deref(), Some("reply 2"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn ignores_unknown_lines_and_malformed_json() {
        let p = write_temp(
            "garbage",
            &[
                r"not json at all",
                r#"{"type":"compacted","payload":{}}"#,
                r#"{"type":"event_msg","payload":{"type":"user_message","message":"hi"}}"#,
                r#"{"type":"event_msg","payload":{"type":"turn_started","turn_id":"t1"}}"#,
                r#"{"type":"event_msg","payload":{"type":"agent_message","message":"hello"}}"#,
            ],
        );
        let parsed = parse_last(&p).unwrap();
        assert_eq!(parsed.user.as_deref(), Some("hi"));
        assert_eq!(parsed.assistant.as_deref(), Some("hello"));
        std::fs::remove_file(&p).ok();
    }
}

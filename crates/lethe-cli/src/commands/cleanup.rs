//! `lethe cleanup` — delete dead/empty Claude Code & Codex transcripts.
//!
//! Scans both stores for transcripts whose repo no longer exists on disk
//! or that yield no memories, previews them (dry-run by default), and
//! deletes on `--yes`. Irreversible — always preview first.

use anyhow::Result;
use lethe_core::maintenance::{self, human_bytes, StaleTranscript};
use lethe_core::transcript_index::Source;
use serde::Serialize;

#[derive(Serialize)]
struct Item {
    path: String,
    source: &'static str,
    reason: &'static str,
    cwd: Option<String>,
    bytes: u64,
}

pub fn run(yes: bool, json_output: bool) -> Result<i32> {
    let stale = maintenance::scan_stale_transcripts();
    let total_bytes: u64 = stale.iter().map(|s| s.bytes).sum();

    if json_output {
        let items: Vec<Item> = stale.iter().map(to_item).collect();
        if yes {
            let r = maintenance::delete_transcripts(&stale);
            println!(
                "{}",
                serde_json::json!({
                    "deleted": items,
                    "transcripts_deleted": r.transcripts,
                    "reclaimed_bytes": r.bytes,
                })
            );
        } else {
            println!("{}", serde_json::to_string_pretty(&items)?);
        }
        return Ok(0);
    }

    if stale.is_empty() {
        println!("no dead or empty transcripts found");
        return Ok(0);
    }

    if !yes {
        println!(
            "{} stale transcript(s), {} reclaimable (dry run — pass --yes to delete):",
            stale.len(),
            human_bytes(total_bytes),
        );
        for s in &stale {
            print_row(s);
        }
        return Ok(0);
    }

    let r = maintenance::delete_transcripts(&stale);
    println!(
        "deleted {} transcript(s); reclaimed {}",
        r.transcripts,
        human_bytes(r.bytes),
    );
    Ok(0)
}

fn source_label(s: Source) -> &'static str {
    match s {
        Source::ClaudeCode => "claude",
        Source::Codex => "codex",
    }
}

fn print_row(s: &StaleTranscript) {
    println!(
        "  {:<7} {:<12} {:>9}  {}",
        source_label(s.source),
        s.reason.label(),
        human_bytes(s.bytes),
        s.path.display(),
    );
}

fn to_item(s: &StaleTranscript) -> Item {
    Item {
        path: s.path.to_string_lossy().into_owned(),
        source: source_label(s.source),
        reason: s.reason.label(),
        cwd: s.cwd.clone(),
        bytes: s.bytes,
    }
}

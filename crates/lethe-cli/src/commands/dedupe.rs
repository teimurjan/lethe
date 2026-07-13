//! `lethe dedupe` — offline near-duplicate compaction of a project's
//! index.
//!
//! Clusters entry embeddings, merges each near-duplicate group into a
//! single canonical entry (folding RIF metadata), and records the
//! absorbed ids in an alias table so a later `lethe index` can't
//! resurrect them. `--dry-run` reports what would be merged without
//! touching the index. `--threshold` overrides the configured
//! `dedup_threshold`. `--all` compacts every registered project.

use anyhow::Result;
use lethe_core::registry;
use serde::Serialize;

use crate::paths::{resolve, Paths};

use super::store_helpers::{load_config, open_store, CliConfig};

#[derive(Serialize)]
struct DedupeCounts {
    scanned: usize,
    groups: usize,
    absorbed: usize,
    dry_run: bool,
}

pub fn run(
    root: Option<&str>,
    threshold: Option<f32>,
    dry_run: bool,
    json_output: bool,
    all: bool,
) -> Result<i32> {
    if all {
        return run_all(threshold, dry_run, json_output);
    }
    let paths = resolve(root);
    let cfg = load_config(&paths.config_path())?;
    let report = dedupe_project(&paths, &cfg, threshold, dry_run)?;

    let payload = DedupeCounts {
        scanned: report.scanned,
        groups: report.groups,
        absorbed: report.absorbed,
        dry_run,
    };
    if json_output {
        println!("{}", serde_json::to_string(&payload)?);
    } else {
        let verb = if dry_run { "would merge" } else { "merged" };
        println!(
            "dedupe: scanned={} groups={} {verb}={}",
            payload.scanned, payload.groups, payload.absorbed
        );
    }
    Ok(0)
}

/// Compact one project's index. Encoders aren't needed — dedupe works on
/// stored embeddings — so this is cheap (no model load). A dry run opens
/// read-only so it can't mutate.
fn dedupe_project(
    paths: &Paths,
    cfg: &CliConfig,
    threshold: Option<f32>,
    dry_run: bool,
) -> Result<lethe_core::memory_store::DedupeReport> {
    let store = open_store(&paths.index(), cfg, false, dry_run)?;
    let tau = threshold.unwrap_or(cfg.dedup_threshold);
    let report = store.dedupe(tau, dry_run)?;
    if !dry_run {
        store.save()?;
    }
    Ok(report)
}

fn run_all(threshold: Option<f32>, dry_run: bool, json_output: bool) -> Result<i32> {
    let entries = registry::load();
    if entries.is_empty() {
        eprintln!("no registered projects — run `lethe index` in a repo first");
        return Ok(1);
    }
    let cfg = load_config(&registry::registry_dir().join("config.toml"))?;

    #[derive(Serialize)]
    struct ProjectResult {
        slug: String,
        root: String,
        scanned: usize,
        groups: usize,
        absorbed: usize,
    }
    let mut results: Vec<ProjectResult> = Vec::new();
    let (mut grand_groups, mut grand_absorbed) = (0usize, 0usize);

    for entry in &entries {
        let paths = Paths {
            root: entry.root.clone(),
        };
        // Skip projects that were never indexed (no index dir yet).
        if !paths.index().join("lethe.duckdb").exists() {
            continue;
        }
        match dedupe_project(&paths, &cfg, threshold, dry_run) {
            Ok(report) => {
                grand_groups += report.groups;
                grand_absorbed += report.absorbed;
                results.push(ProjectResult {
                    slug: entry.slug.clone(),
                    root: entry.root.to_string_lossy().into_owned(),
                    scanned: report.scanned,
                    groups: report.groups,
                    absorbed: report.absorbed,
                });
            }
            Err(e) => eprintln!("  skip {}: {e}", entry.root.display()),
        }
    }

    if json_output {
        println!("{}", serde_json::to_string(&results)?);
    } else {
        let verb = if dry_run { "would merge" } else { "merged" };
        println!(
            "dedupe {} projects: groups={grand_groups} {verb}={grand_absorbed}",
            results.len()
        );
    }
    Ok(0)
}

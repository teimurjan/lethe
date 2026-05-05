//! `lethe search` — local + cross-project retrieval.

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use lethe_core::encoders::{BiEncoder, CrossEncoder};
use lethe_core::memory_store::StoreConfig;
use lethe_core::registry;
use lethe_core::rif::RifConfig;
use lethe_core::union_store::UnionStore;
use serde::Serialize;

use crate::paths::resolve;

use super::store_helpers::{load_config, open_store};

#[derive(Serialize)]
struct LocalHit {
    id: String,
    content: String,
    score: f32,
}

#[derive(Serialize)]
struct UnionHitJson {
    project_slug: String,
    project_root: String,
    id: String,
    content: String,
    score: f32,
}

/// Single-project search. By default the index is opened read-write
/// and retrieval-driven state (suppression scores, tier transitions,
/// retrieval counts) is persisted after the query so RIF evolves.
/// Pass `read_only=true` (`--read-only` on the CLI) to share the
/// index with other concurrent lethe processes — at the cost of not
/// learning from this query.
pub fn run_local(
    root: Option<&str>,
    query: &str,
    top_k: usize,
    json_output: bool,
    read_only: bool,
) -> Result<i32> {
    let paths = resolve(root);
    let cfg = load_config(&paths.config_path())?;
    let store = open_store(&paths.index(), &cfg, true, read_only)?;
    let hits = store.retrieve(query, top_k)?;
    if !read_only {
        store.save()?;
    }

    if json_output {
        let payload: Vec<LocalHit> = hits
            .iter()
            .map(|h| LocalHit {
                id: h.id.clone(),
                content: h.content.clone(),
                score: h.score,
            })
            .collect();
        println!("{}", serde_json::to_string(&payload)?);
        return Ok(0);
    }
    if hits.is_empty() {
        println!("(no results)");
        return Ok(0);
    }
    for h in &hits {
        println!("[{:+.2}] {}  {}", h.score, h.id, snippet(&h.content, 160));
    }
    Ok(0)
}

pub fn run_union(
    query: &str,
    top_k: usize,
    json_output: bool,
    projects_filter: Option<&str>,
) -> Result<i32> {
    let all_entries = registry::load();
    let entries = if let Some(filter) = projects_filter {
        let wanted: HashSet<String> = filter
            .split(',')
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        let known_slugs: HashSet<String> = all_entries.iter().map(|e| e.slug.clone()).collect();
        let known_paths: HashSet<String> = all_entries
            .iter()
            .map(|e| e.root.to_string_lossy().into_owned())
            .collect();
        for name in &wanted {
            if !known_slugs.contains(name) && !known_paths.contains(name) {
                eprintln!("[lethe] unknown project: {name}");
            }
        }
        all_entries
            .into_iter()
            .filter(|e| {
                wanted.contains(&e.slug) || wanted.contains(&e.root.to_string_lossy().into_owned())
            })
            .collect::<Vec<_>>()
    } else {
        all_entries
    };

    if entries.is_empty() {
        eprintln!(
            "no projects registered — run `lethe index` in each project, \
             or pass --projects <slug,slug>"
        );
        return Ok(1);
    }

    // Pull encoder names from the first project that has a config.toml.
    let cfg_root = entries
        .iter()
        .map(|e| e.root.clone())
        .find(|r| r.join(".lethe").join("config.toml").exists())
        .unwrap_or_else(|| entries[0].root.clone());
    let cfg = load_config(&cfg_root.join(".lethe").join("config.toml"))?;
    let bi = Arc::new(BiEncoder::from_repo(&cfg.bi_encoder)?);
    let cross = Arc::new(CrossEncoder::from_repo(&cfg.cross_encoder)?);
    let store_cfg = StoreConfig {
        dim: bi.dim(),
        rif: RifConfig {
            n_clusters: cfg.n_clusters,
            use_rank_gap: cfg.use_rank_gap,
            ..RifConfig::default()
        },
        ..StoreConfig::default()
    };
    let union = UnionStore::open(entries, Some(bi), Some(cross), store_cfg);
    let hits = union.retrieve(query, top_k)?;

    if json_output {
        let payload: Vec<UnionHitJson> = hits
            .iter()
            .map(|h| UnionHitJson {
                project_slug: h.project_slug.clone(),
                project_root: h.project_root.to_string_lossy().into_owned(),
                id: h.id.clone(),
                content: h.content.clone(),
                score: h.score,
            })
            .collect();
        println!("{}", serde_json::to_string(&payload)?);
        return Ok(0);
    }
    if hits.is_empty() {
        println!("(no results)");
        return Ok(0);
    }
    for h in &hits {
        println!(
            "[{}] [{:+.2}] {}  {}",
            h.project_slug,
            h.score,
            h.id,
            snippet(&h.content, 160)
        );
    }
    Ok(0)
}

fn snippet(content: &str, width: usize) -> String {
    for line in content.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if s.starts_with('#') {
            continue;
        }
        if s.starts_with("<!--") && s.ends_with("-->") {
            continue;
        }
        if s.len() > width {
            return s.chars().take(width).collect();
        }
        return s.to_owned();
    }
    "(heading only)".to_owned()
}

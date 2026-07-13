//! Shared store-construction + config helpers.
//!
//! Mirrors the Python `_open_store` + `load_config` + `save_config`
//! family. Defaults match `DEFAULT_CONFIG` in `research_playground/lethe_reference/lethe/cli.py`.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use lethe_core::encoders::{BiEncoder, CrossEncoder};
use lethe_core::memory_store::{MemoryStore, StoreConfig};
use lethe_core::rif::RifConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CliConfig {
    pub bi_encoder: String,
    pub cross_encoder: String,
    pub top_k: usize,
    pub n_clusters: u32,
    pub use_rank_gap: bool,
    /// Cosine cutoff for near-duplicate detection (the write-time gate
    /// and the offline `lethe dedupe` compaction).
    pub dedup_threshold: f32,
    /// Catch-all for unknown keys so `config get` still surfaces them.
    #[serde(flatten)]
    pub extra: BTreeMap<String, toml::Value>,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            bi_encoder: "all-MiniLM-L6-v2".to_owned(),
            cross_encoder: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_owned(),
            top_k: 5,
            n_clusters: 30,
            use_rank_gap: true,
            dedup_threshold: 0.95,
            extra: BTreeMap::new(),
        }
    }
}

pub fn load_config(config_path: &Path) -> Result<CliConfig> {
    if !config_path.exists() {
        return Ok(CliConfig::default());
    }
    let txt = std::fs::read_to_string(config_path)?;
    // Surface parse errors instead of silently falling back to defaults
    // — `config.toml` is user-controlled and a typo there should fail
    // loudly with a line/column from the toml crate.
    toml::from_str::<CliConfig>(&txt).with_context(|| format!("parse {}", config_path.display()))
}

pub fn save_config(config_path: &Path, cfg: &CliConfig) -> Result<()> {
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = toml::to_string_pretty(cfg)?;
    std::fs::write(config_path, s)?;
    Ok(())
}

/// Build a `MemoryStore` rooted at `<paths.index()>` honoring `cfg`.
/// `need_encoders=true` triggers the (potentially network-bound)
/// model download + ONNX session warm-up. `read_only=true` opens the
/// DuckDB connection in `AccessMode::ReadOnly` so the recall path can
/// run alongside other lethe processes (multi-reader, single-writer).
pub fn open_store(
    index_dir: &Path,
    cfg: &CliConfig,
    need_encoders: bool,
    read_only: bool,
) -> Result<MemoryStore> {
    let bi = if need_encoders {
        Some(Arc::new(BiEncoder::from_repo(&cfg.bi_encoder)?))
    } else {
        None
    };
    let cross = if need_encoders {
        Some(Arc::new(CrossEncoder::from_repo(&cfg.cross_encoder)?))
    } else {
        None
    };
    let dim = bi.as_ref().map_or(384, |b| b.dim());
    let rif = RifConfig {
        n_clusters: cfg.n_clusters,
        use_rank_gap: cfg.use_rank_gap,
        ..RifConfig::default()
    };
    let store_cfg = StoreConfig {
        dim,
        rif,
        read_only,
        dedup_threshold: cfg.dedup_threshold,
        ..StoreConfig::default()
    };
    let store = MemoryStore::open(index_dir, bi, cross, store_cfg)?;
    Ok(store)
}

/// Wipe the index dir when its on-disk format predates the running
/// binary, so the next `open_store` + `ensure_fresh` rebuilds from the
/// (authoritative) transcripts. No-op when the index is absent or
/// current. Transcripts are never touched. Must run **before**
/// `open_store` grabs the writer lock.
pub fn ensure_index_format(index_dir: &Path) -> Result<()> {
    use lethe_core::db::{MemoryDb, INDEX_FORMAT, INDEX_FORMAT_KEY};

    let db_path = index_dir.join("lethe.duckdb");
    if !db_path.exists() {
        return Ok(());
    }
    // Read the marker in its own scope so the connection is dropped
    // before we remove the directory.
    let stale = {
        match MemoryDb::open_with_mode(&db_path, true) {
            Ok(db) => db
                .get_stat(INDEX_FORMAT_KEY, "0")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(0)
                < INDEX_FORMAT,
            // Unreadable (locked / corrupt): leave it — open_store will
            // surface a real error rather than silently nuking data.
            Err(_) => false,
        }
    };
    if stale {
        std::fs::remove_dir_all(index_dir)
            .with_context(|| format!("rebuild: remove {}", index_dir.display()))?;
        eprintln!("lethe: index format changed — rebuilding from transcripts…");
    }
    Ok(())
}

/// Coerce a config value string the same way Python does.
pub fn coerce_scalar(s: &str) -> toml::Value {
    if let Ok(b) = s.parse::<bool>() {
        return toml::Value::Boolean(b);
    }
    if let Ok(i) = s.parse::<i64>() {
        return toml::Value::Integer(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return toml::Value::Float(f);
    }
    toml::Value::String(s.to_owned())
}

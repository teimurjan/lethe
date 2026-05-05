//! Cross-project read-only retrieval — port of
//! `research_playground/lethe_reference/lethe/union_store.py`.
//!
//! For each registered project: opens the per-project DuckDB read-only,
//! fans the hybrid retrieve out via the same BM25/dense/RRF stack,
//! then cross-encoder reranks the merged pool.
//!
//! The Python implementation uses one in-memory DuckDB and ATTACHes
//! every project to it. The Rust port goes simpler: open one
//! `MemoryStore` per project in read-only mode (no add/save), call
//! retrieve, then merge across projects via a per-project RRF.
//!
//! Each per-project DuckDB is opened with `AccessMode::ReadOnly`, so
//! many `lethe search --all` / `recall-global` invocations can stack
//! shared-lock readers on the same index file. DuckDB's cross-process
//! semantics are one writer xor many readers per file, so an ingest
//! writer (`lethe index` / stop hook) still serializes against active
//! readers — it waits through the open-with-retry helper in `db.rs`.

use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;

use crate::encoders::{BiEncoder, CrossEncoder};
use crate::memory_store::{Hit, MemoryStore, StoreConfig};
use crate::registry::ProjectEntry;

/// Per-project candidates pulled before the global rerank. Larger
/// values trade rerank batch size for recall; 30 matches the default
/// single-project `k_shallow` so behavior is comparable.
const UNION_K_GATHER: usize = 30;
/// Cap on the merged pool that gets cross-encoder reranked. The whole
/// pool goes through one batched ONNX forward, so this is the dominant
/// latency knob — keep it small enough to fit in one Mutex acquisition
/// of the cross-encoder session.
const UNION_K_RERANK: usize = 60;

/// One ranked, project-tagged hit.
#[derive(Debug, Clone)]
pub struct UnionHit {
    pub project_slug: String,
    pub project_root: PathBuf,
    pub id: String,
    pub content: String,
    pub score: f32,
}

#[derive(Debug)]
pub struct UnionStore {
    projects: Vec<UnionProject>,
    bi_encoder: Option<Arc<BiEncoder>>,
    cross_encoder: Option<Arc<CrossEncoder>>,
}

#[derive(Debug)]
struct UnionProject {
    entry: ProjectEntry,
    store: MemoryStore,
}

impl UnionStore {
    /// Build a UnionStore from a list of registered projects. Bad
    /// projects (missing index, schema mismatch) are skipped with a
    /// warning to stderr — same behavior as the Python helper. Each
    /// store opens in parallel since the work is dominated by DuckDB
    /// hydration + embedding load and is independent across projects.
    pub fn open(
        projects: Vec<ProjectEntry>,
        bi_encoder: Option<Arc<BiEncoder>>,
        cross_encoder: Option<Arc<CrossEncoder>>,
        mut config: StoreConfig,
    ) -> Self {
        config.read_only = true;
        let handles: Vec<UnionProject> = projects
            .into_par_iter()
            .filter_map(|entry| {
                let store_path = entry.root.join(".lethe").join("index");
                if !store_path.join("lethe.duckdb").exists() {
                    return None;
                }
                match MemoryStore::open(
                    &store_path,
                    bi_encoder.clone(),
                    cross_encoder.clone(),
                    config.clone(),
                ) {
                    Ok(store) => Some(UnionProject { entry, store }),
                    Err(e) => {
                        eprintln!("[lethe] skipping {}: {e}", entry.root.display());
                        None
                    }
                }
            })
            .collect();
        Self {
            projects: handles,
            bi_encoder,
            cross_encoder,
        }
    }

    /// `(slug, root, entry_count)` for diagnostics.
    pub fn stats(&self) -> Vec<(String, PathBuf, usize)> {
        self.projects
            .iter()
            .map(|p| (p.entry.slug.clone(), p.entry.root.clone(), p.store.size()))
            .collect()
    }

    /// Cross-project retrieve in two phases:
    ///
    /// **Phase A** (parallel): each project runs `retrieve_shallow` —
    /// hybrid BM25 + dense + suppression penalty, no rerank. The
    /// bi-encoder query embedding is computed once globally and shared
    /// across projects.
    ///
    /// **Phase B** (single batched call): the per-project candidates
    /// are merged, capped to `UNION_K_RERANK` by gather score, then
    /// reranked through the cross-encoder in **one** ONNX forward.
    ///
    /// This avoids the trap that the per-project pipeline used to fall
    /// into: every parallel rayon task serializing through the
    /// cross-encoder's `Mutex<Session>`, multiplying rerank work by
    /// project count for no quality benefit.
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<UnionHit>, crate::Error> {
        if self.projects.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let bi = self
            .bi_encoder
            .as_ref()
            .ok_or(crate::Error::NotInitialized(
                "bi_encoder required for union retrieve",
            ))?;
        let query_emb = bi.encode(query)?;

        // Phase A — parallel shallow gather.
        let gathered: Vec<UnionHit> = self
            .projects
            .par_iter()
            .map(|proj| -> Result<Vec<UnionHit>, crate::Error> {
                let hits: Vec<Hit> =
                    proj.store
                        .retrieve_shallow(query_emb.view(), query, UNION_K_GATHER)?;
                Ok(hits
                    .into_iter()
                    .map(|h| UnionHit {
                        project_slug: proj.entry.slug.clone(),
                        project_root: proj.entry.root.clone(),
                        id: h.id,
                        content: h.content,
                        score: h.score,
                    })
                    .collect())
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        if gathered.is_empty() {
            return Ok(Vec::new());
        }

        // Cap merged pool to UNION_K_RERANK before the single rerank
        // call. Sort by gather score so we keep the strongest
        // candidates regardless of project.
        let mut pool = gathered;
        pool.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.project_slug.cmp(&b.project_slug))
                .then_with(|| a.id.cmp(&b.id))
        });
        pool.truncate(UNION_K_RERANK);

        // Phase B — one batched cross-encoder rerank on the merged
        // pool. If no cross-encoder is wired (test path), keep gather
        // scores as-is.
        if let Some(xenc) = self.cross_encoder.as_ref() {
            let pairs: Vec<(&str, &str)> =
                pool.iter().map(|h| (query, h.content.as_str())).collect();
            let scores = xenc.predict(&pairs)?;
            for (hit, score) in pool.iter_mut().zip(scores) {
                hit.score = score;
            }
            pool.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.project_slug.cmp(&b.project_slug))
                    .then_with(|| a.id.cmp(&b.id))
            });
        }

        pool.truncate(k);
        Ok(pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn empty_projects_yields_empty() {
        let union = UnionStore::open(Vec::new(), None, None, StoreConfig::default());
        assert!(union.retrieve("anything", 5).unwrap().is_empty());
        assert!(union.stats().is_empty());
    }

    #[test]
    fn skips_projects_without_index() {
        let dir = tempdir().unwrap();
        // ProjectEntry pointing at a directory that has no .lethe/index/lethe.duckdb.
        let entry = ProjectEntry {
            root: dir.path().to_path_buf(),
            slug: "p_test_00000000".into(),
            registered_at: "2026-01-01T00:00:00+00:00".into(),
        };
        let union = UnionStore::open(vec![entry], None, None, StoreConfig::default());
        assert!(union.stats().is_empty());
    }
}

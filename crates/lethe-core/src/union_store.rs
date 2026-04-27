//! Cross-project read-only retrieval — port of
//! `legacy/lethe/union_store.py`.
//!
//! For each registered project: opens the per-project DuckDB read-only
//! (so the per-project hook writers can keep mutating freely), fans
//! the hybrid retrieve out via the same BM25/dense/RRF stack, then
//! cross-encoder reranks the merged pool.
//!
//! The Python implementation uses one in-memory DuckDB and ATTACHes
//! every project to it. The Rust port goes simpler: open one
//! `MemoryStore` per project in read-only mode (no add/save), call
//! retrieve, then merge across projects via a per-project RRF.
//!
//! Read-only here means: we never call `save()` and we never call
//! `add()`. The per-project DuckDB connections are still opened with
//! write intent (DuckDB's Rust crate can't reliably attach read-only
//! across versions), but we strictly avoid mutation.

use std::path::PathBuf;
use std::sync::Arc;

use rayon::prelude::*;

use crate::encoders::{BiEncoder, CrossEncoder};
use crate::memory_store::{Hit, MemoryStore, StoreConfig};
use crate::registry::ProjectEntry;

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
        config: StoreConfig,
    ) -> Self {
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
        Self { projects: handles }
    }

    /// `(slug, root, entry_count)` for diagnostics.
    pub fn stats(&self) -> Vec<(String, PathBuf, usize)> {
        self.projects
            .iter()
            .map(|p| (p.entry.slug.clone(), p.entry.root.clone(), p.store.size()))
            .collect()
    }

    /// Cross-project retrieve. Per-project top-`k` results are
    /// concatenated then re-sorted by their rerank scores; the
    /// `k` highest overall are returned.
    ///
    /// Per-project pipelines run in parallel via rayon — the heavy
    /// cost (BM25 + dense + cross-encoder rerank) is independent
    /// across projects. The bi-encoder query embedding is computed
    /// once per project but the duplicated work (~5 ms each) is
    /// dwarfed by the parallel speedup on multi-core hosts.
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<UnionHit>, crate::Error> {
        if self.projects.is_empty() {
            return Ok(Vec::new());
        }
        let mut all: Vec<UnionHit> = self
            .projects
            .par_iter()
            .map(|proj| -> Result<Vec<UnionHit>, crate::Error> {
                let per_project: Vec<Hit> = proj.store.retrieve(query, k)?;
                Ok(per_project
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
        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.project_slug.cmp(&b.project_slug))
                .then_with(|| a.id.cmp(&b.id))
        });
        all.truncate(k);
        Ok(all)
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

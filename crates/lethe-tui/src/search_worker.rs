//! Background thread that owns the encoders and runs retrieval calls.
//! The TUI thread enqueues `SearchRequest`s and drains `SearchOutput`s
//! out-of-band so the UI never blocks during a query.

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use lethe_core::encoders::{BiEncoder, CrossEncoder};
use lethe_core::memory_store::{MemoryStore, StoreConfig};
use lethe_core::registry;
use lethe_core::rif::RifConfig;
use lethe_core::union_store::UnionStore;
use std::sync::Arc;

use crate::app::{ResultRow, Scope};

#[derive(Debug)]
pub struct SearchRequest {
    pub query: String,
    pub scope: Scope,
    pub top_k: usize,
}

#[derive(Debug)]
pub enum SearchOutput {
    Hits(Vec<ResultRow>),
    Error(String),
}

const DEFAULT_BI: &str = "all-MiniLM-L6-v2";
const DEFAULT_CROSS: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";

/// Spawn the worker. Returns the request-sender + result-receiver.
pub fn spawn() -> (Sender<SearchRequest>, Receiver<SearchOutput>) {
    let (req_tx, req_rx) = mpsc::channel::<SearchRequest>();
    let (out_tx, out_rx) = mpsc::channel::<SearchOutput>();
    thread::spawn(move || {
        let mut state = WorkerState::default();
        while let Ok(req) = req_rx.recv() {
            let result = state.handle(req);
            let _ = out_tx.send(result);
        }
    });
    (req_tx, out_rx)
}

#[derive(Default)]
struct WorkerState {
    bi: Option<Arc<BiEncoder>>,
    cross: Option<Arc<CrossEncoder>>,
    /// Cached union store keyed by the registered project list.
    union_cache_key: Option<String>,
    union_store: Option<UnionStore>,
    /// Cached single-project store keyed by slug.
    single_cache_slug: Option<String>,
    single_store: Option<MemoryStore>,
}

impl WorkerState {
    fn ensure_encoders(&mut self) -> Result<(Arc<BiEncoder>, Arc<CrossEncoder>), String> {
        if self.bi.is_none() {
            let b = BiEncoder::from_repo(DEFAULT_BI).map_err(|e| e.to_string())?;
            self.bi = Some(Arc::new(b));
        }
        if self.cross.is_none() {
            let c = CrossEncoder::from_repo(DEFAULT_CROSS).map_err(|e| e.to_string())?;
            self.cross = Some(Arc::new(c));
        }
        Ok((self.bi.clone().unwrap(), self.cross.clone().unwrap()))
    }

    fn config_with_dim(dim: usize) -> StoreConfig {
        StoreConfig {
            dim,
            rif: RifConfig {
                n_clusters: 30,
                use_rank_gap: true,
                ..RifConfig::default()
            },
            ..StoreConfig::default()
        }
    }

    fn handle(&mut self, req: SearchRequest) -> SearchOutput {
        match self.run(req) {
            Ok(rows) => SearchOutput::Hits(rows),
            Err(e) => SearchOutput::Error(e),
        }
    }

    fn run(&mut self, req: SearchRequest) -> Result<Vec<ResultRow>, String> {
        let (bi, cross) = self.ensure_encoders()?;
        let cfg = Self::config_with_dim(bi.dim());
        match req.scope {
            Scope::AllProjects => {
                let entries = registry::load();
                let key = entries
                    .iter()
                    .map(|e| e.slug.clone())
                    .collect::<Vec<_>>()
                    .join(",");
                if self.union_cache_key.as_deref() != Some(&key) {
                    self.union_store = None;
                    self.single_store = None;
                    self.single_cache_slug = None;
                    let union = UnionStore::open(
                        entries,
                        Some(bi.clone()),
                        Some(cross.clone()),
                        cfg.clone(),
                    );
                    self.union_store = Some(union);
                    self.union_cache_key = Some(key);
                }
                let store = self.union_store.as_ref().expect("just set");
                let hits = store
                    .retrieve(&req.query, req.top_k)
                    .map_err(|e| e.to_string())?;
                Ok(hits
                    .into_iter()
                    .map(|h| ResultRow {
                        project_slug: Some(h.project_slug),
                        project_root: Some(h.project_root),
                        id: h.id,
                        content: h.content,
                        score: h.score,
                    })
                    .collect())
            }
            Scope::Single(entry) => {
                if self.single_cache_slug.as_deref() != Some(entry.slug.as_str()) {
                    self.union_store = None;
                    self.union_cache_key = None;
                    let store_path = entry.root.join(".lethe").join("index");
                    let store = MemoryStore::open(
                        &store_path,
                        Some(bi.clone()),
                        Some(cross.clone()),
                        cfg.clone(),
                    )
                    .map_err(|e| e.to_string())?;
                    self.single_store = Some(store);
                    self.single_cache_slug = Some(entry.slug.clone());
                }
                let store = self.single_store.as_ref().expect("just set");
                let hits = store
                    .retrieve(&req.query, req.top_k)
                    .map_err(|e| e.to_string())?;
                Ok(hits
                    .into_iter()
                    .map(|h| ResultRow {
                        project_slug: None,
                        project_root: None,
                        id: h.id,
                        content: h.content,
                        score: h.score,
                    })
                    .collect())
            }
        }
    }
}

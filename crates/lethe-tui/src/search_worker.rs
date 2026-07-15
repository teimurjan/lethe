//! Background worker that owns the encoders and runs everything that
//! touches a project's DuckDB — retrieval, indexing, and cleanup. Keeping
//! it single-threaded means the read-only caches it holds for search can
//! be dropped before it opens a store read-write to index, so a writer
//! never deadlocks against the TUI's own readers.
//!
//! The TUI thread enqueues [`WorkerRequest`]s and drains [`WorkerOutput`]s
//! out-of-band so the UI never blocks.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use lethe_core::encoders::{BiEncoder, CrossEncoder};
use lethe_core::maintenance::{self, Reclaimed, StaleTranscript};
use lethe_core::memory_store::{MemoryStore, StoreConfig};
use lethe_core::registry::{self, ProjectEntry};
use lethe_core::rif::RifConfig;
use lethe_core::transcript_index;
use lethe_core::union_store::UnionStore;

use crate::app::{ResultRow, Scope};

#[derive(Debug)]
pub struct SearchQuery {
    pub query: String,
    pub scope: Scope,
    pub top_k: usize,
}

#[derive(Debug)]
pub enum WorkerRequest {
    Search(SearchQuery),
    /// Index the given projects (one, or all registered).
    Index(Vec<ProjectEntry>),
    /// Scan Claude/Codex storage for stale transcripts.
    Scan,
    /// Delete the given stale transcripts from disk.
    DeleteStale(Vec<StaleTranscript>),
    /// Delete stored memory records (age-based) from the given projects.
    DeleteRecords {
        entries: Vec<ProjectEntry>,
        days: u32,
    },
    /// Remove whole projects (index + registry, optionally transcripts).
    DeleteProjects {
        entries: Vec<ProjectEntry>,
        with_transcripts: bool,
    },
    /// Wipe each project's index dir, then re-parse every transcript from
    /// scratch. Unlike [`Index`] (incremental), this picks up parser/format
    /// fixes on transcripts whose mtime/size never changed.
    Rebuild(Vec<ProjectEntry>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    LoadingBiEncoder,
    LoadingCrossEncoder,
    LoadingIndex,
    Searching,
    Indexing,
    Scanning,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Phase::LoadingBiEncoder => "loading bi-encoder weights",
            Phase::LoadingCrossEncoder => "loading cross-encoder weights",
            Phase::LoadingIndex => "loading project index",
            Phase::Searching => "searching",
            Phase::Indexing => "indexing transcripts",
            Phase::Scanning => "scanning transcripts",
        }
    }
}

#[derive(Debug)]
pub enum WorkerOutput {
    Phase(Phase),
    Hits(Vec<ResultRow>),
    Indexed { added: usize, projects: usize },
    Scanned(Vec<StaleTranscript>),
    Deleted(Reclaimed),
    Error(String),
}

const DEFAULT_BI: &str = "all-MiniLM-L6-v2";
const DEFAULT_CROSS: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";

/// Spawn the worker. Returns the request-sender + result-receiver.
pub fn spawn() -> (Sender<WorkerRequest>, Receiver<WorkerOutput>) {
    let (req_tx, req_rx) = mpsc::channel::<WorkerRequest>();
    let (out_tx, out_rx) = mpsc::channel::<WorkerOutput>();
    thread::spawn(move || {
        let mut state = WorkerState::default();
        while let Ok(req) = req_rx.recv() {
            state.handle(req, &out_tx);
        }
    });
    (req_tx, out_rx)
}

#[derive(Default)]
struct WorkerState {
    bi: Option<Arc<BiEncoder>>,
    cross: Option<Arc<CrossEncoder>>,
    union_cache_key: Option<String>,
    union_store: Option<UnionStore>,
    single_cache_slug: Option<String>,
    single_store: Option<MemoryStore>,
}

impl WorkerState {
    fn ensure_encoders(
        &mut self,
        tx: &Sender<WorkerOutput>,
    ) -> Result<(Arc<BiEncoder>, Arc<CrossEncoder>), String> {
        if self.bi.is_none() {
            let _ = tx.send(WorkerOutput::Phase(Phase::LoadingBiEncoder));
            let b = BiEncoder::from_repo(DEFAULT_BI).map_err(|e| e.to_string())?;
            self.bi = Some(Arc::new(b));
        }
        if self.cross.is_none() {
            let _ = tx.send(WorkerOutput::Phase(Phase::LoadingCrossEncoder));
            let c = CrossEncoder::from_repo(DEFAULT_CROSS).map_err(|e| e.to_string())?;
            self.cross = Some(Arc::new(c));
        }
        Ok((self.bi.clone().unwrap(), self.cross.clone().unwrap()))
    }

    /// Drop cached read handles so the underlying DuckDB files are unlocked.
    fn release_caches(&mut self) {
        self.union_store = None;
        self.union_cache_key = None;
        self.single_store = None;
        self.single_cache_slug = None;
    }

    fn config(dim: usize, read_only: bool) -> StoreConfig {
        StoreConfig {
            dim,
            rif: RifConfig {
                n_clusters: 30,
                use_rank_gap: true,
                ..RifConfig::default()
            },
            read_only,
            ..StoreConfig::default()
        }
    }

    fn handle(&mut self, req: WorkerRequest, tx: &Sender<WorkerOutput>) {
        match req {
            WorkerRequest::Search(q) => {
                let out = match self.search(q, tx) {
                    Ok(rows) => WorkerOutput::Hits(rows),
                    Err(e) => WorkerOutput::Error(e),
                };
                let _ = tx.send(out);
            }
            WorkerRequest::Index(projects) => {
                let out = match self.index(&projects, tx) {
                    Ok((added, n)) => WorkerOutput::Indexed { added, projects: n },
                    Err(e) => WorkerOutput::Error(e),
                };
                let _ = tx.send(out);
            }
            WorkerRequest::Scan => {
                let _ = tx.send(WorkerOutput::Phase(Phase::Scanning));
                let _ = tx.send(WorkerOutput::Scanned(maintenance::scan_stale_transcripts()));
            }
            WorkerRequest::DeleteStale(items) => {
                self.release_caches();
                let _ = tx.send(WorkerOutput::Deleted(maintenance::delete_transcripts(
                    &items,
                )));
            }
            WorkerRequest::DeleteRecords { entries, days } => {
                self.release_caches();
                let _ = tx.send(WorkerOutput::Deleted(maintenance::delete_records(
                    &entries, days,
                )));
            }
            WorkerRequest::DeleteProjects {
                entries,
                with_transcripts,
            } => {
                self.release_caches();
                let mut total = Reclaimed::default();
                for e in &entries {
                    total =
                        add_reclaimed(total, maintenance::delete_project_data(e, with_transcripts));
                }
                let _ = tx.send(WorkerOutput::Deleted(total));
            }
            WorkerRequest::Rebuild(projects) => {
                self.release_caches();
                // Drop each index dir so `ensure_fresh` has no manifest to
                // trust and re-parses every transcript from scratch.
                for e in &projects {
                    let dir = registry::registry_dir().join("index").join(&e.slug);
                    let _ = std::fs::remove_dir_all(&dir);
                }
                let out = match self.index(&projects, tx) {
                    Ok((added, n)) => WorkerOutput::Indexed { added, projects: n },
                    Err(e) => WorkerOutput::Error(e),
                };
                let _ = tx.send(out);
            }
        }
    }

    /// Open each project read-write and freshen it. Drops read caches
    /// first so the writer doesn't fight the TUI's own read locks.
    fn index(
        &mut self,
        projects: &[ProjectEntry],
        tx: &Sender<WorkerOutput>,
    ) -> Result<(usize, usize), String> {
        let (bi, cross) = self.ensure_encoders(tx)?;
        self.release_caches();
        let cfg = Self::config(bi.dim(), false);
        let _ = tx.send(WorkerOutput::Phase(Phase::Indexing));
        let mut added = 0usize;
        let mut done = 0usize;
        for entry in projects {
            let dir = registry::registry_dir().join("index").join(&entry.slug);
            let store = MemoryStore::open(&dir, Some(bi.clone()), Some(cross.clone()), cfg.clone())
                .map_err(|e| e.to_string())?;
            let counts =
                transcript_index::ensure_fresh(&store, &entry.root).map_err(|e| e.to_string())?;
            store.save().map_err(|e| e.to_string())?;
            if !registry::is_disabled() {
                let _ = registry::register(&entry.root);
            }
            added += counts.added;
            done += 1;
        }
        Ok((added, done))
    }

    fn search(
        &mut self,
        req: SearchQuery,
        tx: &Sender<WorkerOutput>,
    ) -> Result<Vec<ResultRow>, String> {
        let (bi, cross) = self.ensure_encoders(tx)?;
        let cfg = Self::config(bi.dim(), true);
        match req.scope {
            Scope::AllProjects => {
                let entries = registry::load();
                let key = entries
                    .iter()
                    .map(|e| e.slug.clone())
                    .collect::<Vec<_>>()
                    .join(",");
                if self.union_cache_key.as_deref() != Some(&key) {
                    let _ = tx.send(WorkerOutput::Phase(Phase::LoadingIndex));
                    self.release_caches();
                    let union = UnionStore::open(
                        entries,
                        Some(bi.clone()),
                        Some(cross.clone()),
                        cfg.clone(),
                    );
                    self.union_store = Some(union);
                    self.union_cache_key = Some(key);
                }
                let _ = tx.send(WorkerOutput::Phase(Phase::Searching));
                let store = self.union_store.as_ref().expect("just set");
                let hits = store
                    .retrieve(&req.query, req.top_k)
                    .map_err(|e| e.to_string())?;
                Ok(hits
                    .into_iter()
                    .map(|h| ResultRow {
                        project_name: crate::app::project_name(&h.project_root),
                        project_root: Some(h.project_root),
                        id: h.id,
                        content: h.content,
                        score: h.score,
                        date_epoch: 0,
                    })
                    .collect())
            }
            Scope::Single(entry) => {
                if self.single_cache_slug.as_deref() != Some(entry.slug.as_str()) {
                    let _ = tx.send(WorkerOutput::Phase(Phase::LoadingIndex));
                    self.release_caches();
                    let store_path = registry::registry_dir().join("index").join(&entry.slug);
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
                let _ = tx.send(WorkerOutput::Phase(Phase::Searching));
                let store = self.single_store.as_ref().expect("just set");
                let hits = store
                    .retrieve(&req.query, req.top_k)
                    .map_err(|e| e.to_string())?;
                Ok(hits
                    .into_iter()
                    .map(|h| ResultRow {
                        project_root: None,
                        project_name: String::new(),
                        id: h.id,
                        content: h.content,
                        score: h.score,
                        date_epoch: 0,
                    })
                    .collect())
            }
        }
    }
}

/// Field-wise sum of two [`Reclaimed`] totals (its own `add` is private).
fn add_reclaimed(a: Reclaimed, b: Reclaimed) -> Reclaimed {
    Reclaimed {
        projects: a.projects + b.projects,
        transcripts: a.transcripts + b.transcripts,
        records: a.records + b.records,
        bytes: a.bytes + b.bytes,
    }
}

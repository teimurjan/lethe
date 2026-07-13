//! Production memory store — port of `research_playground/lethe_reference/lethe/memory_store.py`.
//!
//! Layers:
//! * `MemoryDb` (DuckDB) for entry rows + clustered RIF persistence +
//!   embedding BLOBs.
//! * `FlatIp` (ndarray) for dense top-k search.
//! * `BM25Okapi` for sparse top-k search; both fused via `rrf::rrf_merge`.
//! * `Reranker` (cross-encoder) reorders the top candidates.
//! * `RifConfig` + `ClusteredSuppressionState` apply the per-cluster
//!   suppression penalty before reranking and update it after.
//!
//! Lifecycle: `add()` embeds, dedupes (hash + cosine), inserts, and
//! marks structure dirty. `retrieve()` runs the full pipeline and
//! advances tier transitions. `save()` persists; with structure-dirty
//! tracking it skips the FAISS-equivalent rebuild when only retrieve
//! happened (PR #15 contract).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayView1};

use crate::bm25::BM25Okapi;
use crate::db::MemoryDb;
use crate::dedup::is_near_duplicate;
use crate::encoders::{BiEncoder, CrossEncoder};
use crate::entry::{MemoryEntry, Tier};
use crate::faiss_flat::FlatIp;
use crate::kmeans::{assign_cluster, build_clusters};
use crate::rif::{
    apply_suppression_penalty, update_suppression, ClusteredSuppressionState, CompetitorRow,
    RifConfig,
};
use crate::rrf::rrf_merge;
use crate::tokenize::tokenize_bm25;

const DEFAULT_K_SHALLOW: usize = 30;
const DEFAULT_K_DEEP: usize = 100;
const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 4.0;
const DEFAULT_DEDUP_THRESHOLD: f32 = 0.95;

const TIER_GC_RETRIEVALS: i64 = 3;
const TIER_MEMORY_AFFINITY: f32 = 0.65;
const TIER_MEMORY_RETRIEVALS: i64 = 5;
const APOPTOSIS_AFFINITY: f32 = 0.15;
const APOPTOSIS_IDLE_STEPS: i64 = 1000;

/// Configuration mirror of the Python `MemoryStore.__init__` kwargs.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub dim: usize,
    pub k_shallow: usize,
    pub k_deep: usize,
    pub confidence_threshold: f32,
    pub dedup_threshold: f32,
    pub rif: RifConfig,
    /// Open the underlying DuckDB read-only and turn `save()` into a
    /// no-op. Lets multiple processes share the same index for recall
    /// without fighting over the writer lock.
    pub read_only: bool,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            dim: 384,
            k_shallow: DEFAULT_K_SHALLOW,
            k_deep: DEFAULT_K_DEEP,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            dedup_threshold: DEFAULT_DEDUP_THRESHOLD,
            rif: RifConfig::default(),
            read_only: false,
        }
    }
}

/// One ranked retrieval result.
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: String,
    pub content: String,
    pub score: f32,
}

/// Outcome of a [`MemoryStore::dedupe`] pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct DedupeReport {
    /// Entries examined.
    pub scanned: usize,
    /// Near-duplicate groups found (size > 1).
    pub groups: usize,
    /// Entries absorbed into a canonical, i.e. `Σ (group_size − 1)`.
    pub absorbed: usize,
}

#[derive(Clone)]
pub struct MemoryStore {
    pub path: PathBuf,
    config: StoreConfig,
    bi_encoder: Option<Arc<BiEncoder>>,
    cross_encoder: Option<Arc<CrossEncoder>>,
    /// `duckdb::Connection` is `Send` but not `Sync`; wrap in a Mutex
    /// so a `MemoryStore` can be cloned and used across threads (the
    /// UnionStore code path opens many at once).
    db: Arc<Mutex<MemoryDb>>,
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for MemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryStore")
            .field("path", &self.path)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

struct Inner {
    flat: FlatIp,
    bm25: Option<BM25Okapi>,
    entries: HashMap<String, MemoryEntry>,
    /// Insertion-order id list — keeps FAISS-equivalent row indices
    /// stable as long as no add/delete happens between rebuilds.
    ids: Vec<String>,
    embeddings: HashMap<String, Array1<f32>>,
    step: i64,
    cluster_state: Option<ClusteredSuppressionState>,
    cluster_centroids: Option<Array2<f32>>,
    query_emb_buffer: Vec<Array1<f32>>,
    min_cluster_queries: usize,
    structure_dirty: bool,
    defer_rebuild: bool,
}

impl MemoryStore {
    /// Open or create a store at `path` (a directory; the DuckDB lives
    /// at `<path>/lethe.duckdb`). The encoder fields are optional so
    /// tests can mock them; `add` and `retrieve` will return errors
    /// when called without the corresponding encoder configured.
    #[allow(clippy::too_many_lines)] // hydration is one straight-line operation
    pub fn open(
        path: impl AsRef<Path>,
        bi_encoder: Option<Arc<BiEncoder>>,
        cross_encoder: Option<Arc<CrossEncoder>>,
        config: StoreConfig,
    ) -> Result<Self, crate::Error> {
        let path = path.as_ref().to_path_buf();
        if !config.read_only {
            std::fs::create_dir_all(&path)?;
        }
        let raw_db = MemoryDb::open_with_mode(path.join("lethe.duckdb"), config.read_only)?;
        let step = raw_db.get_stat("step", "0")?.parse::<i64>().unwrap_or(0);
        let db = Arc::new(Mutex::new(raw_db));

        let cluster_state = if config.rif.n_clusters > 0 {
            Some(ClusteredSuppressionState::new())
        } else {
            None
        };
        let min_cluster_queries = (config.rif.n_clusters as usize * 10).max(30);

        let mut inner = Inner {
            flat: FlatIp::new(config.dim),
            bm25: None,
            entries: HashMap::new(),
            ids: Vec::new(),
            embeddings: HashMap::new(),
            step,
            cluster_state,
            cluster_centroids: None,
            query_emb_buffer: Vec::new(),
            min_cluster_queries,
            structure_dirty: true,
            defer_rebuild: false,
        };

        // Hydrate from disk. Embeddings are stored as DuckDB BLOBs in
        // the `entry_embeddings` table — Rust-native, no dependence on
        // numpy semantics.
        let (rows, emb_rows, suppression_state, centroids) = {
            let g = db.lock().unwrap();
            let rows = g.load_all_entries()?;
            let emb_rows = g.load_all_embeddings()?;
            let mut state_in: Option<crate::rif::ClusteredSnapshot> = None;
            if config.rif.n_clusters > 0 {
                let (scores, last) = g.load_cluster_suppression()?;
                if !scores.is_empty() {
                    state_in = Some(crate::rif::ClusteredSnapshot {
                        scores,
                        last_updated: last,
                    });
                }
            }
            let centroids = if config.rif.n_clusters > 0 {
                g.load_cluster_centroids()?
            } else {
                None
            };
            (rows, emb_rows, state_in, centroids)
        };
        let emb_map: HashMap<String, Array1<f32>> = emb_rows.into_iter().collect();
        for r in rows {
            let Some(emb) = emb_map.get(&r.id) else {
                continue;
            };
            let emb = emb.clone();
            let entry = MemoryEntry {
                id: r.id.clone(),
                content: r.content,
                embedding: emb.clone(),
                session_id: r.session_id,
                turn_idx: r.turn_idx,
                affinity: r.affinity,
                retrieval_count: r.retrieval_count,
                generation: 0,
                last_retrieved_step: r.last_retrieved_step,
                tier: r.tier,
                suppression: r.suppression,
            };
            inner.embeddings.insert(r.id.clone(), emb);
            inner.ids.push(r.id.clone());
            inner.entries.insert(r.id, entry);
        }
        if !inner.ids.is_empty() {
            inner.rebuild_index_inplace(config.dim);
            inner.structure_dirty = false;
        }

        if let Some(state) = inner.cluster_state.as_mut() {
            if let Some(snap) = suppression_state {
                state.restore(snap);
            }
            if let Some(c) = centroids {
                if c.ncols() == config.dim {
                    inner.cluster_centroids = Some(c);
                }
            }
        }

        Ok(Self {
            path,
            config,
            bi_encoder,
            cross_encoder,
            db,
            inner: Arc::new(std::sync::Mutex::new(inner)),
        })
    }

    /// Number of live entries (excluding apoptotic).
    #[must_use]
    pub fn size(&self) -> usize {
        self.inner.lock().map(|g| g.entries.len()).unwrap_or(0)
    }

    /// Snapshot of currently-loaded entry ids. Used by markdown
    /// `reindex` to compute add/unchanged/remove deltas.
    #[must_use]
    pub fn live_ids(&self) -> std::collections::HashSet<String> {
        self.inner
            .lock()
            .map(|g| g.entries.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Read-only access to the bi-encoder. Used by callers that batch
    /// many embeddings before calling [`add_with_embedding`] in a loop
    /// (e.g. `markdown_store::reindex`).
    #[must_use]
    pub fn bi_encoder(&self) -> Option<&Arc<BiEncoder>> {
        self.bi_encoder.as_ref()
    }

    /// Add a new memory entry. Returns the id, or `None` when the
    /// content was deduped (exact-hash or near-cosine match against an
    /// existing longer entry). Mirrors `MemoryStore.add`.
    pub fn add(
        &self,
        content: &str,
        entry_id: Option<&str>,
        session_id: &str,
        turn_idx: i64,
    ) -> Result<Option<String>, crate::Error> {
        let bi = self
            .bi_encoder
            .as_ref()
            .ok_or(crate::Error::NotInitialized("bi_encoder required for add"))?;
        // Exact hash dedup before paying for the encoder.
        if self.db.lock().unwrap().has_content_hash(content)? {
            return Ok(None);
        }
        let emb = bi.encode(content)?;
        self.add_with_embedding(content, emb, entry_id, session_id, turn_idx)
    }

    /// Insert an entry whose embedding is already computed. Skips the
    /// bi-encoder call so callers can batch many encodings into one
    /// ORT inference launch (`encode_batch`) and then loop here.
    pub fn add_with_embedding(
        &self,
        content: &str,
        emb: Array1<f32>,
        entry_id: Option<&str>,
        session_id: &str,
        turn_idx: i64,
    ) -> Result<Option<String>, crate::Error> {
        if self.db.lock().unwrap().has_content_hash(content)? {
            return Ok(None);
        }
        if emb.len() != self.config.dim {
            return Err(crate::Error::DimensionMismatch {
                expected: self.config.dim,
                actual: emb.len(),
            });
        }
        let mut inner = self.inner.lock().unwrap();
        // Near-duplicate cosine check.
        if !inner.embeddings.is_empty() {
            let stacked = stack_embeddings(&inner.ids, &inner.embeddings);
            if let Some(idx) =
                is_near_duplicate(emb.view(), stacked.view(), self.config.dedup_threshold)
            {
                let existing_id = inner.ids[idx].clone();
                let existing = inner
                    .entries
                    .get(&existing_id)
                    .expect("near-duplicate id must be live");
                if content.len() <= existing.content.len() {
                    return Ok(None);
                }
                // New content is longer: replace existing.
                {
                    let g = self.db.lock().unwrap();
                    g.delete_entry(&existing_id)?;
                    g.delete_embedding(&existing_id)?;
                }
                inner.entries.remove(&existing_id);
                inner.embeddings.remove(&existing_id);
                inner.ids.retain(|i| i != &existing_id);
            }
        }

        let id = entry_id.map_or_else(short_uuid, str::to_owned);

        let entry = MemoryEntry::new(&id, content, emb.view(), session_id, turn_idx)?;
        {
            let g = self.db.lock().unwrap();
            g.insert_entry(&entry)?;
            g.save_embedding(&id, entry.embedding.view())?;
        }
        inner.entries.insert(id.clone(), entry.clone());
        inner.embeddings.insert(id.clone(), entry.embedding.clone());
        inner.ids.push(id.clone());
        inner.structure_dirty = true;
        if !inner.defer_rebuild {
            inner.rebuild_index_inplace(self.config.dim);
        }
        Ok(Some(id))
    }

    /// Remove an entry by id. Returns whether it existed.
    pub fn delete(&self, id: &str) -> Result<bool, crate::Error> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.entries.contains_key(id) {
            return Ok(false);
        }
        {
            let g = self.db.lock().unwrap();
            g.delete_entry(id)?;
            g.delete_embedding(id)?;
        }
        inner.entries.remove(id);
        inner.embeddings.remove(id);
        inner.ids.retain(|i| i != id);
        inner.structure_dirty = true;
        if !inner.defer_rebuild {
            inner.rebuild_index_inplace(self.config.dim);
        }
        Ok(true)
    }

    /// Retrieve the top-`k` most relevant entries to `query`.
    ///
    /// Linear pipeline (encode → hybrid → suppress → rerank → deep
    /// pass → RIF update → tier transitions). Lives as one function
    /// because every step's output feeds the next; splitting helps
    /// nothing here.
    #[allow(clippy::too_many_lines)]
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Hit>, crate::Error> {
        let bi = self
            .bi_encoder
            .as_ref()
            .ok_or(crate::Error::NotInitialized(
                "bi_encoder required for retrieve",
            ))?;
        let query_emb = bi.encode(query)?;

        let cfg = self.config.clone();
        let mut inner = self.inner.lock().unwrap();
        if inner.entries.is_empty() {
            return Ok(Vec::new());
        }

        // Query buffer for cluster build (only while centroids unbuilt).
        if inner.cluster_state.is_some() && inner.cluster_centroids.is_none() {
            inner.query_emb_buffer.push(query_emb.clone());
        }
        // Lazily build clusters once the buffer is large enough.
        if inner.cluster_state.is_some() && inner.cluster_centroids.is_none() {
            let buf_len = inner.query_emb_buffer.len();
            if buf_len >= inner.min_cluster_queries {
                let stacked = stack_embeddings_from_vec(&inner.query_emb_buffer);
                let centroids = build_clusters(stacked.view(), cfg.rif.n_clusters as usize);
                if !centroids.is_empty() {
                    inner.cluster_centroids = Some(centroids);
                }
            }
        }

        // Determine effective suppression scope.
        let (cluster_id, suppression, last_updated) = if let (Some(state), Some(centroids)) = (
            inner.cluster_state.as_ref(),
            inner.cluster_centroids.as_ref(),
        ) {
            let cid = assign_cluster(query_emb.view(), centroids.view()) as u32;
            (
                Some(cid),
                state.cluster_scores(cid),
                state.cluster_last_updated(cid),
            )
        } else {
            let mut sup: HashMap<String, f32> = HashMap::new();
            let mut last: HashMap<String, i64> = HashMap::new();
            for (eid, entry) in &inner.entries {
                sup.insert(eid.clone(), entry.suppression);
                last.insert(eid.clone(), entry.last_retrieved_step);
            }
            (None, sup, last)
        };

        // Shallow hybrid search.
        let raw = hybrid_scored(&inner, query_emb.view(), query, cfg.k_shallow);
        let raw_filtered: Vec<(String, f32)> = raw
            .into_iter()
            .filter(|(eid, _)| inner.entries.contains_key(eid))
            .collect();
        let adjusted = apply_suppression_penalty(&raw_filtered, &suppression, cfg.rif.alpha);
        let candidate_ids: Vec<String> = adjusted
            .iter()
            .take(cfg.k_shallow)
            .map(|(e, _)| e.clone())
            .collect();

        let scored = self.rerank_candidates(&inner, query, &candidate_ids)?;
        let mut scored = scored;
        let scores: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();

        // Adaptive deep pass.
        if needs_deep_search(&scores, cfg.confidence_threshold) {
            let deep_raw = hybrid_scored(&inner, query_emb.view(), query, cfg.k_deep);
            let deep_filtered: Vec<(String, f32)> = deep_raw
                .into_iter()
                .filter(|(eid, _)| inner.entries.contains_key(eid))
                .collect();
            let deep_adjusted =
                apply_suppression_penalty(&deep_filtered, &suppression, cfg.rif.alpha);
            let already: HashSet<&str> = candidate_ids.iter().map(String::as_str).collect();
            let new_ids: Vec<String> = deep_adjusted
                .iter()
                .filter(|(e, _)| !already.contains(e.as_str()) && inner.entries.contains_key(e))
                .map(|(e, _)| e.clone())
                .collect();
            if !new_ids.is_empty() {
                let mut new_scored = self.rerank_candidates(&inner, query, &new_ids)?;
                scored.append(&mut new_scored);
                scored.sort_by(|(a_id, a), (b_id, b)| {
                    b.partial_cmp(a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a_id.cmp(b_id))
                });
            }
        }

        // RIF state update.
        inner.step += 1;
        let step = inner.step;
        let mut winner_ids: HashSet<String> = HashSet::new();
        for (eid, score) in scored.iter().take(k) {
            if let Some(entry) = inner.entries.get_mut(eid) {
                winner_ids.insert(eid.clone());
                entry.retrieval_count += 1;
                entry.last_retrieved_step = step;
                let norm = 1.0_f32 / (1.0 + (-*score).exp());
                entry.affinity = 0.8 * entry.affinity + 0.2 * norm;
            }
        }
        let rank_lookup: HashMap<String, usize> = adjusted
            .iter()
            .enumerate()
            .map(|(r, (e, _))| (e.clone(), r))
            .collect();
        let xenc_lookup: HashMap<String, f32> =
            scored.iter().map(|(e, s)| (e.clone(), *s)).collect();
        let xenc_rank_lookup: HashMap<String, usize> = scored
            .iter()
            .enumerate()
            .map(|(r, (e, _))| (e.clone(), r))
            .collect();
        let fallback_rank = scored.len();
        let competitors: Vec<CompetitorRow> = candidate_ids
            .iter()
            .map(|eid| CompetitorRow {
                eid: eid.clone(),
                initial_rank: rank_lookup.get(eid).copied().unwrap_or(adjusted.len()),
                xenc_rank: xenc_rank_lookup.get(eid).copied().unwrap_or(fallback_rank),
                xenc_score: xenc_lookup.get(eid).copied().unwrap_or(0.0),
            })
            .collect();
        let rif_updates = update_suppression(
            &winner_ids,
            &competitors,
            &suppression,
            candidate_ids.len(),
            &cfg.rif,
            step,
            &last_updated,
        );
        if let (Some(cid), Some(state)) = (cluster_id, inner.cluster_state.as_mut()) {
            state.update_cluster(cid, &rif_updates, step);
        } else {
            for (eid, new_supp) in &rif_updates {
                if let Some(entry) = inner.entries.get_mut(eid) {
                    entry.suppression = *new_supp;
                }
            }
        }

        // Tier transitions.
        for entry in inner.entries.values_mut() {
            if entry.tier == Tier::Naive && entry.retrieval_count >= TIER_GC_RETRIEVALS {
                entry.tier = Tier::Gc;
            } else if entry.tier == Tier::Gc
                && entry.affinity >= TIER_MEMORY_AFFINITY
                && entry.retrieval_count >= TIER_MEMORY_RETRIEVALS
            {
                entry.tier = Tier::Memory;
            }
            if !matches!(entry.tier, Tier::Memory | Tier::Apoptotic)
                && entry.affinity < APOPTOSIS_AFFINITY
                && step - entry.last_retrieved_step > APOPTOSIS_IDLE_STEPS
            {
                entry.tier = Tier::Apoptotic;
            }
        }

        // Materialize hits.
        let hits: Vec<Hit> = scored
            .into_iter()
            .take(k)
            .filter_map(|(eid, score)| {
                inner.entries.get(&eid).map(|e| Hit {
                    id: eid.clone(),
                    content: e.content.clone(),
                    score,
                })
            })
            .collect();
        Ok(hits)
    }

    /// Persist mutable state. Embeddings are written incrementally
    /// during `add`/`delete` (DuckDB BLOB), so save() only flushes
    /// per-retrieve metadata and resets the structure-dirty flag.
    /// No-op when the store was opened read-only.
    pub fn save(&self) -> Result<(), crate::Error> {
        if self.config.read_only {
            return Ok(());
        }
        let mut inner = self.inner.lock().unwrap();
        inner.structure_dirty = false;
        let entries: Vec<&MemoryEntry> = inner.entries.values().collect();
        let g = self.db.lock().unwrap();
        g.batch_update_entries(&entries)?;
        g.set_stat("step", &inner.step.to_string())?;
        if let Some(state) = inner.cluster_state.as_ref() {
            let snap = state.snapshot();
            g.save_cluster_suppression(&snap.scores, &snap.last_updated)?;
        }
        if let Some(centroids) = inner.cluster_centroids.as_ref() {
            g.save_cluster_centroids(centroids)?;
        }
        Ok(())
    }

    pub fn close(self) -> Result<(), crate::Error> {
        self.save()?;
        Ok(())
    }

    /// Run a closure with the per-add rebuild deferred. The closure
    /// can call `add` / `delete` repeatedly; we rebuild the in-memory
    /// indexes once on successful exit. Mirrors the Python
    /// `MemoryStore.bulk_add()` context manager.
    pub fn bulk_add<F, T>(&self, f: F) -> Result<T, crate::Error>
    where
        F: FnOnce() -> Result<T, crate::Error>,
    {
        {
            let mut g = self.inner.lock().unwrap();
            g.defer_rebuild = true;
        }
        let result = f();
        {
            let mut g = self.inner.lock().unwrap();
            g.defer_rebuild = false;
            if !g.entries.is_empty() {
                g.rebuild_index_inplace(self.config.dim);
            }
        }
        result
    }

    /// Borrow the DB mutex briefly. `lethe expand`-style content
    /// lookups call this and use `get_content`. The lock is held for
    /// the duration of `f`'s execution.
    pub fn with_db<R>(&self, f: impl FnOnce(&MemoryDb) -> R) -> R {
        let g = self.db.lock().unwrap();
        f(&g)
    }

    /// Stamp the current on-disk index format. Call after a full
    /// (re)build so a later `ensure_index_format` recognizes the index as
    /// current-format. Stamping only here (never on a plain open) keeps a
    /// stale populated index detectable until it is actually rebuilt.
    /// No-op when read-only.
    pub fn mark_index_format(&self) -> Result<(), crate::Error> {
        if self.config.read_only {
            return Ok(());
        }
        self.with_db(|db| {
            db.set_stat(
                crate::db::INDEX_FORMAT_KEY,
                &crate::db::INDEX_FORMAT.to_string(),
            )
        })
    }

    /// Bulk-load pre-embedded entries into memory only — the DuckDB file
    /// is left untouched, so nothing is persisted and state is lost on
    /// drop. For ephemeral, throwaway stores (benchmarks, offline
    /// experiments) over a large corpus, where a per-add similarity scan
    /// would be quadratic and a full DB write would dominate. Each item
    /// is `(id, content, embedding)`; the embedding is L2-normalized on
    /// construction. Rebuilds the retrieval index once. Duplicate ids
    /// overwrite.
    pub fn ingest_ephemeral(
        &self,
        items: impl IntoIterator<Item = (String, String, Array1<f32>)>,
    ) -> Result<(), crate::Error> {
        let mut inner = self.inner.lock().unwrap();
        for (id, content, emb) in items {
            if emb.len() != self.config.dim {
                return Err(crate::Error::DimensionMismatch {
                    expected: self.config.dim,
                    actual: emb.len(),
                });
            }
            let entry = MemoryEntry::new(&id, &content, emb.view(), "", 0)?;
            inner.embeddings.insert(id.clone(), entry.embedding.clone());
            inner.ids.push(id.clone());
            inner.entries.insert(id, entry);
        }
        inner.structure_dirty = true;
        inner.rebuild_index_inplace(self.config.dim);
        Ok(())
    }

    /// Absorbed chunk ids from prior `dedupe` runs. Transcript sync
    /// treats these as already-present so a rewritten (dirty) transcript
    /// doesn't resurrect a merged-away turn.
    #[must_use]
    pub fn aliased_ids(&self) -> HashSet<String> {
        self.with_db(|db| db.aliased_ids().unwrap_or_default())
    }

    /// Offline near-duplicate compaction (SemDeDup-style). Clusters the
    /// **entry** embeddings, unions within each cluster any pair with
    /// cosine ≥ `threshold`, elects a canonical per group (most-retrieved,
    /// then longest content), merges RIF metadata into it, and deletes the
    /// rest — recording each absorbed id in the alias table so a later
    /// re-index can't resurrect it. `dry_run` reports groups without
    /// mutating anything.
    #[allow(clippy::too_many_lines)] // one straight-line compaction pass
    pub fn dedupe(&self, threshold: f32, dry_run: bool) -> Result<DedupeReport, crate::Error> {
        let cfg = self.config.clone();
        if cfg.read_only && !dry_run {
            return Err(crate::Error::NotInitialized(
                "dedupe requires a writable index",
            ));
        }
        let mut inner = self.inner.lock().unwrap();
        let n = inner.ids.len();
        let mut report = DedupeReport {
            scanned: n,
            ..Default::default()
        };
        if n < 2 {
            return Ok(report);
        }

        // Entry-embedding matrix in `ids` order (row i ↔ ids[i]).
        let ids = inner.ids.clone();
        let stacked = stack_embeddings(&ids, &inner.embeddings);

        // Cluster the ENTRY embeddings — RIF centroids are built from
        // *query* embeddings, so they're the wrong granularity here.
        // Empty centroids (n_clusters == 0 or too few rows) → one bucket
        // = full pairwise, which is correct if slower.
        let centroids = build_clusters(stacked.view(), cfg.rif.n_clusters as usize);
        let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        if centroids.is_empty() {
            buckets.insert(0, (0..n).collect());
        } else {
            for i in 0..n {
                let c = assign_cluster(stacked.row(i), centroids.view());
                buckets.entry(c).or_default().push(i);
            }
        }

        // Union near-duplicate pairs within each cluster only — O(Σ nᵢ²).
        let mut parent: Vec<usize> = (0..n).collect();
        for rows in buckets.values() {
            for (a, &i) in rows.iter().enumerate() {
                for &j in &rows[a + 1..] {
                    if stacked.row(i).dot(&stacked.row(j)) >= threshold {
                        uf_union(&mut parent, i, j);
                    }
                }
            }
        }

        // Bucket row indices by union-find root.
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let r = uf_find(&mut parent, i);
            groups.entry(r).or_default().push(i);
        }

        let mut removed: HashSet<String> = HashSet::new();
        for members in groups.values() {
            if members.len() < 2 {
                continue;
            }
            report.groups += 1;
            report.absorbed += members.len() - 1;
            if dry_run {
                continue;
            }

            // Canonical: max retrieval_count, then longest content, then
            // smallest id (stable across runs).
            let canonical = *members
                .iter()
                .max_by(|&&x, &&y| {
                    let ex = &inner.entries[&ids[x]];
                    let ey = &inner.entries[&ids[y]];
                    ex.retrieval_count
                        .cmp(&ey.retrieval_count)
                        .then_with(|| ex.content.len().cmp(&ey.content.len()))
                        .then_with(|| ey.id.cmp(&ex.id))
                })
                .expect("group is non-empty");
            let canonical_id = ids[canonical].clone();

            // Fold RIF metadata across the whole group into the canonical.
            let mut sum_rc = 0_i64;
            let mut max_aff = f32::MIN;
            let mut min_supp = f32::MAX;
            let mut max_last = i64::MIN;
            for &m in members {
                let e = &inner.entries[&ids[m]];
                sum_rc += e.retrieval_count;
                max_aff = max_aff.max(e.affinity);
                min_supp = min_supp.min(e.suppression);
                max_last = max_last.max(e.last_retrieved_step);
            }

            // Delete losers (db + memory), alias them to the canonical.
            for &m in members {
                if m == canonical {
                    continue;
                }
                let loser_id = ids[m].clone();
                {
                    let g = self.db.lock().unwrap();
                    g.delete_entry(&loser_id)?;
                    g.delete_embedding(&loser_id)?;
                    g.insert_alias(&loser_id, &canonical_id)?;
                }
                inner.entries.remove(&loser_id);
                inner.embeddings.remove(&loser_id);
                removed.insert(loser_id);
            }

            // Apply merged metadata and persist the canonical.
            {
                let c = inner
                    .entries
                    .get_mut(&canonical_id)
                    .expect("canonical stays live");
                c.retrieval_count = sum_rc;
                c.affinity = max_aff;
                c.suppression = min_supp;
                c.last_retrieved_step = max_last;
            }
            let canon = inner.entries[&canonical_id].clone();
            self.db.lock().unwrap().insert_entry(&canon)?;
        }

        if !removed.is_empty() {
            inner.ids.retain(|id| !removed.contains(id));
            inner.structure_dirty = true;
            inner.rebuild_index_inplace(cfg.dim);
        }
        Ok(report)
    }

    /// Read-only retrieve used by `UnionStore` cross-project gather.
    ///
    /// Returns top-`k` hits ranked by hybrid (BM25 + dense) RRF score
    /// with the suppression penalty applied. Skips cross-encoder
    /// rerank, the adaptive deep pass, RIF state mutation, and tier
    /// transitions — the union path runs a single batched rerank over
    /// the merged pool, and `--all` never calls `save()` so per-project
    /// state mutation would be discarded anyway.
    ///
    /// Takes a precomputed `query_emb` so the caller can amortize the
    /// bi-encoder forward across all projects.
    pub fn retrieve_shallow(
        &self,
        query_emb: ArrayView1<'_, f32>,
        query: &str,
        k: usize,
    ) -> Result<Vec<Hit>, crate::Error> {
        let cfg = self.config.clone();
        let inner = self.inner.lock().unwrap();
        if inner.entries.is_empty() {
            return Ok(Vec::new());
        }

        // Reuse existing cluster centroids if present; do not update
        // the query buffer or build new clusters here (that's a
        // mutation reserved for the owning project's full `retrieve`).
        let suppression: HashMap<String, f32> = if let (Some(state), Some(centroids)) = (
            inner.cluster_state.as_ref(),
            inner.cluster_centroids.as_ref(),
        ) {
            let cid = assign_cluster(query_emb, centroids.view()) as u32;
            state.cluster_scores(cid)
        } else {
            let mut sup = HashMap::with_capacity(inner.entries.len());
            for (eid, entry) in &inner.entries {
                sup.insert(eid.clone(), entry.suppression);
            }
            sup
        };

        let raw = hybrid_scored(&inner, query_emb, query, k);
        let raw_filtered: Vec<(String, f32)> = raw
            .into_iter()
            .filter(|(eid, _)| inner.entries.contains_key(eid))
            .collect();
        let adjusted = apply_suppression_penalty(&raw_filtered, &suppression, cfg.rif.alpha);

        let hits: Vec<Hit> = adjusted
            .into_iter()
            .take(k)
            .filter_map(|(eid, score)| {
                inner.entries.get(&eid).map(|e| Hit {
                    id: eid,
                    content: e.content.clone(),
                    score,
                })
            })
            .collect();
        Ok(hits)
    }

    fn rerank_candidates(
        &self,
        inner: &Inner,
        query: &str,
        ids: &[String],
    ) -> Result<Vec<(String, f32)>, crate::Error> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        // Build (id, stripped_content) pairs. `embed_content` drops the
        // anchor + heading lines so the cross-encoder scores the actual
        // turn text, matching what the bi-encoder embedded. Idempotent
        // for entries already stored stripped (legacy markdown).
        let pairs: Vec<(String, String)> = ids
            .iter()
            .filter_map(|eid| {
                inner.entries.get(eid).map(|e| {
                    (
                        eid.clone(),
                        crate::markdown_store::embed_content(&e.content),
                    )
                })
            })
            .collect();
        let Some(xenc) = &self.cross_encoder else {
            // No cross-encoder: rank by RRF position from the gather
            // step. Tests use this path.
            return Ok(pairs
                .into_iter()
                .enumerate()
                .map(|(i, (id, _))| (id, -(i as f32)))
                .collect());
        };
        let xpairs: Vec<(&str, &str)> = pairs.iter().map(|(_, c)| (query, c.as_str())).collect();
        let scores = xenc.predict(&xpairs)?;
        let mut scored: Vec<(String, f32)> = pairs
            .iter()
            .zip(scores)
            .map(|((eid, _), s)| (eid.clone(), s))
            .collect();
        scored.sort_by(|(a_id, a), (b_id, b)| {
            b.partial_cmp(a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a_id.cmp(b_id))
        });
        Ok(scored)
    }
}

impl Inner {
    fn rebuild_index_inplace(&mut self, dim: usize) {
        if self.ids.is_empty() {
            self.flat = FlatIp::new(dim);
            self.bm25 = None;
            return;
        }
        let stacked = stack_embeddings(&self.ids, &self.embeddings);
        let mut flat = FlatIp::new(dim);
        let _ = flat.add_batch(stacked.view());
        self.flat = flat;
        let tokenized: Vec<Vec<String>> = self
            .ids
            .iter()
            .map(|eid| {
                self.entries
                    .get(eid)
                    .map(|e| tokenize_bm25(&crate::markdown_store::embed_content(&e.content)))
                    .unwrap_or_default()
            })
            .collect();
        self.bm25 = Some(BM25Okapi::new(&tokenized));
    }
}

/// Union-find find with path halving. Used by [`MemoryStore::dedupe`].
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Union-find union. Roots merge arbitrarily (grouping is set-membership;
/// canonical election is a separate, deterministic step).
fn uf_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

fn stack_embeddings(ids: &[String], map: &HashMap<String, Array1<f32>>) -> Array2<f32> {
    if ids.is_empty() {
        return Array2::<f32>::zeros((0, 0));
    }
    let dim = map[&ids[0]].len();
    let mut out = Array2::<f32>::zeros((ids.len(), dim));
    for (i, eid) in ids.iter().enumerate() {
        if let Some(v) = map.get(eid) {
            out.row_mut(i).assign(v);
        }
    }
    out
}

fn stack_embeddings_from_vec(buf: &[Array1<f32>]) -> Array2<f32> {
    if buf.is_empty() {
        return Array2::<f32>::zeros((0, 0));
    }
    let dim = buf[0].len();
    let mut out = Array2::<f32>::zeros((buf.len(), dim));
    for (i, v) in buf.iter().enumerate() {
        out.row_mut(i).assign(v);
    }
    out
}

/// BM25 + dense IP top-k merged via RRF. Returns `(eid, rrf_score)` pairs.
///
/// BM25 and dense top-k are computed concurrently via `rayon::join` —
/// they're independent and their costs are similar enough that the
/// joined wall time is roughly `max(bm25, dense)` instead of the sum.
fn hybrid_scored(
    inner: &Inner,
    query_emb: ArrayView1<'_, f32>,
    query: &str,
    k: usize,
) -> Vec<(String, f32)> {
    let (bm25_ids, dense_ids) = rayon::join(
        || -> Vec<String> {
            let Some(bm25) = inner.bm25.as_ref() else {
                return Vec::new();
            };
            let tokens = tokenize_bm25(query);
            if tokens.is_empty() {
                return Vec::new();
            }
            let scores = bm25.get_scores(&tokens);
            let order = crate::faiss_flat::top_k_desc(ndarray::ArrayView1::from(&scores), k);
            order
                .into_iter()
                .filter_map(|(i, _)| inner.ids.get(i).cloned())
                .collect()
        },
        || -> Vec<String> {
            inner
                .flat
                .search(query_emb, k)
                .map(|hits| {
                    hits.into_iter()
                        .filter_map(|(i, _)| inner.ids.get(i).cloned())
                        .collect()
                })
                .unwrap_or_default()
        },
    );
    rrf_merge(&[bm25_ids, dense_ids])
}

fn needs_deep_search(scores: &[f32], threshold: f32) -> bool {
    let Some(top) = scores.first() else {
        return true;
    };
    *top < threshold
}

fn short_uuid() -> String {
    // 12-char hex uuid stand-in (matches Python `str(uuid.uuid4())[:12]`
    // length and character set). Uses os-randomness so multiple Rust
    // processes don't collide.
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut hasher = sha2::Sha256::new();
    use sha2::Digest as _;
    hasher.update(nanos.to_le_bytes());
    hasher.update(std::process::id().to_le_bytes());
    // Mix in monotonic counter via static AtomicU64 so back-to-back
    // calls within the same nanosecond still differ.
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    hasher.update(n.to_le_bytes());
    let hex_digest = hex::encode(hasher.finalize());
    hex_digest[..12].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use tempfile::tempdir;

    /// Deterministic mock bi-encoder that hashes input text into a
    /// reproducible 16-D unit vector — same recipe as the Python
    /// `MockBiEncoder` in tests/conftest.py.
    fn mock_emb(text: &str, dim: usize) -> Array1<f32> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();
        let mut state = seed | 1;
        let mut v = Array1::<f32>::zeros(dim);
        for x in v.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert 64-bit state into a roughly-Gaussian f32.
            let bits = (state >> 32) as u32;
            *x = (bits as f32 / u32::MAX as f32) - 0.5;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.mapv_inplace(|x| x / n);
        }
        v
    }

    /// Tiny "bi-encoder" that bypasses ONNX so we can drive
    /// MemoryStore in unit tests without downloading a model. We
    /// don't have a trait abstraction over BiEncoder yet, so the test
    /// path stages embeddings via direct DB writes through a helper
    /// that calls `add` after manually inserting the embedding.
    /// In practice, MemoryStore.add requires a real bi_encoder, so
    /// the test instead exercises `retrieve` after manually wiring
    /// entries via the underlying DB and rebuilding the index.
    fn seed_entry(
        store: &MemoryStore,
        id: &str,
        content: &str,
        emb: Array1<f32>,
    ) -> Result<(), crate::Error> {
        let entry = MemoryEntry::new(id, content, emb.view(), "sess", 0)?;
        {
            let dbg = store.db.lock().unwrap();
            dbg.insert_entry(&entry)?;
            dbg.save_embedding(id, entry.embedding.view())?;
        }
        let mut g = store.inner.lock().unwrap();
        g.embeddings.insert(id.to_owned(), entry.embedding.clone());
        g.entries.insert(id.to_owned(), entry);
        g.ids.push(id.to_owned());
        g.structure_dirty = true;
        g.rebuild_index_inplace(store.config.dim);
        Ok(())
    }

    #[test]
    fn open_creates_directory_and_persists_step() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("store");
        let cfg = StoreConfig {
            dim: 16,
            ..Default::default()
        };
        {
            let s = MemoryStore::open(&path, None, None, cfg.clone()).unwrap();
            assert_eq!(s.size(), 0);
            // Force step bump via direct mutation.
            {
                let mut g = s.inner.lock().unwrap();
                g.step = 7;
            }
            s.save().unwrap();
        }
        let s2 = MemoryStore::open(&path, None, None, cfg).unwrap();
        let g = s2.inner.lock().unwrap();
        assert_eq!(g.step, 7);
    }

    #[test]
    fn retrieve_without_encoder_errors() {
        let dir = tempdir().unwrap();
        let s = MemoryStore::open(
            dir.path().join("store"),
            None,
            None,
            StoreConfig {
                dim: 16,
                ..Default::default()
            },
        )
        .unwrap();
        let res = s.retrieve("hi", 5);
        assert!(matches!(res, Err(crate::Error::NotInitialized(_))));
    }

    #[test]
    fn dedupe_merges_near_duplicates_and_writes_alias() {
        let dir = tempdir().unwrap();
        let cfg = StoreConfig {
            dim: 3,
            // Disable the write-time gate so we can plant a near-dup pair.
            dedup_threshold: 2.0,
            rif: RifConfig {
                n_clusters: 0,
                ..RifConfig::default()
            },
            ..Default::default()
        };
        let store = MemoryStore::open(dir.path().join("s"), None, None, cfg).unwrap();
        seed_entry(&store, "a", "alpha", ndarray::arr1(&[1.0, 0.0, 0.0])).unwrap();
        seed_entry(&store, "b", "alpha too", ndarray::arr1(&[0.999, 0.001, 0.0])).unwrap();
        seed_entry(&store, "c", "beta", ndarray::arr1(&[0.0, 1.0, 0.0])).unwrap();
        assert_eq!(store.size(), 3);

        // Dry run reports the group but changes nothing.
        let dry = store.dedupe(0.95, true).unwrap();
        assert_eq!((dry.groups, dry.absorbed), (1, 1));
        assert_eq!(store.size(), 3);
        assert!(store.aliased_ids().is_empty());

        // Real run merges the near-dup pair, keeps the distinct entry.
        let report = store.dedupe(0.95, false).unwrap();
        assert_eq!((report.groups, report.absorbed), (1, 1));
        assert_eq!(store.size(), 2);

        let live = store.live_ids();
        assert!(live.contains("c"), "distinct entry survives");
        assert!(live.contains("a") ^ live.contains("b"), "exactly one survives");
        let aliased = store.aliased_ids();
        assert_eq!(aliased.len(), 1, "loser is aliased to the survivor");
    }

    #[test]
    fn seeded_entries_round_trip_through_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("store");
        let cfg = StoreConfig {
            dim: 16,
            ..Default::default()
        };
        {
            let s = MemoryStore::open(&path, None, None, cfg.clone()).unwrap();
            for (id, text) in [("a", "alpha beta"), ("b", "gamma delta")] {
                seed_entry(&s, id, text, mock_emb(text, 16)).unwrap();
            }
            s.save().unwrap();
            assert_eq!(s.size(), 2);
        }
        let s2 = MemoryStore::open(&path, None, None, cfg).unwrap();
        assert_eq!(s2.size(), 2);
        let g = s2.inner.lock().unwrap();
        assert!(g.entries.contains_key("a"));
        assert!(g.entries.contains_key("b"));
    }

    #[test]
    fn delete_removes_from_memory_and_db() {
        let dir = tempdir().unwrap();
        let s = MemoryStore::open(
            dir.path().join("store"),
            None,
            None,
            StoreConfig {
                dim: 16,
                ..Default::default()
            },
        )
        .unwrap();
        seed_entry(&s, "a", "alpha", mock_emb("alpha", 16)).unwrap();
        seed_entry(&s, "b", "beta", mock_emb("beta", 16)).unwrap();
        assert_eq!(s.size(), 2);
        assert!(s.delete("a").unwrap());
        assert_eq!(s.size(), 1);
        assert!(!s.delete("a").unwrap());
    }

    #[test]
    fn save_is_noop_when_read_only() {
        // Seed an index in writable mode so the file exists.
        let dir = tempdir().unwrap();
        let path = dir.path().join("store");
        let writer_cfg = StoreConfig {
            dim: 16,
            ..Default::default()
        };
        {
            let s = MemoryStore::open(&path, None, None, writer_cfg).unwrap();
            seed_entry(&s, "a", "alpha", mock_emb("alpha", 16)).unwrap();
            s.save().unwrap();
        }
        // Reopen read-only; save() must succeed without trying to write.
        let ro_cfg = StoreConfig {
            dim: 16,
            read_only: true,
            ..Default::default()
        };
        let s = MemoryStore::open(&path, None, None, ro_cfg).unwrap();
        assert_eq!(s.size(), 1);
        s.save().unwrap();
    }
}

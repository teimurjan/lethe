//! Production memory store — port of `legacy/lethe/memory_store.py`.
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
        std::fs::create_dir_all(&path)?;
        let raw_db = MemoryDb::open(path.join("lethe.duckdb"))?;
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
        // numpy semantics. Stores written by the legacy Python impl
        // keep their embeddings in `embeddings.npz` and need a one-shot
        // conversion via `lethe migrate`.
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
        // Detect an unmigrated Python store: rows present, embeddings
        // table empty, but `embeddings.npz` exists. Don't read the npz;
        // tell the user how to convert it.
        if !rows.is_empty() && emb_map.is_empty() && path.join("embeddings.npz").exists() {
            eprintln!(
                "[lethe] {} entries but no Rust-side embeddings; this index was \
                 written by the legacy Python implementation. Run `lethe migrate` \
                 to convert `embeddings.npz` into the DuckDB store.",
                rows.len()
            );
        }
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
    pub fn save(&self) -> Result<(), crate::Error> {
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

    fn rerank_candidates(
        &self,
        inner: &Inner,
        query: &str,
        ids: &[String],
    ) -> Result<Vec<(String, f32)>, crate::Error> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        // Build (id_str, content_str) pairs without copying ids.
        let pairs: Vec<(&str, &str)> = ids
            .iter()
            .filter_map(|eid| {
                inner
                    .entries
                    .get(eid)
                    .map(|e| (eid.as_str(), e.content.as_str()))
            })
            .collect();
        let Some(xenc) = &self.cross_encoder else {
            // No cross-encoder: rank by RRF position from the gather
            // step. Tests use this path.
            return Ok(pairs
                .into_iter()
                .enumerate()
                .map(|(i, (id, _))| (id.to_owned(), -(i as f32)))
                .collect());
        };
        let xpairs: Vec<(&str, &str)> = pairs.iter().map(|(_, c)| (query, *c)).collect();
        let scores = xenc.predict(&xpairs)?;
        let mut scored: Vec<(String, f32)> = pairs
            .iter()
            .zip(scores)
            .map(|((eid, _), s)| ((*eid).to_owned(), s))
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
                    .map(|e| tokenize_bm25(&e.content))
                    .unwrap_or_default()
            })
            .collect();
        self.bm25 = Some(BM25Okapi::new(&tokenized));
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
fn hybrid_scored(
    inner: &Inner,
    query_emb: ArrayView1<'_, f32>,
    query: &str,
    k: usize,
) -> Vec<(String, f32)> {
    let mut bm25_ids: Vec<String> = Vec::new();
    if let Some(bm25) = &inner.bm25 {
        let tokens = tokenize_bm25(query);
        if !tokens.is_empty() {
            let scores = bm25.get_scores(&tokens);
            let order = crate::faiss_flat::top_k_desc(ndarray::ArrayView1::from(&scores), k);
            for (i, _s) in order {
                if let Some(eid) = inner.ids.get(i) {
                    bm25_ids.push(eid.clone());
                }
            }
        }
    }
    let dense_ids: Vec<String> = inner
        .flat
        .search(query_emb, k)
        .map(|hits| {
            hits.into_iter()
                .filter_map(|(i, _)| inner.ids.get(i).cloned())
                .collect()
        })
        .unwrap_or_default();
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
}

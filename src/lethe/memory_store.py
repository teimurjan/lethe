"""MemoryStore: production-ready memory for LLM agents.

Combines BM25 + dense vector hybrid retrieval with cross-encoder
reranking, adaptive search depth, deduplication, and tier lifecycle.
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lethe.db import MemoryDB
from lethe.dedup import content_hash, is_near_duplicate
from lethe.entry import MemoryEntry, Tier, create_entry
from lethe.reranker import Reranker
from lethe.rif import (
    ClusteredSuppressionState,
    RIFConfig,
    apply_suppression_penalty,
    assign_cluster,
    build_clusters,
    update_suppression,
)
from lethe.vectors import VectorIndex


class MemoryStore:
    """Self-improving memory store for LLM agents.

    Usage:
        store = MemoryStore("./memories")
        store.add("I prefer window seats", session_id="trip_chat")
        results = store.retrieve("travel preferences", k=5)
    """

    def __init__(
        self,
        path: str | Path,
        bi_encoder: Any = None,
        cross_encoder: Any = None,
        dim: int = 384,
        k_shallow: int = 30,
        k_deep: int = 100,
        confidence_threshold: float = 4.0,
        dedup_threshold: float = 0.95,
        rif_config: RIFConfig | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.bi_encoder = bi_encoder
        self.dim = dim
        self.k_shallow = k_shallow
        self.k_deep = k_deep
        self.dedup_threshold = dedup_threshold
        self.rif = rif_config or RIFConfig()

        # Storage layers
        self.db = MemoryDB(self.path / "lethe.duckdb")
        self.index = VectorIndex(dim=dim)
        self.reranker = Reranker(cross_encoder, confidence_threshold)

        # In-memory state
        self.entries: dict[str, MemoryEntry] = {}
        self._embeddings: dict[str, npt.NDArray[np.float32]] = {}
        self._step = int(self.db.get_stat("step", "0"))

        # Clustered RIF state (only used when rif.n_clusters > 0)
        self._cluster_state: ClusteredSuppressionState | None = (
            ClusteredSuppressionState() if self.rif.n_clusters > 0 else None
        )
        self._cluster_centroids: npt.NDArray[np.float32] | None = None
        self._cluster_dirty: bool = True
        # Query embeddings collected during retrieve() — used to build
        # clusters that reflect user intent (not entry content).
        self._query_emb_buffer: list[npt.NDArray[np.float32]] = []
        self._min_cluster_queries: int = max(self.rif.n_clusters * 10, 30)

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load entries from SQLite, rebuild indexes."""
        rows = self.db.load_all_entries()
        emb_path = self.path / "embeddings.npz"

        if emb_path.exists() and rows:
            data = np.load(str(emb_path), allow_pickle=True)
            saved_ids = list(data["ids"])
            saved_embs = data["embeddings"].astype(np.float32)
            emb_map = dict(zip(saved_ids, saved_embs))
        else:
            emb_map = {}

        ids, embs, contents = [], [], []
        for row in rows:
            eid = str(row["id"])
            content = str(row["content"])
            tier = Tier(str(row["tier"]))

            emb = emb_map.get(eid)
            if emb is None:
                continue  # skip entries without embeddings

            entry = MemoryEntry(
                id=eid, content=content,
                base_embedding=emb.copy(), embedding=emb.copy(),
                adapter=np.zeros(self.dim, dtype=np.float32),
                session_id=str(row.get("session_id", "")),
                turn_idx=int(row.get("turn_idx", 0)),
                affinity=float(row.get("affinity", 0.5)),
                retrieval_count=int(row.get("retrieval_count", 0)),
                last_retrieved_step=int(row.get("last_retrieved_step", 0)),
                tier=tier,
                suppression=float(row.get("suppression", 0.0)),
            )
            self.entries[eid] = entry
            self._embeddings[eid] = emb
            ids.append(eid)
            embs.append(emb)
            contents.append(content)

        if ids:
            self.index.build(ids, np.stack(embs), contents)

        # Rehydrate clustered RIF state if configured
        if self._cluster_state is not None:
            scores, last_updated = self.db.load_cluster_suppression()
            if scores:
                self._cluster_state.restore(scores, last_updated)
            centroids = self.db.load_cluster_centroids()
            if centroids is not None and centroids.shape[1] == self.dim:
                self._cluster_centroids = centroids
                self._cluster_dirty = False
            qbuf_path = self.path / "query_embeddings.npz"
            if qbuf_path.exists():
                qdata = np.load(str(qbuf_path), allow_pickle=True)
                self._query_emb_buffer = list(qdata["embeddings"])

    def add(
        self,
        content: str,
        entry_id: str | None = None,
        session_id: str = "",
        turn_idx: int = 0,
    ) -> str | None:
        """Add a memory entry. Returns entry ID, or None if deduplicated.

        Checks exact hash dedup and cosine near-duplicate before adding.
        """
        # Exact dedup
        if self.db.has_content_hash(content):
            return None

        # Embed
        if self.bi_encoder is None:
            raise ValueError("bi_encoder required for add()")
        emb = self.bi_encoder.encode(content, normalize_embeddings=True).astype(np.float32)

        # Near-duplicate check
        if self._embeddings:
            all_embs = np.stack(list(self._embeddings.values()))
            dup_idx = is_near_duplicate(emb, all_embs, self.dedup_threshold)
            if dup_idx is not None:
                existing_id = list(self._embeddings.keys())[dup_idx]
                existing = self.entries[existing_id]
                # Keep the longer one
                if len(content) <= len(existing.content):
                    return None  # new is shorter, skip
                # New is longer, replace existing
                self.db.delete_entry(existing_id)
                del self.entries[existing_id]
                del self._embeddings[existing_id]

        # Create entry
        eid = entry_id or str(uuid.uuid4())[:12]
        entry = create_entry(eid, content, emb, session_id, turn_idx)
        self.entries[eid] = entry
        self._embeddings[eid] = emb
        self.db.insert_entry(entry)

        # Rebuild index
        self._rebuild_index()
        return eid

    def _ensure_clusters(self) -> None:
        """Lazily build/rebuild cluster centroids over past query embeddings.

        Clusters reflect *user intent* — queries about travel land in the
        same cluster and share a suppression namespace. Clustering entry
        embeddings (content topics) is a much weaker proxy and gives ~3x
        less NDCG lift than query-based clusters in our benchmarks.

        Until enough queries have been seen (>= n_clusters), the store
        falls back to global suppression.
        """
        if self._cluster_state is None or self.rif.n_clusters <= 0:
            return
        if self._cluster_centroids is not None:
            return  # built once, frozen
        if len(self._query_emb_buffer) < self._min_cluster_queries:
            self._cluster_centroids = None
            return
        # Use the full buffer for the one-time build (more queries → better
        # centroids). After this, _cluster_dirty stays False forever.
        embs = np.stack(self._query_emb_buffer).astype(np.float32)
        self._cluster_centroids = build_clusters(embs, self.rif.n_clusters)
        self._cluster_dirty = False
        self._cluster_frozen = True

    def retrieve(
        self, query: str, k: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Retrieve top-k memories. Returns list of (id, content, score).

        Pipeline: BM25 + FAISS hybrid → RIF suppression filter →
                  cross-encoder rerank → adaptive depth → RIF update.
        """
        if self.bi_encoder is None:
            raise ValueError("bi_encoder required for retrieve()")

        query_emb = self.bi_encoder.encode(query, normalize_embeddings=True).astype(np.float32)

        # Feed query embedding into the cluster-building buffer.
        # Centroids are built once (when buffer reaches n_clusters) and then
        # frozen — rebuilding invalidates cluster IDs which breaks the
        # per-cluster suppression state.
        if self._cluster_state is not None:
            self._query_emb_buffer.append(query_emb.copy())

        self._ensure_clusters()
        if self._cluster_state is not None and self._cluster_centroids is not None:
            cluster_id: int | None = assign_cluster(query_emb, self._cluster_centroids)
            suppression_scores = self._cluster_state.get_cluster_scores(cluster_id)
            last_updated = self._cluster_state.get_cluster_last_updated(cluster_id)
        else:
            cluster_id = None
            suppression_scores = {
                eid: self.entries[eid].suppression for eid in self.entries
            }
            last_updated = {
                eid: self.entries[eid].last_retrieved_step for eid in self.entries
            }

        # Shallow pass: BM25 + vector hybrid with RRF scores
        raw_candidates = self.index.search_hybrid_scored(query_emb, query, self.k_shallow)
        raw_candidates = [(eid, s) for eid, s in raw_candidates if eid in self.entries]

        # Apply RIF suppression penalty before cross-encoder
        adjusted = apply_suppression_penalty(raw_candidates, suppression_scores, self.rif.alpha)
        candidate_ids = [eid for eid, _ in adjusted[:self.k_shallow]]

        candidates = [(eid, self.entries[eid].content) for eid in candidate_ids]

        # Cross-encoder rerank
        scored = self.reranker.rerank(query, candidates)
        scores = [s for _, s in scored]

        # Adaptive depth: if shallow confidence is low, go deeper
        if self.reranker.needs_deep_search(scores):
            deep_raw = self.index.search_hybrid_scored(query_emb, query, self.k_deep)
            deep_adjusted = apply_suppression_penalty(
                [(eid, s) for eid, s in deep_raw if eid in self.entries],
                suppression_scores, self.rif.alpha,
            )
            already = set(candidate_ids)
            new_ids = [eid for eid, _ in deep_adjusted if eid not in already and eid in self.entries]
            if new_ids:
                new_cands = [(eid, self.entries[eid].content) for eid in new_ids]
                new_scored = self.reranker.rerank(query, new_cands)
                scored = sorted(scored + new_scored, key=lambda x: x[1], reverse=True)

        # Identify winners and competitors for RIF update
        self._step += 1
        winner_ids = set()
        for eid, score in scored[:k]:
            if eid in self.entries:
                winner_ids.add(eid)
                entry = self.entries[eid]
                entry.retrieval_count += 1
                entry.last_retrieved_step = self._step
                norm_score = 1.0 / (1.0 + math.exp(-score))
                entry.affinity = 0.8 * entry.affinity + 0.2 * norm_score

        # RIF update: suppress competitors, reinforce winners
        rank_lookup = {eid: rank for rank, (eid, _) in enumerate(adjusted)}
        xenc_lookup = {eid: s for eid, s in scored}
        xenc_rank_lookup = {eid: rank for rank, (eid, _) in enumerate(scored)}
        fallback_rank = len(scored)
        if self.rif.use_rank_gap:
            competitor_data: list[Any] = [
                (
                    eid,
                    rank_lookup.get(eid, len(adjusted)),
                    xenc_rank_lookup.get(eid, fallback_rank),
                    xenc_lookup.get(eid, 0.0),
                )
                for eid in candidate_ids
            ]
        else:
            competitor_data = [
                (eid, rank_lookup.get(eid, len(adjusted)), xenc_lookup.get(eid, 0.0))
                for eid in candidate_ids
            ]
        rif_updates = update_suppression(
            winner_ids, competitor_data, suppression_scores,
            len(candidate_ids), self.rif, self._step, last_updated,
        )
        if cluster_id is not None and self._cluster_state is not None:
            self._cluster_state.update_cluster(cluster_id, rif_updates, self._step)
        else:
            for eid, new_supp in rif_updates.items():
                if eid in self.entries:
                    self.entries[eid].suppression = new_supp

        # Tier transitions
        self._check_tiers()

        # Return top-k
        return [(eid, self.entries[eid].content, score)
                for eid, score in scored[:k] if eid in self.entries]

    def decay(self) -> None:
        """Apply time decay to affinities. Call periodically."""
        factor = math.exp(-0.01)
        for entry in self.entries.values():
            if entry.tier == Tier.MEMORY:
                continue
            if self._step - entry.last_retrieved_step < 100:
                continue
            entry.affinity *= factor

    def save(self) -> None:
        """Persist state to disk."""
        # Save embeddings
        if self._embeddings:
            ids = list(self._embeddings.keys())
            embs = np.stack([self._embeddings[eid] for eid in ids])
            np.savez(str(self.path / "embeddings.npz"),
                     ids=np.array(ids), embeddings=embs)
        # Save FAISS index
        self.index.save(self.path)
        # Update entries in SQLite
        self.db.batch_update_entries(list(self.entries.values()))
        self.db.set_stat("step", str(self._step))
        # Clustered RIF state
        if self._cluster_state is not None:
            scores, last = self._cluster_state.snapshot()
            self.db.save_cluster_suppression(scores, last)
        if self._cluster_centroids is not None:
            self.db.save_cluster_centroids(self._cluster_centroids)
        # Query embedding buffer for cluster rebuilds across sessions
        if self._query_emb_buffer:
            np.savez(
                str(self.path / "query_embeddings.npz"),
                embeddings=np.stack(self._query_emb_buffer),
            )

    def close(self) -> None:
        """Save and close."""
        self.save()
        self.db.close()

    def _rebuild_index(self) -> None:
        ids = list(self.entries.keys())
        if not ids:
            return
        embs = np.stack([self._embeddings[eid] for eid in ids])
        contents = [self.entries[eid].content for eid in ids]
        self.index.build(ids, embs, contents)

    def _check_tiers(self) -> None:
        for entry in self.entries.values():
            if entry.tier == Tier.NAIVE and entry.retrieval_count >= 3:
                entry.tier = Tier.GC
            elif (entry.tier == Tier.GC
                  and entry.affinity >= 0.65
                  and entry.retrieval_count >= 5):
                entry.tier = Tier.MEMORY
            if (entry.tier not in (Tier.MEMORY, Tier.APOPTOTIC)
                and entry.affinity < 0.15
                and self._step - entry.last_retrieved_step > 1000):
                entry.tier = Tier.APOPTOTIC

    @property
    def size(self) -> int:
        return len(self.entries)

    def stats(self) -> dict[str, object]:
        tiers = {t.value: 0 for t in Tier}
        for e in self.entries.values():
            tiers[e.tier.value] += 1
        return {
            "total_entries": len(self.entries),
            "tiers": tiers,
            "step": self._step,
            "index_size": self.index.size,
        }

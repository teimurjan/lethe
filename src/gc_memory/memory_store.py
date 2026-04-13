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

from gc_memory.db import MemoryDB
from gc_memory.dedup import content_hash, is_near_duplicate
from gc_memory.entry import MemoryEntry, Tier, create_entry
from gc_memory.reranker import Reranker
from gc_memory.rif import RIFConfig, apply_suppression_penalty, update_suppression
from gc_memory.vectors import VectorIndex


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
        k_deep: int = 200,
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
        self.db = MemoryDB(self.path / "gc_memory.db")
        self.index = VectorIndex(dim=dim)
        self.reranker = Reranker(cross_encoder, confidence_threshold)

        # In-memory state
        self.entries: dict[str, MemoryEntry] = {}
        self._embeddings: dict[str, npt.NDArray[np.float32]] = {}
        self._step = int(self.db.get_stat("step", "0"))

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

        # Shallow pass: BM25 + vector hybrid with RRF scores
        raw_candidates = self.index.search_hybrid_scored(query_emb, query, self.k_shallow)
        raw_candidates = [(eid, s) for eid, s in raw_candidates if eid in self.entries]

        # Apply RIF suppression penalty before cross-encoder
        suppression_scores = {eid: self.entries[eid].suppression for eid in self.entries}
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
        competitor_data = [
            (eid, rank_lookup.get(eid, len(adjusted)), xenc_lookup.get(eid, 0.0))
            for eid in candidate_ids
        ]
        last_updated = {eid: self.entries[eid].last_retrieved_step for eid in self.entries}
        rif_updates = update_suppression(
            winner_ids, competitor_data, suppression_scores,
            len(candidate_ids), self.rif, self._step, last_updated,
        )
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

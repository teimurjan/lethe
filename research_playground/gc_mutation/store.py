from __future__ import annotations

import math
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from research_playground.gc_mutation.config import Config
from lethe.entry import MemoryEntry, Tier, effective_embedding
from research_playground.gc_mutation.graph import RelevanceGraph
from research_playground.gc_mutation.rescue_index import RescueIndex


class GCMemoryStore:
    """Memory store with FAISS retrieval, cross-encoder reranking,
    co-relevance graph expansion, tier lifecycle, and time decay."""

    def __init__(
        self,
        entries: list[MemoryEntry],
        config: Config,
        rng: np.random.Generator,
        cross_encoder: Any = None,
        bi_encoder: Any = None,
    ) -> None:
        self.entries: dict[str, MemoryEntry] = {e.id: e for e in entries}
        self.config = config
        self.rng = rng
        self.cross_encoder = cross_encoder
        self.bi_encoder = bi_encoder
        self.graph = RelevanceGraph(max_neighbors=config.graph_max_neighbors)
        self.rescue = RescueIndex(max_size=config.rescue_max_size)
        self._step_count = 0
        self._id_order: list[str] = []
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(0)
        self._bm25: BM25Okapi | None = None
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        active = [
            (eid, e) for eid, e in self.entries.items() if e.tier != Tier.APOPTOTIC
        ]
        self._id_order = [eid for eid, _ in active]
        if not active:
            self._index = faiss.IndexFlatIP(384)
            self._bm25 = None
            return
        # FAISS vector index
        embeddings = np.stack([e.embedding for _, e in active]).astype(np.float32)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        # BM25 sparse index
        tokenized = [e.content.lower().split() for _, e in active]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        k: int,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve: FAISS top-k_fetch → graph expand → xenc rerank → top-k."""
        scored, _ = self._retrieve_with_candidates(query, query_text)
        return scored[:k]

    def _retrieve_with_candidates(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
    ) -> tuple[list[tuple[MemoryEntry, float]], list[tuple[MemoryEntry, float]]]:
        """Returns (top_k_results, all_scored_candidates) for graph learning."""
        if self._index.ntotal == 0:
            return [], []

        # Stage 1: FAISS retrieval (dense vector)
        n_fetch = min(self.config.k_fetch, self._index.ntotal)
        query_2d = query.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query_2d, n_fetch)

        faiss_ids = []
        for idx in indices[0]:
            if idx >= 0:
                faiss_ids.append(self._id_order[idx])

        # Stage 2: BM25 retrieval (sparse keyword)
        bm25_ids: list[str] = []
        if self._bm25 is not None and query_text:
            tokens = query_text.lower().split()
            scores = self._bm25.get_scores(tokens)
            top_bm25 = np.argsort(scores)[::-1][:self.config.k_fetch]
            bm25_ids = [self._id_order[i] for i in top_bm25 if i < len(self._id_order)]

        # Stage 3: Rescue index lookup (learned cache of FAISS misses)
        rescue_ids = self.rescue.lookup(
            query,
            top_k=self.config.rescue_lookup_k,
            similarity_threshold=self.config.rescue_similarity_threshold,
        )

        # Stage 4: Graph expansion
        graph_ids = self.graph.expand(
            faiss_ids[:self.config.graph_seed_k],
            top_k_per_seed=self.config.graph_expand_per_seed,
        )
        all_candidate_ids = list(dict.fromkeys(faiss_ids + bm25_ids + rescue_ids + graph_ids))

        # Stage 3: Cross-encoder rerank (or bi-encoder fallback)
        candidates = [
            (self.entries[eid], 0.0)
            for eid in all_candidate_ids
            if eid in self.entries and self.entries[eid].tier != Tier.APOPTOTIC
        ]

        if self.cross_encoder is not None and candidates:
            pairs = [(query_text, e.content) for e, _ in candidates]
            xenc_scores = self.cross_encoder.predict(pairs)
            tier_weights = self._tier_weights()
            scored = [
                (entry, float(xscore) * tier_weights[entry.tier])
                for (entry, _), xscore in zip(candidates, xenc_scores)
            ]
        else:
            tier_weights = self._tier_weights()
            scored = []
            for eid in all_candidate_ids:
                if eid not in self.entries or self.entries[eid].tier == Tier.APOPTOTIC:
                    continue
                entry = self.entries[eid]
                bienc = float(np.dot(query, entry.embedding))
                scored.append((entry, bienc * tier_weights[entry.tier]))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored, scored  # all_candidates is the full scored set

    def update_after_retrieval(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
        step: int,
        all_candidates: list[tuple[MemoryEntry, float]] | None = None,
    ) -> None:
        """Update affinities, graph, tiers, and periodically rescue deep entries."""
        for entry, _ in retrieved:
            entry.retrieval_count += 1
            entry.last_retrieved_step = step

        self._update_affinities(query_text, retrieved)
        graph_pool = all_candidates if all_candidates is not None else retrieved
        self._update_graph_from_scores(graph_pool)

        # Periodic deep mining: rescue FAISS-missed but xenc-loved entries
        self._step_count += 1
        if (
            self.cross_encoder is not None
            and self._step_count % self.config.rescue_mine_interval == 0
        ):
            self._deep_mine(query, query_text)

        changed = self._check_tier_transitions(step)
        if changed:
            self._rebuild_index()

    def _deep_mine(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
    ) -> None:
        """Look beyond FAISS top-k_fetch for entries that score high on xenc.

        Retrieves top-k_mine from FAISS, scores all of them with cross-encoder,
        and adds entries that scored high but were below position k_fetch
        to the rescue index.
        """
        cfg = self.config
        n_mine = min(cfg.rescue_mine_k, self._index.ntotal)
        if n_mine <= cfg.k_fetch:
            return

        query_2d = query.reshape(1, -1).astype(np.float32)
        _, indices = self._index.search(query_2d, n_mine)

        # Take entries from positions [k_fetch, k_mine) — the "deep" zone
        deep_idxs = [idx for idx in indices[0][cfg.k_fetch:] if idx >= 0]
        if not deep_idxs:
            return

        deep_ids = [self._id_order[idx] for idx in deep_idxs]
        deep_entries = [
            self.entries[eid]
            for eid in deep_ids
            if eid in self.entries and self.entries[eid].tier != Tier.APOPTOTIC
        ]
        if not deep_entries:
            return

        # Score with cross-encoder
        pairs = [(query_text, e.content) for e in deep_entries]
        xenc_scores = self.cross_encoder.predict(pairs)

        # Rescue any entry that scores above threshold
        for entry, score in zip(deep_entries, xenc_scores):
            if float(score) > cfg.rescue_score_threshold:
                self.rescue.add(query, entry.id, float(score))

    def retrieve_and_update(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        k: int,
        step: int,
    ) -> list[tuple[MemoryEntry, float]]:
        """Atomic retrieve + update. Reuses cross-encoder scores for graph learning."""
        scored, all_candidates = self._retrieve_with_candidates(query, query_text)
        top_k = scored[:k]
        self.update_after_retrieval(query, query_text, top_k, step, all_candidates=all_candidates)
        return top_k

    def _update_graph_from_scores(
        self,
        scored_candidates: list[tuple[MemoryEntry, float]],
    ) -> None:
        """Learn graph edges from pre-computed scores (no extra xenc calls).

        Strategy: only reinforce edges between entries that BOTH score
        significantly above the median for this query. This filters out
        co-occurrence noise from queries where the top-10 contains few
        actual relevant entries.
        """
        if len(scored_candidates) < 4:
            return

        cfg = self.config
        scores = np.array([s for _, s in scored_candidates], dtype=np.float64)
        median = float(np.median(scores))
        max_score = float(np.max(scores))

        # If even the best score is below the relevance threshold, skip:
        # this query has no good answers in the candidate set, learning
        # from co-occurrence here would only add noise.
        if max_score < cfg.xenc_relevant:
            for entry, score in scored_candidates:
                if score < cfg.xenc_irrelevant:
                    self.graph.weaken(entry.id, amount=0.05)
            return

        # Threshold: significantly above median AND above relevance floor
        threshold = max(median + 0.5, cfg.xenc_relevant)
        relevant_ids = [
            e.id for e, s in scored_candidates if s >= threshold
        ][: cfg.graph_learn_top_k]

        for entry, score in scored_candidates:
            if score < cfg.xenc_irrelevant:
                self.graph.weaken(entry.id, amount=0.05)

        if len(relevant_ids) >= 2:
            self.graph.reinforce(relevant_ids, weight=cfg.graph_reinforce_weight)

    def _update_graph(
        self,
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
    ) -> None:
        """Legacy: learn graph edges by calling cross-encoder again. Used as fallback."""
        if self.cross_encoder is None:
            return
        pairs = [(query_text, e.content) for e, _ in retrieved]
        xenc_scores = self.cross_encoder.predict(pairs)
        scored = [(e, float(s)) for (e, _), s in zip(retrieved, xenc_scores)]
        self._update_graph_from_scores(scored)

    def run_decay(self, step: int) -> None:
        lam = self.config.lambda_decay
        interval = self.config.decay_interval
        decay_factor = math.exp(-lam * interval / 100.0)
        frozen_ids: set[str] = set()
        for entry in self.entries.values():
            if entry.tier == Tier.MEMORY:
                frozen_ids.add(entry.id)
                continue
            if step - entry.last_retrieved_step < interval:
                continue
            entry.affinity *= decay_factor

        # Decay graph edges (memory-tier edges are frozen)
        self.graph.decay(factor=0.995, frozen_ids=frozen_ids)

    def get_all_entries(self) -> list[MemoryEntry]:
        return list(self.entries.values())

    def get_active_entries(self) -> list[MemoryEntry]:
        return [e for e in self.entries.values() if e.tier != Tier.APOPTOTIC]

    def _tier_weights(self) -> dict[Tier, float]:
        return {
            Tier.NAIVE: 1.0,
            Tier.GC: 1.0,
            Tier.MEMORY: self.config.tier_weight_memory,
            Tier.APOPTOTIC: 0.0,
        }

    def _update_affinities(
        self,
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
    ) -> None:
        alpha = self.config.alpha
        if self.cross_encoder is not None:
            pairs = [(query_text, e.content) for e, _ in retrieved]
            xenc_scores = self.cross_encoder.predict(pairs)
            for (entry, _), xscore in zip(retrieved, xenc_scores):
                normalized = 1.0 / (1.0 + math.exp(-float(xscore)))
                entry.affinity = (1.0 - alpha) * entry.affinity + alpha * normalized
        else:
            for entry, score in retrieved:
                entry.affinity = (1.0 - alpha) * entry.affinity + alpha * max(0.0, min(1.0, score))

    def _check_tier_transitions(self, step: int) -> bool:
        changed = False
        cfg = self.config
        for entry in self.entries.values():
            old_tier = entry.tier

            if entry.tier == Tier.NAIVE and entry.retrieval_count >= cfg.promote_naive_threshold:
                entry.tier = Tier.GC
            elif (
                entry.tier == Tier.GC
                and entry.affinity >= cfg.promote_memory_affinity
                and entry.retrieval_count >= cfg.promote_memory_generation
            ):
                entry.tier = Tier.MEMORY

            if (
                entry.tier != Tier.MEMORY
                and entry.tier != Tier.APOPTOTIC
                and entry.affinity < cfg.apoptosis_affinity
                and (step - entry.last_retrieved_step) > cfg.apoptosis_idle_steps
            ):
                entry.tier = Tier.APOPTOTIC

            if entry.tier != old_tier:
                changed = True
        return changed

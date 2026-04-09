from __future__ import annotations

import math
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt

from gc_memory.config import Config
from gc_memory.entry import MemoryEntry, Tier, effective_embedding
from gc_memory.mutation import (
    compute_sigma,
    generate_adapter_mutants,
    select_best_adapter,
)


class GCMemoryStore:
    """Germinal center memory store with cross-encoder guided adapter mutation.

    The cross-encoder breaks the fitness circularity: it scores (query, content)
    pairs using full text, providing information the bi-encoder doesn't encode.
    Mutation direction is driven by disagreement between bi-encoder rank and
    cross-encoder relevance.
    """

    def __init__(
        self,
        entries: list[MemoryEntry],
        config: Config,
        rng: np.random.Generator,
        cross_encoder: Any = None,
    ) -> None:
        self.entries: dict[str, MemoryEntry] = {e.id: e for e in entries}
        self.config = config
        self.rng = rng
        self.cross_encoder = cross_encoder
        self._id_order: list[str] = []
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(0)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from effective embeddings, excluding apoptotic."""
        active = [
            (eid, e) for eid, e in self.entries.items() if e.tier != Tier.APOPTOTIC
        ]
        self._id_order = [eid for eid, _ in active]
        if not active:
            self._index = faiss.IndexFlatIP(384)
            return
        embeddings = np.stack([e.embedding for _, e in active]).astype(np.float32)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def retrieve(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        k: int,
    ) -> list[tuple[MemoryEntry, float]]:
        """Retrieve top-k: bi-encoder fetch → cross-encoder rerank → tier weight."""
        if self._index.ntotal == 0:
            return []

        n_fetch = min(self.config.k_fetch, self._index.ntotal)
        query_2d = query.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query_2d, n_fetch)

        # Gather candidates (exclude apoptotic at index level already)
        candidates: list[tuple[MemoryEntry, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            eid = self._id_order[idx]
            entry = self.entries[eid]
            candidates.append((entry, float(dist)))

        # Cross-encoder rerank if available
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
            scored = [
                (entry, bienc * tier_weights[entry.tier])
                for entry, bienc in candidates
            ]

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def update_after_retrieval(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
        step: int,
    ) -> None:
        """Post-retrieval: bookkeeping, cross-encoder guided mutation, tier transitions."""
        for entry, _ in retrieved:
            entry.retrieval_count += 1
            entry.last_retrieved_step = step

        self._update_affinities(query_text, retrieved)
        mutated = self._mutate_gc_entries(query, query_text, retrieved)
        changed = self._check_tier_transitions(step)

        if mutated or changed:
            self._rebuild_index()

    def run_decay(self, step: int) -> None:
        """Time decay: affinity *= exp(-lambda * interval / 100) per period."""
        lam = self.config.lambda_decay
        interval = self.config.decay_interval
        decay_factor = math.exp(-lam * interval / 100.0)
        for entry in self.entries.values():
            if entry.tier == Tier.MEMORY:
                continue
            if step - entry.last_retrieved_step < interval:
                continue
            entry.affinity *= decay_factor

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
        """Update affinity using cross-encoder scores (or retrieval score as fallback)."""
        alpha = self.config.alpha
        if self.cross_encoder is not None:
            pairs = [(query_text, e.content) for e, _ in retrieved]
            xenc_scores = self.cross_encoder.predict(pairs)
            for (entry, _), xscore in zip(retrieved, xenc_scores):
                # Sigmoid to normalize logit to [0, 1]
                normalized = 1.0 / (1.0 + math.exp(-float(xscore)))
                entry.affinity = (1.0 - alpha) * entry.affinity + alpha * normalized
        else:
            for entry, score in retrieved:
                entry.affinity = (1.0 - alpha) * entry.affinity + alpha * max(0.0, min(1.0, score))

    def _get_sigma(self, affinity: float) -> float:
        """Compute mutation sigma. Override in baselines for fixed rate."""
        return float(self.config.sigma_0 * (1.0 - affinity) ** self.config.gamma)

    def _mutate_gc_entries(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
    ) -> bool:
        """Cross-encoder guided adapter mutation.

        The cross-encoder score decides IF and HOW to mutate:
        - High xenc + low bi-enc rank → under-retrieved, mutate toward query
        - Low xenc + high bi-enc rank → false positive, mutate away from query
        - High xenc + high bi-enc rank → correctly retrieved, boost affinity only
        """
        any_mutated = False
        cfg = self.config

        # Get cross-encoder scores for retrieved entries
        gc_entries = [(entry, score) for entry, score in retrieved if entry.tier == Tier.GC]
        if not gc_entries:
            return False

        if self.cross_encoder is not None:
            pairs = [(query_text, e.content) for e, _ in gc_entries]
            xenc_scores = self.cross_encoder.predict(pairs)
        else:
            # Without cross-encoder, assume all retrieved entries are relevant
            xenc_scores = np.ones(len(gc_entries)) * (cfg.xenc_relevant + 1.0)

        for (entry, bienc_rank_score), xenc_score in zip(gc_entries, xenc_scores):
            xenc_score_f = float(xenc_score)

            # Determine mutation direction from disagreement
            if xenc_score_f > cfg.xenc_relevant:
                # Cross-encoder says relevant: mutate toward query
                toward = True
            elif xenc_score_f < cfg.xenc_irrelevant:
                # Cross-encoder says strongly irrelevant: mutate away
                toward = False
            else:
                # Ambiguous: skip mutation for this entry
                continue

            sigma = self._get_sigma(entry.affinity)
            adapter_mutants, eff_mutants = generate_adapter_mutants(
                entry.adapter, entry.base_embedding, query,
                sigma, cfg.n_mutants, cfg.max_adapter_norm, self.rng,
                toward_query=toward,
            )

            best_idx = select_best_adapter(
                query, entry.embedding, adapter_mutants, eff_mutants, cfg.delta,
            )
            if best_idx is not None:
                entry.adapter = adapter_mutants[best_idx].copy()
                entry.embedding = eff_mutants[best_idx].copy()
                entry.generation += 1
                any_mutated = True

        return any_mutated

    def _check_tier_transitions(self, step: int) -> bool:
        """Check and apply tier transitions for all entries."""
        changed = False
        cfg = self.config
        for entry in self.entries.values():
            old_tier = entry.tier

            if entry.tier == Tier.NAIVE and entry.retrieval_count >= cfg.promote_naive_threshold:
                entry.tier = Tier.GC
            elif (
                entry.tier == Tier.GC
                and entry.affinity >= cfg.promote_memory_affinity
                and entry.generation >= cfg.promote_memory_generation
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

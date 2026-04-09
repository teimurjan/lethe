from __future__ import annotations

import math

import faiss
import numpy as np
import numpy.typing as npt

from gc_memory.config import Config
from gc_memory.entry import MemoryEntry, Tier
from gc_memory.mutation import generate_mutants, select_best_mutant


class GCMemoryStore:
    def __init__(
        self,
        entries: list[MemoryEntry],
        config: Config,
        rng: np.random.Generator,
    ) -> None:
        self.entries: dict[str, MemoryEntry] = {e.id: e for e in entries}
        self.config = config
        self.rng = rng
        self._id_order: list[str] = []
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(0)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current embeddings, excluding apoptotic."""
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
        k: int,
    ) -> list[tuple[MemoryEntry, float]]:
        """Return top-k entries by effective score (cosine * tier_weight).

        Apoptotic entries are excluded at the index level.
        """
        if self._index.ntotal == 0:
            return []

        n_fetch = min(k * 3, self._index.ntotal)
        query_2d = query.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(query_2d, n_fetch)

        tier_weights = self._tier_weights()
        scored: list[tuple[MemoryEntry, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            eid = self._id_order[idx]
            entry = self.entries[eid]
            weight = tier_weights[entry.tier]
            scored.append((entry, float(dist) * weight))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def update_after_retrieval(
        self,
        query: npt.NDArray[np.float32],
        retrieved: list[tuple[MemoryEntry, float]],
        step: int,
    ) -> None:
        """Post-retrieval updates: bookkeeping, affinity EMA, mutation, tier transitions."""
        for entry, _ in retrieved:
            entry.retrieval_count += 1
            entry.last_retrieved_step = step

        self._update_affinities(query, retrieved)
        mutated = self._mutate_gc_entries(query, retrieved)
        changed = self._check_tier_transitions(step)

        if mutated or changed:
            self._rebuild_index()

    def run_decay(self, step: int) -> None:
        """Time decay: affinity *= exp(-lambda * delta_steps / 100) for non-memory entries."""
        lam = self.config.lambda_decay
        for entry in self.entries.values():
            if entry.tier == Tier.MEMORY:
                continue
            delta_steps = step - entry.last_retrieved_step
            if delta_steps <= 0:
                continue
            entry.affinity *= math.exp(-lam * delta_steps / 100.0)

    def get_all_entries(self) -> list[MemoryEntry]:
        """Return all entries (including apoptotic)."""
        return list(self.entries.values())

    def get_active_entries(self) -> list[MemoryEntry]:
        """Return all non-apoptotic entries."""
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
        query: npt.NDArray[np.float32],
        retrieved: list[tuple[MemoryEntry, float]],
    ) -> None:
        """EMA affinity update for all retrieved entries."""
        alpha = self.config.alpha
        for entry, _ in retrieved:
            cos_sim = float(np.dot(query, entry.embedding))
            entry.affinity = (1.0 - alpha) * entry.affinity + alpha * cos_sim

    def _get_sigma(self, affinity: float) -> float:
        """Compute mutation sigma. Override in baselines for fixed rate."""
        return float(self.config.sigma_0 * (1.0 - affinity) ** self.config.gamma)

    def _mutate_gc_entries(
        self,
        query: npt.NDArray[np.float32],
        retrieved: list[tuple[MemoryEntry, float]],
    ) -> bool:
        """Generate mutants for GC-tier entries, apply selection. Returns True if any accepted."""
        any_mutated = False
        for entry, _ in retrieved:
            if entry.tier != Tier.GC:
                continue
            sigma = self._get_sigma(entry.affinity)
            # Use sigma directly to generate mutants (bypass compute_sigma in generate_mutants)
            dim = entry.embedding.shape[0]
            noise = self.rng.normal(0.0, sigma, size=(self.config.n_mutants, dim)).astype(
                np.float32
            )
            mutants = entry.embedding + noise
            norms = np.linalg.norm(mutants, axis=1, keepdims=True)
            mutants = mutants / norms

            result = select_best_mutant(
                query,
                entry.embedding,
                entry.original_embedding,
                mutants,
                self.config.delta,
                self.config.theta_anchor,
            )
            if result is not None:
                entry.embedding = result
                entry.generation += 1
                any_mutated = True
        return any_mutated

    def _check_tier_transitions(self, step: int) -> bool:
        """Check and apply tier transitions for all entries. Returns True if any changed."""
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

            # Apoptosis: any non-memory tier
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

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from gc_memory.config import Config
from gc_memory.entry import MemoryEntry
from gc_memory.store import GCMemoryStore


class StaticStore(GCMemoryStore):
    """Baseline: cross-encoder reranking but no mutation, no decay, no tier transitions."""

    def update_after_retrieval(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
        step: int,
    ) -> None:
        pass

    def run_decay(self, step: int) -> None:
        pass


class RandomMutationStore(GCMemoryStore):
    """Control: adapter mutation with fixed sigma, query-as-teacher (no cross-encoder for mutation)."""

    def __init__(
        self,
        entries: list[MemoryEntry],
        config: Config,
        rng: np.random.Generator,
        cross_encoder: Any = None,
        fixed_sigma: float = 0.025,
    ) -> None:
        super().__init__(entries, config, rng, cross_encoder)
        self._fixed_sigma = fixed_sigma

    def _get_sigma(self, affinity: float) -> float:
        return self._fixed_sigma

    def _mutate_gc_entries(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
    ) -> bool:
        """Random mutation: always toward query, ignores cross-encoder for mutation decisions."""
        from gc_memory.mutation import generate_adapter_mutants, select_best_adapter

        any_mutated = False
        cfg = self.config
        for entry, _ in retrieved:
            if entry.tier != Tier.GC:
                continue
            sigma = self._get_sigma(entry.affinity)
            adapter_mutants, eff_mutants = generate_adapter_mutants(
                entry.adapter, entry.base_embedding, query,
                sigma, cfg.n_mutants, cfg.max_adapter_norm, self.rng,
                toward_query=True,
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


# Need Tier import for the method above
from gc_memory.entry import Tier  # noqa: E402

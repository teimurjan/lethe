from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from gc_memory.config import Config
from gc_memory.entry import MemoryEntry
from gc_memory.store import GCMemoryStore


class StaticStore(GCMemoryStore):
    """Baseline: cross-encoder reranking but no graph learning, no decay, no tiers.

    Uses the same FAISS + xenc rerank pipeline but the graph stays empty.
    """

    def update_after_retrieval(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
        step: int,
        all_candidates: list[tuple[MemoryEntry, float]] | None = None,
    ) -> None:
        pass

    def run_decay(self, step: int) -> None:
        pass


class NoGraphStore(GCMemoryStore):
    """Control: same as GCMemoryStore but graph expansion disabled.

    Still updates affinities and tiers, but doesn't learn or use the graph.
    Isolates the graph contribution.
    """

    def _update_graph_from_scores(
        self,
        scored_candidates: list[tuple[MemoryEntry, float]],
    ) -> None:
        pass  # no graph learning

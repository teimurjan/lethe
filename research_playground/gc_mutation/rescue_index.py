"""Rescue index: learned cache of FAISS-missed but xenc-loved entries.

When a query is processed, we periodically do a deep FAISS retrieval (top-100)
and score all candidates with the cross-encoder. Entries that score high but
were buried below position 30 are "rescued" and indexed by query embedding.

On future queries, we look up rescued entries by query similarity and include
them in the candidate set. This recovers recall lost to FAISS limitations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class RescueEntry:
    """An entry that was rescued for a particular query embedding."""
    query_embedding: npt.NDArray[np.float32]
    entry_id: str
    xenc_score: float
    age: int = 0  # incremented each step, decays usefulness


class RescueIndex:
    """Stores (query_emb, entry_id, score) tuples and looks them up by similarity."""

    def __init__(self, max_size: int = 5000) -> None:
        self.entries: list[RescueEntry] = []
        self._query_matrix: npt.NDArray[np.float32] | None = None
        self.max_size = max_size

    def add(
        self,
        query_emb: npt.NDArray[np.float32],
        entry_id: str,
        xenc_score: float,
    ) -> None:
        self.entries.append(RescueEntry(
            query_embedding=query_emb.copy(),
            entry_id=entry_id,
            xenc_score=xenc_score,
        ))
        self._query_matrix = None  # invalidate cache
        # Evict oldest if over capacity
        if len(self.entries) > self.max_size:
            # Sort by xenc_score descending, keep top max_size
            self.entries.sort(key=lambda e: e.xenc_score, reverse=True)
            self.entries = self.entries[: self.max_size]

    def lookup(
        self,
        query_emb: npt.NDArray[np.float32],
        top_k: int = 20,
        similarity_threshold: float = 0.7,
    ) -> list[str]:
        """Find rescue entries with similar query embeddings.

        Returns up to top_k entry IDs whose stored query is similar to the
        current query (cosine >= threshold), sorted by combined similarity * xenc_score.
        """
        if not self.entries:
            return []

        # Build query matrix lazily
        if self._query_matrix is None or self._query_matrix.shape[0] != len(self.entries):
            self._query_matrix = np.stack([e.query_embedding for e in self.entries])

        # Cosine similarity (queries are already unit-normalized)
        sims = self._query_matrix @ query_emb.astype(np.float32)

        # Filter by threshold and rank by sim * xenc_score
        results = []
        for i, sim in enumerate(sims):
            if sim < similarity_threshold:
                continue
            entry = self.entries[i]
            # Score: similarity to query × historical xenc score
            results.append((entry.entry_id, float(sim) * entry.xenc_score))

        # Sort by combined score, dedupe entry_ids (keep best score)
        seen: dict[str, float] = {}
        for eid, score in results:
            if eid not in seen or score > seen[eid]:
                seen[eid] = score

        sorted_ids = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        return [eid for eid, _ in sorted_ids[:top_k]]

    def age_step(self, decay_age: int = 5000) -> None:
        """Increment age of all entries, evict ones that are too old."""
        for e in self.entries:
            e.age += 1
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.age < decay_age]
        if len(self.entries) != before:
            self._query_matrix = None

    @property
    def size(self) -> int:
        return len(self.entries)

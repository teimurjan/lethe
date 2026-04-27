"""Co-relevance graph for memory retrieval expansion.

Learns which entries are co-relevant from cross-encoder feedback.
When entry A and entry B both score high for the same query,
they get connected. On retrieval, graph neighbors of top FAISS
results are added to the candidate set before cross-encoder reranking.
"""
from __future__ import annotations

from collections import defaultdict


class RelevanceGraph:
    """Weighted undirected graph of co-relevant entries.

    Edge weight represents confidence that two entries are co-relevant.
    Higher weight = more often seen as co-relevant by cross-encoder.
    """

    def __init__(self, max_neighbors: int = 20) -> None:
        self._edges: dict[str, dict[str, float]] = defaultdict(dict)
        self._max_neighbors = max_neighbors

    def reinforce(self, entry_ids: list[str], weight: float = 1.0) -> None:
        """Strengthen edges between all pairs in entry_ids.

        Called when cross-encoder identifies these entries as co-relevant
        for a query. Weight scales the reinforcement (e.g., by xenc score).
        """
        for i, a in enumerate(entry_ids):
            for b in entry_ids[i + 1 :]:
                self._edges[a][b] = self._edges[a].get(b, 0.0) + weight
                self._edges[b][a] = self._edges[b].get(a, 0.0) + weight
        # Prune to max_neighbors per node (keep highest weight)
        for eid in entry_ids:
            if len(self._edges[eid]) > self._max_neighbors:
                sorted_neighbors = sorted(
                    self._edges[eid].items(), key=lambda x: x[1], reverse=True
                )
                self._edges[eid] = dict(sorted_neighbors[: self._max_neighbors])

    def weaken(self, entry_id: str, amount: float = 0.1) -> None:
        """Weaken all edges for an entry (called when it scores low)."""
        if entry_id not in self._edges:
            return
        to_remove = []
        for neighbor in self._edges[entry_id]:
            self._edges[entry_id][neighbor] -= amount
            if neighbor in self._edges:
                self._edges[neighbor][entry_id] = self._edges[entry_id][neighbor]
            if self._edges[entry_id][neighbor] <= 0:
                to_remove.append(neighbor)
        for n in to_remove:
            del self._edges[entry_id][n]
            if n in self._edges and entry_id in self._edges[n]:
                del self._edges[n][entry_id]

    def decay(self, factor: float = 0.99, frozen_ids: set[str] | None = None) -> None:
        """Decay all edge weights. Frozen entries' edges are exempt."""
        frozen = frozen_ids or set()
        to_remove: list[tuple[str, str]] = []
        for a in list(self._edges.keys()):
            if a in frozen:
                continue
            for b in list(self._edges[a].keys()):
                if b in frozen:
                    continue
                self._edges[a][b] *= factor
                if self._edges[a][b] < 0.01:
                    to_remove.append((a, b))
        for a, b in to_remove:
            if b in self._edges.get(a, {}):
                del self._edges[a][b]
            if a in self._edges.get(b, {}):
                del self._edges[b][a]

    def neighbors(self, entry_id: str, top_k: int = 20) -> list[str]:
        """Return top-k neighbors by edge weight."""
        if entry_id not in self._edges:
            return []
        sorted_neighbors = sorted(
            self._edges[entry_id].items(), key=lambda x: x[1], reverse=True
        )
        return [n for n, _ in sorted_neighbors[:top_k]]

    def expand(self, seed_ids: list[str], top_k_per_seed: int = 10) -> list[str]:
        """Expand a set of seed entries with their graph neighbors.

        Returns deduplicated list of neighbors not in the seed set.
        """
        seeds = set(seed_ids)
        expanded: dict[str, float] = {}
        for sid in seed_ids:
            for neighbor, weight in self._edges.get(sid, {}).items():
                if neighbor not in seeds:
                    expanded[neighbor] = max(expanded.get(neighbor, 0.0), weight)
        # Sort by weight, return top entries
        sorted_expanded = sorted(expanded.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_expanded[:top_k_per_seed * len(seed_ids)]]

    @property
    def num_nodes(self) -> int:
        return len(self._edges)

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self._edges.values()) // 2

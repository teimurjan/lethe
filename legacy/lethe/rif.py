"""Retrieval-Induced Forgetting: principled competitor suppression.

When a memory is successfully retrieved, competing memories (those activated
but not selected) are suppressed proportionally to their competition strength.
Over time, chronic distractors get pushed down, freeing candidate slots for
previously-excluded relevant entries.

Based on Anderson's inhibition theory (1994) and the SAM competitive sampling
model (Raaijmakers & Shiffrin, 1981). First implementation in an AI memory system.

Supports two modes:
- Global (cue-independent): one suppression score per entry.
- Clustered (cue-dependent): suppression per (entry, query_cluster) pair.
  An entry suppressed for "travel" queries stays unsuppressed for "food" queries.
  Based on the SAM model's cue-dependent retrieval probability.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RIFConfig:
    suppression_rate: float = 0.1
    reinforcement_rate: float = 0.05
    max_suppression: float = 1.0
    decay_lambda: float = 0.005
    alpha: float = 0.3  # weight of suppression in score adjustment
    n_clusters: int = 0  # 0 = global (cue-independent), >0 = clustered
    use_rank_gap: bool = False  # use rank-gap competition formula instead of original


def competition_strength(
    initial_rank: int,
    pool_size: int,
    xenc_score: float,
) -> float:
    """Compute how strongly an entry competed but lost.

    High initial rank (retrieved early by BM25/FAISS) + low cross-encoder
    score = strong competitor (distractor). This follows the cognitive science
    finding that stronger competitors receive more suppression.

    The nonmonotonic property: very weak competitors (high rank) produce
    near-zero strength. Peak suppression hits entries that looked relevant
    on surface but weren't.
    """
    if pool_size <= 1:
        return 0.0
    rank_score = 1.0 - (initial_rank / pool_size)
    rejection = 1.0 / (1.0 + math.exp(xenc_score))  # sigmoid(-score)
    return rank_score * rejection


def competition_strength_gap(
    initial_rank: int,
    xenc_rank: int,
    pool_size: int,
    xenc_score: float,
) -> float:
    """Rank-gap competition: measures how much an entry dropped after rerank.

    Intuition: an entry ranked #1 by BM25 but #25 by cross-encoder is
    the most misleading distractor. Entries that kept a similar rank in
    both (e.g., #15 → #15) are not particularly misleading — they just
    didn't make the cut.

    Formula: gap = max(0, xenc_rank - initial_rank) / pool_size
             rejection = sigmoid(-xenc_score)
             competition = gap * rejection

    The rejection factor ensures we only suppress entries the cross-encoder
    actively disliked (xenc_score < 0), not near-winners that just lost out.
    """
    if pool_size <= 1:
        return 0.0
    gap = max(0, xenc_rank - initial_rank) / pool_size
    rejection = 1.0 / (1.0 + math.exp(xenc_score))
    return gap * rejection


def apply_suppression_penalty(
    candidates: list[tuple[str, float]],
    suppression_scores: dict[str, float],
    alpha: float,
) -> list[tuple[str, float]]:
    """Adjust candidate scores by subtracting suppression penalty.

    Returns re-sorted candidates with effective scores.
    """
    adjusted = [
        (eid, score - alpha * suppression_scores.get(eid, 0.0))
        for eid, score in candidates
    ]
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


def update_suppression(
    winner_ids: set[str],
    competitor_scores: list[tuple[str, int, float]] | list[tuple[str, int, int, float]],
    current_suppression: dict[str, float],
    pool_size: int,
    config: RIFConfig,
    current_step: int,
    last_updated: dict[str, int],
) -> dict[str, float]:
    """Apply RIF updates after a retrieval event.

    1. Decay existing suppressions based on elapsed steps
    2. Suppress competitors proportional to competition strength
    3. Reinforce winners (reduce their suppression)

    competitor_scores tuple format:
    - (id, initial_rank, xenc_score) for default formula
    - (id, initial_rank, xenc_rank, xenc_score) when config.use_rank_gap=True

    Returns updated suppression scores for affected entries.
    """
    updated: dict[str, float] = {}

    # Decay + suppress competitors
    for tup in competitor_scores:
        if config.use_rank_gap:
            eid, initial_rank, xenc_rank, xenc_score = tup  # type: ignore[misc]
        else:
            eid, initial_rank, xenc_score = tup  # type: ignore[misc]
            xenc_rank = 0
        if eid in winner_ids:
            continue
        old = current_suppression.get(eid, 0.0)
        # Decay based on steps elapsed since last update
        elapsed = current_step - last_updated.get(eid, current_step)
        if elapsed > 0:
            old *= math.exp(-config.decay_lambda * elapsed)
        # Add competition-proportional suppression
        if config.use_rank_gap:
            strength = competition_strength_gap(
                initial_rank, xenc_rank, pool_size, xenc_score,
            )
        else:
            strength = competition_strength(initial_rank, pool_size, xenc_score)
        new = min(old + strength * config.suppression_rate, config.max_suppression)
        updated[eid] = new

    # Reinforce winners
    for eid in winner_ids:
        old = current_suppression.get(eid, 0.0)
        elapsed = current_step - last_updated.get(eid, current_step)
        if elapsed > 0:
            old *= math.exp(-config.decay_lambda * elapsed)
        updated[eid] = max(0.0, old - config.reinforcement_rate)

    return updated


# --- Clustered (cue-dependent) suppression ---


def build_clusters(
    embeddings: npt.NDArray[np.float32],
    n_clusters: int,
) -> npt.NDArray[np.float32]:
    """K-means clustering on query embeddings. Returns centroids."""
    import faiss

    dim = embeddings.shape[1]
    kmeans = faiss.Kmeans(dim, n_clusters, niter=20, seed=42)
    kmeans.train(embeddings.astype(np.float32))
    return kmeans.centroids


def assign_cluster(
    query_emb: npt.NDArray[np.float32],
    centroids: npt.NDArray[np.float32],
) -> int:
    """Assign a query to its nearest cluster centroid."""
    sims = query_emb @ centroids.T
    return int(np.argmax(sims))


class ClusteredSuppressionState:
    """Per-(entry, cluster) suppression state for cue-dependent RIF."""

    def __init__(self) -> None:
        # cluster_id → {entry_id → suppression_score}
        self._scores: dict[int, dict[str, float]] = defaultdict(dict)
        self._last_updated: dict[int, dict[str, int]] = defaultdict(dict)

    def get_cluster_scores(self, cluster_id: int) -> dict[str, float]:
        return self._scores[cluster_id]

    def get_cluster_last_updated(self, cluster_id: int) -> dict[str, int]:
        return self._last_updated[cluster_id]

    def update_cluster(
        self, cluster_id: int, updates: dict[str, float], step: int,
    ) -> None:
        for eid, score in updates.items():
            self._scores[cluster_id][eid] = score
            self._last_updated[cluster_id][eid] = step

    def total_suppressed(self, threshold: float = 0.01) -> int:
        seen: set[str] = set()
        for cluster_scores in self._scores.values():
            for eid, s in cluster_scores.items():
                if s > threshold:
                    seen.add(eid)
        return len(seen)

    def max_suppression(self) -> float:
        mx = 0.0
        for cluster_scores in self._scores.values():
            for s in cluster_scores.values():
                mx = max(mx, s)
        return mx

    def mean_suppression(self, threshold: float = 0.01) -> float:
        values = [
            s for cluster_scores in self._scores.values()
            for s in cluster_scores.values() if s > threshold
        ]
        return float(np.mean(values)) if values else 0.0

    def snapshot(
        self,
    ) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, int]]]:
        """Plain-dict copies suitable for persistence."""
        scores = {cid: dict(d) for cid, d in self._scores.items()}
        last = {cid: dict(d) for cid, d in self._last_updated.items()}
        return scores, last

    def restore(
        self,
        scores: dict[int, dict[str, float]],
        last_updated: dict[int, dict[str, int]],
    ) -> None:
        """Replace state from a previously saved snapshot."""
        self._scores = defaultdict(dict)
        self._last_updated = defaultdict(dict)
        for cid, d in scores.items():
            self._scores[cid].update(d)
        for cid, d in last_updated.items():
            self._last_updated[cid].update(d)

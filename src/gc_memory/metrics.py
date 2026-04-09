from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from gc_memory.entry import MemoryEntry, Tier


def compute_diversity(
    embeddings: npt.NDArray[np.float32],
    n_pairs: int,
    rng: np.random.Generator,
) -> float:
    """Mean pairwise cosine distance over n_pairs random pairs.

    Distance = 1 - cosine_similarity. Embeddings assumed unit-normalized.
    """
    n = embeddings.shape[0]
    if n < 2:
        return 0.0
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    # Re-draw collisions
    mask = idx_a == idx_b
    while mask.any():
        idx_b[mask] = rng.integers(0, n, size=int(mask.sum()))
        mask = idx_a == idx_b
    dots = np.sum(embeddings[idx_a] * embeddings[idx_b], axis=1)
    return float(np.mean(1.0 - dots))


def compute_anchor_drift(
    entries: Sequence[MemoryEntry],
    n_sample: int,
    rng: np.random.Generator,
) -> float:
    """Mean adapter norm over n_sample random entries.

    With adapter-based mutation, drift is bounded by max_adapter_norm.
    Adapter norm directly measures how far the effective embedding has
    moved from the base embedding.
    """
    if len(entries) == 0:
        return 0.0
    n_sample = min(n_sample, len(entries))
    indices = rng.choice(len(entries), size=n_sample, replace=False)
    norms = []
    for idx in indices:
        entry = entries[idx]
        norms.append(float(np.linalg.norm(entry.adapter)))
    return float(np.mean(norms))


def compute_tier_distribution(entries: Sequence[MemoryEntry]) -> dict[Tier, int]:
    """Count entries per tier."""
    counts: dict[Tier, int] = {tier: 0 for tier in Tier}
    for entry in entries:
        counts[entry.tier] += 1
    return counts


def compute_mean_generation(entries: Sequence[MemoryEntry]) -> float:
    """Mean generation across non-naive entries. Returns 0.0 if none."""
    non_naive = [e for e in entries if e.tier != Tier.NAIVE]
    if not non_naive:
        return 0.0
    return float(np.mean([e.generation for e in non_naive]))


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    """NDCG@k using graded relevance (gain = 2^rel - 1)."""
    dcg = _dcg(retrieved_ids[:k], relevance)
    # Ideal: sort all relevant docs by relevance descending
    ideal_order = sorted(relevance.keys(), key=lambda d: relevance[d], reverse=True)
    idcg = _dcg(ideal_order[:k], relevance)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _dcg(doc_ids: Sequence[str], relevance: dict[str, int]) -> float:
    total = 0.0
    for i, doc_id in enumerate(doc_ids):
        rel = relevance.get(doc_id, 0)
        gain = 2.0**rel - 1.0
        total += gain / math.log2(i + 2)  # i+2 because rank starts at 1, log2(1+1)
    return total


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    relevant = {doc_id for doc_id, rel in relevance.items() if rel > 0}
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(relevant & retrieved_set) / len(relevant)

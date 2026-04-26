"""Deduplication: exact hash + cosine near-duplicate detection."""
from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def is_near_duplicate(
    new_embedding: npt.NDArray[np.float32],
    existing_embeddings: npt.NDArray[np.float32],
    threshold: float = 0.95,
) -> int | None:
    """Check if new_embedding is a near-duplicate of any existing embedding.

    Returns the index of the most similar existing entry if cosine >= threshold,
    or None if no duplicate found.
    """
    if existing_embeddings.shape[0] == 0:
        return None
    sims = existing_embeddings @ new_embedding
    best_idx = int(np.argmax(sims))
    if float(sims[best_idx]) >= threshold:
        return best_idx
    return None

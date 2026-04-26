"""Cross-encoder reranker with adaptive depth."""
from __future__ import annotations

from typing import Any

import numpy as np


class Reranker:
    """Wraps a cross-encoder model with adaptive depth logic.

    Shallow pass: score k_fetch candidates.
    If max score < threshold, trigger deep pass: score k_deep candidates.
    """

    def __init__(
        self,
        cross_encoder: Any,
        confidence_threshold: float = 4.0,
    ) -> None:
        self.cross_encoder = cross_encoder
        self.confidence_threshold = confidence_threshold
        self._loaded = cross_encoder is not None

    def rerank(
        self,
        query_text: str,
        candidates: list[tuple[str, str]],  # (entry_id, content)
    ) -> list[tuple[str, float]]:
        """Score candidates with cross-encoder, return sorted (id, score) pairs."""
        if not self._loaded or not candidates:
            return [(eid, 0.0) for eid, _ in candidates]
        pairs = [(query_text, content) for _, content in candidates]
        scores = self.cross_encoder.predict(pairs)
        scored = [(eid, float(s)) for (eid, _), s in zip(candidates, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def needs_deep_search(self, scores: list[float]) -> bool:
        """Returns True if the shallow pass didn't find anything confident."""
        if not scores:
            return True
        return max(scores) < self.confidence_threshold

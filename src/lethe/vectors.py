"""Vector index management: FAISS dense + BM25 sparse."""
from __future__ import annotations

import re
from pathlib import Path

import faiss
import numpy as np
import numpy.typing as npt
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize_bm25(text: str) -> list[str]:
    """Lowercase regex word-tokenizer for BM25.

    The canonical BM25 tokenizer for lethe. Used by ``VectorIndex`` at
    both build and query time, and by the benchmark scripts so every
    measurement uses the same rules as production. Kept as a
    public-style name (no leading underscore) because it has stable
    callers outside this module.

    Replaces the earlier ``text.lower().split()`` which kept
    punctuation glued to adjacent words (e.g. ``"hello,"`` never
    matched ``"hello"``). Switching to a regex word-tokenizer gave
    +3.68pp NDCG@10 / +6.79pp Recall@10 on LongMemEval — see
    ``benchmarks/results/BENCHMARKS_BM25_TOKENIZER.md``. Stopword
    removal and Porter stemming were both tested and regressed or
    barely recovered, so we keep the tokenizer simple and punctuation
    handling is the whole lever.
    """
    return _WORD_RE.findall(text.lower())


def _top_k_desc(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-``k`` entries by ``scores``, descending.

    ``argpartition`` avoids sorting the whole array for large corpora
    (O(N) vs O(N log N)); we then sort only the k winners."""
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if k >= n:
        return np.argsort(scores)[::-1]
    part = np.argpartition(scores, -k)[-k:]
    return part[np.argsort(scores[part])[::-1]]


class VectorIndex:
    """Manages FAISS dense vector index and BM25 sparse keyword index."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._faiss: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._bm25: BM25Okapi | None = None
        self._contents: list[str] = []

    def build(
        self,
        ids: list[str],
        embeddings: npt.NDArray[np.float32],
        contents: list[str],
    ) -> None:
        """Build both indexes from scratch."""
        self._ids = list(ids)
        self._contents = list(contents)
        # FAISS
        self._faiss = faiss.IndexFlatIP(self.dim)
        if len(embeddings) > 0:
            self._faiss.add(embeddings.astype(np.float32))
        # BM25
        tokenized = [tokenize_bm25(c) for c in contents]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search_vector(
        self, query_emb: npt.NDArray[np.float32], k: int,
    ) -> list[str]:
        """Return top-k entry IDs by dense vector similarity."""
        if self._faiss.ntotal == 0:
            return []
        n = min(k, self._faiss.ntotal)
        _, indices = self._faiss.search(query_emb.reshape(1, -1).astype(np.float32), n)
        return [self._ids[i] for i in indices[0] if 0 <= i < len(self._ids)]

    def search_vector_scored(
        self, query_emb: npt.NDArray[np.float32], k: int,
    ) -> list[tuple[str, float]]:
        """Return top-k (entry_id, cosine_score) by dense vector similarity."""
        if self._faiss.ntotal == 0:
            return []
        n = min(k, self._faiss.ntotal)
        distances, indices = self._faiss.search(
            query_emb.reshape(1, -1).astype(np.float32), n,
        )
        return [
            (self._ids[i], float(distances[0][rank]))
            for rank, i in enumerate(indices[0])
            if 0 <= i < len(self._ids)
        ]

    def search_bm25(self, query_text: str, k: int) -> list[str]:
        """Return top-k entry IDs by BM25 keyword match."""
        if self._bm25 is None or not query_text:
            return []
        tokens = tokenize_bm25(query_text)
        if not tokens:
            # Empty token list (punctuation-only query, etc.) —
            # BM25Okapi.get_scores([]) returns all-zero scores and the
            # top-k would be arbitrary. Treat it as a no-match query.
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = _top_k_desc(scores, k)
        return [self._ids[i] for i in top_idx if i < len(self._ids)]

    def search_bm25_scored(self, query_text: str, k: int) -> list[tuple[str, float]]:
        """Return top-k (entry_id, bm25_score) by keyword match."""
        if self._bm25 is None or not query_text:
            return []
        tokens = tokenize_bm25(query_text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        top_idx = _top_k_desc(scores, k)
        return [
            (self._ids[i], float(scores[i]))
            for i in top_idx if i < len(self._ids)
        ]

    def search_hybrid(
        self, query_emb: npt.NDArray[np.float32], query_text: str, k: int,
    ) -> list[str]:
        """Return deduplicated union of vector and BM25 top-k results."""
        vec_ids = self.search_vector(query_emb, k)
        bm25_ids = self.search_bm25(query_text, k)
        return list(dict.fromkeys(vec_ids + bm25_ids))

    def search_hybrid_scored(
        self, query_emb: npt.NDArray[np.float32], query_text: str, k: int,
    ) -> list[tuple[str, float]]:
        """Return deduplicated union with RRF scores (1/rank fusion)."""
        vec_scored = self.search_vector_scored(query_emb, k)
        bm25_scored = self.search_bm25_scored(query_text, k)
        rrf: dict[str, float] = {}
        for rank, (eid, _) in enumerate(vec_scored):
            rrf[eid] = rrf.get(eid, 0.0) + 1.0 / (rank + 60)  # RRF k=60
        for rank, (eid, _) in enumerate(bm25_scored):
            rrf[eid] = rrf.get(eid, 0.0) + 1.0 / (rank + 60)
        scored = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return scored

    def save(self, path: Path) -> None:
        """Persist FAISS index to disk."""
        faiss.write_index(self._faiss, str(path / "faiss.index"))

    def load(self, path: Path, ids: list[str], contents: list[str]) -> None:
        """Load persisted FAISS index, rebuild BM25."""
        index_path = path / "faiss.index"
        if index_path.exists():
            self._faiss = faiss.read_index(str(index_path))
        self._ids = list(ids)
        self._contents = list(contents)
        tokenized = [tokenize_bm25(c) for c in contents]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    @property
    def size(self) -> int:
        return self._faiss.ntotal

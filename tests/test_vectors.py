"""Unit tests for lethe.vectors.VectorIndex (FAISS + BM25 + RRF hybrid)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lethe.vectors import VectorIndex, _top_k_desc


DIM = 16


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


@pytest.fixture
def index() -> VectorIndex:
    idx = VectorIndex(dim=DIM)
    # 5 docs with known relationships:
    # - "dog cat" and "dog bird" share "dog"
    # - "python programming" standalone
    ids = ["a", "b", "c", "d", "e"]
    contents = [
        "dog cat running",
        "dog bird flying",
        "python programming language",
        "quantum physics paper",
        "window seat preference",
    ]
    # Make embeddings that match the narrative: a ≈ b similar, others orthogonal
    rng = np.random.default_rng(7)
    embs = np.zeros((5, DIM), dtype=np.float32)
    embs[0] = _unit(rng.standard_normal(DIM))
    # b shares direction with a (cosine ~0.9)
    embs[1] = _unit(0.9 * embs[0] + 0.1 * rng.standard_normal(DIM))
    for i in range(2, 5):
        embs[i] = _unit(rng.standard_normal(DIM))
    idx.build(ids, embs, contents)
    return idx


def test_build_populates_both_indexes(index: VectorIndex) -> None:
    assert index.size == 5
    assert index._bm25 is not None  # BM25 built
    assert len(index._ids) == 5


def test_search_vector_returns_nearest_by_cosine(index: VectorIndex) -> None:
    # Query close to "a" → "a" and "b" should be top
    query = index._faiss.reconstruct(0)  # same as embs[0]
    results = index.search_vector(query, k=2)
    assert "a" in results
    assert "b" in results


def test_search_bm25_matches_shared_keywords(index: VectorIndex) -> None:
    results = index.search_bm25("dog", k=5)
    # "a" and "b" both contain "dog", should rank above others
    assert "a" in results[:2]
    assert "b" in results[:2]


def test_search_bm25_empty_query_returns_empty(index: VectorIndex) -> None:
    assert index.search_bm25("", k=5) == []


def test_search_hybrid_dedupes_ids(index: VectorIndex) -> None:
    query = index._faiss.reconstruct(0)
    ids = index.search_hybrid(query, "dog", k=5)
    # ID appears at most once
    assert len(ids) == len(set(ids))


def test_search_hybrid_scored_uses_rrf(index: VectorIndex) -> None:
    query = index._faiss.reconstruct(0)
    scored = index.search_hybrid_scored(query, "dog", k=5)
    assert all(isinstance(score, float) and score > 0 for _, score in scored)
    # Results are descending by score
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)
    # "a" should be top (best of vector near + BM25 match)
    assert scored[0][0] == "a"


def test_search_on_empty_index_returns_empty() -> None:
    idx = VectorIndex(dim=DIM)
    # Nothing indexed → vector search returns []
    assert idx.search_vector(np.zeros(DIM, dtype=np.float32), k=5) == []
    # BM25 not built → returns []
    assert idx.search_bm25("anything", k=5) == []
    # Hybrid returns empty
    assert idx.search_hybrid(np.zeros(DIM, dtype=np.float32), "q", k=5) == []


def test_save_and_load_preserves_faiss_index(index: VectorIndex, tmp_path: Path) -> None:
    index.save(tmp_path)
    assert (tmp_path / "faiss.index").exists()

    fresh = VectorIndex(dim=DIM)
    fresh.load(tmp_path, ids=index._ids, contents=index._contents)
    # Searching the reloaded index should return the same top result for the same query
    q = index._faiss.reconstruct(0)
    before = index.search_vector(q, k=1)
    after = fresh.search_vector(q, k=1)
    assert before == after


def test_search_bm25_scored_returns_sorted() -> None:
    idx = VectorIndex(dim=DIM)
    idx.build(
        ["a", "b", "c"],
        np.eye(3, DIM, dtype=np.float32),
        ["alpha beta gamma", "beta delta", "epsilon zeta"],
    )
    results = idx.search_bm25_scored("beta", k=3)
    # "a" (1 hit) and "b" (1 hit) both contain "beta"; "c" does not
    ids = [eid for eid, _ in results[:2]]
    assert "a" in ids and "b" in ids
    # Results sorted descending by score
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# ---------- _top_k_desc ----------

def test_top_k_desc_matches_argsort_tail() -> None:
    """``_top_k_desc`` must return the same k winners, same order, as
    the previous ``np.argsort(scores)[::-1][:k]`` implementation."""
    rng = np.random.default_rng(0)
    for n in (0, 1, 10, 200, 5000):
        scores = rng.standard_normal(n).astype(np.float32)
        for k in (1, 5, 50, n, n + 1):
            if k == 0:
                continue
            got = _top_k_desc(scores, k)
            expected = np.argsort(scores)[::-1][:k]
            assert np.array_equal(
                scores[got], scores[expected]
            ), f"n={n} k={k}: tie-breaks may differ but score sequence must match"

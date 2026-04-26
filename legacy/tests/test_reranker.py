"""Unit tests for lethe.reranker.Reranker."""
from __future__ import annotations

from lethe.reranker import Reranker


class FakeEncoder:
    """Returns scores based on a predefined mapping from content → score."""

    def __init__(self, mapping: dict[str, float]) -> None:
        self.mapping = mapping
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs):  # type: ignore[no-untyped-def]
        self.calls.append(list(pairs))
        return [self.mapping.get(content, 0.0) for _, content in pairs]


def test_rerank_sorts_by_cross_encoder_score() -> None:
    enc = FakeEncoder({"c1": 3.0, "c2": 1.0, "c3": 5.0})
    r = Reranker(enc)
    out = r.rerank("q", [("a", "c1"), ("b", "c2"), ("c", "c3")])
    assert [eid for eid, _ in out] == ["c", "a", "b"]
    assert [s for _, s in out] == [5.0, 3.0, 1.0]


def test_rerank_with_no_encoder_returns_zero_scores() -> None:
    r = Reranker(cross_encoder=None)
    out = r.rerank("q", [("a", "x"), ("b", "y")])
    assert out == [("a", 0.0), ("b", 0.0)]


def test_rerank_empty_candidates_returns_empty() -> None:
    enc = FakeEncoder({})
    r = Reranker(enc)
    assert r.rerank("q", []) == []
    assert enc.calls == []  # model not even invoked


def test_needs_deep_search_true_when_max_below_threshold() -> None:
    r = Reranker(cross_encoder=None, confidence_threshold=4.0)
    assert r.needs_deep_search([1.0, 2.0, 3.5]) is True


def test_needs_deep_search_false_when_max_above_threshold() -> None:
    r = Reranker(cross_encoder=None, confidence_threshold=4.0)
    assert r.needs_deep_search([1.0, 4.5]) is False


def test_needs_deep_search_true_on_empty_scores() -> None:
    r = Reranker(cross_encoder=None, confidence_threshold=0.0)
    assert r.needs_deep_search([]) is True


def test_rerank_pairs_query_with_each_candidate() -> None:
    enc = FakeEncoder({"cat": 1.0, "dog": 2.0})
    r = Reranker(enc)
    r.rerank("my query", [("1", "cat"), ("2", "dog")])
    assert enc.calls[0] == [("my query", "cat"), ("my query", "dog")]

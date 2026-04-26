from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import math

import numpy as np
import pytest

from lethe.entry import Tier, create_entry
from benchmarks._lib.metrics import (
    compute_anchor_drift,
    compute_diversity,
    compute_mean_generation,
    compute_tier_distribution,
    ndcg_at_k,
    recall_at_k,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestDiversity:
    def test_identical_zero(self, rng: np.random.Generator) -> None:
        v = np.ones((5, 384), dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        assert compute_diversity(v, 100, rng) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_one(self, rng: np.random.Generator) -> None:
        embeddings = np.zeros((5, 384), dtype=np.float32)
        for i in range(5):
            embeddings[i, i] = 1.0
        assert compute_diversity(embeddings, 1000, rng) == pytest.approx(1.0, abs=0.01)


class TestAnchorDrift:
    def test_zero_for_unmutated(self, rng: np.random.Generator) -> None:
        entries = [create_entry(f"e{i}", "c", rng.standard_normal(384).astype(np.float32)) for i in range(10)]
        assert compute_anchor_drift(entries, 10, rng) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_mutated(self, rng: np.random.Generator) -> None:
        entry = create_entry("e0", "c", rng.standard_normal(384).astype(np.float32))
        entry.adapter = rng.standard_normal(384).astype(np.float32) * 0.1
        assert compute_anchor_drift([entry], 1, rng) > 0.0


class TestTierDistribution:
    def test_counts(self) -> None:
        rng = np.random.default_rng(0)
        entries = [create_entry(f"n{i}", "c", rng.standard_normal(384).astype(np.float32)) for i in range(3)]
        gc = create_entry("g0", "c", rng.standard_normal(384).astype(np.float32))
        gc.tier = Tier.GC
        entries.append(gc)
        dist = compute_tier_distribution(entries)
        assert dist[Tier.NAIVE] == 3
        assert dist[Tier.GC] == 1


class TestNDCG:
    def test_perfect(self) -> None:
        assert ndcg_at_k(["a", "b", "c"], {"a": 3, "b": 2, "c": 1}, 3) == pytest.approx(1.0)

    def test_inverse(self) -> None:
        score = ndcg_at_k(["c", "b", "a"], {"a": 3, "b": 2, "c": 1}, 3)
        assert score < 1.0
        dcg = 1.0 / 1.0 + 3.0 / math.log2(3) + 7.0 / math.log2(4)
        idcg = 7.0 / 1.0 + 3.0 / math.log2(3) + 1.0 / math.log2(4)
        assert score == pytest.approx(dcg / idcg, abs=1e-6)

    def test_no_relevant(self) -> None:
        assert ndcg_at_k(["x", "y"], {}, 2) == 0.0


class TestRecall:
    def test_perfect(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a": 1, "b": 2}, 3) == pytest.approx(1.0)

    def test_partial(self) -> None:
        assert recall_at_k(["a", "x"], {"a": 1, "b": 2, "c": 1}, 2) == pytest.approx(1 / 3)

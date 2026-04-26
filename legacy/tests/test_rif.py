from __future__ import annotations

import math

import pytest

import numpy as np

from lethe.rif import (
    ClusteredSuppressionState,
    RIFConfig,
    apply_suppression_penalty,
    assign_cluster,
    build_clusters,
    competition_strength,
    competition_strength_gap,
    update_suppression,
)


class TestCompetitionStrength:
    def test_high_rank_low_xenc_produces_high_strength(self) -> None:
        # Rank 0 out of 30 (top candidate), xenc = -5 (rejected hard)
        s = competition_strength(initial_rank=0, pool_size=30, xenc_score=-5.0)
        assert s > 0.9  # strong competitor: looked great, wasn't

    def test_low_rank_produces_low_strength(self) -> None:
        # Rank 29 out of 30 (bottom), xenc = -5
        s = competition_strength(initial_rank=29, pool_size=30, xenc_score=-5.0)
        assert s < 0.05  # weak competitor: barely made the pool

    def test_high_xenc_produces_low_strength(self) -> None:
        # Rank 0 out of 30, xenc = 5.0 (cross-encoder liked it)
        s = competition_strength(initial_rank=0, pool_size=30, xenc_score=5.0)
        assert s < 0.01  # not really a competitor — it was relevant

    def test_singleton_pool(self) -> None:
        assert competition_strength(0, 1, -3.0) == 0.0

    def test_symmetry_around_zero_xenc(self) -> None:
        pos = competition_strength(0, 30, 1.0)
        neg = competition_strength(0, 30, -1.0)
        assert neg > pos  # negative xenc = more rejection = more suppression


class TestCompetitionStrengthGap:
    def test_large_gap_with_rejection_is_high(self) -> None:
        # Rank #0 initially, #25 by xenc, strongly rejected: classic distractor
        s = competition_strength_gap(initial_rank=0, xenc_rank=25, pool_size=30, xenc_score=-5.0)
        assert s > 0.7

    def test_no_gap_is_zero(self) -> None:
        # Same rank in both = not a distractor, just ranked where it belongs
        s = competition_strength_gap(initial_rank=5, xenc_rank=5, pool_size=30, xenc_score=-2.0)
        assert s == 0.0

    def test_negative_gap_is_zero(self) -> None:
        # xenc promoted it (better rank) — not a competitor
        s = competition_strength_gap(initial_rank=15, xenc_rank=3, pool_size=30, xenc_score=2.0)
        assert s == 0.0

    def test_gap_with_positive_xenc_is_low(self) -> None:
        # Dropped from #0 to #20 but xenc still likes it = near-winner, not distractor
        s = competition_strength_gap(initial_rank=0, xenc_rank=20, pool_size=30, xenc_score=3.0)
        assert s < 0.1

    def test_singleton_pool(self) -> None:
        assert competition_strength_gap(0, 0, 1, -3.0) == 0.0


class TestApplySuppressionPenalty:
    def test_reorders_candidates(self) -> None:
        candidates = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        suppression = {"a": 1.0, "b": 0.0, "c": 0.0}
        adjusted = apply_suppression_penalty(candidates, suppression, alpha=0.3)
        ids = [eid for eid, _ in adjusted]
        # "a" should drop: 1.0 - 0.3*1.0 = 0.7 < 0.9 and 0.8
        assert ids[0] == "b"
        assert ids[1] == "c"
        assert ids[2] == "a"

    def test_no_suppression_preserves_order(self) -> None:
        candidates = [("a", 1.0), ("b", 0.5)]
        adjusted = apply_suppression_penalty(candidates, {}, alpha=0.3)
        assert [eid for eid, _ in adjusted] == ["a", "b"]

    def test_alpha_zero_ignores_suppression(self) -> None:
        candidates = [("a", 1.0), ("b", 0.5)]
        suppression = {"a": 10.0}
        adjusted = apply_suppression_penalty(candidates, suppression, alpha=0.0)
        assert adjusted[0][0] == "a"


class TestUpdateSuppression:
    @pytest.fixture
    def config(self) -> RIFConfig:
        return RIFConfig(
            suppression_rate=0.1,
            reinforcement_rate=0.05,
            max_suppression=1.0,
            decay_lambda=0.005,
        )

    def test_competitors_get_suppressed(self, config: RIFConfig) -> None:
        winners = {"a"}
        competitors = [
            ("a", 0, 5.0),   # winner
            ("b", 1, -3.0),  # competitor: high rank, low xenc
            ("c", 5, -1.0),  # competitor: mid rank, mid xenc
        ]
        result = update_suppression(
            winners, competitors, {}, pool_size=10,
            config=config, current_step=1, last_updated={},
        )
        assert result["b"] > 0.0  # b got suppressed
        assert result["c"] > 0.0  # c got suppressed
        assert result["b"] > result["c"]  # b suppressed more (higher rank, worse xenc)

    def test_winners_get_reinforced(self, config: RIFConfig) -> None:
        winners = {"a"}
        competitors = [("a", 0, 5.0)]
        result = update_suppression(
            winners, competitors,
            current_suppression={"a": 0.5},
            pool_size=10, config=config,
            current_step=1, last_updated={"a": 0},
        )
        assert result["a"] < 0.5  # suppression decreased

    def test_suppression_caps_at_max(self, config: RIFConfig) -> None:
        winners = set()
        competitors = [("b", 0, -10.0)]  # maximum competition
        # Start near max
        result = update_suppression(
            winners, competitors,
            current_suppression={"b": 0.95},
            pool_size=10, config=config,
            current_step=1, last_updated={"b": 1},
        )
        assert result["b"] <= config.max_suppression

    def test_decay_reduces_existing_suppression(self, config: RIFConfig) -> None:
        winners = set()
        # Competitor with no new competition (not in candidate pool)
        competitors = [("b", 5, 0.0)]
        result = update_suppression(
            winners, competitors,
            current_suppression={"b": 0.8},
            pool_size=10, config=config,
            current_step=200, last_updated={"b": 0},
        )
        # Decay should reduce the 0.8 significantly before adding new suppression
        expected_decayed = 0.8 * math.exp(-config.decay_lambda * 200)
        assert result["b"] < 0.8  # net reduction from decay

    def test_winner_suppression_floors_at_zero(self, config: RIFConfig) -> None:
        winners = {"a"}
        competitors = [("a", 0, 5.0)]
        result = update_suppression(
            winners, competitors,
            current_suppression={"a": 0.01},
            pool_size=10, config=config,
            current_step=1, last_updated={"a": 1},
        )
        assert result["a"] >= 0.0

    def test_gap_formula_routed_through_config(self, config: RIFConfig) -> None:
        gap_cfg = RIFConfig(**{**config.__dict__, "use_rank_gap": True})
        winners: set[str] = set()
        # Format for gap: (id, initial_rank, xenc_rank, xenc_score)
        competitors = [("a", 0, 25, -5.0), ("b", 5, 5, -5.0)]
        result = update_suppression(
            winners, competitors, {}, pool_size=30,
            config=gap_cfg, current_step=1, last_updated={},
        )
        # "a" has a large rank drop (0→25) → suppressed; "b" didn't drop (5→5) → not.
        assert result["a"] > 0.0
        assert result["b"] == 0.0


class TestBuildClusters:
    def test_returns_centroids_with_expected_shape(self) -> None:
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((200, 16)).astype(np.float32)
        centroids = build_clusters(embeddings, n_clusters=5)
        assert centroids.shape == (5, 16)
        assert centroids.dtype == np.float32

    def test_deterministic_with_seed(self) -> None:
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((200, 16)).astype(np.float32)
        c1 = build_clusters(embeddings, n_clusters=5)
        c2 = build_clusters(embeddings, n_clusters=5)
        # FAISS kmeans w/ fixed seed is deterministic
        assert np.allclose(c1, c2)


class TestAssignCluster:
    def test_returns_nearest_centroid_index(self) -> None:
        centroids = np.array(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        q = np.array([0.9, 0.1], dtype=np.float32)
        assert assign_cluster(q, centroids) == 0
        q2 = np.array([-0.95, 0.2], dtype=np.float32)
        assert assign_cluster(q2, centroids) == 2


class TestClusteredSuppressionState:
    def test_isolates_per_cluster(self) -> None:
        s = ClusteredSuppressionState()
        s.update_cluster(0, {"e1": 0.7}, step=1)
        s.update_cluster(5, {"e2": 0.4}, step=2)

        # Cluster 0 only sees e1, not e2
        assert s.get_cluster_scores(0) == {"e1": 0.7}
        assert s.get_cluster_scores(5) == {"e2": 0.4}
        # Empty cluster returns empty dict (not a shared reference)
        assert s.get_cluster_scores(99) == {}

    def test_update_cluster_independent(self) -> None:
        s = ClusteredSuppressionState()
        s.update_cluster(0, {"e1": 0.5}, step=1)
        s.update_cluster(0, {"e1": 0.8, "e2": 0.3}, step=5)
        scores = s.get_cluster_scores(0)
        assert scores["e1"] == 0.8  # overwritten
        assert scores["e2"] == 0.3

    def test_update_records_last_updated_step(self) -> None:
        s = ClusteredSuppressionState()
        s.update_cluster(3, {"x": 0.2}, step=42)
        assert s.get_cluster_last_updated(3) == {"x": 42}

    def test_total_suppressed_counts_unique_entries_over_threshold(self) -> None:
        s = ClusteredSuppressionState()
        s.update_cluster(0, {"a": 0.2, "b": 0.001}, step=1)
        s.update_cluster(1, {"b": 0.5, "c": 0.4}, step=2)
        # Above default 0.01 threshold: a, b (cluster 1), c. "b in cluster 0" below threshold.
        assert s.total_suppressed() == 3

    def test_max_suppression(self) -> None:
        s = ClusteredSuppressionState()
        assert s.max_suppression() == 0.0  # empty state
        s.update_cluster(0, {"a": 0.3}, step=1)
        s.update_cluster(1, {"b": 0.9}, step=2)
        assert s.max_suppression() == 0.9

    def test_mean_suppression_handles_empty(self) -> None:
        s = ClusteredSuppressionState()
        assert s.mean_suppression() == 0.0
        s.update_cluster(0, {"a": 0.2, "b": 0.4}, step=1)
        mean = s.mean_suppression(threshold=0.0)
        assert mean == pytest.approx(0.3)

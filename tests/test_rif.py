from __future__ import annotations

import math

import pytest

from gc_memory.rif import (
    RIFConfig,
    apply_suppression_penalty,
    competition_strength,
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

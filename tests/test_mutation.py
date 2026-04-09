from __future__ import annotations

import numpy as np
import pytest

from gc_memory.config import Config
from gc_memory.mutation import compute_sigma, generate_mutants, select_best_mutant


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def unit_embedding(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


class TestComputeSigma:
    def test_formula(self, config: Config) -> None:
        """sigma = sigma_0 * (1 - affinity) ** gamma"""
        assert compute_sigma(0.0, config.sigma_0, config.gamma) == pytest.approx(
            config.sigma_0
        )
        assert compute_sigma(1.0, config.sigma_0, config.gamma) == pytest.approx(0.0)
        expected = config.sigma_0 * (1.0 - 0.5) ** config.gamma
        assert compute_sigma(0.5, config.sigma_0, config.gamma) == pytest.approx(
            expected
        )

    def test_high_affinity_small_sigma(self, config: Config) -> None:
        high = compute_sigma(0.9, config.sigma_0, config.gamma)
        low = compute_sigma(0.1, config.sigma_0, config.gamma)
        assert high < low


class TestGenerateMutants:
    def test_preserves_unit_norm(
        self, unit_embedding: np.ndarray, config: Config, rng: np.random.Generator
    ) -> None:
        mutants = generate_mutants(
            unit_embedding, 0.5, config.n_mutants, config.sigma_0, config.gamma, rng
        )
        norms = np.linalg.norm(mutants, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_returns_correct_shape(
        self, unit_embedding: np.ndarray, config: Config, rng: np.random.Generator
    ) -> None:
        mutants = generate_mutants(
            unit_embedding, 0.5, config.n_mutants, config.sigma_0, config.gamma, rng
        )
        assert mutants.shape == (config.n_mutants, 384)

    def test_mutants_are_distinct(
        self, unit_embedding: np.ndarray, config: Config, rng: np.random.Generator
    ) -> None:
        mutants = generate_mutants(
            unit_embedding, 0.5, config.n_mutants, config.sigma_0, config.gamma, rng
        )
        for i in range(config.n_mutants):
            for j in range(i + 1, config.n_mutants):
                assert not np.allclose(mutants[i], mutants[j])

    def test_high_affinity_mutants_closer(
        self, unit_embedding: np.ndarray, config: Config, rng: np.random.Generator
    ) -> None:
        """High affinity -> small sigma -> mutants closer to original."""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        high_mutants = generate_mutants(
            unit_embedding, 0.9, config.n_mutants, config.sigma_0, config.gamma, rng_a
        )
        low_mutants = generate_mutants(
            unit_embedding, 0.1, config.n_mutants, config.sigma_0, config.gamma, rng_b
        )
        high_dist = np.mean(1.0 - high_mutants @ unit_embedding)
        low_dist = np.mean(1.0 - low_mutants @ unit_embedding)
        assert high_dist < low_dist


class TestSelectBestMutant:
    def _make_query_and_original(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(123)
        q = rng.standard_normal(384).astype(np.float32)
        q /= np.linalg.norm(q)
        o = rng.standard_normal(384).astype(np.float32)
        o /= np.linalg.norm(o)
        return q, o

    def test_accepts_improving_mutant(self) -> None:
        query, original = self._make_query_and_original()
        # Create a mutant that is very close to query (high cosine)
        mutant = (query * 0.95 + original * 0.05).astype(np.float32)
        mutant /= np.linalg.norm(mutant)
        mutants = mutant.reshape(1, -1)
        result = select_best_mutant(
            query, original, original, mutants, delta=0.01, theta_anchor=0.0
        )
        assert result is not None

    def test_rejects_insufficient_improvement(self) -> None:
        query, original = self._make_query_and_original()
        # Create a mutant barely different from original
        mutant = original + np.random.default_rng(0).standard_normal(384).astype(
            np.float32
        ) * 1e-6
        mutant /= np.linalg.norm(mutant)
        mutants = mutant.reshape(1, -1)
        result = select_best_mutant(
            query, original, original, mutants, delta=0.01, theta_anchor=0.0
        )
        assert result is None

    def test_rejects_anchor_violation(self) -> None:
        query, original = self._make_query_and_original()
        # Mutant is essentially the query — far from original_embedding
        mutant = query.copy()
        mutants = mutant.reshape(1, -1)
        # Use a very strict anchor threshold
        result = select_best_mutant(
            query, original, original, mutants, delta=0.0, theta_anchor=0.999
        )
        assert result is None

    def test_returns_copy(self) -> None:
        query, original = self._make_query_and_original()
        mutant = (query * 0.95 + original * 0.05).astype(np.float32)
        mutant /= np.linalg.norm(mutant)
        mutants = mutant.reshape(1, -1)
        result = select_best_mutant(
            query, original, original, mutants, delta=0.01, theta_anchor=0.0
        )
        assert result is not None
        # Mutating result should not affect the input
        result[0] = 999.0
        assert mutants[0, 0] != 999.0

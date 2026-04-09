from __future__ import annotations

import numpy as np
import pytest

from gc_memory.config import Config
from gc_memory.mutation import compute_sigma, generate_adapter_mutants, select_best_adapter


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def base_embedding(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def query(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(384).astype(np.float32)
    return v / np.linalg.norm(v)


class TestComputeSigma:
    def test_formula(self, config: Config) -> None:
        assert compute_sigma(0.0, config.sigma_0, config.gamma) == pytest.approx(config.sigma_0)
        assert compute_sigma(1.0, config.sigma_0, config.gamma) == pytest.approx(0.0)
        expected = config.sigma_0 * (1.0 - 0.5) ** config.gamma
        assert compute_sigma(0.5, config.sigma_0, config.gamma) == pytest.approx(expected)

    def test_high_affinity_small_sigma(self, config: Config) -> None:
        assert compute_sigma(0.9, config.sigma_0, config.gamma) < compute_sigma(0.1, config.sigma_0, config.gamma)


class TestGenerateAdapterMutants:
    def test_effective_preserves_unit_norm(
        self, base_embedding: np.ndarray, query: np.ndarray, config: Config, rng: np.random.Generator,
    ) -> None:
        adapter = np.zeros(384, dtype=np.float32)
        _, eff = generate_adapter_mutants(
            adapter, base_embedding, query, 0.05, config.n_mutants, config.max_adapter_norm, rng,
        )
        np.testing.assert_allclose(np.linalg.norm(eff, axis=1), 1.0, atol=1e-6)

    def test_adapter_norms_clipped(
        self, base_embedding: np.ndarray, query: np.ndarray, config: Config, rng: np.random.Generator,
    ) -> None:
        adapter = np.zeros(384, dtype=np.float32)
        adapters, _ = generate_adapter_mutants(
            adapter, base_embedding, query, 1.0, 100, config.max_adapter_norm, rng,
        )
        norms = np.linalg.norm(adapters, axis=1)
        assert np.all(norms <= config.max_adapter_norm + 1e-6)

    def test_returns_correct_shape(
        self, base_embedding: np.ndarray, query: np.ndarray, config: Config, rng: np.random.Generator,
    ) -> None:
        adapter = np.zeros(384, dtype=np.float32)
        adapters, eff = generate_adapter_mutants(
            adapter, base_embedding, query, 0.05, config.n_mutants, config.max_adapter_norm, rng,
        )
        assert adapters.shape == (config.n_mutants, 384)
        assert eff.shape == (config.n_mutants, 384)

    def test_toward_query_bias(
        self, base_embedding: np.ndarray, query: np.ndarray, rng: np.random.Generator,
    ) -> None:
        """Mutants biased toward query should have higher avg cosine with query."""
        adapter = np.zeros(384, dtype=np.float32)
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        _, eff_toward = generate_adapter_mutants(
            adapter, base_embedding, query, 0.05, 50, 0.5, rng_a, toward_query=True,
        )
        _, eff_away = generate_adapter_mutants(
            adapter, base_embedding, query, 0.05, 50, 0.5, rng_b, toward_query=False,
        )
        mean_toward = float(np.mean(eff_toward @ query))
        mean_away = float(np.mean(eff_away @ query))
        assert mean_toward > mean_away


class TestSelectBestAdapter:
    def test_accepts_improving_mutant(self) -> None:
        rng = np.random.default_rng(123)
        query = rng.standard_normal(384).astype(np.float32)
        query /= np.linalg.norm(query)
        base = rng.standard_normal(384).astype(np.float32)
        base /= np.linalg.norm(base)
        eff_mutant = (query * 0.95 + base * 0.05).astype(np.float32)
        eff_mutant /= np.linalg.norm(eff_mutant)
        adapter_mutants = np.zeros((1, 384), dtype=np.float32)
        result = select_best_adapter(query, base, adapter_mutants, eff_mutant.reshape(1, -1), delta=0.005)
        assert result is not None

    def test_rejects_insufficient_improvement(self) -> None:
        rng = np.random.default_rng(123)
        query = rng.standard_normal(384).astype(np.float32)
        query /= np.linalg.norm(query)
        base = rng.standard_normal(384).astype(np.float32)
        base /= np.linalg.norm(base)
        eff_mutant = base + rng.standard_normal(384).astype(np.float32) * 1e-6
        eff_mutant /= np.linalg.norm(eff_mutant)
        adapter_mutants = np.zeros((1, 384), dtype=np.float32)
        result = select_best_adapter(query, base, adapter_mutants, eff_mutant.reshape(1, -1), delta=0.005)
        assert result is None

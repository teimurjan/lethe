from __future__ import annotations

import math

import numpy as np
import pytest

from gc_memory.config import Config
from gc_memory.entry import MemoryEntry, Tier, create_entry
from gc_memory.store import GCMemoryStore


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def config() -> Config:
    return Config()


def _make_entry(
    entry_id: str,
    direction: np.ndarray,
    rng: np.random.Generator,
    tier: Tier = Tier.NAIVE,
    affinity: float = 0.5,
    retrieval_count: int = 0,
    generation: int = 0,
) -> MemoryEntry:
    v = direction.astype(np.float32)
    entry = create_entry(entry_id, f"content-{entry_id}", v)
    entry.tier = tier
    entry.affinity = affinity
    entry.retrieval_count = retrieval_count
    entry.generation = generation
    return entry


def _basis_vector(idx: int, dim: int = 384) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


class TestRetrieve:
    def test_top_k_by_effective_score(self, config: Config, rng: np.random.Generator) -> None:
        query = _basis_vector(0)
        entries = [
            _make_entry("a", _basis_vector(0), rng),
            _make_entry("b", _basis_vector(1), rng),
        ]
        store = GCMemoryStore(entries, config, rng)
        results = store.retrieve(query, "test query", k=2)
        assert results[0][0].id == "a"

    def test_apoptotic_excluded(self, config: Config, rng: np.random.Generator) -> None:
        query = _basis_vector(0)
        entries = [
            _make_entry("alive", _basis_vector(0), rng),
            _make_entry("dead", _basis_vector(0), rng, tier=Tier.APOPTOTIC),
        ]
        store = GCMemoryStore(entries, config, rng)
        results = store.retrieve(query, "test", k=10)
        ids = [e.id for e, _ in results]
        assert "dead" not in ids

    def test_empty_store(self, config: Config, rng: np.random.Generator) -> None:
        store = GCMemoryStore([], config, rng)
        assert store.retrieve(_basis_vector(0), "test", k=10) == []


class TestTierTransitions:
    def test_naive_to_gc(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, retrieval_count=2)
        store = GCMemoryStore([entry], config, rng)
        query = _basis_vector(0)
        retrieved = store.retrieve(query, "test", k=1)
        store.update_after_retrieval(query, "test", retrieved, step=0)
        assert store.entries["a"].tier == Tier.GC

    def test_apoptosis(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, tier=Tier.GC, affinity=0.1)
        entry.last_retrieved_step = 0
        store = GCMemoryStore([entry], config, rng)
        store._check_tier_transitions(step=1500)
        assert store.entries["a"].tier == Tier.APOPTOTIC

    def test_memory_exempt_from_apoptosis(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, tier=Tier.MEMORY, affinity=0.05)
        entry.last_retrieved_step = 0
        store = GCMemoryStore([entry], config, rng)
        store._check_tier_transitions(step=5000)
        assert store.entries["a"].tier == Tier.MEMORY


class TestDecay:
    def test_reduces_affinity(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, affinity=0.5)
        entry.last_retrieved_step = 0
        store = GCMemoryStore([entry], config, rng)
        store.run_decay(step=200)
        expected = 0.5 * math.exp(-config.lambda_decay * config.decay_interval / 100)
        assert store.entries["a"].affinity == pytest.approx(expected)

    def test_memory_exempt(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, tier=Tier.MEMORY, affinity=0.8)
        entry.last_retrieved_step = 0
        store = GCMemoryStore([entry], config, rng)
        store.run_decay(step=1000)
        assert store.entries["a"].affinity == 0.8


class TestAdapterMutation:
    def test_adapter_changes_without_cross_encoder(self) -> None:
        """Without cross-encoder, mutation uses query cosine (fallback path)."""
        test_rng = np.random.default_rng(99)
        test_config = Config(sigma_0=0.01, gamma=0.0, delta=0.0, max_adapter_norm=0.5, n_mutants=20)
        query = test_rng.standard_normal(384).astype(np.float32)
        query /= np.linalg.norm(query)
        noise = test_rng.standard_normal(384).astype(np.float32) * 0.1
        entry = _make_entry("a", query + noise, test_rng, tier=Tier.GC, affinity=0.3)
        store = GCMemoryStore([entry], test_config, test_rng)  # no cross-encoder
        for step in range(30):
            retrieved = store.retrieve(query, "test query", k=1)
            store.update_after_retrieval(query, "test query", retrieved, step=step)
        assert float(np.linalg.norm(store.entries["a"].adapter)) > 0.0

    def test_adapter_norm_bounded(self) -> None:
        test_rng = np.random.default_rng(99)
        max_norm = 0.3
        test_config = Config(sigma_0=0.5, gamma=0.0, delta=0.0, max_adapter_norm=max_norm, n_mutants=20)
        query = test_rng.standard_normal(384).astype(np.float32)
        query /= np.linalg.norm(query)
        noise = test_rng.standard_normal(384).astype(np.float32) * 0.1
        entry = _make_entry("a", query + noise, test_rng, tier=Tier.GC, affinity=0.3)
        store = GCMemoryStore([entry], test_config, test_rng)
        for step in range(50):
            retrieved = store.retrieve(query, "test", k=1)
            store.update_after_retrieval(query, "test", retrieved, step=step)
        assert float(np.linalg.norm(store.entries["a"].adapter)) <= max_norm + 1e-6


class TestUtilities:
    def test_get_all_entries(self, config: Config, rng: np.random.Generator) -> None:
        entries = [
            _make_entry("a", _basis_vector(0), rng),
            _make_entry("b", _basis_vector(1), rng, tier=Tier.APOPTOTIC),
        ]
        store = GCMemoryStore(entries, config, rng)
        assert len(store.get_all_entries()) == 2

    def test_get_active_entries(self, config: Config, rng: np.random.Generator) -> None:
        entries = [
            _make_entry("a", _basis_vector(0), rng),
            _make_entry("b", _basis_vector(1), rng, tier=Tier.APOPTOTIC),
        ]
        store = GCMemoryStore(entries, config, rng)
        active = store.get_active_entries()
        assert len(active) == 1
        assert active[0].id == "a"

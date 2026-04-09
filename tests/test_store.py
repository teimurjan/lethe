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
    """Helper: create an entry with embedding pointing in given direction."""
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
    def test_top_k_by_effective_score(
        self, config: Config, rng: np.random.Generator
    ) -> None:
        # Entry a: aligned with query, entry b: orthogonal
        query = _basis_vector(0)
        entries = [
            _make_entry("a", _basis_vector(0), rng),
            _make_entry("b", _basis_vector(1), rng),
            _make_entry("c", _basis_vector(2), rng),
        ]
        store = GCMemoryStore(entries, config, rng)
        results = store.retrieve(query, k=2)
        assert len(results) == 2
        assert results[0][0].id == "a"
        assert results[0][1] > results[1][1]

    def test_memory_tier_weight_boost(
        self, config: Config, rng: np.random.Generator
    ) -> None:
        # Two entries with similar cosine, but one is memory-tier
        v = rng.standard_normal(384).astype(np.float32)
        v /= np.linalg.norm(v)
        entry_naive = _make_entry("naive", v, rng, tier=Tier.NAIVE)
        entry_mem = _make_entry("mem", v, rng, tier=Tier.MEMORY)
        store = GCMemoryStore([entry_naive, entry_mem], config, rng)
        results = store.retrieve(v, k=2)
        # Memory should rank first due to 1.15x weight
        assert results[0][0].id == "mem"
        assert results[0][1] > results[1][1]

    def test_apoptotic_excluded(
        self, config: Config, rng: np.random.Generator
    ) -> None:
        query = _basis_vector(0)
        entries = [
            _make_entry("alive", _basis_vector(0), rng),
            _make_entry("dead", _basis_vector(0), rng, tier=Tier.APOPTOTIC),
        ]
        store = GCMemoryStore(entries, config, rng)
        results = store.retrieve(query, k=10)
        ids = [e.id for e, _ in results]
        assert "dead" not in ids
        assert "alive" in ids

    def test_empty_store(self, config: Config, rng: np.random.Generator) -> None:
        store = GCMemoryStore([], config, rng)
        assert store.retrieve(_basis_vector(0), k=10) == []


class TestTierTransitions:
    def test_naive_to_gc(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, retrieval_count=2)
        store = GCMemoryStore([entry], config, rng)
        query = _basis_vector(0)
        retrieved = store.retrieve(query, k=1)
        # retrieval_count becomes 3 after update
        store.update_after_retrieval(query, retrieved, step=0)
        assert store.entries["a"].tier == Tier.GC

    def test_gc_to_memory(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry(
            "a",
            _basis_vector(0),
            rng,
            tier=Tier.GC,
            affinity=0.6,
            generation=5,
        )
        store = GCMemoryStore([entry], config, rng)
        # Query aligned with entry to push affinity above 0.75
        query = _basis_vector(0)
        retrieved = store.retrieve(query, k=1)
        store.update_after_retrieval(query, retrieved, step=0)
        # Affinity: (1-0.2)*0.6 + 0.2*1.0 = 0.68 — not enough yet
        # Need more queries to push affinity above 0.75
        for step in range(1, 10):
            retrieved = store.retrieve(query, k=1)
            store.update_after_retrieval(query, retrieved, step=step)
        assert store.entries["a"].affinity >= config.promote_memory_affinity
        assert store.entries["a"].tier == Tier.MEMORY

    def test_apoptosis(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, tier=Tier.GC, affinity=0.1)
        entry.last_retrieved_step = 0
        store = GCMemoryStore([entry], config, rng)
        # Trigger tier check at step > 1000
        query = _basis_vector(1)  # orthogonal — won't retrieve entry "a" at top
        # Force tier transition check directly
        store._check_tier_transitions(step=1500)
        assert store.entries["a"].tier == Tier.APOPTOTIC

    def test_memory_exempt_from_apoptosis(
        self, config: Config, rng: np.random.Generator
    ) -> None:
        entry = _make_entry(
            "a", _basis_vector(0), rng, tier=Tier.MEMORY, affinity=0.05
        )
        entry.last_retrieved_step = 0
        store = GCMemoryStore([entry], config, rng)
        store._check_tier_transitions(step=5000)
        assert store.entries["a"].tier == Tier.MEMORY


class TestAffinityUpdate:
    def test_ema_update(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, affinity=0.5)
        store = GCMemoryStore([entry], config, rng)
        query = _basis_vector(0)  # cosine = 1.0 with entry
        retrieved = [(store.entries["a"], 1.0)]
        store._update_affinities(query, retrieved)
        # affinity = (1-0.2)*0.5 + 0.2*1.0 = 0.6
        assert store.entries["a"].affinity == pytest.approx(0.6)

    def test_ema_with_low_similarity(
        self, config: Config, rng: np.random.Generator
    ) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, affinity=0.5)
        store = GCMemoryStore([entry], config, rng)
        query = _basis_vector(1)  # orthogonal, cosine ≈ 0
        retrieved = [(store.entries["a"], 0.0)]
        store._update_affinities(query, retrieved)
        # affinity = (1-0.2)*0.5 + 0.2*0.0 = 0.4
        assert store.entries["a"].affinity == pytest.approx(0.4)


class TestDecay:
    def test_reduces_affinity(self, config: Config, rng: np.random.Generator) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, affinity=0.5)
        entry.last_retrieved_step = 0
        store = GCMemoryStore([entry], config, rng)
        store.run_decay(step=200)
        # Fixed decay factor per interval: exp(-lambda * interval / 100)
        expected = 0.5 * math.exp(-config.lambda_decay * config.decay_interval / 100)
        assert store.entries["a"].affinity == pytest.approx(expected)

    def test_memory_exempt_from_decay(
        self, config: Config, rng: np.random.Generator
    ) -> None:
        entry = _make_entry("a", _basis_vector(0), rng, tier=Tier.MEMORY, affinity=0.8)
        entry.last_retrieved_step = 0
        store = GCMemoryStore([entry], config, rng)
        store.run_decay(step=1000)
        assert store.entries["a"].affinity == 0.8


class TestMutation:
    def test_generation_increments_on_accepted_mutation(self) -> None:
        # Dense vectors: entry is a perturbed version of query (cos ~0.85),
        # so small mutations can improve alignment
        test_rng = np.random.default_rng(99)
        test_config = Config(
            sigma_0=0.01, gamma=0.0, delta=0.0, theta_anchor=0.5, n_mutants=20,
        )
        query = test_rng.standard_normal(384).astype(np.float32)
        query /= np.linalg.norm(query)
        # Entry embedding: query + small noise (cos ~0.95 with query)
        noise = test_rng.standard_normal(384).astype(np.float32) * 0.1
        entry_vec = query + noise
        entry = _make_entry("a", entry_vec, test_rng, tier=Tier.GC, affinity=0.3)
        store = GCMemoryStore([entry], test_config, test_rng)
        initial_gen = entry.generation
        for step in range(30):
            retrieved = store.retrieve(query, k=1)
            store.update_after_retrieval(query, retrieved, step=step)
        assert store.entries["a"].generation > initial_gen

    def test_anchor_constraint_prevents_drift(
        self, rng: np.random.Generator,
    ) -> None:
        """With theta_anchor=0.99, almost no mutation should be accepted."""
        strict_config = Config(theta_anchor=0.99, delta=0.0)
        entry = _make_entry("a", _basis_vector(0), rng, tier=Tier.GC, affinity=0.3)
        store = GCMemoryStore([entry], strict_config, rng)
        query = _basis_vector(0)
        for step in range(20):
            retrieved = store.retrieve(query, k=1)
            store.update_after_retrieval(query, retrieved, step=step)
        # Embedding should barely have moved
        drift = 1.0 - float(
            np.dot(store.entries["a"].embedding, store.entries["a"].original_embedding)
        )
        assert drift < 0.01


class TestUtilities:
    def test_get_all_entries(self, config: Config, rng: np.random.Generator) -> None:
        entries = [
            _make_entry("a", _basis_vector(0), rng),
            _make_entry("b", _basis_vector(1), rng, tier=Tier.APOPTOTIC),
        ]
        store = GCMemoryStore(entries, config, rng)
        all_entries = store.get_all_entries()
        assert len(all_entries) == 2

    def test_get_active_entries(self, config: Config, rng: np.random.Generator) -> None:
        entries = [
            _make_entry("a", _basis_vector(0), rng),
            _make_entry("b", _basis_vector(1), rng, tier=Tier.APOPTOTIC),
        ]
        store = GCMemoryStore(entries, config, rng)
        active = store.get_active_entries()
        assert len(active) == 1
        assert active[0].id == "a"

    def test_decay_skips_current_step(
        self, config: Config, rng: np.random.Generator
    ) -> None:
        """Entry last retrieved at step=100, decay at step=100 -> no change."""
        entry = _make_entry("a", _basis_vector(0), rng, affinity=0.5)
        entry.last_retrieved_step = 100
        store = GCMemoryStore([entry], config, rng)
        store.run_decay(step=100)
        assert store.entries["a"].affinity == 0.5

"""Unit tests for lethe.entry (MemoryEntry, Tier, effective_embedding, create_entry)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lethe.entry import MemoryEntry, Tier, create_entry, effective_embedding


def test_create_entry_normalizes_embedding() -> None:
    raw = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32)  # norm = 5
    entry = create_entry("id1", "text", raw)
    assert math.isclose(float(np.linalg.norm(entry.base_embedding)), 1.0, rel_tol=1e-5)
    assert math.isclose(float(np.linalg.norm(entry.embedding)), 1.0, rel_tol=1e-5)


def test_create_entry_zero_norm_raises() -> None:
    with pytest.raises(ValueError, match="Zero-norm embedding"):
        create_entry("id1", "text", np.zeros(4, dtype=np.float32))


def test_create_entry_zeros_adapter() -> None:
    raw = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    entry = create_entry("x", "x", raw)
    assert entry.adapter.shape == (3,)
    assert np.allclose(entry.adapter, 0.0)
    assert entry.adapter.dtype == np.float32


def test_effective_embedding_zero_adapter_returns_normalized_base() -> None:
    base = np.array([0.6, 0.8, 0.0], dtype=np.float32)  # already unit-norm
    adapter = np.zeros(3, dtype=np.float32)
    eff = effective_embedding(base, adapter)
    assert np.allclose(eff, base, atol=1e-6)


def test_effective_embedding_is_normalized_when_adapter_nonzero() -> None:
    base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    adapter = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    eff = effective_embedding(base, adapter)
    assert math.isclose(float(np.linalg.norm(eff)), 1.0, rel_tol=1e-5)
    # direction should be in the (base+adapter) quadrant
    assert eff[0] > 0 and eff[1] > 0 and eff[2] == 0


def test_effective_embedding_degenerate_returns_base_copy() -> None:
    base = np.array([1.0, 0.0], dtype=np.float32)
    adapter = np.array([-1.0, 0.0], dtype=np.float32)  # cancels base exactly
    eff = effective_embedding(base, adapter)
    # norm(base + adapter) == 0 → returns base.copy()
    assert np.allclose(eff, base)
    assert eff is not base  # must be a copy


def test_tier_enum_values_and_membership() -> None:
    assert Tier.NAIVE.value == "naive"
    assert Tier.GC.value == "gc"
    assert Tier.MEMORY.value == "memory"
    assert Tier.APOPTOTIC.value == "apoptotic"
    assert len(Tier) == 4
    assert Tier("memory") is Tier.MEMORY


def test_memory_entry_defaults(make_entry) -> None:
    e = make_entry("e1")
    assert e.tier is Tier.NAIVE
    assert e.affinity == 0.5
    assert e.retrieval_count == 0
    assert e.generation == 0
    assert e.last_retrieved_step == 0
    assert e.suppression == 0.0
    assert e.session_id == ""
    assert e.turn_idx == 0

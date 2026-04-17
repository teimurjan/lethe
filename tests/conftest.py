"""Shared pytest fixtures for lethe production tests.

Mocks the external models (SentenceTransformer, CrossEncoder, AsyncAnthropic)
so tests are fast, deterministic, and don't require network / disk / GPU.
"""
from __future__ import annotations

# FAISS + PyTorch OMP workaround on macOS (harmless elsewhere).
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from lethe.entry import MemoryEntry, Tier, create_entry


# ---------- Determinism helpers ----------

TEST_DIM = 16  # small embedding dim → fast FAISS; fine for clustering tests


def _hash_to_unit_vector(text: str, dim: int = TEST_DIM) -> np.ndarray:
    """Deterministic unit vector derived from text via SHA-256 → seeded RNG."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


# ---------- Model mocks ----------

class MockBiEncoder:
    """Drop-in replacement for a sentence-transformers bi-encoder."""

    def __init__(self, dim: int = TEST_DIM) -> None:
        self.dim = dim

    def encode(
        self,
        texts: str | Iterable[str],
        normalize_embeddings: bool = True,  # noqa: ARG002 - always normalized
        show_progress_bar: bool = False,  # noqa: ARG002
        batch_size: int = 32,  # noqa: ARG002
    ) -> np.ndarray:
        if isinstance(texts, str):
            return _hash_to_unit_vector(texts, self.dim)
        arr = np.stack([_hash_to_unit_vector(t, self.dim) for t in texts])
        return arr.astype(np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim


class MockCrossEncoder:
    """Drop-in replacement scoring (query, content) pairs by shared tokens.

    Score is the raw count of shared lowercased whitespace-tokens between
    query and content, so it's deterministic and monotonically useful for
    test assertions ("the candidate containing more query words should
    outrank the one containing fewer").
    """

    def predict(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = np.zeros(len(pairs), dtype=np.float32)
        for i, (q, c) in enumerate(pairs):
            q_tokens = set(q.lower().split())
            c_tokens = set(c.lower().split())
            scores[i] = float(len(q_tokens & c_tokens))
        return scores


# ---------- Fixtures ----------

@pytest.fixture(autouse=True)
def isolate_registry(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point HOME at a throwaway dir so tests can't pollute the real ~/.lethe.

    Any test that cares about the registry can still write to
    ``Path.home() / ".lethe" / "projects.json"`` — it just lands under a tmp
    dir that pytest cleans up.
    """
    fake_home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(fake_home))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def mock_bi_encoder() -> MockBiEncoder:
    return MockBiEncoder()


@pytest.fixture
def mock_cross_encoder() -> MockCrossEncoder:
    return MockCrossEncoder()


@pytest.fixture
def tmp_store_path(tmp_path: Path) -> Path:
    """A temporary directory for a MemoryStore. Cleaned up by pytest."""
    return tmp_path / "store"


@pytest.fixture
def make_entry():
    """Factory: build a MemoryEntry with a deterministic embedding."""
    def _make(
        entry_id: str,
        content: str | None = None,
        tier: Tier = Tier.NAIVE,
        affinity: float = 0.5,
        retrieval_count: int = 0,
        session_id: str = "",
        turn_idx: int = 0,
    ) -> MemoryEntry:
        text = content if content is not None else f"content-{entry_id}"
        emb = _hash_to_unit_vector(text)
        entry = create_entry(entry_id, text, emb, session_id=session_id, turn_idx=turn_idx)
        entry.tier = tier
        entry.affinity = affinity
        entry.retrieval_count = retrieval_count
        return entry
    return _make


@dataclass
class _SampleSet:
    entries: list[MemoryEntry]
    by_id: dict[str, MemoryEntry]


@pytest.fixture
def sample_entries(make_entry) -> _SampleSet:
    """A handful of distinctly-worded entries useful in several tests."""
    items = [
        make_entry("a", "I prefer window seats on flights"),
        make_entry("b", "My wife needs aisle seats"),
        make_entry("c", "I work at Google as a software engineer"),
        make_entry("d", "We flew from Amsterdam to Paris in January"),
        make_entry("e", "Meeting with Sam about the Q3 roadmap"),
    ]
    return _SampleSet(entries=items, by_id={e.id: e for e in items})

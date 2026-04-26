"""Unit tests for lethe.dedup."""
from __future__ import annotations

import numpy as np
import pytest

from lethe.dedup import content_hash, is_near_duplicate


def test_content_hash_deterministic() -> None:
    assert content_hash("hello world") == content_hash("hello world")


def test_content_hash_differs_for_different_content() -> None:
    assert content_hash("hello world") != content_hash("hello universe")


def test_content_hash_is_hex_sha256_length() -> None:
    h = content_hash("anything")
    assert len(h) == 64
    int(h, 16)  # must be valid hex — raises if not


def test_is_near_duplicate_above_threshold_returns_index() -> None:
    # Two unit vectors with cosine = 0.99
    existing = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    new = np.array([0.99, 0.1, 0.0], dtype=np.float32)
    new /= np.linalg.norm(new)
    idx = is_near_duplicate(new, existing, threshold=0.95)
    assert idx == 0


def test_is_near_duplicate_below_threshold_returns_none() -> None:
    # Orthogonal → cosine = 0
    existing = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    new = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert is_near_duplicate(new, existing, threshold=0.5) is None


def test_is_near_duplicate_empty_matrix_returns_none() -> None:
    existing = np.zeros((0, 4), dtype=np.float32)
    new = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert is_near_duplicate(new, existing) is None


def test_is_near_duplicate_picks_most_similar() -> None:
    # Three candidates with different cosines to the query
    existing = np.array(
        [
            [0.0, 1.0, 0.0],   # cos = 0
            [0.9, 0.436, 0.0], # cos ≈ 0.9
            [0.99, 0.141, 0.0],# cos ≈ 0.99
        ],
        dtype=np.float32,
    )
    # Normalize rows for a fair cosine test
    existing = existing / np.linalg.norm(existing, axis=1, keepdims=True)
    new = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    idx = is_near_duplicate(new, existing, threshold=0.95)
    assert idx == 2  # the .99 row

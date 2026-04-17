"""Tests for UnionStore (cross-project retrieval)."""
from __future__ import annotations

import shutil
from pathlib import Path

from lethe.memory_store import MemoryStore
from lethe.union_store import UnionStore


def _seed_project(
    root: Path, contents: list[str], *, bi_encoder, cross_encoder
) -> None:
    index_dir = root / ".lethe" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(
        index_dir, bi_encoder=bi_encoder, cross_encoder=cross_encoder, dim=bi_encoder.dim,
    )
    for i, c in enumerate(contents):
        store.add(c, session_id="seed", turn_idx=i)
    store.save()
    store.close()


def test_union_searches_across_two_projects_and_tags_hits(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder
) -> None:
    proj_a = tmp_path / "proj_a"
    proj_b = tmp_path / "proj_b"
    _seed_project(
        proj_a,
        ["MongoDB connection pool spiked to 3k", "Coffee preference note"],
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
    )
    _seed_project(
        proj_b,
        ["Payload defaultDepth 10 to 2 cut query count", "Blog post landing page"],
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
    )

    union = UnionStore(
        [proj_a, proj_b],
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        hits = union.retrieve("Payload defaultDepth MongoDB pool", k=5)
    finally:
        union.close()

    assert len(hits) > 0
    slugs = {h.project_slug for h in hits}
    # At least one hit per project when both are represented in the query terms.
    assert len(slugs) >= 2
    for h in hits:
        assert h.project_root in (proj_a.resolve(), proj_b.resolve())


def test_union_skips_uninitialized_project(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder
) -> None:
    real = tmp_path / "real"
    empty = tmp_path / "empty"  # no .lethe/ at all
    empty.mkdir()
    _seed_project(
        real,
        ["mongodb pool spike during release"],
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
    )
    union = UnionStore(
        [real, empty],
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        hits = union.retrieve("mongodb pool", k=5)
    finally:
        union.close()
    assert hits
    assert all(h.project_root == real.resolve() for h in hits)


def test_union_is_read_only(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder
) -> None:
    proj = tmp_path / "proj"
    _seed_project(
        proj,
        ["entry alpha about engines", "entry beta about coffee"],
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
    )

    # Snapshot the DB file bytes before retrieval.
    dbfile = proj / ".lethe" / "index" / "lethe.duckdb"
    before = dbfile.read_bytes()

    union = UnionStore(
        [proj], bi_encoder=mock_bi_encoder, cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        union.retrieve("coffee engines", k=5)
    finally:
        union.close()

    after = dbfile.read_bytes()
    assert before == after, "UnionStore.retrieve must not mutate the per-project DB"


def test_union_handles_stale_faiss(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder
) -> None:
    proj = tmp_path / "proj"
    _seed_project(
        proj,
        ["coffee arabica flavor profile", "espresso preparation"],
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
    )
    # Remove FAISS index file — BM25 should still cover it.
    (proj / ".lethe" / "index" / "faiss.index").unlink()

    union = UnionStore(
        [proj], bi_encoder=mock_bi_encoder, cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        hits = union.retrieve("coffee arabica", k=5)
    finally:
        union.close()
    assert hits
    assert hits[0].content == "coffee arabica flavor profile"

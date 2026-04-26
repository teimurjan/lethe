"""Tests for ``lethe.tui``. Kept minimal — rendering is covered by the
layout debug and manual pilot testing. What we lock down here are the
pure-Python paths that previously caused crashes."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("textual")

from lethe.memory_store import MemoryStore  # noqa: E402
from lethe.tui import Scope, SearchHit, _expand  # noqa: E402
from lethe.union_store import UnionStore  # noqa: E402


def _seed(root: Path, contents: list[str], *, bi_encoder, cross_encoder) -> None:
    idx = root / ".lethe" / "index"
    idx.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(
        idx, bi_encoder=bi_encoder, cross_encoder=cross_encoder, dim=bi_encoder.dim,
    )
    for i, c in enumerate(contents):
        store.add(c, session_id="seed", turn_idx=i)
    store.save()
    store.close()


def test_expand_does_not_reopen_duckdb_while_union_store_is_active(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    """Regression: opening a fresh ``duckdb.connect(path)`` via MemoryDB
    while ``UnionStore`` has that file ATTACHed raises
    ``Binder Error: Unique file handle conflict``. ``_expand`` used to do
    this on every Enter in the all-projects scope."""
    proj = tmp_path / "proj"
    _seed(
        proj, ["MongoDB pool spiked to 3k during release"],
        bi_encoder=mock_bi_encoder, cross_encoder=mock_cross_encoder,
    )

    union = UnionStore(
        [proj], bi_encoder=mock_bi_encoder, cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        hits = union.retrieve("MongoDB pool", k=3)
        assert hits
        # _expand must NOT crash while UnionStore still holds the ATTACH.
        hit = SearchHit(
            id=hits[0].id, content=hits[0].content, score=hits[0].score,
            project_slug=hits[0].project_slug,
        )
        expanded = _expand(Scope(project=None), hit)
    finally:
        union.close()

    assert expanded == hit.content
    assert "MongoDB" in expanded

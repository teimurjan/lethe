"""Tests for lethe.markdown_store.

Covers chunking, hash stability, anchor parsing, and the reindex diff logic
against a real (mocked-encoder) MemoryStore.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lethe.markdown_store import (
    Chunk,
    MarkdownStore,
    chunk_hash,
    parse_anchor,
    split_into_chunks,
)
from lethe.memory_store import MemoryStore


# ---------- Pure chunking ----------

def test_chunk_hash_is_stable_and_content_addressed() -> None:
    h1 = chunk_hash("## Heading\n- bullet")
    h2 = chunk_hash("  ## Heading\n- bullet  \n")  # whitespace-insensitive
    assert h1 == h2
    assert len(h1) == 16
    assert h1 != chunk_hash("## Heading\n- different bullet")


def test_split_into_chunks_respects_double_headings(tmp_path: Path) -> None:
    src = tmp_path / "2026-04-16.md"
    md = """# 2026-04-16

Preamble should be captured as an anonymous chunk.

## Session 09:00

### 09:00
<!-- session:s1 turn:t1 transcript:/tmp/x.jsonl -->
- First bullet
- Second bullet

## Session 10:00
- Just a bullet under the session heading
"""
    chunks = split_into_chunks(md, src)
    # The "## Session 09:00" section has no body of its own (just the heading
    # followed by a deeper heading), so it's dropped to keep retrieval clean.
    assert [c.heading for c in chunks] == [
        "",  # preamble block under the top-level `#` heading
        "09:00",  # ### boundary — the actual turn entry
        "Session 10:00",  # ## boundary — body follows the heading
    ]
    assert all(c.source == src for c in chunks)
    ids = {c.id for c in chunks}
    assert len(ids) == len(chunks)
    assert all(len(cid) == 16 for cid in ids)


def test_split_into_chunks_drops_heading_only_sections(tmp_path: Path) -> None:
    """Duplicate SessionStart fires produce heading-only chunks that
    pollute retrieval. They must be dropped at index time."""
    src = tmp_path / "dup.md"
    md = """# 2026-04-16

## Session 13:45

## Session 13:45

### 13:45
<!-- session:s1 turn:t1 transcript:/tmp/x.jsonl -->
- real content
"""
    chunks = split_into_chunks(md, src)
    # Preamble is title-only; both "## Session 13:45" are body-less; only the
    # final "### 13:45" chunk has bullets and should survive.
    assert [c.heading for c in chunks] == ["13:45"]
    turn_chunk = next(c for c in chunks if c.heading == "13:45")
    assert turn_chunk.anchor == {
        "session": "s1", "turn": "t1", "transcript": "/tmp/x.jsonl",
    }


def test_split_into_chunks_ignores_top_level_single_hash(tmp_path: Path) -> None:
    md = "# Title\nbody\n## Real Section\n- b\n"
    src = tmp_path / "x.md"
    chunks = split_into_chunks(md, src)
    # The leading "# Title\nbody" becomes one anonymous chunk; "## Real Section" is the second.
    assert len(chunks) == 2
    assert chunks[1].heading == "Real Section"


def test_split_into_chunks_drops_empty_whitespace(tmp_path: Path) -> None:
    chunks = split_into_chunks("\n\n   \n", tmp_path / "blank.md")
    assert chunks == []


# ---------- Anchor parsing ----------

def test_parse_anchor_extracts_session_turn_transcript() -> None:
    chunk = "<!-- session:abc turn:def transcript:/tmp/x.jsonl -->\n- body\n"
    assert parse_anchor(chunk) == {
        "session": "abc",
        "turn": "def",
        "transcript": "/tmp/x.jsonl",
    }


def test_parse_anchor_returns_none_when_missing() -> None:
    assert parse_anchor("- just a bullet") is None


def test_parse_anchor_handles_path_with_spaces() -> None:
    chunk = "<!-- session:s turn:t transcript:/path with spaces/x.jsonl -->"
    anchor = parse_anchor(chunk)
    assert anchor is not None
    assert anchor["transcript"] == "/path with spaces/x.jsonl"


# ---------- MarkdownStore scan / reindex ----------

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_scan_reads_all_md_files_in_order(tmp_path: Path) -> None:
    memory = tmp_path / "memory"
    _write(memory / "2026-04-15.md", "## A\n- a\n")
    _write(memory / "2026-04-16.md", "## B\n- b\n")
    store = MarkdownStore(memory_dir=memory, index_dir=tmp_path / "index")
    chunks = store.scan()
    assert [c.heading for c in chunks] == ["A", "B"]


def test_reindex_adds_new_chunks_and_persists_chunk_map(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder
) -> None:
    memory = tmp_path / "memory"
    _write(
        memory / "2026-04-16.md",
        "## Session 09:00\n- first bullet\n\n## Session 10:00\n- second bullet\n",
    )

    md = MarkdownStore(memory_dir=memory, index_dir=tmp_path / "index")
    store = MemoryStore(
        tmp_path / "store",
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        counts = md.reindex(store)
    finally:
        store.close()

    assert counts == {"added": 2, "removed": 0, "unchanged": 0, "total": 2}
    # chunk map file exists and has both chunk ids
    chunk_map = md._load_chunk_map()
    assert len(chunk_map) == 2
    assert all(len(cid) == 16 for cid in chunk_map)


def test_reindex_is_idempotent(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder
) -> None:
    memory = tmp_path / "memory"
    _write(memory / "today.md", "## A\n- alpha\n")

    md = MarkdownStore(memory_dir=memory, index_dir=tmp_path / "index")

    def open_store() -> MemoryStore:
        return MemoryStore(
            tmp_path / "store",
            bi_encoder=mock_bi_encoder,
            cross_encoder=mock_cross_encoder,
            dim=mock_bi_encoder.dim,
        )

    first = open_store()
    try:
        counts1 = md.reindex(first)
        first.save()
    finally:
        first.close()

    second = open_store()
    try:
        counts2 = md.reindex(second)
    finally:
        second.close()

    assert counts1["added"] == 1
    assert counts2 == {"added": 0, "removed": 0, "unchanged": 1, "total": 1}


def test_reindex_removes_chunks_deleted_from_markdown(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder
) -> None:
    memory = tmp_path / "memory"
    md_path = memory / "today.md"
    _write(md_path, "## A\n- alpha\n\n## B\n- beta\n")

    md = MarkdownStore(memory_dir=memory, index_dir=tmp_path / "index")

    store1 = MemoryStore(
        tmp_path / "store",
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        md.reindex(store1)
        store1.save()
    finally:
        store1.close()

    # Drop the "B" chunk and reindex.
    _write(md_path, "## A\n- alpha\n")
    store2 = MemoryStore(
        tmp_path / "store",
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        counts = md.reindex(store2)
        assert counts["added"] == 0
        assert counts["removed"] == 1
        assert counts["total"] == 1
        assert len(store2.entries) == 1
    finally:
        store2.close()


def test_get_chunk_returns_full_markdown(
    tmp_path: Path, mock_bi_encoder, mock_cross_encoder
) -> None:
    memory = tmp_path / "memory"
    body = "## Session 09:00\n<!-- session:s turn:t transcript:/x -->\n- bullet\n"
    _write(memory / "today.md", body)
    md = MarkdownStore(memory_dir=memory, index_dir=tmp_path / "index")

    store = MemoryStore(
        tmp_path / "store",
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=mock_bi_encoder.dim,
    )
    try:
        md.reindex(store)
    finally:
        store.close()

    [chunk] = md.scan()
    expanded = md.get_chunk(chunk.id)
    assert expanded is not None
    assert "bullet" in expanded
    assert parse_anchor(expanded) == {
        "session": "s", "turn": "t", "transcript": "/x",
    }


def test_get_chunk_returns_none_for_unknown_id(tmp_path: Path) -> None:
    md = MarkdownStore(memory_dir=tmp_path / "memory", index_dir=tmp_path / "index")
    assert md.get_chunk("deadbeef" * 2) is None


def test_chunk_anchor_property_pulls_from_content() -> None:
    chunk = Chunk(
        id="a" * 16,
        source=Path("x.md"),
        heading="h",
        content="<!-- session:s turn:t transcript:/p -->\n- body",
    )
    assert chunk.anchor == {"session": "s", "turn": "t", "transcript": "/p"}

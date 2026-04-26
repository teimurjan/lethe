"""Unit tests for lethe.db.MemoryDB backed by DuckDB."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lethe.db import MemoryDB
from lethe.entry import Tier


@pytest.fixture
def db(tmp_path: Path) -> MemoryDB:
    """Fresh per-test DuckDB file."""
    return MemoryDB(tmp_path / "test.duckdb")


def test_insert_entry_persists_all_fields(db: MemoryDB, make_entry) -> None:
    e = make_entry("x", content="hello world", session_id="sess-1", turn_idx=3)
    e.affinity = 0.8
    e.retrieval_count = 5
    e.last_retrieved_step = 42
    e.tier = Tier.MEMORY
    e.suppression = 0.25
    db.insert_entry(e)

    rows = db.load_all_entries()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "x"
    assert row["content"] == "hello world"
    assert row["session_id"] == "sess-1"
    assert row["turn_idx"] == 3
    assert row["tier"] == "memory"
    assert row["affinity"] == pytest.approx(0.8)
    assert row["retrieval_count"] == 5
    assert row["last_retrieved_step"] == 42
    assert row["suppression"] == pytest.approx(0.25)


def test_has_content_hash_detects_existing(db: MemoryDB, make_entry) -> None:
    db.insert_entry(make_entry("a", content="unique-one"))
    assert db.has_content_hash("unique-one") is True
    assert db.has_content_hash("never-stored") is False


def test_insert_replaces_on_same_id(db: MemoryDB, make_entry) -> None:
    first = make_entry("dup", content="v1")
    second = make_entry("dup", content="v2")
    db.insert_entry(first)
    db.insert_entry(second)
    rows = db.load_all_entries()
    assert len(rows) == 1
    assert rows[0]["content"] == "v2"


def test_update_entry_persists_mutable_fields(db: MemoryDB, make_entry) -> None:
    e = make_entry("u")
    db.insert_entry(e)

    e.affinity = 0.91
    e.retrieval_count = 12
    e.last_retrieved_step = 99
    e.suppression = 0.33
    e.tier = Tier.GC
    db.update_entry(e)

    row = db.load_all_entries()[0]
    assert row["affinity"] == pytest.approx(0.91)
    assert row["retrieval_count"] == 12
    assert row["last_retrieved_step"] == 99
    assert row["suppression"] == pytest.approx(0.33)
    assert row["tier"] == "gc"


def test_batch_update_entries(db: MemoryDB, make_entry) -> None:
    es = [make_entry(f"e{i}") for i in range(5)]
    for e in es:
        db.insert_entry(e)
    for i, e in enumerate(es):
        e.affinity = 0.1 * i
        e.retrieval_count = i
    db.batch_update_entries(es)
    rows = {r["id"]: r for r in db.load_all_entries()}
    for i in range(5):
        assert rows[f"e{i}"]["affinity"] == pytest.approx(0.1 * i)
        assert rows[f"e{i}"]["retrieval_count"] == i


def test_load_all_entries_skips_apoptotic(db: MemoryDB, make_entry) -> None:
    alive = make_entry("alive", tier=Tier.NAIVE)
    dead = make_entry("dead", tier=Tier.APOPTOTIC)
    db.insert_entry(alive)
    db.insert_entry(dead)
    ids = [r["id"] for r in db.load_all_entries()]
    assert "alive" in ids
    assert "dead" not in ids


def test_delete_entry_removes_row(db: MemoryDB, make_entry) -> None:
    db.insert_entry(make_entry("x"))
    assert db.count() == 1
    db.delete_entry("x")
    assert db.count() == 0
    assert db.load_all_entries() == []


def test_count_excludes_apoptotic(db: MemoryDB, make_entry) -> None:
    db.insert_entry(make_entry("a", tier=Tier.NAIVE))
    db.insert_entry(make_entry("b", tier=Tier.MEMORY))
    db.insert_entry(make_entry("c", tier=Tier.APOPTOTIC))
    assert db.count() == 2


def test_stats_get_set_roundtrip(db: MemoryDB) -> None:
    assert db.get_stat("step", default="0") == "0"
    db.set_stat("step", "7")
    assert db.get_stat("step") == "7"
    db.set_stat("step", "8")  # overwrite
    assert db.get_stat("step") == "8"


def test_rescue_cache_insert_load_clear(db: MemoryDB) -> None:
    db.insert_rescue("qhash-1", "entry-a", 3.5)
    db.insert_rescue("qhash-1", "entry-b", 1.2)
    rows = db.load_rescue_entries()
    assert len(rows) == 2
    scores = sorted(r["xenc_score"] for r in rows)
    assert scores == [pytest.approx(1.2), pytest.approx(3.5)]
    db.clear_rescue()
    assert db.load_rescue_entries() == []


def test_close_makes_conn_unusable(db: MemoryDB) -> None:
    db.close()
    # Subsequent queries on a closed DuckDB connection raise.
    with pytest.raises(Exception):  # noqa: BLE001 — duckdb.ConnectionException
        db.count()


def test_legacy_sqlite_file_raises(tmp_path: Path) -> None:
    """If a pre-migration lethe.db exists and no lethe.duckdb is present,
    we refuse to open silently — users should `lethe reset` explicitly."""
    legacy = tmp_path / "lethe.db"
    # Create a minimal valid SQLite file so the check triggers on existence.
    conn = sqlite3.connect(str(legacy))
    conn.execute("CREATE TABLE placeholder (x INTEGER)")
    conn.close()

    with pytest.raises(RuntimeError, match="lethe reset"):
        MemoryDB(tmp_path / "lethe.duckdb")


def test_legacy_and_new_coexist_allows_open(tmp_path: Path) -> None:
    """If both files exist, we prefer the new DuckDB and don't error out."""
    # Create DuckDB first so the legacy-check branch doesn't fire.
    MemoryDB(tmp_path / "lethe.duckdb").close()

    # Now drop a legacy sqlite file alongside it; subsequent opens must succeed.
    legacy = tmp_path / "lethe.db"
    conn = sqlite3.connect(str(legacy))
    conn.execute("CREATE TABLE placeholder (x INTEGER)")
    conn.close()

    db = MemoryDB(tmp_path / "lethe.duckdb")
    assert db.count() == 0
    db.close()

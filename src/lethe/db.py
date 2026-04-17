"""DuckDB persistence for lethe entries and rescue cache.

The metadata store is a per-project DuckDB file at
``<project>/.lethe/index/lethe.duckdb``. DuckDB was chosen over SQLite so that
``lethe search --all`` can ATTACH many project databases in one connection
without the 10/125 SQLite ceiling; see ``lethe.union_store``.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import numpy.typing as npt

from lethe.entry import MemoryEntry


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    session_id TEXT DEFAULT '',
    turn_idx INTEGER DEFAULT 0,
    tier TEXT DEFAULT 'naive',
    affinity DOUBLE DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    last_retrieved_step INTEGER DEFAULT 0,
    content_hash TEXT,
    suppression DOUBLE DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE SEQUENCE IF NOT EXISTS rescue_cache_id_seq;

CREATE TABLE IF NOT EXISTS rescue_cache (
    id BIGINT PRIMARY KEY DEFAULT nextval('rescue_cache_id_seq'),
    query_embedding_hash TEXT,
    entry_id TEXT,
    xenc_score DOUBLE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS cluster_suppression (
    cluster_id INTEGER NOT NULL,
    entry_id TEXT NOT NULL,
    suppression_score DOUBLE NOT NULL,
    step_updated INTEGER NOT NULL,
    PRIMARY KEY (cluster_id, entry_id)
);

CREATE TABLE IF NOT EXISTS cluster_centroids (
    cluster_id INTEGER PRIMARY KEY,
    centroid BLOB NOT NULL,
    dim INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_tier ON entries(tier);
CREATE INDEX IF NOT EXISTS idx_entries_hash ON entries(content_hash);
CREATE INDEX IF NOT EXISTS idx_rescue_entry ON rescue_cache(entry_id);
CREATE INDEX IF NOT EXISTS idx_cluster_suppression_entry
    ON cluster_suppression(entry_id);
"""


def _rows_as_dicts(cur: Any) -> list[dict[str, object]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


class MemoryDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        legacy = self.path.with_name("lethe.db")
        if legacy.exists() and not self.path.exists():
            raise RuntimeError(
                "Legacy SQLite index detected at "
                f"{legacy}. lethe now uses DuckDB; run `lethe reset` then "
                "`lethe index` to rebuild from markdown."
            )
        self._conn = duckdb.connect(str(self.path))
        for stmt in _split_statements(SCHEMA):
            self._conn.execute(stmt)

    def close(self) -> None:
        self._conn.close()

    # --- Entries ---

    def insert_entry(self, entry: MemoryEntry) -> None:
        content_hash = hashlib.sha256(entry.content.encode()).hexdigest()
        self._conn.execute(
            """INSERT OR REPLACE INTO entries
            (id, content, session_id, turn_idx, tier, affinity,
             retrieval_count, last_retrieved_step, content_hash, suppression)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [entry.id, entry.content, entry.session_id, entry.turn_idx,
             entry.tier.value, entry.affinity, entry.retrieval_count,
             entry.last_retrieved_step, content_hash, entry.suppression],
        )

    def has_content_hash(self, content: str) -> bool:
        h = hashlib.sha256(content.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT 1 FROM entries WHERE content_hash = ? LIMIT 1", [h]
        ).fetchone()
        return row is not None

    def update_entry(self, entry: MemoryEntry) -> None:
        self._conn.execute(
            """UPDATE entries SET tier=?, affinity=?, retrieval_count=?,
            last_retrieved_step=?, suppression=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?""",
            [entry.tier.value, entry.affinity, entry.retrieval_count,
             entry.last_retrieved_step, entry.suppression, entry.id],
        )

    def batch_update_entries(self, entries: list[MemoryEntry]) -> None:
        if not entries:
            return
        self._conn.executemany(
            """UPDATE entries SET tier=?, affinity=?, retrieval_count=?,
            last_retrieved_step=?, suppression=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?""",
            [[e.tier.value, e.affinity, e.retrieval_count,
              e.last_retrieved_step, e.suppression, e.id] for e in entries],
        )

    def load_all_entries(self) -> list[dict[str, object]]:
        cur = self._conn.execute(
            "SELECT id, content, session_id, turn_idx, tier, affinity, "
            "retrieval_count, last_retrieved_step, suppression "
            "FROM entries WHERE tier != 'apoptotic'"
        )
        return _rows_as_dicts(cur)

    def delete_entry(self, entry_id: str) -> None:
        self._conn.execute("DELETE FROM entries WHERE id=?", [entry_id])

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM entries WHERE tier != 'apoptotic'"
        ).fetchone()
        return int(row[0]) if row else 0

    # --- Rescue cache ---

    def insert_rescue(self, query_hash: str, entry_id: str, xenc_score: float) -> None:
        self._conn.execute(
            "INSERT INTO rescue_cache (query_embedding_hash, entry_id, xenc_score) "
            "VALUES (?, ?, ?)",
            [query_hash, entry_id, xenc_score],
        )

    def load_rescue_entries(self) -> list[dict[str, object]]:
        cur = self._conn.execute(
            "SELECT query_embedding_hash, entry_id, xenc_score FROM rescue_cache"
        )
        return _rows_as_dicts(cur)

    def clear_rescue(self) -> None:
        self._conn.execute("DELETE FROM rescue_cache")

    # --- Stats ---

    def get_stat(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM stats WHERE key=?", [key]
        ).fetchone()
        return str(row[0]) if row else default

    def set_stat(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)",
            [key, value],
        )

    # --- Clustered RIF state ---

    def save_cluster_suppression(
        self,
        scores: dict[int, dict[str, float]],
        last_updated: dict[int, dict[str, int]],
    ) -> None:
        """Rewrite cluster_suppression table from in-memory state.

        Full replacement is fine at typical sizes (n_clusters * entries ~=
        few thousand rows). Avoids drift between memory and disk.
        """
        self._conn.execute("DELETE FROM cluster_suppression")
        rows: list[list[object]] = []
        for cid, entry_scores in scores.items():
            cid_updates = last_updated.get(cid, {})
            for eid, score in entry_scores.items():
                rows.append(
                    [int(cid), eid, float(score), int(cid_updates.get(eid, 0))]
                )
        if rows:
            self._conn.executemany(
                "INSERT INTO cluster_suppression "
                "(cluster_id, entry_id, suppression_score, step_updated) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )

    def load_cluster_suppression(
        self,
    ) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, int]]]:
        cur = self._conn.execute(
            "SELECT cluster_id, entry_id, suppression_score, step_updated "
            "FROM cluster_suppression"
        )
        scores: dict[int, dict[str, float]] = {}
        last_updated: dict[int, dict[str, int]] = {}
        for cid, eid, score, step in cur.fetchall():
            scores.setdefault(int(cid), {})[str(eid)] = float(score)
            last_updated.setdefault(int(cid), {})[str(eid)] = int(step)
        return scores, last_updated

    def save_cluster_centroids(
        self, centroids: npt.NDArray[np.float32] | None,
    ) -> None:
        self._conn.execute("DELETE FROM cluster_centroids")
        if centroids is None or centroids.size == 0:
            return
        arr = centroids.astype(np.float32)
        dim = int(arr.shape[1])
        rows = [
            [i, arr[i].tobytes(), dim] for i in range(arr.shape[0])
        ]
        self._conn.executemany(
            "INSERT INTO cluster_centroids (cluster_id, centroid, dim) "
            "VALUES (?, ?, ?)",
            rows,
        )

    def load_cluster_centroids(self) -> npt.NDArray[np.float32] | None:
        cur = self._conn.execute(
            "SELECT cluster_id, centroid, dim FROM cluster_centroids "
            "ORDER BY cluster_id"
        )
        rows = cur.fetchall()
        if not rows:
            return None
        dim = int(rows[0][2])
        arr = np.empty((len(rows), dim), dtype=np.float32)
        for i, (_, blob, _dim) in enumerate(rows):
            arr[i] = np.frombuffer(bytes(blob), dtype=np.float32)
        return arr


def _split_statements(sql: str) -> list[str]:
    """DuckDB's Python API doesn't have executescript; run each statement once."""
    return [stmt.strip() for stmt in sql.split(";") if stmt.strip()]

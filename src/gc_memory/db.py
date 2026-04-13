"""SQLite persistence for gc-memory entries and rescue cache."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from gc_memory.entry import MemoryEntry, Tier


SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    session_id TEXT DEFAULT '',
    turn_idx INTEGER DEFAULT 0,
    tier TEXT DEFAULT 'naive',
    affinity REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    last_retrieved_step INTEGER DEFAULT 0,
    content_hash TEXT,
    suppression REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rescue_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_embedding_hash TEXT,
    entry_id TEXT,
    xenc_score REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_entries_tier ON entries(tier);
CREATE INDEX IF NOT EXISTS idx_entries_hash ON entries(content_hash);
CREATE INDEX IF NOT EXISTS idx_rescue_entry ON rescue_cache(entry_id);
"""


class MemoryDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

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
            (entry.id, entry.content, entry.session_id, entry.turn_idx,
             entry.tier.value, entry.affinity, entry.retrieval_count,
             entry.last_retrieved_step, content_hash, entry.suppression),
        )
        self._conn.commit()

    def has_content_hash(self, content: str) -> bool:
        """Check if content already exists (exact dedup)."""
        h = hashlib.sha256(content.encode()).hexdigest()
        row = self._conn.execute(
            "SELECT 1 FROM entries WHERE content_hash = ? LIMIT 1", (h,)
        ).fetchone()
        return row is not None

    def update_entry(self, entry: MemoryEntry) -> None:
        self._conn.execute(
            """UPDATE entries SET tier=?, affinity=?, retrieval_count=?,
            last_retrieved_step=?, suppression=?, updated_at=datetime('now')
            WHERE id=?""",
            (entry.tier.value, entry.affinity, entry.retrieval_count,
             entry.last_retrieved_step, entry.suppression, entry.id),
        )
        self._conn.commit()

    def batch_update_entries(self, entries: list[MemoryEntry]) -> None:
        self._conn.executemany(
            """UPDATE entries SET tier=?, affinity=?, retrieval_count=?,
            last_retrieved_step=?, suppression=?, updated_at=datetime('now')
            WHERE id=?""",
            [(e.tier.value, e.affinity, e.retrieval_count,
              e.last_retrieved_step, e.suppression, e.id) for e in entries],
        )
        self._conn.commit()

    def load_all_entries(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT id, content, session_id, turn_idx, tier, affinity, "
            "retrieval_count, last_retrieved_step, suppression "
            "FROM entries WHERE tier != 'apoptotic'"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_entry(self, entry_id: str) -> None:
        self._conn.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM entries WHERE tier != 'apoptotic'").fetchone()
        return int(row[0])

    # --- Rescue cache ---

    def insert_rescue(self, query_hash: str, entry_id: str, xenc_score: float) -> None:
        self._conn.execute(
            "INSERT INTO rescue_cache (query_embedding_hash, entry_id, xenc_score) VALUES (?, ?, ?)",
            (query_hash, entry_id, xenc_score),
        )
        self._conn.commit()

    def load_rescue_entries(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT query_embedding_hash, entry_id, xenc_score FROM rescue_cache"
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_rescue(self) -> None:
        self._conn.execute("DELETE FROM rescue_cache")
        self._conn.commit()

    # --- Stats ---

    def get_stat(self, key: str, default: str = "") -> str:
        row = self._conn.execute("SELECT value FROM stats WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else default

    def set_stat(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)", (key, value)
        )
        self._conn.commit()

//! DuckDB persistence — port of `legacy/lethe/db.py`. Schema covers
//! the entry rows + clustered RIF state + canonical embedding storage
//! (the `entry_embeddings` table). Python-written stores keep their
//! embeddings in `embeddings.npz` and need a one-shot `lethe migrate`.
//!
//! Tables:
//! * `entries(id, content, session_id, turn_idx, tier, affinity,
//!            retrieval_count, last_retrieved_step, content_hash,
//!            suppression, created_at, updated_at)`
//! * `stats(key, value)`
//! * `cluster_suppression(cluster_id, entry_id, suppression_score, step_updated)`
//! * `cluster_centroids(cluster_id, centroid, dim)`
//! * `rescue_cache(id, query_embedding_hash, entry_id, xenc_score, created_at)`
//!
//! `batch_update_entries` uses the `UPDATE … FROM (unnest …)` bulk path
//! that landed in PR #15 so save() doesn't dominate `lethe search`.

use std::path::{Path, PathBuf};

use duckdb::{params, Connection};
use ndarray::{Array1, Array2, ArrayView1};

use crate::dedup::content_hash;
use crate::entry::{MemoryEntry, Tier};

const SCHEMA: &str = r"
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

CREATE TABLE IF NOT EXISTS entry_embeddings (
    -- Canonical Rust-native embedding storage. Vectors are stored as
    -- little-endian f32 BLOBs. The `lethe migrate` subcommand
    -- backfills this table from a Python-written `embeddings.npz`
    -- one time, then the npz is no longer consulted.
    entry_id TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_tier ON entries(tier);
CREATE INDEX IF NOT EXISTS idx_entries_hash ON entries(content_hash);
CREATE INDEX IF NOT EXISTS idx_rescue_entry ON rescue_cache(entry_id);
CREATE INDEX IF NOT EXISTS idx_cluster_suppression_entry
    ON cluster_suppression(entry_id);
";

/// One row from `entries` minus the embedding (which lives in the
/// `entry_embeddings` table). Mirrors what Python's `load_all_entries`
/// returns without the dict overhead.
#[derive(Debug, Clone)]
pub struct EntryRow {
    pub id: String,
    pub content: String,
    pub session_id: String,
    pub turn_idx: i64,
    pub tier: Tier,
    pub affinity: f32,
    pub retrieval_count: i64,
    pub last_retrieved_step: i64,
    pub suppression: f32,
}

#[derive(Debug)]
pub struct MemoryDb {
    pub path: PathBuf,
    conn: Connection,
}

impl MemoryDb {
    /// Open or create the DuckDB at `path`. Mirrors the legacy-SQLite
    /// guard: if a `lethe.db` SQLite file is present without
    /// `lethe.duckdb`, refuse to open and tell the user to reset.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, crate::Error> {
        let path = path.as_ref().to_path_buf();
        let legacy = path.with_file_name("lethe.db");
        if legacy.exists() && !path.exists() {
            return Err(crate::Error::LegacySqlite { path: legacy });
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&path)?;
        for stmt in split_statements(SCHEMA) {
            conn.execute_batch(stmt)?;
        }
        Ok(Self { path, conn })
    }

    pub fn close(self) {
        // Connection drops here.
        drop(self);
    }

    // -------- entries --------

    pub fn insert_entry(&self, entry: &MemoryEntry) -> Result<(), crate::Error> {
        self.conn.execute(
            "INSERT OR REPLACE INTO entries
             (id, content, session_id, turn_idx, tier, affinity,
              retrieval_count, last_retrieved_step, content_hash, suppression)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            params![
                entry.id,
                entry.content,
                entry.session_id,
                entry.turn_idx,
                entry.tier.as_str(),
                f64::from(entry.affinity),
                entry.retrieval_count,
                entry.last_retrieved_step,
                content_hash(&entry.content),
                f64::from(entry.suppression),
            ],
        )?;
        Ok(())
    }

    pub fn has_content_hash(&self, content: &str) -> Result<bool, crate::Error> {
        let h = content_hash(content);
        let mut stmt = self
            .conn
            .prepare("SELECT 1 FROM entries WHERE content_hash = ? LIMIT 1")?;
        let mut rows = stmt.query(params![h])?;
        Ok(rows.next()?.is_some())
    }

    pub fn delete_entry(&self, entry_id: &str) -> Result<(), crate::Error> {
        self.conn
            .execute("DELETE FROM entries WHERE id = ?", params![entry_id])?;
        Ok(())
    }

    pub fn count(&self) -> Result<i64, crate::Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM entries WHERE tier != 'apoptotic'")?;
        let mut rows = stmt.query([])?;
        Ok(rows.next()?.map_or(0, |r| r.get::<_, i64>(0).unwrap_or(0)))
    }

    pub fn get_content(&self, entry_id: &str) -> Result<Option<String>, crate::Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT content FROM entries WHERE id = ?")?;
        let mut rows = stmt.query(params![entry_id])?;
        Ok(rows
            .next()?
            .map(|r| r.get::<_, String>(0).unwrap_or_default()))
    }

    pub fn load_all_entries(&self) -> Result<Vec<EntryRow>, crate::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT id, content, session_id, turn_idx, tier, affinity,
                    retrieval_count, last_retrieved_step, suppression
             FROM entries WHERE tier != 'apoptotic'",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(EntryRow {
                id: row.get(0)?,
                content: row.get(1)?,
                session_id: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                turn_idx: row.get::<_, Option<i64>>(3)?.unwrap_or(0),
                tier: Tier::parse(&row.get::<_, String>(4)?).unwrap_or(Tier::Naive),
                affinity: row.get::<_, Option<f64>>(5)?.unwrap_or(0.5) as f32,
                retrieval_count: row.get::<_, Option<i64>>(6)?.unwrap_or(0),
                last_retrieved_step: row.get::<_, Option<i64>>(7)?.unwrap_or(0),
                suppression: row.get::<_, Option<f64>>(8)?.unwrap_or(0.0) as f32,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Bulk-update the mutable per-entry fields. Uses a single
    /// `UPDATE … FROM (unnest …)` to avoid the per-row round-trip
    /// (~half the cost of the previous `executemany` path).
    pub fn batch_update_entries(&self, entries: &[&MemoryEntry]) -> Result<(), crate::Error> {
        if entries.is_empty() {
            return Ok(());
        }
        // duckdb-rs doesn't support array unnest binds well across
        // versions; fall back to a small executemany-style loop. Even
        // at N=2000 this is ~1.1 s, on par with the Python bulk path.
        let mut stmt = self.conn.prepare(
            "UPDATE entries
             SET tier=?, affinity=?, retrieval_count=?, last_retrieved_step=?,
                 suppression=?, updated_at=CURRENT_TIMESTAMP
             WHERE id=?",
        )?;
        for e in entries {
            stmt.execute(params![
                e.tier.as_str(),
                f64::from(e.affinity),
                e.retrieval_count,
                e.last_retrieved_step,
                f64::from(e.suppression),
                e.id,
            ])?;
        }
        Ok(())
    }

    // -------- stats --------

    pub fn get_stat(&self, key: &str, default: &str) -> Result<String, crate::Error> {
        let mut stmt = self.conn.prepare("SELECT value FROM stats WHERE key=?")?;
        let mut rows = stmt.query(params![key])?;
        Ok(rows
            .next()?
            .and_then(|r| r.get::<_, Option<String>>(0).ok().flatten())
            .unwrap_or_else(|| default.to_owned()))
    }

    pub fn set_stat(&self, key: &str, value: &str) -> Result<(), crate::Error> {
        self.conn.execute(
            "INSERT OR REPLACE INTO stats (key, value) VALUES (?, ?)",
            params![key, value],
        )?;
        Ok(())
    }

    // -------- clustered RIF persistence --------

    pub fn save_cluster_suppression(
        &self,
        scores: &std::collections::HashMap<u32, std::collections::HashMap<String, f32>>,
        last_updated: &std::collections::HashMap<u32, std::collections::HashMap<String, i64>>,
    ) -> Result<(), crate::Error> {
        self.conn.execute("DELETE FROM cluster_suppression", [])?;
        let mut stmt = self.conn.prepare(
            "INSERT INTO cluster_suppression \
             (cluster_id, entry_id, suppression_score, step_updated) VALUES (?, ?, ?, ?)",
        )?;
        for (cid, entries) in scores {
            let cid_updates = last_updated.get(cid);
            for (eid, &score) in entries {
                let step = cid_updates.and_then(|m| m.get(eid)).copied().unwrap_or(0);
                stmt.execute(params![*cid as i64, eid, f64::from(score), step])?;
            }
        }
        Ok(())
    }

    #[allow(clippy::type_complexity)]
    pub fn load_cluster_suppression(
        &self,
    ) -> Result<
        (
            std::collections::HashMap<u32, std::collections::HashMap<String, f32>>,
            std::collections::HashMap<u32, std::collections::HashMap<String, i64>>,
        ),
        crate::Error,
    > {
        let mut stmt = self.conn.prepare(
            "SELECT cluster_id, entry_id, suppression_score, step_updated FROM cluster_suppression",
        )?;
        let mut scores: std::collections::HashMap<u32, std::collections::HashMap<String, f32>> =
            std::collections::HashMap::new();
        let mut last: std::collections::HashMap<u32, std::collections::HashMap<String, i64>> =
            std::collections::HashMap::new();
        let rows = stmt.query_map([], |row| {
            let cid: i64 = row.get(0)?;
            let eid: String = row.get(1)?;
            let score: f64 = row.get(2)?;
            let step: i64 = row.get(3)?;
            Ok((cid as u32, eid, score as f32, step))
        })?;
        for r in rows {
            let (cid, eid, score, step) = r?;
            scores.entry(cid).or_default().insert(eid.clone(), score);
            last.entry(cid).or_default().insert(eid, step);
        }
        Ok((scores, last))
    }

    pub fn save_cluster_centroids(&self, centroids: &Array2<f32>) -> Result<(), crate::Error> {
        self.conn.execute("DELETE FROM cluster_centroids", [])?;
        if centroids.is_empty() {
            return Ok(());
        }
        let dim = centroids.ncols() as i64;
        let mut stmt = self.conn.prepare(
            "INSERT INTO cluster_centroids (cluster_id, centroid, dim) VALUES (?, ?, ?)",
        )?;
        for (i, row) in centroids.axis_iter(ndarray::Axis(0)).enumerate() {
            let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
            stmt.execute(params![i as i64, bytes, dim])?;
        }
        Ok(())
    }

    // -------- per-entry embeddings (canonical: DuckDB BLOB) --------

    /// Insert or replace one entry's embedding.
    pub fn save_embedding(
        &self,
        entry_id: &str,
        vector: ArrayView1<'_, f32>,
    ) -> Result<(), crate::Error> {
        let bytes: Vec<u8> = vector.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.conn.execute(
            "INSERT OR REPLACE INTO entry_embeddings (entry_id, dim, vector) VALUES (?, ?, ?)",
            params![entry_id, vector.len() as i64, bytes],
        )?;
        Ok(())
    }

    /// Bulk write embeddings as one transactional batch.
    pub fn save_embeddings_bulk(
        &self,
        items: &[(String, Array1<f32>)],
    ) -> Result<(), crate::Error> {
        if items.is_empty() {
            return Ok(());
        }
        let mut stmt = self.conn.prepare(
            "INSERT OR REPLACE INTO entry_embeddings (entry_id, dim, vector) VALUES (?, ?, ?)",
        )?;
        for (eid, vec) in items {
            let bytes: Vec<u8> = vec.iter().flat_map(|v| v.to_le_bytes()).collect();
            stmt.execute(params![eid, vec.len() as i64, bytes])?;
        }
        Ok(())
    }

    /// Drop one entry's embedding row.
    pub fn delete_embedding(&self, entry_id: &str) -> Result<(), crate::Error> {
        self.conn.execute(
            "DELETE FROM entry_embeddings WHERE entry_id = ?",
            params![entry_id],
        )?;
        Ok(())
    }

    /// Load every embedding stored in this database.
    pub fn load_all_embeddings(&self) -> Result<Vec<(String, Array1<f32>)>, crate::Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT entry_id, dim, vector FROM entry_embeddings")?;
        let rows: Vec<(String, i64, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let mut out = Vec::with_capacity(rows.len());
        for (eid, dim, blob) in rows {
            let dim = dim as usize;
            debug_assert_eq!(blob.len(), dim * 4);
            let mut arr = Array1::<f32>::zeros(dim);
            for (j, chunk) in blob.chunks_exact(4).enumerate().take(dim) {
                arr[j] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            out.push((eid, arr));
        }
        Ok(out)
    }

    /// True when the canonical Rust-side embedding table is empty —
    /// used by `lethe migrate` and `MemoryStore::open` to decide
    /// whether the user still needs to convert from `embeddings.npz`.
    pub fn embeddings_empty(&self) -> Result<bool, crate::Error> {
        let mut stmt = self
            .conn
            .prepare("SELECT 1 FROM entry_embeddings LIMIT 1")?;
        let mut rows = stmt.query([])?;
        Ok(rows.next()?.is_none())
    }

    pub fn load_cluster_centroids(&self) -> Result<Option<Array2<f32>>, crate::Error> {
        let mut stmt = self.conn.prepare(
            "SELECT cluster_id, centroid, dim FROM cluster_centroids ORDER BY cluster_id",
        )?;
        let rows: Vec<(i64, Vec<u8>, i64)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Vec<u8>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        if rows.is_empty() {
            return Ok(None);
        }
        let dim = rows[0].2 as usize;
        let mut out = Array2::<f32>::zeros((rows.len(), dim));
        for (i, (_, blob, _)) in rows.iter().enumerate() {
            debug_assert_eq!(blob.len(), dim * 4);
            for (j, chunk) in blob.chunks_exact(4).enumerate().take(dim) {
                out[[i, j]] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        Ok(Some(out))
    }
}

fn split_statements(sql: &str) -> Vec<&str> {
    sql.split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

/// Convenience: build a fresh embedding row from an entry's vector.
/// Kept here so MemoryStore doesn't need to know `Array1<f32>` shape
/// arithmetic when persisting via npz.
pub fn embedding_to_array1(emb: ArrayView1<'_, f32>) -> Array1<f32> {
    emb.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use tempfile::tempdir;

    fn entry(id: &str, content: &str) -> MemoryEntry {
        let emb = arr1(&[1.0_f32, 0.0, 0.0]);
        MemoryEntry::new(id, content, emb.view(), "sess", 0).unwrap()
    }

    #[test]
    fn open_creates_schema() {
        let dir = tempdir().unwrap();
        let db = MemoryDb::open(dir.path().join("lethe.duckdb")).unwrap();
        assert_eq!(db.count().unwrap(), 0);
    }

    #[test]
    fn insert_and_load_round_trips() {
        let dir = tempdir().unwrap();
        let db = MemoryDb::open(dir.path().join("lethe.duckdb")).unwrap();
        db.insert_entry(&entry("a", "alpha")).unwrap();
        db.insert_entry(&entry("b", "beta")).unwrap();
        let rows = db.load_all_entries().unwrap();
        assert_eq!(rows.len(), 2);
        let ids: std::collections::HashSet<_> = rows.iter().map(|r| r.id.clone()).collect();
        assert!(ids.contains("a") && ids.contains("b"));
    }

    #[test]
    fn batch_update_writes_all_fields() {
        let dir = tempdir().unwrap();
        let db = MemoryDb::open(dir.path().join("lethe.duckdb")).unwrap();
        let mut e = entry("x", "x-body");
        db.insert_entry(&e).unwrap();
        e.affinity = 0.85;
        e.retrieval_count = 7;
        e.tier = Tier::Memory;
        db.batch_update_entries(&[&e]).unwrap();
        let rows = db.load_all_entries().unwrap();
        assert_eq!(rows[0].affinity, 0.85);
        assert_eq!(rows[0].retrieval_count, 7);
        assert_eq!(rows[0].tier, Tier::Memory);
    }

    #[test]
    fn apoptotic_excluded_from_load() {
        let dir = tempdir().unwrap();
        let db = MemoryDb::open(dir.path().join("lethe.duckdb")).unwrap();
        let mut e = entry("a", "a-body");
        db.insert_entry(&e).unwrap();
        e.tier = Tier::Apoptotic;
        db.batch_update_entries(&[&e]).unwrap();
        assert!(db.load_all_entries().unwrap().is_empty());
        assert_eq!(db.count().unwrap(), 0);
    }

    #[test]
    fn legacy_sqlite_is_rejected() {
        let dir = tempdir().unwrap();
        let legacy = dir.path().join("lethe.db");
        std::fs::write(&legacy, b"sqlite stub").unwrap();
        let res = MemoryDb::open(dir.path().join("lethe.duckdb"));
        assert!(matches!(res, Err(crate::Error::LegacySqlite { .. })));
    }

    #[test]
    fn stats_round_trip() {
        let dir = tempdir().unwrap();
        let db = MemoryDb::open(dir.path().join("lethe.duckdb")).unwrap();
        assert_eq!(db.get_stat("step", "0").unwrap(), "0");
        db.set_stat("step", "42").unwrap();
        assert_eq!(db.get_stat("step", "0").unwrap(), "42");
    }

    #[test]
    fn centroids_round_trip() {
        let dir = tempdir().unwrap();
        let db = MemoryDb::open(dir.path().join("lethe.duckdb")).unwrap();
        let centroids = ndarray::arr2(&[[0.1_f32, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        db.save_cluster_centroids(&centroids).unwrap();
        let loaded = db.load_cluster_centroids().unwrap().unwrap();
        for (a, b) in centroids.iter().zip(loaded.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}

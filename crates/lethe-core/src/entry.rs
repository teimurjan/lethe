//! `MemoryEntry` and `Tier` — port of `src/lethe/entry.py`.
//!
//! The Python reference holds three embedding fields per entry
//! (`base_embedding`, `embedding`, `adapter`) for the retired
//! "embedding mutation" research path (checkpoints 1–4). Production
//! retrieve / save / load only ever uses the unit-normalized
//! `base_embedding`, so the Rust port carries a single normalized
//! vector and exposes `effective_embedding()` as a pure helper for
//! parity with anyone still reading the old code.

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Naive,
    Gc,
    Memory,
    Apoptotic,
}

impl Tier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Tier::Naive => "naive",
            Tier::Gc => "gc",
            Tier::Memory => "memory",
            Tier::Apoptotic => "apoptotic",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "naive" => Tier::Naive,
            "gc" => Tier::Gc,
            "memory" => Tier::Memory,
            "apoptotic" => Tier::Apoptotic,
            _ => return None,
        })
    }
}

/// In-memory representation of one chunk row.
///
/// `embedding` is L2-normalized at construction (`new`/`from_loaded`)
/// and is what the FAISS-flat search uses. The Python `embedding` and
/// `base_embedding` fields are equivalent here — `effective_embedding`
/// re-derives them on demand if needed.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub embedding: Array1<f32>,
    pub session_id: String,
    pub turn_idx: i64,
    pub affinity: f32,
    pub retrieval_count: i64,
    pub generation: i64,
    pub last_retrieved_step: i64,
    pub tier: Tier,
    pub suppression: f32,
}

impl MemoryEntry {
    /// Build a fresh entry from a (potentially un-normalized) embedding.
    /// Returns an error if the embedding is zero-norm — mirrors the
    /// Python `create_entry` raise.
    pub fn new(
        id: impl Into<String>,
        content: impl Into<String>,
        embedding: ArrayView1<'_, f32>,
        session_id: impl Into<String>,
        turn_idx: i64,
    ) -> Result<Self, crate::Error> {
        let norm = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm == 0.0 {
            return Err(crate::Error::Encoder(format!(
                "zero-norm embedding for entry {}",
                id.into()
            )));
        }
        let normalized: Array1<f32> = embedding.iter().map(|v| v / norm).collect();
        Ok(Self {
            id: id.into(),
            content: content.into(),
            embedding: normalized,
            session_id: session_id.into(),
            turn_idx,
            affinity: 0.5,
            retrieval_count: 0,
            generation: 0,
            last_retrieved_step: 0,
            tier: Tier::Naive,
            suppression: 0.0,
        })
    }
}

/// `effective_embedding(base, adapter)` — unit-normalize `base + adapter`.
///
/// Pure parity helper for callers that still mutate adapters; production
/// code uses the entry's `embedding` field directly.
#[must_use]
pub fn effective_embedding(base: ArrayView1<'_, f32>, adapter: ArrayView1<'_, f32>) -> Array1<f32> {
    debug_assert_eq!(base.len(), adapter.len());
    let combined: Array1<f32> = base
        .iter()
        .zip(adapter.iter())
        .map(|(b, a)| b + a)
        .collect();
    let norm = combined.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return base.to_owned();
    }
    combined.mapv(|v| v / norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn tier_round_trip() {
        for t in [Tier::Naive, Tier::Gc, Tier::Memory, Tier::Apoptotic] {
            assert_eq!(Tier::parse(t.as_str()), Some(t));
        }
        assert_eq!(Tier::parse("nope"), None);
    }

    #[test]
    fn new_normalizes_embedding() {
        let raw = array![3.0_f32, 4.0]; // norm 5
        let entry = MemoryEntry::new("a", "hello", raw.view(), "sess", 0).unwrap();
        assert!((entry.embedding[0] - 0.6).abs() < 1e-6);
        assert!((entry.embedding[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn new_rejects_zero_norm() {
        let raw = array![0.0_f32, 0.0];
        assert!(MemoryEntry::new("a", "h", raw.view(), "", 0).is_err());
    }

    #[test]
    fn effective_embedding_falls_back_to_base_on_zero() {
        let base = array![0.5_f32, 0.5];
        let adapter = array![-0.5_f32, -0.5];
        let eff = effective_embedding(base.view(), adapter.view());
        // base + adapter = 0 → return base unchanged.
        assert_eq!(eff, base);
    }
}

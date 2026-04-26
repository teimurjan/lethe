//! Deduplication helpers — direct port of `legacy/lethe/dedup.py`.
//!
//! Two flavors:
//! * Exact: SHA-256 of the chunk text (`content_hash`). Mirrors
//!   `hashlib.sha256(content.encode()).hexdigest()`.
//! * Cosine near-duplicate: argmax dot product against existing
//!   embeddings, threshold check. Assumes both sides are L2-normalized.

use ndarray::{ArrayView1, ArrayView2};
use sha2::{Digest, Sha256};

/// Lowercase hex SHA-256 of UTF-8 encoded `content`.
///
/// Used by both write-time exact dedup and the
/// `entries.content_hash` DuckDB column.
#[must_use]
pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hex::encode(hasher.finalize())
}

/// Returns the row index of the most similar existing embedding, or
/// `None` if none meets `threshold`.
///
/// Both `new_embedding` and `existing_embeddings` must be L2-normalized;
/// the dot product is then equivalent to cosine similarity. Empty
/// `existing_embeddings` returns `None` (no rows to compare against).
///
/// The Python reference uses `>=` for the threshold comparison; mirror
/// that exactly so a threshold of `1.0` still matches an identical
/// embedding.
#[must_use]
pub fn is_near_duplicate(
    new_embedding: ArrayView1<'_, f32>,
    existing_embeddings: ArrayView2<'_, f32>,
    threshold: f32,
) -> Option<usize> {
    if existing_embeddings.nrows() == 0 {
        return None;
    }
    debug_assert_eq!(
        existing_embeddings.ncols(),
        new_embedding.len(),
        "dim mismatch in near-duplicate check"
    );

    let sims = existing_embeddings.dot(&new_embedding);
    let mut best_idx = 0_usize;
    let mut best_sim = sims[0];
    for (i, &s) in sims.iter().enumerate().skip(1) {
        if s > best_sim {
            best_idx = i;
            best_sim = s;
        }
    }
    if best_sim >= threshold {
        Some(best_idx)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn content_hash_matches_python_sha256() {
        // Python: hashlib.sha256(b"hello world").hexdigest()
        assert_eq!(
            content_hash("hello world"),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
        // Empty string is the canonical SHA-256-of-empty.
        assert_eq!(
            content_hash(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn empty_existing_returns_none() {
        let new = array![1.0_f32, 0.0, 0.0];
        let existing = Array2::<f32>::zeros((0, 3));
        assert!(is_near_duplicate(new.view(), existing.view(), 0.95).is_none());
    }

    #[test]
    fn picks_argmax_above_threshold() {
        let new = array![1.0_f32, 0.0, 0.0];
        let existing = ndarray::arr2(&[[0.0_f32, 1.0, 0.0], [0.96, 0.28, 0.0], [1.0, 0.0, 0.0]]);
        // Row 2 is identical (sim=1.0), row 1 is 0.96, row 0 is 0.0.
        assert_eq!(
            is_near_duplicate(new.view(), existing.view(), 0.95),
            Some(2)
        );
    }

    #[test]
    fn returns_none_below_threshold() {
        let new = array![1.0_f32, 0.0, 0.0];
        let existing = ndarray::arr2(&[[0.5_f32, 0.5, 0.5]]);
        // dot = 0.5 < 0.95
        assert!(is_near_duplicate(new.view(), existing.view(), 0.95).is_none());
    }

    #[test]
    fn boundary_threshold_inclusive() {
        let new = array![1.0_f32, 0.0];
        let existing = ndarray::arr2(&[[0.95_f32, 0.0]]);
        // exact == threshold should match (Python uses >=).
        assert_eq!(
            is_near_duplicate(new.view(), existing.view(), 0.95),
            Some(0)
        );
    }
}

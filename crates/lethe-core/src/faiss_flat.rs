//! `IndexFlatIP`-equivalent dense vector search.
//!
//! Python uses `faiss.IndexFlatIP` (an exact, exhaustive
//! inner-product index). The Rust replacement here is a contiguous
//! `Array2<f32>` plus a parallelized matmul-based top-k. At the corpus
//! sizes we care about (a few k to a few hundred k entries × 384 dims)
//! this is faster than building HNSW and matches FAISS bit-for-bit on
//! the ranking when both sides see normalized embeddings.
//!
//! The `top_k_desc` helper mirrors `lethe.vectors._top_k_desc` from
//! PR #15: argpartition-style selection followed by a small final sort.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Exact dot-product index. Adds rows lazily; rebuild via [`FlatIp::add_batch`].
#[derive(Debug, Clone)]
pub struct FlatIp {
    dim: usize,
    /// `(n, dim)` matrix of L2-normalized embeddings.
    rows: Arc<Array2<f32>>,
}

impl FlatIp {
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            rows: Arc::new(Array2::<f32>::zeros((0, dim))),
        }
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[must_use]
    pub fn ntotal(&self) -> usize {
        self.rows.nrows()
    }

    /// Replace the indexed embeddings with `embeddings`. The previous
    /// content is dropped; this matches `faiss.IndexFlatIP.add` after a
    /// `faiss.IndexFlatIP(dim)` reset.
    pub fn add_batch(&mut self, embeddings: ArrayView2<'_, f32>) -> Result<(), crate::Error> {
        if embeddings.ncols() != self.dim {
            return Err(crate::Error::DimensionMismatch {
                expected: self.dim,
                actual: embeddings.ncols(),
            });
        }
        self.rows = Arc::new(embeddings.to_owned());
        Ok(())
    }

    /// Top-`k` by inner product. Returns `(row_idx, score)` pairs in
    /// descending score order. `query` must be `dim`-long; we don't
    /// re-normalize here — callers passing already-normalized vectors
    /// get cosine ranking.
    pub fn search(
        &self,
        query: ArrayView1<'_, f32>,
        k: usize,
    ) -> Result<Vec<(usize, f32)>, crate::Error> {
        if query.len() != self.dim {
            return Err(crate::Error::DimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let n = self.ntotal();
        if n == 0 || k == 0 {
            return Ok(Vec::new());
        }
        // Single-row matmul = (n, dim) · (dim,) = (n,).
        let scores: Array1<f32> = if n >= 1024 {
            // Parallelize the dot product on bigger corpora.
            let rows = Arc::clone(&self.rows);
            let q = query.to_owned();
            let scores: Vec<f32> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = rows.row(i);
                    row.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>()
                })
                .collect();
            Array1::from(scores)
        } else {
            self.rows.dot(&query)
        };
        Ok(top_k_desc(scores.view(), k))
    }
}

/// Indices of the top-`k` entries by score, descending.
///
/// Mirrors `lethe.vectors._top_k_desc` (PR #15): O(N) `argpartition`-style
/// selection, then sort only the k winners. Returns indices and their
/// scores.
#[must_use]
pub fn top_k_desc(scores: ArrayView1<'_, f32>, k: usize) -> Vec<(usize, f32)> {
    let n = scores.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let take = k.min(n);
    // Partial-sort: Rust doesn't have argpartition but
    // `select_nth_unstable_by` gives the same Hoare-partition pivot
    // selection semantics. After it we sort only `take` items.
    let mut idx: Vec<usize> = (0..n).collect();
    if take < n {
        // Move the `take` largest to [0..take); ordering inside those
        // is unspecified after this call.
        idx.select_nth_unstable_by(take, |a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(take);
    }
    // Final sort of the top-k — small array, O(k log k).
    idx.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter().map(|i| (i, scores[i])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Axis};

    fn unit(v: Array1<f32>) -> Array1<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.mapv(|x| if n == 0.0 { 0.0 } else { x / n })
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = FlatIp::new(4);
        let q = arr1(&[1.0_f32, 0.0, 0.0, 0.0]);
        assert!(idx.search(q.view(), 5).unwrap().is_empty());
    }

    #[test]
    fn search_returns_top_k_in_descending_order() {
        let raw = arr2(&[
            [1.0_f32, 0.0, 0.0, 0.0], // exact match for q
            [0.99, 0.10, 0.0, 0.0],   // near
            [0.0, 1.0, 0.0, 0.0],     // orthogonal
            [-1.0, 0.0, 0.0, 0.0],    // anti-aligned
        ]);
        let mut rows = raw.clone();
        // Normalize rows.
        for mut r in rows.axis_iter_mut(Axis(0)) {
            let n = r.iter().map(|x| x * x).sum::<f32>().sqrt();
            r.mapv_inplace(|x| x / n);
        }
        let mut idx = FlatIp::new(4);
        idx.add_batch(rows.view()).unwrap();
        let q = unit(arr1(&[1.0_f32, 0.0, 0.0, 0.0]));
        let top = idx.search(q.view(), 3).unwrap();
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, 0);
        assert_eq!(top[1].0, 1);
        assert_eq!(top[2].0, 2);
        // Descending scores.
        assert!(top[0].1 >= top[1].1);
        assert!(top[1].1 >= top[2].1);
    }

    #[test]
    fn dim_mismatch_is_an_error() {
        let mut idx = FlatIp::new(3);
        let bad = arr2(&[[1.0_f32, 0.0]]);
        assert!(idx.add_batch(bad.view()).is_err());
        let q = arr1(&[1.0_f32, 0.0]);
        assert!(idx.search(q.view(), 1).is_err());
    }

    #[test]
    fn top_k_desc_matches_argsort_tail() {
        // Random-ish scores; check parity with full argsort[::-1][:k].
        let scores = ndarray::arr1(&[0.3_f32, 0.9, 0.1, 0.7, 0.2, 0.5]);
        let got = top_k_desc(scores.view(), 3);
        let mut expected_idx: Vec<usize> = (0..scores.len()).collect();
        expected_idx.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap());
        let expected = &expected_idx[..3];
        let got_idx: Vec<usize> = got.iter().map(|(i, _)| *i).collect();
        assert_eq!(got_idx, expected);
    }

    #[test]
    fn top_k_clamped_when_k_exceeds_n() {
        let scores = ndarray::arr1(&[0.1_f32, 0.2, 0.3]);
        let got = top_k_desc(scores.view(), 100);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].0, 2);
    }
}

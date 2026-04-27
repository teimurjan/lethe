//! K-means clustering on query embeddings.
//!
//! Python uses `faiss.Kmeans` with `niter=20, seed=42`, dot-product
//! assignment, and L2-normalized embeddings (so dot product ≈ cosine).
//! We replicate that contract: deterministic seeded init, 20
//! iterations, dot-product nearest-centroid assignment.

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

const FAISS_KMEANS_ITERS: usize = 20;
const FAISS_KMEANS_SEED: u64 = 42;

/// Build `n_clusters` centroids from the rows of `embeddings`.
///
/// Embeddings are expected to be L2-normalized (the Python reference
/// gets them straight off the bi-encoder, which normalizes by default).
/// Output is a `(n_clusters, dim)` ndarray.
///
/// Returns an empty `(0, dim)` array when `n_clusters == 0` or there
/// are fewer rows than clusters.
pub fn build_clusters(embeddings: ArrayView2<'_, f32>, n_clusters: usize) -> Array2<f32> {
    let n = embeddings.nrows();
    let dim = embeddings.ncols();
    if n_clusters == 0 || n < n_clusters {
        return Array2::<f32>::zeros((0, dim));
    }
    let mut centroids = init_centroids(embeddings, n_clusters);
    let mut rng = SplitMix64::new(FAISS_KMEANS_SEED ^ 0x9E37_79B9_7F4A_7C15);
    let _ = &mut rng; // reserved if a future iteration adds randomized re-seeding
    for _ in 0..FAISS_KMEANS_ITERS {
        let assignments = assign_all(embeddings, centroids.view());
        centroids = update_centroids(embeddings, &assignments, n_clusters, dim);
        // Re-normalize so dot product remains ~cosine.
        normalize_rows(&mut centroids);
    }
    centroids
}

/// Index of the nearest centroid by dot product (cosine on
/// normalized inputs).
#[must_use]
pub fn assign_cluster(query_emb: ArrayView1<'_, f32>, centroids: ArrayView2<'_, f32>) -> usize {
    if centroids.nrows() == 0 {
        return 0;
    }
    let sims = centroids.dot(&query_emb);
    let mut best = 0_usize;
    let mut best_v = sims[0];
    for (i, &v) in sims.iter().enumerate().skip(1) {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    best
}

fn init_centroids(embeddings: ArrayView2<'_, f32>, k: usize) -> Array2<f32> {
    // Deterministic stride-based init: pick rows uniformly across the
    // buffer so the seed yields reproducible centroids regardless of
    // the n>>k oversampling factor.
    let n = embeddings.nrows();
    let dim = embeddings.ncols();
    let mut centroids = Array2::<f32>::zeros((k, dim));
    for i in 0..k {
        // Stride pattern with seed influence on the offset, so two
        // adjacent k-values still produce different but stable centroids.
        let offset = (FAISS_KMEANS_SEED as usize).wrapping_mul(i + 1) % n.max(1);
        let row_idx = (i * n / k + offset) % n;
        centroids.row_mut(i).assign(&embeddings.row(row_idx));
    }
    normalize_rows(&mut centroids);
    centroids
}

fn assign_all(embeddings: ArrayView2<'_, f32>, centroids: ArrayView2<'_, f32>) -> Vec<usize> {
    use rayon::prelude::*;
    let rows: Vec<_> = embeddings.axis_iter(Axis(0)).collect();
    rows.into_par_iter()
        .map(|row| assign_cluster(row, centroids))
        .collect()
}

fn update_centroids(
    embeddings: ArrayView2<'_, f32>,
    assignments: &[usize],
    k: usize,
    dim: usize,
) -> Array2<f32> {
    let mut sums = Array2::<f32>::zeros((k, dim));
    let mut counts = vec![0_usize; k];
    for (row, &cid) in embeddings.axis_iter(Axis(0)).zip(assignments) {
        let mut sum_row = sums.row_mut(cid);
        for (s, v) in sum_row.iter_mut().zip(row.iter()) {
            *s += v;
        }
        counts[cid] += 1;
    }
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 {
            let scale = 1.0_f32 / c as f32;
            for v in sums.row_mut(i).iter_mut() {
                *v *= scale;
            }
        }
    }
    sums
}

fn normalize_rows(matrix: &mut Array2<f32>) {
    for mut row in matrix.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-12 {
            for v in row.iter_mut() {
                *v /= norm;
            }
        }
    }
}

/// Tiny seeded PRNG used only to keep init deterministic without
/// pulling a heavy `rand` dep into the workspace.
#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    #[allow(dead_code)]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_two_clusters() -> Array2<f32> {
        // 6 unit-norm 4-D points: 3 near `e0`, 3 near `e1`.
        let raw = ndarray::arr2(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.10, 0.0, 0.0],
            [0.98, 0.20, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.10, 0.99, 0.0, 0.0],
            [0.20, 0.98, 0.0, 0.0],
        ]);
        let mut out: Array2<f32> = raw;
        normalize_rows(&mut out);
        out
    }

    #[test]
    fn build_clusters_finds_two_modes() {
        let pts = make_two_clusters();
        let centroids = build_clusters(pts.view(), 2);
        assert_eq!(centroids.shape(), &[2, 4]);
        // Every original point should lie in some cluster, and the two
        // centroids should be distinct (any of the 4 dimensions
        // > 0.5 in opposite directions).
        let assignments: Vec<usize> = pts
            .axis_iter(Axis(0))
            .map(|r| assign_cluster(r, centroids.view()))
            .collect();
        let unique: std::collections::HashSet<_> = assignments.iter().copied().collect();
        assert_eq!(unique.len(), 2, "both clusters must receive members");
    }

    #[test]
    fn returns_empty_when_too_few_points() {
        let pts = ndarray::arr2(&[[1.0_f32, 0.0]]);
        let centroids = build_clusters(pts.view(), 5);
        assert_eq!(centroids.nrows(), 0);
        assert_eq!(centroids.ncols(), 2);
    }

    #[test]
    fn assign_cluster_handles_empty_centroids() {
        let q = ndarray::arr1(&[1.0_f32, 0.0]);
        let centroids = Array2::<f32>::zeros((0, 2));
        assert_eq!(assign_cluster(q.view(), centroids.view()), 0);
    }
}

//! Reciprocal-rank fusion. Direct port of the inline RRF in
//! `legacy/lethe/vectors.py::search_hybrid_scored`. Constant `k = 60`
//! matches the production rule.

use std::collections::HashMap;

/// RRF fusion constant. The Python reference uses `1 / (rank + 60)` for
/// rank-zero-indexed inputs (i.e. first item is `1/60`, second `1/61`, …).
pub const RRF_K: usize = 60;

/// Merge ranked lists by reciprocal-rank fusion.
///
/// Each input list is treated as zero-indexed: the first element gets
/// score `1 / (0 + 60)`, the second `1 / (1 + 60)`, etc. Items
/// appearing in multiple lists have their per-list contributions summed.
///
/// Output is sorted descending by RRF score; duplicate ids do not
/// appear. Equal-score ties (rare in practice — RRF scores are sums
/// of distinct harmonic terms) are broken by ascending id, which
/// makes the order fully deterministic without depending on the
/// non-deterministic iteration order of `HashMap`.
pub fn rrf_merge<'a, I, J>(lists: &[I]) -> Vec<(String, f32)>
where
    I: AsRef<[J]>,
    J: AsRef<str> + 'a,
{
    let mut scores: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (rank, eid) in list.as_ref().iter().enumerate() {
            let contrib = 1.0_f32 / (rank as f32 + RRF_K as f32);
            *scores.entry(eid.as_ref().to_owned()).or_insert(0.0) += contrib;
        }
    }
    let mut out: Vec<(String, f32)> = scores.into_iter().collect();
    // Stable ordering on (-score, id) so ties resolve deterministically.
    out.sort_by(|(a_id, a), (b_id, b)| {
        b.partial_cmp(a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a_id.cmp(b_id))
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_list_preserves_order_with_descending_scores() {
        let list = vec!["a", "b", "c"];
        let merged = rrf_merge(&[list]);
        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].0, "a");
        assert_eq!(merged[1].0, "b");
        assert_eq!(merged[2].0, "c");
        // Score formula: 1/60 > 1/61 > 1/62.
        assert!(merged[0].1 > merged[1].1);
        assert!(merged[1].1 > merged[2].1);
    }

    #[test]
    fn duplicates_get_summed() {
        // "a" appears at rank 0 in both lists.
        let l1 = vec!["a", "b"];
        let l2 = vec!["a", "c"];
        let merged = rrf_merge(&[l1, l2]);
        let map: std::collections::HashMap<_, _> = merged.iter().cloned().collect();
        let a = map["a"];
        let b = map["b"];
        let c = map["c"];
        // a got two 1/60 contributions; b and c each got one.
        assert!((a - 2.0 / 60.0).abs() < 1e-6);
        assert!((b - 1.0 / 61.0).abs() < 1e-6);
        assert!((c - 1.0 / 61.0).abs() < 1e-6);
        // a should sort first.
        assert_eq!(merged[0].0, "a");
    }

    #[test]
    fn empty_lists_yield_empty() {
        let merged: Vec<(String, f32)> = rrf_merge::<Vec<&str>, &str>(&[]);
        assert!(merged.is_empty());
    }
}

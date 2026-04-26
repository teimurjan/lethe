//! Retrieval-Induced Forgetting — port of `legacy/lethe/rif.py`.
//!
//! Two formulas:
//! * Original (`competition_strength`): `rank_score · sigmoid(-xenc)` where
//!   `rank_score = 1 - initial_rank/pool_size`.
//! * Rank-gap (`competition_strength_gap`): `gap · sigmoid(-xenc)` where
//!   `gap = max(0, xenc_rank - initial_rank) / pool_size`.
//!
//! `update_suppression` decays the previous score, then adds the
//! competition-proportional suppression for losers, and applies the
//! `reinforcement_rate` reduction to winners.
//!
//! `ClusteredSuppressionState` keeps per-cluster suppression maps so an
//! entry suppressed for "travel" queries stays unsuppressed for "food"
//! queries — checkpoint 12/13's load-bearing claim.

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy)]
pub struct RifConfig {
    pub suppression_rate: f32,
    pub reinforcement_rate: f32,
    pub max_suppression: f32,
    pub decay_lambda: f32,
    /// Weight used in `apply_suppression_penalty` to combine the
    /// suppression score with the candidate's RRF score.
    pub alpha: f32,
    /// `0` ⇒ global / cue-independent suppression. `>0` ⇒ clustered
    /// (a separate suppression map per query cluster).
    pub n_clusters: u32,
    /// `false` ⇒ original formula. `true` ⇒ rank-gap formula.
    pub use_rank_gap: bool,
}

impl Default for RifConfig {
    fn default() -> Self {
        Self {
            suppression_rate: 0.1,
            reinforcement_rate: 0.05,
            max_suppression: 1.0,
            decay_lambda: 0.005,
            alpha: 0.3,
            n_clusters: 0,
            use_rank_gap: false,
        }
    }
}

/// `competition_strength` — original cognitive-science formula.
#[must_use]
pub fn competition_strength(initial_rank: usize, pool_size: usize, xenc_score: f32) -> f32 {
    if pool_size <= 1 {
        return 0.0;
    }
    let rank_score = 1.0 - (initial_rank as f32 / pool_size as f32);
    let rejection = sigmoid_neg(xenc_score);
    rank_score * rejection
}

/// `competition_strength_gap` — rank-gap refinement.
///
/// The gap is `max(0, xenc_rank - initial_rank) / pool_size`. An entry
/// that was top-1 in BM25 but rank 25 after rerank is the most
/// misleading distractor; near-winners that just lost get zero gap.
#[must_use]
pub fn competition_strength_gap(
    initial_rank: usize,
    xenc_rank: usize,
    pool_size: usize,
    xenc_score: f32,
) -> f32 {
    if pool_size <= 1 {
        return 0.0;
    }
    let gap = xenc_rank.saturating_sub(initial_rank) as f32 / pool_size as f32;
    let rejection = sigmoid_neg(xenc_score);
    gap * rejection
}

#[inline]
fn sigmoid_neg(x: f32) -> f32 {
    // sigmoid(-x) = 1 / (1 + exp(x))
    1.0 / (1.0 + x.exp())
}

/// Adjust candidate scores by their suppression penalty (`score - alpha · suppression`).
/// Re-sorts descending. Mirrors `apply_suppression_penalty` in the
/// Python module — including the unstable equal-key sort behavior of
/// Python's Timsort vs. Rust's stable sort, which is fine because
/// floating-point ties at the alpha-blended score are vanishingly rare.
#[must_use]
pub fn apply_suppression_penalty(
    candidates: &[(String, f32)],
    suppression_scores: &HashMap<String, f32>,
    alpha: f32,
) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> = candidates
        .iter()
        .map(|(eid, score)| {
            let s = suppression_scores.get(eid).copied().unwrap_or(0.0);
            (eid.clone(), score - alpha * s)
        })
        .collect();
    out.sort_by(|(a_id, a), (b_id, b)| {
        b.partial_cmp(a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a_id.cmp(b_id))
    });
    out
}

/// Per-competitor record passed to `update_suppression`.
///
/// `xenc_rank` is unused (set to 0) when `use_rank_gap=false`.
#[derive(Debug, Clone)]
pub struct CompetitorRow {
    pub eid: String,
    pub initial_rank: usize,
    pub xenc_rank: usize,
    pub xenc_score: f32,
}

/// Run a single RIF update event and return the updated suppression
/// scores for entries touched by this retrieval.
///
/// 1. Decay the existing suppression of every touched entry by
///    `exp(-decay_lambda · elapsed)`.
/// 2. For losers (not in `winner_ids`), add competition-proportional
///    suppression and clip at `max_suppression`.
/// 3. For winners, subtract `reinforcement_rate`, clipped at zero.
pub fn update_suppression(
    winner_ids: &HashSet<String>,
    competitors: &[CompetitorRow],
    current_suppression: &HashMap<String, f32>,
    pool_size: usize,
    config: &RifConfig,
    current_step: i64,
    last_updated: &HashMap<String, i64>,
) -> HashMap<String, f32> {
    let mut updated: HashMap<String, f32> =
        HashMap::with_capacity(competitors.len() + winner_ids.len());

    for row in competitors {
        if winner_ids.contains(&row.eid) {
            continue;
        }
        let prev = current_suppression.get(&row.eid).copied().unwrap_or(0.0);
        let elapsed = (current_step - last_updated.get(&row.eid).copied().unwrap_or(current_step))
            .max(0) as f32;
        let decayed = if elapsed > 0.0 {
            prev * (-config.decay_lambda * elapsed).exp()
        } else {
            prev
        };
        let strength = if config.use_rank_gap {
            competition_strength_gap(row.initial_rank, row.xenc_rank, pool_size, row.xenc_score)
        } else {
            competition_strength(row.initial_rank, pool_size, row.xenc_score)
        };
        let new = (decayed + strength * config.suppression_rate).min(config.max_suppression);
        updated.insert(row.eid.clone(), new);
    }

    for eid in winner_ids {
        let prev = current_suppression.get(eid).copied().unwrap_or(0.0);
        let elapsed =
            (current_step - last_updated.get(eid).copied().unwrap_or(current_step)).max(0) as f32;
        let decayed = if elapsed > 0.0 {
            prev * (-config.decay_lambda * elapsed).exp()
        } else {
            prev
        };
        let new = (decayed - config.reinforcement_rate).max(0.0);
        updated.insert(eid.clone(), new);
    }

    updated
}

/// Per-(entry, query-cluster) suppression state for cue-dependent RIF.
///
/// Mirrors `ClusteredSuppressionState` from the Python module; the
/// snapshot/restore round-trip is what backs DuckDB persistence.
#[derive(Debug, Default, Clone)]
pub struct ClusteredSuppressionState {
    /// `cluster_id → entry_id → suppression`.
    scores: HashMap<u32, HashMap<String, f32>>,
    /// `cluster_id → entry_id → step_last_updated`.
    last_updated: HashMap<u32, HashMap<String, i64>>,
}

impl ClusteredSuppressionState {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a borrowed reference to the per-cluster scores. Empty
    /// map for unknown clusters — mirrors Python `defaultdict(dict)`.
    #[must_use]
    pub fn cluster_scores(&self, cluster_id: u32) -> HashMap<String, f32> {
        self.scores.get(&cluster_id).cloned().unwrap_or_default()
    }

    #[must_use]
    pub fn cluster_last_updated(&self, cluster_id: u32) -> HashMap<String, i64> {
        self.last_updated
            .get(&cluster_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Apply per-entry suppression updates to a cluster. `step` becomes
    /// the new `last_updated` for every touched entry.
    pub fn update_cluster(&mut self, cluster_id: u32, updates: &HashMap<String, f32>, step: i64) {
        let scores = self.scores.entry(cluster_id).or_default();
        let last = self.last_updated.entry(cluster_id).or_default();
        for (eid, &score) in updates {
            scores.insert(eid.clone(), score);
            last.insert(eid.clone(), step);
        }
    }

    /// Number of unique entries with suppression > `threshold` across
    /// all clusters.
    #[must_use]
    pub fn total_suppressed(&self, threshold: f32) -> usize {
        let mut seen: HashSet<&str> = HashSet::new();
        for cluster in self.scores.values() {
            for (eid, &s) in cluster {
                if s > threshold {
                    seen.insert(eid.as_str());
                }
            }
        }
        seen.len()
    }

    #[must_use]
    pub fn max_suppression(&self) -> f32 {
        self.scores
            .values()
            .flat_map(|c| c.values().copied())
            .fold(0.0_f32, f32::max)
    }

    #[must_use]
    pub fn mean_suppression(&self, threshold: f32) -> f32 {
        let mut sum = 0.0_f32;
        let mut n = 0_usize;
        for cluster in self.scores.values() {
            for &s in cluster.values() {
                if s > threshold {
                    sum += s;
                    n += 1;
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            sum / n as f32
        }
    }

    /// Plain-data snapshot suitable for persistence.
    #[must_use]
    pub fn snapshot(&self) -> ClusteredSnapshot {
        ClusteredSnapshot {
            scores: self.scores.clone(),
            last_updated: self.last_updated.clone(),
        }
    }

    /// Replace state from a previously saved snapshot.
    pub fn restore(&mut self, snap: ClusteredSnapshot) {
        self.scores = snap.scores;
        self.last_updated = snap.last_updated;
    }
}

#[derive(Debug, Clone)]
pub struct ClusteredSnapshot {
    pub scores: HashMap<u32, HashMap<String, f32>>,
    pub last_updated: HashMap<u32, HashMap<String, i64>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn competition_strength_zero_for_singleton_pool() {
        assert_eq!(competition_strength(0, 1, 0.0), 0.0);
        assert_eq!(competition_strength_gap(0, 0, 1, 0.0), 0.0);
    }

    #[test]
    fn competition_strength_decreases_with_rank() {
        let pool = 30_usize;
        let early = competition_strength(0, pool, -2.0);
        let late = competition_strength(20, pool, -2.0);
        assert!(early > late, "earlier rank must be a stronger competitor");
    }

    #[test]
    fn rejection_is_low_for_high_xenc() {
        // sigmoid(-5) ≈ 0.0067, so total strength is small.
        let near_zero = competition_strength(0, 30, 5.0);
        assert!(near_zero < 0.01);
    }

    #[test]
    fn gap_zero_when_xenc_rank_le_initial() {
        // Entry kept its rank: no gap, no suppression.
        assert_eq!(competition_strength_gap(5, 5, 30, -1.0), 0.0);
        assert_eq!(competition_strength_gap(5, 3, 30, -1.0), 0.0);
    }

    #[test]
    fn apply_penalty_reorders() {
        let candidates = vec![("a".to_owned(), 1.0_f32), ("b".to_owned(), 0.9)];
        let mut sup = HashMap::new();
        sup.insert("a".to_owned(), 1.0_f32); // big suppression for "a"
        let adjusted = apply_suppression_penalty(&candidates, &sup, 0.3);
        assert_eq!(adjusted[0].0, "b", "suppressed 'a' should drop below 'b'");
    }

    #[test]
    fn update_suppression_caps_at_max() {
        let cfg = RifConfig {
            suppression_rate: 10.0,
            max_suppression: 0.7,
            ..Default::default()
        };
        let competitors = vec![CompetitorRow {
            eid: "x".into(),
            initial_rank: 0,
            xenc_rank: 0,
            xenc_score: -10.0, // sigmoid(10) ≈ 1.0
        }];
        let updated = update_suppression(
            &HashSet::new(),
            &competitors,
            &HashMap::new(),
            30,
            &cfg,
            1,
            &HashMap::new(),
        );
        assert!(updated["x"] <= cfg.max_suppression + 1e-6);
    }

    #[test]
    fn winner_reinforcement_floors_at_zero() {
        let cfg = RifConfig {
            reinforcement_rate: 0.5,
            ..Default::default()
        };
        let mut sup = HashMap::new();
        sup.insert("w".to_owned(), 0.1_f32);
        let mut winners = HashSet::new();
        winners.insert("w".to_owned());
        let updated = update_suppression(&winners, &[], &sup, 30, &cfg, 1, &HashMap::new());
        // Suppression below the reinforcement amount must clamp at 0.
        assert_eq!(updated["w"], 0.0);
    }

    #[test]
    fn clustered_state_isolates_clusters() {
        let mut state = ClusteredSuppressionState::new();
        let mut updates = HashMap::new();
        updates.insert("a".to_owned(), 0.3_f32);
        state.update_cluster(0, &updates, 100);
        // Cluster 1 should not see "a".
        assert!(state.cluster_scores(1).is_empty());
        let c0 = state.cluster_scores(0);
        assert_eq!(c0["a"], 0.3);
        assert_eq!(state.total_suppressed(0.01), 1);
    }

    #[test]
    fn snapshot_roundtrip() {
        let mut state = ClusteredSuppressionState::new();
        let mut u0 = HashMap::new();
        u0.insert("a".to_owned(), 0.3_f32);
        state.update_cluster(0, &u0, 100);
        let snap = state.snapshot();

        let mut restored = ClusteredSuppressionState::new();
        restored.restore(snap);
        assert_eq!(restored.cluster_scores(0)["a"], 0.3);
        assert_eq!(restored.cluster_last_updated(0)["a"], 100);
    }
}

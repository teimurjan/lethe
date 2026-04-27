//! Cross-encoder reranking + adaptive deep-pass — port of
//! `legacy/lethe/reranker.py`.
//!
//! `needs_deep_search` mirrors the Python check: trigger the deep
//! candidate pass when the top cross-encoder score from the shallow
//! batch is below `confidence_threshold`. Default 4.0 — picked in
//! checkpoint 6 and unchanged since.

use std::sync::Arc;

use crate::encoders::CrossEncoder;
use crate::Error;

/// Rerank shallow candidates, with the deep-pass decision baked in.
#[derive(Clone)]
pub struct Reranker {
    cross_encoder: Option<Arc<CrossEncoder>>,
    pub confidence_threshold: f32,
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reranker")
            .field("has_cross_encoder", &self.cross_encoder.is_some())
            .field("confidence_threshold", &self.confidence_threshold)
            .finish()
    }
}

impl Reranker {
    #[must_use]
    pub fn new(cross_encoder: Option<Arc<CrossEncoder>>, confidence_threshold: f32) -> Self {
        Self {
            cross_encoder,
            confidence_threshold,
        }
    }

    /// Score `(query, content)` pairs. Returns `(eid, score)` sorted
    /// descending by score. When no cross-encoder is configured, the
    /// candidates are returned in input order with score `0.0` for
    /// every entry (this is the test-only path; callers must not
    /// treat the score as meaningful in that case).
    pub fn rerank(
        &self,
        query: &str,
        candidates: &[(&str, &str)],
    ) -> Result<Vec<(String, f32)>, Error> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let Some(xenc) = &self.cross_encoder else {
            return Ok(candidates
                .iter()
                .map(|(eid, _)| ((*eid).to_owned(), 0.0))
                .collect());
        };
        let pairs: Vec<(&str, &str)> = candidates.iter().map(|(_, c)| (query, *c)).collect();
        let scores = xenc.predict(&pairs)?;
        let mut scored: Vec<(String, f32)> = candidates
            .iter()
            .zip(scores)
            .map(|((eid, _), s)| ((*eid).to_owned(), s))
            .collect();
        scored.sort_by(|(a_id, a), (b_id, b)| {
            b.partial_cmp(a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a_id.cmp(b_id))
        });
        Ok(scored)
    }

    /// Adaptive deep-pass trigger: top score below threshold (or no
    /// scores at all). Mirrors `Reranker.needs_deep_search`.
    #[must_use]
    pub fn needs_deep_search(&self, scores: &[f32]) -> bool {
        let Some(top) = scores.first() else {
            return true;
        };
        *top < self.confidence_threshold
    }
}

//! BM25Okapi port — bit-faithful to `rank_bm25.BM25Okapi` as used by
//! `src/lethe/vectors.py`.
//!
//! The `rank_bm25` reference uses these defaults:
//! * `k1 = 1.5`
//! * `b = 0.75`
//! * `epsilon = 0.25` — negative-IDF clipping coefficient
//!
//! IDF formula (per term `t` with document frequency `df_t`, corpus
//! size `N`):
//!     idf(t) = ln(N - df_t + 0.5) - ln(df_t + 0.5)
//! Negative IDFs (terms in more than ~half the corpus) are clipped to
//! `epsilon * average_idf` where `average_idf` is the unweighted mean
//! of all per-term IDFs (negatives included before clipping).
//!
//! Score for a query `q` over document `d`:
//!     score(d, q) = Σ_{t ∈ q} idf(t) · (tf_{t,d} · (k1+1))
//!                              / (tf_{t,d} + k1 · (1 - b + b · |d| / avgdl))
//!
//! Tokens repeated in the query contribute multiple times (linearly), as
//! `BM25Okapi.get_scores` iterates over the input list verbatim.

use std::collections::HashMap;

const DEFAULT_K1: f32 = 1.5;
const DEFAULT_B: f32 = 0.75;
const DEFAULT_EPSILON: f32 = 0.25;

/// Inverted-index BM25 over a fixed corpus.
///
/// Internals are `f64` to match `rank_bm25`'s numpy-default precision —
/// the Python reference accumulates per-doc scores in a `np.float64`
/// vector and using `f32` here was costing ~0.005 NDCG@10 on
/// LongMemEval purely from accumulation drift over 200k documents.
/// `get_scores` casts down to `f32` at the end so the public API and
/// downstream FAISS-flat path stay homogeneous.
///
/// Memory layout:
/// * `doc_freqs`: per-doc term-frequency map (sparse, keyed by token).
/// * `doc_len`: per-doc length (token count).
/// * `idf`: per-term IDF (f64), with negative-IDF clipping pre-applied.
#[derive(Debug, Clone)]
pub struct BM25Okapi {
    k1: f64,
    b: f64,
    avgdl: f64,
    doc_freqs: Vec<HashMap<String, u32>>,
    doc_len: Vec<u32>,
    idf: HashMap<String, f64>,
}

impl BM25Okapi {
    /// Build a BM25Okapi index from already-tokenized documents.
    ///
    /// `corpus` is a slice of token vectors, one per document (matching
    /// `BM25Okapi(tokenized_corpus)` in Python).
    #[must_use]
    pub fn new(corpus: &[Vec<String>]) -> Self {
        Self::with_params(corpus, DEFAULT_K1, DEFAULT_B, DEFAULT_EPSILON)
    }

    pub fn with_params(corpus: &[Vec<String>], k1: f32, b: f32, epsilon: f32) -> Self {
        let k1 = f64::from(k1);
        let b = f64::from(b);
        let epsilon = f64::from(epsilon);
        let n = corpus.len();
        let mut doc_freqs: Vec<HashMap<String, u32>> = Vec::with_capacity(n);
        let mut doc_len: Vec<u32> = Vec::with_capacity(n);
        let mut nd: HashMap<String, u32> = HashMap::new();
        let mut total_len: u64 = 0;

        for tokens in corpus {
            let mut freqs: HashMap<String, u32> = HashMap::with_capacity(tokens.len());
            for tok in tokens {
                *freqs.entry(tok.clone()).or_insert(0) += 1;
            }
            for tok in freqs.keys() {
                *nd.entry(tok.clone()).or_insert(0) += 1;
            }
            doc_len.push(tokens.len() as u32);
            total_len += tokens.len() as u64;
            doc_freqs.push(freqs);
        }

        let avgdl: f64 = if n == 0 {
            0.0
        } else {
            total_len as f64 / n as f64
        };

        let idf = compute_idf(&nd, n, epsilon);

        Self {
            k1,
            b,
            avgdl,
            doc_freqs,
            doc_len,
            idf,
        }
    }

    /// Number of indexed documents.
    #[must_use]
    pub fn corpus_size(&self) -> usize {
        self.doc_freqs.len()
    }

    /// BM25Okapi.get_scores port.
    ///
    /// Returns one score per document, in the same order as `corpus`
    /// passed to `new`. Tokens not in the IDF map (i.e. not present in
    /// any document) contribute zero — matching the Python `idf.get(q) or 0`.
    pub fn get_scores(&self, query: &[String]) -> Vec<f32> {
        let n = self.doc_freqs.len();
        if n == 0 || query.is_empty() {
            return vec![0.0_f32; n];
        }
        // Accumulate in f64 to match `rank_bm25` (numpy default), then
        // cast the public output to f32. Without this we drift ~5e-3
        // on NDCG@10 vs the Python reference at 200k corpus.
        let mut scores = vec![0.0_f64; n];
        for q in query {
            let Some(&idf) = self.idf.get(q) else {
                continue;
            };
            for (i, freqs) in self.doc_freqs.iter().enumerate() {
                let tf = f64::from(*freqs.get(q).unwrap_or(&0));
                if tf == 0.0 {
                    continue;
                }
                let dl = f64::from(self.doc_len[i]);
                let denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl);
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom;
            }
        }
        scores.into_iter().map(|x| x as f32).collect()
    }
}

fn compute_idf(nd: &HashMap<String, u32>, n: usize, epsilon: f64) -> HashMap<String, f64> {
    let n_f = n as f64;
    let mut idf: HashMap<String, f64> = HashMap::with_capacity(nd.len());
    let mut idf_sum = 0.0_f64;
    let mut negatives: Vec<String> = Vec::new();
    for (word, freq) in nd {
        let df = f64::from(*freq);
        let v = (n_f - df + 0.5).ln() - (df + 0.5).ln();
        idf.insert(word.clone(), v);
        idf_sum += v;
        if v < 0.0 {
            negatives.push(word.clone());
        }
    }
    if !idf.is_empty() {
        let average_idf = idf_sum / idf.len() as f64;
        let eps = epsilon * average_idf;
        for word in negatives {
            idf.insert(word, eps);
        }
    }
    idf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str) -> Vec<String> {
        crate::tokenize::tokenize_bm25(text)
    }

    #[test]
    fn empty_corpus_returns_empty_scores() {
        let bm = BM25Okapi::new(&[]);
        let scores = bm.get_scores(&tok("hello"));
        assert!(scores.is_empty());
    }

    #[test]
    fn empty_query_returns_zero_scores() {
        let docs = vec![tok("alpha beta"), tok("gamma delta")];
        let bm = BM25Okapi::new(&docs);
        let scores = bm.get_scores(&[]);
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn unknown_query_term_yields_zero() {
        let docs = vec![tok("alpha beta"), tok("gamma delta")];
        let bm = BM25Okapi::new(&docs);
        let scores = bm.get_scores(&tok("unrelated"));
        assert!(scores.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn rare_term_score_higher_than_common_term() {
        // 5 docs, "common" in 3 of them, "rare" in only doc 0.
        let docs = vec![
            tok("rare common alpha"),
            tok("common beta gamma"),
            tok("common delta epsilon"),
            tok("zeta eta theta"),
            tok("iota kappa lambda"),
        ];
        let bm = BM25Okapi::new(&docs);
        let by_rare = bm.get_scores(&tok("rare"));
        let by_common = bm.get_scores(&tok("common"));
        assert!(
            by_rare[0] > by_common[0],
            "rare term should outweigh common term, got {} vs {}",
            by_rare[0],
            by_common[0]
        );
    }

    #[test]
    fn returns_score_for_matching_doc() {
        let docs = vec![
            tok("MongoDB pool sizing notes."),
            tok("unrelated content about cats"),
            tok("another note about cars"),
            tok("another about cooking recipes"),
            tok("yet another about music theory"),
        ];
        let bm = BM25Okapi::new(&docs);
        // Tokenizer strips '?' from query; doc 0 has "mongodb".
        let scores = bm.get_scores(&tok("MongoDB?"));
        // Only doc 0 contains "mongodb"; in this small corpus IDF for
        // "mongodb" is positive, so its score must be the unique max.
        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 0);
        assert!(scores[0] > 0.0);
    }
}

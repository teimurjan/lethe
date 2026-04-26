//! ONNX bi-encoder + cross-encoder via the `ort` crate.
//!
//! Mirrors the Python `lethe.encoders` module which wraps `fastembed`.
//! For the Rust port we go straight to ONNX Runtime so we can feed the
//! same model files directly:
//!
//! * Bi-encoder default: `Xenova/all-MiniLM-L6-v2` (384-dim, mean-pooled,
//!   L2-normalized — same architecture as
//!   `sentence-transformers/all-MiniLM-L6-v2`).
//! * Cross-encoder default: `Xenova/ms-marco-MiniLM-L-6-v2` (binary
//!   relevance logit per `(query, passage)` pair).
//!
//! Both repos lay out `onnx/model.onnx` + `tokenizer.json` at the
//! HuggingFace root, downloaded via `hf-hub` into the standard cache
//! directory at `~/.cache/huggingface/hub/`.
//!
//! The shipped `lethe.encoders` aliases `cross-encoder/ms-marco-...`
//! to `Xenova/ms-marco-...`; we honor the same alias here for parity
//! with anyone who has model names in their config.toml.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use hf_hub::api::sync::Api;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::{DynValue, Value};
use tokenizers::Tokenizer;

use crate::Error;

const DEFAULT_BI: &str = "Xenova/all-MiniLM-L6-v2";
const DEFAULT_CROSS: &str = "Xenova/ms-marco-MiniLM-L-6-v2";

/// Resolve the legacy `sentence-transformers/...` and
/// `cross-encoder/...` model names that pre-existing config files may
/// carry to their Xenova ONNX equivalents.
#[must_use]
pub fn resolve_bi_name(name: &str) -> &str {
    match name {
        "sentence-transformers/all-MiniLM-L6-v2" | "all-MiniLM-L6-v2" => DEFAULT_BI,
        other => other,
    }
}

#[must_use]
pub fn resolve_cross_name(name: &str) -> &str {
    match name {
        "cross-encoder/ms-marco-MiniLM-L-6-v2" | "ms-marco-MiniLM-L-6-v2" => DEFAULT_CROSS,
        other => other,
    }
}

fn fetch_model(repo: &str) -> Result<(PathBuf, PathBuf), Error> {
    let api = Api::new().map_err(|e| Error::Encoder(format!("hf-hub init: {e}")))?;
    let repo_handle = api.model(repo.to_owned());

    // Common Xenova layout: onnx/model.onnx at the root + tokenizer.json.
    // Some repos use `model.onnx` directly. Try the nested path first;
    // fall back if it 404s.
    let model_path = repo_handle
        .get("onnx/model.onnx")
        .or_else(|_| repo_handle.get("model.onnx"))
        .map_err(|e| Error::Encoder(format!("download model.onnx for {repo}: {e}")))?;
    let tokenizer_path = repo_handle
        .get("tokenizer.json")
        .map_err(|e| Error::Encoder(format!("download tokenizer.json for {repo}: {e}")))?;
    Ok((model_path, tokenizer_path))
}

fn build_session(onnx_path: &std::path::Path) -> Result<Session, Error> {
    let mut builder =
        Session::builder().map_err(|e| Error::Encoder(format!("session builder: {e}")))?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Encoder(format!("optimization level: {e}")))?;
    // ONNX Runtime defaults to using all available cores for matmul; we
    // let it. Capping `intra_threads = 1` (the previous setting) made
    // the cross-encoder rerank ~5× slower than Python on the
    // LongMemEval bench because Python's fastembed does not cap
    // intra-op parallelism.
    builder
        .commit_from_file(onnx_path)
        .map_err(|e| Error::Encoder(format!("commit ONNX: {e}")))
}

/// Cap on `(input_ids)` length we feed into the model. Both the
/// MiniLM bi-encoder and the MS-MARCO cross-encoder are BERT-style
/// with `max_position_embeddings = 512`. Without an explicit
/// truncation config the `tokenizers` crate happily produces
/// 700-token (q, passage) pairs, and ONNX Runtime then aborts with
/// `Attempting to broadcast an axis by a dimension other than 1.
/// 512 by 700`. fastembed (Python) truncates implicitly; we must
/// match that.
const MODEL_MAX_LEN: usize = 512;

fn load_tokenizer(path: &std::path::Path) -> Result<Tokenizer, Error> {
    let mut tok = Tokenizer::from_file(path)
        .map_err(|e| Error::Encoder(format!("tokenizer.json: {e}")))?;
    tok.with_truncation(Some(tokenizers::TruncationParams {
        max_length: MODEL_MAX_LEN,
        // `LongestFirst` matches HuggingFace transformers' default for
        // sequence-pair encoding, which is what the cross-encoder uses.
        strategy: tokenizers::TruncationStrategy::LongestFirst,
        stride: 0,
        direction: tokenizers::TruncationDirection::Right,
    }))
    .map_err(|e| Error::Encoder(format!("set truncation: {e}")))?;
    Ok(tok)
}

/// Bi-encoder producing L2-normalized embeddings. Single text and
/// batched APIs match the Python contract: single → 1-D, batch → 2-D.
#[derive(Clone)]
pub struct BiEncoder {
    inner: Arc<EncoderInner>,
    dim: usize,
}

impl std::fmt::Debug for BiEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BiEncoder")
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

struct EncoderInner {
    /// `ort`'s `Session::run` requires `&mut self`. The underlying ONNX
    /// Runtime API is internally thread-safe, so we wrap the session
    /// in a `Mutex` and lock briefly per inference call. Retrieve in
    /// lethe doesn't have within-query parallelism on the model side,
    /// so this never contends meaningfully.
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// `true` if the ONNX graph requires `token_type_ids` as input
    /// (most BERT-family checkpoints do).
    needs_token_type_ids: bool,
}

impl BiEncoder {
    pub fn from_repo(repo: &str) -> Result<Self, Error> {
        let resolved = resolve_bi_name(repo);
        let (onnx, tok) = fetch_model(resolved)?;
        let session = build_session(&onnx)?;
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let tokenizer = load_tokenizer(&tok)?;
        let inner = Arc::new(EncoderInner {
            session: Mutex::new(session),
            tokenizer,
            needs_token_type_ids,
        });
        let dim = probe_dim_bi(&inner)?;
        Ok(Self { inner, dim })
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Encode one text. Returns a `dim`-long L2-normalized vector.
    pub fn encode(&self, text: &str) -> Result<Array1<f32>, Error> {
        let batch = self.encode_batch(&[text])?;
        Ok(batch.row(0).to_owned())
    }

    /// Encode a batch of texts. Returns `(N, dim)` L2-normalized.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Array2<f32>, Error> {
        if texts.is_empty() {
            return Ok(Array2::<f32>::zeros((0, self.dim)));
        }
        let encodings = self
            .inner
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Encoder(format!("tokenize: {e}")))?;
        let (input_ids, attention_mask, token_type_ids, max_len) =
            tensors_from_encodings(&encodings);
        let value = run_inference(
            &self.inner,
            input_ids,
            attention_mask.clone(),
            token_type_ids,
        )?;
        let last_hidden = value
            .try_extract_array::<f32>()
            .map_err(|e| Error::Encoder(format!("extract last_hidden_state: {e}")))?;
        let last_hidden = last_hidden
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| Error::Encoder(format!("reshape last_hidden_state: {e}")))?;
        let pooled = mean_pool(last_hidden.view(), &attention_mask, max_len);
        Ok(l2_normalize_rows(pooled))
    }
}

/// Cross-encoder producing one relevance score per `(query, passage)` pair.
#[derive(Clone)]
pub struct CrossEncoder {
    inner: Arc<EncoderInner>,
}

impl std::fmt::Debug for CrossEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossEncoder").finish_non_exhaustive()
    }
}

impl CrossEncoder {
    pub fn from_repo(repo: &str) -> Result<Self, Error> {
        let resolved = resolve_cross_name(repo);
        let (onnx, tok) = fetch_model(resolved)?;
        let session = build_session(&onnx)?;
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        let tokenizer = load_tokenizer(&tok)?;
        Ok(Self {
            inner: Arc::new(EncoderInner {
                session: Mutex::new(session),
                tokenizer,
                needs_token_type_ids,
            }),
        })
    }

    /// Score a slice of (query, passage) pairs.
    pub fn predict(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, Error> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        // The tokenizers crate accepts `(String, String)` for the
        // sequence-pair encoding form.
        let inputs: Vec<(String, String)> = pairs
            .iter()
            .map(|(a, b)| ((*a).to_owned(), (*b).to_owned()))
            .collect();
        let encodings = self
            .inner
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| Error::Encoder(format!("tokenize pairs: {e}")))?;
        let (input_ids, attention_mask, token_type_ids, _max_len) =
            tensors_from_encodings(&encodings);
        let value = run_inference(&self.inner, input_ids, attention_mask, token_type_ids)?;
        let logits = value
            .try_extract_array::<f32>()
            .map_err(|e| Error::Encoder(format!("extract logits: {e}")))?;
        // Cross-encoder MS-MARCO outputs `(batch, 1)`. Flatten to a vec.
        let flat: Vec<f32> = logits.iter().copied().collect();
        Ok(flat)
    }
}

fn probe_dim_bi(inner: &EncoderInner) -> Result<usize, Error> {
    let enc = inner
        .tokenizer
        .encode("hello".to_string(), true)
        .map_err(|e| Error::Encoder(format!("probe tokenize: {e}")))?;
    let (input_ids, attention_mask, token_type_ids, max_len) = tensors_from_encodings(&[enc]);
    let value = run_inference(inner, input_ids, attention_mask, token_type_ids)?;
    let arr = value
        .try_extract_array::<f32>()
        .map_err(|e| Error::Encoder(format!("probe extract: {e}")))?;
    let arr = arr
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| Error::Encoder(format!("probe reshape: {e}")))?;
    debug_assert_eq!(arr.shape()[1], max_len);
    Ok(arr.shape()[2])
}

fn tensors_from_encodings(
    encodings: &[tokenizers::Encoding],
) -> (Array2<i64>, Array2<i64>, Array2<i64>, usize) {
    let n = encodings.len();
    let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
    let mut input_ids = Array2::<i64>::zeros((n, max_len));
    let mut attention_mask = Array2::<i64>::zeros((n, max_len));
    let mut token_type_ids = Array2::<i64>::zeros((n, max_len));
    for (i, enc) in encodings.iter().enumerate() {
        for (j, &tok) in enc.get_ids().iter().enumerate() {
            input_ids[[i, j]] = i64::from(tok);
        }
        for (j, &m) in enc.get_attention_mask().iter().enumerate() {
            attention_mask[[i, j]] = i64::from(m);
        }
        for (j, &t) in enc.get_type_ids().iter().enumerate() {
            token_type_ids[[i, j]] = i64::from(t);
        }
    }
    (input_ids, attention_mask, token_type_ids, max_len)
}

fn run_inference(
    inner: &EncoderInner,
    input_ids: Array2<i64>,
    attention_mask: Array2<i64>,
    token_type_ids: Array2<i64>,
) -> Result<DynValue, Error> {
    let ids_value = Value::from_array(input_ids)
        .map_err(|e| Error::Encoder(format!("input_ids tensor: {e}")))?;
    let mask_value = Value::from_array(attention_mask)
        .map_err(|e| Error::Encoder(format!("attention_mask tensor: {e}")))?;

    let mut session = inner
        .session
        .lock()
        .map_err(|e| Error::Encoder(format!("session lock poisoned: {e}")))?;

    let outputs = if inner.needs_token_type_ids {
        let tt_value = Value::from_array(token_type_ids)
            .map_err(|e| Error::Encoder(format!("token_type_ids tensor: {e}")))?;
        session
            .run(ort::inputs![
                "input_ids" => ids_value,
                "attention_mask" => mask_value,
                "token_type_ids" => tt_value,
            ])
            .map_err(|e| Error::Encoder(format!("session.run: {e}")))?
    } else {
        session
            .run(ort::inputs![
                "input_ids" => ids_value,
                "attention_mask" => mask_value,
            ])
            .map_err(|e| Error::Encoder(format!("session.run: {e}")))?
    };

    // Take the first (and usually only) output. Both bi-encoder
    // (last_hidden_state) and cross-encoder (logits) ship as the first
    // declared output in the Xenova ONNX exports we target.
    let (_, value) = outputs
        .into_iter()
        .next()
        .ok_or_else(|| Error::Encoder("session returned no outputs".to_owned()))?;
    Ok(value)
}

fn mean_pool(
    last_hidden: ndarray::ArrayView3<'_, f32>,
    attention_mask: &Array2<i64>,
    max_len: usize,
) -> Array2<f32> {
    let n = last_hidden.shape()[0];
    let dim = last_hidden.shape()[2];
    let mut pooled = Array2::<f32>::zeros((n, dim));
    for i in 0..n {
        let mut count = 0_f32;
        for t in 0..max_len {
            if attention_mask[[i, t]] != 0 {
                let row = last_hidden.slice(ndarray::s![i, t, ..]);
                for (k, v) in row.iter().enumerate() {
                    pooled[[i, k]] += v;
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            for k in 0..dim {
                pooled[[i, k]] /= count;
            }
        }
    }
    pooled
}

fn l2_normalize_rows(mut matrix: Array2<f32>) -> Array2<f32> {
    for mut row in matrix.axis_iter_mut(Axis(0)) {
        let n = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n > 0.0 {
            for v in row.iter_mut() {
                *v /= n;
            }
        }
    }
    matrix
}

/// Convenience: L2-normalize a single 1-D vector. Useful for caller
/// code that handles already-encoded query embeddings.
#[must_use]
pub fn l2_normalize(view: ArrayView1<'_, f32>) -> Array1<f32> {
    let n = view.iter().map(|v| v * v).sum::<f32>().sqrt();
    if n == 0.0 {
        view.to_owned()
    } else {
        view.mapv(|v| v / n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_resolve_to_xenova_paths() {
        assert_eq!(
            resolve_bi_name("sentence-transformers/all-MiniLM-L6-v2"),
            DEFAULT_BI
        );
        assert_eq!(
            resolve_cross_name("cross-encoder/ms-marco-MiniLM-L-6-v2"),
            DEFAULT_CROSS
        );
        // Pass-through for already-resolved names.
        assert_eq!(resolve_bi_name(DEFAULT_BI), DEFAULT_BI);
    }

    #[test]
    fn l2_normalize_unit_vec_is_identity() {
        let v = ndarray::arr1(&[0.0_f32, 1.0]);
        let n = l2_normalize(v.view());
        assert!((n[0] - 0.0).abs() < 1e-6);
        assert!((n[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_returns_zero() {
        let v = ndarray::arr1(&[0.0_f32, 0.0]);
        let n = l2_normalize(v.view());
        assert_eq!(n, v);
    }

    // Live ONNX tests are gated behind an env var so `cargo test`
    // doesn't pull ~100 MB of model weights on every run. Set
    // `LETHE_TEST_ENCODERS=1` (and ensure network access) to exercise
    // the full path.
    #[test]
    #[ignore = "downloads ~100MB; opt in via LETHE_TEST_ENCODERS=1 cargo test --ignored"]
    fn bi_encoder_produces_unit_vector() {
        if std::env::var("LETHE_TEST_ENCODERS").ok().as_deref() != Some("1") {
            return;
        }
        let bi = BiEncoder::from_repo(DEFAULT_BI).expect("download bi-encoder");
        let v = bi.encode("hello world").expect("encode");
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected unit-norm, got {norm}");
        assert_eq!(v.len(), bi.dim());
    }

    #[test]
    #[ignore = "downloads ~100MB"]
    fn cross_encoder_orders_relevance() {
        if std::env::var("LETHE_TEST_ENCODERS").ok().as_deref() != Some("1") {
            return;
        }
        let cross = CrossEncoder::from_repo(DEFAULT_CROSS).expect("download cross");
        let pairs: Vec<(&str, &str)> = vec![
            (
                "MongoDB pool sizing",
                "Use asyncpg.create_pool with min_size=5, max_size=20.",
            ),
            ("MongoDB pool sizing", "The cat sat on the mat."),
        ];
        let scores = cross.predict(&pairs).expect("predict");
        assert_eq!(scores.len(), 2);
        // Relevant pair must outscore the irrelevant one.
        assert!(
            scores[0] > scores[1],
            "expected first pair > second, got {scores:?}"
        );
    }
}

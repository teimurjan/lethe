//! `lethe-bench` — Rust counterpart to `benchmarks/run_benchmark.py`.
//!
//! Runs the same five retrieval configurations on LongMemEval data,
//! reusing the precomputed embeddings exported by
//! `bench/prepare_lme_data.py`. Outputs JSON to stdout in the same
//! shape as `bench/run_lme_python.py` so the two are directly
//! diffable.
//!
//! The cross-encoder rerank step uses the production Rust path
//! (`lethe_core::encoders::CrossEncoder` over ONNX Runtime via `ort`),
//! while Python uses `lethe.encoders.OnnxCrossEncoder` (fastembed). Both
//! load `Xenova/ms-marco-MiniLM-L-6-v2`, so any drift between the two
//! pipelines is bounded by ONNX numerical precision, not algorithmic
//! difference.

#![allow(clippy::print_stdout)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use lethe_core::bm25::BM25Okapi;
use lethe_core::encoders::CrossEncoder;
use lethe_core::faiss_flat::FlatIp;
use lethe_core::tokenize::tokenize_bm25;
use ndarray::{Array2, ArrayView1};

#[derive(Parser, Debug)]
#[command(
    name = "lethe-bench",
    version,
    about = "Rust parity bench. Each subcommand emits JSON to stdout in a shape the matching Python harness can diff."
)]
struct Cli {
    /// Path to `data/` containing longmemeval_*.json + lme_rust/.
    #[arg(long, global = true, default_value = "data")]
    data: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the 5 LongMemEval retrieval configs and emit NDCG/Recall as JSON.
    Longmemeval,
    /// BM25 score vector for each query in stdin JSON `{"queries": ["..."]}`.
    Bm25 {
        /// Path to JSON `{"queries": [...]}`. Use `-` for stdin.
        #[arg(long)]
        queries: PathBuf,
    },
    /// FlatIP top-K for each query embedding in stdin JSON `{"queries": [[f32...]], "k": 30}`.
    FlatIp {
        #[arg(long)]
        queries: PathBuf,
    },
    /// Cross-encoder logits for `{"pairs": [["q","content"], ...]}`.
    Xenc {
        #[arg(long)]
        pairs: PathBuf,
    },
}

#[derive(serde::Serialize)]
struct ConfigMetrics {
    ndcg: f64,
    recall: f64,
    n_eval: usize,
    time_s: f64,
}

#[derive(serde::Serialize)]
struct Output {
    impl_: String,
    configs: HashMap<String, ConfigMetrics>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Longmemeval => cmd_longmemeval(&cli.data),
        Cmd::Bm25 { queries } => cmd_bm25(&cli.data, &queries),
        Cmd::FlatIp { queries } => cmd_flat_ip(&cli.data, &queries),
        Cmd::Xenc { pairs } => cmd_xenc(&pairs),
    }
}

#[allow(clippy::too_many_lines)]
fn cmd_longmemeval(data: &std::path::Path) -> Result<()> {
    let cli_data = data;
    let lme_rust = cli_data.join("lme_rust");
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(lme_rust.join("meta.json")).context("read meta.json")?,
    )?;
    let n_corpus = meta["n_corpus"].as_u64().unwrap() as usize;
    let n_queries = meta["n_queries"].as_u64().unwrap() as usize;
    let dim = meta["dim"].as_u64().unwrap() as usize;
    eprintln!("[rust] n_corpus={n_corpus}  n_queries={n_queries}  dim={dim}");

    let corpus_ids: Vec<String> = std::fs::read_to_string(lme_rust.join("corpus_ids.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    let query_ids: Vec<String> = std::fs::read_to_string(lme_rust.join("query_ids.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    if corpus_ids.len() != n_corpus || query_ids.len() != n_queries {
        return Err(anyhow!("id count mismatch vs meta.json"));
    }

    eprintln!("[rust] reading corpus embeddings…");
    let corpus_embs = read_f32_matrix(&lme_rust.join("corpus_embeddings.bin"), n_corpus, dim)?;
    eprintln!("[rust] reading query embeddings…");
    let query_embs = read_f32_matrix(&lme_rust.join("query_embeddings.bin"), n_queries, dim)?;

    let sampled: Vec<usize> = std::fs::read_to_string(lme_rust.join("sampled_query_indices.txt"))?
        .lines()
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    eprintln!("[rust] sampled {} queries", sampled.len());

    let qrels: HashMap<String, HashMap<String, f64>> = serde_json::from_str(
        &std::fs::read_to_string(cli_data.join("longmemeval_qrels.json"))?,
    )?;
    let corpus_content: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        cli_data.join("longmemeval_corpus.json"),
    )?)?;
    let query_texts: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        cli_data.join("longmemeval_queries.json"),
    )?)?;

    eprintln!("[rust] building FAISS-equivalent FlatIp…");
    let mut flat = FlatIp::new(dim);
    flat.add_batch(corpus_embs.view())?;

    eprintln!("[rust] tokenizing corpus…");
    let tokenized: Vec<Vec<String>> = corpus_ids
        .iter()
        .map(|cid| tokenize_bm25(corpus_content.get(cid).map_or("", String::as_str)))
        .collect();
    eprintln!("[rust] building BM25…");
    let bm25 = BM25Okapi::new(&tokenized);

    eprintln!("[rust] loading cross-encoder (ONNX)…");
    let xenc = CrossEncoder::from_repo("Xenova/ms-marco-MiniLM-L-6-v2")
        .map_err(|e| anyhow!("cross-encoder: {e}"))?;
    let _ = xenc.predict(&[("warm", "warm")]);

    let mut configs: HashMap<String, ConfigMetrics> = HashMap::new();

    // 1. vector_only
    {
        let t0 = Instant::now();
        let mut ndcgs = Vec::new();
        let mut recalls = Vec::new();
        for &i in &sampled {
            let qid = &query_ids[i];
            let qrel = match qrels.get(qid) {
                Some(q) if !q.is_empty() => q,
                _ => continue,
            };
            let qe = query_embs.row(i);
            let top = flat.search(qe, 10).map_err(|e| anyhow!("flat: {e}"))?;
            let top_ids: Vec<String> = top
                .into_iter()
                .map(|(idx, _)| corpus_ids[idx].clone())
                .collect();
            ndcgs.push(ndcg_at_k(&top_ids, qrel, 10));
            recalls.push(recall_at_k(&top_ids, qrel, 10));
        }
        configs.insert(
            "vector_only".to_owned(),
            ConfigMetrics {
                ndcg: mean(&ndcgs),
                recall: mean(&recalls),
                n_eval: ndcgs.len(),
                time_s: t0.elapsed().as_secs_f64(),
            },
        );
    }

    // 2. bm25_only
    {
        let t0 = Instant::now();
        let mut ndcgs = Vec::new();
        let mut recalls = Vec::new();
        for &i in &sampled {
            let qid = &query_ids[i];
            let qrel = match qrels.get(qid) {
                Some(q) if !q.is_empty() => q,
                _ => continue,
            };
            let qt = query_texts.get(qid).cloned().unwrap_or_default();
            let scores = bm25.get_scores(&tokenize_bm25(&qt));
            let top_ids = top_k_ids(&scores, &corpus_ids, 10);
            ndcgs.push(ndcg_at_k(&top_ids, qrel, 10));
            recalls.push(recall_at_k(&top_ids, qrel, 10));
        }
        configs.insert(
            "bm25_only".to_owned(),
            ConfigMetrics {
                ndcg: mean(&ndcgs),
                recall: mean(&recalls),
                n_eval: ndcgs.len(),
                time_s: t0.elapsed().as_secs_f64(),
            },
        );
    }

    // 3. hybrid_rrf
    {
        let t0 = Instant::now();
        let mut ndcgs = Vec::new();
        let mut recalls = Vec::new();
        for &i in &sampled {
            let qid = &query_ids[i];
            let qrel = match qrels.get(qid) {
                Some(q) if !q.is_empty() => q,
                _ => continue,
            };
            let qt = query_texts.get(qid).cloned().unwrap_or_default();
            let qe = query_embs.row(i);
            let vec_ids: Vec<&str> = flat
                .search(qe, 30)?
                .into_iter()
                .map(|(idx, _)| corpus_ids[idx].as_str())
                .collect();
            let scores = bm25.get_scores(&tokenize_bm25(&qt));
            let bm_ids: Vec<&str> = top_k_ids_ref(&scores, &corpus_ids, 30);
            let merged = lethe_core::rrf::rrf_merge(&[bm_ids, vec_ids]);
            let top_ids: Vec<String> = merged.into_iter().take(10).map(|(s, _)| s).collect();
            ndcgs.push(ndcg_at_k(&top_ids, qrel, 10));
            recalls.push(recall_at_k(&top_ids, qrel, 10));
        }
        configs.insert(
            "hybrid_rrf".to_owned(),
            ConfigMetrics {
                ndcg: mean(&ndcgs),
                recall: mean(&recalls),
                n_eval: ndcgs.len(),
                time_s: t0.elapsed().as_secs_f64(),
            },
        );
    }

    // 4. vector_xenc
    {
        let t0 = Instant::now();
        let mut ndcgs = Vec::new();
        let mut recalls = Vec::new();
        for &i in &sampled {
            let qid = &query_ids[i];
            let qrel = match qrels.get(qid) {
                Some(q) if !q.is_empty() => q,
                _ => continue,
            };
            let qt = query_texts.get(qid).cloned().unwrap_or_default();
            let qe = query_embs.row(i);
            let cand: Vec<usize> = flat.search(qe, 30)?.into_iter().map(|(i, _)| i).collect();
            let cand_ids: Vec<String> = cand.iter().map(|&i| corpus_ids[i].clone()).collect();
            let pairs: Vec<(&str, &str)> = cand_ids
                .iter()
                .map(|id| {
                    (
                        qt.as_str(),
                        corpus_content.get(id).map_or("", String::as_str),
                    )
                })
                .collect();
            let scores = xenc.predict(&pairs).map_err(|e| anyhow!("xenc: {e}"))?;
            let mut scored: Vec<(String, f32)> = cand_ids.into_iter().zip(scores).collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_ids: Vec<String> = scored.into_iter().take(10).map(|(s, _)| s).collect();
            ndcgs.push(ndcg_at_k(&top_ids, qrel, 10));
            recalls.push(recall_at_k(&top_ids, qrel, 10));
        }
        configs.insert(
            "vector_xenc".to_owned(),
            ConfigMetrics {
                ndcg: mean(&ndcgs),
                recall: mean(&recalls),
                n_eval: ndcgs.len(),
                time_s: t0.elapsed().as_secs_f64(),
            },
        );
    }

    // 5. lethe_full = BM25+vector dedup union → cross-encoder rerank
    {
        let t0 = Instant::now();
        let mut ndcgs = Vec::new();
        let mut recalls = Vec::new();
        for &i in &sampled {
            let qid = &query_ids[i];
            let qrel = match qrels.get(qid) {
                Some(q) if !q.is_empty() => q,
                _ => continue,
            };
            let qt = query_texts.get(qid).cloned().unwrap_or_default();
            let qe = query_embs.row(i);
            let vec_ids: Vec<String> = flat
                .search(qe, 30)?
                .into_iter()
                .map(|(idx, _)| corpus_ids[idx].clone())
                .collect();
            let bm_scores = bm25.get_scores(&tokenize_bm25(&qt));
            let bm_ids: Vec<String> = top_k_ids(&bm_scores, &corpus_ids, 30);
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut union: Vec<String> = Vec::with_capacity(60);
            for id in vec_ids.into_iter().chain(bm_ids) {
                if seen.insert(id.clone()) {
                    union.push(id);
                }
            }
            let pairs: Vec<(&str, &str)> = union
                .iter()
                .map(|id| {
                    (
                        qt.as_str(),
                        corpus_content.get(id).map_or("", String::as_str),
                    )
                })
                .collect();
            let scores = xenc.predict(&pairs).map_err(|e| anyhow!("xenc: {e}"))?;
            let mut scored: Vec<(String, f32)> = union.into_iter().zip(scores).collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_ids: Vec<String> = scored.into_iter().take(10).map(|(s, _)| s).collect();
            ndcgs.push(ndcg_at_k(&top_ids, qrel, 10));
            recalls.push(recall_at_k(&top_ids, qrel, 10));
        }
        configs.insert(
            "lethe_full".to_owned(),
            ConfigMetrics {
                ndcg: mean(&ndcgs),
                recall: mean(&recalls),
                n_eval: ndcgs.len(),
                time_s: t0.elapsed().as_secs_f64(),
            },
        );
    }

    let out = Output {
        impl_: "rust".to_owned(),
        configs,
    };
    let json = serde_json::to_string_pretty(&serde_json::json!({
        "impl": out.impl_,
        "configs": out.configs,
    }))?;
    println!("{json}");
    Ok(())
}

// -------- component subcommands ---------------------------------------------
//
// These exist so `bench/components.py` can ask the Rust core for a single
// piece of the pipeline (BM25 score vector, FlatIP top-K, cross-encoder
// logits) and diff it against the Python reference numerically.

fn load_corpus_for_components(data: &std::path::Path) -> Result<(Vec<String>, Array2<f32>)> {
    let lme_rust = data.join("lme_rust");
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(lme_rust.join("meta.json")).context("meta.json")?,
    )?;
    let n_corpus = meta["n_corpus"].as_u64().unwrap() as usize;
    let dim = meta["dim"].as_u64().unwrap() as usize;
    let ids: Vec<String> = std::fs::read_to_string(lme_rust.join("corpus_ids.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    let embs = read_f32_matrix(&lme_rust.join("corpus_embeddings.bin"), n_corpus, dim)?;
    Ok((ids, embs))
}

#[derive(serde::Deserialize)]
struct QueriesInput {
    queries: Vec<String>,
}

fn cmd_bm25(data: &std::path::Path, queries_path: &std::path::Path) -> Result<()> {
    let input: QueriesInput = serde_json::from_str(&read_input(queries_path)?)?;
    let (corpus_ids, _embs) = load_corpus_for_components(data)?;
    let corpus_content: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        data.join("longmemeval_corpus.json"),
    )?)?;
    let tokenized: Vec<Vec<String>> = corpus_ids
        .iter()
        .map(|cid| tokenize_bm25(corpus_content.get(cid).map_or("", String::as_str)))
        .collect();
    let bm25 = BM25Okapi::new(&tokenized);
    // Per query: emit the f32 score vector. Python diffs element-wise.
    let mut out = Vec::with_capacity(input.queries.len());
    for q in &input.queries {
        let scores = bm25.get_scores(&tokenize_bm25(q));
        out.push(scores);
    }
    let payload = serde_json::json!({
        "impl": "rust",
        "n_queries": input.queries.len(),
        "n_corpus": corpus_ids.len(),
        "scores": out,
    });
    println!("{}", serde_json::to_string(&payload)?);
    Ok(())
}

#[derive(serde::Deserialize)]
struct EmbeddingQueriesInput {
    /// Indices into `query_embeddings.bin`. Avoids serializing 384-d
    /// vectors as JSON which is slow and lossy at f32 precision.
    query_indices: Vec<usize>,
    k: usize,
}

fn cmd_flat_ip(data: &std::path::Path, queries_path: &std::path::Path) -> Result<()> {
    let input: EmbeddingQueriesInput = serde_json::from_str(&read_input(queries_path)?)?;
    let (corpus_ids, corpus_embs) = load_corpus_for_components(data)?;
    let lme_rust = data.join("lme_rust");
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(lme_rust.join("meta.json"))?)?;
    let n_queries = meta["n_queries"].as_u64().unwrap() as usize;
    let dim = meta["dim"].as_u64().unwrap() as usize;
    let query_embs = read_f32_matrix(&lme_rust.join("query_embeddings.bin"), n_queries, dim)?;
    let mut flat = FlatIp::new(dim);
    flat.add_batch(corpus_embs.view())?;
    let mut out = Vec::with_capacity(input.query_indices.len());
    for &qi in &input.query_indices {
        let qe = query_embs.row(qi);
        let top: Vec<(String, f32)> = flat
            .search(qe, input.k)?
            .into_iter()
            .map(|(idx, score)| (corpus_ids[idx].clone(), score))
            .collect();
        out.push(top);
    }
    let payload = serde_json::json!({
        "impl": "rust",
        "k": input.k,
        "results": out,
    });
    println!("{}", serde_json::to_string(&payload)?);
    Ok(())
}

#[derive(serde::Deserialize)]
struct PairsInput {
    pairs: Vec<(String, String)>,
}

fn cmd_xenc(pairs_path: &std::path::Path) -> Result<()> {
    let input: PairsInput = serde_json::from_str(&read_input(pairs_path)?)?;
    let xenc = CrossEncoder::from_repo("Xenova/ms-marco-MiniLM-L-6-v2")
        .map_err(|e| anyhow!("cross-encoder: {e}"))?;
    let pairs_ref: Vec<(&str, &str)> = input
        .pairs
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let scores = xenc
        .predict(&pairs_ref)
        .map_err(|e| anyhow!("predict: {e}"))?;
    let payload = serde_json::json!({
        "impl": "rust",
        "n_pairs": input.pairs.len(),
        "logits": scores,
    });
    println!("{}", serde_json::to_string(&payload)?);
    Ok(())
}

fn read_input(path: &std::path::Path) -> Result<String> {
    if path == std::path::Path::new("-") {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        Ok(buf)
    } else {
        Ok(std::fs::read_to_string(path)?)
    }
}

fn read_f32_matrix(path: &std::path::Path, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let expected = rows * cols * 4;
    if bytes.len() != expected {
        return Err(anyhow!(
            "{}: expected {} bytes, got {}",
            path.display(),
            expected,
            bytes.len()
        ));
    }
    let mut floats = vec![0_f32; rows * cols];
    for (chunk, dst) in bytes.chunks_exact(4).zip(floats.iter_mut()) {
        *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(Array2::from_shape_vec((rows, cols), floats)?)
}

fn top_k_ids(scores: &[f32], ids: &[String], k: usize) -> Vec<String> {
    let view = ArrayView1::from(scores);
    lethe_core::faiss_flat::top_k_desc(view, k)
        .into_iter()
        .map(|(i, _)| ids[i].clone())
        .collect()
}

fn top_k_ids_ref<'a>(scores: &[f32], ids: &'a [String], k: usize) -> Vec<&'a str> {
    let view = ArrayView1::from(scores);
    lethe_core::faiss_flat::top_k_desc(view, k)
        .into_iter()
        .map(|(i, _)| ids[i].as_str())
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn ndcg_at_k(retrieved: &[String], qrels: &HashMap<String, f64>, k: usize) -> f64 {
    let mut dcg = 0.0;
    for (i, eid) in retrieved.iter().take(k).enumerate() {
        let rel = qrels.get(eid).copied().unwrap_or(0.0);
        dcg += (2_f64.powf(rel) - 1.0) / ((i as f64 + 2.0).log2());
    }
    let mut sorted_rels: Vec<f64> = qrels.values().copied().collect();
    sorted_rels.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut ideal = 0.0;
    for (i, r) in sorted_rels.iter().take(k).enumerate() {
        ideal += (2_f64.powf(*r) - 1.0) / ((i as f64 + 2.0).log2());
    }
    if ideal > 0.0 {
        dcg / ideal
    } else {
        0.0
    }
}

fn recall_at_k(retrieved: &[String], qrels: &HashMap<String, f64>, k: usize) -> f64 {
    if qrels.is_empty() {
        return 0.0;
    }
    let mut hits = 0_usize;
    for eid in retrieved.iter().take(k) {
        if qrels.contains_key(eid) {
            hits += 1;
        }
    }
    hits as f64 / qrels.len() as f64
}

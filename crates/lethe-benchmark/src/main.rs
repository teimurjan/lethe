//! `lethe-benchmark` — Rust counterpart to `research_playground/baseline/run.py`.
//!
//! Runs the same five retrieval configurations on LongMemEval data,
//! reusing the precomputed embeddings exported by
//! `research_playground/rust_migration/prepare.py`. Outputs JSON to stdout in the
//! same shape as `research_playground/rust_migration/longmemeval.py --impl python`
//! so the two are directly diffable.
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
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use lethe_core::bm25::BM25Okapi;
use lethe_core::encoders::{BiEncoder, CrossEncoder, Pooling};
use lethe_core::faiss_flat::FlatIp;
use lethe_core::fields::{extract_entities, extract_title};
use lethe_core::memory_store::{MemoryStore, StoreConfig};
use lethe_core::rif::RifConfig;
use lethe_core::tokenize::tokenize_bm25;
use ndarray::{Array2, ArrayView1};

#[derive(Parser, Debug)]
#[command(
    name = "lethe-benchmark",
    version,
    about = "Rust parity bench. Each subcommand emits JSON to stdout in a shape the matching Python harness can diff."
)]
struct Cli {
    /// Path to `tmp_data/` containing longmemeval_*.json + lme_rust/.
    #[arg(long, global = true, default_value = "tmp_data")]
    data: PathBuf,

    /// HuggingFace repo of the cross-encoder reranker. Must export an
    /// ONNX model + tokenizer.json at the standard paths. Default
    /// matches the production pipeline. A different value here only
    /// changes the rerank stage; bi-encoder embeddings stay fixed
    /// because they're loaded from the pre-baked `lme_rust/` files.
    #[arg(long, global = true, default_value = "Xenova/ms-marco-MiniLM-L-6-v2")]
    cross_encoder: String,

    /// Cap the number of evaluated queries — uses the first N entries
    /// of `sampled_query_indices.txt`. Useful for fast iteration on
    /// expensive rerankers. `0` means no cap.
    #[arg(long, global = true, default_value_t = 0)]
    sample_limit: usize,

    /// HuggingFace repo of the bi-encoder. The default reads the
    /// existing `lme_rust/` cache (Python-prepped MiniLM-L6 embeddings).
    /// Setting a different value reads `lme_<sanitized>/` instead, which
    /// must be generated first via the `prepare-embeddings` subcommand.
    #[arg(long, global = true, default_value = "Xenova/all-MiniLM-L6-v2")]
    bi_encoder: String,

    /// Pooling strategy for the bi-encoder. mean = MiniLM/mpnet,
    /// cls = BGE/Snowflake/Nomic. Wrong choice silently destroys NDCG.
    #[arg(long, global = true, default_value = "mean")]
    pooling: String,

    /// Optional path-within-repo of an ONNX variant (e.g.
    /// `onnx/model_int8.onnx`). When unset, uses the canonical
    /// `onnx/model.onnx`. Quantized variants give 3-4× CPU speedup
    /// at small NDCG cost, useful for fast prep iterations.
    #[arg(long, global = true)]
    bi_encoder_onnx: Option<String>,

    /// Read late-chunking embeddings (`lme_<sanitized>_late/`)
    /// produced by `prepare-embeddings-late`, instead of the
    /// standard `lme_<sanitized>/` cache.
    #[arg(long, global = true)]
    late: bool,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the 5 LongMemEval retrieval configs and emit NDCG/Recall as JSON.
    Longmemeval,
    /// Re-encode the LongMemEval corpus + queries with `--bi-encoder`
    /// and write a fresh `lme_<sanitized>/` cache the bench can read.
    /// Idempotent — does nothing if the cache already exists, unless
    /// `--force` is passed.
    PrepareEmbeddings {
        #[arg(long)]
        force: bool,
        /// Encoding batch size. 32 fits comfortably for 768-d models on
        /// 16GB; bump to 64 for smaller models.
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
    },
    /// Late-chunking variant: re-encode each LongMemEval *session*
    /// in a single long-context forward pass, then mean-pool over each
    /// turn's token span. Requires a long-context bi-encoder
    /// (e.g. `nomic-ai/nomic-embed-text-v1.5`, 8k context).
    /// Writes to `lme_<sanitized>_late/` so the bench can read it via
    /// `--bi-encoder <repo> --late`.
    PrepareEmbeddingsLate {
        #[arg(long)]
        force: bool,
        /// Tokenizer/model max sequence length. Must be ≥ the model's
        /// `max_position_embeddings`. Sessions longer than this fall
        /// back to per-turn encoding (no late-chunking benefit).
        #[arg(long, default_value_t = 8192)]
        max_len: usize,
    },
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
    /// Full-pipeline A/B on LongMemEval through the production
    /// `MemoryStore` (RIF + cross-encoder rerank): baseline vs. after an
    /// offline `dedupe` compaction. Emits NDCG@10 / Recall@10 as JSON.
    RifEval {
        /// Cosine cutoff for the compaction pass.
        #[arg(long, default_value_t = 0.95)]
        threshold: f32,
        /// RIF clusters (matches the production default).
        #[arg(long, default_value_t = 30)]
        n_clusters: u32,
    },
}

/// Map a HuggingFace repo to a filesystem-safe subdir of `--data`.
/// `Xenova/all-MiniLM-L6-v2` is the canonical baseline whose cache lives
/// at `lme_rust/` (matching `research_playground/rust_migration/prepare.py`); every
/// other model gets a parallel `lme_<sanitized>/` directory.
fn lme_dir_name(bi_encoder: &str) -> String {
    if bi_encoder == "Xenova/all-MiniLM-L6-v2"
        || bi_encoder == "sentence-transformers/all-MiniLM-L6-v2"
        || bi_encoder == "all-MiniLM-L6-v2"
    {
        return "lme_rust".to_owned();
    }
    let sanitized: String = bi_encoder
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("lme_{sanitized}")
}

fn lme_late_dir_name(bi_encoder: &str) -> String {
    format!("{}_late", lme_dir_name(bi_encoder))
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
    let pooling = Pooling::parse(&cli.pooling).map_err(|e| anyhow!(e.to_string()))?;
    match cli.cmd {
        Cmd::Longmemeval => cmd_longmemeval(
            &cli.data,
            &cli.cross_encoder,
            cli.sample_limit,
            &cli.bi_encoder,
            cli.late,
        ),
        Cmd::PrepareEmbeddings { force, batch_size } => cmd_prepare_embeddings(
            &cli.data,
            &cli.bi_encoder,
            cli.bi_encoder_onnx.as_deref(),
            pooling,
            force,
            batch_size,
        ),
        Cmd::PrepareEmbeddingsLate { force, max_len } => cmd_prepare_embeddings_late(
            &cli.data,
            &cli.bi_encoder,
            cli.bi_encoder_onnx.as_deref(),
            pooling,
            force,
            max_len,
        ),
        Cmd::Bm25 { queries } => cmd_bm25(&cli.data, &queries),
        Cmd::FlatIp { queries } => cmd_flat_ip(&cli.data, &queries),
        Cmd::Xenc { pairs } => cmd_xenc(&pairs, &cli.cross_encoder),
        Cmd::RifEval {
            threshold,
            n_clusters,
        } => cmd_rif_eval(
            &cli.data,
            &cli.cross_encoder,
            &cli.bi_encoder,
            cli.sample_limit,
            threshold,
            n_clusters,
        ),
    }
}

#[allow(clippy::too_many_lines)]
fn cmd_longmemeval(
    data: &std::path::Path,
    cross_encoder_repo: &str,
    sample_limit: usize,
    bi_encoder_repo: &str,
    late: bool,
) -> Result<()> {
    let cli_data = data;
    let lme_rust = if late {
        cli_data.join(lme_late_dir_name(bi_encoder_repo))
    } else {
        cli_data.join(lme_dir_name(bi_encoder_repo))
    };
    eprintln!("[rust] reading from {}", lme_rust.display());
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(lme_rust.join("meta.json"))
            .with_context(|| format!("read {}/meta.json", lme_rust.display()))?,
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

    let mut sampled: Vec<usize> =
        std::fs::read_to_string(lme_rust.join("sampled_query_indices.txt"))?
            .lines()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
    if sample_limit > 0 && sampled.len() > sample_limit {
        sampled.truncate(sample_limit);
    }
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

    // Multi-field views — extracted once, indexed in their own BM25.
    // Hypothesis: BM25 is the load-bearing leg on this corpus
    // (`bm25_only` 2× `vector_only`), so giving BM25 richer signal —
    // entity tokens and the first-line "title" — should lift NDCG
    // more than swapping models. See "Implementation note: reranker
    // + bi-encoder ablation" in `RESEARCH_JOURNEY.md`.
    eprintln!("[rust] extracting entity / title fields…");
    let t_fields = Instant::now();
    let tokenized_entities: Vec<Vec<String>> = corpus_ids
        .iter()
        .map(|cid| {
            let body = corpus_content.get(cid).map_or("", String::as_str);
            // Entity extractor already lowercases — we only need to
            // split on word boundaries to match BM25's expected token
            // shape. tokenize_bm25 handles punctuation in joined form.
            let ents = extract_entities(body);
            tokenize_bm25(&ents.join(" "))
        })
        .collect();
    let tokenized_titles: Vec<Vec<String>> = corpus_ids
        .iter()
        .map(|cid| {
            let body = corpus_content.get(cid).map_or("", String::as_str);
            tokenize_bm25(&extract_title(body))
        })
        .collect();
    let n_with_entities = tokenized_entities.iter().filter(|v| !v.is_empty()).count();
    let n_with_titles = tokenized_titles.iter().filter(|v| !v.is_empty()).count();
    eprintln!(
        "[rust] extracted: entities on {n_with_entities}/{n_corpus} chunks, \
         titles on {n_with_titles}/{n_corpus} ({:.1}s)",
        t_fields.elapsed().as_secs_f64()
    );
    eprintln!("[rust] building entity + title BM25 indexes…");
    let bm25_entities = BM25Okapi::new(&tokenized_entities);
    let bm25_titles = BM25Okapi::new(&tokenized_titles);

    // Field-boosted body: append entity tokens (×2 for upweighting)
    // and title tokens to the body before BM25 indexing. Single
    // BM25 instead of RRF-merging separate fields — equal-weight
    // RRF on this corpus regressed `lethe_full` by 3.6pp because
    // the entity field has 63% empty docs that pollute the merge.
    // Field-boosted indexing lets entity tokens raise TF in the
    // single body index without making them an independent
    // retrieval source.
    eprintln!("[rust] building field-boosted body BM25…");
    let tokenized_boosted: Vec<Vec<String>> = (0..n_corpus)
        .map(|i| {
            let mut toks = tokenized[i].clone();
            // 2× weight on entity tokens: appear once via the
            // body text already, twice more here. Title tokens
            // get +1 boost.
            toks.extend(tokenized_entities[i].iter().cloned());
            toks.extend(tokenized_entities[i].iter().cloned());
            toks.extend(tokenized_titles[i].iter().cloned());
            toks
        })
        .collect();
    let bm25_boosted = BM25Okapi::new(&tokenized_boosted);

    eprintln!("[rust] loading cross-encoder (ONNX) repo={cross_encoder_repo}…");
    let xenc = CrossEncoder::from_repo(cross_encoder_repo)
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

    // 6. lethe_multifield = BM25(body) ⊕ BM25(entities) ⊕ BM25(title)
    //    ⊕ vector → RRF → cross-encoder rerank.
    //    Hypothesis: separating the entity / title signal from the
    //    body BM25 lets the reranker see candidates the single-field
    //    BM25 was diluting. Top-K mirrors the memo'd prototype:
    //    body=30 / entities=20 / title=20 / dense=30.
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

            // Body BM25 query is the raw query text. Entities &
            // title BM25s receive the same query text — extracting
            // entities from a typical question pulls back too few
            // tokens to score reliably (verified manually on a few
            // LongMemEval queries).
            let q_tokens = tokenize_bm25(&qt);
            let body_scores = bm25.get_scores(&q_tokens);
            let body_ids: Vec<&str> = top_k_ids_ref(&body_scores, &corpus_ids, 30);
            let entity_scores = bm25_entities.get_scores(&q_tokens);
            let entity_ids: Vec<&str> = top_k_ids_ref(&entity_scores, &corpus_ids, 20);
            let title_scores = bm25_titles.get_scores(&q_tokens);
            let title_ids: Vec<&str> = top_k_ids_ref(&title_scores, &corpus_ids, 20);

            let merged = lethe_core::rrf::rrf_merge(&[body_ids, entity_ids, title_ids, vec_ids]);
            // Keep up to 60 candidates for the rerank pool — same
            // budget as `lethe_full` so the comparison isolates
            // signal quality, not pool size.
            let union: Vec<String> = merged.into_iter().take(60).map(|(s, _)| s).collect();

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
            "lethe_multifield".to_owned(),
            ConfigMetrics {
                ndcg: mean(&ndcgs),
                recall: mean(&recalls),
                n_eval: ndcgs.len(),
                time_s: t0.elapsed().as_secs_f64(),
            },
        );
    }

    // 7. lethe_field_boost = BM25(body + 2×entities + title) ⊕ vector
    //    → dedup union → cross-encoder rerank.
    //    Same shape as `lethe_full`, just with entity- and title-
    //    boosted indexing on the BM25 leg. Tests whether enriching
    //    the single index lifts NDCG (vs the parallel-field
    //    multi-RRF approach above which regressed).
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
            let bm_scores = bm25_boosted.get_scores(&tokenize_bm25(&qt));
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
            "lethe_field_boost".to_owned(),
            ConfigMetrics {
                ndcg: mean(&ndcgs),
                recall: mean(&recalls),
                n_eval: ndcgs.len(),
                time_s: t0.elapsed().as_secs_f64(),
            },
        );
    }

    // 8. bm25_boost_only = BM25(body + 2×entities + title), top-10.
    //    Sanity: did field-boosted indexing change BM25-alone
    //    rankings? If `bm25_boost_only` ≥ `bm25_only`, the boost
    //    helps the index; if ≤, it doesn't and `lethe_field_boost`
    //    can't be expected to win either.
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
            let scores = bm25_boosted.get_scores(&tokenize_bm25(&qt));
            let top_ids = top_k_ids(&scores, &corpus_ids, 10);
            ndcgs.push(ndcg_at_k(&top_ids, qrel, 10));
            recalls.push(recall_at_k(&top_ids, qrel, 10));
        }
        configs.insert(
            "bm25_boost_only".to_owned(),
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
        "cross_encoder": cross_encoder_repo,
        "bi_encoder": bi_encoder_repo,
        "lme_dir": lme_rust.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "configs": out.configs,
    }))?;
    println!("{json}");
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn cmd_prepare_embeddings(
    data: &std::path::Path,
    bi_encoder_repo: &str,
    onnx_variant: Option<&str>,
    pooling: Pooling,
    force: bool,
    batch_size: usize,
) -> Result<()> {
    let out_dir = data.join(lme_dir_name(bi_encoder_repo));
    let meta_path = out_dir.join("meta.json");
    if meta_path.exists() && !force {
        eprintln!("[prep] up to date: {}", meta_path.display());
        return Ok(());
    }
    std::fs::create_dir_all(&out_dir)?;

    let canonical = data.join("lme_rust");
    let corpus_ids: Vec<String> = std::fs::read_to_string(canonical.join("corpus_ids.txt"))
        .with_context(|| format!("read {}/corpus_ids.txt", canonical.display()))?
        .lines()
        .map(str::to_owned)
        .collect();
    let query_ids: Vec<String> = std::fs::read_to_string(canonical.join("query_ids.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    let sampled_indices: String =
        std::fs::read_to_string(canonical.join("sampled_query_indices.txt"))?;

    let corpus_content: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        data.join("longmemeval_corpus.json"),
    )?)?;
    let query_texts: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        data.join("longmemeval_queries.json"),
    )?)?;

    eprintln!(
        "[prep] loading bi-encoder repo={bi_encoder_repo} pooling={pooling:?} onnx={}…",
        onnx_variant.unwrap_or("(default)")
    );
    let bi = BiEncoder::from_repo_full(bi_encoder_repo, onnx_variant, pooling)
        .map_err(|e| anyhow!("bi-encoder load: {e}"))?;
    let dim = bi.dim();
    eprintln!("[prep] dim={dim}");

    let n_corpus = corpus_ids.len();
    let n_queries = query_ids.len();

    eprintln!("[prep] encoding {n_corpus} corpus chunks…");
    let mut corpus_embs = Array2::<f32>::zeros((n_corpus, dim));
    encode_into(
        &bi,
        &corpus_ids,
        &corpus_content,
        batch_size,
        &mut corpus_embs,
        "corpus",
    )?;
    write_f32_matrix(&out_dir.join("corpus_embeddings.bin"), &corpus_embs)?;
    std::fs::write(out_dir.join("corpus_ids.txt"), corpus_ids.join("\n"))?;

    eprintln!("[prep] encoding {n_queries} queries…");
    let mut query_embs = Array2::<f32>::zeros((n_queries, dim));
    encode_into(
        &bi,
        &query_ids,
        &query_texts,
        batch_size,
        &mut query_embs,
        "queries",
    )?;
    write_f32_matrix(&out_dir.join("query_embeddings.bin"), &query_embs)?;
    std::fs::write(out_dir.join("query_ids.txt"), query_ids.join("\n"))?;

    std::fs::write(
        out_dir.join("sampled_query_indices.txt"),
        sampled_indices,
    )?;

    let meta = serde_json::json!({
        "n_corpus": n_corpus,
        "n_queries": n_queries,
        "dim": dim,
        "bi_encoder": bi_encoder_repo,
        "pooling": format!("{pooling:?}").to_lowercase(),
    });
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
    eprintln!("[prep] wrote {}", out_dir.display());
    println!("{}", serde_json::to_string_pretty(&meta)?);
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn cmd_prepare_embeddings_late(
    data: &std::path::Path,
    bi_encoder_repo: &str,
    onnx_variant: Option<&str>,
    pooling: Pooling,
    force: bool,
    max_len: usize,
) -> Result<()> {
    let out_dir = data.join(lme_late_dir_name(bi_encoder_repo));
    let meta_path = out_dir.join("meta.json");
    if meta_path.exists() && !force {
        eprintln!("[prep-late] up to date: {}", meta_path.display());
        return Ok(());
    }
    std::fs::create_dir_all(&out_dir)?;

    // Reuse the canonical lme_rust IDs/sample so the bench can read
    // this dir interchangeably with the standard prep.
    let canonical = data.join("lme_rust");
    let corpus_ids: Vec<String> = std::fs::read_to_string(canonical.join("corpus_ids.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    let query_ids: Vec<String> = std::fs::read_to_string(canonical.join("query_ids.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    let sampled_indices: String =
        std::fs::read_to_string(canonical.join("sampled_query_indices.txt"))?;

    let corpus_content: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        data.join("longmemeval_corpus.json"),
    )?)?;
    let query_texts: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        data.join("longmemeval_queries.json"),
    )?)?;

    // longmemeval_meta.json: corpus_id → {session_id, turn_idx}.
    // Used to group turns into sessions for late-chunking.
    #[derive(serde::Deserialize)]
    struct MetaEntry {
        session_id: String,
        turn_idx: usize,
    }
    let meta_map: HashMap<String, MetaEntry> = serde_json::from_str(&std::fs::read_to_string(
        data.join("longmemeval_meta.json"),
    )?)?;

    eprintln!(
        "[prep-late] loading bi-encoder repo={bi_encoder_repo} pooling={pooling:?} \
         max_len={max_len} onnx={}…",
        onnx_variant.unwrap_or("(default)")
    );
    let bi = BiEncoder::from_repo_full_with_max_len(
        bi_encoder_repo,
        onnx_variant,
        pooling,
        max_len,
    )
    .map_err(|e| anyhow!("bi-encoder load: {e}"))?;
    let dim = bi.dim();
    eprintln!("[prep-late] dim={dim}");

    // Group corpus_ids by session, preserving canonical order so we
    // can scatter results back to the right rows in corpus_embeddings.bin.
    let mut sessions: std::collections::BTreeMap<String, Vec<(usize, usize)>> =
        std::collections::BTreeMap::new();
    for (row, cid) in corpus_ids.iter().enumerate() {
        let m = meta_map
            .get(cid)
            .ok_or_else(|| anyhow!("corpus_id {cid} missing from meta"))?;
        sessions
            .entry(m.session_id.clone())
            .or_default()
            .push((m.turn_idx, row));
    }
    // Sort each session's turns by turn_idx — chronological order is
    // the right context for late-chunking.
    for v in sessions.values_mut() {
        v.sort_by_key(|t| t.0);
    }
    eprintln!(
        "[prep-late] {} sessions, mean {:.1} turns/session",
        sessions.len(),
        corpus_ids.len() as f64 / sessions.len() as f64
    );

    let n_corpus = corpus_ids.len();
    let mut corpus_embs = Array2::<f32>::zeros((n_corpus, dim));
    let mut n_late = 0_usize;
    let mut n_fallback = 0_usize;
    let mut next_log = 0;
    // Log every ~2% of sessions so progress is visible during the
    // multi-hour prep without spamming stderr.
    let log_every = (sessions.len() / 50).max(1);
    let t0 = Instant::now();

    for (sess_idx, (sess_id, turns)) in sessions.iter().enumerate() {
        let texts: Vec<&str> = turns
            .iter()
            .map(|(_, row)| {
                corpus_content
                    .get(&corpus_ids[*row])
                    .map_or("", String::as_str)
            })
            .collect();
        let rows: Vec<usize> = turns.iter().map(|(_, r)| *r).collect();

        let result = bi.encode_session(&texts);
        let embs = match result {
            Ok(e) => {
                n_late += 1;
                e
            }
            Err(_) => {
                // Session too long — fall back to per-turn encoding.
                // Loses late-chunking benefit for this session but
                // keeps the bench workable on outliers.
                n_fallback += 1;
                bi.encode_batch(&texts)
                    .map_err(|e| anyhow!("encode_batch fallback: {e}"))?
            }
        };
        for (j, row) in rows.iter().enumerate() {
            for (k, v) in embs.row(j).iter().enumerate() {
                corpus_embs[[*row, k]] = *v;
            }
        }
        if sess_idx >= next_log {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = (sess_idx + 1) as f64 / elapsed.max(1e-3);
            let eta = (sessions.len() - sess_idx - 1) as f64 / rate.max(1e-6);
            eprintln!(
                "[prep-late] session {}/{} ({:.1}/s, eta {:.0}s, late={} fallback={}) [{sess_id}]",
                sess_idx + 1,
                sessions.len(),
                rate,
                eta,
                n_late,
                n_fallback
            );
            next_log = sess_idx + log_every;
        }
    }
    eprintln!(
        "[prep-late] done: late={n_late} fallback={n_fallback} ({:.1}% late)",
        100.0 * n_late as f64 / (n_late + n_fallback).max(1) as f64
    );

    write_f32_matrix(&out_dir.join("corpus_embeddings.bin"), &corpus_embs)?;
    std::fs::write(out_dir.join("corpus_ids.txt"), corpus_ids.join("\n"))?;

    // Queries are stand-alone — no late-chunking applies. Encode
    // them with the standard batch path.
    eprintln!("[prep-late] encoding {} queries…", query_ids.len());
    let mut query_embs = Array2::<f32>::zeros((query_ids.len(), dim));
    encode_into(
        &bi,
        &query_ids,
        &query_texts,
        32,
        &mut query_embs,
        "queries",
    )?;
    write_f32_matrix(&out_dir.join("query_embeddings.bin"), &query_embs)?;
    std::fs::write(out_dir.join("query_ids.txt"), query_ids.join("\n"))?;
    std::fs::write(
        out_dir.join("sampled_query_indices.txt"),
        sampled_indices,
    )?;

    let meta = serde_json::json!({
        "n_corpus": n_corpus,
        "n_queries": query_ids.len(),
        "dim": dim,
        "bi_encoder": bi_encoder_repo,
        "pooling": format!("{pooling:?}").to_lowercase(),
        "max_len": max_len,
        "late_chunking": true,
        "n_sessions_late": n_late,
        "n_sessions_fallback": n_fallback,
    });
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)?;
    eprintln!("[prep-late] wrote {}", out_dir.display());
    println!("{}", serde_json::to_string_pretty(&meta)?);
    Ok(())
}

fn encode_into(
    bi: &BiEncoder,
    ids: &[String],
    text_map: &HashMap<String, String>,
    batch_size: usize,
    out: &mut Array2<f32>,
    label: &str,
) -> Result<()> {
    let n = ids.len();
    let mut row = 0;
    let mut next_log = 0;
    let log_every = (n / 20).max(1);
    let t0 = Instant::now();
    while row < n {
        let end = (row + batch_size).min(n);
        let batch_texts: Vec<&str> = (row..end)
            .map(|i| text_map.get(&ids[i]).map_or("", String::as_str))
            .collect();
        let batch = bi
            .encode_batch(&batch_texts)
            .map_err(|e| anyhow!("encode_batch: {e}"))?;
        for (j, src_row) in batch.outer_iter().enumerate() {
            for (k, v) in src_row.iter().enumerate() {
                out[[row + j, k]] = *v;
            }
        }
        row = end;
        if row >= next_log {
            let elapsed = t0.elapsed().as_secs_f64();
            let rate = row as f64 / elapsed.max(1e-3);
            let eta = (n - row) as f64 / rate.max(1e-6);
            eprintln!(
                "[prep] {label}: {row}/{n} ({rate:.0}/s, eta {eta:.0}s)"
            );
            next_log = row + log_every;
        }
    }
    Ok(())
}

fn write_f32_matrix(path: &std::path::Path, m: &Array2<f32>) -> Result<()> {
    let bytes: Vec<u8> = m
        .as_slice()
        .ok_or_else(|| anyhow!("non-contiguous matrix"))?
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(path, bytes)?;
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

fn cmd_xenc(pairs_path: &std::path::Path, cross_encoder_repo: &str) -> Result<()> {
    let input: PairsInput = serde_json::from_str(&read_input(pairs_path)?)?;
    let xenc = CrossEncoder::from_repo(cross_encoder_repo)
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

/// Full-pipeline A/B: ingest the LongMemEval corpus into a real
/// `MemoryStore`, then evaluate NDCG@10 / Recall@10 for the baseline vs.
/// after an offline `dedupe` compaction. Each arm ingests a fresh
/// ephemeral store so RIF starts cold and the arms are independent. The
/// compaction arm remaps qrels through the alias table so a gold turn
/// absorbed into a canonical still counts when the canonical is
/// retrieved.
#[allow(clippy::too_many_lines)]
fn cmd_rif_eval(
    data: &std::path::Path,
    cross_encoder_repo: &str,
    bi_encoder_repo: &str,
    sample_limit: usize,
    threshold: f32,
    n_clusters: u32,
) -> Result<()> {
    let lme_rust = data.join(lme_dir_name(bi_encoder_repo));
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(lme_rust.join("meta.json"))
            .with_context(|| format!("read {}/meta.json", lme_rust.display()))?,
    )?;
    let n_corpus = meta["n_corpus"].as_u64().unwrap() as usize;
    let dim = meta["dim"].as_u64().unwrap() as usize;

    let corpus_ids: Vec<String> = std::fs::read_to_string(lme_rust.join("corpus_ids.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    let query_ids: Vec<String> = std::fs::read_to_string(lme_rust.join("query_ids.txt"))?
        .lines()
        .map(str::to_owned)
        .collect();
    eprintln!("[rif-eval] reading corpus embeddings ({n_corpus}×{dim})…");
    let corpus_embs = read_f32_matrix(&lme_rust.join("corpus_embeddings.bin"), n_corpus, dim)?;

    let mut sampled: Vec<usize> =
        std::fs::read_to_string(lme_rust.join("sampled_query_indices.txt"))?
            .lines()
            .map(str::parse)
            .collect::<Result<_, _>>()?;
    if sample_limit > 0 && sampled.len() > sample_limit {
        sampled.truncate(sample_limit);
    }

    let qrels: HashMap<String, HashMap<String, f64>> = serde_json::from_str(
        &std::fs::read_to_string(data.join("longmemeval_qrels.json"))?,
    )?;
    let corpus_content: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        data.join("longmemeval_corpus.json"),
    )?)?;
    let query_texts: HashMap<String, String> = serde_json::from_str(&std::fs::read_to_string(
        data.join("longmemeval_queries.json"),
    )?)?;

    let eval_queries: Vec<(String, String)> = sampled
        .iter()
        .filter_map(|&i| {
            let qid = query_ids.get(i)?.clone();
            let text = query_texts.get(&qid)?.clone();
            Some((qid, text))
        })
        .collect();
    eprintln!("[rif-eval] {} eval queries", eval_queries.len());

    eprintln!("[rif-eval] loading encoders…");
    let bi = Arc::new(BiEncoder::from_repo(bi_encoder_repo)?);
    let cross = Arc::new(CrossEncoder::from_repo(cross_encoder_repo)?);

    let store_dir = data.join(".rif_eval_store");

    // Build a fresh in-memory store over the whole corpus. Ephemeral:
    // nothing is persisted, so each arm starts with cold RIF and the
    // arms stay independent. The DuckDB at `store_dir` is only used by
    // `dedupe` for the (tiny) alias table.
    let ingest = || -> Result<MemoryStore> {
        let _ = std::fs::remove_dir_all(&store_dir);
        let cfg = StoreConfig {
            dim,
            dedup_threshold: threshold,
            rif: RifConfig {
                n_clusters,
                ..RifConfig::default()
            },
            ..StoreConfig::default()
        };
        let store = MemoryStore::open(&store_dir, Some(bi.clone()), Some(cross.clone()), cfg)?;
        let items = corpus_ids.iter().enumerate().map(|(row, cid)| {
            let content = corpus_content.get(cid).cloned().unwrap_or_default();
            (cid.clone(), content, corpus_embs.row(row).to_owned())
        });
        let t = Instant::now();
        store.ingest_ephemeral(items)?;
        eprintln!("[rif-eval]   ingested {n_corpus} in {:.1}s", t.elapsed().as_secs_f64());
        Ok(store)
    };

    // Baseline (no compaction).
    eprintln!("[rif-eval] arm: baseline…");
    let baseline = eval_arm(&ingest()?, &eval_queries, &qrels)?;

    // Compaction arm: ingest, compact, remap qrels through the aliases so
    // an absorbed gold turn still scores when its canonical is retrieved.
    eprintln!("[rif-eval] arm: +dedupe (τ={threshold:.2})…");
    let store = ingest()?;
    let report = store.dedupe(threshold, false)?;
    let alias = store.with_db(lethe_core::db::MemoryDb::alias_map)?;
    eprintln!(
        "[rif-eval] dedupe: scanned={} groups={} absorbed={}",
        report.scanned, report.groups, report.absorbed
    );
    let qrels_after: HashMap<String, HashMap<String, f64>> = qrels
        .iter()
        .map(|(qid, rels)| {
            let mut m: HashMap<String, f64> = HashMap::new();
            for (gid, &rel) in rels {
                let key = alias.get(gid).cloned().unwrap_or_else(|| gid.clone());
                let e = m.entry(key).or_insert(0.0);
                *e = e.max(rel);
            }
            (qid.clone(), m)
        })
        .collect();
    let dedupe = eval_arm(&store, &eval_queries, &qrels_after)?;
    drop(store);

    let _ = std::fs::remove_dir_all(&store_dir);

    let arm = |m: &ArmMetrics| {
        serde_json::json!({"ndcg": m.ndcg, "recall": m.recall, "n_eval": m.n_eval, "time_s": m.time_s})
    };
    let payload = serde_json::json!({
        "impl": "rust",
        "bi_encoder": bi_encoder_repo,
        "cross_encoder": cross_encoder_repo,
        "n_corpus": n_corpus,
        "n_corpus_after": n_corpus - report.absorbed,
        "threshold": threshold,
        "n_clusters": n_clusters,
        "dedupe": {"scanned": report.scanned, "groups": report.groups, "absorbed": report.absorbed},
        "arms": {
            "baseline": arm(&baseline),
            "dedupe": arm(&dedupe),
        }
    });
    println!("{}", serde_json::to_string_pretty(&payload)?);

    eprintln!("\narm        ndcg@10  recall@10   n   time_s");
    for (name, m) in [("baseline", &baseline), ("+dedupe", &dedupe)] {
        eprintln!(
            "{name:<10} {:.4}   {:.4}    {}  {:.1}",
            m.ndcg, m.recall, m.n_eval, m.time_s
        );
    }
    Ok(())
}

struct ArmMetrics {
    ndcg: f64,
    recall: f64,
    n_eval: usize,
    time_s: f64,
}

/// Run every eval query through `store.retrieve` and average NDCG@10 /
/// Recall@10 against `qrels`.
fn eval_arm(
    store: &MemoryStore,
    queries: &[(String, String)],
    qrels: &HashMap<String, HashMap<String, f64>>,
) -> Result<ArmMetrics> {
    let t = Instant::now();
    let mut ndcgs = Vec::new();
    let mut recalls = Vec::new();
    for (qid, text) in queries {
        let Some(rel) = qrels.get(qid) else { continue };
        if rel.is_empty() {
            continue;
        }
        let ids: Vec<String> = store.retrieve(text, 10)?.into_iter().map(|h| h.id).collect();
        ndcgs.push(ndcg_at_k(&ids, rel, 10));
        recalls.push(recall_at_k(&ids, rel, 10));
    }
    Ok(ArmMetrics {
        ndcg: mean(&ndcgs),
        recall: mean(&recalls),
        n_eval: ndcgs.len(),
        time_s: t.elapsed().as_secs_f64(),
    })
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

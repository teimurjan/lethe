"""Compare bi-encoder candidates on load time, throughput, and pipeline NDCG.

Uses the same BM25 + dense + cross-encoder rerank pipeline as
``run_benchmark.py``, but evaluated on a **candidate-pool subset** of the
corpus (relevant docs + BM25 top-K + dense top-K per query, deduped) so
the expensive full-corpus re-embed for a new bi-encoder drops from 30+
minutes to under 2 minutes. The candidate pool is model-agnostic
(derived once using the baseline) and held constant across models — so
NDCG numbers reported here are **relative**, apples-to-apples between
models, not directly comparable to run_benchmark.py's full-corpus NDCG.

Currently evaluates:
    - sentence-transformers/all-MiniLM-L6-v2 (fp32, current default)
    - BAAI/bge-small-en-v1.5              (int8, qdrant's `-onnx-q` variant)

Run: ``uv run python benchmarks/run_int8.py``
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from benchmarks._lib.metrics import ndcg_at_k, recall_at_k

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "benchmarks" / "results" / "BENCHMARKS_INT8.md"
EMB_CACHE = ROOT / "benchmarks" / "results" / "int8_embeds"

MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6-v2 (fp32)"),
    ("BAAI/bge-small-en-v1.5",                 "bge-small-en-v1.5 (int8)"),
]

# Candidate-pool eval: number of sampled queries, BM25/dense depth per query.
SAMPLE_QUERIES = 100
CAND_DEPTH = 60
RERANK_POOL = 30  # top-N per retriever fed into xenc rerank


def cold_start_ms(model_name: str, trials: int = 3) -> list[float]:
    """Cold-start load time (seconds) in a fresh Python process per trial."""
    code = (
        "import os, time; os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1';"
        "t0=time.monotonic();"
        "from fastembed import TextEmbedding;"
        f"e=TextEmbedding({model_name!r});"
        "list(e.embed(['warm']));"
        "print(time.monotonic()-t0)"
    )
    out = []
    for _ in range(trials):
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT),
        )
        if r.returncode != 0:
            raise RuntimeError(f"cold-start probe failed for {model_name}: {r.stderr}")
        last = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1]
        out.append(float(last))
    return out


def throughput_items_per_sec(model_name: str, n: int = 500) -> float:
    from fastembed import TextEmbedding
    e = TextEmbedding(model_name)
    texts = [f"benchmark sentence {i} containing varied content for throughput measurement." for i in range(n)]
    list(e.embed(texts[:5]))
    t0 = time.monotonic()
    list(e.embed(texts))
    t1 = time.monotonic()
    return n / (t1 - t0)


def build_candidate_pool(
    corpus_ids: list[str],
    corpus_content: dict,
    corpus_embs_full: np.ndarray,
    query_ids_sample: list[str],
    query_texts: dict,
    query_embs_sample: np.ndarray,
    qrels: dict,
    cand_depth: int = CAND_DEPTH,
) -> tuple[list[str], dict[str, int]]:
    """Build a reduced pool: qrels + BM25 top-K + dense top-K per sampled
    query. Returns (pool_ids, id→index in pool)."""
    # Full-corpus BM25 + FAISS so candidate selection mirrors run_benchmark.py.
    print(f"    building BM25 index on {len(corpus_ids)} docs...", flush=True)
    tokenized = [corpus_content.get(cid, "").lower().split() for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)

    print(f"    building FAISS index ({corpus_embs_full.shape})...", flush=True)
    faiss_idx = faiss.IndexFlatIP(corpus_embs_full.shape[1])
    faiss_idx.add(corpus_embs_full)

    pool: set[str] = set()
    for qi, qe in zip(query_ids_sample, query_embs_sample):
        # relevant docs
        for cid in (qrels.get(qi, {}) or {}):
            pool.add(cid)
        # BM25
        scores = bm25.get_scores(query_texts.get(qi, "").lower().split())
        for j in np.argsort(scores)[::-1][:cand_depth]:
            pool.add(corpus_ids[j])
        # Dense
        _, I = faiss_idx.search(qe.reshape(1, -1).astype(np.float32), cand_depth)
        for j in I[0]:
            if j >= 0:
                pool.add(corpus_ids[j])

    pool_ids = sorted(pool)  # stable order
    id_to_idx = {cid: i for i, cid in enumerate(pool_ids)}
    return pool_ids, id_to_idx


def embed_texts_batched(model_name: str, texts: list[str], label: str) -> np.ndarray:
    """Embed a list of texts with progress prints per batch."""
    from fastembed import TextEmbedding
    e = TextEmbedding(model_name)
    # warm
    list(e.embed(["warm"]))

    BATCH = 1000
    out: list[np.ndarray] = []
    t0 = time.monotonic()
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        vecs = list(e.embed(chunk, batch_size=256))
        out.extend(vecs)
        done = min(i + BATCH, len(texts))
        rate = done / (time.monotonic() - t0)
        print(f"      {label}: {done}/{len(texts)}  ({rate:.0f} items/s)", flush=True)
    return np.asarray(out, dtype=np.float32)


def get_pool_embeddings(
    model_name: str,
    pool_ids: list[str],
    corpus_content: dict,
    query_ids_sample: list[str],
    query_texts: dict,
    baseline_corpus_embs_full: np.ndarray,
    baseline_query_embs_full: np.ndarray,
    corpus_id_to_full_idx: dict[str, int],
    query_id_to_full_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool + query embeddings for the given model. Cached per model."""
    safe = model_name.replace("/", "__")
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    cache = EMB_CACHE / f"{safe}.pool.npz"
    if cache.exists():
        print(f"    (cache hit: {cache.name})", flush=True)
        data = np.load(str(cache))
        return data["pool"], data["queries"]

    if model_name == "sentence-transformers/all-MiniLM-L6-v2":
        # Reuse prepared.npz vectors — sentence-transformers MiniLM and
        # fastembed-ONNX MiniLM produce near-identical vectors (same weights).
        print("    (using precomputed longmemeval_prepared.npz for MiniLM)", flush=True)
        pool_embs = np.stack([baseline_corpus_embs_full[corpus_id_to_full_idx[cid]] for cid in pool_ids])
        q_embs = np.stack([baseline_query_embs_full[query_id_to_full_idx[qi]] for qi in query_ids_sample])
    else:
        pool_texts = [corpus_content.get(cid, "") for cid in pool_ids]
        print(f"    embedding pool ({len(pool_texts)} docs)...", flush=True)
        pool_embs = embed_texts_batched(model_name, pool_texts, "pool")
        query_list = [query_texts.get(qi, "") for qi in query_ids_sample]
        print(f"    embedding queries ({len(query_list)})...", flush=True)
        q_embs = embed_texts_batched(model_name, query_list, "queries")

    np.savez(str(cache), pool=pool_embs, queries=q_embs)
    return pool_embs, q_embs


def evaluate_on_pool(
    pool_ids: list[str], pool_embs: np.ndarray, corpus_content: dict,
    query_ids_sample: list[str], query_texts: dict, q_embs: np.ndarray, qrels: dict,
    xenc, rerank_pool: int = RERANK_POOL,
) -> tuple[float, float]:
    dim = pool_embs.shape[1]
    faiss_idx = faiss.IndexFlatIP(dim)
    faiss_idx.add(pool_embs)

    pool_tokens = [corpus_content.get(cid, "").lower().split() for cid in pool_ids]
    pool_bm25 = BM25Okapi(pool_tokens)

    ndcgs, recalls = [], []
    for qi, qe in zip(query_ids_sample, q_embs):
        qr = qrels.get(qi, {})
        if not qr:
            continue
        qt = query_texts.get(qi, "")

        _, I = faiss_idx.search(qe.reshape(1, -1).astype(np.float32), rerank_pool)
        vec_ids = [pool_ids[j] for j in I[0] if j >= 0]
        scores = pool_bm25.get_scores(qt.lower().split())
        bm25_ids = [pool_ids[j] for j in np.argsort(scores)[::-1][:rerank_pool]]
        all_ids = list(dict.fromkeys(vec_ids + bm25_ids))

        pairs = [(qt, corpus_content.get(c, "")) for c in all_ids]
        xs = xenc.predict(pairs)
        ranked = sorted(zip(all_ids, xs), key=lambda x: x[1], reverse=True)
        top10 = [c for c, _ in ranked[:10]]

        ndcgs.append(ndcg_at_k(top10, qr, 10))
        recalls.append(recall_at_k(top10, qr, 10))

    return float(np.mean(ndcgs)), float(np.mean(recalls))


def main() -> None:
    print("=" * 64)
    print("bi-encoder comparison: load time, throughput, pool-NDCG")
    print("=" * 64)

    print("\nLoading LongMemEval data...", flush=True)
    with open(DATA / "longmemeval_qrels.json") as f:
        qrels = json.load(f)
    with open(DATA / "longmemeval_corpus.json") as f:
        corpus_content = json.load(f)
    with open(DATA / "longmemeval_queries.json") as f:
        query_texts = json.load(f)
    prep = np.load(str(DATA / "longmemeval_prepared.npz"), allow_pickle=True)
    corpus_ids = list(prep["corpus_ids"])
    query_ids = list(prep["query_ids"])
    baseline_corpus_embs = prep["corpus_embeddings"].astype(np.float32)
    baseline_query_embs = prep["query_embeddings"].astype(np.float32)
    print(f"  corpus={len(corpus_ids)}  queries={len(query_ids)}  eval sample={SAMPLE_QUERIES}")

    corpus_id_to_full_idx = {cid: i for i, cid in enumerate(corpus_ids)}
    query_id_to_full_idx = {qi: i for i, qi in enumerate(query_ids)}

    # Sample queries once, used across models
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(query_ids), size=SAMPLE_QUERIES, replace=False)
    query_ids_sample = [query_ids[i] for i in sample_idx]
    query_embs_sample = baseline_query_embs[sample_idx]

    # Candidate pool derived once (using baseline) — held constant across models
    print("\nBuilding shared candidate pool (qrels + BM25 top-K + dense top-K)...", flush=True)
    t0 = time.monotonic()
    pool_ids, _ = build_candidate_pool(
        corpus_ids, corpus_content, baseline_corpus_embs,
        query_ids_sample, query_texts, query_embs_sample, qrels,
        cand_depth=CAND_DEPTH,
    )
    print(f"  pool size = {len(pool_ids)} docs  [{time.monotonic()-t0:.1f}s]", flush=True)

    print("\nLoading cross-encoder (shared across runs)...", flush=True)
    from lethe.encoders import OnnxCrossEncoder
    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")

    results = []
    for model_name, label in MODELS:
        print(f"\n=== {label} ===", flush=True)

        print("  cold-start (3 trials)...", flush=True)
        cs = cold_start_ms(model_name, trials=3)
        cs_med = statistics.median(cs)
        print(f"    median={cs_med:.2f}s  min={min(cs):.2f}s  max={max(cs):.2f}s", flush=True)

        print("  warm throughput (500 items)...", flush=True)
        thr = throughput_items_per_sec(model_name, n=500)
        print(f"    {thr:.0f} items/s", flush=True)

        print("  pool-NDCG eval...", flush=True)
        pool_embs, q_embs = get_pool_embeddings(
            model_name, pool_ids, corpus_content,
            query_ids_sample, query_texts,
            baseline_corpus_embs, baseline_query_embs,
            corpus_id_to_full_idx, query_id_to_full_idx,
        )

        t0 = time.monotonic()
        ndcg, recall = evaluate_on_pool(
            pool_ids, pool_embs, corpus_content,
            query_ids_sample, query_texts, q_embs, qrels, xenc,
        )
        t_eval = time.monotonic() - t0
        print(f"    NDCG@10={ndcg:.4f}  Recall@10={recall:.4f}  [{t_eval:.1f}s]", flush=True)

        results.append({
            "model": model_name, "label": label,
            "cold_start_med": cs_med, "cold_start_min": min(cs), "cold_start_max": max(cs),
            "throughput": thr, "ndcg": ndcg, "recall": recall, "eval_s": t_eval,
        })

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    base_cs = results[0]["cold_start_med"]
    base_thr = results[0]["throughput"]
    base_ndcg = results[0]["ndcg"]

    md = [
        "# bi-encoder benchmark (int8 comparison)\n",
        f"Dataset: LongMemEval S ({len(corpus_ids):,} turns, {len(query_ids)} queries; "
        f"**{SAMPLE_QUERIES}**-query eval sample, seed=0)",
        f"Pipeline: BM25 + dense (depth {CAND_DEPTH} each) + cross-encoder rerank of top-{RERANK_POOL} each",
        f"Candidate pool (shared across models): {len(pool_ids):,} docs = qrels ∪ BM25 top-{CAND_DEPTH} ∪ dense top-{CAND_DEPTH} per sampled query",
        f"Cross-encoder: Xenova/ms-marco-MiniLM-L-6-v2 (held constant)",
        f"Date: {time.strftime('%Y-%m-%d')}",
        f"Note: NDCG is on the reduced pool, not the full corpus — **relative** comparison between models.",
        "",
        "| model | cold-start (s, med / min) | throughput (items/s) | NDCG@10 | Recall@10 | Δ NDCG | load speedup | throughput speedup |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        d_ndcg = (r["ndcg"] - base_ndcg) * 100
        load_speed = base_cs / r["cold_start_med"]
        thr_speed = r["throughput"] / base_thr
        md.append(
            f"| {r['label']} "
            f"| {r['cold_start_med']:.2f} / {r['cold_start_min']:.2f} "
            f"| {r['throughput']:.0f} "
            f"| {r['ndcg']:.4f} | {r['recall']:.4f} "
            f"| {d_ndcg:+.2f}pp | {load_speed:.2f}× | {thr_speed:.2f}× |"
        )
    RESULTS.write_text("\n".join(md) + "\n")

    print("\n" + "=" * 64)
    print("\n".join(md))
    print("\nWrote:", RESULTS)


if __name__ == "__main__":
    main()

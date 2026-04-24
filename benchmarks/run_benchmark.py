"""Benchmark lethe against baseline approaches on LongMemEval.

Produces clean output and saves results to BENCHMARKS.md.
Usage: uv run python benchmarks/run_benchmark.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from benchmarks._lib.metrics import ndcg_at_k, recall_at_k
from lethe.encoders import OnnxCrossEncoder  # production rerank runtime
from lethe.vectors import tokenize_bm25  # production BM25 tokenizer

DATA = Path("data")
BENCHMARKS_MD = Path("BENCHMARKS.md")


def load_data() -> tuple:
    data = np.load(str(DATA / "longmemeval_prepared.npz"), allow_pickle=True)
    corpus_ids = list(data["corpus_ids"])
    corpus_embs = data["corpus_embeddings"].astype(np.float32)
    query_ids = list(data["query_ids"])
    query_embs = data["query_embeddings"].astype(np.float32)
    with open(DATA / "longmemeval_qrels.json") as f:
        qrels = json.load(f)
    with open(DATA / "longmemeval_corpus.json") as f:
        corpus_content = json.load(f)
    with open(DATA / "longmemeval_queries.json") as f:
        query_texts = json.load(f)
    return corpus_ids, corpus_embs, query_ids, query_embs, qrels, corpus_content, query_texts


def evaluate(
    name: str,
    get_top10: callable,
    query_ids: list,
    query_embs: np.ndarray,
    query_texts: dict,
    qrels: dict,
    sample_size: int = 200,
) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(query_ids), size=sample_size, replace=False)
    ndcgs, recalls = [], []
    for i in idx:
        qi = query_ids[i]
        qr = qrels.get(qi, {})
        if not qr:
            continue
        top10 = get_top10(qi, query_embs[i], query_texts.get(qi, ""))
        ndcgs.append(ndcg_at_k(top10, qr, 10))
        recalls.append(recall_at_k(top10, qr, 10))
    return float(np.mean(ndcgs)), float(np.mean(recalls))


def main() -> None:
    print("=" * 60)
    print("lethe benchmark on LongMemEval")
    print("=" * 60)
    print()

    print("Loading data...", flush=True)
    corpus_ids, corpus_embs, query_ids, query_embs, qrels, corpus_content, query_texts = load_data()
    print(f"Corpus: {len(corpus_ids)} entries, Queries: {len(query_ids)}")

    # Build indexes
    print("Building FAISS index...", flush=True)
    index = faiss.IndexFlatIP(384)
    index.add(corpus_embs)

    print("Building BM25 index...", flush=True)
    tokenized = [tokenize_bm25(corpus_content.get(cid, "")) for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)

    print("Loading cross-encoder (ONNX)...", flush=True)
    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    _ = xenc.predict([("warm", "warm")])  # JIT-warm the session
    print()

    results = []

    # 1. Vector only (bi-encoder top-10)
    def vector_only(qi, qe, qt):
        D, I = index.search(qe.reshape(1, -1), 10)
        return [corpus_ids[i] for i in I[0] if i >= 0]

    t0 = time.time()
    ndcg, recall = evaluate("Vector only", vector_only, query_ids, query_embs, query_texts, qrels)
    t1 = time.time()
    results.append(("Vector only (MiniLM top-10)", ndcg, recall, t1 - t0))
    print(f"  Vector only (MiniLM top-10):         NDCG={ndcg:.4f}  Recall={recall:.4f}  [{t1-t0:.1f}s]", flush=True)

    # 2. BM25 only top-10
    def bm25_only(qi, qe, qt):
        scores = bm25.get_scores(tokenize_bm25(qt))
        top = np.argsort(scores)[::-1][:10]
        return [corpus_ids[i] for i in top]

    t0 = time.time()
    ndcg, recall = evaluate("BM25 only", bm25_only, query_ids, query_embs, query_texts, qrels)
    t1 = time.time()
    results.append(("BM25 only (top-10)", ndcg, recall, t1 - t0))
    print(f"  BM25 only (top-10):                  NDCG={ndcg:.4f}  Recall={recall:.4f}  [{t1-t0:.1f}s]", flush=True)

    # 3. Hybrid BM25+vector RRF (memsearch style)
    def hybrid_rrf(qi, qe, qt):
        D, I = index.search(qe.reshape(1, -1), 30)
        vec_ids = [corpus_ids[i] for i in I[0] if i >= 0]
        scores = bm25.get_scores(tokenize_bm25(qt))
        bm25_ids = [corpus_ids[i] for i in np.argsort(scores)[::-1][:30]]
        rrf_scores: dict[str, float] = {}
        for rank, cid in enumerate(vec_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (60 + rank + 1)
        for rank, cid in enumerate(bm25_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (60 + rank + 1)
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:10]]

    t0 = time.time()
    ndcg, recall = evaluate("Hybrid RRF", hybrid_rrf, query_ids, query_embs, query_texts, qrels)
    t1 = time.time()
    results.append(("Hybrid BM25+vector RRF (memsearch)", ndcg, recall, t1 - t0))
    print(f"  Hybrid BM25+vector RRF (memsearch):  NDCG={ndcg:.4f}  Recall={recall:.4f}  [{t1-t0:.1f}s]", flush=True)

    # 4. Vector + cross-encoder rerank (our static baseline)
    def vector_xenc(qi, qe, qt):
        D, I = index.search(qe.reshape(1, -1), 30)
        cand_ids = [corpus_ids[i] for i in I[0] if i >= 0]
        pairs = [(qt, corpus_content.get(c, "")) for c in cand_ids]
        xs = xenc.predict(pairs)
        ranked = sorted(zip(cand_ids, xs), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:10]]

    t0 = time.time()
    ndcg, recall = evaluate("Vector+xenc", vector_xenc, query_ids, query_embs, query_texts, qrels)
    t1 = time.time()
    results.append(("Vector + cross-encoder rerank", ndcg, recall, t1 - t0))
    print(f"  Vector + cross-encoder rerank:       NDCG={ndcg:.4f}  Recall={recall:.4f}  [{t1-t0:.1f}s]", flush=True)

    # 5. lethe full stack (BM25 + vector + xenc rerank)
    def lethe_full(qi, qe, qt):
        D, I = index.search(qe.reshape(1, -1), 30)
        vec_ids = [corpus_ids[i] for i in I[0] if i >= 0]
        scores = bm25.get_scores(tokenize_bm25(qt))
        bm25_ids = [corpus_ids[i] for i in np.argsort(scores)[::-1][:30]]
        all_ids = list(dict.fromkeys(vec_ids + bm25_ids))
        pairs = [(qt, corpus_content.get(c, "")) for c in all_ids]
        xs = xenc.predict(pairs)
        ranked = sorted(zip(all_ids, xs), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:10]]

    t0 = time.time()
    ndcg, recall = evaluate("lethe", lethe_full, query_ids, query_embs, query_texts, qrels)
    t1 = time.time()
    results.append(("lethe (BM25 + vector + xenc)", ndcg, recall, t1 - t0))
    print(f"  lethe (BM25 + vector + xenc):        NDCG={ndcg:.4f}  Recall={recall:.4f}  [{t1-t0:.1f}s]", flush=True)

    # Write BENCHMARKS.md
    print(f"\nWriting {BENCHMARKS_MD}...", flush=True)
    baseline_ndcg = results[0][1]
    with open(BENCHMARKS_MD, "w") as f:
        f.write("# Benchmarks\n\n")
        f.write(f"Dataset: LongMemEval S variant ({len(corpus_ids)} conversation turns, {len(query_ids)} questions)\n")
        f.write(f"Eval: 200-query random sample, seed=0\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("| System | NDCG@10 | Recall@10 | vs baseline | Time |\n")
        f.write("|--------|---------|-----------|-------------|------|\n")
        for name, ndcg, recall, elapsed in results:
            delta = (ndcg - baseline_ndcg) / baseline_ndcg * 100
            delta_str = f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%"
            if name == results[0][0]:
                delta_str = "baseline"
            f.write(f"| {name} | {ndcg:.4f} | {recall:.4f} | {delta_str} | {elapsed:.1f}s |\n")
        f.write("\n## How to reproduce\n\n")
        f.write("```bash\n")
        f.write("uv run python experiments/prep_longmemeval.py\n")
        f.write("uv run python benchmarks/run_benchmark.py\n")
        f.write("```\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

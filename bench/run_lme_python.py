"""Run the same 5 retrieval configurations as `benchmarks/run_benchmark.py`
but use the prepared sample subset and emit JSON to stdout instead of
overwriting `BENCHMARKS.md`. Used by the Rust↔Python comparison harness.

The configs and pipeline are byte-for-byte the same as
`run_benchmark.py` so the resulting NDCG/Recall numbers can be directly
compared against the production headline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from benchmarks._lib.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from lethe.encoders import OnnxCrossEncoder  # noqa: E402
from lethe.vectors import tokenize_bm25  # noqa: E402

DATA = REPO / "data"
LME_RUST = DATA / "lme_rust"


def load_inputs() -> tuple:
    if not LME_RUST.exists():
        sys.stderr.write(
            "error: data/lme_rust/ missing. "
            "Run `uv run python bench/prepare_lme_data.py` first.\n"
        )
        sys.exit(2)
    with (DATA / "longmemeval_qrels.json").open() as f:
        qrels = json.load(f)
    with (DATA / "longmemeval_corpus.json").open() as f:
        corpus_content = json.load(f)
    with (DATA / "longmemeval_queries.json").open() as f:
        query_texts = json.load(f)
    prepared = np.load(str(DATA / "longmemeval_prepared.npz"), allow_pickle=True)
    corpus_ids = list(prepared["corpus_ids"])
    corpus_embs = prepared["corpus_embeddings"].astype(np.float32)
    query_ids = list(prepared["query_ids"])
    query_embs = prepared["query_embeddings"].astype(np.float32)
    sampled = [int(x) for x in (LME_RUST / "sampled_query_indices.txt").read_text().split()]
    return (
        qrels,
        corpus_content,
        query_texts,
        corpus_ids,
        corpus_embs,
        query_ids,
        query_embs,
        sampled,
    )


def evaluate(name, get_top10, sampled, query_ids, query_embs, query_texts, qrels):
    ndcgs, recalls = [], []
    for i in sampled:
        qi = query_ids[i]
        qr = qrels.get(qi, {})
        if not qr:
            continue
        top10 = get_top10(qi, query_embs[i], query_texts.get(qi, ""))
        ndcgs.append(ndcg_at_k(top10, qr, 10))
        recalls.append(recall_at_k(top10, qr, 10))
    return float(np.mean(ndcgs)), float(np.mean(recalls)), len(ndcgs)


def main() -> int:
    (
        qrels,
        corpus_content,
        query_texts,
        corpus_ids,
        corpus_embs,
        query_ids,
        query_embs,
        sampled,
    ) = load_inputs()

    sys.stderr.write("[python] building FAISS…\n")
    index = faiss.IndexFlatIP(corpus_embs.shape[1])
    index.add(corpus_embs)
    sys.stderr.write("[python] tokenizing corpus…\n")
    tokenized = [tokenize_bm25(corpus_content.get(cid, "")) for cid in corpus_ids]
    sys.stderr.write("[python] building BM25…\n")
    bm25 = BM25Okapi(tokenized)
    sys.stderr.write("[python] loading cross-encoder (ONNX)…\n")
    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    _ = xenc.predict([("warm", "warm")])

    results = {}

    def vector_only(_qi, qe, _qt):
        _D, I = index.search(qe.reshape(1, -1), 10)
        return [corpus_ids[i] for i in I[0] if i >= 0]

    t0 = time.time()
    n, r, nev = evaluate("Vector only", vector_only, sampled, query_ids, query_embs, query_texts, qrels)
    results["vector_only"] = {"ndcg": n, "recall": r, "n_eval": nev, "time_s": time.time() - t0}

    def bm25_only(_qi, _qe, qt):
        scores = bm25.get_scores(tokenize_bm25(qt))
        top = np.argsort(scores)[::-1][:10]
        return [corpus_ids[i] for i in top]

    t0 = time.time()
    n, r, nev = evaluate("BM25 only", bm25_only, sampled, query_ids, query_embs, query_texts, qrels)
    results["bm25_only"] = {"ndcg": n, "recall": r, "n_eval": nev, "time_s": time.time() - t0}

    def hybrid_rrf(_qi, qe, qt):
        _D, I = index.search(qe.reshape(1, -1), 30)
        vec_ids = [corpus_ids[i] for i in I[0] if i >= 0]
        scores = bm25.get_scores(tokenize_bm25(qt))
        bm_ids = [corpus_ids[i] for i in np.argsort(scores)[::-1][:30]]
        rrf: dict[str, float] = {}
        for rank, cid in enumerate(vec_ids):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        for rank, cid in enumerate(bm_ids):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:10]]

    t0 = time.time()
    n, r, nev = evaluate("Hybrid RRF", hybrid_rrf, sampled, query_ids, query_embs, query_texts, qrels)
    results["hybrid_rrf"] = {"ndcg": n, "recall": r, "n_eval": nev, "time_s": time.time() - t0}

    def vector_xenc(_qi, qe, qt):
        _D, I = index.search(qe.reshape(1, -1), 30)
        cand_ids = [corpus_ids[i] for i in I[0] if i >= 0]
        pairs = [(qt, corpus_content.get(c, "")) for c in cand_ids]
        xs = xenc.predict(pairs)
        ranked = sorted(zip(cand_ids, xs, strict=False), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:10]]

    t0 = time.time()
    n, r, nev = evaluate("Vector+xenc", vector_xenc, sampled, query_ids, query_embs, query_texts, qrels)
    results["vector_xenc"] = {"ndcg": n, "recall": r, "n_eval": nev, "time_s": time.time() - t0}

    def lethe_full(_qi, qe, qt):
        _D, I = index.search(qe.reshape(1, -1), 30)
        vec_ids = [corpus_ids[i] for i in I[0] if i >= 0]
        scores = bm25.get_scores(tokenize_bm25(qt))
        bm_ids = [corpus_ids[i] for i in np.argsort(scores)[::-1][:30]]
        all_ids = list(dict.fromkeys(vec_ids + bm_ids))
        pairs = [(qt, corpus_content.get(c, "")) for c in all_ids]
        xs = xenc.predict(pairs)
        ranked = sorted(zip(all_ids, xs, strict=False), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:10]]

    t0 = time.time()
    n, r, nev = evaluate("lethe full", lethe_full, sampled, query_ids, query_embs, query_texts, qrels)
    results["lethe_full"] = {"ndcg": n, "recall": r, "n_eval": nev, "time_s": time.time() - t0}

    print(json.dumps({"impl": "python", "configs": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

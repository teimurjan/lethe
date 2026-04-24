"""RIF benchmark: does retrieval-induced forgetting improve candidate pool quality?

Tests whether suppressing chronic false positives (entries that compete but
lose repeatedly) improves NDCG by letting previously-excluded relevant entries
into the cross-encoder pool.

Protocol (sequential — RIF is cumulative):
  Phase 1: Simulate 2000-query usage pattern, accumulating RIF suppression
  Phase 2: Evaluate on 200-query sample, comparing RIF vs static baseline

Key diagnostic: candidate pool recall@30 before vs after RIF.
"""
from __future__ import annotations

import json
import math
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

from benchmarks._lib.metrics import ndcg_at_k
from lethe.encoders import OnnxCrossEncoder  # production rerank runtime
from lethe.rif import RIFConfig, apply_suppression_penalty, competition_strength, update_suppression
from lethe.vectors import tokenize_bm25  # production BM25 tokenizer

DATA = Path("data")
RESULTS = Path("benchmarks/results/BENCHMARKS_RIF.md")


def search_hybrid_scored(
    query_emb: np.ndarray,
    query_text: str,
    k: int,
    index: faiss.IndexFlatIP,
    bm25: BM25Okapi,
    corpus_ids: list[str],
) -> list[tuple[str, float]]:
    """Hybrid search returning (id, RRF_score) pairs."""
    D, I = index.search(query_emb.reshape(1, -1), k)
    vec_results = [(corpus_ids[i], float(D[0][rank])) for rank, i in enumerate(I[0]) if i >= 0]

    tokens = tokenize_bm25(query_text)
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    bm25_results = [(corpus_ids[i], float(scores[i])) for i in top_idx]

    rrf: dict[str, float] = {}
    for rank, (eid, _) in enumerate(vec_results):
        rrf[eid] = rrf.get(eid, 0.0) + 1.0 / (rank + 60)
    for rank, (eid, _) in enumerate(bm25_results):
        rrf[eid] = rrf.get(eid, 0.0) + 1.0 / (rank + 60)

    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def main() -> None:
    print("=" * 60)
    print("RIF benchmark: retrieval-induced forgetting")
    print("=" * 60)
    print()

    # Load data
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

    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    _ = xenc.predict([("warm", "warm")])  # JIT-warm
    index = faiss.IndexFlatIP(384)
    index.add(corpus_embs)
    tokenized = [tokenize_bm25(corpus_content.get(cid, "")) for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    qid_to_idx = {q: i for i, q in enumerate(query_ids)}

    rif_config = RIFConfig()

    print(f"Corpus: {len(corpus_ids)} entries")
    print(f"Queries: {len(query_ids)}")
    print(f"RIF config: rate={rif_config.suppression_rate}, "
          f"reinforce={rif_config.reinforcement_rate}, "
          f"alpha={rif_config.alpha}, decay={rif_config.decay_lambda}")
    print()

    # State tracking
    suppression: dict[str, float] = {}
    last_updated: dict[str, int] = {}

    # Phase 1: Simulate usage with RIF
    print("Phase 1: Simulating 2000-query usage pattern with RIF...", flush=True)

    schedule_rng = np.random.default_rng(42)
    n_hot = max(1, int(len(query_ids) * 0.2))
    hot_idx = schedule_rng.choice(len(query_ids), size=n_hot, replace=False)
    hot_ids = [query_ids[i] for i in hot_idx]
    cold_ids = [q for q in query_ids if q not in set(hot_ids)]

    k_shallow = 30
    k_final = 10

    for step in range(2000):
        qid = (
            hot_ids[schedule_rng.integers(len(hot_ids))]
            if schedule_rng.random() < 0.7
            else cold_ids[schedule_rng.integers(len(cold_ids))]
        )
        qe = query_embs[qid_to_idx[qid]]
        qt = query_texts.get(qid, "")

        # Hybrid search with RIF penalty
        raw = search_hybrid_scored(qe, qt, k_shallow, index, bm25, corpus_ids)
        adjusted = apply_suppression_penalty(raw, suppression, rif_config.alpha)
        candidate_ids = [eid for eid, _ in adjusted[:k_shallow]]

        # Cross-encoder rerank
        pairs = [(qt, corpus_content.get(c, "")) for c in candidate_ids]
        xs = xenc.predict(pairs)
        scored = sorted(zip(candidate_ids, xs), key=lambda x: x[1], reverse=True)

        # Identify winners and competitors
        winner_ids = {eid for eid, _ in scored[:k_final]}

        rank_lookup = {eid: rank for rank, (eid, _) in enumerate(adjusted)}
        xenc_lookup = {eid: float(s) for eid, s in scored}
        competitor_data = [
            (eid, rank_lookup.get(eid, len(adjusted)), xenc_lookup.get(eid, 0.0))
            for eid in candidate_ids
        ]

        rif_updates = update_suppression(
            winner_ids, competitor_data, suppression,
            len(candidate_ids), rif_config, step, last_updated,
        )
        suppression.update(rif_updates)
        for eid in rif_updates:
            last_updated[eid] = step

        if (step + 1) % 500 == 0:
            n_suppressed = sum(1 for s in suppression.values() if s > 0.01)
            max_supp = max(suppression.values()) if suppression else 0
            mean_supp = np.mean([s for s in suppression.values() if s > 0.01]) if n_suppressed else 0
            print(f"  Step {step + 1}: {n_suppressed} entries suppressed, "
                  f"max={max_supp:.3f}, mean={mean_supp:.3f}", flush=True)

    print(flush=True)

    # Phase 2: Evaluate
    print("Phase 2: Evaluating (static vs RIF)...", flush=True)
    print()

    static_ndcgs = []
    rif_ndcgs = []
    static_recall_30 = []
    rif_recall_30 = []

    for qi in query_ids:
        qr = qrels.get(qi, {})
        if not qr:
            continue
        qe = query_embs[qid_to_idx[qi]]
        qt = query_texts.get(qi, "")
        relevant_set = set(qr.keys())

        raw = search_hybrid_scored(qe, qt, k_shallow, index, bm25, corpus_ids)

        # Static: no suppression
        static_candidates = [eid for eid, _ in raw[:k_shallow]]
        static_recall = len(relevant_set & set(static_candidates)) / len(relevant_set) if relevant_set else 0
        static_recall_30.append(static_recall)

        pairs_s = [(qt, corpus_content.get(c, "")) for c in static_candidates]
        xs_s = xenc.predict(pairs_s)
        scored_s = sorted(zip(static_candidates, xs_s), key=lambda x: x[1], reverse=True)
        static_top10 = [c for c, _ in scored_s[:k_final]]
        static_ndcgs.append(ndcg_at_k(static_top10, qr, k_final))

        # RIF: with suppression
        adjusted = apply_suppression_penalty(raw, suppression, rif_config.alpha)
        rif_candidates = [eid for eid, _ in adjusted[:k_shallow]]
        rif_recall = len(relevant_set & set(rif_candidates)) / len(relevant_set) if relevant_set else 0
        rif_recall_30.append(rif_recall)

        pairs_r = [(qt, corpus_content.get(c, "")) for c in rif_candidates]
        xs_r = xenc.predict(pairs_r)
        scored_r = sorted(zip(rif_candidates, xs_r), key=lambda x: x[1], reverse=True)
        rif_top10 = [c for c, _ in scored_r[:k_final]]
        rif_ndcgs.append(ndcg_at_k(rif_top10, qr, k_final))

    # Results
    s_ndcg = np.mean(static_ndcgs)
    r_ndcg = np.mean(rif_ndcgs)
    delta = (r_ndcg - s_ndcg) / s_ndcg * 100 if s_ndcg > 0 else 0

    s_recall = np.mean(static_recall_30)
    r_recall = np.mean(rif_recall_30)
    recall_delta = (r_recall - s_recall) / s_recall * 100 if s_recall > 0 else 0

    n_suppressed = sum(1 for s in suppression.values() if s > 0.01)
    n_heavy = sum(1 for s in suppression.values() if s > 0.5)

    print(f"{'Metric':<30} | {'Static':>8} | {'RIF':>8} | {'Δ':>8}")
    print("-" * 65)
    print(f"{'NDCG@10':<30} | {s_ndcg:>8.4f} | {r_ndcg:>8.4f} | {delta:>+7.1f}%")
    print(f"{'Recall@30 (candidate pool)':<30} | {s_recall:>8.4f} | {r_recall:>8.4f} | {recall_delta:>+7.1f}%")
    print(f"\nSuppression stats: {n_suppressed} entries suppressed, {n_heavy} heavily (>0.5)")
    print(f"Queries evaluated: {len(static_ndcgs)}")

    # Write results
    print(f"\nWriting {RESULTS}...", flush=True)
    with open(RESULTS, "w") as f:
        f.write("# RIF Benchmark\n\n")
        f.write("Retrieval-Induced Forgetting: principled competitor suppression.\n\n")
        f.write(f"**Protocol**: Simulate 2000 queries to build suppression state, "
                f"then evaluate on {len(static_ndcgs)} queries.\n\n")
        f.write(f"**RIF config**: rate={rif_config.suppression_rate}, "
                f"reinforce={rif_config.reinforcement_rate}, "
                f"alpha={rif_config.alpha}, decay={rif_config.decay_lambda}\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("| Metric | Static | RIF | Δ |\n")
        f.write("|--------|--------|-----|---|\n")
        f.write(f"| NDCG@10 | {s_ndcg:.4f} | {r_ndcg:.4f} | {delta:+.1f}% |\n")
        f.write(f"| Recall@30 | {s_recall:.4f} | {r_recall:.4f} | {recall_delta:+.1f}% |\n")
        f.write(f"\nSuppression: {n_suppressed} entries suppressed, {n_heavy} heavily (>0.5)\n")
        f.write(f"\n## Mechanism\n\n")
        f.write("After each retrieval, entries that competed (made it into the candidate pool)\n")
        f.write("but lost (ranked low by cross-encoder) accumulate suppression. Suppression\n")
        f.write("is applied as a penalty to hybrid search scores before cross-encoder reranking,\n")
        f.write("pushing chronic distractors down and allowing new candidates in.\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

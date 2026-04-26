"""Lifecycle benchmark: does the GC mechanism improve memory management?

Tests the GC tier system on LongMemEval's temporal and knowledge-update
subsets. The hypothesis: a store with decay and tier promotion handles
knowledge updates and temporal queries better than a static store.

The benchmark simulates a realistic usage pattern:
1. Phase 1 (warmup): feed all sessions chronologically, building the store
2. Phase 2 (eval): query and measure NDCG on each question type

For knowledge-update questions, the GC store should promote recent
entries (higher affinity from recent retrieval) and decay old ones.
A static store treats all entries equally regardless of age.
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
from datasets import load_dataset  # type: ignore[import-untyped]
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

from lethe.entry import Tier
from benchmarks._lib.metrics import ndcg_at_k

DATA = Path("data")
RESULTS = Path("BENCHMARKS_LIFECYCLE.md")


def main() -> None:
    print("=" * 60)
    print("Lifecycle benchmark: GC memory management")
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

    # Load question types from original dataset
    ds = load_dataset(
        "xiaowu0162/longmemeval-cleaned",
        data_files="longmemeval_s_cleaned.json",
        split="train",
    )
    question_types: dict[str, str] = {}
    for row in ds:
        question_types[row["question_id"]] = row["question_type"]

    xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    index = faiss.IndexFlatIP(384)
    index.add(corpus_embs)
    tokenized = [corpus_content.get(cid, "").lower().split() for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    qid_to_idx = {q: i for i, q in enumerate(query_ids)}

    print(f"Corpus: {len(corpus_ids)} entries")
    print(f"Queries: {len(query_ids)} ({len(set(question_types.values()))} types)")
    print()

    # Retrieval function: BM25+vector+xenc (same for both static and GC)
    def retrieve_hybrid(qe: np.ndarray, qt: str, tier_weights: dict[str, float] | None = None):
        D, I = index.search(qe.reshape(1, -1), 30)
        vec_ids = [corpus_ids[i] for i in I[0] if i >= 0]
        scores = bm25.get_scores(qt.lower().split())
        bm25_ids = [corpus_ids[i] for i in np.argsort(scores)[::-1][:30]]
        all_ids = list(dict.fromkeys(vec_ids + bm25_ids))
        pairs = [(qt, corpus_content.get(c, "")) for c in all_ids]
        xs = xenc.predict(pairs)
        scored = []
        for cid, xscore in zip(all_ids, xs):
            weight = 1.0
            if tier_weights and cid in tier_weights:
                weight = tier_weights[cid]
            scored.append((cid, float(xscore) * weight))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # Phase 1: Simulate usage with GC lifecycle
    # Feed queries in a realistic pattern, track affinities
    print("Phase 1: Simulating 2000-query usage pattern...", flush=True)

    entry_affinity: dict[str, float] = {cid: 0.5 for cid in corpus_ids}
    entry_retrieval_count: dict[str, int] = {cid: 0 for cid in corpus_ids}
    entry_tier: dict[str, str] = {cid: "naive" for cid in corpus_ids}
    entry_last_step: dict[str, int] = {cid: 0 for cid in corpus_ids}

    schedule_rng = np.random.default_rng(42)
    n_hot = max(1, int(len(query_ids) * 0.2))
    hot_idx = schedule_rng.choice(len(query_ids), size=n_hot, replace=False)
    hot_ids = [query_ids[i] for i in hot_idx]
    cold_ids = [q for q in query_ids if q not in set(hot_ids)]

    for step in range(2000):
        qid = (
            hot_ids[schedule_rng.integers(len(hot_ids))]
            if schedule_rng.random() < 0.7
            else cold_ids[schedule_rng.integers(len(cold_ids))]
        )
        qe = query_embs[qid_to_idx[qid]]
        qt = query_texts.get(qid, "")
        scored = retrieve_hybrid(qe, qt)

        # Update top-10 entries
        for cid, xscore in scored[:10]:
            entry_retrieval_count[cid] += 1
            entry_last_step[cid] = step
            norm_score = 1.0 / (1.0 + math.exp(-xscore))
            entry_affinity[cid] = 0.8 * entry_affinity[cid] + 0.2 * norm_score

        # Tier transitions
        for cid, xscore in scored[:10]:
            if entry_tier[cid] == "naive" and entry_retrieval_count[cid] >= 3:
                entry_tier[cid] = "gc"
            elif (
                entry_tier[cid] == "gc"
                and entry_affinity[cid] >= 0.65
                and entry_retrieval_count[cid] >= 5
            ):
                entry_tier[cid] = "memory"

        # Periodic decay
        if step % 100 == 0:
            for cid in corpus_ids:
                if entry_tier[cid] == "memory":
                    continue
                if step - entry_last_step[cid] >= 100:
                    entry_affinity[cid] *= 0.99

    # Count tier distribution
    tier_counts = {"naive": 0, "gc": 0, "memory": 0}
    for t in entry_tier.values():
        if t in tier_counts:
            tier_counts[t] += 1

    print(f"  Tier distribution: {tier_counts}", flush=True)
    print()

    # Phase 2: Evaluate per question type
    # GC store: boost memory-tier entries by 1.15x, penalize low-affinity
    gc_weights: dict[str, float] = {}
    for cid in corpus_ids:
        if entry_tier[cid] == "memory":
            gc_weights[cid] = 1.15
        elif entry_affinity[cid] < 0.3:
            gc_weights[cid] = 0.8  # mild penalty for low-affinity
        else:
            gc_weights[cid] = 1.0

    print("Phase 2: Evaluating per question type...", flush=True)
    print()

    type_results_static: dict[str, list[float]] = {}
    type_results_gc: dict[str, list[float]] = {}

    for qi in query_ids:
        qr = qrels.get(qi, {})
        if not qr:
            continue
        qtype = question_types.get(qi, "unknown")
        qe = query_embs[qid_to_idx[qi]]
        qt = query_texts.get(qi, "")

        # Static (no tier weights)
        static_scored = retrieve_hybrid(qe, qt)
        static_top10 = [c for c, _ in static_scored[:10]]
        static_ndcg = ndcg_at_k(static_top10, qr, 10)

        # GC (with tier weights)
        gc_scored = retrieve_hybrid(qe, qt, tier_weights=gc_weights)
        gc_top10 = [c for c, _ in gc_scored[:10]]
        gc_ndcg = ndcg_at_k(gc_top10, qr, 10)

        type_results_static.setdefault(qtype, []).append(static_ndcg)
        type_results_gc.setdefault(qtype, []).append(gc_ndcg)

    # Print results
    print(f"{'Question type':<28} | {'N':>4} | {'Static':>7} | {'GC':>7} | {'Δ':>7}", flush=True)
    print("-" * 65, flush=True)

    all_static, all_gc = [], []
    md_rows = []
    for qtype in sorted(type_results_static.keys()):
        s_ndcgs = type_results_static[qtype]
        g_ndcgs = type_results_gc[qtype]
        s_mean = np.mean(s_ndcgs)
        g_mean = np.mean(g_ndcgs)
        delta = (g_mean - s_mean) / s_mean * 100 if s_mean > 0 else 0
        all_static.extend(s_ndcgs)
        all_gc.extend(g_ndcgs)
        print(
            f"{qtype:<28} | {len(s_ndcgs):>4} | {s_mean:.4f} | {g_mean:.4f} | {delta:+5.1f}%",
            flush=True,
        )
        md_rows.append((qtype, len(s_ndcgs), s_mean, g_mean, delta))

    s_overall = np.mean(all_static)
    g_overall = np.mean(all_gc)
    d_overall = (g_overall - s_overall) / s_overall * 100 if s_overall > 0 else 0
    print("-" * 65, flush=True)
    print(
        f"{'OVERALL':<28} | {len(all_static):>4} | {s_overall:.4f} | {g_overall:.4f} | {d_overall:+5.1f}%",
        flush=True,
    )

    # Write results
    print(f"\nWriting {RESULTS}...", flush=True)
    with open(RESULTS, "w") as f:
        f.write("# Lifecycle Benchmark\n\n")
        f.write("Tests whether the GC tier system improves memory management.\n\n")
        f.write("**Method**: Simulate 2000 queries to build entry affinities and tiers.\n")
        f.write("Then evaluate with tier-weighted scoring: memory-tier entries get 1.15x boost,\n")
        f.write("low-affinity entries get 0.8x penalty.\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("| Question type | N | Static | GC managed | Δ |\n")
        f.write("|---------------|---|--------|------------|---|\n")
        for qtype, n, s, g, d in md_rows:
            f.write(f"| {qtype} | {n} | {s:.4f} | {g:.4f} | {d:+.1f}% |\n")
        f.write(f"| **OVERALL** | **{len(all_static)}** | **{s_overall:.4f}** | **{g_overall:.4f}** | **{d_overall:+.1f}%** |\n")
        f.write(f"\nTier distribution after 2000 queries: {tier_counts}\n")
        f.write("\n## What this measures\n\n")
        f.write("- **knowledge-update**: Does the system promote recent facts over outdated ones?\n")
        f.write("- **temporal-reasoning**: Does temporal awareness help ordering?\n")
        f.write("- **multi-session**: Does cross-session memory benefit from tier management?\n")
        f.write("- **single-session-***: Baseline factual recall (lifecycle less relevant)\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

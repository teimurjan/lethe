"""RIF hyperparameter sweep: find the strongest configuration.

Shares expensive computation across configs: FAISS, BM25, and cross-encoder
calls are done once per query. Each config only varies the suppression
application. This makes a 6-config sweep ~1.7x the cost of a single run
instead of 6x.

Approach: cross-encode the FULL hybrid pool (~60 entries) once, then each
config applies its own suppression state to select winners/competitors.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

from benchmarks._lib.metrics import ndcg_at_k
from lethe.rif import RIFConfig, apply_suppression_penalty, update_suppression

DATA = Path("data")
RESULTS = Path("BENCHMARKS_RIF_SWEEP.md")


@dataclass
class SweepConfig:
    name: str
    rif: RIFConfig
    burn_in: int = 5000


CONFIGS = [
    SweepConfig("baseline (no RIF)", RIFConfig(alpha=0.0)),
    SweepConfig("v1-conservative", RIFConfig(suppression_rate=0.1, reinforcement_rate=0.05, alpha=0.3, decay_lambda=0.005, max_suppression=1.0)),
    SweepConfig("v2-higher-alpha", RIFConfig(suppression_rate=0.1, reinforcement_rate=0.05, alpha=1.0, decay_lambda=0.005, max_suppression=1.0)),
    SweepConfig("v3-fast-accum", RIFConfig(suppression_rate=0.3, reinforcement_rate=0.1, alpha=1.0, decay_lambda=0.005, max_suppression=2.0)),
    SweepConfig("v4-no-decay", RIFConfig(suppression_rate=0.1, reinforcement_rate=0.05, alpha=1.0, decay_lambda=0.0, max_suppression=1.0)),
    SweepConfig("v5-aggressive", RIFConfig(suppression_rate=0.5, reinforcement_rate=0.1, alpha=2.0, decay_lambda=0.001, max_suppression=3.0)),
    SweepConfig("v6-extreme", RIFConfig(suppression_rate=0.5, reinforcement_rate=0.1, alpha=5.0, decay_lambda=0.001, max_suppression=5.0)),
]


def hybrid_search_scored(
    query_emb: np.ndarray,
    query_text: str,
    k: int,
    index: faiss.IndexFlatIP,
    bm25: BM25Okapi,
    corpus_ids: list[str],
) -> list[tuple[str, float]]:
    D, I = index.search(query_emb.reshape(1, -1), k)
    vec_results = [(corpus_ids[i], float(D[0][rank])) for rank, i in enumerate(I[0]) if i >= 0]
    tokens = query_text.lower().split()
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
    print("RIF hyperparameter sweep")
    print("=" * 60)
    print()

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

    xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    index = faiss.IndexFlatIP(384)
    index.add(corpus_embs)
    tokenized = [corpus_content.get(cid, "").lower().split() for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    qid_to_idx = {q: i for i, q in enumerate(query_ids)}

    print(f"Corpus: {len(corpus_ids)} entries")
    print(f"Queries: {len(query_ids)}")
    print(f"Configs: {len(CONFIGS)}")
    print()
    for sc in CONFIGS:
        r = sc.rif
        print(f"  {sc.name}: alpha={r.alpha} rate={r.suppression_rate} "
              f"decay={r.decay_lambda} max={r.max_suppression}")
    print()

    # Per-config state
    states: list[dict[str, float]] = [{} for _ in CONFIGS]
    last_updated: list[dict[str, int]] = [{} for _ in CONFIGS]

    # Phase 1: shared burn-in
    max_burn = max(sc.burn_in for sc in CONFIGS)
    print(f"Phase 1: {max_burn}-step burn-in ({len(CONFIGS)} configs, shared search+xenc)...",
          flush=True)

    schedule_rng = np.random.default_rng(42)
    n_hot = max(1, int(len(query_ids) * 0.2))
    hot_idx = schedule_rng.choice(len(query_ids), size=n_hot, replace=False)
    hot_ids = [query_ids[i] for i in hot_idx]
    cold_ids = [q for q in query_ids if q not in set(hot_ids)]
    k_shallow = 30
    k_final = 10

    for step in range(max_burn):
        qid = (
            hot_ids[schedule_rng.integers(len(hot_ids))]
            if schedule_rng.random() < 0.7
            else cold_ids[schedule_rng.integers(len(cold_ids))]
        )
        qe = query_embs[qid_to_idx[qid]]
        qt = query_texts.get(qid, "")

        # Shared: hybrid search + cross-encode full pool
        raw = hybrid_search_scored(qe, qt, k_shallow, index, bm25, corpus_ids)
        all_pool_ids = [eid for eid, _ in raw]
        pairs = [(qt, corpus_content.get(c, "")) for c in all_pool_ids]
        xenc_scores = xenc.predict(pairs)
        xenc_map = {eid: float(s) for eid, s in zip(all_pool_ids, xenc_scores)}

        # Per-config: apply suppression, determine winners/competitors, update
        for ci, sc in enumerate(CONFIGS):
            if step >= sc.burn_in:
                continue
            adjusted = apply_suppression_penalty(raw, states[ci], sc.rif.alpha)
            candidate_ids = [eid for eid, _ in adjusted[:k_shallow]]

            # Select winners from pre-computed xenc scores
            scored = sorted(
                [(eid, xenc_map.get(eid, -10.0)) for eid in candidate_ids],
                key=lambda x: x[1], reverse=True,
            )
            winner_ids = {eid for eid, _ in scored[:k_final]}

            # RIF update
            rank_lookup = {eid: rank for rank, (eid, _) in enumerate(adjusted)}
            competitor_data = [
                (eid, rank_lookup.get(eid, len(adjusted)), xenc_map.get(eid, 0.0))
                for eid in candidate_ids
            ]
            rif_updates = update_suppression(
                winner_ids, competitor_data, states[ci],
                len(candidate_ids), sc.rif, step, last_updated[ci],
            )
            states[ci].update(rif_updates)
            for eid in rif_updates:
                last_updated[ci][eid] = step

        if (step + 1) % 1000 == 0:
            print(f"  Step {step + 1}:", flush=True)
            for ci, sc in enumerate(CONFIGS):
                if sc.rif.alpha == 0:
                    continue
                n_supp = sum(1 for s in states[ci].values() if s > 0.01)
                mx = max(states[ci].values()) if states[ci] else 0
                mn = np.mean([s for s in states[ci].values() if s > 0.01]) if n_supp else 0
                print(f"    {sc.name}: {n_supp} suppressed, max={mx:.3f}, mean={mn:.3f}",
                      flush=True)

    print(flush=True)

    # Phase 2: eval (shared search + xenc)
    print("Phase 2: Evaluating all configs (shared search+xenc)...", flush=True)
    print()

    # Per-config results
    config_ndcgs: list[list[float]] = [[] for _ in CONFIGS]
    config_recalls: list[list[float]] = [[] for _ in CONFIGS]

    for qi in query_ids:
        qr = qrels.get(qi, {})
        if not qr:
            continue
        qe = query_embs[qid_to_idx[qi]]
        qt = query_texts.get(qi, "")
        relevant_set = set(qr.keys())

        # Shared: hybrid search + cross-encode full pool
        raw = hybrid_search_scored(qe, qt, k_shallow, index, bm25, corpus_ids)
        all_pool_ids = [eid for eid, _ in raw]
        pairs = [(qt, corpus_content.get(c, "")) for c in all_pool_ids]
        xenc_scores = xenc.predict(pairs)
        xenc_map = {eid: float(s) for eid, s in zip(all_pool_ids, xenc_scores)}

        # Per-config: apply suppression, compute NDCG
        for ci, sc in enumerate(CONFIGS):
            adjusted = apply_suppression_penalty(raw, states[ci], sc.rif.alpha)
            candidate_ids = [eid for eid, _ in adjusted[:k_shallow]]

            recall = len(relevant_set & set(candidate_ids)) / len(relevant_set) if relevant_set else 0
            config_recalls[ci].append(recall)

            scored = sorted(
                [(eid, xenc_map.get(eid, -10.0)) for eid in candidate_ids],
                key=lambda x: x[1], reverse=True,
            )
            top10 = [c for c, _ in scored[:k_final]]
            config_ndcgs[ci].append(ndcg_at_k(top10, qr, k_final))

    # Results
    print(f"{'Config':<25} | {'NDCG@10':>8} | {'Δ':>7} | {'Recall@30':>9} | {'Δ':>7} | {'Suppressed':>10}")
    print("-" * 85)

    baseline_ndcg = np.mean(config_ndcgs[0])
    baseline_recall = np.mean(config_recalls[0])

    md_rows = []
    for ci, sc in enumerate(CONFIGS):
        ndcg = np.mean(config_ndcgs[ci])
        recall = np.mean(config_recalls[ci])
        d_ndcg = (ndcg - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg > 0 else 0
        d_recall = (recall - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else 0
        n_supp = sum(1 for s in states[ci].values() if s > 0.01)
        mx = max(states[ci].values()) if states[ci] else 0

        print(f"{sc.name:<25} | {ndcg:>8.4f} | {d_ndcg:>+6.1f}% | {recall:>9.4f} | {d_recall:>+6.1f}% | {n_supp:>10}")
        md_rows.append((sc.name, sc.rif, ndcg, d_ndcg, recall, d_recall, n_supp, mx))

    n_eval = len(config_ndcgs[0])
    print(f"\nQueries evaluated: {n_eval}")

    # Write results
    print(f"\nWriting {RESULTS}...", flush=True)
    with open(RESULTS, "w") as f:
        f.write("# RIF Hyperparameter Sweep\n\n")
        f.write(f"Burn-in: {max_burn} steps, Eval: {n_eval} queries\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("| Config | alpha | rate | decay | max | NDCG@10 | Δ | Recall@30 | Δ | Suppressed |\n")
        f.write("|--------|-------|------|-------|-----|---------|---|-----------|---|------------|\n")
        for name, rif, ndcg, d_ndcg, recall, d_recall, n_supp, mx in md_rows:
            f.write(f"| {name} | {rif.alpha} | {rif.suppression_rate} | "
                    f"{rif.decay_lambda} | {rif.max_suppression} | "
                    f"{ndcg:.4f} | {d_ndcg:+.1f}% | {recall:.4f} | {d_recall:+.1f}% | {n_supp} |\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

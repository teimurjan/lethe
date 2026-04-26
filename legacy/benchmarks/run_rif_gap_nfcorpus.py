"""Rank-gap RIF on NFCorpus (BEIR medical IR, 3,633 docs, 323 queries).

Second-dataset replication of the LongMemEval rank-gap RIF benchmark. Same
configs, same suppression formulas, different corpus. Because NFCorpus is
non-conversational ad-hoc IR with graded relevance judgments, this tests
whether clustered RIF generalizes beyond long-term conversation memory.

Burn-in is reduced to 3000 steps (from 5000 on LongMemEval) because the
query pool is only 323 items; with-replacement sampling still gives each
query around 9 exposures during burn-in.
"""
from __future__ import annotations

import json
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
from lethe.rif import (
    ClusteredSuppressionState,
    RIFConfig,
    apply_suppression_penalty,
    assign_cluster,
    build_clusters,
    update_suppression,
)

DATA = Path("data")
RESULTS = Path("benchmarks/results/BENCHMARKS_RIF_GAP_NFCORPUS.md")
PER_QUERY_OUT = Path("benchmarks/results/rif_gap_per_query_nfcorpus.json")
DATASET_PREFIX = "nfcorpus"


@dataclass
class Config:
    name: str
    rif: RIFConfig
    n_clusters: int  # 0 = global


BASE_PARAMS = dict(
    alpha=0.3, suppression_rate=0.1, reinforcement_rate=0.05, decay_lambda=0.005,
)

CONFIGS = [
    Config("baseline", RIFConfig(alpha=0.0), 0),
    Config("global-original", RIFConfig(**BASE_PARAMS), 0),
    Config("global-gap", RIFConfig(**BASE_PARAMS, use_rank_gap=True), 0),
    Config("clustered30-original", RIFConfig(**BASE_PARAMS), 30),
    Config("clustered30-gap", RIFConfig(**BASE_PARAMS, use_rank_gap=True), 30),
]

BURN_IN = 3000
K_SHALLOW = 30
K_FINAL = 10


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
    print("Rank-gap RIF: competition from rank drop signal")
    print("=" * 60)
    print()

    data = np.load(str(DATA / f"{DATASET_PREFIX}_prepared.npz"), allow_pickle=True)
    corpus_ids = list(data["corpus_ids"])
    corpus_embs = data["corpus_embeddings"].astype(np.float32)
    query_ids = list(data["query_ids"])
    query_embs = data["query_embeddings"].astype(np.float32)
    with open(DATA / f"{DATASET_PREFIX}_qrels.json") as f:
        qrels = json.load(f)
    with open(DATA / f"{DATASET_PREFIX}_corpus.json") as f:
        corpus_content = json.load(f)
    with open(DATA / f"{DATASET_PREFIX}_queries.json") as f:
        query_texts = json.load(f)

    xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    index = faiss.IndexFlatIP(384)
    index.add(corpus_embs)
    tokenized = [corpus_content.get(cid, "").lower().split() for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    qid_to_idx = {q: i for i, q in enumerate(query_ids)}

    cluster_centroids: dict[int, np.ndarray] = {}
    unique_n = {c.n_clusters for c in CONFIGS if c.n_clusters > 0}
    for n in unique_n:
        print(f"Building {n} query clusters...", flush=True)
        cluster_centroids[n] = build_clusters(query_embs, n)

    print(f"\nCorpus: {len(corpus_ids)} entries")
    print(f"Queries: {len(query_ids)}")
    print(f"Burn-in: {BURN_IN} steps")
    print(f"Configs: {len(CONFIGS)}")
    for c in CONFIGS:
        formula = "gap" if c.rif.use_rank_gap else "original"
        mode = "global" if c.n_clusters == 0 else f"{c.n_clusters} clusters"
        print(f"  {c.name}: formula={formula}, {mode}")
    print()

    # State
    global_states: list[dict[str, float]] = [{} for _ in CONFIGS]
    global_last_updated: list[dict[str, int]] = [{} for _ in CONFIGS]
    clustered_states: list[ClusteredSuppressionState | None] = [
        ClusteredSuppressionState() if c.n_clusters > 0 else None for c in CONFIGS
    ]

    # Phase 1: burn-in
    print(f"Phase 1: {BURN_IN}-step burn-in...", flush=True)

    schedule_rng = np.random.default_rng(42)
    n_hot = max(1, int(len(query_ids) * 0.2))
    hot_idx = schedule_rng.choice(len(query_ids), size=n_hot, replace=False)
    hot_ids = [query_ids[i] for i in hot_idx]
    cold_ids = [q for q in query_ids if q not in set(hot_ids)]

    for step in range(BURN_IN):
        qid = (
            hot_ids[schedule_rng.integers(len(hot_ids))]
            if schedule_rng.random() < 0.7
            else cold_ids[schedule_rng.integers(len(cold_ids))]
        )
        qe = query_embs[qid_to_idx[qid]]
        qt = query_texts.get(qid, "")

        raw = hybrid_search_scored(qe, qt, K_SHALLOW, index, bm25, corpus_ids)
        all_pool_ids = [eid for eid, _ in raw]
        pairs = [(qt, corpus_content.get(c, "")) for c in all_pool_ids]
        xenc_scores = xenc.predict(pairs)
        xenc_map = {eid: float(s) for eid, s in zip(all_pool_ids, xenc_scores)}

        for ci, cfg in enumerate(CONFIGS):
            if cfg.rif.alpha == 0:
                continue

            if cfg.n_clusters == 0:
                supp_scores = global_states[ci]
                lu = global_last_updated[ci]
            else:
                cid = assign_cluster(qe, cluster_centroids[cfg.n_clusters])
                cs = clustered_states[ci]
                supp_scores = cs.get_cluster_scores(cid)
                lu = cs.get_cluster_last_updated(cid)

            adjusted = apply_suppression_penalty(raw, supp_scores, cfg.rif.alpha)
            candidate_ids = [eid for eid, _ in adjusted[:K_SHALLOW]]

            scored = sorted(
                [(eid, xenc_map.get(eid, -10.0)) for eid in candidate_ids],
                key=lambda x: x[1], reverse=True,
            )
            winner_ids = {eid for eid, _ in scored[:K_FINAL]}

            rank_lookup = {eid: rank for rank, (eid, _) in enumerate(adjusted)}
            xenc_rank_lookup = {eid: rank for rank, (eid, _) in enumerate(scored)}

            if cfg.rif.use_rank_gap:
                competitor_data = [
                    (eid, rank_lookup.get(eid, len(adjusted)),
                     xenc_rank_lookup.get(eid, len(scored)),
                     xenc_map.get(eid, 0.0))
                    for eid in candidate_ids
                ]
            else:
                competitor_data = [
                    (eid, rank_lookup.get(eid, len(adjusted)), xenc_map.get(eid, 0.0))
                    for eid in candidate_ids
                ]

            rif_updates = update_suppression(
                winner_ids, competitor_data, supp_scores,
                len(candidate_ids), cfg.rif, step, lu,
            )

            if cfg.n_clusters == 0:
                global_states[ci].update(rif_updates)
                for eid in rif_updates:
                    global_last_updated[ci][eid] = step
            else:
                clustered_states[ci].update_cluster(cid, rif_updates, step)

        if (step + 1) % 1000 == 0:
            print(f"  Step {step + 1}:", flush=True)
            for ci, cfg in enumerate(CONFIGS):
                if cfg.rif.alpha == 0:
                    continue
                if cfg.n_clusters == 0:
                    n_s = sum(1 for s in global_states[ci].values() if s > 0.01)
                    mx = max(global_states[ci].values()) if global_states[ci] else 0
                    mn = np.mean([s for s in global_states[ci].values() if s > 0.01]) if n_s else 0
                else:
                    cs = clustered_states[ci]
                    n_s = cs.total_suppressed()
                    mx = cs.max_suppression()
                    mn = cs.mean_suppression()
                print(f"    {cfg.name}: {n_s} suppressed, max={mx:.3f}, mean={mn:.3f}",
                      flush=True)

    print(flush=True)

    # Phase 2
    print("Phase 2: Evaluating...", flush=True)
    print()

    config_ndcgs: list[list[float]] = [[] for _ in CONFIGS]
    config_recalls: list[list[float]] = [[] for _ in CONFIGS]
    evaluated_qids: list[str] = []

    for qi in query_ids:
        qr = qrels.get(qi, {})
        if not qr:
            continue
        evaluated_qids.append(qi)
        qe = query_embs[qid_to_idx[qi]]
        qt = query_texts.get(qi, "")
        relevant_set = set(qr.keys())

        raw = hybrid_search_scored(qe, qt, K_SHALLOW, index, bm25, corpus_ids)
        all_pool_ids = [eid for eid, _ in raw]
        pairs = [(qt, corpus_content.get(c, "")) for c in all_pool_ids]
        xenc_scores = xenc.predict(pairs)
        xenc_map = {eid: float(s) for eid, s in zip(all_pool_ids, xenc_scores)}

        for ci, cfg in enumerate(CONFIGS):
            if cfg.n_clusters == 0:
                supp_scores = global_states[ci]
            else:
                cid = assign_cluster(qe, cluster_centroids[cfg.n_clusters])
                supp_scores = clustered_states[ci].get_cluster_scores(cid)

            adjusted = apply_suppression_penalty(raw, supp_scores, cfg.rif.alpha)
            candidate_ids = [eid for eid, _ in adjusted[:K_SHALLOW]]

            recall = len(relevant_set & set(candidate_ids)) / len(relevant_set) if relevant_set else 0
            config_recalls[ci].append(recall)

            scored = sorted(
                [(eid, xenc_map.get(eid, -10.0)) for eid in candidate_ids],
                key=lambda x: x[1], reverse=True,
            )
            top10 = [c for c, _ in scored[:K_FINAL]]
            config_ndcgs[ci].append(ndcg_at_k(top10, qr, K_FINAL))

    baseline_ndcg = np.mean(config_ndcgs[0])
    baseline_recall = np.mean(config_recalls[0])

    print(f"{'Config':<25} | {'NDCG@10':>8} | {'Δ':>7} | {'Recall@30':>9} | {'Δ':>7}")
    print("-" * 70)

    md_rows = []
    for ci, cfg in enumerate(CONFIGS):
        ndcg = np.mean(config_ndcgs[ci])
        recall = np.mean(config_recalls[ci])
        d_ndcg = (ndcg - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg > 0 else 0
        d_recall = (recall - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else 0
        print(f"{cfg.name:<25} | {ndcg:>8.4f} | {d_ndcg:>+6.1f}% | {recall:>9.4f} | {d_recall:>+6.1f}%")
        md_rows.append((cfg.name, cfg.n_clusters, cfg.rif.use_rank_gap, ndcg, d_ndcg, recall, d_recall))

    n_eval = len(config_ndcgs[0])
    print(f"\nQueries evaluated: {n_eval}")

    print(f"\nWriting {RESULTS}...", flush=True)
    with open(RESULTS, "w") as f:
        f.write("# Rank-gap RIF Benchmark (NFCorpus)\n\n")
        f.write("Competition strength from rank drop (initial vs xenc) instead of rank alone.\n\n")
        f.write(f"Burn-in: {BURN_IN} steps, Eval: {n_eval} queries\n")
        f.write(f"RIF params: alpha=0.3, rate=0.1, decay=0.005 (all configs)\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("| Config | Clusters | Formula | NDCG@10 | Δ | Recall@30 | Δ |\n")
        f.write("|--------|----------|---------|---------|---|-----------|---|\n")
        for name, nc, gap, ndcg, d_ndcg, recall, d_recall in md_rows:
            cl = "global" if nc == 0 else str(nc)
            fm = "gap" if gap else "original"
            f.write(f"| {name} | {cl} | {fm} | {ndcg:.4f} | {d_ndcg:+.1f}% | "
                    f"{recall:.4f} | {d_recall:+.1f}% |\n")

    # Persist per-query arrays so bootstrap CIs and permutation tests can
    # run against the same data without re-running the full benchmark.
    PER_QUERY_OUT.parent.mkdir(parents=True, exist_ok=True)
    per_query_payload = {
        "queries": evaluated_qids,
        "configs": [
            {
                "name": cfg.name,
                "n_clusters": cfg.n_clusters,
                "use_rank_gap": bool(cfg.rif.use_rank_gap),
                "ndcg": config_ndcgs[ci],
                "recall": config_recalls[ci],
            }
            for ci, cfg in enumerate(CONFIGS)
        ],
        "meta": {
            "burn_in": BURN_IN,
            "n_eval": n_eval,
            "date": time.strftime("%Y-%m-%d"),
        },
    }
    with open(PER_QUERY_OUT, "w") as f:
        json.dump(per_query_payload, f)
    print(f"Wrote {PER_QUERY_OUT}", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

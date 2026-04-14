"""Exploration + rescue list: discover relevant entries beyond top-30.

RIF pushes false positives DOWN. Exploration pulls false negatives UP.

Mechanism: periodically, fetch a deeper pool (top-80) and cross-encode
positions 31-80. High-scoring finds get added to a per-cluster rescue
list. On subsequent queries in the same cluster, rescue-list entries
are injected into the candidate pool even if they aren't in the normal
top-30.

Fast version: precomputes xenc scores for top-80 of every query so all
configs can share the expensive computation.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]

from gc_memory.metrics import ndcg_at_k
from gc_memory.rif import (
    ClusteredSuppressionState,
    RIFConfig,
    apply_suppression_penalty,
    assign_cluster,
    build_clusters,
    update_suppression,
)

DATA = Path("data")
RESULTS = Path("BENCHMARKS_RIF_EXPLORE.md")


@dataclass
class Config:
    name: str
    rif: RIFConfig
    n_clusters: int
    explore_every: int  # 0 = no exploration, N = explore every Nth step
    rescue_threshold: float  # xenc score required to enter rescue list
    rescue_max_per_cluster: int  # cap rescue list size


BASE = dict(
    alpha=0.3, suppression_rate=0.1, reinforcement_rate=0.05, decay_lambda=0.005,
    use_rank_gap=True,
)

CONFIGS = [
    Config("baseline-no-rif", RIFConfig(alpha=0.0), 0, 0, 1.0, 50),
    Config("current-best", RIFConfig(**BASE), 30, 0, 1.0, 50),
    Config("+rescue-sparse", RIFConfig(**BASE), 30, 20, 1.0, 50),
    Config("+rescue-moderate", RIFConfig(**BASE), 30, 5, 1.0, 50),
    Config("+rescue-dense", RIFConfig(**BASE), 30, 1, 1.0, 50),
]

BURN_IN = 800
EVAL_N = 100
K_SHALLOW = 30
K_EXPLORE = 80
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


def run_query_for_config(
    step: int,
    cfg: Config,
    qe: np.ndarray,
    raw: list[tuple[str, float]],  # top-80 hybrid results (shared)
    xenc_map: dict[str, float],  # precomputed xenc scores for all in raw
    cluster_centroids: np.ndarray | None,
    clustered_state: ClusteredSuppressionState | None,
    global_state: dict[str, float] | None,
    global_lu: dict[str, int] | None,
    rescue_lists: dict[int, set[str]],
) -> tuple[set[str], list[tuple[str, float]]]:
    """Run one retrieval for a config. Returns (winner_ids, top-k results)."""
    # Full pool = top-80 hybrid (lets suppression demote entries and pull
    # from positions 30-80). Matches the pool-size behavior of clustered.py.
    candidate_pool = list(raw)
    all_pool_ids = {eid for eid, _ in raw}
    # Track what was originally in the top-30 (for exploration detection)
    top_30_ids = {eid for eid, _ in raw[:K_SHALLOW]}

    # Cluster assignment
    if cfg.n_clusters > 0:
        cid = assign_cluster(qe, cluster_centroids)
        # Inject rescue list entries from this cluster (entries discovered
        # in past exploration that aren't in the current top-80)
        for eid in rescue_lists.get(cid, set()):
            if eid not in all_pool_ids and eid in xenc_map:
                candidate_pool.append((eid, 0.04))  # above max RRF, priority inject
                all_pool_ids.add(eid)
    else:
        cid = 0

    # Get suppression state for this cluster
    if cfg.n_clusters > 0 and clustered_state is not None:
        supp = clustered_state.get_cluster_scores(cid)
        lu = clustered_state.get_cluster_last_updated(cid)
    elif global_state is not None:
        supp = global_state
        lu = global_lu  # type: ignore[assignment]
    else:
        supp = {}
        lu = {}

    # Apply RIF suppression
    adjusted = apply_suppression_penalty(candidate_pool, supp, cfg.rif.alpha)
    candidate_ids = [eid for eid, _ in adjusted[:K_SHALLOW]]

    # Score via precomputed xenc
    scored = sorted(
        [(eid, xenc_map.get(eid, -10.0)) for eid in candidate_ids],
        key=lambda x: x[1], reverse=True,
    )
    winner_ids = {eid for eid, _ in scored[:K_FINAL]}

    # RIF update
    if cfg.rif.alpha > 0:
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
            winner_ids, competitor_data, supp,
            len(candidate_ids), cfg.rif, step, lu,
        )
        if cfg.n_clusters > 0 and clustered_state is not None:
            clustered_state.update_cluster(cid, rif_updates, step)
        elif global_state is not None:
            global_state.update(rif_updates)
            for eid in rif_updates:
                global_lu[eid] = step  # type: ignore[index]

    # Exploration: check positions 30-80 for high scorers, add to rescue list
    if cfg.explore_every > 0 and step >= 0 and step % cfg.explore_every == 0:
        for eid, _ in raw[K_SHALLOW:]:  # positions 30-80
            if eid in top_30_ids:
                continue  # entry was in normal pool, not a "rescue" candidate
            score = xenc_map.get(eid, -10.0)
            if score >= cfg.rescue_threshold:
                rl = rescue_lists.setdefault(cid, set())
                if len(rl) < cfg.rescue_max_per_cluster:
                    rl.add(eid)

    return winner_ids, scored


def main() -> None:
    print("=" * 60)
    print("Exploration + rescue list")
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

    # Build 30 clusters (our sweet spot)
    print("Building 30 query clusters...", flush=True)
    cluster_centroids = build_clusters(query_embs, 30)

    print(f"\nCorpus: {len(corpus_ids)} entries")
    print(f"Burn-in: {BURN_IN} steps, Eval: {EVAL_N} queries")
    print(f"Pool: top-{K_SHALLOW} normal, top-{K_EXPLORE} exploration")
    print(f"Configs: {len(CONFIGS)}")
    for c in CONFIGS:
        print(f"  {c.name}: explore_every={c.explore_every}")
    print()

    # Per-config state
    global_states: list[dict[str, float]] = [{} for _ in CONFIGS]
    global_lus: list[dict[str, int]] = [{} for _ in CONFIGS]
    clustered_states: list[ClusteredSuppressionState | None] = [
        ClusteredSuppressionState() if c.n_clusters > 0 else None for c in CONFIGS
    ]
    rescue_lists: list[dict[int, set[str]]] = [defaultdict(set) for _ in CONFIGS]

    # Sampling pattern
    schedule_rng = np.random.default_rng(42)
    n_hot = max(1, int(len(query_ids) * 0.2))
    hot_idx = schedule_rng.choice(len(query_ids), size=n_hot, replace=False)
    hot_ids = [query_ids[i] for i in hot_idx]
    cold_ids = [q for q in query_ids if q not in set(hot_ids)]

    # Phase 1: burn-in (shared xenc over top-80 pool + per-config rescue xenc)
    print(f"Phase 1: {BURN_IN}-step burn-in...", flush=True)

    for step in range(BURN_IN):
        qid = (
            hot_ids[schedule_rng.integers(len(hot_ids))]
            if schedule_rng.random() < 0.7
            else cold_ids[schedule_rng.integers(len(cold_ids))]
        )
        qe = query_embs[qid_to_idx[qid]]
        qt = query_texts.get(qid, "")

        # Shared: hybrid top-80 + xenc all
        raw = hybrid_search_scored(qe, qt, K_EXPLORE, index, bm25, corpus_ids)
        all_pool_ids = [eid for eid, _ in raw]
        top_80_set = set(all_pool_ids)
        pairs = [(qt, corpus_content.get(c, "")) for c in all_pool_ids]
        xenc_scores = xenc.predict(pairs)
        shared_xenc_map = {eid: float(s) for eid, s in zip(all_pool_ids, xenc_scores)}

        # Per-config: augment xenc_map with rescue entries outside top-80
        for ci, cfg in enumerate(CONFIGS):
            xenc_map = dict(shared_xenc_map)
            if cfg.n_clusters > 0:
                cid = assign_cluster(qe, cluster_centroids)
                rescue_outside = [
                    e for e in rescue_lists[ci].get(cid, set()) if e not in top_80_set
                ]
                if rescue_outside:
                    extra_pairs = [(qt, corpus_content.get(e, "")) for e in rescue_outside]
                    extra_scores = xenc.predict(extra_pairs)
                    for e, s in zip(rescue_outside, extra_scores):
                        xenc_map[e] = float(s)
            run_query_for_config(
                step, cfg, qe, raw, xenc_map, cluster_centroids,
                clustered_states[ci], global_states[ci], global_lus[ci],
                rescue_lists[ci],
            )

        if (step + 1) % 200 == 0:
            print(f"  Step {step + 1}:", flush=True)
            for ci, cfg in enumerate(CONFIGS):
                rl = rescue_lists[ci]
                total_rescued = sum(len(s) for s in rl.values())
                clusters_with_rescue = sum(1 for s in rl.values() if s)
                if cfg.explore_every > 0:
                    print(f"    {cfg.name}: {total_rescued} rescued across "
                          f"{clusters_with_rescue} clusters", flush=True)

    print(flush=True)

    # Phase 2: eval on sample
    print(f"Phase 2: Evaluating on {EVAL_N} queries...", flush=True)
    print()

    eval_rng = np.random.default_rng(7)
    eval_qids = eval_rng.choice(query_ids, size=EVAL_N, replace=False)

    config_ndcgs: list[list[float]] = [[] for _ in CONFIGS]
    config_recalls: list[list[float]] = [[] for _ in CONFIGS]

    for qi in eval_qids:
        qr = qrels.get(qi, {})
        if not qr:
            continue
        qe = query_embs[qid_to_idx[qi]]
        qt = query_texts.get(qi, "")
        relevant_set = set(qr.keys())

        raw = hybrid_search_scored(qe, qt, K_EXPLORE, index, bm25, corpus_ids)
        all_pool_ids = [eid for eid, _ in raw]
        top_80_set = set(all_pool_ids)
        pairs = [(qt, corpus_content.get(c, "")) for c in all_pool_ids]
        xenc_scores = xenc.predict(pairs)
        shared_xenc_map = {eid: float(s) for eid, s in zip(all_pool_ids, xenc_scores)}

        for ci, cfg in enumerate(CONFIGS):
            xenc_map = dict(shared_xenc_map)
            if cfg.n_clusters > 0:
                cid = assign_cluster(qe, cluster_centroids)
                rescue_outside = [
                    e for e in rescue_lists[ci].get(cid, set()) if e not in top_80_set
                ]
                if rescue_outside:
                    extra_pairs = [(qt, corpus_content.get(e, "")) for e in rescue_outside]
                    extra_scores = xenc.predict(extra_pairs)
                    for e, s in zip(rescue_outside, extra_scores):
                        xenc_map[e] = float(s)

            # Use a step that won't trigger any exploration modulus
            _, scored = run_query_for_config(
                step=-1, cfg=cfg, qe=qe, raw=raw, xenc_map=xenc_map,
                cluster_centroids=cluster_centroids,
                clustered_state=clustered_states[ci],
                global_state=global_states[ci],
                global_lu=global_lus[ci],
                rescue_lists=rescue_lists[ci],
            )
            top10 = [c for c, _ in scored[:K_FINAL]]
            config_ndcgs[ci].append(ndcg_at_k(top10, qr, K_FINAL))
            # Recall@30: relevant entries making it into the eval candidate set
            candidate_ids_for_eval = [eid for eid, _ in scored[:K_SHALLOW]]
            r = len(relevant_set & set(candidate_ids_for_eval)) / len(relevant_set) if relevant_set else 0
            config_recalls[ci].append(r)

    baseline_ndcg = np.mean(config_ndcgs[0])
    baseline_recall = np.mean(config_recalls[0])

    print(f"{'Config':<20} | {'NDCG@10':>8} | {'Δ':>7} | {'Recall@30':>9} | {'Δ':>7} | {'Rescued':>8}")
    print("-" * 80)

    md_rows = []
    for ci, cfg in enumerate(CONFIGS):
        ndcg = np.mean(config_ndcgs[ci])
        recall = np.mean(config_recalls[ci])
        d_ndcg = (ndcg - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg > 0 else 0
        d_recall = (recall - baseline_recall) / baseline_recall * 100 if baseline_recall > 0 else 0
        rescued = sum(len(s) for s in rescue_lists[ci].values())
        print(f"{cfg.name:<20} | {ndcg:>8.4f} | {d_ndcg:>+6.1f}% | {recall:>9.4f} | {d_recall:>+6.1f}% | {rescued:>8}")
        md_rows.append((cfg.name, cfg.explore_every, ndcg, d_ndcg, recall, d_recall, rescued))

    n_eval = len(config_ndcgs[0])
    print(f"\nQueries evaluated: {n_eval}")

    print(f"\nWriting {RESULTS}...", flush=True)
    with open(RESULTS, "w") as f:
        f.write("# Exploration + Rescue List Benchmark (fast)\n\n")
        f.write("Symmetric mechanism: RIF pushes false positives down, rescue list pulls false negatives up.\n\n")
        f.write(f"Burn-in: {BURN_IN} steps, Eval: {n_eval} queries (random sample)\n")
        f.write(f"Pool: top-{K_SHALLOW} normal, top-{K_EXPLORE} exploration zone\n")
        f.write(f"RIF params: clustered30+gap (current best)\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("| Config | explore_every | NDCG@10 | Δ | Recall@30 | Δ | Rescued |\n")
        f.write("|--------|---------------|---------|---|-----------|---|---------|\n")
        for name, ee, ndcg, d_ndcg, recall, d_recall, rescued in md_rows:
            f.write(f"| {name} | {ee if ee > 0 else '—'} | {ndcg:.4f} | "
                    f"{d_ndcg:+.1f}% | {recall:.4f} | {d_recall:+.1f}% | {rescued} |\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

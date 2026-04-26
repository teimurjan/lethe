"""Extended metrics: baseline vs clustered+gap RIF (checkpoint 13 winner).

Adds five behavior metrics on top of NDCG:
- exact_episode_accuracy: top-1 is in qrels.
- sibling_confusion_rate: top-1 is from the right session but wrong turn.
- wrong_family_retrieval: top-1 is from a session with no relevant turns.
- abstention_rate: top-1 xenc score below threshold (low-confidence retrieval).
- stale_fact_retrieval: for knowledge-update questions only — top-1 from
  answer-session but not in qrels (proxy for pulling an older version).

All retrievers return top-10 from the same BM25+FAISS+xenc pipeline;
RIF only changes which 30 entries reach the xenc reranker.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
from datasets import load_dataset  # type: ignore[import-untyped]
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
RESULTS = Path("BENCHMARKS_RIF_EXTENDED.md")

BURN_IN = 5000
K_SHALLOW = 30
K_FINAL = 10
N_CLUSTERS = 30
ABSTAIN_THRESHOLDS = (0.0, 2.0, 4.0)

RIF = RIFConfig(
    alpha=0.3, suppression_rate=0.1, reinforcement_rate=0.05,
    decay_lambda=0.005, use_rank_gap=True,
)


def hybrid_search_scored(
    qe: np.ndarray, qt: str, k: int,
    index: faiss.IndexFlatIP, bm25: BM25Okapi, corpus_ids: list[str],
) -> list[tuple[str, float]]:
    D, I = index.search(qe.reshape(1, -1), k)
    vec_results = [(corpus_ids[i], float(D[0][rank]))
                   for rank, i in enumerate(I[0]) if i >= 0]
    tokens = qt.lower().split()
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
    print("=" * 80)
    print("Extended metrics: baseline vs clustered+gap RIF")
    print("=" * 80)
    print()

    # --- Load ---
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
    with open(DATA / "longmemeval_meta.json") as f:
        corpus_meta = json.load(f)  # entry_id → {session_id, turn_idx}

    # Question types from source dataset
    ds = load_dataset(
        "xiaowu0162/longmemeval-cleaned",
        data_files="longmemeval_s_cleaned.json",
        split="train",
    )
    question_types: dict[str, str] = {row["question_id"]: row["question_type"] for row in ds}

    # --- Setup ---
    xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    index = faiss.IndexFlatIP(384)
    index.add(corpus_embs)
    tokenized = [corpus_content.get(cid, "").lower().split() for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    qid_to_idx = {q: i for i, q in enumerate(query_ids)}

    print("Building 30 query clusters...", flush=True)
    centroids = build_clusters(query_embs, N_CLUSTERS)

    # session lookup
    def session_of(eid: str) -> str:
        return corpus_meta.get(eid, {}).get("session_id", "")

    # relevant sessions per query
    def rel_sessions(qid: str) -> set[str]:
        return {session_of(eid) for eid in qrels.get(qid, {}).keys()}

    print(f"Corpus: {len(corpus_ids)}, Queries: {len(query_ids)}")
    print(f"RIF: clustered30+gap ({BURN_IN}-step burn-in)")
    print()

    # --- Phase 1: burn-in for RIF ---
    clustered_state = ClusteredSuppressionState()
    schedule_rng = np.random.default_rng(42)
    n_hot = max(1, int(len(query_ids) * 0.2))
    hot_idx = schedule_rng.choice(len(query_ids), size=n_hot, replace=False)
    hot_ids = [query_ids[i] for i in hot_idx]
    cold_ids = [q for q in query_ids if q not in set(hot_ids)]

    print(f"Phase 1: {BURN_IN}-step RIF burn-in...", flush=True)
    t0 = time.time()

    for step in range(BURN_IN):
        qid = (
            hot_ids[schedule_rng.integers(len(hot_ids))]
            if schedule_rng.random() < 0.7
            else cold_ids[schedule_rng.integers(len(cold_ids))]
        )
        qe = query_embs[qid_to_idx[qid]]
        qt = query_texts.get(qid, "")

        raw = hybrid_search_scored(qe, qt, K_SHALLOW, index, bm25, corpus_ids)
        cid = assign_cluster(qe, centroids)
        supp = clustered_state.get_cluster_scores(cid)
        lu = clustered_state.get_cluster_last_updated(cid)

        adjusted = apply_suppression_penalty(raw, supp, RIF.alpha)
        candidate_ids = [eid for eid, _ in adjusted[:K_SHALLOW]]
        pairs = [(qt, corpus_content.get(c, "")) for c in candidate_ids]
        xenc_scores = xenc.predict(pairs)
        scored = sorted(zip(candidate_ids, xenc_scores), key=lambda x: x[1], reverse=True)

        winner_ids = {eid for eid, _ in scored[:K_FINAL]}
        rank_lookup = {eid: r for r, (eid, _) in enumerate(adjusted)}
        xenc_rank_lookup = {eid: r for r, (eid, _) in enumerate(scored)}
        xenc_map = {eid: float(s) for eid, s in scored}
        competitor_data = [
            (eid, rank_lookup.get(eid, len(adjusted)),
             xenc_rank_lookup.get(eid, len(scored)),
             xenc_map.get(eid, 0.0))
            for eid in candidate_ids
        ]
        updates = update_suppression(
            winner_ids, competitor_data, supp, len(candidate_ids), RIF, step, lu,
        )
        clustered_state.update_cluster(cid, updates, step)

        if (step + 1) % 1000 == 0:
            print(f"  step {step+1}/{BURN_IN} (elapsed {(time.time()-t0)/60:.1f}m)", flush=True)

    print(flush=True)

    # --- Phase 2: evaluate both systems with shared xenc on top-80 ---
    print("Phase 2: evaluating on all queries...", flush=True)

    def blank_metrics() -> dict[str, list[float]]:
        return defaultdict(list)

    baseline_m: dict[str, dict[str, list[float]]] = defaultdict(blank_metrics)
    rif_m: dict[str, dict[str, list[float]]] = defaultdict(blank_metrics)

    for qi in query_ids:
        qr = qrels.get(qi, {})
        if not qr:
            continue
        qtype = question_types.get(qi, "unknown")
        qe = query_embs[qid_to_idx[qi]]
        qt = query_texts.get(qi, "")
        rel_ids = set(qr.keys())
        rel_sess = rel_sessions(qi)

        # Shared hybrid search + xenc on top-80 so both systems see same scores.
        raw = hybrid_search_scored(qe, qt, 80, index, bm25, corpus_ids)
        pool_ids = [eid for eid, _ in raw]
        pairs = [(qt, corpus_content.get(c, "")) for c in pool_ids]
        xenc_scores = xenc.predict(pairs)
        xenc_map = {eid: float(s) for eid, s in zip(pool_ids, xenc_scores)}

        # Apply RIF suppression for clustered+gap system
        cid = assign_cluster(qe, centroids)
        supp = clustered_state.get_cluster_scores(cid)

        # Baseline: no suppression (alpha=0)
        baseline_adjusted = list(raw)  # already sorted by RRF
        # RIF: suppression-penalized
        rif_adjusted = apply_suppression_penalty(list(raw), supp, RIF.alpha)

        def score_top1(adjusted: list[tuple[str, float]]) -> tuple[str, float, list[str]]:
            cand = [eid for eid, _ in adjusted[:K_SHALLOW]]
            cand_scores = sorted(
                [(eid, xenc_map.get(eid, -10.0)) for eid in cand],
                key=lambda x: x[1], reverse=True,
            )
            top_id, top_score = cand_scores[0]
            top10 = [c for c, _ in cand_scores[:K_FINAL]]
            return top_id, top_score, top10

        for name, adjusted, bucket in (
            ("baseline", baseline_adjusted, baseline_m),
            ("rif", rif_adjusted, rif_m),
        ):
            top_id, top_score, top10 = score_top1(adjusted)
            top_sess = session_of(top_id)

            # Metrics
            p1 = 1.0 if top_id in rel_ids else 0.0
            ndcg = ndcg_at_k(top10, qr, K_FINAL)
            sibling_confuse = 1.0 if (top_id not in rel_ids and top_sess in rel_sess) else 0.0
            wrong_family = 1.0 if top_sess and top_sess not in rel_sess else 0.0

            for threshold in ABSTAIN_THRESHOLDS:
                abstain = 1.0 if top_score < threshold else 0.0
                bucket["overall"][f"abstain@{threshold}"].append(abstain)
                bucket[qtype][f"abstain@{threshold}"].append(abstain)

            for b in (bucket["overall"], bucket[qtype]):
                b["exact_episode"].append(p1)
                b["ndcg@10"].append(ndcg)
                b["sibling_confusion"].append(sibling_confuse)
                b["wrong_family"].append(wrong_family)

            # Stale-fact proxy: knowledge-update only
            if qtype == "knowledge-update":
                stale = 1.0 if (top_id not in rel_ids and top_sess in rel_sess) else 0.0
                for b in (bucket["overall"], bucket[qtype]):
                    b["stale_fact"].append(stale)

    # --- Report ---
    def mean_or_nan(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    metric_order = [
        "exact_episode", "ndcg@10", "sibling_confusion",
        "wrong_family", "stale_fact",
        f"abstain@{ABSTAIN_THRESHOLDS[0]}", f"abstain@{ABSTAIN_THRESHOLDS[1]}",
        f"abstain@{ABSTAIN_THRESHOLDS[2]}",
    ]

    def fmt_row(label: str, bl: dict[str, list[float]], rf: dict[str, list[float]]) -> None:
        for m in metric_order:
            b = mean_or_nan(bl.get(m, []))
            r = mean_or_nan(rf.get(m, []))
            if np.isnan(b) and np.isnan(r):
                continue
            delta = r - b
            print(f"  {label:<24} | {m:<20} | {b:>7.3f} | {r:>7.3f} | {delta:>+7.3f}")

    print()
    print("=" * 80)
    print(f"{'bucket':<24} | {'metric':<20} | {'base':>7} | {'RIF':>7} | {'Δ':>7}")
    print("-" * 80)
    fmt_row("overall", baseline_m["overall"], rif_m["overall"])
    print("-" * 80)
    for qtype in sorted(k for k in baseline_m.keys() if k != "overall"):
        fmt_row(qtype, baseline_m[qtype], rif_m[qtype])
        print("-" * 80)

    # Write markdown
    print(f"\nWriting {RESULTS}...", flush=True)
    with open(RESULTS, "w") as f:
        f.write("# Extended metrics: baseline vs clustered+gap RIF\n\n")
        f.write(f"Burn-in: {BURN_IN} steps, Eval: all queries with qrels\n")
        f.write(f"Config: clustered30+gap, alpha={RIF.alpha}, rate={RIF.suppression_rate}, decay={RIF.decay_lambda}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        f.write("Metrics:\n")
        f.write("- **exact_episode**: top-1 in qrels.\n")
        f.write("- **ndcg@10**: standard IR ranking quality.\n")
        f.write("- **sibling_confusion**: top-1 is from an answer-session but not in qrels (same-topic, wrong turn).\n")
        f.write("- **wrong_family**: top-1 is from a session with no relevant turns (unrelated topic).\n")
        f.write("- **stale_fact** (knowledge-update only): top-1 is from answer-session but not in qrels — proxy for pulling an older version of the fact.\n")
        f.write(f"- **abstain@T**: fraction of queries where top-1 xenc score is below threshold T (would trigger abstention).\n\n")
        buckets = ["overall"] + sorted(k for k in baseline_m.keys() if k != "overall")
        for bucket in buckets:
            f.write(f"## {bucket}\n\n")
            f.write("| metric | baseline | RIF | Δ |\n")
            f.write("|--------|----------|-----|---|\n")
            for m in metric_order:
                b = mean_or_nan(baseline_m[bucket].get(m, []))
                r = mean_or_nan(rif_m[bucket].get(m, []))
                if np.isnan(b) and np.isnan(r):
                    continue
                f.write(f"| {m} | {b:.3f} | {r:.3f} | {r-b:+.3f} |\n")
            f.write("\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

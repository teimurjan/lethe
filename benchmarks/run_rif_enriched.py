"""Three-arm benchmark: baseline vs checkpoint 13 vs checkpoint 13 + LLM enrichment.

Arms share the same BM25 + vector + cross-encoder pipeline; they differ in:
- Whether RIF suppression is applied (baseline: off; checkpoints 13 & enriched: on).
- Whether BM25/vector indexes use enriched text (only the third arm).

Enriched text = original + gist + anticipated_queries + entities + temporal_markers.
The cross-encoder still scores against ORIGINAL text (what a user would see).

Reports the full extended metrics suite:
- exact_episode, ndcg@10, sibling_confusion, wrong_family, stale_fact (knowledge-update only)
- abstain@T for T in {0, 2, 4}
- Per-question-type breakdown.

Usage:
  uv run python experiments/enrich_longmemeval.py  # one-time: build enrichments
  uv run python benchmarks/run_rif_enriched.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
from datasets import load_dataset  # type: ignore[import-untyped]
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gc_memory.enrichment import load_enrichments  # noqa: E402
from gc_memory.metrics import ndcg_at_k  # noqa: E402
from gc_memory.rif import (  # noqa: E402
    ClusteredSuppressionState,
    RIFConfig,
    apply_suppression_penalty,
    assign_cluster,
    build_clusters,
    update_suppression,
)

DATA = Path("data")
ENRICHED_PATH = DATA / "longmemeval_enriched.jsonl"
RESULTS = Path("BENCHMARKS_RIF_ENRICHED.md")

BURN_IN = 5000
K_SHALLOW = 30
K_FINAL = 10
N_CLUSTERS = 30
ABSTAIN_THRESHOLDS = (0.0, 2.0, 4.0)

RIF = RIFConfig(
    alpha=0.3, suppression_rate=0.1, reinforcement_rate=0.05,
    decay_lambda=0.005, use_rank_gap=True,
)


def build_text_for_index(
    entry_id: str,
    original: dict[str, str],
    enriched: dict,  # entry_id -> Enrichment
) -> str:
    """For enriched arm: concat original + enrichment fields."""
    base = original.get(entry_id, "")
    enr = enriched.get(entry_id)
    if enr is None:
        return base
    return f"{base} {enr.as_search_text()}"


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


def build_enriched_index(
    corpus_ids: list[str],
    corpus_content: dict[str, str],
    enriched: dict,
) -> tuple[faiss.IndexFlatIP, BM25Okapi, np.ndarray]:
    """Build FAISS + BM25 over enriched text. Re-encodes with MiniLM."""
    print("  Building enriched index (re-encoding with MiniLM)...", flush=True)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = [build_text_for_index(eid, corpus_content, enriched) for eid in corpus_ids]
    t0 = time.time()
    embs = encoder.encode(
        texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256,
    ).astype(np.float32)
    print(f"  Encoded {len(texts):,} docs in {time.time()-t0:.1f}s", flush=True)
    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    return idx, bm25, embs


def main() -> None:
    print("=" * 80)
    print("Three-arm benchmark: baseline vs RIF vs RIF + LLM enrichment")
    print("=" * 80)
    print()

    if not ENRICHED_PATH.exists():
        print(f"ERROR: {ENRICHED_PATH} not found.", file=sys.stderr)
        print("Run `uv run python experiments/enrich_longmemeval.py` first.",
              file=sys.stderr)
        sys.exit(1)

    # --- Load LongMemEval ---
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
        corpus_meta = json.load(f)

    ds = load_dataset(
        "xiaowu0162/longmemeval-cleaned",
        data_files="longmemeval_s_cleaned.json",
        split="train",
    )
    question_types: dict[str, str] = {row["question_id"]: row["question_type"] for row in ds}

    # --- Load enrichments ---
    enriched = load_enrichments(ENRICHED_PATH)
    print(f"Loaded {len(enriched):,} enrichments from {ENRICHED_PATH}")
    coverage = len(enriched) / len(corpus_ids) * 100
    print(f"Corpus coverage: {coverage:.1f}% ({len(enriched):,} / {len(corpus_ids):,})")
    print()

    # --- Build indexes ---
    xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # Original index (for baseline + RIF-only arms)
    index_orig = faiss.IndexFlatIP(384)
    index_orig.add(corpus_embs)
    bm25_orig = BM25Okapi([corpus_content.get(cid, "").lower().split() for cid in corpus_ids])
    # Enriched index (for third arm)
    index_enr, bm25_enr, _ = build_enriched_index(corpus_ids, corpus_content, enriched)

    qid_to_idx = {q: i for i, q in enumerate(query_ids)}
    centroids = build_clusters(query_embs, N_CLUSTERS)

    def session_of(eid: str) -> str:
        return corpus_meta.get(eid, {}).get("session_id", "")

    def rel_sessions(qid: str) -> set[str]:
        return {session_of(eid) for eid in qrels.get(qid, {}).keys()}

    # Queries whose qrels have at least one enriched entry — where the
    # enrichment mechanism actually has something to bite on.
    enriched_ids_set = set(enriched.keys())
    covered_query_ids = {
        qid for qid, relmap in qrels.items()
        if set(relmap.keys()) & enriched_ids_set
    }
    print(f"Coverage: {len(covered_query_ids)}/{len(qrels)} queries have at least "
          f"one enriched qrels entry.")
    print()

    # --- Phase 1: burn-in RIF suppression (separately for each arm that uses RIF) ---
    # The enriched arm has a different candidate pool, so it needs its own burn-in state.
    rif_state_orig = ClusteredSuppressionState()
    rif_state_enr = ClusteredSuppressionState()

    schedule_rng = np.random.default_rng(42)
    n_hot = max(1, int(len(query_ids) * 0.2))
    hot_idx = schedule_rng.choice(len(query_ids), size=n_hot, replace=False)
    hot_ids = [query_ids[i] for i in hot_idx]
    cold_ids = [q for q in query_ids if q not in set(hot_ids)]

    print(f"Phase 1: {BURN_IN}-step RIF burn-in (two states: orig and enriched)...",
          flush=True)
    t0 = time.time()

    def run_burn_in(state, index, bm25):
        local_rng = np.random.default_rng(42)
        for step in range(BURN_IN):
            qid = (
                hot_ids[local_rng.integers(len(hot_ids))]
                if local_rng.random() < 0.7
                else cold_ids[local_rng.integers(len(cold_ids))]
            )
            qe = query_embs[qid_to_idx[qid]]
            qt = query_texts.get(qid, "")

            raw = hybrid_search_scored(qe, qt, K_SHALLOW, index, bm25, corpus_ids)
            cid = assign_cluster(qe, centroids)
            supp = state.get_cluster_scores(cid)
            lu = state.get_cluster_last_updated(cid)

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
            state.update_cluster(cid, updates, step)

            if (step + 1) % 1000 == 0:
                print(f"    step {step+1}/{BURN_IN} (elapsed {(time.time()-t0)/60:.1f}m)",
                      flush=True)

    print("  Burn-in for RIF-only arm (original index)...")
    run_burn_in(rif_state_orig, index_orig, bm25_orig)
    print("  Burn-in for RIF + enrichment arm (enriched index)...")
    run_burn_in(rif_state_enr, index_enr, bm25_enr)
    print(f"  Burn-in complete in {(time.time()-t0)/60:.1f}m total", flush=True)
    print()

    # --- Phase 2: eval ---
    def blank_metrics() -> dict[str, list[float]]:
        return defaultdict(list)

    arms = {
        "baseline": defaultdict(blank_metrics),
        "rif":      defaultdict(blank_metrics),
        "enriched": defaultdict(blank_metrics),
    }

    print("Phase 2: evaluating on all queries...", flush=True)
    for qi in query_ids:
        qr = qrels.get(qi, {})
        if not qr:
            continue
        qtype = question_types.get(qi, "unknown")
        qe = query_embs[qid_to_idx[qi]]
        qt = query_texts.get(qi, "")
        rel_ids = set(qr.keys())
        rel_sess = rel_sessions(qi)

        # --- Baseline & RIF share the ORIGINAL index ---
        raw_orig = hybrid_search_scored(qe, qt, 80, index_orig, bm25_orig, corpus_ids)
        pool_orig_ids = [eid for eid, _ in raw_orig]
        pairs_orig = [(qt, corpus_content.get(c, "")) for c in pool_orig_ids]
        xenc_orig = xenc.predict(pairs_orig)
        xmap_orig = {eid: float(s) for eid, s in zip(pool_orig_ids, xenc_orig)}

        cid = assign_cluster(qe, centroids)
        supp_orig = rif_state_orig.get_cluster_scores(cid)

        # --- Enriched arm uses the ENRICHED index; xenc still scores ORIGINAL text ---
        raw_enr = hybrid_search_scored(qe, qt, 80, index_enr, bm25_enr, corpus_ids)
        pool_enr_ids = [eid for eid, _ in raw_enr]
        pairs_enr = [(qt, corpus_content.get(c, "")) for c in pool_enr_ids]
        xenc_enr = xenc.predict(pairs_enr)
        xmap_enr = {eid: float(s) for eid, s in zip(pool_enr_ids, xenc_enr)}
        supp_enr = rif_state_enr.get_cluster_scores(cid)

        arm_configs = [
            ("baseline", raw_orig, xmap_orig, None),
            ("rif",      raw_orig, xmap_orig, supp_orig),
            ("enriched", raw_enr,  xmap_enr,  supp_enr),
        ]

        for arm_name, raw, xmap, supp in arm_configs:
            if supp is None:
                adjusted = list(raw)
            else:
                adjusted = apply_suppression_penalty(list(raw), supp, RIF.alpha)
            cand = [eid for eid, _ in adjusted[:K_SHALLOW]]
            cand_scores = sorted(
                [(eid, xmap.get(eid, -10.0)) for eid in cand],
                key=lambda x: x[1], reverse=True,
            )
            top_id, top_score = cand_scores[0]
            top10 = [c for c, _ in cand_scores[:K_FINAL]]
            top_sess = session_of(top_id)

            p1 = 1.0 if top_id in rel_ids else 0.0
            ndcg = ndcg_at_k(top10, qr, K_FINAL)
            sib = 1.0 if (top_id not in rel_ids and top_sess in rel_sess) else 0.0
            wfam = 1.0 if top_sess and top_sess not in rel_sess else 0.0

            bucket = arms[arm_name]
            is_covered = qi in covered_query_ids
            target_buckets = [bucket["overall"], bucket[qtype]]
            if is_covered:
                target_buckets.append(bucket["covered"])
                target_buckets.append(bucket[f"covered::{qtype}"])
            else:
                target_buckets.append(bucket["uncovered"])

            for b in target_buckets:
                b["exact_episode"].append(p1)
                b["ndcg@10"].append(ndcg)
                b["sibling_confusion"].append(sib)
                b["wrong_family"].append(wfam)
            if qtype == "knowledge-update":
                stale = 1.0 if (top_id not in rel_ids and top_sess in rel_sess) else 0.0
                for b in target_buckets:
                    b["stale_fact"].append(stale)
            for threshold in ABSTAIN_THRESHOLDS:
                abstain = 1.0 if top_score < threshold else 0.0
                for b in target_buckets:
                    b[f"abstain@{threshold}"].append(abstain)

    # --- Report ---
    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    metric_order = [
        "exact_episode", "ndcg@10", "sibling_confusion", "wrong_family",
        "stale_fact",
        f"abstain@{ABSTAIN_THRESHOLDS[0]}",
        f"abstain@{ABSTAIN_THRESHOLDS[1]}",
        f"abstain@{ABSTAIN_THRESHOLDS[2]}",
    ]

    print()
    print("=" * 90)
    print(f"{'bucket':<26} | {'metric':<20} | {'base':>7} | {'RIF':>7} | {'+enr':>7}")
    print("-" * 90)
    # Order: overall → covered/uncovered → per-qtype → covered::qtype
    all_buckets = list(arms["baseline"].keys())
    priority = ["overall", "covered", "uncovered"]
    others = sorted(b for b in all_buckets if b not in priority)
    buckets_order = [b for b in priority if b in all_buckets] + others
    for bucket in buckets_order:
        for m in metric_order:
            b = mean(arms["baseline"][bucket].get(m, []))
            r = mean(arms["rif"][bucket].get(m, []))
            e = mean(arms["enriched"][bucket].get(m, []))
            if np.isnan(b) and np.isnan(r) and np.isnan(e):
                continue
            print(f"  {bucket:<24} | {m:<20} | {b:>7.3f} | {r:>7.3f} | {e:>7.3f}")
        print("-" * 90)

    # Markdown
    print(f"\nWriting {RESULTS}...", flush=True)
    with open(RESULTS, "w") as f:
        f.write("# Three-arm benchmark: baseline vs checkpoint 13 vs checkpoint 13 + enrichment\n\n")
        f.write(f"Burn-in: {BURN_IN} steps per RIF arm\n")
        f.write(f"Coverage: {len(enriched):,} / {len(corpus_ids):,} "
                f"({coverage:.1f}%) corpus enriched\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d')}\n\n")
        for bucket in buckets_order:
            f.write(f"## {bucket}\n\n")
            f.write("| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |\n")
            f.write("|--------|----------|-----|--------------|-----------|\n")
            for m in metric_order:
                b = mean(arms["baseline"][bucket].get(m, []))
                r = mean(arms["rif"][bucket].get(m, []))
                e = mean(arms["enriched"][bucket].get(m, []))
                if np.isnan(b) and np.isnan(r) and np.isnan(e):
                    continue
                f.write(f"| {m} | {b:.3f} | {r:.3f} | {e:.3f} | {e-r:+.3f} |\n")
            f.write("\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

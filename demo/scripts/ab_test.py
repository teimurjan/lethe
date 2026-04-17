"""Fast A/B: benchmark primitives vs MemoryStore on the same 200 queries.

Runs both paths on identical data and schedule, prints per-query NDCG
diff so we can pinpoint where MemoryStore diverges from the published
+6.5% benchmark result.

Usage:
    uv run python demo/scripts/ab_test.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss  # noqa: E402 — before torch
import numpy as np  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

from benchmarks._lib.metrics import ndcg_at_k  # noqa: E402
from lethe.entry import MemoryEntry, Tier  # noqa: E402
from lethe.memory_store import MemoryStore  # noqa: E402
from lethe.rif import (  # noqa: E402
    ClusteredSuppressionState,
    RIFConfig,
    apply_suppression_penalty,
    assign_cluster,
    build_clusters,
    update_suppression,
)

DATA = REPO / "data"
N = 1000
K_SHALLOW = 30
K_FINAL = 10
SEED = 42
N_CLUSTERS = 30


def hybrid_search_scored(qe, qt, k, index, bm25, corpus_ids):
    D, I = index.search(qe.reshape(1, -1), k)
    vec = [(corpus_ids[i], float(D[0][r])) for r, i in enumerate(I[0]) if i >= 0]
    tokens = qt.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    bm25_r = [(corpus_ids[i], float(scores[i])) for i in top_idx]
    rrf: dict[str, float] = {}
    for r, (eid, _) in enumerate(vec):
        rrf[eid] = rrf.get(eid, 0.0) + 1.0 / (r + 60)
    for r, (eid, _) in enumerate(bm25_r):
        rrf[eid] = rrf.get(eid, 0.0) + 1.0 / (r + 60)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


def main() -> None:
    print("Loading data...", flush=True)
    data = np.load(str(DATA / "longmemeval_prepared.npz"), allow_pickle=True)
    corpus_ids = [str(c) for c in data["corpus_ids"].tolist()]
    corpus_embs = data["corpus_embeddings"].astype(np.float32)
    query_ids_all = [str(q) for q in data["query_ids"].tolist()]
    query_embs = data["query_embeddings"].astype(np.float32)
    qrels = json.loads((DATA / "longmemeval_qrels.json").read_text())
    corpus_content = json.loads((DATA / "longmemeval_corpus.json").read_text())
    query_texts = json.loads((DATA / "longmemeval_queries.json").read_text())
    eval_ids = [q for q in query_ids_all if qrels.get(q)]
    qid_to_idx = {q: i for i, q in enumerate(query_ids_all)}

    # Build cluster centroids from query embeddings (matching benchmark exactly)
    print(f"Building {N_CLUSTERS} clusters from {len(eval_ids)} query embeddings...", flush=True)
    all_qembs = np.stack([query_embs[qid_to_idx[q]] for q in eval_ids])
    centroids = build_clusters(all_qembs.astype(np.float32), N_CLUSTERS)

    # Schedule
    rng = np.random.default_rng(SEED)
    n_hot = max(1, int(len(eval_ids) * 0.2))
    hot_idx = rng.choice(len(eval_ids), size=n_hot, replace=False)
    hot_set = {eval_ids[i] for i in hot_idx}
    hot_ids = sorted(hot_set)
    cold_ids = [q for q in eval_ids if q not in hot_set]
    schedule = []
    for _ in range(N):
        if rng.random() < 0.7:
            schedule.append(hot_ids[rng.integers(len(hot_ids))])
        else:
            schedule.append(cold_ids[rng.integers(len(cold_ids))])

    print("Loading cross-encoder...", flush=True)
    xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Path A: benchmark primitives (known to give +6.5%) ---
    print("\n=== Path A: Benchmark primitives ===", flush=True)
    faiss_idx = faiss.IndexFlatIP(384)
    faiss_idx.add(corpus_embs)
    tokenized = [corpus_content.get(cid, "").lower().split() for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)

    cfg = RIFConfig(alpha=0.3, use_rank_gap=True, suppression_rate=0.1)
    cs = ClusteredSuppressionState()

    a_baseline_ndcgs = []
    a_lethe_ndcgs = []
    for step, qid in enumerate(schedule):
        qe = query_embs[qid_to_idx[qid]]
        qt = query_texts.get(qid, "")
        qr = qrels.get(qid, {})

        raw = hybrid_search_scored(qe, qt, K_SHALLOW, faiss_idx, bm25, corpus_ids)
        pool_ids = [eid for eid, _ in raw]
        pairs = [(qt, corpus_content.get(c, "")) for c in pool_ids]
        xenc_scores = xenc.predict(pairs, show_progress_bar=False)
        xenc_map = {eid: float(s) for eid, s in zip(pool_ids, xenc_scores)}

        # Baseline (no RIF)
        scored_base = sorted(
            [(eid, xenc_map.get(eid, -10.0)) for eid in pool_ids[:K_SHALLOW]],
            key=lambda x: x[1], reverse=True,
        )
        a_baseline_ndcgs.append(ndcg_at_k([e for e, _ in scored_base[:K_FINAL]], qr, K_FINAL))

        # Lethe (clustered RIF)
        cid = assign_cluster(qe, centroids)
        supp = cs.get_cluster_scores(cid)
        lu = cs.get_cluster_last_updated(cid)
        adjusted = apply_suppression_penalty(raw, supp, cfg.alpha)
        cand_ids = [eid for eid, _ in adjusted[:K_SHALLOW]]
        scored = sorted(
            [(eid, xenc_map.get(eid, -10.0)) for eid in cand_ids],
            key=lambda x: x[1], reverse=True,
        )
        top_ids = [e for e, _ in scored[:K_FINAL]]
        a_lethe_ndcgs.append(ndcg_at_k(top_ids, qr, K_FINAL))

        # RIF update
        winner_ids = {eid for eid, _ in scored[:K_FINAL]}
        rank_lookup = {e: r for r, (e, _) in enumerate(adjusted)}
        xrank_lookup = {e: r for r, (e, _) in enumerate(scored)}
        cdata = [
            (e, rank_lookup.get(e, len(adjusted)),
             xrank_lookup.get(e, len(scored)),
             xenc_map.get(e, 0.0))
            for e in cand_ids
        ]
        updates = update_suppression(winner_ids, cdata, supp, len(cand_ids), cfg, step, lu)
        cs.update_cluster(cid, updates, step)

    a_b_mean = np.mean(a_baseline_ndcgs)
    a_l_mean = np.mean(a_lethe_ndcgs)
    print(f"  baseline={a_b_mean:.4f}  lethe={a_l_mean:.4f}  delta={a_l_mean-a_b_mean:+.4f} ({(a_l_mean-a_b_mean)/a_b_mean*100:+.1f}%)")

    # --- Path B: MemoryStore ---
    print("\n=== Path B: MemoryStore ===", flush=True)

    class _Enc:
        def __init__(self, st):
            self._st = st
        def encode(self, text, **kw):
            return self._st.encode(text, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    bi = _Enc(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(
            path=Path(tmp) / "store",
            bi_encoder=bi, cross_encoder=xenc,
            dim=384, k_shallow=30, k_deep=30,
            rif_config=RIFConfig(alpha=0.3, use_rank_gap=True, suppression_rate=0.1, n_clusters=N_CLUSTERS),
        )
        # Bulk populate
        dim = corpus_embs.shape[1]
        for i, eid in enumerate(corpus_ids):
            emb = corpus_embs[i]
            store.entries[eid] = MemoryEntry(
                id=eid, content=corpus_content.get(eid, ""),
                base_embedding=emb.copy(), embedding=emb.copy(),
                adapter=np.zeros(dim, dtype=np.float32), tier=Tier.NAIVE,
            )
            store._embeddings[eid] = emb
        store._rebuild_index()

        # Inject the SAME centroids as Path A (bypass the query-buffer bootstrap)
        store._cluster_centroids = centroids
        store._cluster_dirty = False

        b_baseline_ndcgs = a_baseline_ndcgs  # same baseline (deterministic, no RIF)
        b_lethe_ndcgs = []
        for step, qid in enumerate(schedule):
            qt = query_texts.get(qid, "")
            qr = qrels.get(qid, {})
            hits = store.retrieve(qt, k=K_FINAL)
            b_lethe_ndcgs.append(ndcg_at_k([h[0] for h in hits], qr, K_FINAL))

        store.close()

    b_l_mean = np.mean(b_lethe_ndcgs)
    print(f"  baseline={a_b_mean:.4f}  lethe={b_l_mean:.4f}  delta={b_l_mean-a_b_mean:+.4f} ({(b_l_mean-a_b_mean)/a_b_mean*100:+.1f}%)")

    # --- Diff ---
    print("\n=== Comparison ===")
    print(f"  Path A (primitives): delta={a_l_mean-a_b_mean:+.4f} ({(a_l_mean-a_b_mean)/a_b_mean*100:+.1f}%)")
    print(f"  Path B (MemoryStore): delta={b_l_mean-a_b_mean:+.4f} ({(b_l_mean-a_b_mean)/a_b_mean*100:+.1f}%)")
    print(f"  Gap: {(a_l_mean - b_l_mean):+.4f}")

    # Per-query diff
    diffs = [a - b for a, b in zip(a_lethe_ndcgs, b_lethe_ndcgs)]
    n_same = sum(1 for d in diffs if abs(d) < 0.001)
    n_a_wins = sum(1 for d in diffs if d > 0.001)
    n_b_wins = sum(1 for d in diffs if d < -0.001)
    print(f"  Per-query: {n_same} same, {n_a_wins} A>B, {n_b_wins} B>A")
    if n_a_wins + n_b_wins > 0:
        print(f"  Mean |diff| on divergent queries: {np.mean([abs(d) for d in diffs if abs(d) > 0.001]):.4f}")


if __name__ == "__main__":
    main()

"""Sweep the adaptive deep-pass cap to measure the NDCG-vs-latency tradeoff.

Mirrors the candidate-gather + RRF-merge + rerank pipeline from
lethe.union_store / lethe.memory_store so results are representative of
the production deep pass.

Pipeline (single project, standing in for the cross-project case):
  1. Shallow: BM25 top-30 + FAISS top-30 → RRF merge → take top-30
  2. Rerank top-30 with cross-encoder
  3. If best score < confidence_threshold (4.0):
       Deep: BM25 top-k_deep + FAISS top-k_deep → RRF merge → take top-k_deep
       Rerank top-k_deep  ← this is what we sweep
"""
from __future__ import annotations
import os, sys, time, json, warnings
from pathlib import Path
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from benchmarks._lib.metrics import ndcg_at_k, recall_at_k
from lethe.encoders import OnnxCrossEncoder

DATA = ROOT / "data"

K_SHALLOW = 30
CONFIDENCE = 4.0
K_SWEEPS = [30, 60, 100, 200]  # k_deep values
SAMPLE = 100

print("=" * 68)
print("adaptive deep-pass: NDCG vs latency sweep")
print("=" * 68)
print(f"k_shallow={K_SHALLOW}  confidence_threshold={CONFIDENCE}  sample={SAMPLE}")

# ---- load ----
print("\nloading LongMemEval...", flush=True)
with open(DATA / "longmemeval_qrels.json") as f: qrels = json.load(f)
with open(DATA / "longmemeval_corpus.json") as f: corpus_content = json.load(f)
with open(DATA / "longmemeval_queries.json") as f: query_texts = json.load(f)
prep = np.load(str(DATA / "longmemeval_prepared.npz"), allow_pickle=True)
corpus_ids = list(prep["corpus_ids"])
corpus_embs = prep["corpus_embeddings"].astype(np.float32)
query_ids = list(prep["query_ids"])
query_embs = prep["query_embeddings"].astype(np.float32)
print(f"  corpus={len(corpus_ids)}  queries={len(query_ids)}")

rng = np.random.default_rng(0)
idx = rng.choice(len(query_ids), size=SAMPLE, replace=False)

# ---- indexes ----
print("building FAISS...", flush=True); t0 = time.monotonic()
faiss_idx = faiss.IndexFlatIP(corpus_embs.shape[1])
faiss_idx.add(corpus_embs)
print(f"  {time.monotonic()-t0:.1f}s")

print("building BM25...", flush=True); t0 = time.monotonic()
tokenized = [corpus_content.get(cid, "").lower().split() for cid in corpus_ids]
bm25 = BM25Okapi(tokenized)
print(f"  {time.monotonic()-t0:.1f}s")

print("loading cross-encoder...", flush=True); t0 = time.monotonic()
xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
_ = xenc.predict([("warm", "warm")])
print(f"  {time.monotonic()-t0:.1f}s")

def rrf_merge(bm25_ids, dense_ids, topn):
    scores = {}
    for rank, cid in enumerate(bm25_ids):
        scores[cid] = scores.get(cid, 0) + 1.0 / (60 + rank + 1)
    for rank, cid in enumerate(dense_ids):
        scores[cid] = scores.get(cid, 0) + 1.0 / (60 + rank + 1)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [cid for cid, _ in ranked[:topn]]

def gather(qt, qe, k):
    s = bm25.get_scores(qt.lower().split())
    bids = [corpus_ids[j] for j in np.argsort(s)[::-1][:k]]
    _, I = faiss_idx.search(qe.reshape(1, -1), k)
    dids = [corpus_ids[j] for j in I[0] if j >= 0]
    return rrf_merge(bids, dids, k)

def rerank(qt, cand_ids):
    if not cand_ids: return []
    pairs = [(qt, corpus_content.get(c, "")) for c in cand_ids]
    scores = xenc.predict(pairs)
    return sorted(zip(cand_ids, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)

def run_query(qt, qe, k_deep):
    shallow_cands = gather(qt, qe, K_SHALLOW)
    ranked = rerank(qt, shallow_cands)
    triggered = False
    if ranked and ranked[0][1] < CONFIDENCE and k_deep > K_SHALLOW:
        triggered = True
        deep_cands = gather(qt, qe, k_deep)
        ranked = rerank(qt, deep_cands)
    top10 = [c for c, _ in ranked[:10]]
    return top10, triggered

# ---- sweep ----
results = []
for k_deep in K_SWEEPS:
    label = "shallow-only (no deep pass)" if k_deep == K_SHALLOW else f"k_deep={k_deep}"
    print(f"\n=== {label} ===", flush=True)
    ndcgs, recalls, latencies = [], [], []
    n_triggered = 0
    t_all = time.monotonic()
    for i in idx:
        qi = query_ids[i]
        qr = qrels.get(qi, {})
        if not qr: continue
        qt = query_texts.get(qi, "")
        qe = query_embs[i].astype(np.float32)
        t0 = time.monotonic()
        top10, triggered = run_query(qt, qe, k_deep)
        latencies.append((time.monotonic() - t0) * 1000)
        if triggered: n_triggered += 1
        ndcgs.append(ndcg_at_k(top10, qr, 10))
        recalls.append(recall_at_k(top10, qr, 10))
    lat = np.array(latencies)
    ndcg_mean = float(np.mean(ndcgs)); recall_mean = float(np.mean(recalls))
    p50 = float(np.percentile(lat, 50)); p95 = float(np.percentile(lat, 95)); p99 = float(np.percentile(lat, 99))
    trig_rate = n_triggered / len(ndcgs) * 100
    wall = time.monotonic() - t_all
    print(f"  NDCG@10 = {ndcg_mean:.4f}  Recall@10 = {recall_mean:.4f}")
    print(f"  latency   p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms  mean={lat.mean():.0f}ms")
    print(f"  deep-pass triggered: {n_triggered}/{len(ndcgs)} ({trig_rate:.0f}%)  |  wall={wall:.0f}s")
    results.append({"k_deep": k_deep, "label": label, "ndcg": ndcg_mean, "recall": recall_mean,
                    "p50": p50, "p95": p95, "p99": p99, "mean": lat.mean(), "trig": trig_rate})

# ---- summary ----
base = next(r for r in results if r["k_deep"] == 200)
print("\n" + "=" * 68)
print(f"{'config':30s} {'NDCG@10':>9} {'ΔNDCG':>7} {'p50':>7} {'p95':>7} {'p99':>7} {'trig%':>6}")
for r in results:
    d = (r["ndcg"] - base["ndcg"]) * 100
    print(f"{r['label']:30s} {r['ndcg']:>9.4f} {d:>+7.2f} {r['p50']:>5.0f}ms {r['p95']:>5.0f}ms {r['p99']:>5.0f}ms {r['trig']:>5.0f}")

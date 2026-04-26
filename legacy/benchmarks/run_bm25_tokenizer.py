"""Sweep BM25 tokenizers to see whether a smarter tokenizer beats
the current `text.lower().split()` baseline on LongMemEval.

Compares four tokenizers over the same hybrid pipeline
(BM25 + FAISS + cross-encoder rerank):

1. baseline       — lower().split()                          (production)
2. regex_words    — regex word-tokens, punctuation stripped
3. regex_stop     — regex + remove a tiny English stop list
4. regex_stem     — regex + Porter stemmer (if nltk available)

Metrics: NDCG@10, Recall@10, p50/p95 latency, over a fixed query
sample. k_deep held at the production default (100) so deep-pass
behavior matches shipped code.
"""
from __future__ import annotations
import os, re, sys, time, json, warnings
from pathlib import Path
from typing import Callable
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
K_DEEP = 100
CONFIDENCE = 4.0
SAMPLE = 100

# ---- tokenizers ----

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
    "as", "by", "with", "from", "about", "into", "over", "so", "than",
    "then", "do", "does", "did", "have", "has", "had", "but", "if",
    "not", "no", "yes", "my", "your", "our", "their",
})


def tok_baseline(text: str) -> list[str]:
    return text.lower().split()


def tok_regex_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def tok_regex_stop(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in STOP]


def _build_stemmer():
    try:
        from nltk.stem.porter import PorterStemmer  # type: ignore
    except Exception:
        return None
    return PorterStemmer()


_STEMMER = _build_stemmer()


def tok_regex_stem(text: str) -> list[str]:
    if _STEMMER is None:
        return tok_regex_stop(text)
    return [_STEMMER.stem(t) for t in _WORD_RE.findall(text.lower()) if t not in STOP]


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {
    "baseline (lower+split)": tok_baseline,
    "regex words": tok_regex_words,
    "regex + stopwords": tok_regex_stop,
    "regex + stop + stem": tok_regex_stem,
}

print("=" * 72)
print("BM25 tokenizer sweep on LongMemEval")
print("=" * 72)
print(f"k_shallow={K_SHALLOW}  k_deep={K_DEEP}  confidence={CONFIDENCE}  sample={SAMPLE}")
if _STEMMER is None:
    print("note: nltk PorterStemmer not installed → 'stem' tokenizer falls back to stop-only")

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

# ---- shared ----
print("building FAISS...", flush=True); t0 = time.monotonic()
faiss_idx = faiss.IndexFlatIP(corpus_embs.shape[1])
faiss_idx.add(corpus_embs)
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


def rerank(qt, cand_ids):
    if not cand_ids: return []
    pairs = [(qt, corpus_content.get(c, "")) for c in cand_ids]
    scores = xenc.predict(pairs)
    return sorted(zip(cand_ids, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)


def sweep_one(name: str, tok_fn) -> dict:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.monotonic()
    tokenized = [tok_fn(corpus_content.get(cid, "")) for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    build_ms = (time.monotonic() - t0) * 1000
    print(f"  bm25 build: {build_ms:.0f} ms")

    def gather(qt, qe, k):
        s = bm25.get_scores(tok_fn(qt))
        bids = [corpus_ids[j] for j in np.argsort(s)[::-1][:k]]
        _, I = faiss_idx.search(qe.reshape(1, -1), k)
        dids = [corpus_ids[j] for j in I[0] if j >= 0]
        return rrf_merge(bids, dids, k)

    def run_query(qt, qe):
        shallow = gather(qt, qe, K_SHALLOW)
        ranked = rerank(qt, shallow)
        triggered = False
        if ranked and ranked[0][1] < CONFIDENCE and K_DEEP > K_SHALLOW:
            triggered = True
            deep = gather(qt, qe, K_DEEP)
            ranked = rerank(qt, deep)
        return [c for c, _ in ranked[:10]], triggered

    ndcgs, recalls, latencies = [], [], []
    n_triggered = 0
    wall_t0 = time.monotonic()
    for i in idx:
        qi = query_ids[i]
        qr = qrels.get(qi, {})
        if not qr: continue
        qt = query_texts.get(qi, "")
        qe = query_embs[i].astype(np.float32)
        t0 = time.monotonic()
        top10, triggered = run_query(qt, qe)
        latencies.append((time.monotonic() - t0) * 1000)
        if triggered: n_triggered += 1
        ndcgs.append(ndcg_at_k(top10, qr, 10))
        recalls.append(recall_at_k(top10, qr, 10))
    if not ndcgs:
        # Every sampled query was filtered out (empty qrels, etc.).
        # Don't crash on np.percentile([]) or the trig division —
        # surface the gap so the user reruns with a saner sample.
        raise RuntimeError(
            f"sweep '{name}': no queries with qrels in the sampled set "
            f"(size={SAMPLE}). Check longmemeval_qrels.json or raise SAMPLE."
        )
    lat = np.array(latencies)
    wall = time.monotonic() - wall_t0
    ndcg = float(np.mean(ndcgs)); rec = float(np.mean(recalls))
    p50 = float(np.percentile(lat, 50)); p95 = float(np.percentile(lat, 95))
    trig = n_triggered / len(ndcgs) * 100
    print(f"  NDCG@10={ndcg:.4f}  Recall@10={rec:.4f}")
    print(f"  p50={p50:.0f}ms  p95={p95:.0f}ms  trig={trig:.0f}%  wall={wall:.0f}s")
    return {"name": name, "ndcg": ndcg, "recall": rec, "p50": p50, "p95": p95,
            "trig": trig, "build_ms": build_ms, "n_eval": len(ndcgs)}


# ---- sweep ----
results = []
for name, fn in TOKENIZERS.items():
    results.append(sweep_one(name, fn))

# ---- summary ----
base = results[0]
print("\n" + "=" * 72)
print(f"{'tokenizer':28s} {'NDCG@10':>9} {'ΔNDCG':>8} {'Recall@10':>11} {'p50':>7} {'p95':>7} {'build':>8}")
for r in results:
    d = (r["ndcg"] - base["ndcg"]) * 100
    print(f"{r['name']:28s} {r['ndcg']:>9.4f} {d:>+7.2f}pp {r['recall']:>11.4f} {r['p50']:>5.0f}ms {r['p95']:>5.0f}ms {r['build_ms']:>6.0f}ms")
print(f"\nn_eval={results[0]['n_eval']}  (queries with at least one qrel)")

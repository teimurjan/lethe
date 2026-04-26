"""Single-pass lethe worker with pre-computed centroids injected.

Like _pass.py but pre-computes centroids from ALL query embeddings
(matching the benchmark exactly) and injects them into MemoryStore.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
REPO = DEMO.parent
sys.path.insert(0, str(REPO / "legacy"))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np  # noqa: E402
import faiss  # noqa: E402,F401

from benchmarks._lib.metrics import ndcg_at_k  # noqa: E402
from lethe.entry import MemoryEntry, Tier  # noqa: E402
from lethe.memory_store import MemoryStore  # noqa: E402
from lethe.rif import RIFConfig, build_clusters  # noqa: E402

DATA = REPO / "data"
NUM_QUERIES = 5000
K_FINAL = 10
SEED = 42
N_CLUSTERS = 30


class _Enc:
    def __init__(self, st):
        self._st = st

    def encode(self, text, **kw):
        return self._st.encode(
            text, normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)


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

    # Pre-compute centroids from ALL query embeddings (matches benchmark)
    print(f"Building {N_CLUSTERS} centroids from {len(eval_ids)} query embeddings...", flush=True)
    all_qembs = np.stack([query_embs[qid_to_idx[q]] for q in eval_ids]).astype(np.float32)
    centroids = build_clusters(all_qembs, N_CLUSTERS)
    print(f"Centroids shape: {centroids.shape}", flush=True)

    # Now load torch/sentence-transformers
    from sentence_transformers import CrossEncoder, SentenceTransformer
    bi = _Enc(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))
    xe = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Schedule
    rng = np.random.default_rng(SEED)
    n_hot = max(1, int(len(eval_ids) * 0.2))
    hot_idx = rng.choice(len(eval_ids), size=n_hot, replace=False)
    hot_set = {eval_ids[i] for i in hot_idx}
    hot_ids = sorted(hot_set)
    cold_ids = [q for q in eval_ids if q not in hot_set]
    schedule: list[str] = []
    for _ in range(NUM_QUERIES):
        if rng.random() < 0.7:
            schedule.append(hot_ids[rng.integers(len(hot_ids))])
        else:
            schedule.append(cold_ids[rng.integers(len(cold_ids))])
    print(f"Schedule: {NUM_QUERIES} queries", flush=True)

    cfg = RIFConfig(alpha=0.3, use_rank_gap=True, suppression_rate=0.1, n_clusters=N_CLUSTERS)

    with tempfile.TemporaryDirectory() as tmp:
        store = MemoryStore(
            path=Path(tmp) / "lethe",
            bi_encoder=bi, cross_encoder=xe,
            dim=384, k_shallow=30, k_deep=30,
            rif_config=cfg,
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

        # Inject pre-computed centroids
        store._cluster_centroids = centroids
        store._cluster_dirty = False
        print("Centroids injected.", flush=True)

        # Free big objects
        del corpus_content, data, corpus_embs
        import gc; gc.collect()

        ndcgs: list[float] = []
        qids: list[str] = []
        t0 = time.time()
        running = 0.0
        for step, qid in enumerate(schedule):
            qt = query_texts.get(qid, "")
            qr = qrels.get(qid, {})
            hits = store.retrieve(qt, k=K_FINAL)
            ndcg = ndcg_at_k([h[0] for h in hits], qr, K_FINAL)
            ndcgs.append(float(ndcg))
            qids.append(qid)
            running += ndcg
            if (step + 1) % 100 == 0:
                n = step + 1
                elapsed = time.time() - t0
                print(
                    f"[lethe] step {n}/{NUM_QUERIES}  "
                    f"mean={running / n:.3f}  "
                    f"({elapsed:.0f}s, {n / elapsed:.1f} q/s)",
                    flush=True,
                )
        store.close()

    out_path = DEMO / ".pass_out" / "lethe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"qids": qids, "ndcgs": ndcgs}))
    print(f"Wrote {out_path}  (mean={running / NUM_QUERIES:.3f})", flush=True)


if __name__ == "__main__":
    main()

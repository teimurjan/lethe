"""Single-pass worker for the demo collector.

Reads sys.argv:
    _pass.py <label> <alpha> <use_rank_gap> <n_clusters> <out_json>

Runs NUM_QUERIES on one RIFConfig against the full LongMemEval corpus and
writes a compact JSON list of NDCG@10 values.

Isolated in its own process so the two passes (baseline, lethe) never
share memory — lethe's clustered k-means was OOM'ing when the baseline
store was still resident.
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

# Import faiss BEFORE torch / sentence-transformers to avoid an OpenMP
# conflict that segfaults faiss.Kmeans on the 199k × 384 corpus.
import faiss  # noqa: E402,F401

from benchmarks._lib.metrics import ndcg_at_k  # noqa: E402
from lethe.entry import MemoryEntry, Tier  # noqa: E402
from lethe.memory_store import MemoryStore  # noqa: E402
from lethe.rif import RIFConfig  # noqa: E402

DATA = REPO / "data"
NUM_QUERIES = 5000
K_FINAL = 10
SEED = 42


class _PrebuiltEncoder:
    def __init__(self, st: SentenceTransformer) -> None:
        self._st = st

    def encode(
        self,
        text: str,
        normalize_embeddings: bool = True,  # noqa: ARG002
        show_progress_bar: bool = False,  # noqa: ARG002
        batch_size: int = 32,  # noqa: ARG002
    ) -> np.ndarray:
        return self._st.encode(
            text, normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)


def main() -> None:
    label = sys.argv[1]
    alpha = float(sys.argv[2])
    use_rank_gap = sys.argv[3] == "True"
    n_clusters = int(sys.argv[4])
    out_path = Path(sys.argv[5])

    print(
        f"[{label}] alpha={alpha} rank_gap={use_rank_gap} "
        f"n_clusters={n_clusters}",
        flush=True,
    )

    data = np.load(str(DATA / "longmemeval_prepared.npz"), allow_pickle=True)
    corpus_ids = [str(c) for c in data["corpus_ids"].tolist()]
    corpus_embs = data["corpus_embeddings"].astype(np.float32)
    query_ids_all = [str(q) for q in data["query_ids"].tolist()]
    qrels = json.loads((DATA / "longmemeval_qrels.json").read_text())
    corpus_content = json.loads((DATA / "longmemeval_corpus.json").read_text())
    query_texts = json.loads((DATA / "longmemeval_queries.json").read_text())
    eval_ids = [q for q in query_ids_all if qrels.get(q)]

    from sentence_transformers import CrossEncoder, SentenceTransformer
    bi_real = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    xe = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    bi = _PrebuiltEncoder(bi_real)

    cfg = RIFConfig(
        alpha=alpha,
        use_rank_gap=use_rank_gap,
        n_clusters=n_clusters,
        suppression_rate=0.1,
    )

    rng = np.random.default_rng(SEED)
    n_hot = max(1, int(len(eval_ids) * 0.2))
    hot_idx = rng.choice(len(eval_ids), size=n_hot, replace=False)
    hot_set = {eval_ids[i] for i in hot_idx}
    # sorted() is critical: Python sets don't preserve insertion order
    # across processes (hash randomization), which otherwise makes the
    # schedule nondeterministic between the baseline and lethe subprocess.
    hot_ids = sorted(hot_set)
    cold_ids = [q for q in eval_ids if q not in hot_set]
    schedule: list[str] = []
    for _ in range(NUM_QUERIES):
        if rng.random() < 0.7:
            schedule.append(hot_ids[rng.integers(len(hot_ids))])
        else:
            schedule.append(cold_ids[rng.integers(len(cold_ids))])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store = MemoryStore(
            path=tmp_path / label,
            bi_encoder=bi,
            cross_encoder=xe,
            dim=384,
            # Match benchmarks/run_rif_gap.py: flat k=30, no adaptive depth.
            # This is the baseline the README's +6.5% number is measured against.
            k_shallow=30,
            k_deep=30,
            rif_config=cfg,
        )
        dim = corpus_embs.shape[1]
        for i, eid in enumerate(corpus_ids):
            emb = corpus_embs[i]
            content = corpus_content.get(eid, "")
            store.entries[eid] = MemoryEntry(
                id=eid,
                content=content,
                base_embedding=emb.copy(),
                embedding=emb.copy(),
                adapter=np.zeros(dim, dtype=np.float32),
                tier=Tier.NAIVE,
            )
            store._embeddings[eid] = emb
        store._rebuild_index()
        print(f"[{label}] index built ({len(store.entries)} entries)", flush=True)

        # Free the big decoded dicts / raw embeddings array — we won't need
        # them again, and they're competing for memory with k-means.
        del corpus_content
        del data
        del corpus_embs
        import gc
        gc.collect()

        if cfg.n_clusters > 0:
            print(
                f"[{label}] clustered RIF active (n_clusters={n_clusters}). "
                f"Centroids will be built from queries after {n_clusters} "
                f"retrievals.",
                flush=True,
            )

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
                    f"[{label}] step {n}/{NUM_QUERIES}  "
                    f"mean={running / n:.3f}  "
                    f"({elapsed:.0f}s, {n / elapsed:.1f} q/s)",
                    flush=True,
                )
        store.close()

    out_path.write_text(json.dumps({"qids": qids, "ndcgs": ndcgs}))
    print(f"[{label}] wrote {out_path}  (mean={running / NUM_QUERIES:.3f})", flush=True)


if __name__ == "__main__":
    main()

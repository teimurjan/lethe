"""Single-pass worker for the replay demo.

Reads sys.argv:
    _pass_replay.py <label> <alpha> <use_rank_gap> <n_clusters> \
        <n_unique> <n_replay> <n_rounds> <n_warmup> <out_json>

If <n_warmup> > 0, Phase 0 runs <n_warmup> queries drawn from the pool
but disjoint from the cold set ("warmup" phase). This seeds the
clustered-RIF centroids and suppression state so Phase 1 is genuinely
cold per-cluster. Phase 1 then runs <n_unique> distinct queries in a
fixed order ("cold"), followed by <n_rounds> replay rounds, each
playing the first <n_replay> queries in the same order, tagged
"warm1", "warm2", ... , "warm<n_rounds>". Each step's NDCG@10 and
phase tag is written to <out_json>.

Isolated in its own process so baseline and lethe passes never share
memory — clustered k-means OOMs otherwise.
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

# Import faiss BEFORE torch / sentence-transformers to avoid the OpenMP
# conflict that segfaults faiss.Kmeans on the 199k × 384 corpus.
import faiss  # noqa: E402,F401

from benchmarks._lib.metrics import ndcg_at_k  # noqa: E402
from lethe.entry import MemoryEntry, Tier  # noqa: E402
from lethe.memory_store import MemoryStore  # noqa: E402
from lethe.rif import RIFConfig  # noqa: E402

DATA = REPO / "data"
K_FINAL = 10
SEED = 42


class _PrebuiltEncoder:
    def __init__(self, st) -> None:
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
    n_unique = int(sys.argv[5])
    n_replay = int(sys.argv[6])
    n_rounds = int(sys.argv[7])
    n_warmup = int(sys.argv[8])
    out_path = Path(sys.argv[9])

    assert n_replay <= n_unique, "n_replay must be <= n_unique"
    assert n_rounds >= 1, "n_rounds must be >= 1"
    assert n_warmup >= 0, "n_warmup must be >= 0"

    print(
        f"[{label}] alpha={alpha} rank_gap={use_rank_gap} "
        f"n_clusters={n_clusters} unique={n_unique} replay={n_replay} "
        f"rounds={n_rounds} warmup={n_warmup}",
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

    # Deterministic: pick n_unique cold queries first (preserves legacy
    # selection), then n_warmup warmup queries from the remaining pool
    # (disjoint from cold). sorted() guarantees cross-process stability.
    rng = np.random.default_rng(SEED)
    pool = sorted(eval_ids)
    cold_picks = rng.choice(len(pool), size=n_unique, replace=False)
    cold_set = set(cold_picks.tolist())
    unique_q = [pool[i] for i in cold_picks]
    replay_q = unique_q[:n_replay]
    warmup_q: list[str] = []
    if n_warmup > 0:
        remaining = np.array(
            [i for i in range(len(pool)) if i not in cold_set],
            dtype=np.int64,
        )
        assert len(remaining) > 0, "no queries available for warmup"
        # Sample with replacement: LongMemEval's eval pool is only ~500,
        # so n_warmup can exceed len(remaining). Repetition in warmup is
        # fine — it mirrors a real workload and gives RIF more training
        # signal per cluster.
        replace = n_warmup > len(remaining)
        warmup_picks = rng.choice(len(remaining), size=n_warmup, replace=replace)
        warmup_q = [pool[int(remaining[i])] for i in warmup_picks]
        print(
            f"[{label}] warmup draws {n_warmup} from {len(remaining)} "
            f"non-cold queries (replace={replace})",
            flush=True,
        )

    schedule: list[tuple[str, str]] = [(q, "warmup") for q in warmup_q]
    schedule += [(q, "cold") for q in unique_q]
    for r in range(1, n_rounds + 1):
        tag = f"warm{r}"
        schedule += [(q, tag) for q in replay_q]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store = MemoryStore(
            path=tmp_path / label,
            bi_encoder=bi,
            cross_encoder=xe,
            dim=384,
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

        del corpus_content
        del data
        del corpus_embs
        import gc
        gc.collect()

        if cfg.n_clusters > 0:
            print(
                f"[{label}] clustered RIF active (n_clusters={n_clusters}). "
                f"Centroids bootstrap from queries after {n_clusters} "
                f"retrievals.",
                flush=True,
            )

        ndcgs: list[float] = []
        qids: list[str] = []
        phases: list[str] = []
        t0 = time.time()
        total = len(schedule)
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for step, (qid, phase) in enumerate(schedule):
            qt = query_texts.get(qid, "")
            qr = qrels.get(qid, {})
            hits = store.retrieve(qt, k=K_FINAL)
            ndcg = ndcg_at_k([h[0] for h in hits], qr, K_FINAL)
            ndcgs.append(float(ndcg))
            qids.append(qid)
            phases.append(phase)
            sums[phase] = sums.get(phase, 0.0) + ndcg
            counts[phase] = counts.get(phase, 0) + 1
            if (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                phase_order = ["warmup", "cold"] + [
                    f"warm{r}" for r in range(1, n_rounds + 1)
                ]
                means = " ".join(
                    f"{p}={sums[p] / counts[p]:.3f}"
                    for p in phase_order
                    if p in counts
                )
                print(
                    f"[{label}] step {step + 1}/{total}  {means}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )
        store.close()

    out_path.write_text(
        json.dumps({"qids": qids, "ndcgs": ndcgs, "phases": phases})
    )
    cold_mean = sums.get("cold", 0.0) / max(counts.get("cold", 1), 1)
    summary = []
    if "warmup" in counts:
        summary.append(f"warmup={sums['warmup'] / counts['warmup']:.3f}")
    summary.append(f"cold={cold_mean:.3f}")
    for r in range(1, n_rounds + 1):
        tag = f"warm{r}"
        if tag in counts:
            m = sums[tag] / counts[tag]
            summary.append(f"{tag}={m:.3f} (Δ{m - cold_mean:+.3f})")
    print(f"[{label}] wrote {out_path}  " + "  ".join(summary), flush=True)


if __name__ == "__main__":
    main()

"""Export `data/longmemeval_prepared.npz` to flat files the Rust bench
can read without a numpy dependency.

Writes (all under `data/lme_rust/`):
  corpus_ids.txt              one id per line
  corpus_embeddings.bin       f32 little-endian, shape (n_corpus, dim) row-major
  query_ids.txt               one id per line
  query_embeddings.bin        f32 little-endian, shape (n_queries, dim)
  sampled_query_indices.txt   the 200-query subset both impls evaluate
  meta.json                   {"n_corpus", "n_queries", "dim", "sample_seed", "sample_size"}

The same Python and Rust harnesses both consume `sampled_query_indices.txt`,
so the eval set is identical between the two implementations even though
their random generators differ.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "longmemeval_prepared.npz"
OUT_DIR = REPO / "data" / "lme_rust"

SAMPLE_SIZE = 200
SAMPLE_SEED = 0


def main() -> int:
    if not SRC.exists():
        sys.stderr.write(
            f"error: {SRC} not present. "
            "Run `uv run python experiments/prep_longmemeval.py` first.\n"
        )
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(str(SRC), allow_pickle=True)
    corpus_ids = data["corpus_ids"]
    corpus_embs = data["corpus_embeddings"].astype(np.float32)
    query_ids = data["query_ids"]
    query_embs = data["query_embeddings"].astype(np.float32)

    n_corpus = len(corpus_ids)
    n_queries = len(query_ids)
    dim = corpus_embs.shape[1]

    (OUT_DIR / "corpus_ids.txt").write_text("\n".join(str(x) for x in corpus_ids))
    (OUT_DIR / "corpus_embeddings.bin").write_bytes(corpus_embs.tobytes(order="C"))
    (OUT_DIR / "query_ids.txt").write_text("\n".join(str(x) for x in query_ids))
    (OUT_DIR / "query_embeddings.bin").write_bytes(query_embs.tobytes(order="C"))

    rng = np.random.default_rng(SAMPLE_SEED)
    sample = rng.choice(n_queries, size=min(SAMPLE_SIZE, n_queries), replace=False)
    (OUT_DIR / "sampled_query_indices.txt").write_text(
        "\n".join(str(int(i)) for i in sample)
    )

    meta = {
        "n_corpus": int(n_corpus),
        "n_queries": int(n_queries),
        "dim": int(dim),
        "sample_seed": SAMPLE_SEED,
        "sample_size": int(len(sample)),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {OUT_DIR}/")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

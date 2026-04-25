"""One-time data prep for the bench suites.

Exports `data/longmemeval_prepared.npz` to flat files the Rust bench
binary can read without a numpy dependency:

  data/lme_rust/corpus_ids.txt              one id per line
  data/lme_rust/corpus_embeddings.bin       f32 LE row-major
  data/lme_rust/query_ids.txt
  data/lme_rust/query_embeddings.bin
  data/lme_rust/sampled_query_indices.txt   200-query subset both impls evaluate
  data/lme_rust/meta.json                   {n_corpus, n_queries, dim, sample_seed, sample_size}

Both Python and Rust harnesses consume `sampled_query_indices.txt` so
the eval set is identical regardless of which language seeds the RNG.
This is the only persistent intermediate state — gitignored, kept
across runs so the bench is fast to re-invoke.

Idempotent: bails out early if `meta.json` is already present unless
``--force`` is passed.
"""
from __future__ import annotations

import argparse
import json
import sys

from _lib import DATA, LME_RUST

SAMPLE_SIZE = 200
SAMPLE_SEED = 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Export LongMemEval data to Rust-readable files.")
    p.add_argument("--force", action="store_true", help="Re-export even if files are up to date.")
    args = p.parse_args(argv)

    src = DATA / "longmemeval_prepared.npz"
    if not src.exists():
        sys.stderr.write(
            f"error: {src} not found. Run `uv run python experiments/prep_longmemeval.py` first.\n"
        )
        return 2

    LME_RUST.mkdir(parents=True, exist_ok=True)
    meta_path = LME_RUST / "meta.json"
    if meta_path.exists() and not args.force:
        meta = json.loads(meta_path.read_text())
        print(f"[prepare] up to date: {meta}")
        return 0

    import numpy as np  # noqa: PLC0415

    data = np.load(str(src), allow_pickle=True)
    corpus_ids = data["corpus_ids"]
    corpus_embs = data["corpus_embeddings"].astype(np.float32)
    query_ids = data["query_ids"]
    query_embs = data["query_embeddings"].astype(np.float32)
    n_corpus = len(corpus_ids)
    n_queries = len(query_ids)
    dim = int(corpus_embs.shape[1])

    (LME_RUST / "corpus_ids.txt").write_text("\n".join(str(x) for x in corpus_ids))
    (LME_RUST / "corpus_embeddings.bin").write_bytes(corpus_embs.tobytes(order="C"))
    (LME_RUST / "query_ids.txt").write_text("\n".join(str(x) for x in query_ids))
    (LME_RUST / "query_embeddings.bin").write_bytes(query_embs.tobytes(order="C"))

    rng = np.random.default_rng(SAMPLE_SEED)
    sample = rng.choice(n_queries, size=min(SAMPLE_SIZE, n_queries), replace=False)
    (LME_RUST / "sampled_query_indices.txt").write_text(
        "\n".join(str(int(i)) for i in sample)
    )

    meta = {
        "n_corpus": int(n_corpus),
        "n_queries": int(n_queries),
        "dim": dim,
        "sample_seed": SAMPLE_SEED,
        "sample_size": int(len(sample)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[prepare] wrote {LME_RUST}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(DATA.parent / "bench"))  # for _lib import when run directly
    sys.exit(main(sys.argv[1:]))

# Python ↔ Rust parity bench

The migration-confidence ladder. Three suites, all 1-1 between Python
and Rust: each script accepts `--impl python` or `--impl rust` and
emits the same JSON shape; `--compare` runs both and writes a single
markdown report under `migration_benchmarks/results/`.

## Layout

```
migration_benchmarks/
├── README.md
├── _lib.py            # shared utilities
├── prepare.py         # one-time data export (gitignored output)
├── longmemeval.py     # Layer 1: end-to-end NDCG/Recall on LongMemEval
├── components.py      # Layer 3: per-component numerical diff
├── latency.py         # cold-start + warm-retrieve timing
└── results/           # only persistent output: COMPARE_*.md reports
```

The Rust side is a single binary, `target/release/lethe-bench`, with
matching subcommands (`longmemeval`, `bm25`, `flat-ip`, `xenc`). Each
Python suite shells out to the corresponding subcommand and parses
JSON.

## Run

One-time setup (writes `tmp_data/lme_rust/` — flat-binary mirror of the
prepared LongMemEval data, gitignored):

```bash
uv run python migration_benchmarks/prepare.py
```

Run any suite in compare mode:

```bash
./migration_benchmarks/longmemeval.py --compare    # quality: NDCG@10, Recall@10
./migration_benchmarks/components.py --compare     # numerical diff per component
./migration_benchmarks/latency.py --compare        # cold + warm timing
```

Or single-impl, JSON to stdout:

```bash
./migration_benchmarks/longmemeval.py --impl python
./migration_benchmarks/longmemeval.py --impl rust
```

`migration_benchmarks/results/COMPARE_<SUITE>_<host>_<date>.md` is the only file
written. Intermediate JSON outputs go through `tempfile` and are
cleaned up on exit.

## What each suite measures

**longmemeval** — end-to-end. Five retrieval configs (vector only,
BM25 only, hybrid RRF, vector+xenc, lethe full), 200 sampled queries
from LongMemEval, NDCG@10 + Recall@10. Pass: per-config
`|ΔNDCG| ≤ 0.005` and `|ΔRecall| ≤ 0.005`. The strongest single signal.

**components** — per-piece numerical diff. BM25 score vector
(elementwise), FlatIP top-30 (id-set Jaccard), cross-encoder logits
(elementwise). Pass: `|ΔBM25| ≤ 1e-4`, FlatIP Jaccard `≥ 0.99`,
`|Δxenc| ≤ 1e-3`. Useful for isolating the cause when `longmemeval`
drifts.

**latency** — cold start (median of 3 fresh process boots) + warm
retrieve (subprocess-per-query for Rust, in-process for Python) at
corpus sizes `{500, 5000, 20000}`. Reports p50/p95 and the relative
speedup.

## Migration cutover

```
longmemeval --compare PASS                 → safe to keep developing in parallel
+ alternate seed (edit prepare.py SAMPLE_SEED) PASS  → safe for opt-in users
+ components --compare PASS                → safe to make Rust the default
+ shadow logging clean over a week         → safe to remove the Python impl
```

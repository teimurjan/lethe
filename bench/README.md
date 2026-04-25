# Python ↔ Rust comparison harness

End-to-end timer that measures cold-start (process boot) and warm
retrieve latency for both implementations on the local machine, then
writes a markdown report to `bench/results/COMPARE_<host>_<date>.md`.

## Run

```bash
./bench/run_compare.sh
```

The script ensures `target/release/lethe-rs` exists (builds it if
not), then invokes the Python harness via `uv run`. End-to-end
runtime is dominated by the model download on first use; subsequent
runs are ~3–5 minutes total.

## What it measures

1. **Cold start (3 samples each)** — `python -c "import lethe.memory_store; import lethe.encoders"` vs. `lethe-rs --version`. Reports the median wall time.
2. **Warm retrieve (10 timed queries × 3 corpus sizes)** — for each `N ∈ {500, 5000, 20000}`, seeds a Python store with synthetic entries, then times retrieval through the Python in-process API and through `lethe-rs search` (fresh subprocess per query). Reports p50 and p95.

The Rust warm path is intentionally "subprocess per query" so it captures the realistic command-line invocation pattern: every Claude Code hook firing pays a binary boot. Python's in-process number is the upper bound for what an embedded Python user sees.

## Reading the output

| Header | Means |
|---|---|
| `Cold start` | How long the implementation takes to be ready. Rust's win is mostly here — Python imports cost ~3 s, the Rust binary boots in ~0.25 s. |
| `Warm retrieve p50` | Median of 10 retrievals at corpus size N. Includes encoder + BM25 + rerank. |
| `p50 speedup` | `python_p50 / rust_p50`. Greater than 1 means Rust is faster. |

Cross-encoder rerank is fixed cost regardless of language (ONNX Runtime is C++ either way), so the win narrows on small corpora and widens as N grows because Python's BM25 score loop is the linear-with-N piece that Rust's port displaces.

## Criterion micro-benches

The Rust crate ships criterion benchmarks under `crates/lethe-core/benches/`:

```bash
cargo bench -p lethe-core
```

Useful for tracking regressions in the individual building blocks (BM25 build/score, FlatIP search, RRF merge, RIF update) without waiting for the full end-to-end harness.

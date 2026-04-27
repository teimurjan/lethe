# lethe-memory

Python bindings to the Rust core of
[lethe](https://github.com/teimurjan/lethe), a self-improving memory store
for LLM agents. Built with PyO3 + maturin, distributed as wheels.

## Install

```bash
pip install lethe-memory
```

## Use

```python
from lethe_memory import MemoryStore

store = MemoryStore("./my_memories")  # encoders default to MiniLM

store.add("I prefer window seats on flights",      session_id="trip")
store.add("My wife needs aisle seats",             session_id="trip")
store.add("I work at Google as a software engineer", session_id="work")

for hit in store.retrieve("What are my travel preferences?", k=5):
    print(f"  [{hit.score:.1f}] {hit.content}")

store.save()
```

The wheel ships the Rust `lethe-core` retrieval pipeline (BM25 + dense
hybrid, cross-encoder rerank, clustered RIF) — same code path the CLI
uses, exposed via PyO3 with `Python::allow_threads` around all native
work so multi-threaded Python callers see real parallelism.

## Constructor knobs

```python
MemoryStore(
    path,
    bi_encoder="Xenova/all-MiniLM-L6-v2",
    cross_encoder="Xenova/ms-marco-MiniLM-L-6-v2",
    rif_config=None,                # RIFConfig(n_clusters=30, ...)
    k_shallow=30,
    k_deep=100,
    confidence_threshold=4.0,
    dedup_threshold=0.95,
)
```

## See also

- [Project landing page](https://github.com/teimurjan/lethe) — architecture, benchmarks, research journey
- [`lethe-cli`](https://crates.io/crates/lethe-cli) (Homebrew: `lethe`) — the standalone CLI

License: MIT.

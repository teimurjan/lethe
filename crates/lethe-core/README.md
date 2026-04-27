# lethe-core

Core retrieval library behind [lethe](https://github.com/teimurjan/lethe) — a
self-improving memory store for LLM agents. Hybrid BM25 + dense retrieval,
cross-encoder reranking, clustered retrieval-induced forgetting (RIF), DuckDB
persistence, all in pure Rust.

```
cargo add lethe-core
```

## Quick start

```rust
use std::sync::Arc;
use lethe_core::encoders::{BiEncoder, CrossEncoder};
use lethe_core::memory_store::{MemoryStore, StoreConfig};

# fn main() -> anyhow::Result<()> {
let bi    = Arc::new(BiEncoder::from_repo("Xenova/all-MiniLM-L6-v2")?);
let cross = Arc::new(CrossEncoder::from_repo("Xenova/ms-marco-MiniLM-L-6-v2")?);

let store = MemoryStore::open(
    "./.lethe/index",
    Some(bi.clone()),
    Some(cross.clone()),
    StoreConfig { dim: bi.dim(), ..StoreConfig::default() },
)?;

store.add("Postgres is great for transactional workloads.", None, "", 0)?;
store.add("DuckDB is built for embedded analytics.",          None, "", 0)?;
store.save()?;

for hit in store.retrieve("which db for OLAP?", 5)? {
    println!("[{:.2}] {}", hit.score, hit.content);
}
# Ok(())
# }
```

## What you get

- `MemoryStore` — open / add / retrieve / delete / save lifecycle
- `BM25Okapi` — bit-faithful port of `rank_bm25.BM25Okapi`
- `FlatIp` — top-K dense IP search (FAISS-equivalent)
- `BiEncoder` / `CrossEncoder` — ONNX wrappers via `ort`
- `RifConfig` + `ClusteredSuppressionState` — RIF mechanism
- `MemoryDb` — DuckDB schema with `entries`, `entry_embeddings`,
  `cluster_suppression`, `cluster_centroids`, `stats`
- `union_store::UnionStore` — read-only cross-project retrieval

Embeddings live as DuckDB BLOBs; the in-memory `FlatIp` is rebuilt from
those on every `open()` (~100 ms on 200k entries).

## See also

- Project landing page + benchmark numbers: [github.com/teimurjan/lethe](https://github.com/teimurjan/lethe)
- CLI built on top of this crate: [`lethe-cli`](https://crates.io/crates/lethe-cli)
- Python binding: [`lethe-memory`](https://pypi.org/project/lethe-memory/)
- Node binding: [`lethe`](https://www.npmjs.com/package/lethe)

License: MIT.

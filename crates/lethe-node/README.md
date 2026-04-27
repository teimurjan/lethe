# lethe

Node.js bindings to the Rust core of
[lethe](https://github.com/teimurjan/lethe), a self-improving memory store
for LLM agents. Built with [napi-rs](https://napi.rs/), distributed as
prebuilt platform binaries — no Rust toolchain needed at install time.

## Install

```bash
npm install lethe
# or
bun add lethe
```

## Use

```typescript
import { MemoryStore } from "lethe";

const store = await MemoryStore.open("./my_memories");
await store.add("I prefer window seats on flights");
await store.add("My wife needs aisle seats");
await store.add("I work at Google as a software engineer");

const hits = await store.retrieve("What are my travel preferences?", 5);
for (const h of hits) {
  console.log(`  [${h.score.toFixed(1)}] ${h.content}`);
}

await store.save();
```

`add` / `retrieve` / `save` are async — napi-rs runs them via
`tokio::spawn_blocking` so the Node event loop never stalls on the ONNX
inference or DuckDB I/O.

## Open options

```typescript
MemoryStore.open(path, {
  biEncoder?:    string,   // default "Xenova/all-MiniLM-L6-v2"
  crossEncoder?: string,   // default "Xenova/ms-marco-MiniLM-L-6-v2"
  nClusters?:    number,   // RIF clusters; 0 disables clustering
  useRankGap?:   boolean,
  kShallow?:     number,   // default 30
  kDeep?:        number,   // default 100
});
```

## See also

- [Project landing page](https://github.com/teimurjan/lethe) — architecture, benchmarks, research journey
- [`lethe-cli`](https://crates.io/crates/lethe-cli) (Homebrew: `lethe`) — the standalone CLI

License: MIT.

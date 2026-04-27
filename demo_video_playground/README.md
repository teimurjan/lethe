# demo

Visualizations for lethe's retrieval behavior on LongMemEval. A
Remotion (React) project under `src/` consumes JSON traces from
`public/`. The traces are produced by the data collector script.

## Collector

```bash
bun install
bun run collect          # writes public/napi_run.json
```

`scripts/collect_napi.ts` opens a `MemoryStore` via the lethe
N-API binding (`lethe` on npm, locally `file:../crates/lethe-node`),
indexes the small fixture under `data/`, runs the queries from
`data/queries.json`, and writes per-query top-5 hits to
`public/napi_run.json` for the Remotion compositions.

## Frozen data

`public/run.json`, `public/run_replay.json`,
`public/run_replay_warm.json`, and `public/run_replay_extended.json`
are committed snapshots from the original Python data-collection
pipeline (used by the published video). The Python collectors are no
longer in the repo; reproduce headline numbers via the migration
benchmarks at `migration_benchmarks/longmemeval.py --compare`.

## Layout

```
demo/
├── scripts/collect_napi.ts   # N-API data collector
├── src/                      # Remotion scenes
├── public/                   # generated + frozen run*.json
└── data/                     # corpus/queries fixtures
```

## Render

```bash
bun run dev                   # interactive Remotion studio
bun run build:mp4             # render to MP4
bun run build:gif             # render to GIF
```

// Demo data collection via the lethe N-API binding.
//
// This is the Rust-native path: opens a MemoryStore via napi-rs,
// indexes the small `demo/data/corpus.json` fixture, runs the
// `demo/data/queries.json` set, and writes per-query top-5 hits to
// `demo/public/napi_run.json` for the Remotion compositions.
//
// The legacy Python collectors live under `legacy/demo_scripts/` and
// remain the source of truth for the full RIF A/B benchmarks; this
// script is the binding-only demo that lives alongside the TS demo.

import { MemoryStore } from "lethe";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";

interface Entry {
  id?: string;
  content: string;
}

interface Query {
  id: string;
  text: string;
}

interface RunResult {
  query: string;
  hits: { id: string; content: string; score: number }[];
}

async function main(): Promise<void> {
  const repo = path.resolve(import.meta.dir ?? path.dirname(new URL(import.meta.url).pathname), "..");
  const corpusPath = path.join(repo, "data/corpus.json");
  const queriesPath = path.join(repo, "data/queries.json");
  const outPath = path.join(repo, "public/napi_run.json");

  const corpus: Entry[] = JSON.parse(await fs.readFile(corpusPath, "utf-8"));
  const queries: Query[] = JSON.parse(await fs.readFile(queriesPath, "utf-8"));

  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "lethe-demo-napi-"));
  try {
    const store = await MemoryStore.open(tmp);
    for (const entry of corpus) {
      await store.add(entry.content);
    }
    await store.save();

    const results: RunResult[] = [];
    for (const q of queries) {
      const hits = await store.retrieve(q.text, 5);
      results.push({ query: q.text, hits });
    }
    await fs.writeFile(outPath, JSON.stringify(results, null, 2));
    console.log(`wrote ${outPath} (${results.length} queries, store size=${store.size()})`);
  } finally {
    await fs.rm(tmp, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

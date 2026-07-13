# Heuristic Chunk Dedupe

## Problem Statement

How might we keep recall results diverse as the global `~/.lethe` index accumulates
near-identical transcript chunks across sessions — without full pairwise scans, and
without losing the RIF bookkeeping (affinity, retrieval_count, suppression) that makes
lethe improve with use?

## Context

Lethe already dedupes at write time, in two layers:

- **Exact:** SHA-256 `content_hash` guard in `MemoryStore::add` / `add_with_embedding`
  (`memory_store.rs:279,297`, index `idx_entries_hash`).
- **Near:** cosine argmax against all in-memory embeddings with hardcoded
  `dedup_threshold = 0.95` (`dedup.rs:34`, applied at `memory_store.rs:308-331`),
  policy = keep the longer content.

Both are undermined by the anchor line `<!-- session:S turn:T transcript:P -->` that
`build_chunk()` (`transcript_store.rs:41`) bakes into content *before* hashing and
embedding: identical turns from a copied/moved transcript never hash equal, and every
embedding carries anchor noise. Nothing ever cleans up what has already accumulated.

## Recommended Direction

Three layers, cheapest first.

### Layer 0 — anchor-aware canonicalization (bug-fix tier)

Strip the anchor before `chunk_hash` and before encoding; keep it as metadata. The
existing exact-SHA gate starts actually firing, and every cosine comparison downstream
sharpens. Requires a one-time id migration (or full re-index).

### Layer 1 — SemDeDup-style offline compaction (`lethe dedupe`)

State of the art for semantic dedup at scale (SemDeDup, Meta 2023): cluster embeddings,
compare pairs only within clusters — O(Σ nᵢ²) instead of O(n²). Lethe already maintains
k-means clusters for RIF (`cluster_centroids`) — reuse them.

Within each cluster:

1. Union-find chunks with cosine ≥ `dedup_threshold` into dupe-groups.
2. Elect a canonical: most-retrieved, then longest.
3. Merge metadata: sum `retrieval_count`, max `affinity`, min `suppression`, keep
   absorbed session anchors.
4. Delete the rest, recording absorbed ids in an **alias table** (absorbed id →
   canonical id).

The alias table is load-bearing: `TranscriptStore::sync` skips chunks via `live_ids()`,
so without it the next re-index resurrects every absorbed chunk. `sync` must consult
the alias table.

Expose `dedup_threshold` through `CliConfig` (same plumbing pattern as `n_clusters` in
`store_helpers.rs:89` and `commands/config.rs`). Ship as an explicit command first;
auto-run post-sync once trusted.

### Layer 2 — thin retrieval-time guard

After reranking, drop any hit with cosine ≥ τ against an already-selected hit and
backfill from deeper candidates (MMR-lite). Embeddings are already in memory
(`memory_store.rs:115`); ~40 lines. Guards result quality even between compaction runs.

## Key Assumptions to Validate

- [ ] **RIF clusters are fine-grained enough for dedupe** — near-dup pairs could
  straddle cluster boundaries. Test: on a real index, compare within-cluster dedupe
  recall vs. a one-off full pairwise pass; if misses > ~5%, also compare each chunk
  against its 2nd-nearest centroid's cluster.
- [ ] **0.95 cosine is the right τ after anchor-stripping** — sample 50 pairs at
  0.90/0.93/0.95/0.97 from a real index and eyeball; thresholds tuned on anchor-noisy
  embeddings don't transfer.
- [ ] **Metadata merge doesn't distort RIF scoring** — summing `retrieval_count` and
  maxing `affinity` changes suppression dynamics. Test: run the benchmark suite
  (BENCHMARKS.md) before/after a compaction pass; retrieval quality must not regress.
- [ ] **Alias table prevents resurrection** — re-run `lethe index` after `lethe dedupe`
  and assert chunk count is stable.

## MVP Scope

**In:**

- Anchor strip + id migration.
- `lethe dedupe` command: per-cluster union-find, canonical election, metadata merge,
  alias table consulted by `sync`.
- `dedup_threshold` in CLI config.
- Retrieval-time similarity guard.
- `--dry-run` flag reporting dupe-groups without deleting.

**Out:** everything below.

## Not Doing (and Why)

- **MinHash/SimHash-LSH ingest gate** — right tool only when insert latency is a
  measured problem; nothing suggests it is yet. Revisit if the per-insert linear scan
  shows up in profiles.
- **HNSW/ANN index** — solves search speed, not dedupe; separate decision, don't
  couple them.
- **LLM-judge or cross-encoder pair verification** — ~100× cost per pair for marginal
  precision at τ ≥ 0.95. Keep "heuristic" honest.
- **Non-destructive dupe-groups with a browsing UI** — keep-one-merge-metadata was the
  chosen policy; groups add machinery the TUI would then have to render. `--dry-run`
  covers the safety need.
- **RIF-native lazy dedupe** (fold losers into winners at retrieval time) — on-theme
  for the paper, but unvalidated convergence makes it research, not a plan.

## Open Questions

- Auto-run compaction after every `sync`, or manual-only with a staleness nudge in
  the TUI?
- Should the alias table also redirect `rescue_cache` entries pointing at absorbed
  ids, or just invalidate them?
- Does the id migration for anchor-stripping warrant a version bump in the DB schema /
  `transcript_manifest`?

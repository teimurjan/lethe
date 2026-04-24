# Research Journey

How `lethe` (originally `gc-memory`) evolved from a biology experiment into a three-stage retrieval system: hybrid IR → clustered RIF → LLM enrichment.

All numbers are NDCG@10 on the full LongMemEval S corpus (199,509 turns, 500 queries) unless explicitly scoped to a smaller eval. Benchmark methodology note at the bottom.

## Summary

| # | Approach | Result | Phase |
|---|----------|--------|-------|
| 1 | Direct embedding mutation (Gaussian noise, adaptive rate) | −5.9% to −2.8% | 1 |
| 2 | Adapter-based mutation + centroid fitness | −0.9% to +1.2% | 1 |
| 3 | Cross-encoder reranking | **+63%** over bi-encoder | 1 |
| 4 | Learned MLP adapter | +0.4% (stable) | 1 |
| 5 | Recall analysis (turning point) | diagnostic | 1 |
| 6 | Adaptive depth + rescue cache | +28.7% | 1 |
| 7 | Deduplication | +16.5% | 1 |
| 8 | BM25 hybrid retrieval | **+76%** over vector alone | 1 |
| 9 | GC routing index | −11% | 1 |
| 10 | Integrity audit | 0.3680 NDCG is from hybrid+xenc alone | 1 |
| 11 | Global RIF (cue-independent suppression) | +1.1% | 2 |
| 12 | Clustered RIF (cue-dependent, 30 clusters) | +5.8% | 2 |
| 13 | Clustered + rank-gap formula | **+6.5% / +9.5% recall@30 — retrieval-only best** | 2 |
| 14 | Exploration + rescue list | −2.6pp (negative) | 2 |
| 15 | Sparse Distributed Memory prototype | FAISS wins outright | 3 |
| 16 | Extended behavior metrics for ck 13 | RIF's gain is wrong_family only | 4 |
| 17 | LLM enrichment layer | **+8.3pp NDCG on covered — largest single lever** | 4 |
| 18 | Statistical rigor + NFCorpus replication | **clustered-RIF p<0.002 on LongMemEval; regresses on NFCorpus** | 5 |

**Current best retrieval pipeline:** hybrid BM25+vector → clustered+gap RIF → cross-encoder rerank (checkpoint 13). **Best overall:** that stack + write-time LLM enrichment (checkpoint 17, partial coverage). **Scope (checkpoint 18):** the mechanism is workload-specific — it helps on long-term conversational memory and does not generalize to non-conversational ad-hoc IR.

---

## Phase 1 — Biology-inspired mutation (checkpoints 1-10)

The original thesis: port germinal-center antibody mutation to vector memory. Adaptive mutation rate driven by retrieval affinity. Ten variants tested; none improved retrieval quality over a static hybrid+xenc baseline. What *did* work during this phase: standard IR techniques discovered along the way.

### Checkpoint 1 — Direct embedding mutation (NFCorpus, 3,633 docs)

Gaussian noise added to pretrained embeddings. Three arms: static, random-sigma, GC-adaptive-sigma.

| Arm | NDCG@10 | vs Static |
|-----|---------|-----------|
| Static | 0.3165 | baseline |
| Random | 0.2978 | −5.9% |
| GC | 0.3076 | −2.8% |

**Finding**: adaptive beats random (+3.3%), but both lose to static. Gaussian noise pushes embeddings off the pretrained manifold.
Bug found along the way: decay formula used cumulative `delta_steps`, producing `exp(−50.5)`. Fixed in `13e7f43`.

### Checkpoint 2 — Adapter-based mutation (LongMemEval, 19k sessions)

Froze base embeddings, added mutable adapter vectors. `effective = normalize(base + adapter)`, adapter norm clipped to 0.5.

| Arm | NDCG@10 | vs Static |
|-----|---------|-----------|
| Static | 0.1813 | baseline |
| Random adapter | 0.1834 | +1.2% |
| GC adapter (centroid fitness) | 0.1797 | −0.9% |

**Finding**: random adapter mutation beat static for the first time. Co-retrieval centroid as a fitness signal washed out whatever adaptive signal existed.

### Checkpoint 3 — Cross-encoder reranking (LongMemEval, 200k turns)

Added `ms-marco-MiniLM-L-6-v2` cross-encoder on top of FAISS candidates.

| Arm | NDCG@10 | vs Static |
|-----|---------|-----------|
| Static + rerank | 0.2217 | baseline |
| Random + rerank | 0.2206 | −0.5% |
| GC + rerank | 0.2202 | −0.7% |

**Finding**: reranking is **+63% over bi-encoder alone**. But on top of reranking, mutation adds nothing — the cross-encoder overrides FAISS ordering entirely.

### Checkpoint 4 — Learned MLP adapter

Replaced Gaussian noise with `delta = f(query, embedding, xenc_score)`, 148k-parameter MLP, trained online.

| Config | Peak NDCG | vs Static |
|--------|-----------|-----------|
| MLP (no gates) | 0.3362 | +0.6% |
| MLP + probability gate + low promotion | 0.3356 | +0.4% (stable through step 1250) |

**Finding**: first stable positive result. MLP learns the right direction, but deltas are too small to change FAISS rankings.

### Checkpoint 5 — Recall analysis (turning point)

Diagnosed the real bottleneck:

```
Bi-encoder recall@30:   29.0%
Bi-encoder recall@100:  43.8%
Bi-encoder recall@500:  63.6%
Oracle recall:          100%

k_fetch=30 + xenc:  NDCG=0.2217
k_fetch=100 + xenc: NDCG=0.2697 (+21.6%)
Oracle + xenc:      NDCG=0.4709 (+112%)
```

78% of relevant entries never reach the cross-encoder. The mutation was optimizing the wrong layer.

### Checkpoint 6 — Adaptive depth + rescue cache

Shallow k=30 for easy queries, deep k=200 when cross-encoder confidence is low. Rescue cache: deep-search results saved for future similar queries.

| Method | Hot NDCG | vs Static |
|--------|---------|-----------|
| Static (k=30 + xenc) | 0.1958 | baseline |
| Rescue cache only | 0.2203 | +12.6% |
| Adaptive k + rescue | 0.2853 | +28.7% |

### Checkpoint 7 — Deduplication

Cosine-similarity > 0.95 dedup with qrel remapping.

| Method | NDCG |
|--------|------|
| Original static | 0.2217 |
| Deduped static | 0.2582 (+16.5%) |
| Dedup + adaptive + rescue | 0.3017 (+54.1%) |

4.6% of entries are near-duplicates wasting retrieval slots.

### Checkpoint 8 — BM25 hybrid retrieval

Added BM25 keyword search alongside FAISS.

| System | NDCG@10 |
|--------|---------|
| Vector only | 0.1376 |
| BM25 only | 0.2420 |
| Memsearch-style (BM25+vector RRF) | 0.2171 |
| Vector + xenc | 0.2217 |
| **Full stack (dedup + BM25 + xenc)** | **0.3395** |

**Finding**: BM25 is the single biggest lever on conversation data — **+76% over vector alone**. Keyword matching beats semantic similarity for fact retrieval.

### Checkpoint 9 — GC routing index (final GC attempt)

Formalized rescue cache as a GC-managed routing index. Clusters query embeddings, maintains per-cluster entry associations with full tier lifecycle.

| System | Hot NDCG |
|--------|---------|
| Static hybrid+xenc | 0.3680 |
| + GC routing (2000 steps) | 0.3273 (−11%) |

Routing index added noise candidates that diluted the cross-encoder pool.

### Checkpoint 10 — Integrity audit

Independent audit: all systems, identical 200-query eval, identical qrels, top-10 results. Verified that the **0.3680 is from BM25+vector+cross-encoder alone**. No GC mechanism contributes.

**Phase 1 conclusion**: the biology-inspired control loop (adaptive rate, tier lifecycle, decay) is sound engineering for **memory management**. It is not useful for **retrieval quality**. Modern IR is the right tool for retrieval. Future work should focus on either a different retrieval mechanism or on lifecycle/cost layers.

---

## Phase 2 — Cognitive-science-inspired forgetting (checkpoints 11-14)

Retrieval-induced forgetting (Anderson, 1994; SAM model — Raaijmakers & Shiffrin, 1981): when memory A is retrieved, competing memories activated-but-not-selected are actively suppressed. Zero AI implementations. This phase tested it as a learned retrieval mechanism.

### Checkpoint 11 — Global RIF

After each retrieval, penalize the candidate-pool losers via a scalar per-entry `suppression`. Applied to RRF scores before the cross-encoder sees them. 2000-step burn-in, 200-query eval.

| Metric | Static | RIF | Δ |
|--------|--------|-----|---|
| NDCG@10 | 0.2960 | 0.3020 | **+2.0%** |
| Recall@30 | 0.4103 | 0.4196 | +2.3% |

Sweep: conservative config (α=0.3) was optimal; aggressive variants all went negative because cue-independent suppression hurts entries that are distractors for some queries but relevant for others.

### Checkpoint 12 — Clustered RIF

K-means cluster query embeddings (30 clusters); store suppression per `(entry, cluster)` pair. An entry suppressed for "travel" queries stays available for "food" queries. Based on SAM's cue-dependent retrieval probability.

| Config | NDCG@10 | vs baseline |
|--------|---------|-------------|
| baseline (no RIF) | 0.2960 | — |
| global RIF | 0.2993 | +1.1% |
| 10 clusters | 0.3113 | +5.2% |
| **30 clusters** | **0.3132** | **+5.8%** |
| 50 clusters | 0.3067 | +3.6% |
| 100 clusters | 0.3092 | +4.5% |

Inverted-U on cluster count: 10 too coarse, 100 too fine. 30 clusters ≈ 17 queries per cluster, enough signal without diluting.

### Checkpoint 13 — Rank-gap competition formula (retrieval-only best)

Original formula: `(1 − initial_rank/pool) × sigmoid(−xenc)`. Treats any top-ranked entry as a strong competitor when rejected.

Rank-gap formula: `max(0, xenc_rank − initial_rank) / pool × sigmoid(−xenc)`. Only penalizes entries that actually *dropped* in rank AND were actively rejected — not entries that just lost a close race.

| Config | NDCG@10 | Recall@30 |
|--------|---------|-----------|
| baseline | 0.2960 | 0.4103 |
| global + original | 0.2993 (+1.1%) | 0.4142 (+0.9%) |
| global + gap | 0.3037 (+2.6%) | 0.4250 (+3.6%) |
| clustered30 + original | 0.3132 (+5.8%) | 0.4381 (+6.8%) |
| **clustered30 + gap** | **0.3152 (+6.5%)** | **0.4494 (+9.5%)** |

Gap suppresses 30% fewer entries but produces stronger results — more targeted, not more aggressive. Biggest recall gain of any checkpoint.

### Checkpoint 14 — Exploration + rescue list (negative result)

Symmetric mechanism to RIF: pull up false negatives. Periodically fetch top-80 and score positions 31-80 with xenc; high-scoring finds enter a per-cluster rescue list; future queries in that cluster inject top-K rescues into the pool.

Iterated three times on a 100-query fast benchmark. Smart top-3 injection showed +2.5% vs +1.1% for RIF-only — looked promising.

Full-scale validation (3000 burn-in, 500-query eval) flipped the sign:

| Config | NDCG | vs baseline |
|--------|------|-------------|
| baseline | 0.2926 | — |
| clustered+gap | 0.3124 | +6.8% |
| +rescue-top3 | 0.3049 | +4.2% |
| +rescue-top5 | 0.3011 | +2.9% |
| +rescue-top10 | 0.3003 | +2.7% |

**Finding**: fast-benchmark +2.5% was variance against the noisy small-sample baseline. At scale, rescue *degrades* retrieval by −2.6pp. Causes: injection creates noise competitors that mislead RIF's winner/loser identification; fixed injection score creates bias; one-time xenc validation is too weak a persistence signal.

---

## Phase 3 — Alternative retrieval paradigms (checkpoint 15)

### Checkpoint 15 — Sparse Distributed Memory prototype (negative)

Built a from-scratch SDM (Kanerva, 1988) in `sdm/`: random binary hard locations in 512-bit address space, bipolar counters, top-N activation, optional iterative cleanup. Tested on a synthetic episodic dataset (250 events, 4 families with 5 near-duplicate siblings, 1000 queries across 4 noise modes).

Sanity check (exact queries, verifies implementation): SDM 206/250 (82%), SDM+cleanup 65/250 (26%), FAISS 233/250 (93%).

Full eval, precision@1:

| Mode | SDM | SDM+cleanup | FAISS |
|------|-----|-------------|-------|
| partial | 0.132 | 0.064 | 0.344 |
| paraphrase | 0.476 | 0.176 | 0.900 |
| fragment | 0.088 | 0.036 | 0.348 |
| noisy | 0.204 | 0.060 | 0.892 |
| **overall** | **0.225** | **0.084** | **0.621** |

**Findings**:
1. FAISS wins outright. Dense cosine beats random binary projection on this dataset.
2. SDM's gap to FAISS is *smallest on partial/fragment* (content missing), *largest on noisy/paraphrase* (content preserved). Binary quantization throws away signal exactly where dense cosine would exploit it.
3. SDM has *lower sibling confusion* on partial/fragment — when it fails, it tends to miss entirely rather than swap in a near-duplicate. Qualitatively different failure mode.
4. *Iterative cleanup HURTS* on every noise mode. Cleanup is a prototype-extraction attractor — destructive for distinct-episode recall.

**Conclusion**: the paradigm shift didn't yield a win. Closes the alternative-retrieval thread.

---

## Phase 4 — Behavior diagnosis and LLM enrichment (checkpoints 16-17)

### Checkpoint 16 — Extended behavior metrics for checkpoint 13

Before adding more mechanisms, decomposed checkpoint 13's +6.8% NDCG into behavior-level metrics:

- **exact_episode** — top-1 ∈ qrels
- **sibling_confusion** — top-1 from answer-session but not in qrels (right topic, wrong turn)
- **wrong_family** — top-1 from a session with no relevant turns
- **stale_fact** (knowledge-update only) — proxy for pulling an older version
- **abstain@T** — fraction where top-1 xenc score < T

Overall (500 queries, 5000-step burn-in):

| Metric | baseline | RIF | Δ |
|--------|----------|-----|---|
| exact_episode | 0.208 | 0.222 | +0.014 |
| ndcg@10 | 0.293 | 0.316 | +0.023 |
| **wrong_family** | **0.688** | **0.672** | **−0.016** |
| sibling_confusion | 0.104 | 0.106 | +0.002 |
| stale_fact | 0.205 | 0.205 | 0.000 |
| abstain@2 | 0.324 | 0.310 | −0.014 |

**Findings**:
1. RIF's NDCG gain is **primarily wrong_family reduction** (−1.6pp). Cross-topic pruning.
2. **Sibling confusion unchanged**. RIF doesn't operate at within-session granularity.
3. **Stale-fact rate identical**. RIF has no concept of time.
4. **Abstention stable**. Cross-encoder's rank-1 confidence doesn't shift.

Biggest per-type gains on temporal-reasoning and single-session-user. Story is "cross-topic noise pruning," not "general retrieval improvement."

### Checkpoint 17 — LLM enrichment layer (largest single lever)

Write-time structured extraction via `claude-haiku-4-5`. For each memory, produce: gist, 3 anticipated queries, entities, temporal markers. Indexed alongside original text in BM25/vector; cross-encoder still scores against the original.

Enriched 975 entries (first 1000 of the ~10k answer-relevant set), covering 15% of queries (75/500 have at least one enriched qrels entry).

**Overall (500 queries, 85% uncovered — diluted):**

| Metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.208 | 0.218 | 0.220 | +0.002 |
| ndcg@10 | 0.293 | 0.312 | 0.324 | +0.012 |
| wrong_family | 0.688 | 0.676 | 0.674 | −0.002 |

**Covered bucket (75 queries — where the mechanism applies):**

| Metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.267 | 0.280 | **0.333** | **+0.053 (+19% rel)** |
| ndcg@10 | 0.350 | 0.390 | **0.473** | **+0.083 (+21% rel)** |
| wrong_family | 0.720 | 0.707 | **0.653** | **−0.053** |
| sibling_confusion | 0.013 | 0.013 | 0.013 | 0 |
| abstain@4 | 0.280 | 0.253 | 0.227 | −0.027 (more confident) |

**Findings**:
1. **Enrichment works where applied.** +8.3pp NDCG and +5.3pp exact_episode over RIF — 3.6× larger than RIF's own contribution. **Largest single-lever gain in the whole journey.**
2. Driven by **wrong_family reduction** (−5.3pp). Anticipated queries shrinks vocabulary mismatch.
3. **Sibling confusion still doesn't move**. When all in-session turns get enriched, they all get richer simultaneously.
4. **Stale-fact didn't get a fair test**. Knowledge-update qrels fell outside the 1000-entry subset; temporal markers not exercised.
5. **Abstention drops** (more confident) on covered queries.

**Cost**: $1.6 for 1000 entries on Haiku (prompt caching off — system prompt too short to cache on this model). Projected $16 for full coverage.

---

## Phase 5 — Statistical rigor and cross-dataset scope (checkpoint 18)

### Checkpoint 18 — Bootstrap CIs, permutation tests, and NFCorpus replication

Before claiming "clustered RIF improves retrieval" in the arXiv writeup, we re-ran the checkpoint-13 benchmark with per-query NDCG arrays persisted (`benchmarks/run_rif_gap.py` patched to dump `rif_gap_per_query.json`), then computed 95% bootstrap percentile CIs (10k resamples) and two-sided paired permutation test p-values (10k permutations) via `benchmarks/bootstrap_rif_gap_ci.py`.

**LongMemEval S, 500-query eval, same 5000-step burn-in:**

| Config | NDCG@10 [95% CI] | Δ NDCG [95% CI] | p (perm) |
|--------|------------------|------------------|----------|
| baseline | 0.2960 [0.2647, 0.3281] | — | — |
| global-original | 0.2992 [0.2671, 0.3311] | +0.0032 [−0.0097, +0.0161] | 0.622 |
| global-gap | 0.3038 [0.2713, 0.3364] | +0.0079 [−0.0034, +0.0192] | 0.175 |
| clustered30-original | 0.3132 [0.2814, 0.3453] | **+0.0173 [+0.0063, +0.0285]** | **0.002** |
| **clustered30-gap** | **0.3152 [0.2836, 0.3473]** | **+0.0192 [+0.0102, +0.0292]** | **0.0001** |

**Findings (LongMemEval):**
1. **Clustering is the significant component.** Both clustered variants reject the null at Bonferroni-adjusted p<0.01; both global variants do not reject even without correction.
2. **Rank-gap refinement is not individually significant over the uniform rule.** Pairwise (clustered30-gap vs clustered30-original): Δ NDCG +0.0020 p=0.548, Δ Recall +0.0112 p=0.055. Direction consistent, significance not established at n=500. Report as an efficiency win (30% fewer suppressed entries), not a quality win.
3. Previous published numbers (+1.1% global, +6.5% clustered+gap) were correct as point estimates. The CIs reframe what they mean: the clustering gain is real, the global and rank-gap deltas are inside or at the edge of noise.

**NFCorpus (BEIR medical IR), 323-query eval, 3,000-step burn-in, identical hyperparameters:**

| Config | NDCG@10 | Δ NDCG [95% CI] | p (perm) |
|--------|---------|------------------|----------|
| baseline | 0.3462 [0.3120, 0.3812] | — | — |
| global-original | 0.3198 | **−0.0264 [−0.0398, −0.0144]** | **0.0001** |
| global-gap | 0.3341 | **−0.0121 [−0.0219, −0.0035]** | **0.007** |
| clustered30-original | 0.3247 | **−0.0215 [−0.0352, −0.0100]** | **0.0002** |
| clustered30-gap | 0.3423 | −0.0039 [−0.0101, +0.0021] | 0.199 |

**Findings (NFCorpus):**
1. **Three of four variants significantly regress.** Only clustered+gap stays within noise of baseline.
2. **Corpus saturation is the likely root cause.** 3,000 burn-in steps × 30 candidates over a 3,633-doc corpus with a 323-query pool sampled with replacement gives ~25 suppression updates per entry; 68% of the corpus accumulates non-zero suppression. On LongMemEval the same burn-in budget touches ~4% of the 199k corpus.
3. **Workload mismatch is the structural cause.** NFCorpus queries are independent medical questions, not a single user's overlapping information needs. The cue-dependent suppression mechanism requires recurring cues; when clusters correspond to static topic labels rather than recurring retrieval intents, within-cluster suppression does not generalize.

**Conclusion:** the mechanism is workload-specific. It targets the chronic-false-positive pattern characteristic of a single user's accumulating long-term conversation memory, and actively hurts on ad-hoc retrieval where that pattern does not exist. The arXiv paper (`arxiv/paper.tex`, §5.3) documents this scope explicitly; public write-ups (`writeup/post_v2.md`, `writeup/launch_posts.md`) reflect the narrower claim.

Artifacts: `benchmarks/results/rif_gap_per_query.json`, `benchmarks/results/rif_gap_per_query_nfcorpus.json`, `benchmarks/results/rif_gap_ci.md`, `benchmarks/results/rif_gap_nfcorpus_ci.md`.

---

## Benchmark methodology note

All numbers above are **NDCG@10 over turn-level retrieval on the full 199,509-turn LongMemEval S corpus** — needle-in-haystack among 200k candidates.

Other memory-tool benchmarks commonly report **recall@5 on per-query fresh DBs of ~50 sessions at session granularity**. That's a ~2000× easier task (random baseline 10% vs 0.005%); some implementations additionally leak ground truth via annotation fields at indexing time. Published numbers in the 95-99% range on that methodology are state-of-the-art *for that methodology* — they are not directly comparable to the numbers here.

A head-to-head comparison on a shared methodology (either side) is a separate experiment that hasn't been run.

---

### Implementation note: query-based vs entry-based clustering for production RIF

When wiring clustered RIF into `MemoryStore.retrieve()` (shipping checkpoint 13 to production), we tested two clustering strategies:

- **Entry-based**: cluster the corpus embeddings (199k turns). Available from the first query. Gave **+1.8% NDCG** — 3.6× less than the benchmark's +6.5%.
- **Query-based**: cluster the query embeddings (matching the benchmark protocol). Requires collecting queries before building centroids. Gave **+9.5% NDCG** (via MemoryStore, A/B validated against raw primitives at N=1000).

Root cause: cue-dependent suppression scopes forgetting to *query context* ("travel-like queries" vs "food-like queries"). Entry-based clusters group *content topics*, which is a weak proxy for user intent — a query about PostgreSQL pool config might land in a cluster of PostgreSQL install guides rather than a cluster of pooling-related queries.

Production implementation: `MemoryStore` collects query embeddings during `retrieve()`, builds centroids via k-means once `10 × n_clusters` queries have been seen, then freezes. Centroids persist to DuckDB across sessions.

---

### Implementation note: speed-up attempts that didn't pan out (int8 swap, CoreML EP)

Before finding the real latency lever, we tried two obvious ones that failed on this workload. Logging them so future explorers don't rediscover.

**1. Swap the bi-encoder to an int8-quantized variant.** Hypothesis: a smaller, pre-quantized ONNX (`BAAI/bge-small-en-v1.5` at 67 MB via `qdrant/bge-small-en-v1.5-onnx-q`) would halve load time and raise throughput.

| metric | all-MiniLM-L6-v2 (fp32, current) | bge-small-en-v1.5 (int8) |
|---|---|---|
| cold-start, median of 3 subprocess probes | 0.39 s | 0.44 s |
| warm throughput, **synthetic** fixed-length sentences | 108 items/s | **532 items/s** (4.9×) |
| warm throughput, **real LongMemEval conversation turns** | ~47 items/s | **~11 items/s** (0.23×, *regression*) |

The 4.9× win on synthetic text inverts to a ~4× regression on real conversational turns. Root cause: BGE-small's max context is 512 tokens vs MiniLM's 256; variable-length LongMemEval turns (often > 256 tokens) push BGE to ~2× the per-item compute, and the token-throughput differential dwarfs any tensor-width win from int8. Synthetic short sentences hide this by staying well under both caps.

We did not complete the NDCG arm (killed at ~20% of corpus re-embed once the throughput inversion made the speed premise moot). The quality question is open, but there's no speed incentive to answer it on this workload.

Reproducer: `benchmarks/run_int8.py`. Raw table: `benchmarks/results/BENCHMARKS_INT8.md`.

**2. CoreML execution provider on Apple Silicon.** Hypothesis: route onnxruntime through the Neural Engine / Metal via `providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]`.

| metric | CPU EP | CoreML EP |
|---|---|---|
| bi-encoder load + warm | 114 ms | **1195 ms** (10.5× slower) |
| bi-encoder embed 150 docs | 1249 ms | **9427 ms** (7.5× slower) |
| xenc load + warm | 68 ms | **1606 ms** (23× slower) |
| xenc rerank 60 pairs | 229 ms | **1886 ms** (8× slower) |

Classic "model too small for accelerator dispatch." onnxruntime's CoreML partitioner only covers ~72 % of the graph (232/323 nodes); the remaining 28 % run on CPU, so every forward pass pays a round-trip Metal/ANE transfer cost that would only amortize on models ~5-10× larger than MiniLM-L6. The same calculus applies to `fastembed-gpu` (CUDA on NVIDIA) — small MiniLM-family models don't saturate a GPU either.

Takeaway: for the current 22M-param bi-encoder + 22M-param cross-encoder, the CPU path is Pareto-optimal, and the latency lever lives in the retrieval pipeline (rerank pool sizing), not the hardware provider — leading to the `k_deep=200 → 100` calibration below.

---

### Implementation note: adaptive deep-pass `k_deep` re-calibration (200 → 100)

Checkpoint 6 introduced the adaptive depth: shallow `k=30`, deep `k=200` when cross-encoder confidence is low. The 200 was never measured against smaller alternatives — it was picked as "large enough to never hurt." Once the TUI started making the deep pass user-visible (cross-project search over N registered projects pays `N × k_deep` rerank cost in the worst case), we re-ran the sweep.

LongMemEval S, 100-query random sample (seed 0), full production pipeline (BM25 + FAISS → RRF → cross-encoder rerank), deep pass fires on 59% of queries:

| config | NDCG@10 | Recall@10 | ΔNDCG vs baseline | p50 | p95 |
|---|---|---|---|---|---|
| shallow-only (no deep pass) | 0.2866 | 0.3493 | −1.56 pp | 1689 ms | 2371 ms |
| k_deep=60  | 0.2910 | 0.3527 | −1.11 pp | 4159 ms | 5936 ms |
| **k_deep=100** | **0.3022** | **0.3731** | **0.00 pp** | **5651 ms** | **7440 ms** |
| k_deep=200 (old) | 0.3022 | 0.3727 | baseline | 9800 ms | 12650 ms |

`k_deep=100` gives **identical NDCG@10 and Recall@10** to the old 200 while cutting p50 ~42% and p95 ~41%. The cross-encoder's top-10 picks stabilize by merged rank ~100 on this workload — ranks 101-200 never won a top-10 slot, so the old 200 was pure latency tax.

`k_deep=60` is cheaper still but costs 1.1 pp NDCG, above the noise floor — rejected. If a user ever reports a query pattern where 100 is insufficient, the knob is exposed on the `MemoryStore` / `UnionStore` constructors.

Reproducer: `benchmarks/run_deep_pass.py`. Raw table: `benchmarks/results/BENCHMARKS_DEEP_PASS.md`.

### Implementation note: BM25 tokenizer (`lower().split()` → regex word-tokens)

A Codex code review flagged that BM25 tokenization was still `text.lower().split()` — the first implementation from checkpoint 8. Under `split()`, `"MongoDB?"` and `"MongoDB"` are different tokens, so any query ending in punctuation silently misses the matching corpus turn. We swept four tokenizers on a 100-query random sample (seed 0), shipped pipeline, `k_deep=100`:

| Tokenizer | NDCG@10 | Recall@10 | BM25 build |
|-----------|---------|-----------|------------|
| baseline (`lower().split()`) | 0.3022 | 0.3731 | 11.2s |
| **regex `[A-Za-z0-9_]+`** | **0.3390 (+3.68 pp)** | **0.4410 (+6.79 pp)** | 10.4s |
| regex + stopword removal | 0.3084 (+0.63 pp) | 0.4027 | 10.6s |
| regex + Porter stemming | 0.3153 (+1.31 pp) | 0.4143 | 187.8s |

Two non-obvious results:

- **Stopword removal regresses.** Dropping function words ("the", "of", "is", …) erases most of the gain. Conversational memory has short, specific queries where function words act as syntactic anchors — removing them is net-harmful on this corpus.
- **Porter stemming is a trap.** Build cost jumps 17× (10s → 188s), NDCG tops out at +1.31 pp — well below plain regex. The over-conflation cost (e.g. `generate`/`general` → `gener`) exceeds the vocabulary-compression gain on conversational text.

On the full 200-query headline benchmark the swap lifted the production pipeline from **0.3680 → 0.3817 NDCG@10** (BM25-only jumped 0.2420 → 0.3171). Bigger single-lever quality win than clustered-RIF on the same corpus, measured on the same eval sample. The lift is punctuation-only — the 200k conversational turns are heavy on trailing `?`, `.`, contractions, and hook-written session anchors that previously didn't tokenize cleanly.

Reproducer: `benchmarks/run_bm25_tokenizer.py`. Raw table: `benchmarks/results/BENCHMARKS_BM25_TOKENIZER.md`.

#### Downstream effect on RIF: absolute gains up, relative gains halved

Re-ran the full checkpoint-11/12/13 bench with the regex tokenizer (500-query full eval, 5000-step burn-in):

| Config | NDCG@10 | Rel gain | Prev tokenizer |
|--------|---------|----------|----------------|
| Baseline (no RIF) | 0.3311 | — | 0.2960 (+3.51pp from tokenizer) |
| Global RIF (gap) | 0.3369 | +1.8% | 0.3037 (+2.6%) |
| Clustered RIF 30 (gap, prod default) | 0.3422 | +3.4% | 0.3152 (+6.5%) |
| Clustered RIF 10 (gap) | 0.3444 | +4.0% | — |

Two findings, both expected under a "better base leaves less to recover" mental model:

- **Absolute numbers improved across the board.** Clustered RIF 30 absolute went 0.3152 → 0.3422 (+2.70 pp); Recall@30 went 0.4494 → 0.4972 (+4.78 pp). The mechanism is net-positive on the stronger baseline.
- **Relative gains halved.** Clustered RIF 30's NDCG gain went from +6.5% to +3.4%. Global RIF is now basically flat (+1.8%, measured separately, +0.1% here). This strengthens the original checkpoint-12/13 story: the cue-dependence of *clustered* RIF is what keeps it in the money when the base retrieval improves; global suppression does not.
- **10-clusters narrowly beats 30-clusters on NDCG** (+4.0% vs +3.4%) but 30-clusters wins Recall@30. Single-sample finding — CIs on this workload typically overlap in that range, so not switching defaults without a proper sweep.

Significance testing (paired permutation, bootstrap CI) from checkpoint 18 was on the old tokenizer. The clustered-RIF claim is expected to survive re-testing (absolute effect size did not collapse), but that hasn't been re-verified.

Reproducer: `benchmarks/run_rif_clustered.py`. Raw table: `benchmarks/results/BENCHMARKS_RIF_CLUSTERED.md`.

---

## What's next

Four open directions, in rough priority:

1. **Scale enrichment to full answer-relevant coverage** (~$16, ~1h enrichment + 3.5h benchmark). Confirms covered-bucket numbers on the full eval. Gets us the clean +8pp NDCG story.
2. **Replicate clustered RIF on a second long-term conversation memory benchmark** (e.g., LoCoMo, MSC, LongMemEval M). The NFCorpus negative result shows the mechanism doesn't transfer to ad-hoc IR; a second in-scope dataset would strengthen the conversational-memory claim independently.
3. **Head-to-head against other memory tools on shared methodology.** Run our system under their setup (per-query ~50 sessions, recall@5), and/or theirs under ours. Honest comparison.
4. **Move failure modes that haven't budged**: sibling_confusion (within-session discrimination) and stale_fact (temporal awareness). Candidate mechanisms: session-structured reranking, temporal-aware tie-breaking, explicit fact extraction with validity windows.

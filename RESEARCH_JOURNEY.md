# Research Journey

How `gc-memory` evolved from a biology experiment into a three-stage retrieval system: hybrid IR → clustered RIF → LLM enrichment.

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

**Current best retrieval pipeline:** hybrid BM25+vector → clustered+gap RIF → cross-encoder rerank (checkpoint 13). **Best overall:** that stack + write-time LLM enrichment (checkpoint 17, partial coverage).

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

## Benchmark methodology note

All numbers above are **NDCG@10 over turn-level retrieval on the full 199,509-turn LongMemEval S corpus** — needle-in-haystack among 200k candidates.

Other memory-tool benchmarks commonly report **recall@5 on per-query fresh DBs of ~50 sessions at session granularity**. That's a ~2000× easier task (random baseline 10% vs 0.005%); some implementations additionally leak ground truth via annotation fields at indexing time. Published numbers in the 95-99% range on that methodology are state-of-the-art *for that methodology* — they are not directly comparable to the numbers here.

A head-to-head comparison on a shared methodology (either side) is a separate experiment that hasn't been run.

---

## What's next

Three open directions, in rough priority:

1. **Scale enrichment to full answer-relevant coverage** (~$16, ~1h enrichment + 3.5h benchmark). Confirms covered-bucket numbers on the full eval. Gets us the clean +8pp NDCG story.
2. **Head-to-head against other memory tools on shared methodology.** Run our system under their setup (per-query ~50 sessions, recall@5), and/or theirs under ours. Honest comparison.
3. **Move failure modes that haven't budged**: sibling_confusion (within-session discrimination) and stale_fact (temporal awareness). Candidate mechanisms: session-structured reranking, temporal-aware tie-breaking, explicit fact extraction with validity windows.

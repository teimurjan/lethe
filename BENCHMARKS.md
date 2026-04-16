# Benchmarks

Canonical results for `gc-memory`. For the narrative (why each experiment was run, what failed, how we got here), see [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).

## Methodology

### Dataset
- **LongMemEval S variant** via HuggingFace `xiaowu0162/longmemeval-cleaned`.
- **199,509 conversation turns** (the full "haystack") + **500 evaluation questions** with 1–3 relevant turns each, annotated via qrels.

### Retrieval setup
- Global corpus — all 200k turns indexed together, no per-query pre-filtering.
- Turn-level retrieval (not session-level).
- Return top-10 results per query.

### Primary metric — NDCG@10
Position-sensitive; penalizes putting the right turn at rank 8 vs rank 1. Ideal DCG computed from qrels.

### Supporting metrics
- **Recall@k** (k ∈ {5, 10, 30}) — fraction of qrels hits appearing in top-k.
- **exact_episode** — top-1 ∈ qrels (precision@1 by another name).
- **sibling_confusion** — top-1 is from an answer-session but not in qrels (right topic, wrong turn).
- **wrong_family** — top-1 is from a session with no relevant turns (unrelated topic).
- **stale_fact** (knowledge-update only) — top-1 from answer-session but not in qrels (proxy for older version of the fact).
- **abstain@T** — fraction where top-1 xenc score < T (would trigger abstention).

### Burn-in protocol for RIF results
RIF is a learned mechanism — needs usage history to accumulate suppression. All RIF numbers below use a **5000-step burn-in** (query schedule: 70% hot queries drawn from a 20% subset, 30% cold queries) before the eval pass.

### Integrity checks
- All systems evaluated on identical queries with identical qrels.
- All systems return exactly 10 results to the scorer.
- No hyperparameters tuned on the eval set.
- RRF vs our hybrid merge produce identical numbers on the baseline — confirmed the 0.3680 number is from the cross-encoder, not from the merge strategy (see checkpoint 10).

---

## Results

### Retrieval baselines

200-query eval sample, checkpoint 10 audit:

| System | NDCG@10 | Recall@10 | Notes |
|--------|---------|-----------|-------|
| Vector only (MiniLM top-10) | 0.1376 | 0.2173 | Bi-encoder baseline |
| BM25 only (top-10) | 0.2420 | 0.3264 | Sparse keyword baseline |
| Hybrid BM25+vector RRF (memsearch style) | 0.2171 | 0.3334 | Rank fusion, no reranker |
| Vector + cross-encoder rerank (k=30) | 0.2425 | 0.2892 | Dense + reranker |
| **Hybrid + cross-encoder rerank (k=30)** | **0.3680** | **0.4694** | **Best without any learned component** |

500-query full eval (the default for all Phase 2+ results):

| System | NDCG@10 | Recall@30 |
|--------|---------|-----------|
| Baseline (hybrid + xenc, no RIF) | 0.2960 | 0.4103 |

### Retrieval-induced forgetting (RIF)

Checkpoints 11–13. All on 500-query full eval, 5000-step burn-in.

| Config | NDCG@10 | Recall@30 | Notes |
|--------|---------|-----------|-------|
| Global RIF (original formula) | 0.2993 (+1.1%) | 0.4142 (+0.9%) | Cue-independent; conservative is optimal |
| Global RIF (gap formula) | 0.3037 (+2.6%) | 0.4250 (+3.6%) | Gap formula alone is 2.4× better than original |
| Clustered RIF 30 (original) | 0.3132 (+5.8%) | 0.4381 (+6.8%) | K-means 30 query clusters, per-cluster suppression |
| **Clustered RIF 30 + gap formula** | **0.3152 (+6.5%)** | **0.4494 (+9.5%)** | **Best retrieval-only configuration** |

Default config for the best row: `alpha=0.3`, `suppression_rate=0.1`, `reinforcement_rate=0.05`, `decay_lambda=0.005`, `n_clusters=30`, `use_rank_gap=True`.

**Mechanism**: entries that make the candidate pool but lose to the cross-encoder accumulate a per-cluster suppression score. On subsequent retrievals in the same query cluster, suppression penalizes RRF scores before the cross-encoder sees them.
- **Clustered**: `{entry_id → {cluster_id → float}}` instead of `{entry_id → float}`. An entry suppressed for "travel" queries stays available for "food" queries.
- **Gap formula**: `max(0, xenc_rank − initial_rank) / pool × sigmoid(−xenc)`. Only penalizes entries that actually dropped in rank AND were actively rejected.

### Extended behavior metrics (checkpoint 16)

Decomposes the +6.5% NDCG gain into behavior-level changes. 500-query eval.

| Metric | baseline | clustered+gap RIF | Δ |
|--------|----------|-------------------|---|
| exact_episode | 0.208 | 0.222 | +0.014 |
| ndcg@10 | 0.293 | 0.316 | +0.023 |
| **wrong_family** | **0.688** | **0.672** | **−0.016** |
| sibling_confusion | 0.104 | 0.106 | +0.002 |
| stale_fact (knowledge-update only) | 0.205 | 0.205 | 0.000 |
| abstain@0 | 0.180 | 0.166 | −0.014 |
| abstain@2 | 0.324 | 0.310 | −0.014 |
| abstain@4 | 0.512 | 0.506 | −0.006 |

**Read**: the NDCG gain comes primarily from **reducing cross-topic retrieval** (−1.6pp `wrong_family`). RIF has no effect on within-session discrimination (`sibling_confusion`) or temporal awareness (`stale_fact`).

### LLM enrichment layer (checkpoint 17)

Write-time structured extraction via `claude-haiku-4-5`: gist, 3 anticipated queries, entities, temporal markers. Indexed alongside original text in BM25/vector. Cross-encoder still scores original text.

Coverage: 975 entries enriched (15% of queries have ≥1 enriched qrels entry). Partial-coverage result — overall numbers are diluted by the 85% of uncovered queries.

**Overall (500 queries):**

| Metric | baseline | RIF | RIF + enrichment | Δ(enr−RIF) |
|--------|----------|-----|------------------|-----------|
| exact_episode | 0.208 | 0.218 | 0.220 | +0.002 |
| ndcg@10 | 0.293 | 0.312 | 0.324 | +0.012 |
| wrong_family | 0.688 | 0.676 | 0.674 | −0.002 |

**Covered bucket (75 queries — where the mechanism applies):**

| Metric | baseline | RIF | RIF + enrichment | Δ(enr−RIF) |
|--------|----------|-----|------------------|-----------|
| exact_episode | 0.267 | 0.280 | **0.333** | **+0.053 (+19% rel)** |
| ndcg@10 | 0.350 | 0.390 | **0.473** | **+0.083 (+21% rel)** |
| wrong_family | 0.720 | 0.707 | **0.653** | **−0.053** |
| sibling_confusion | 0.013 | 0.013 | 0.013 | 0 |
| abstain@4 | 0.280 | 0.253 | 0.227 | −0.027 (more confident) |

The covered-bucket +8.3pp NDCG gain is **3.6× larger than RIF's own contribution** — the largest single-lever improvement in the project. Mechanism: anticipated_queries reduces vocabulary mismatch, so correct sessions reach the candidate pool more often.

**Cost**: $1.60 for 1000 entries on Haiku; system prompt is below the 4096-token cache minimum, so no prompt caching. Projected ~$16 for full coverage of the ~10k answer-relevant entries.

### What doesn't work (negative results)

Summarized here; full numbers in [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).

| Approach | Phase | Result |
|----------|-------|--------|
| Embedding mutation (direct Gaussian, adapter, MLP) | 1 | No gain or negative on retrieval quality |
| Text segmentation | 1 | −46% NDCG |
| Co-relevance graph | 1 | −2.7% on LongMemEval (too sparse) |
| GC routing index | 1 | −11% |
| Exploration + rescue list (top-K injection) | 2 | −2.6pp at scale (looked positive on small eval, variance) |
| Sparse Distributed Memory (binary LSH + cleanup) | 3 | FAISS wins outright on synthetic episodic data |
| Iterative cleanup on SDM | 3 | Drifts to sibling prototypes — ≈3× precision drop |

---

## Benchmark methodology note (comparison with other tools)

All numbers on this page measure **full-corpus NDCG@10 with 200k-turn needle-in-haystack retrieval**. This is deliberately close to production — in a real deployed memory, you don't know upfront which sessions are relevant to a query.

Other memory-tool benchmarks commonly report **recall@5 over per-query fresh DBs of ~50 sessions at session granularity** — a ~2000× easier task (random baseline 10% vs 0.005%). Some implementations additionally leak ground-truth signal at indexing time (e.g. filtering assistant turns on `has_answer=true`, which is a dataset annotation). Published numbers in the 95-99% range on that methodology are state-of-the-art *for that methodology*; they are not comparable to numbers here.

Running either side under a shared methodology is a separate experiment that hasn't been done.

---

## How to reproduce

```bash
# Prep data
uv run python experiments/data_prep.py --dataset longmemeval

# Retrieval-only
uv run python benchmarks/run_benchmark.py          # checkpoint 10 baselines
uv run python benchmarks/run_rif_benchmark.py      # checkpoint 11 (global RIF)
uv run python benchmarks/run_rif_clustered.py      # checkpoint 12
uv run python benchmarks/run_rif_gap.py            # checkpoint 13 (best retrieval-only)
uv run python benchmarks/run_rif_extended_metrics.py  # checkpoint 16

# LLM enrichment (needs ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
uv run python experiments/enrich_longmemeval.py    # ~$16 for 10k entries
uv run python benchmarks/run_rif_enriched.py       # checkpoint 17 (3-arm)
```

Raw per-run outputs from each benchmark script are checked into `benchmarks/results/` for auditability.

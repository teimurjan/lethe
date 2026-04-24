# Benchmarks

Canonical results for `lethe`. For the narrative (why each experiment was run, what failed, how we got here), see [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).

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

200-query eval sample. Current numbers use the regex BM25 tokenizer (`[A-Za-z0-9_]+`), which replaced `lower().split()` on 2026-04-24. See [BENCHMARKS_BM25_TOKENIZER.md](results/BENCHMARKS_BM25_TOKENIZER.md) for the ablation — the change stripped trailing punctuation (so `"MongoDB?"` matches `"mongodb"`) and beat three alternative tokenizers on the 100-query sweep.

| System | NDCG@10 | Recall@10 | Notes |
|--------|---------|-----------|-------|
| Vector only (MiniLM top-10) | 0.1376 | 0.2173 | Bi-encoder baseline; no BM25, unchanged |
| BM25 only (top-10) | 0.3171 | 0.4152 | Sparse keyword baseline (was 0.2420 with `lower().split()`) |
| Hybrid BM25+vector RRF (memsearch style) | 0.2408 | 0.3554 | Rank fusion, no reranker (was 0.2171) |
| Vector + cross-encoder rerank (k=30) | 0.2425 | 0.2892 | Dense + reranker; no BM25, unchanged |
| **Hybrid + cross-encoder rerank (k=30)** | **0.3817** | **0.4964** | **Best without any learned component** (was 0.3680) |

500-query full eval (the default for all Phase 2+ results). Current values use the regex BM25 tokenizer (2026-04-24); original `lower().split()` numbers kept for reference.

| System | NDCG@10 | Recall@30 | Prev (lower+split) |
|--------|---------|-----------|--------------------|
| Baseline (hybrid + xenc, no RIF) | **0.3311** | **0.4739** | 0.2960 / 0.4103 |

### Retrieval-induced forgetting (RIF)

Checkpoints 11–13. All on 500-query full eval, 5000-step burn-in. Re-measured 2026-04-24 with the regex BM25 tokenizer — relative gains shrink because the baseline is stronger, but absolute NDCG and Recall both improve across every config.

| Config | NDCG@10 | Recall@30 | Prev (lower+split) |
|--------|---------|-----------|--------------------|
| Global RIF (gap formula) | 0.3369 (+1.8%) | 0.4741 (+0.0%) | 0.3037 (+2.6%) / 0.4250 (+3.6%) |
| Clustered RIF 10 (gap formula) | **0.3444 (+4.0%)** | 0.4917 (+3.8%) | — |
| **Clustered RIF 30 (gap formula)**¹ | **0.3422 (+3.4%)** | **0.4972 (+4.9%)** | 0.3152 (+6.5%) / 0.4494 (+9.5%) |
| Clustered RIF 50 (gap formula) | 0.3385 (+2.3%) | 0.4942 (+4.3%) | — |
| Clustered RIF 100 (gap formula) | 0.3411 (+3.0%) | 0.4900 (+3.4%) | — |

¹ Production default config. 10-clusters beat 30-clusters on NDCG (+4.0% vs +3.4%) on this single run; recommend a proper sweep with confidence intervals before considering a default change — CIs likely overlap given n=500.

Default config for the production row: `alpha=0.3`, `suppression_rate=0.1`, `reinforcement_rate=0.05`, `decay_lambda=0.005`, `n_clusters=30`, `use_rank_gap=True`. Reproducer: `benchmarks/run_rif_clustered.py`. Stat-significance tests (paired permutation, bootstrap CI) from checkpoint 18 were on the old tokenizer; results with the new tokenizer have not been re-verified for significance.

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
| **RIF transfer to NFCorpus (non-conversational IR)** | **5** | **3 of 4 variants significantly regress (p ≤ 0.02); mechanism is workload-specific (checkpoint 18)** |

### Scope (checkpoint 18)

The retrieval-only gains above hold on LongMemEval S (long-term conversational memory). Running the same five RIF configurations on NFCorpus (BEIR medical IR, 3,633 docs, 323 queries):

| Config | NFCorpus NDCG@10 | Δ vs baseline | p (perm) |
|--------|------------------|---------------|----------|
| Baseline | 0.3462 | — | — |
| Global RIF (original) | 0.3198 | **−0.0264** | **0.0001** |
| Global RIF (gap) | 0.3341 | **−0.0121** | **0.007** |
| Clustered RIF (original) | 0.3247 | **−0.0215** | **0.0002** |
| Clustered RIF + gap | 0.3423 | −0.0039 | 0.199 |

Three of four variants significantly regress. Only clustered+gap stays within noise of baseline. Diagnosis: (i) corpus saturation — 68% of the 3,633-doc corpus has non-zero suppression by the end of 3,000-step burn-in vs 4% on the 199k LongMemEval corpus; (ii) workload mismatch — independent medical queries don't share the recurring information needs RIF's cue-dependent suppression depends on.

**Interpretation:** the mechanism is scoped to long-term conversational memory. Don't apply it to general ad-hoc retrieval.

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

# Statistical rigor + cross-dataset scope (checkpoint 18)
uv run python benchmarks/bootstrap_rif_gap_ci.py   # LongMemEval CIs + perm tests
uv run python benchmarks/run_rif_gap_nfcorpus.py   # NFCorpus replication (~25 min)
uv run python benchmarks/bootstrap_rif_gap_ci.py \
    --input  benchmarks/results/rif_gap_per_query_nfcorpus.json \
    --output benchmarks/results/rif_gap_nfcorpus_ci.md
```

Raw per-run outputs from each benchmark script are checked into `benchmarks/results/` for auditability.

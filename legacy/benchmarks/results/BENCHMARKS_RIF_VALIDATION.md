# Full Validation: Top-K Rescue Injection

Validates fast-benchmark findings on full 500-query eval.

Burn-in: 3000 steps, Eval: 500 queries
Base: clustered30+gap RIF (checkpoint 13)
Date: 2026-04-15

| Config | explore_every | inject_top_k | NDCG@10 | Δ | Recall@30 | Δ | Rescued |
|--------|---------------|--------------|---------|---|-----------|---|---------|
| baseline-no-rif | — | — | 0.2926 | +0.0% | 0.4065 | +0.0% | 0 |
| clustered+gap | — | — | 0.3124 | +6.8% | 0.4481 | +10.3% | 0 |
| +rescue-top3 | 5 | 3 | 0.3049 | +4.2% | 0.4366 | +7.4% | 1006 |
| +rescue-top5 | 5 | 5 | 0.3011 | +2.9% | 0.4300 | +5.8% | 1006 |
| +rescue-top10 | 5 | 10 | 0.3003 | +2.7% | 0.4276 | +5.2% | 1006 |

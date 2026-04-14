# Rescue-list Threshold Sweep

Burn-in: 800 steps, Eval: 100 queries
Base RIF: clustered30+gap (current best)
Date: 2026-04-14

| Config | explore_every | threshold | NDCG@10 | Δ | Recall@30 | Δ | Rescued |
|--------|---------------|-----------|---------|---|-----------|---|---------|
| baseline-no-rif | — | — | 0.3029 | +0.0% | 0.4333 | +0.0% | 0 |
| current-best | — | — | 0.3061 | +1.1% | 0.4400 | +1.5% | 0 |
| sparse-t1.0 | 20 | 1.0 | 0.3082 | +1.8% | 0.4400 | +1.5% | 127 |
| sparse-t2.0 | 20 | 2.0 | 0.3065 | +1.2% | 0.4400 | +1.5% | 71 |
| sparse-t3.0 | 20 | 3.0 | 0.3061 | +1.1% | 0.4400 | +1.5% | 46 |
| moderate-t2.0 | 5 | 2.0 | 0.3075 | +1.6% | 0.4383 | +1.2% | 232 |
| moderate-t3.0 | 5 | 3.0 | 0.3065 | +1.2% | 0.4383 | +1.2% | 146 |
| dense-t3.0 | 1 | 3.0 | 0.3033 | +0.1% | 0.4283 | -1.2% | 486 |

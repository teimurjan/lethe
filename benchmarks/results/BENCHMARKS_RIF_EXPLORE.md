# Exploration + Rescue List Benchmark (fast)

Symmetric mechanism: RIF pushes false positives down, rescue list pulls false negatives up.

Burn-in: 800 steps, Eval: 100 queries (random sample)
Pool: top-30 normal, top-80 exploration zone
RIF params: clustered30+gap (current best)
Date: 2026-04-14

| Config | explore_every | NDCG@10 | Δ | Recall@30 | Δ | Rescued |
|--------|---------------|---------|---|-----------|---|---------|
| baseline-no-rif | — | 0.3029 | +0.0% | 0.4333 | +0.0% | 0 |
| current-best | — | 0.3061 | +1.1% | 0.4400 | +1.5% | 0 |
| +rescue-sparse | 20 | 0.3082 | +1.8% | 0.4400 | +1.5% | 127 |
| +rescue-moderate | 5 | 0.3078 | +1.6% | 0.4383 | +1.2% | 344 |
| +rescue-dense | 1 | 0.3004 | -0.8% | 0.4287 | -1.1% | 804 |

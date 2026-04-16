# Smart Rescue Injection Benchmark

Top-K + optional bandit decay/prune.

Burn-in: 800 steps, Eval: 100 queries
Base RIF: clustered30+gap (current best)
Date: 2026-04-14

| Config | inject_top | bandit_decay | NDCG@10 | Δ | Recall@30 | Δ | Rescued |
|--------|------------|--------------|---------|---|-----------|---|---------|
| baseline-no-rif | — | — | 0.3029 | +0.0% | 0.4333 | +0.0% | 0 |
| current-best | — | — | 0.3061 | +1.1% | 0.4400 | +1.5% | 0 |
| rescue-all | all | — | 0.3028 | -0.0% | 0.4233 | -2.3% | 456 |
| rescue-top3 | 3 | — | 0.3104 | +2.5% | 0.4433 | +2.3% | 456 |
| rescue-top5 | 5 | — | 0.3065 | +1.2% | 0.4383 | +1.2% | 456 |
| rescue-top10 | 10 | — | 0.3078 | +1.6% | 0.4383 | +1.2% | 456 |
| rescue-bandit-soft | 5 | 0.3 | 0.3067 | +1.3% | 0.4383 | +1.2% | 456 |
| rescue-bandit-hard | 5 | 0.5 | 0.3068 | +1.3% | 0.4383 | +1.2% | 364 |

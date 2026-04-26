# Rank-gap RIF Benchmark

Competition strength from rank drop (initial vs xenc) instead of rank alone.

Burn-in: 5000 steps, Eval: 500 queries
RIF params: alpha=0.3, rate=0.1, decay=0.005 (all configs)
Date: 2026-04-21

| Config | Clusters | Formula | NDCG@10 | Δ | Recall@30 | Δ |
|--------|----------|---------|---------|---|-----------|---|
| baseline | global | original | 0.2960 | +0.0% | 0.4103 | +0.0% |
| global-original | global | original | 0.2992 | +1.1% | 0.4142 | +0.9% |
| global-gap | global | gap | 0.3038 | +2.7% | 0.4258 | +3.8% |
| clustered30-original | 30 | original | 0.3132 | +5.8% | 0.4381 | +6.8% |
| clustered30-gap | 30 | gap | 0.3152 | +6.5% | 0.4494 | +9.5% |

# Rank-gap RIF Benchmark (NFCorpus)

Competition strength from rank drop (initial vs xenc) instead of rank alone.

Burn-in: 3000 steps, Eval: 323 queries
RIF params: alpha=0.3, rate=0.1, decay=0.005 (all configs)
Date: 2026-04-21

| Config | Clusters | Formula | NDCG@10 | Δ | Recall@30 | Δ |
|--------|----------|---------|---------|---|-----------|---|
| baseline | global | original | 0.3462 | +0.0% | 0.2131 | +0.0% |
| global-original | global | original | 0.3198 | -7.6% | 0.1905 | -10.6% |
| global-gap | global | gap | 0.3341 | -3.5% | 0.1998 | -6.3% |
| clustered30-original | 30 | original | 0.3247 | -6.2% | 0.1838 | -13.8% |
| clustered30-gap | 30 | gap | 0.3423 | -1.1% | 0.2036 | -4.5% |

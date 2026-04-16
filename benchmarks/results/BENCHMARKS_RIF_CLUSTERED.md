# Clustered RIF Benchmark

Cue-dependent suppression: per (entry, query_cluster) instead of global.

Burn-in: 5000 steps, Eval: 500 queries
RIF params: alpha=0.3, rate=0.1, decay=0.005 (conservative, same for all)
Date: 2026-04-13

| Config | Clusters | NDCG@10 | Δ | Recall@30 | Δ |
|--------|----------|---------|---|-----------|---|
| baseline | global | 0.2960 | +0.0% | 0.4103 | +0.0% |
| global | global | 0.2993 | +1.1% | 0.4142 | +0.9% |
| 10-clusters | 10 | 0.3113 | +5.2% | 0.4284 | +4.4% |
| 30-clusters | 30 | 0.3132 | +5.8% | 0.4381 | +6.8% |
| 50-clusters | 50 | 0.3067 | +3.6% | 0.4298 | +4.8% |
| 100-clusters | 100 | 0.3092 | +4.5% | 0.4329 | +5.5% |

# Clustered RIF Benchmark

Cue-dependent suppression: per (entry, query_cluster) instead of global.

Burn-in: 5000 steps, Eval: 500 queries
RIF params: alpha=0.3, rate=0.1, decay=0.005 (conservative, same for all)
Date: 2026-04-24

| Config | Clusters | NDCG@10 | Δ | Recall@30 | Δ |
|--------|----------|---------|---|-----------|---|
| baseline | global | 0.3311 | +0.0% | 0.4739 | +0.0% |
| global | global | 0.3314 | +0.1% | 0.4731 | -0.2% |
| 10-clusters | 10 | 0.3444 | +4.0% | 0.4917 | +3.8% |
| 30-clusters | 30 | 0.3422 | +3.4% | 0.4972 | +4.9% |
| 50-clusters | 50 | 0.3385 | +2.3% | 0.4942 | +4.3% |
| 100-clusters | 100 | 0.3411 | +3.0% | 0.4900 | +3.4% |

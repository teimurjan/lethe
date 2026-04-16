# Lifecycle Benchmark

Tests whether the GC tier system improves memory management.

**Method**: Simulate 2000 queries to build entry affinities and tiers.
Then evaluate with tier-weighted scoring: memory-tier entries get 1.15x boost,
low-affinity entries get 0.8x penalty.

Date: 2026-04-12

| Question type | N | Static | GC managed | Δ |
|---------------|---|--------|------------|---|
| knowledge-update | 78 | 0.5231 | 0.5204 | -0.5% |
| multi-session | 133 | 0.1781 | 0.1781 | -0.0% |
| single-session-assistant | 56 | 0.6925 | 0.6928 | +0.0% |
| single-session-preference | 30 | 0.0669 | 0.0669 | +0.0% |
| single-session-user | 70 | 0.3829 | 0.3838 | +0.2% |
| temporal-reasoning | 133 | 0.1870 | 0.1865 | -0.2% |
| **OVERALL** | **500** | **0.3139** | **0.3135** | **-0.1%** |

Tier distribution after 2000 queries: {'naive': 197774, 'gc': 1248, 'memory': 487}

## What this measures

- **knowledge-update**: Does the system promote recent facts over outdated ones?
- **temporal-reasoning**: Does temporal awareness help ordering?
- **multi-session**: Does cross-session memory benefit from tier management?
- **single-session-***: Baseline factual recall (lifecycle less relevant)

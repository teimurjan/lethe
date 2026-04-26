# Extended metrics: baseline vs clustered+gap RIF

Burn-in: 5000 steps, Eval: all queries with qrels
Config: clustered30+gap, alpha=0.3, rate=0.1, decay=0.005
Date: 2026-04-15

Metrics:
- **exact_episode**: top-1 in qrels.
- **ndcg@10**: standard IR ranking quality.
- **sibling_confusion**: top-1 is from an answer-session but not in qrels (same-topic, wrong turn).
- **wrong_family**: top-1 is from a session with no relevant turns (unrelated topic).
- **stale_fact** (knowledge-update only): top-1 is from answer-session but not in qrels — proxy for pulling an older version of the fact.
- **abstain@T**: fraction of queries where top-1 xenc score is below threshold T (would trigger abstention).

## overall

| metric | baseline | RIF | Δ |
|--------|----------|-----|---|
| exact_episode | 0.208 | 0.222 | +0.014 |
| ndcg@10 | 0.293 | 0.316 | +0.023 |
| sibling_confusion | 0.104 | 0.106 | +0.002 |
| wrong_family | 0.688 | 0.672 | -0.016 |
| stale_fact | 0.205 | 0.205 | +0.000 |
| abstain@0.0 | 0.180 | 0.172 | -0.008 |
| abstain@2.0 | 0.324 | 0.310 | -0.014 |
| abstain@4.0 | 0.512 | 0.508 | -0.004 |

## knowledge-update

| metric | baseline | RIF | Δ |
|--------|----------|-----|---|
| exact_episode | 0.410 | 0.423 | +0.013 |
| ndcg@10 | 0.498 | 0.519 | +0.021 |
| sibling_confusion | 0.205 | 0.205 | +0.000 |
| wrong_family | 0.385 | 0.372 | -0.013 |
| stale_fact | 0.205 | 0.205 | +0.000 |
| abstain@0.0 | 0.038 | 0.038 | +0.000 |
| abstain@2.0 | 0.090 | 0.077 | -0.013 |
| abstain@4.0 | 0.218 | 0.205 | -0.013 |

## multi-session

| metric | baseline | RIF | Δ |
|--------|----------|-----|---|
| exact_episode | 0.098 | 0.113 | +0.015 |
| ndcg@10 | 0.151 | 0.176 | +0.025 |
| sibling_confusion | 0.098 | 0.090 | -0.008 |
| wrong_family | 0.805 | 0.797 | -0.008 |
| abstain@0.0 | 0.218 | 0.211 | -0.008 |
| abstain@2.0 | 0.398 | 0.368 | -0.030 |
| abstain@4.0 | 0.571 | 0.579 | +0.008 |

## single-session-assistant

| metric | baseline | RIF | Δ |
|--------|----------|-----|---|
| exact_episode | 0.500 | 0.500 | +0.000 |
| ndcg@10 | 0.671 | 0.680 | +0.009 |
| sibling_confusion | 0.357 | 0.375 | +0.018 |
| wrong_family | 0.143 | 0.125 | -0.018 |
| abstain@0.0 | 0.089 | 0.089 | +0.000 |
| abstain@2.0 | 0.196 | 0.179 | -0.018 |
| abstain@4.0 | 0.625 | 0.625 | +0.000 |

## single-session-preference

| metric | baseline | RIF | Δ |
|--------|----------|-----|---|
| exact_episode | 0.033 | 0.033 | +0.000 |
| ndcg@10 | 0.070 | 0.057 | -0.012 |
| sibling_confusion | 0.000 | 0.000 | +0.000 |
| wrong_family | 0.967 | 0.967 | +0.000 |
| abstain@0.0 | 0.033 | 0.000 | -0.033 |
| abstain@2.0 | 0.200 | 0.200 | +0.000 |
| abstain@4.0 | 0.300 | 0.300 | +0.000 |

## single-session-user

| metric | baseline | RIF | Δ |
|--------|----------|-----|---|
| exact_episode | 0.286 | 0.300 | +0.014 |
| ndcg@10 | 0.362 | 0.405 | +0.043 |
| sibling_confusion | 0.014 | 0.014 | +0.000 |
| wrong_family | 0.700 | 0.686 | -0.014 |
| abstain@0.0 | 0.057 | 0.057 | +0.000 |
| abstain@2.0 | 0.100 | 0.086 | -0.014 |
| abstain@4.0 | 0.243 | 0.229 | -0.014 |

## temporal-reasoning

| metric | baseline | RIF | Δ |
|--------|----------|-----|---|
| exact_episode | 0.075 | 0.098 | +0.023 |
| ndcg@10 | 0.168 | 0.194 | +0.026 |
| sibling_confusion | 0.015 | 0.023 | +0.008 |
| wrong_family | 0.910 | 0.880 | -0.030 |
| abstain@0.0 | 0.361 | 0.346 | -0.015 |
| abstain@2.0 | 0.586 | 0.586 | +0.000 |
| abstain@4.0 | 0.767 | 0.759 | -0.008 |


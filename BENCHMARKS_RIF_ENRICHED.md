# Three-arm benchmark: baseline vs checkpoint 13 vs checkpoint 13 + enrichment

Burn-in: 5000 steps per RIF arm
Coverage: 975 / 199,509 (0.5%) corpus enriched
Date: 2026-04-16

## overall

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.208 | 0.218 | 0.220 | +0.002 |
| ndcg@10 | 0.293 | 0.312 | 0.324 | +0.012 |
| sibling_confusion | 0.104 | 0.106 | 0.106 | +0.000 |
| wrong_family | 0.688 | 0.676 | 0.674 | -0.002 |
| stale_fact | 0.205 | 0.205 | 0.205 | +0.000 |
| abstain@0.0 | 0.180 | 0.166 | 0.168 | +0.002 |
| abstain@2.0 | 0.324 | 0.308 | 0.308 | +0.000 |
| abstain@4.0 | 0.512 | 0.506 | 0.502 | -0.004 |

## covered

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.267 | 0.280 | 0.333 | +0.053 |
| ndcg@10 | 0.350 | 0.390 | 0.473 | +0.083 |
| sibling_confusion | 0.013 | 0.013 | 0.013 | +0.000 |
| wrong_family | 0.720 | 0.707 | 0.653 | -0.053 |
| abstain@0.0 | 0.080 | 0.067 | 0.067 | +0.000 |
| abstain@2.0 | 0.120 | 0.107 | 0.107 | +0.000 |
| abstain@4.0 | 0.280 | 0.253 | 0.227 | -0.027 |

## uncovered

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.198 | 0.207 | 0.200 | -0.007 |
| ndcg@10 | 0.282 | 0.299 | 0.298 | -0.000 |
| sibling_confusion | 0.120 | 0.122 | 0.122 | +0.000 |
| wrong_family | 0.682 | 0.671 | 0.678 | +0.007 |
| stale_fact | 0.205 | 0.205 | 0.205 | +0.000 |
| abstain@0.0 | 0.198 | 0.184 | 0.186 | +0.002 |
| abstain@2.0 | 0.360 | 0.344 | 0.344 | +0.000 |
| abstain@4.0 | 0.553 | 0.551 | 0.551 | +0.000 |

## covered::multi-session

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.000 | 0.000 | 0.167 | +0.167 |
| ndcg@10 | 0.153 | 0.153 | 0.191 | +0.038 |
| sibling_confusion | 0.000 | 0.000 | 0.000 | +0.000 |
| wrong_family | 1.000 | 1.000 | 0.833 | -0.167 |
| abstain@0.0 | 0.333 | 0.167 | 0.167 | +0.000 |
| abstain@2.0 | 0.333 | 0.333 | 0.333 | +0.000 |
| abstain@4.0 | 0.833 | 0.833 | 0.833 | +0.000 |

## covered::single-session-user

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.290 | 0.304 | 0.348 | +0.043 |
| ndcg@10 | 0.367 | 0.411 | 0.497 | +0.086 |
| sibling_confusion | 0.014 | 0.014 | 0.014 | +0.000 |
| wrong_family | 0.696 | 0.681 | 0.638 | -0.043 |
| abstain@0.0 | 0.058 | 0.058 | 0.058 | +0.000 |
| abstain@2.0 | 0.101 | 0.087 | 0.087 | +0.000 |
| abstain@4.0 | 0.232 | 0.203 | 0.174 | -0.029 |

## knowledge-update

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.410 | 0.423 | 0.410 | -0.013 |
| ndcg@10 | 0.498 | 0.515 | 0.513 | -0.002 |
| sibling_confusion | 0.205 | 0.205 | 0.205 | +0.000 |
| wrong_family | 0.385 | 0.372 | 0.385 | +0.013 |
| stale_fact | 0.205 | 0.205 | 0.205 | +0.000 |
| abstain@0.0 | 0.038 | 0.038 | 0.038 | +0.000 |
| abstain@2.0 | 0.090 | 0.077 | 0.077 | +0.000 |
| abstain@4.0 | 0.218 | 0.218 | 0.205 | -0.013 |

## multi-session

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.098 | 0.105 | 0.113 | +0.008 |
| ndcg@10 | 0.151 | 0.172 | 0.178 | +0.005 |
| sibling_confusion | 0.098 | 0.098 | 0.098 | +0.000 |
| wrong_family | 0.805 | 0.797 | 0.789 | -0.008 |
| abstain@0.0 | 0.218 | 0.203 | 0.211 | +0.008 |
| abstain@2.0 | 0.398 | 0.368 | 0.376 | +0.008 |
| abstain@4.0 | 0.571 | 0.571 | 0.579 | +0.008 |

## single-session-assistant

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.500 | 0.500 | 0.500 | +0.000 |
| ndcg@10 | 0.671 | 0.680 | 0.681 | +0.001 |
| sibling_confusion | 0.357 | 0.375 | 0.357 | -0.018 |
| wrong_family | 0.143 | 0.125 | 0.143 | +0.018 |
| abstain@0.0 | 0.089 | 0.089 | 0.089 | +0.000 |
| abstain@2.0 | 0.196 | 0.179 | 0.179 | +0.000 |
| abstain@4.0 | 0.625 | 0.625 | 0.625 | +0.000 |

## single-session-preference

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.033 | 0.033 | 0.033 | +0.000 |
| ndcg@10 | 0.070 | 0.058 | 0.058 | +0.000 |
| sibling_confusion | 0.000 | 0.000 | 0.000 | +0.000 |
| wrong_family | 0.967 | 0.967 | 0.967 | +0.000 |
| abstain@0.0 | 0.033 | 0.000 | 0.000 | +0.000 |
| abstain@2.0 | 0.200 | 0.200 | 0.200 | +0.000 |
| abstain@4.0 | 0.300 | 0.300 | 0.300 | +0.000 |

## single-session-user

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.286 | 0.300 | 0.343 | +0.043 |
| ndcg@10 | 0.362 | 0.405 | 0.490 | +0.085 |
| sibling_confusion | 0.014 | 0.014 | 0.014 | +0.000 |
| wrong_family | 0.700 | 0.686 | 0.643 | -0.043 |
| abstain@0.0 | 0.057 | 0.057 | 0.057 | +0.000 |
| abstain@2.0 | 0.100 | 0.086 | 0.086 | +0.000 |
| abstain@4.0 | 0.243 | 0.214 | 0.186 | -0.029 |

## temporal-reasoning

| metric | baseline | RIF | RIF+enriched | Δ(enr−RIF) |
|--------|----------|-----|--------------|-----------|
| exact_episode | 0.075 | 0.090 | 0.075 | -0.015 |
| ndcg@10 | 0.168 | 0.187 | 0.183 | -0.004 |
| sibling_confusion | 0.015 | 0.015 | 0.023 | +0.008 |
| wrong_family | 0.910 | 0.895 | 0.902 | +0.008 |
| abstain@0.0 | 0.361 | 0.331 | 0.331 | +0.000 |
| abstain@2.0 | 0.586 | 0.579 | 0.571 | -0.008 |
| abstain@4.0 | 0.767 | 0.759 | 0.759 | +0.000 |


# Adaptive Deep-Pass `k_deep` Sweep

Picking `k_deep` by NDCG vs latency. Cross-encoder top-10 stabilizes by merged rank 100, so the old `k_deep=200` was pure latency tax.

Eval: 100-query random sample (seed 0)
Pipeline: BM25 top-N + FAISS top-N → RRF merge → cross-encoder rerank (N = `k_deep`)
Fixed: `k_shallow=30`, `confidence_threshold=4.0`, xenc = `Xenova/ms-marco-MiniLM-L-6-v2` (ONNX, CPU)
Date: 2026-04-23

| Config | NDCG@10 | Δ | Recall@10 | p50 | p95 | p99 | Deep% |
|--------|---------|---|-----------|-----|-----|-----|-------|
| shallow-only (no deep pass) | 0.2866 | -1.56pp | 0.3493 | 1689ms | 2371ms | 3044ms | 0 |
| k_deep=60 | 0.2910 | -1.11pp | 0.3527 | 4159ms | 5936ms | 6707ms | 59 |
| **k_deep=100 (new default)** | **0.3022** | **+0.00pp** | **0.3731** | **5651ms** | **7440ms** | **8262ms** | 59 |
| k_deep=200 (old default) | 0.3022 | baseline | 0.3727 | 9800ms | 12650ms | 13200ms | 59 |

Reproducer: `benchmarks/run_deep_pass.py`.

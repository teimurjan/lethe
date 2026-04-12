# Benchmarks

Dataset: LongMemEval S variant (199509 conversation turns, 500 questions)
Eval: 200-query random sample, seed=0
Date: 2026-04-12

| System | NDCG@10 | Recall@10 | vs baseline | Time |
|--------|---------|-----------|-------------|------|
| Vector only (MiniLM top-10) | 0.1376 | 0.2173 | baseline | 1.3s |
| BM25 only (top-10) | 0.2420 | 0.3264 | +76% | 137.2s |
| Hybrid BM25+vector RRF (memsearch) | 0.2171 | 0.3334 | +58% | 137.9s |
| Vector + cross-encoder rerank | 0.2425 | 0.2892 | +76% | 34.6s |
| gc-memory (BM25 + vector + xenc) | 0.3680 | 0.4694 | +167% | 208.1s |

## How to reproduce

```bash
uv run python experiments/data_prep.py --dataset longmemeval
uv run python benchmarks/run_benchmark.py
```

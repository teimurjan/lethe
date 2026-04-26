# Python ↔ Rust LongMemEval parity

Host: `MacBookPro` · macOS-26.2-arm64-arm-64bit · CPU 10
Date: 2026-04-26T12:12:50
Sample: 200 queries (seed 0)
Tolerance: |ΔNDCG@10| ≤ 0.01, |ΔRecall@10| ≤ 0.01
Verdict: ✅ PASS

## Quality

| Config | Python NDCG@10 | Rust NDCG@10 | Δ NDCG | Python Recall@10 | Rust Recall@10 | Δ Recall | |
|---|---|---|---|---|---|---|---|
| Vector only | 0.1376 | 0.1300 | -0.0076 | 0.2173 | 0.2198 | +0.0025 | ✅ |
| BM25 only | 0.3171 | 0.3113 | -0.0058 | 0.4152 | 0.4152 | +0.0000 | ✅ |
| Hybrid RRF (BM25+vector) | 0.2408 | 0.2416 | +0.0007 | 0.3554 | 0.3579 | +0.0025 | ✅ |
| Vector + cross-encoder rerank | 0.2425 | 0.2365 | -0.0060 | 0.2892 | 0.2892 | +0.0000 | ✅ |
| lethe full (BM25+vector+xenc) | 0.3817 | 0.3797 | -0.0020 | 0.4964 | 0.4962 | -0.0002 | ✅ |

## Wall time

| Config | Python (s) | Rust (s) | Speedup |
|---|---|---|---|
| Vector only | 1.3 | 2.1 | 0.59× |
| BM25 only | 134.9 | 43.7 | 3.09× |
| Hybrid RRF (BM25+vector) | 135.6 | 44.8 | 3.03× |
| Vector + cross-encoder rerank | 209.6 | 214.7 | 0.98× |
| lethe full (BM25+vector+xenc) | 556.9 | 466.0 | 1.19× |

Numerical drift between fastembed (Python) and `ort` (Rust) is bounded by ONNX precision. Anything beyond the tolerance band is a porting bug — investigate BM25 IDF clipping, FAISS top-k tie-breaks, or RRF k=60 first.

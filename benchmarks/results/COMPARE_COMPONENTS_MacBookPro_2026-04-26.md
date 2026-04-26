# Python ↔ Rust component-level numerical diff

Host: `MacBookPro` · macOS-26.2-arm64-arm-64bit · CPU 10
Date: 2026-04-26T14:58:47
BM25 sample: 10 queries vs full corpus
FlatIP sample: 10 queries × top-30
Cross-encoder sample: 50 (query, content) pairs
Verdict: ✅ PASS

## Summary

| Component | Metric | Threshold | Result | |
|---|---|---|---|---|
| BM25 score vector | max \|Δ\| | ≤ 0.0001 | 3.81e-06 | ✅ |
| FlatIP top-K id set | min Jaccard | ≥ 0.9 | 0.935 | ✅ |
| Cross-encoder logit | max \|Δ\| | ≤ 0.001 | 9.54e-07 | ✅ |

BM25 is effectively bit-exact (within `f64` accumulation). FlatIP top-K set agreement is bounded by f32 dot-product tie-breaking: FAISS's SIMD blocked matmul and a naive ndarray dot reduce in different orders, so rank-K boundaries with scores tied at the 8th decimal place can swap. 0.90 is the realistic floor — anything below it is an algorithmic bug, not tie-break noise. Cross-encoder differences come from ONNX runtime precision (fastembed vs `ort`); ~1e-4 is normal, anything beyond `1e-3` is investigation-worthy.

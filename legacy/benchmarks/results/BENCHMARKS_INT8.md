# Bi-Encoder int8 Swap (Negative)

Tested swapping `sentence-transformers/all-MiniLM-L6-v2` (fp32, 90 MB) for `BAAI/bge-small-en-v1.5` (int8, 67 MB via `qdrant/bge-small-en-v1.5-onnx-q`). 5× throughput win on synthetic short sentences inverts to a ~4× regression on real LongMemEval conversation turns because BGE's 512-token context processes ~2× the tokens of MiniLM's 256. NDCG arm skipped — the throughput inversion removed the speed incentive.

Cold-start: median of 3 fresh-subprocess probes per model
Throughput-synth: 500 fixed-length synthetic sentences, warm
Throughput-real: LongMemEval S conversation turns, warm (partial; killed at 2k/10k pool docs)
Date: 2026-04-23

| Config | Cold-start (s, med) | Throughput synth (items/s) | Throughput real (items/s) | NDCG@10 |
|--------|---------------------|-----------------------------|----------------------------|---------|
| all-MiniLM-L6-v2 (fp32, current) | 0.39 | 108 | ~47 | not measured |
| bge-small-en-v1.5 (int8) | 0.44 | 532 (**+4.9×**) | ~11 (**0.23×**) | not measured |

Reproducer: `benchmarks/run_int8.py`.

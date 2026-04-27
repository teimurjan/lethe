# Python ↔ Rust latency

Host: `Teimurs-MacBook-Pro.local` · macOS-26.2-arm64-arm-64bit · CPU 10
Date: 2026-04-26T16:29:20

## Cold start (median over 3 invocations)

| Implementation | Cold start (ms) |
|---|---|
| Python `import lethe.memory_store; import lethe.encoders` | 140 |
| Rust `lethe --version` | 23 |

Rust cold-start speedup: ~6.2×

## Warm retrieve (Python in-process, Rust subprocess-per-query)

| N | Python p50 (ms) | Python p95 (ms) | Rust p50 (ms) | Rust p95 (ms) | p50 speedup |
|---|---|---|---|---|---|
| 500 | 51 | 61 | 243 | 355 | 0.21× |
| 5000 | 50 | 78 | 243 | 250 | 0.21× |
| 20000 | 55 | 95 | 245 | 261 | 0.22× |

Python warm path is in-process (encoders + DB loaded once and reused). Rust warm path is a fresh subprocess per query — that's the realistic command-line shape every Claude Code hook invocation pays. Compare relative trends, not absolute Python numbers; an embedded library user gets even better Python latency.

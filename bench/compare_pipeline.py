"""Python↔Rust comparison harness.

Times cold-start (process boot) and warm retrieval at several corpus
sizes for both implementations on the user's host machine. Output is
appended to ``bench/results/COMPARE_<host>_<date>.md``.

This script does NOT depend on the Rust port being installed in
Python; it shells out to ``lethe-rs`` for the Rust path. The Python
path uses the in-process ``lethe.MemoryStore``.

Usage::

    uv run python bench/compare_pipeline.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "bench" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Corpus sizes to exercise. 200k is the LongMemEval corpus size; we
# sample down so the bench is reproducible without needing the data
# directory present.
SIZES = [500, 5_000, 20_000]
WARMUP_QUERIES = 3
TIMED_QUERIES = 10


def median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000


def time_python_cold() -> float:
    """Wall time of `python -c 'import lethe; lethe.MemoryStore'` in seconds."""
    code = "import lethe.memory_store; import lethe.encoders"
    t0 = time.perf_counter()
    subprocess.check_call(
        [sys.executable, "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=REPO,
    )
    return time.perf_counter() - t0


def time_rust_cold() -> float:
    rust_bin = shutil.which("lethe-rs") or str(
        REPO / "target" / "release" / "lethe-rs"
    )
    t0 = time.perf_counter()
    subprocess.check_call(
        [rust_bin, "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return time.perf_counter() - t0


def seed_python_store(path: Path, n: int) -> None:
    """Build a Python store with N synthetic entries. Done once per size."""
    from lethe.memory_store import MemoryStore
    from lethe.encoders import OnnxBiEncoder, OnnxCrossEncoder

    bi = OnnxBiEncoder("Xenova/all-MiniLM-L6-v2")
    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    store = MemoryStore(path, bi_encoder=bi, cross_encoder=xenc, dim=bi.get_embedding_dimension())
    contents = [
        f"entry {i} alpha session_{i % 17} turn_{i % 31} body text"
        for i in range(n)
    ]
    for c in contents:
        store.add(c)
    store.save()
    store.close()


def time_python_warm(path: Path, n: int) -> tuple[float, float]:
    from lethe.memory_store import MemoryStore
    from lethe.encoders import OnnxBiEncoder, OnnxCrossEncoder

    bi = OnnxBiEncoder("Xenova/all-MiniLM-L6-v2")
    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    store = MemoryStore(path, bi_encoder=bi, cross_encoder=xenc, dim=bi.get_embedding_dimension())
    queries = [f"alpha session_{i % 17}" for i in range(WARMUP_QUERIES + TIMED_QUERIES)]
    for q in queries[:WARMUP_QUERIES]:
        store.retrieve(q, k=10)
    samples: list[float] = []
    for q in queries[WARMUP_QUERIES:]:
        t0 = time.perf_counter()
        store.retrieve(q, k=10)
        samples.append(time.perf_counter() - t0)
    store.save()
    store.close()
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)] if len(samples) >= 2 else samples[-1]
    return p50, p95


def time_rust_warm(path: Path, n: int) -> tuple[float, float]:
    rust_bin = shutil.which("lethe-rs") or str(REPO / "target" / "release" / "lethe-rs")
    queries = [f"alpha session_{i % 17}" for i in range(WARMUP_QUERIES + TIMED_QUERIES)]
    # Each invocation is a fresh process — a fair "Rust binary cold +
    # one query" measurement. The cold-start number above already
    # captures pure boot; this captures boot + retrieval together.
    samples: list[float] = []
    for q in queries[WARMUP_QUERIES:]:
        t0 = time.perf_counter()
        subprocess.check_call(
            [rust_bin, "--root", str(path), "search", q, "--top-k", "10",
             "--json-output"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        samples.append(time.perf_counter() - t0)
    samples.sort()
    p50 = samples[len(samples) // 2]
    p95 = samples[int(len(samples) * 0.95)] if len(samples) >= 2 else samples[-1]
    return p50, p95


def main() -> int:
    if not _python_lethe_available():
        print("error: cannot import lethe.memory_store; run from repo root", file=sys.stderr)
        return 2
    if not (shutil.which("lethe-rs") or (REPO / "target" / "release" / "lethe-rs").exists()):
        print("error: lethe-rs binary not found; build with `cargo build --release -p lethe-cli`", file=sys.stderr)
        return 2

    print("== cold start (process boot) ==", flush=True)
    py_cold = [time_python_cold() for _ in range(3)]
    rs_cold = [time_rust_cold() for _ in range(3)]

    rows: list[dict] = []
    for n in SIZES:
        print(f"\n== N={n} ==", flush=True)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "store"
            print("  seeding Python store…", flush=True)
            seed_python_store(path, n)
            print("  timing Python warm retrieve…", flush=True)
            py_p50, py_p95 = time_python_warm(path, n)
            print("  timing Rust warm retrieve (subprocess)…", flush=True)
            rs_p50, rs_p95 = time_rust_warm(path, n)
            rows.append({
                "n": n,
                "py_p50_ms": py_p50 * 1000,
                "py_p95_ms": py_p95 * 1000,
                "rs_p50_ms": rs_p50 * 1000,
                "rs_p95_ms": rs_p95 * 1000,
            })

    host = platform.node().replace("/", "_") or "unknown"
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = RESULTS_DIR / f"COMPARE_{host}_{today}.md"
    write_report(out_path, py_cold, rs_cold, rows)
    print(f"\nwrote {out_path}")
    return 0


def write_report(path: Path, py_cold: list[float], rs_cold: list[float], rows: list[dict]) -> None:
    py_cold_ms = median_ms(py_cold)
    rs_cold_ms = median_ms(rs_cold)
    lines: list[str] = []
    lines.append("# Python ↔ Rust comparison")
    lines.append("")
    lines.append(f"Host: `{platform.node()}` · {platform.platform()} · CPU {os.cpu_count()}")
    lines.append(f"Date: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Cold start (boot only)")
    lines.append("")
    lines.append(f"| Implementation | median wall (ms) |")
    lines.append(f"|---|---|")
    lines.append(f"| Python `import lethe.memory_store` | {py_cold_ms:.0f} |")
    lines.append(f"| Rust `lethe-rs --version` | {rs_cold_ms:.0f} |")
    if py_cold_ms > 0:
        lines.append(f"")
        lines.append(f"Rust cold start is {py_cold_ms / max(rs_cold_ms, 1.0):.1f}× faster.")
    lines.append("")
    lines.append("## Warm retrieve (boot + query, fresh subprocess for Rust)")
    lines.append("")
    lines.append("| Corpus N | Python p50 (ms) | Python p95 (ms) | Rust p50 (ms) | Rust p95 (ms) | p50 speedup |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        speedup = r["py_p50_ms"] / max(r["rs_p50_ms"], 1.0)
        lines.append(
            f"| {r['n']} | {r['py_p50_ms']:.0f} | {r['py_p95_ms']:.0f} | "
            f"{r['rs_p50_ms']:.0f} | {r['rs_p95_ms']:.0f} | {speedup:.2f}× |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("")
    lines.append(
        "- Python warm path runs in-process; the encoder + DB are loaded once and reused. "
        "Rust warm path is a fresh subprocess per query so it includes binary boot AND query "
        "(an honest \"command-line speed\" comparison). Compare the relative trends rather than "
        "absolute numbers."
    )
    lines.append(
        "- Cross-encoder rerank is fixed cost regardless of language (ONNX Runtime is C++ either "
        "way). Rust's win at small N is dominated by Python import overhead; at large N it shifts "
        "to the BM25 score loop."
    )
    path.write_text("\n".join(lines) + "\n")


def _python_lethe_available() -> bool:
    try:
        __import__("lethe.memory_store")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())

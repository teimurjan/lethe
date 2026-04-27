"""Cold-start + warm-retrieve latency comparison.

Measures the practical command-line latency you actually feel:

  - Cold start: fresh process invocation that just initializes the
    pipeline. Python imports `lethe.memory_store` + `lethe.encoders`;
    Rust runs `lethe --version`.
  - Warm retrieve: subprocess per query, full boot + retrieval. Same
    fixed corpus seeded once via Python, queried by both impls.

Same CLI shape as the other suites. Outputs `bench/results/COMPARE_LATENCY_*.md`.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO, find_rust_bin, host_header, report_path  # noqa: E402

SIZES = [500, 5_000, 20_000]
WARMUP = 3
TIMED = 10


def _python_lethe() -> str:
    return shutil.which("lethe") or "lethe"


def _time(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0


def time_python_cold(n: int = 3) -> list[float]:
    samples: list[float] = []
    for _ in range(n):
        samples.append(
            _time(
                subprocess.check_call,
                [sys.executable, "-c", "import lethe.memory_store; import lethe.encoders"],
            )
        )
    return samples


def time_rust_cold(rs_cli: str, n: int = 3) -> list[float]:
    return [_time(subprocess.check_call, [rs_cli, "--version"]) for _ in range(n)]


def seed_python_store(path: Path, n_entries: int) -> None:
    sys.path.insert(0, str(REPO / "legacy"))
    from lethe.encoders import OnnxBiEncoder, OnnxCrossEncoder  # noqa: PLC0415
    from lethe.memory_store import MemoryStore  # noqa: PLC0415

    bi = OnnxBiEncoder("sentence-transformers/all-MiniLM-L6-v2")
    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    store = MemoryStore(
        path,
        bi_encoder=bi,
        cross_encoder=xenc,
        dim=bi.get_embedding_dimension(),
    )
    for i in range(n_entries):
        store.add(f"entry {i} alpha session_{i % 17} turn_{i % 31} body text")
    store.save()
    store.close()


def warm_queries() -> list[str]:
    return [f"alpha session_{i % 17}" for i in range(WARMUP + TIMED)]


def time_python_warm(path: Path) -> tuple[float, float]:
    sys.path.insert(0, str(REPO / "legacy"))
    from lethe.encoders import OnnxBiEncoder, OnnxCrossEncoder  # noqa: PLC0415
    from lethe.memory_store import MemoryStore  # noqa: PLC0415

    bi = OnnxBiEncoder("sentence-transformers/all-MiniLM-L6-v2")
    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    store = MemoryStore(
        path, bi_encoder=bi, cross_encoder=xenc, dim=bi.get_embedding_dimension()
    )
    qs = warm_queries()
    for q in qs[:WARMUP]:
        store.retrieve(q, k=10)
    samples: list[float] = []
    for q in qs[WARMUP:]:
        t0 = time.perf_counter()
        store.retrieve(q, k=10)
        samples.append(time.perf_counter() - t0)
    store.save()
    store.close()
    samples.sort()
    return samples[len(samples) // 2], samples[int(len(samples) * 0.95)]


def time_rust_warm(path: Path, rs_cli: str) -> tuple[float, float]:
    qs = warm_queries()
    samples: list[float] = []
    for q in qs[WARMUP:]:
        t0 = time.perf_counter()
        subprocess.check_call(
            [
                rs_cli,
                "--root",
                str(path),
                "search",
                q,
                "--top-k",
                "10",
                "--json-output",
            ],
            stdout=subprocess.DEVNULL,
        )
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples[int(len(samples) * 0.95)]


def run_for_impl(impl: str) -> dict:
    """Returns the same JSON shape regardless of which impl ran."""
    rs_cli = shutil.which("lethe") or str(REPO / "target" / "release" / "lethe")
    if impl == "rust" and not Path(rs_cli).exists():
        raise SystemExit(
            "lethe not built; run `cargo build --release -p lethe-cli` first."
        )

    cold_samples = time_rust_cold(rs_cli) if impl == "rust" else time_python_cold()
    rows: list[dict] = []
    for n in SIZES:
        with tempfile.TemporaryDirectory(prefix=f"lat-{n}-") as td:
            path = Path(td) / "store"
            sys.stderr.write(f"[{impl}] seeding N={n}…\n")
            # Python always seeds the store (Rust doesn't expose `add` over CLI yet).
            seed_python_store(path, n)
            sys.stderr.write(f"[{impl}] timing warm retrieve at N={n}…\n")
            if impl == "python":
                p50, p95 = time_python_warm(path)
            else:
                p50, p95 = time_rust_warm(path, rs_cli)
            rows.append({"n": n, "p50_ms": p50 * 1000, "p95_ms": p95 * 1000})
    return {
        "impl": impl,
        "cold_samples_ms": [s * 1000 for s in cold_samples],
        "cold_median_ms": statistics.median(cold_samples) * 1000,
        "warm": rows,
    }


def write_compare_report(py: dict, rs: dict) -> Path:
    out = report_path("latency")
    lines = [
        "# Python ↔ Rust latency",
        "",
        *host_header(),
        "",
        "## Cold start (median over 3 invocations)",
        "",
        "| Implementation | Cold start (ms) |",
        "|---|---|",
        f"| Python `import lethe.memory_store; import lethe.encoders` | {py['cold_median_ms']:.0f} |",
        f"| Rust `lethe --version` | {rs['cold_median_ms']:.0f} |",
        "",
        f"Rust cold-start speedup: ~{py['cold_median_ms'] / max(rs['cold_median_ms'], 1):.1f}×",
        "",
        "## Warm retrieve (Python in-process, Rust subprocess-per-query)",
        "",
        "| N | Python p50 (ms) | Python p95 (ms) | Rust p50 (ms) | Rust p95 (ms) | p50 speedup |",
        "|---|---|---|---|---|---|",
    ]
    for p, r in zip(py["warm"], rs["warm"], strict=True):
        speedup = p["p50_ms"] / max(r["p50_ms"], 1.0)
        lines.append(
            f"| {p['n']} | {p['p50_ms']:.0f} | {p['p95_ms']:.0f} | "
            f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {speedup:.2f}× |"
        )
    lines += [
        "",
        "Python warm path is in-process (encoders + DB loaded once and reused). Rust warm "
        "path is a fresh subprocess per query — that's the realistic command-line shape "
        "every Claude Code hook invocation pays. Compare relative trends, not absolute "
        "Python numbers; an embedded library user gets even better Python latency.",
        "",
    ]
    out.write_text("\n".join(lines))
    return out


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Cold-start + warm-retrieve latency.")
    p.add_argument("--impl", choices=["python", "rust"])
    p.add_argument("--compare", action="store_true")
    args = p.parse_args(argv)
    if args.compare and args.impl:
        sys.stderr.write("--impl and --compare are mutually exclusive\n")
        return 2

    if args.impl:
        print(json.dumps(run_for_impl(args.impl), indent=2))
        return 0
    if not args.compare:
        p.print_help()
        return 2

    # Build release lethe / lethe-bench up front so timing isn't polluted.
    find_rust_bin()  # ensures `lethe-bench` exists; the CLI uses target/release/lethe
    rs_cli = shutil.which("lethe") or str(REPO / "target" / "release" / "lethe")
    if not Path(rs_cli).exists():
        sys.stderr.write("[latency] building release lethe…\n")
        subprocess.check_call(
            ["cargo", "build", "--release", "-p", "lethe-cli"], cwd=REPO
        )

    sys.stderr.write("[latency] running Python…\n")
    py = run_for_impl("python")
    sys.stderr.write("[latency] running Rust…\n")
    rs = run_for_impl("rust")
    out = write_compare_report(py, rs)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

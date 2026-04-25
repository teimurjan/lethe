"""Shared utilities for the parity bench suites under `bench/`.

Every bench script follows the same CLI shape:
  - ``--impl=python``  run one implementation, emit JSON to stdout
  - ``--impl=rust``    same, swapped impl
  - ``--compare``      run both, write a single markdown report under
                       ``bench/results/`` and clean up intermediate
                       JSON via tempfiles

The "1-1" promise: both ``--impl`` paths exercise the same eval logic
on the same inputs and emit JSON of the same shape, so swapping is
trivial and the markdown report can land them in matching columns.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"
DATA = REPO / "data"
LME_RUST = DATA / "lme_rust"
RUST_BIN = REPO / "target" / "release" / "lethe-bench"


def ensure_results_dir() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def find_rust_bin() -> Path:
    """Locate the release `lethe-bench` binary; build it if missing."""
    if RUST_BIN.exists():
        return RUST_BIN
    p = shutil.which("lethe-bench")
    if p:
        return Path(p)
    print("[bench] building release lethe-bench…")
    subprocess.check_call(
        ["cargo", "build", "--release", "-p", "lethe-bench"],
        cwd=REPO,
    )
    if not RUST_BIN.exists():
        raise RuntimeError("lethe-bench build did not produce target/release/lethe-bench")
    return RUST_BIN


def report_path(suite: str) -> Path:
    """Canonical markdown output path: bench/results/COMPARE_<suite>_<host>_<date>.md."""
    host = platform.node().replace("/", "_") or "unknown"
    today = datetime.now().strftime("%Y-%m-%d")
    return ensure_results_dir() / f"COMPARE_{suite.upper()}_{host}_{today}.md"


def host_header() -> list[str]:
    """Markdown lines describing the host the bench ran on."""
    return [
        f"Host: `{platform.node()}` · {platform.platform()} · CPU {os.cpu_count()}",
        f"Date: {datetime.now().isoformat(timespec='seconds')}",
    ]


def load_lme_jsons() -> tuple[dict, dict, dict]:
    """Return (qrels, corpus_content, query_texts) — small enough to hold in memory."""
    qrels = json.loads((DATA / "longmemeval_qrels.json").read_text())
    corpus_content = json.loads((DATA / "longmemeval_corpus.json").read_text())
    query_texts = json.loads((DATA / "longmemeval_queries.json").read_text())
    return qrels, corpus_content, query_texts


def load_lme_npz():
    """Lazily import numpy and load the prepared.npz."""
    import numpy as np  # noqa: PLC0415  - keep numpy out of fast-path imports

    return np.load(str(DATA / "longmemeval_prepared.npz"), allow_pickle=True)


def load_sampled_indices() -> list[int]:
    p = LME_RUST / "sampled_query_indices.txt"
    if not p.exists():
        raise SystemExit(
            f"missing {p}. Run `uv run python bench/prepare.py` first."
        )
    return [int(x) for x in p.read_text().split() if x.strip()]

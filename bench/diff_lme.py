"""Diff `run_lme_python.py` and `lethe-bench` JSON outputs side-by-side.

Writes `bench/results/COMPARE_LONGMEMEVAL_<host>_<date>.md` with a
single table per config showing Python vs. Rust NDCG@10 / Recall@10 +
absolute deltas, then a verdict line.

Pass thresholds (configurable below):
  - Per-config NDCG abs diff ≤ 0.005
  - Per-config Recall abs diff ≤ 0.005

Above that we fail with exit 1 so this can run as a CI gate.
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "bench" / "results"
NDCG_TOLERANCE = 0.005
RECALL_TOLERANCE = 0.005

CONFIGS = [
    ("vector_only", "Vector only"),
    ("bm25_only", "BM25 only"),
    ("hybrid_rrf", "Hybrid RRF (BM25+vector)"),
    ("vector_xenc", "Vector + cross-encoder rerank"),
    ("lethe_full", "lethe full (BM25+vector+xenc)"),
]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: diff_lme.py <python.json> <rust.json>\n")
        return 2
    py = json.loads(Path(argv[0]).read_text())
    rs = json.loads(Path(argv[1]).read_text())
    py_cfg = py["configs"]
    rs_cfg = rs["configs"]

    rows = []
    failed = False
    for key, label in CONFIGS:
        if key not in py_cfg or key not in rs_cfg:
            rows.append({"key": key, "label": label, "status": "missing"})
            failed = True
            continue
        p = py_cfg[key]
        r = rs_cfg[key]
        d_ndcg = r["ndcg"] - p["ndcg"]
        d_recall = r["recall"] - p["recall"]
        ok = abs(d_ndcg) <= NDCG_TOLERANCE and abs(d_recall) <= RECALL_TOLERANCE
        if not ok:
            failed = True
        rows.append({
            "key": key,
            "label": label,
            "py_ndcg": p["ndcg"],
            "py_recall": p["recall"],
            "rs_ndcg": r["ndcg"],
            "rs_recall": r["recall"],
            "d_ndcg": d_ndcg,
            "d_recall": d_recall,
            "py_time": p.get("time_s", 0.0),
            "rs_time": r.get("time_s", 0.0),
            "n_eval": p.get("n_eval", 0),
            "ok": ok,
        })

    print_report(rows)
    write_md(rows)

    if failed:
        print("\nFAIL: parity exceeded thresholds.", file=sys.stderr)
        return 1
    print("\nPASS: every config within tolerance.")
    return 0


def print_report(rows: list[dict]) -> None:
    print(f"\n{'config':32s}  {'py NDCG':>9}  {'rs NDCG':>9}  {'Δ NDCG':>9}  "
          f"{'py Recall':>10}  {'rs Recall':>10}  {'Δ Recall':>10}  {'verdict':>8}")
    for r in rows:
        if r.get("status") == "missing":
            print(f"{r['label']:32s}  -- missing --")
            continue
        verdict = "✅" if r["ok"] else "✗"
        print(
            f"{r['label']:32s}  "
            f"{r['py_ndcg']:>9.4f}  {r['rs_ndcg']:>9.4f}  {r['d_ndcg']:>+9.4f}  "
            f"{r['py_recall']:>10.4f}  {r['rs_recall']:>10.4f}  {r['d_recall']:>+10.4f}  "
            f"{verdict:>8}"
        )


def write_md(rows: list[dict]) -> None:
    host = platform.node().replace("/", "_") or "unknown"
    today = datetime.now().strftime("%Y-%m-%d")
    out = RESULTS / f"COMPARE_LONGMEMEVAL_{host}_{today}.md"
    lines = [
        "# Python ↔ Rust LongMemEval parity",
        "",
        f"Host: `{platform.node()}` · {platform.platform()}",
        f"Date: {datetime.now().isoformat(timespec='seconds')}",
        f"Tolerance: |ΔNDCG| ≤ {NDCG_TOLERANCE}, |ΔRecall| ≤ {RECALL_TOLERANCE}",
        "",
        "## Headline",
        "",
        "| Config | Python NDCG@10 | Rust NDCG@10 | Δ NDCG | Python Recall@10 | Rust Recall@10 | Δ Recall | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("status") == "missing":
            lines.append(f"| {r['label']} | — | — | — | — | — | — | ⚠ missing |")
            continue
        v = "✅" if r["ok"] else "✗"
        lines.append(
            f"| {r['label']} | {r['py_ndcg']:.4f} | {r['rs_ndcg']:.4f} | "
            f"{r['d_ndcg']:+.4f} | {r['py_recall']:.4f} | {r['rs_recall']:.4f} | "
            f"{r['d_recall']:+.4f} | {v} |"
        )
    lines.append("")
    lines.append("## Wall time")
    lines.append("")
    lines.append("| Config | Python (s) | Rust (s) |")
    lines.append("|---|---|---|")
    for r in rows:
        if "py_time" in r:
            lines.append(f"| {r['label']} | {r['py_time']:.1f} | {r['rs_time']:.1f} |")
    lines.append("")
    lines.append(
        "Numerical drift between the two ONNX runtimes (fastembed vs `ort`) is "
        f"expected within ±{NDCG_TOLERANCE:.3f} NDCG. Anything beyond that "
        "indicates a porting bug — investigate BM25 IDF clipping, FAISS top-k "
        "tie-breaks, or RRF k=60 first."
    )
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

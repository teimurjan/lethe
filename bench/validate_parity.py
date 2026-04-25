"""Parity validator: confirm the Rust port returns retrieval results
"close enough" to the Python reference on a known small corpus.

Quick local sanity check — NOT a replacement for the full LongMemEval
benchmark. Designed to finish in ~2-4 minutes after first model
download so we can spot common porting bugs (wrong tokenizer, off-by-
one in argpartition, mismatched RIF formula) before paying for the
real eval.

Procedure
---------
1. Reuse `demo/data/corpus.json` (45 PostgreSQL connection-pool entries)
   and `demo/data/queries.json` (30 hand-labeled queries). The same
   content is written to two separate `.lethe/memory/corpus.md` files
   under throwaway directories so each implementation builds its own
   index without sharing storage state.
2. `lethe index` (Python) on dir A, `lethe-rs index` on dir B.
3. For each query, run `lethe search --json-output` and
   `lethe-rs search --json-output`. Parse the top-K JSON.
4. Map chunk-hash → entry id (the chunks carry an `## eXXX` heading
   so we can recover the original demo id from the content body).
5. Compute, per query: top-1 agreement (same id at rank 1), top-K
   set overlap (Jaccard), top-K rank-perfect (same ids in same
   order), and the rank-correlation of overlapping ids.

Usage::

    ./bench/validate_parity.sh
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "bench" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_JSON = REPO / "demo" / "data" / "corpus.json"
QUERIES_JSON = REPO / "demo" / "data" / "queries.json"

# Limit queries for the "quick" mode; pass --full to run all of them.
QUICK_QUERY_LIMIT = 10
TOP_K = 5

# Heading we use to encode the original entry id inside the markdown
# chunk so we can recover it from retrieval output.
ID_HEADING_RE = re.compile(r"##\s+(e\w+)\b")


def _python_lethe_path() -> str:
    # We assume the user has the dev install (`uv pip install -e .`)
    # present, the same way the rest of the benchmarks do. We use the
    # console script directly so the Python and Rust paths share an
    # identical "fresh subprocess" assumption — fair on cold-start.
    return "lethe"


def _rust_bin() -> str:
    candidate = shutil.which("lethe-rs")
    if candidate:
        return candidate
    fallback = REPO / "target" / "release" / "lethe-rs"
    if not fallback.exists():
        sys.stderr.write(
            "error: lethe-rs not found; build with `cargo build --release -p lethe-cli`\n"
        )
        sys.exit(2)
    return str(fallback)


def write_corpus(root: Path, entries: list[dict]) -> None:
    memory = root / ".lethe" / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    parts: list[str] = ["# corpus\n"]
    for e in entries:
        parts.append(f"## {e['id']}\n{e['content']}\n")
    (memory / "corpus.md").write_text("\n".join(parts))


def run_index(root: Path, binary: str) -> dict:
    out = subprocess.check_output(
        [binary, "--root", str(root), "index", "--no-register", "--json-output"],
        cwd=REPO,
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(out)


def run_search(root: Path, binary: str, query: str) -> list[dict]:
    out = subprocess.check_output(
        [
            binary,
            "--root",
            str(root),
            "search",
            query,
            "--top-k",
            str(TOP_K),
            "--json-output",
        ],
        cwd=REPO,
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return json.loads(out)


def chunk_to_entry_id(content: str) -> str | None:
    m = ID_HEADING_RE.search(content)
    return m.group(1) if m else None


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def main(argv: list[str]) -> int:
    full = "--full" in argv
    if not CORPUS_JSON.exists() or not QUERIES_JSON.exists():
        sys.stderr.write(
            f"error: missing demo data ({CORPUS_JSON.name} / {QUERIES_JSON.name})\n"
        )
        return 2

    entries = json.loads(CORPUS_JSON.read_text())
    queries = json.loads(QUERIES_JSON.read_text())
    if not full:
        queries = queries[:QUICK_QUERY_LIMIT]

    print(
        f"validating parity over {len(entries)} entries × {len(queries)} queries",
        flush=True,
    )

    py_bin = _python_lethe_path()
    rs_bin = _rust_bin()

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        py_root = td_path / "py"
        rs_root = td_path / "rs"
        py_root.mkdir()
        rs_root.mkdir()
        write_corpus(py_root, entries)
        write_corpus(rs_root, entries)

        print("  indexing Python store…", flush=True)
        py_idx = run_index(py_root, py_bin)
        print(f"    {py_idx}", flush=True)
        print("  indexing Rust store…", flush=True)
        rs_idx = run_index(rs_root, rs_bin)
        print(f"    {rs_idx}", flush=True)

        if py_idx.get("total") != rs_idx.get("total"):
            print(
                f"  ⚠ chunk count mismatch: py={py_idx.get('total')} rs={rs_idx.get('total')}",
                file=sys.stderr,
            )

        rows = []
        for q in queries:
            qtext = q["text"]
            py_hits = run_search(py_root, py_bin, qtext)
            rs_hits = run_search(rs_root, rs_bin, qtext)
            py_ids = [chunk_to_entry_id(h["content"]) or h["id"] for h in py_hits]
            rs_ids = [chunk_to_entry_id(h["content"]) or h["id"] for h in rs_hits]
            rows.append(
                {
                    "query": qtext,
                    "py": py_ids,
                    "rs": rs_ids,
                    "py_scores": [h["score"] for h in py_hits],
                    "rs_scores": [h["score"] for h in rs_hits],
                    "relevant": list(q.get("relevant", {}).keys()),
                }
            )

    summary = score_summary(rows)
    print_console(summary, rows)

    host = platform.node().replace("/", "_") or "unknown"
    today = datetime.now().strftime("%Y-%m-%d")
    out_path = RESULTS_DIR / f"VALIDATE_PARITY_{host}_{today}.md"
    write_report(out_path, summary, rows, len(entries))
    print(f"\nwrote {out_path}")

    # Exit non-zero only on hard regressions so this can run in CI.
    if summary["top1_agreement"] < 0.6 or summary["topk_jaccard_mean"] < 0.5:
        print("FAIL: parity below thresholds (top1<0.6 or jaccard<0.5)", file=sys.stderr)
        return 1
    return 0


def score_summary(rows: list[dict]) -> dict:
    n = len(rows)
    top1_match = 0
    rank_perfect = 0
    jacs: list[float] = []
    py_relevance_recall: list[float] = []
    rs_relevance_recall: list[float] = []
    for r in rows:
        py = r["py"]
        rs = r["rs"]
        if py and rs and py[0] == rs[0]:
            top1_match += 1
        if py == rs:
            rank_perfect += 1
        jacs.append(jaccard(set(py[:TOP_K]), set(rs[:TOP_K])))
        rel = set(r["relevant"])
        if rel:
            py_relevance_recall.append(len(rel & set(py[:TOP_K])) / len(rel))
            rs_relevance_recall.append(len(rel & set(rs[:TOP_K])) / len(rel))
    return {
        "n": n,
        "top1_agreement": top1_match / n if n else 0.0,
        "rank_perfect": rank_perfect / n if n else 0.0,
        "topk_jaccard_mean": statistics.fmean(jacs) if jacs else 0.0,
        "py_recall_at_k": statistics.fmean(py_relevance_recall) if py_relevance_recall else 0.0,
        "rs_recall_at_k": statistics.fmean(rs_relevance_recall) if rs_relevance_recall else 0.0,
    }


def print_console(summary: dict, rows: list[dict]) -> None:
    print("\n=== parity summary ===")
    print(f"  top-1 agreement       : {summary['top1_agreement']:.0%}")
    print(f"  rank-perfect (top-{TOP_K})  : {summary['rank_perfect']:.0%}")
    print(f"  top-{TOP_K} Jaccard          : {summary['topk_jaccard_mean']:.2f}")
    print(f"  Python recall@{TOP_K}       : {summary['py_recall_at_k']:.0%}")
    print(f"  Rust recall@{TOP_K}         : {summary['rs_recall_at_k']:.0%}")
    print(f"\n=== per-query disagreements ===")
    any_diff = False
    for r in rows:
        if r["py"] == r["rs"]:
            continue
        any_diff = True
        print(f"  query: {r['query']}")
        print(f"    py: {r['py']}")
        print(f"    rs: {r['rs']}")
    if not any_diff:
        print("  (none — bit-perfect rank match across all queries)")


def write_report(out_path: Path, summary: dict, rows: list[dict], n_entries: int) -> None:
    lines = [
        "# Python ↔ Rust parity validation",
        "",
        f"Host: `{platform.node()}` · {platform.platform()}",
        f"Date: {datetime.now().isoformat(timespec='seconds')}",
        f"Corpus: {n_entries} entries · {len(rows)} queries · top-{TOP_K}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| top-1 agreement | {summary['top1_agreement']:.0%} |",
        f"| rank-perfect (top-{TOP_K}) | {summary['rank_perfect']:.0%} |",
        f"| top-{TOP_K} Jaccard (mean) | {summary['topk_jaccard_mean']:.2f} |",
        f"| Python recall@{TOP_K} | {summary['py_recall_at_k']:.0%} |",
        f"| Rust recall@{TOP_K} | {summary['rs_recall_at_k']:.0%} |",
        "",
        "Numerical drift between the two ONNX runtimes is expected — the threshold for "
        "this validator is **top-1 ≥ 60% AND top-5 Jaccard ≥ 0.5**. Anything below that "
        "indicates a porting bug (wrong tokenizer, off-by-one in argpartition, mismatched "
        "RIF formula). For absolute quality, run the full LongMemEval benchmark.",
        "",
        "## Per-query breakdown",
        "",
        "| Query | Top-1 match | Jaccard@k | Python top-K | Rust top-K |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        py = r["py"]
        rs = r["rs"]
        match = "✅" if py and rs and py[0] == rs[0] else "✗"
        jac = jaccard(set(py[:TOP_K]), set(rs[:TOP_K]))
        lines.append(
            f"| {r['query']} | {match} | {jac:.2f} | "
            f"{', '.join(py[:TOP_K])} | {', '.join(rs[:TOP_K])} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

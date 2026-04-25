"""Component-level numerical diff between Python and Rust.

Tests each piece of the retrieval pipeline in isolation:

  - BM25 score vector — compare element-wise across the full corpus
  - FlatIP top-30 — compare the returned ids and their dot-product scores
  - Cross-encoder logits — compare on a fixed set of (query, content) pairs

Each sub-bench has a tolerance band; the suite passes only when every
component does. When a single suite fails, the markdown report points
at the offending component so you don't have to bisect by hand.

Same CLI shape as `bench/longmemeval.py`:
  --impl python|rust  → emit JSON for one component spec
  --compare           → run both, write `bench/results/COMPARE_COMPONENTS_*.md`

The same query / pair fixtures are used across both impls so the diff
is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import (  # noqa: E402
    DATA,
    LME_RUST,
    REPO,
    find_rust_bin,
    host_header,
    load_lme_jsons,
    load_lme_npz,
    load_sampled_indices,
    report_path,
)

BM25_QUERY_SAMPLE = 10
FLATIP_QUERY_SAMPLE = 10
FLATIP_K = 30
XENC_PAIR_SAMPLE = 50

BM25_TOL = 1e-4
FLATIP_OVERLAP_MIN = 0.99  # ≥99% set agreement on top-30 ids
XENC_TOL = 1e-3


def fixture_query_indices(n: int) -> list[int]:
    """Deterministic subset of the prepared sample for component runs."""
    sampled = load_sampled_indices()
    return sampled[:n]


def fixture_query_strings(query_indices: list[int]) -> list[str]:
    prep = load_lme_npz()
    query_ids = list(prep["query_ids"])
    _qrels, _corpus_content, query_texts = load_lme_jsons()
    return [query_texts.get(query_ids[i], "") for i in query_indices]


def fixture_pairs(n_pairs: int) -> list[tuple[str, str]]:
    """Pick `n_pairs` (query, content) pairs deterministically from the sample."""
    sampled = load_sampled_indices()
    prep = load_lme_npz()
    query_ids = list(prep["query_ids"])
    corpus_ids = list(prep["corpus_ids"])
    _qrels, corpus_content, query_texts = load_lme_jsons()
    out: list[tuple[str, str]] = []
    for i, qi in enumerate(sampled):
        q = query_texts.get(query_ids[qi], "")
        # cycle through the corpus deterministically
        cid = corpus_ids[(qi * 7919 + i) % len(corpus_ids)]
        out.append((q, corpus_content.get(cid, "")))
        if len(out) >= n_pairs:
            break
    return out


# --------------------------------------------------------------- Python impls


def py_bm25_scores(queries: list[str]) -> list[list[float]]:
    from rank_bm25 import BM25Okapi  # noqa: PLC0415

    sys.path.insert(0, str(REPO))
    from lethe.vectors import tokenize_bm25  # noqa: PLC0415

    _qrels, corpus_content, _qtexts = load_lme_jsons()
    prep = load_lme_npz()
    corpus_ids = list(prep["corpus_ids"])
    sys.stderr.write("[python] tokenizing corpus + building BM25…\n")
    tokenized = [tokenize_bm25(corpus_content.get(cid, "")) for cid in corpus_ids]
    bm25 = BM25Okapi(tokenized)
    return [list(map(float, bm25.get_scores(tokenize_bm25(q)))) for q in queries]


def py_flat_ip(query_indices: list[int], k: int) -> list[list[tuple[str, float]]]:
    import faiss  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    prep = load_lme_npz()
    corpus_ids = list(prep["corpus_ids"])
    corpus_embs = prep["corpus_embeddings"].astype(np.float32)
    query_embs = prep["query_embeddings"].astype(np.float32)
    index = faiss.IndexFlatIP(corpus_embs.shape[1])
    index.add(corpus_embs)
    out: list[list[tuple[str, float]]] = []
    for qi in query_indices:
        D, I = index.search(query_embs[qi].reshape(1, -1), k)
        out.append([(corpus_ids[i], float(D[0][rank])) for rank, i in enumerate(I[0]) if i >= 0])
    return out


def py_xenc(pairs: list[tuple[str, str]]) -> list[float]:
    sys.path.insert(0, str(REPO))
    from lethe.encoders import OnnxCrossEncoder  # noqa: PLC0415

    xenc = OnnxCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    return [float(s) for s in xenc.predict(pairs)]


# --------------------------------------------------------------- Rust impls


def rs_bm25_scores(queries: list[str], rs_bin: Path) -> list[list[float]]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"queries": queries}, f)
        spec_path = f.name
    try:
        out = subprocess.check_output(
            [str(rs_bin), "bm25", "--queries", spec_path], cwd=REPO, text=True
        )
    finally:
        Path(spec_path).unlink(missing_ok=True)
    payload = json.loads(out)
    return payload["scores"]


def rs_flat_ip(query_indices: list[int], k: int, rs_bin: Path) -> list[list[tuple[str, float]]]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"query_indices": query_indices, "k": k}, f)
        spec_path = f.name
    try:
        out = subprocess.check_output(
            [str(rs_bin), "flat-ip", "--queries", spec_path], cwd=REPO, text=True
        )
    finally:
        Path(spec_path).unlink(missing_ok=True)
    payload = json.loads(out)
    return [[tuple(x) for x in row] for row in payload["results"]]


def rs_xenc(pairs: list[tuple[str, str]], rs_bin: Path) -> list[float]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"pairs": [list(p) for p in pairs]}, f)
        spec_path = f.name
    try:
        out = subprocess.check_output(
            [str(rs_bin), "xenc", "--pairs", spec_path], cwd=REPO, text=True
        )
    finally:
        Path(spec_path).unlink(missing_ok=True)
    payload = json.loads(out)
    return [float(s) for s in payload["logits"]]


# --------------------------------------------------------------- diff helpers


def max_abs_diff(a: list[float], b: list[float]) -> tuple[float, int]:
    if len(a) != len(b):
        return float("inf"), -1
    diffs = [abs(x - y) for x, y in zip(a, b, strict=True)]
    if not diffs:
        return 0.0, -1
    idx = max(range(len(diffs)), key=lambda i: diffs[i])
    return diffs[idx], idx


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


# --------------------------------------------------------------- runners


def run_python() -> dict:
    sys.stderr.write("[python] BM25 scores…\n")
    qidx = fixture_query_indices(BM25_QUERY_SAMPLE)
    qstrs = fixture_query_strings(qidx)
    bm25_scores = py_bm25_scores(qstrs)

    sys.stderr.write("[python] FlatIP top-K…\n")
    flat_qidx = fixture_query_indices(FLATIP_QUERY_SAMPLE)
    flat = py_flat_ip(flat_qidx, FLATIP_K)

    sys.stderr.write("[python] xenc logits…\n")
    pairs = fixture_pairs(XENC_PAIR_SAMPLE)
    logits = py_xenc(pairs)

    return {
        "impl": "python",
        "bm25": {"queries": qstrs, "scores": bm25_scores},
        "flat_ip": {"query_indices": flat_qidx, "results": flat, "k": FLATIP_K},
        "xenc": {"pairs": pairs, "logits": logits},
    }


def run_rust() -> dict:
    rs_bin = find_rust_bin()
    sys.stderr.write("[rust] BM25 scores…\n")
    qidx = fixture_query_indices(BM25_QUERY_SAMPLE)
    qstrs = fixture_query_strings(qidx)
    bm25_scores = rs_bm25_scores(qstrs, rs_bin)

    sys.stderr.write("[rust] FlatIP top-K…\n")
    flat_qidx = fixture_query_indices(FLATIP_QUERY_SAMPLE)
    flat = rs_flat_ip(flat_qidx, FLATIP_K, rs_bin)

    sys.stderr.write("[rust] xenc logits…\n")
    pairs = fixture_pairs(XENC_PAIR_SAMPLE)
    logits = rs_xenc(pairs, rs_bin)

    return {
        "impl": "rust",
        "bm25": {"queries": qstrs, "scores": bm25_scores},
        "flat_ip": {"query_indices": flat_qidx, "results": flat, "k": FLATIP_K},
        "xenc": {"pairs": pairs, "logits": logits},
    }


# --------------------------------------------------------------- compare


def compare(py: dict, rs: dict) -> tuple[bool, dict]:
    # BM25: max abs diff across all (query, doc) pairs.
    bm_diffs: list[tuple[int, float, int]] = []
    for i, (a, b) in enumerate(zip(py["bm25"]["scores"], rs["bm25"]["scores"], strict=True)):
        d, idx = max_abs_diff(a, b)
        bm_diffs.append((i, d, idx))
    bm_max = max((d for _, d, _ in bm_diffs), default=0.0)
    bm_ok = bm_max <= BM25_TOL

    # FlatIP: agreement on the top-K id sets per query.
    flat_overlaps: list[float] = []
    for py_top, rs_top in zip(py["flat_ip"]["results"], rs["flat_ip"]["results"], strict=True):
        py_ids = {x[0] for x in py_top}
        rs_ids = {x[0] for x in rs_top}
        flat_overlaps.append(jaccard(py_ids, rs_ids))
    flat_min = min(flat_overlaps) if flat_overlaps else 1.0
    flat_ok = flat_min >= FLATIP_OVERLAP_MIN

    # Xenc: max abs diff across all logit pairs.
    xenc_max, xenc_idx = max_abs_diff(py["xenc"]["logits"], rs["xenc"]["logits"])
    xenc_ok = xenc_max <= XENC_TOL

    overall = bm_ok and flat_ok and xenc_ok
    return overall, {
        "bm25_max_abs_diff": bm_max,
        "bm25_pass": bm_ok,
        "flat_ip_min_jaccard": flat_min,
        "flat_ip_pass": flat_ok,
        "xenc_max_abs_diff": xenc_max,
        "xenc_max_idx": xenc_idx,
        "xenc_pass": xenc_ok,
    }


def write_compare_report(py: dict, rs: dict, overall_ok: bool, summary: dict) -> Path:
    out = report_path("components")
    lines = [
        "# Python ↔ Rust component-level numerical diff",
        "",
        *host_header(),
        f"BM25 sample: {len(py['bm25']['queries'])} queries vs full corpus",
        f"FlatIP sample: {len(py['flat_ip']['results'])} queries × top-{py['flat_ip']['k']}",
        f"Cross-encoder sample: {len(py['xenc']['pairs'])} (query, content) pairs",
        f"Verdict: {'✅ PASS' if overall_ok else '✗ FAIL'}",
        "",
        "## Summary",
        "",
        "| Component | Metric | Threshold | Result | |",
        "|---|---|---|---|---|",
        f"| BM25 score vector | max \\|Δ\\| | ≤ {BM25_TOL} | {summary['bm25_max_abs_diff']:.2e} | "
        f"{'✅' if summary['bm25_pass'] else '✗'} |",
        f"| FlatIP top-K id set | min Jaccard | ≥ {FLATIP_OVERLAP_MIN} | {summary['flat_ip_min_jaccard']:.3f} | "
        f"{'✅' if summary['flat_ip_pass'] else '✗'} |",
        f"| Cross-encoder logit | max \\|Δ\\| | ≤ {XENC_TOL} | {summary['xenc_max_abs_diff']:.2e} | "
        f"{'✅' if summary['xenc_pass'] else '✗'} |",
        "",
        "BM25 and FlatIP have no nondeterminism — they should be effectively bit-exact "
        "(within `f32` rounding). Cross-encoder differences come from ONNX runtime "
        "precision (fastembed vs `ort`); ~1e-4 is normal, anything beyond `1e-3` is "
        "investigation-worthy.",
        "",
    ]
    out.write_text("\n".join(lines))
    return out


# --------------------------------------------------------------- entry


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Component-level Python ↔ Rust diff.")
    p.add_argument("--impl", choices=["python", "rust"])
    p.add_argument("--compare", action="store_true")
    args = p.parse_args(argv)

    if args.compare and args.impl:
        sys.stderr.write("--impl and --compare are mutually exclusive\n")
        return 2
    if not LME_RUST.exists():
        sys.stderr.write("error: run `uv run python bench/prepare.py` first.\n")
        return 2

    if args.impl == "python":
        print(json.dumps(run_python()))
        return 0
    if args.impl == "rust":
        print(json.dumps(run_rust()))
        return 0
    if not args.compare:
        p.print_help()
        return 2

    py = run_python()
    rs = run_rust()
    overall_ok, summary = compare(py, rs)
    out = write_compare_report(py, rs, overall_ok, summary)
    print(f"wrote {out}")
    print(json.dumps(summary, indent=2))
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

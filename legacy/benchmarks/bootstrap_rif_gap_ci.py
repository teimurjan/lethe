"""Bootstrap CIs and paired permutation tests for the rank-gap RIF benchmark.

Reads `benchmarks/results/rif_gap_per_query.json` (produced by
`run_rif_gap.py` after the persistence patch), and for each (config, baseline)
pair reports:

  - Mean NDCG and 95% bootstrap CI via the percentile method
  - Paired delta NDCG with 95% bootstrap CI
  - Paired permutation test p-value on the delta (two-sided)

Same for Recall@30.

The baseline is assumed to be the first config in the JSON. If the benchmark
script changes config ordering, pass `--baseline <name>` explicitly.

Usage:
    uv run python benchmarks/bootstrap_rif_gap_ci.py
    uv run python benchmarks/bootstrap_rif_gap_ci.py --n-boot 20000 --n-perm 20000
    uv run python benchmarks/bootstrap_rif_gap_ci.py --baseline baseline --metric ndcg

Outputs a markdown table to stdout and to
`benchmarks/results/rif_gap_ci.md`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

IN_PATH = Path("benchmarks/results/rif_gap_per_query.json")
OUT_PATH = Path("benchmarks/results/rif_gap_ci.md")


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for the 1-alpha bootstrap percentile CI."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(values.mean()), lo, hi


def bootstrap_paired_delta_ci(
    treatment: np.ndarray,
    baseline: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for mean(treatment - baseline) using paired resampling."""
    diffs = treatment - baseline
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(diffs.mean()), lo, hi


def paired_permutation_pvalue(
    treatment: np.ndarray,
    baseline: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    """Two-sided paired permutation test on the mean difference."""
    diffs = treatment - baseline
    observed = float(np.abs(diffs.mean()))
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diffs)))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    # +1 in numerator and denominator: standard small-sample correction.
    return float((np.sum(perm_means >= observed) + 1) / (n_perm + 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=IN_PATH)
    ap.add_argument("--output", type=Path, default=OUT_PATH)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--n-perm", type=int, default=10_000)
    ap.add_argument("--baseline", type=str, default=None,
                    help="Name of the baseline config (default: first in file)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Per-query results not found at {args.input}. "
            f"Run `uv run python benchmarks/run_rif_gap.py` first."
        )

    payload = json.loads(args.input.read_text())
    configs = payload["configs"]
    if not configs:
        raise SystemExit("No configs in input file.")

    baseline_name = args.baseline or configs[0]["name"]
    baseline_cfg = next((c for c in configs if c["name"] == baseline_name), None)
    if baseline_cfg is None:
        names = ", ".join(c["name"] for c in configs)
        raise SystemExit(f"Baseline '{baseline_name}' not found. Available: {names}")

    rng = np.random.default_rng(args.seed)
    b_ndcg = np.asarray(baseline_cfg["ndcg"], dtype=np.float64)
    b_recall = np.asarray(baseline_cfg["recall"], dtype=np.float64)
    n_queries = len(b_ndcg)

    print(f"Baseline: {baseline_name}  |  n_queries = {n_queries}")
    print(f"Bootstrap resamples: {args.n_boot}  |  permutations: {args.n_perm}")
    print()

    rows = []
    for cfg in configs:
        t_ndcg = np.asarray(cfg["ndcg"], dtype=np.float64)
        t_recall = np.asarray(cfg["recall"], dtype=np.float64)

        mean_n, n_lo, n_hi = bootstrap_mean_ci(t_ndcg, args.n_boot, rng)
        mean_r, r_lo, r_hi = bootstrap_mean_ci(t_recall, args.n_boot, rng)

        if cfg["name"] == baseline_name:
            row = {
                "name": cfg["name"],
                "n_ndcg": f"{mean_n:.4f} [{n_lo:.4f}, {n_hi:.4f}]",
                "n_delta": "—",
                "n_p": "—",
                "n_recall": f"{mean_r:.4f} [{r_lo:.4f}, {r_hi:.4f}]",
                "r_delta": "—",
                "r_p": "—",
            }
        else:
            d_n, d_n_lo, d_n_hi = bootstrap_paired_delta_ci(
                t_ndcg, b_ndcg, args.n_boot, rng,
            )
            d_r, d_r_lo, d_r_hi = bootstrap_paired_delta_ci(
                t_recall, b_recall, args.n_boot, rng,
            )
            p_n = paired_permutation_pvalue(t_ndcg, b_ndcg, args.n_perm, rng)
            p_r = paired_permutation_pvalue(t_recall, b_recall, args.n_perm, rng)
            row = {
                "name": cfg["name"],
                "n_ndcg": f"{mean_n:.4f} [{n_lo:.4f}, {n_hi:.4f}]",
                "n_delta": f"{d_n:+.4f} [{d_n_lo:+.4f}, {d_n_hi:+.4f}]",
                "n_p": f"{p_n:.4f}",
                "n_recall": f"{mean_r:.4f} [{r_lo:.4f}, {r_hi:.4f}]",
                "r_delta": f"{d_r:+.4f} [{d_r_lo:+.4f}, {d_r_hi:+.4f}]",
                "r_p": f"{p_r:.4f}",
            }
        rows.append(row)

    header = (
        "| Config | NDCG@10 [95% CI] | Δ NDCG [95% CI] | p (perm) | "
        "Recall@30 [95% CI] | Δ Recall [95% CI] | p (perm) |"
    )
    sep = "|" + "|".join(["---"] * 7) + "|"
    lines = [
        f"# Rank-gap RIF: Bootstrap CIs and Permutation Tests",
        "",
        f"Baseline: `{baseline_name}` | n_queries = {n_queries} | "
        f"n_boot = {args.n_boot} | n_perm = {args.n_perm} | seed = {args.seed}",
        "",
        header,
        sep,
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['n_ndcg']} | {r['n_delta']} | {r['n_p']} | "
            f"{r['n_recall']} | {r['r_delta']} | {r['r_p']} |"
        )
    out = "\n".join(lines) + "\n"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out)

    print(out)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

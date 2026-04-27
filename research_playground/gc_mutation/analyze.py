"""Load experiment results and produce plots + summary table."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


def load_latest_results() -> dict:
    """Load the most recent run_*.json file."""
    pattern = str(RESULTS_DIR / "run_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result files found matching {pattern}")
    latest = files[-1]
    print(f"Loading {latest}")
    with open(latest) as f:
        return json.load(f)


def extract_metric(
    metrics_by_step: list[dict],
    metric_name: str,
) -> tuple[list[int], list[float]]:
    """Extract (steps, values) for a given metric."""
    steps = [m["step"] for m in metrics_by_step]
    values = [m[metric_name] for m in metrics_by_step]
    return steps, values


def plot_ndcg(results: dict) -> None:
    """Plot 1: NDCG@10 over time, three lines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for arm_name, style in [("static", "-"), ("random", "--"), ("gc", "-.")]:
        arm = results["arms"][arm_name]
        steps, values = extract_metric(arm["metrics_by_step"], "ndcg_at_10")
        ax.plot(steps, values, style, label=arm_name, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("NDCG@10")
    ax.set_title("NDCG@10 Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "ndcg_at_10.png"), dpi=150)
    plt.close(fig)


def plot_diversity(results: dict) -> None:
    """Plot 2: Diversity over time, three lines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for arm_name, style in [("static", "-"), ("random", "--"), ("gc", "-.")]:
        arm = results["arms"][arm_name]
        steps, values = extract_metric(arm["metrics_by_step"], "diversity")
        ax.plot(steps, values, style, label=arm_name, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Pairwise Cosine Distance")
    ax.set_title("Diversity Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "diversity.png"), dpi=150)
    plt.close(fig)


def plot_anchor_drift(results: dict) -> None:
    """Plot 3: Anchor drift over time (GC arm only)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    arm = results["arms"]["gc"]
    steps, values = extract_metric(arm["metrics_by_step"], "anchor_drift")
    ax.plot(steps, values, "-", color="tab:red", linewidth=1.5)
    ax.axhline(y=0.25, color="black", linestyle=":", alpha=0.5, label="Circuit breaker (0.25)")

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Anchor Drift (1 - cos)")
    ax.set_title("Anchor Drift Over Time (GC arm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "anchor_drift.png"), dpi=150)
    plt.close(fig)


def plot_tier_distribution(results: dict) -> None:
    """Plot 4: Tier distribution over time (GC arm, stacked area chart)."""
    arm = results["arms"]["gc"]
    metrics = arm["metrics_by_step"]
    steps = [m["step"] for m in metrics]
    tiers = ["naive", "gc", "memory", "apoptotic"]
    tier_data = {t: [m["tier_distribution"].get(t, 0) for m in metrics] for t in tiers}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(
        steps,
        [tier_data[t] for t in tiers],
        labels=tiers,
        alpha=0.8,
    )

    ax.set_xlabel("Step")
    ax.set_ylabel("Entry Count")
    ax.set_title("Tier Distribution Over Time (GC arm)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / "tier_distribution.png"), dpi=150)
    plt.close(fig)


def print_summary(results: dict) -> None:
    """Print the summary table."""
    static_metrics = results["arms"]["static"]["metrics_by_step"]
    random_metrics = results["arms"]["random"]["metrics_by_step"]
    gc_metrics = results["arms"]["gc"]["metrics_by_step"]

    static_ndcg = static_metrics[-1]["ndcg_at_10"] if static_metrics else 0.0
    random_ndcg = random_metrics[-1]["ndcg_at_10"] if random_metrics else 0.0
    gc_ndcg = gc_metrics[-1]["ndcg_at_10"] if gc_metrics else 0.0

    def delta_str(value: float, baseline: float) -> str:
        if baseline == 0:
            return "N/A"
        pct = (value - baseline) / baseline * 100
        return f"{pct:+.1f}%"

    def breaker_str(arm: dict) -> str:
        if arm["halt_reason"]:
            return str(arm["halt_reason"])
        return "-"

    print("\n" + "=" * 70)
    print(f"{'Arm':<12} {'Final NDCG@10':<18} {'Delta vs Static':<18} {'Circuit Breaker'}")
    print("-" * 70)
    print(f"{'static':<12} {static_ndcg:<18.4f} {'-':<18} {breaker_str(results['arms']['static'])}")
    print(
        f"{'random':<12} {random_ndcg:<18.4f} "
        f"{delta_str(random_ndcg, static_ndcg):<18} "
        f"{breaker_str(results['arms']['random'])}"
    )
    print(
        f"{'gc':<12} {gc_ndcg:<18.4f} "
        f"{delta_str(gc_ndcg, static_ndcg):<18} "
        f"{breaker_str(results['arms']['gc'])}"
    )
    print("=" * 70)

    # Check success criteria
    print("\nSuccess Criteria:")
    gc_completed = results["arms"]["gc"]["completed"]
    gc_vs_static = (gc_ndcg - static_ndcg) / static_ndcg * 100 if static_ndcg else 0
    gc_vs_random = (gc_ndcg - random_ndcg) / random_ndcg * 100 if random_ndcg else 0

    print(f"  1. GC completes without circuit breakers: {'PASS' if gc_completed else 'FAIL'}")
    print(f"  2. GC NDCG > Static by >= 3%: {'PASS' if gc_vs_static >= 3 else 'FAIL'} ({gc_vs_static:+.1f}%)")
    print(f"  3. GC NDCG > Random by >= 1.5%: {'PASS' if gc_vs_random >= 1.5 else 'FAIL'} ({gc_vs_random:+.1f}%)")

    # Check monotonicity in rolling 2000-step window after step 2000
    gc_steps_and_ndcg = [
        (m["step"], m["ndcg_at_10"]) for m in gc_metrics if m["step"] >= 2000
    ]
    monotonic = True
    if len(gc_steps_and_ndcg) >= 5:  # need enough points for rolling window
        # Rolling window of ~2000 steps = 4 metric points (logged every 500)
        window_size = 4
        for i in range(window_size, len(gc_steps_and_ndcg)):
            window_start = gc_steps_and_ndcg[i - window_size][1]
            window_end = gc_steps_and_ndcg[i][1]
            if window_end < window_start - 0.001:  # small tolerance
                monotonic = False
                break
    print(f"  4. GC NDCG monotonically non-decreasing (2000-step window after step 2000): "
          f"{'PASS' if monotonic else 'FAIL'}")

    all_pass = gc_completed and gc_vs_static >= 3 and gc_vs_random >= 1.5 and monotonic
    print(f"\nOverall: {'POSITIVE SIGNAL' if all_pass else 'NEGATIVE RESULT'}")


def main() -> None:
    results = load_latest_results()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_ndcg(results)
    plot_diversity(results)
    plot_anchor_drift(results)
    plot_tier_distribution(results)
    print(f"Plots saved to {PLOTS_DIR}/")

    print_summary(results)


if __name__ == "__main__":
    main()

"""Driver for the demo collector.

Runs baseline and lethe passes as separate subprocesses (each with a fresh
Python memory space — otherwise k-means OOMs on a 199k × 384 corpus after
the first store has already been built). Merges the two per-query NDCG
arrays into demo/public/run.json.

Usage:
    uv run python demo/scripts/collect.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent
REPO = DEMO.parent
WORKER = HERE / "_pass.py"
OUT = DEMO / "public" / "run.json"
TMP = DEMO / ".pass_out"


def run_pass(
    label: str, alpha: float, use_rank_gap: bool, n_clusters: int,
) -> dict:
    TMP.mkdir(parents=True, exist_ok=True)
    out_json = TMP / f"{label}.json"
    subprocess.check_call([
        sys.executable,
        str(WORKER),
        label,
        str(alpha),
        str(use_rank_gap),
        str(n_clusters),
        str(out_json),
    ], cwd=str(REPO))
    payload = json.loads(out_json.read_text())
    return payload


def main() -> None:
    base = run_pass("baseline", alpha=0.0, use_rank_gap=False, n_clusters=0)
    leth = run_pass("lethe", alpha=0.3, use_rank_gap=True, n_clusters=30)

    # Align by qid. Baseline is state-free (alpha=0 → no learning), so any
    # occurrence of a qid has the same NDCG — we can look it up regardless
    # of position. We index the video's x-axis off lethe's sequence because
    # that's where state actually evolves.
    baseline_by_qid = dict(zip(base["qids"], base["ndcgs"]))
    missing = set(leth["qids"]) - set(baseline_by_qid)
    assert not missing, f"baseline didn't see {len(missing)} lethe qids"

    rows = []
    for i, qid in enumerate(leth["qids"]):
        rows.append({
            "idx": i,
            "qid": qid,
            "baseline": {"ndcg": round(baseline_by_qid[qid], 4)},
            "lethe": {"ndcg": round(leth["ndcgs"][i], 4)},
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {
            "fps": 30,
            "totalQueries": len(rows),
            "snapshotAt": [],
        },
        "queries": rows,
    }))
    n = len(rows)
    b_mean = sum(r["baseline"]["ndcg"] for r in rows) / n
    l_mean = sum(r["lethe"]["ndcg"] for r in rows) / n
    print(f"\nWrote {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print(
        f"Final mean NDCG@10: baseline={b_mean:.3f} "
        f"lethe={l_mean:.3f}  delta={l_mean - b_mean:+.3f}"
    )


if __name__ == "__main__":
    main()

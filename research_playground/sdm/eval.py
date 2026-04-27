"""Retrieval metrics for the SDM vs. baseline comparison.

Every retriever returns `list[tuple[memory_id, score]]` — we only look at
the memory_id order, not the scores.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Protocol

import numpy as np

from dataset import Dataset, Query


class Retriever(Protocol):
    """Minimal interface both SDM and FaissRetriever satisfy."""

    def read(self, query_text: str, top_k: int = 10) -> list[tuple[int, float]]: ...


def precision_at_1(pred_ids: list[int], target_id: int) -> float:
    return 1.0 if pred_ids and pred_ids[0] == target_id else 0.0


def recall_at_k(pred_ids: list[int], target_id: int, k: int) -> float:
    return 1.0 if target_id in pred_ids[:k] else 0.0


def exact_episode_accuracy(pred_ids: list[int], target_id: int) -> float:
    """Exact same as precision@1 — kept separate per user spec for clarity."""
    return precision_at_1(pred_ids, target_id)


def sibling_confusion(pred_ids: list[int], target_id: int, sibling_ids: list[int]) -> float:
    """1.0 when top-1 is a sibling of the target (same family) but not the target itself.

    This is the key metric for SDM vs. ANN: near-duplicate events should be
    distinguishable. A high confusion rate means the retriever can't tell
    the target apart from its siblings.
    """
    if not pred_ids:
        return 0.0
    top1 = pred_ids[0]
    siblings_without_target = set(sibling_ids) - {target_id}
    return 1.0 if top1 in siblings_without_target else 0.0


def run_eval(
    retriever: Retriever,
    queries: list[Query],
    k_values: tuple[int, ...] = (1, 5, 10),
    max_k: int = 10,
) -> dict[str, dict[str, float]]:
    """Run the retriever on all queries, grouped by noise mode.

    Returns {noise_mode: {metric_name: mean_score}} with an "overall" bucket.
    """
    # Per-mode accumulators.
    by_mode: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for q in queries:
        results = retriever.read(q.text, top_k=max_k)
        pred_ids = [mid for mid, _ in results]

        p1 = precision_at_1(pred_ids, q.target_id)
        confuse = sibling_confusion(pred_ids, q.target_id, q.sibling_ids)
        exact = exact_episode_accuracy(pred_ids, q.target_id)

        by_mode[q.noise_mode]["precision@1"].append(p1)
        by_mode[q.noise_mode]["sibling_confusion"].append(confuse)
        by_mode[q.noise_mode]["exact_episode"].append(exact)
        for k in k_values:
            by_mode[q.noise_mode][f"recall@{k}"].append(recall_at_k(pred_ids, q.target_id, k))

        by_mode["overall"]["precision@1"].append(p1)
        by_mode["overall"]["sibling_confusion"].append(confuse)
        by_mode["overall"]["exact_episode"].append(exact)
        for k in k_values:
            by_mode["overall"][f"recall@{k}"].append(recall_at_k(pred_ids, q.target_id, k))

    return {
        mode: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for mode, metrics in by_mode.items()
    }

"""Run the three-arm GC memory experiment: Static vs Random vs GC."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from gc_memory.baselines import RandomMutationStore, StaticStore
from gc_memory.config import Config
from gc_memory.entry import MemoryEntry, Tier, create_entry
from gc_memory.metrics import (
    compute_anchor_drift,
    compute_diversity,
    compute_mean_generation,
    compute_tier_distribution,
    ndcg_at_k,
    recall_at_k,
)
from gc_memory.store import GCMemoryStore


RESULTS_DIR = Path("results")
DATA_DIR = Path("data")


def load_prepared_data() -> tuple[
    list[str],
    npt.NDArray[np.float32],
    list[str],
    npt.NDArray[np.float32],
    dict[str, dict[str, int]],
    dict[str, str],
    dict[str, str],
]:
    """Load pre-computed embeddings and metadata."""
    data = np.load(str(DATA_DIR / "longmemeval_prepared.npz"), allow_pickle=True)
    corpus_ids = list(data["corpus_ids"])
    corpus_embeddings = data["corpus_embeddings"].astype(np.float32)
    query_ids = list(data["query_ids"])
    query_embeddings = data["query_embeddings"].astype(np.float32)

    with open(DATA_DIR / "longmemeval_qrels.json") as f:
        qrels = json.load(f)
    with open(DATA_DIR / "longmemeval_corpus.json") as f:
        corpus_content = json.load(f)
    with open(DATA_DIR / "longmemeval_queries.json") as f:
        query_texts = json.load(f)

    return corpus_ids, corpus_embeddings, query_ids, query_embeddings, qrels, corpus_content, query_texts


def build_entries(
    corpus_ids: list[str],
    corpus_embeddings: npt.NDArray[np.float32],
    corpus_content: dict[str, str],
) -> list[MemoryEntry]:
    entries = []
    for doc_id, embedding in zip(corpus_ids, corpus_embeddings):
        content = corpus_content.get(doc_id, "")
        entries.append(create_entry(doc_id, content, embedding))
    return entries


def build_query_schedule(
    query_ids: list[str],
    n_queries: int,
    hot_set_fraction: float,
    hot_set_probability: float,
    rng: np.random.Generator,
) -> list[str]:
    n_hot = max(1, int(len(query_ids) * hot_set_fraction))
    hot_indices = rng.choice(len(query_ids), size=n_hot, replace=False)
    hot_ids = [query_ids[i] for i in hot_indices]
    cold_ids = [qid for qid in query_ids if qid not in set(hot_ids)]
    if not cold_ids:
        cold_ids = hot_ids

    schedule = []
    for _ in range(n_queries):
        if rng.random() < hot_set_probability:
            schedule.append(hot_ids[rng.integers(len(hot_ids))])
        else:
            schedule.append(cold_ids[rng.integers(len(cold_ids))])
    return schedule


def log_metrics(
    store: GCMemoryStore,
    step: int,
    eval_query_ids: list[str],
    eval_query_embeddings: npt.NDArray[np.float32],
    eval_query_texts: dict[str, str],
    qrels: dict[str, dict[str, int]],
    rng: np.random.Generator,
) -> dict[str, object]:
    """Compute all metrics. NDCG uses bi-encoder retrieval (no cross-encoder at eval to keep it fast)."""
    all_entries = store.get_all_entries()
    active_entries = store.get_active_entries()

    # Evaluate retrieval quality using FAISS only (no cross-encoder, for speed)
    ndcg_scores = []
    recall_scores = []
    for qid, qemb in zip(eval_query_ids, eval_query_embeddings):
        query_rel = qrels.get(qid, {})
        if not query_rel:
            continue
        qtext = eval_query_texts.get(qid, "")
        retrieved = store.retrieve(qemb, qtext, k=store.config.k)
        retrieved_ids = [e.id for e, _ in retrieved]
        ndcg_scores.append(ndcg_at_k(retrieved_ids, query_rel, store.config.k))
        recall_scores.append(recall_at_k(retrieved_ids, query_rel, store.config.k))
    ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    recall = float(np.mean(recall_scores)) if recall_scores else 0.0

    if active_entries:
        embeddings = np.stack([e.embedding for e in active_entries])
        diversity = compute_diversity(embeddings, min(1000, len(active_entries) * 10), rng)
    else:
        diversity = 0.0

    drift = compute_anchor_drift(active_entries, min(100, len(active_entries)), rng)
    tier_dist = compute_tier_distribution(all_entries)
    mean_gen = compute_mean_generation(all_entries)
    gc_entries = [e for e in all_entries if e.tier == Tier.GC]
    mean_gc_affinity = float(np.mean([e.affinity for e in gc_entries])) if gc_entries else 0.0

    return {
        "step": step,
        "ndcg_at_10": ndcg,
        "recall_at_10": recall,
        "diversity": diversity,
        "anchor_drift": drift,
        "tier_distribution": {t.value: c for t, c in tier_dist.items()},
        "mean_generation": mean_gen,
        "mean_gc_affinity": mean_gc_affinity,
        "n_active": len(active_entries),
    }


def check_circuit_breakers(
    metrics_history: list[dict[str, object]],
    initial_diversity: float,
) -> str | None:
    if not metrics_history:
        return None
    latest = metrics_history[-1]

    diversity = float(latest["diversity"])  # type: ignore[arg-type]
    if initial_diversity > 0 and diversity < 0.9 * initial_diversity:
        return "DEGENERATE_CONVERGENCE"

    drift = float(latest["anchor_drift"])  # type: ignore[arg-type]
    if drift > 0.25:
        return "ANCHOR_DRIFT_EXCEEDED"

    gc_affinity = float(latest["mean_gc_affinity"])  # type: ignore[arg-type]
    gc_count = int(latest.get("tier_distribution", {}).get("gc", 0))  # type: ignore[union-attr]
    if gc_count > 0 and gc_affinity < 0.1:
        return "AFFINITY_COLLAPSE"

    return None


def run_arm(
    name: str,
    store: GCMemoryStore,
    query_schedule: list[str],
    query_id_to_idx: dict[str, int],
    query_ids: list[str],
    query_embeddings: npt.NDArray[np.float32],
    query_texts: dict[str, str],
    qrels: dict[str, dict[str, int]],
    config: Config,
    rng: np.random.Generator,
) -> dict[str, object]:
    metrics_history: list[dict[str, object]] = []
    initial_diversity: float | None = None
    halt_reason: str | None = None

    for step in tqdm(range(config.n_queries), desc=name):
        query_id = query_schedule[step]
        query_idx = query_id_to_idx[query_id]
        query_embedding = query_embeddings[query_idx]
        query_text = query_texts.get(query_id, "")

        retrieved = store.retrieve(query_embedding, query_text, k=config.k)
        store.update_after_retrieval(query_embedding, query_text, retrieved, step)

        if step % config.decay_interval == 0:
            store.run_decay(step)

        if step % 500 == 0:
            m = log_metrics(store, step, query_ids, query_embeddings, query_texts, qrels, rng)
            metrics_history.append(m)

            if initial_diversity is None:
                initial_diversity = float(m["diversity"])  # type: ignore[arg-type]

            halt_reason = check_circuit_breakers(metrics_history, initial_diversity or 0.0)
            if halt_reason:
                print(f"  [{name}] Circuit breaker: {halt_reason} at step {step}")
                break

    return {
        "completed": halt_reason is None,
        "halt_reason": halt_reason,
        "metrics_by_step": metrics_history,
    }


def deep_copy_entries(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    copied = []
    for e in entries:
        copied.append(MemoryEntry(
            id=e.id,
            content=e.content,
            base_embedding=e.base_embedding.copy(),
            embedding=e.embedding.copy(),
            adapter=e.adapter.copy(),
            affinity=e.affinity,
            retrieval_count=e.retrieval_count,
            generation=e.generation,
            last_retrieved_step=e.last_retrieved_step,
            tier=e.tier,
        ))
    return copied


def main() -> None:
    config = Config()
    rng = np.random.default_rng(config.random_seed)

    print("Loading prepared data...")
    corpus_ids, corpus_embeddings, query_ids, query_embeddings, qrels, corpus_content, query_texts = (
        load_prepared_data()
    )

    print(f"Building entries from {len(corpus_ids)} turns...")
    base_entries = build_entries(corpus_ids, corpus_embeddings, corpus_content)

    print("Building query schedule...")
    query_schedule = build_query_schedule(
        query_ids, config.n_queries, config.hot_set_fraction, config.hot_set_probability, rng,
    )
    query_id_to_idx = {qid: i for i, qid in enumerate(query_ids)}

    print("Loading cross-encoder...")
    xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    arms_results: dict[str, dict[str, object]] = {}

    print("\n=== Running Static arm (with cross-encoder rerank) ===")
    static_rng = np.random.default_rng(config.random_seed)
    static_store = StaticStore(deep_copy_entries(base_entries), config, static_rng, cross_encoder=xenc)
    arms_results["static"] = run_arm(
        "static", static_store, query_schedule, query_id_to_idx,
        query_ids, query_embeddings, query_texts, qrels, config, static_rng,
    )

    print("\n=== Running Random arm (with cross-encoder rerank) ===")
    random_rng = np.random.default_rng(config.random_seed)
    random_store = RandomMutationStore(deep_copy_entries(base_entries), config, random_rng, cross_encoder=xenc)
    arms_results["random"] = run_arm(
        "random", random_store, query_schedule, query_id_to_idx,
        query_ids, query_embeddings, query_texts, qrels, config, random_rng,
    )

    print("\n=== Running GC arm (with cross-encoder guided mutation) ===")
    gc_rng = np.random.default_rng(config.random_seed)
    gc_store = GCMemoryStore(deep_copy_entries(base_entries), config, gc_rng, cross_encoder=xenc)
    arms_results["gc"] = run_arm(
        "gc", gc_store, query_schedule, query_id_to_idx,
        query_ids, query_embeddings, query_texts, qrels, config, gc_rng,
    )

    any_halt = any(not arm["completed"] for arm in arms_results.values())
    halt_reasons = {
        name: arm["halt_reason"]
        for name, arm in arms_results.items()
        if arm["halt_reason"]
    }

    results = {
        "config": asdict(config),
        "seed": config.random_seed,
        "completed": not any_halt,
        "halt_reason": halt_reasons if halt_reasons else None,
        "arms": arms_results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"run_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    if halt_reasons:
        print(f"Circuit breakers fired: {halt_reasons}")
    else:
        print("All arms completed successfully.")


if __name__ == "__main__":
    main()

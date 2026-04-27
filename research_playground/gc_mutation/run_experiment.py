"""Run the v4 experiment: Static vs MLP Adapter vs Segmentation.

Usage:
    python research/gc_mutation/run_experiment.py --dataset nfcorpus
    python research/gc_mutation/run_experiment.py --dataset longmemeval
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "legacy"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))

import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from research_playground.gc_mutation.baselines import NoGraphStore, StaticStore
from research_playground.gc_mutation.config import Config
from lethe.entry import MemoryEntry, Tier, create_entry
from benchmarks._lib.metrics import (
    compute_anchor_drift,
    compute_diversity,
    compute_mean_generation,
    compute_tier_distribution,
    ndcg_at_k,
    recall_at_k,
)
from research_playground.gc_mutation.store import GCMemoryStore


RESULTS_DIR = Path("tmp_results")
DATA_DIR = Path("tmp_data")


def load_dataset_files(dataset: str) -> tuple[
    list[str], npt.NDArray[np.float32],
    list[str], npt.NDArray[np.float32],
    dict[str, dict[str, int]], dict[str, str],
    dict[str, str], dict[str, dict[str, object]],
]:
    prefix = dataset
    data = np.load(str(DATA_DIR / f"{prefix}_prepared.npz"), allow_pickle=True)
    corpus_ids = list(data["corpus_ids"])
    corpus_embeddings = data["corpus_embeddings"].astype(np.float32)
    query_ids = list(data["query_ids"])
    query_embeddings = data["query_embeddings"].astype(np.float32)

    with open(DATA_DIR / f"{prefix}_qrels.json") as f:
        qrels = json.load(f)
    with open(DATA_DIR / f"{prefix}_corpus.json") as f:
        corpus_content = json.load(f)
    with open(DATA_DIR / f"{prefix}_queries.json") as f:
        query_texts = json.load(f)
    with open(DATA_DIR / f"{prefix}_meta.json") as f:
        meta = json.load(f)

    return corpus_ids, corpus_embeddings, query_ids, query_embeddings, qrels, corpus_content, query_texts, meta


def build_entries(
    corpus_ids: list[str],
    corpus_embeddings: npt.NDArray[np.float32],
    corpus_content: dict[str, str],
    meta: dict[str, dict[str, object]],
) -> list[MemoryEntry]:
    entries = []
    for doc_id, embedding in zip(corpus_ids, corpus_embeddings):
        content = corpus_content.get(doc_id, "")
        m = meta.get(doc_id, {})
        entries.append(create_entry(
            doc_id, content, embedding,
            session_id=str(m.get("session_id", "")),
            turn_idx=int(m.get("turn_idx", 0)),
        ))
    return entries


def build_query_schedule(
    query_ids: list[str], n_queries: int,
    hot_set_fraction: float, hot_set_probability: float,
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
    store: GCMemoryStore, step: int,
    eval_query_ids: list[str], eval_query_embeddings: npt.NDArray[np.float32],
    eval_query_texts: dict[str, str],
    qrels: dict[str, dict[str, int]], rng: np.random.Generator,
) -> dict[str, object]:
    all_entries = store.get_all_entries()
    active_entries = store.get_active_entries()

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
        sample = active_entries[:min(5000, len(active_entries))]
        embeddings = np.stack([e.embedding for e in sample])
        diversity = compute_diversity(embeddings, min(1000, len(sample) * 10), rng)
    else:
        diversity = 0.0

    drift = compute_anchor_drift(active_entries, min(100, len(active_entries)), rng)
    tier_dist = compute_tier_distribution(all_entries)
    mean_gen = compute_mean_generation(all_entries)
    gc_entries = [e for e in all_entries if e.tier == Tier.GC]
    mean_gc_aff = float(np.mean([e.affinity for e in gc_entries])) if gc_entries else 0.0

    return {
        "step": step, "ndcg_at_10": ndcg, "recall_at_10": recall,
        "diversity": diversity, "anchor_drift": drift,
        "tier_distribution": {t.value: c for t, c in tier_dist.items()},
        "mean_generation": mean_gen, "mean_gc_affinity": mean_gc_aff,
        "n_active": len(active_entries), "n_total": len(all_entries),
        "graph_nodes": store.graph.num_nodes, "graph_edges": store.graph.num_edges,
    }


def check_circuit_breakers(
    metrics_history: list[dict[str, object]], initial_diversity: float,
) -> str | None:
    if not metrics_history:
        return None
    latest = metrics_history[-1]
    div = float(latest["diversity"])  # type: ignore[arg-type]
    if initial_diversity > 0 and div < 0.9 * initial_diversity:
        return "DEGENERATE_CONVERGENCE"
    drift = float(latest["anchor_drift"])  # type: ignore[arg-type]
    if drift > 0.25:
        return "ANCHOR_DRIFT_EXCEEDED"
    gc_aff = float(latest["mean_gc_affinity"])  # type: ignore[arg-type]
    gc_count = int(latest.get("tier_distribution", {}).get("gc", 0))  # type: ignore[union-attr]
    if gc_count > 0 and gc_aff < 0.1:
        return "AFFINITY_COLLAPSE"
    return None


def run_arm(
    name: str, store: GCMemoryStore,
    query_schedule: list[str], query_id_to_idx: dict[str, int],
    query_ids: list[str], query_embeddings: npt.NDArray[np.float32],
    query_texts: dict[str, str], qrels: dict[str, dict[str, int]],
    config: Config, rng: np.random.Generator,
) -> dict[str, object]:
    metrics_history: list[dict[str, object]] = []
    initial_diversity: float | None = None
    halt_reason: str | None = None

    for step in tqdm(range(config.n_queries), desc=name):
        qid = query_schedule[step]
        qidx = query_id_to_idx[qid]
        qemb = query_embeddings[qidx]
        qtxt = query_texts.get(qid, "")

        retrieved = store.retrieve(qemb, qtxt, k=config.k)
        store.update_after_retrieval(qemb, qtxt, retrieved, step)

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

    return {"completed": halt_reason is None, "halt_reason": halt_reason, "metrics_by_step": metrics_history}


def deep_copy_entries(entries: list[MemoryEntry]) -> list[MemoryEntry]:
    return [
        MemoryEntry(
            id=e.id, content=e.content,
            base_embedding=e.base_embedding.copy(), embedding=e.embedding.copy(),
            adapter=e.adapter.copy(), session_id=e.session_id, turn_idx=e.turn_idx,
            affinity=e.affinity, retrieval_count=e.retrieval_count,
            generation=e.generation, last_retrieved_step=e.last_retrieved_step, tier=e.tier,
        )
        for e in entries
    ]


def run_dataset(dataset: str, bi_encoder: SentenceTransformer, xenc: CrossEncoder) -> dict[str, object]:
    config = Config()
    rng = np.random.default_rng(config.random_seed)

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    corpus_ids, corpus_embeddings, query_ids, query_embeddings, qrels, corpus_content, query_texts, meta = (
        load_dataset_files(dataset)
    )
    print(f"Corpus: {len(corpus_ids)} entries, Queries: {len(query_ids)}")

    base_entries = build_entries(corpus_ids, corpus_embeddings, corpus_content, meta)
    query_schedule = build_query_schedule(
        query_ids, config.n_queries, config.hot_set_fraction, config.hot_set_probability, rng,
    )
    query_id_to_idx = {qid: i for i, qid in enumerate(query_ids)}

    arms_results: dict[str, dict[str, object]] = {}

    print("\n--- Static (no graph, no updates) ---")
    s_rng = np.random.default_rng(config.random_seed)
    static_store = StaticStore(deep_copy_entries(base_entries), config, s_rng, cross_encoder=xenc)
    arms_results["static"] = run_arm(
        "static", static_store, query_schedule, query_id_to_idx,
        query_ids, query_embeddings, query_texts, qrels, config, s_rng,
    )

    print("\n--- No-graph (rerank + tiers, no graph expansion) ---")
    n_rng = np.random.default_rng(config.random_seed)
    nograph_store = NoGraphStore(deep_copy_entries(base_entries), config, n_rng, cross_encoder=xenc)
    arms_results["nograph"] = run_arm(
        "nograph", nograph_store, query_schedule, query_id_to_idx,
        query_ids, query_embeddings, query_texts, qrels, config, n_rng,
    )

    print("\n--- Graph GC (full: graph expansion + tiers + decay) ---")
    g_rng = np.random.default_rng(config.random_seed)
    gc_store = GCMemoryStore(deep_copy_entries(base_entries), config, g_rng, cross_encoder=xenc)
    arms_results["gc"] = run_arm(
        "gc", gc_store, query_schedule, query_id_to_idx,
        query_ids, query_embeddings, query_texts, qrels, config, g_rng,
    )
    print(f"  Graph: {gc_store.graph.num_nodes} nodes, {gc_store.graph.num_edges} edges")

    return {
        "dataset": dataset,
        "config": asdict(config),
        "seed": config.random_seed,
        "completed": all(a["completed"] for a in arms_results.values()),
        "arms": arms_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nfcorpus", "longmemeval"], required=True)
    args = parser.parse_args()

    print("Loading cross-encoder...")
    xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    results = run_dataset(args.dataset, None, xenc)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"run_{args.dataset}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

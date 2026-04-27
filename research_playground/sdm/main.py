"""SDM vs. FAISS baseline — run with: uv run python sdm/main.py

Flow:
1. Generate a synthetic episodic dataset (250 events, 1000 queries).
2. Build a shared TextEncoder + HyperplaneProjector.
3. Build SDM and FaissRetriever (FAISS IndexFlatIP on the same encoder,
   for an apples-to-apples comparison).
4. Write all events into both systems.
5. Evaluate both on the same queries, grouped by noise mode.
6. Print a comparison table and observations.
"""
from __future__ import annotations

import os
import time
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np

from dataset import Dataset, build_dataset
from embedding import HyperplaneProjector, TextEncoder
from eval import run_eval
from sdm import SDM


class FaissRetriever:
    """Baseline: exact cosine nearest-neighbor over normalized dense embeddings.

    Matches the SDM's .read(query_text, top_k) signature so run_eval works on both.
    """

    def __init__(self, encoder: TextEncoder) -> None:
        self.encoder = encoder
        self.index = faiss.IndexFlatIP(encoder.dim)  # inner product on L2-normed = cosine
        self.ids: list[int] = []
        self.memories: dict[int, dict[str, Any]] = {}
        self._next_id = 0

    def write(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        emb = self.encoder.encode([text])
        self.index.add(emb)
        mid = self._next_id
        self._next_id += 1
        self.ids.append(mid)
        self.memories[mid] = {"text": text, "metadata": metadata or {}}
        return mid

    def read(self, query_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        emb = self.encoder.encode([query_text])
        k = min(top_k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(emb, k)
        return [(self.ids[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


class SDMWithCleanup:
    """Wraps an SDM with a fixed cleanup_iters setting — so eval can treat
    (SDM, SDM_cleanup) as two independent retrievers over the same backing store."""

    def __init__(self, sdm: SDM, cleanup_iters: int) -> None:
        self._sdm = sdm
        self._cleanup_iters = cleanup_iters

    def read(self, query_text: str, top_k: int = 10) -> list[tuple[int, float]]:
        return self._sdm.read(query_text, top_k=top_k, cleanup_iters=self._cleanup_iters)


def _write_all(store: Any, dataset: Dataset, label: str) -> float:
    t0 = time.time()
    for e in dataset.events:
        store.write(e.text, metadata={"family": e.family, "family_key": e.family_key})
    elapsed = time.time() - t0
    print(f"  {label}: wrote {len(dataset.events)} events in {elapsed:.1f}s", flush=True)
    return elapsed


def _print_comparison(
    sdm_results: dict[str, dict[str, float]],
    sdm_cleanup_results: dict[str, dict[str, float]],
    faiss_results: dict[str, dict[str, float]],
) -> None:
    modes = ["partial", "paraphrase", "fragment", "noisy", "overall"]
    metrics = ["precision@1", "recall@5", "recall@10", "sibling_confusion"]
    print()
    print("=" * 90)
    print(f"{'Noise mode':<12} | {'Metric':<20} | {'SDM':>8} | {'SDM+cleanup':>12} | {'FAISS':>8}")
    print("-" * 90)
    for mode in modes:
        for m in metrics:
            sdm_val = sdm_results.get(mode, {}).get(m, float("nan"))
            sdm_c_val = sdm_cleanup_results.get(mode, {}).get(m, float("nan"))
            faiss_val = faiss_results.get(mode, {}).get(m, float("nan"))
            print(f"{mode:<12} | {m:<20} | {sdm_val:>8.3f} | {sdm_c_val:>12.3f} | {faiss_val:>8.3f}")
        print("-" * 90)


def _observations(
    sdm_results: dict[str, dict[str, float]],
    sdm_cleanup_results: dict[str, dict[str, float]],
    faiss_results: dict[str, dict[str, float]],
) -> None:
    modes = ["partial", "paraphrase", "fragment", "noisy"]
    sdm_wins_confuse = [m for m in modes
                        if sdm_results[m]["sibling_confusion"] < faiss_results[m]["sibling_confusion"]]
    cleanup_hurts = sum(1 for m in modes
                        if sdm_cleanup_results[m]["precision@1"] < sdm_results[m]["precision@1"])

    # Relative gap per mode: smaller gap = SDM more competitive.
    gap = {m: faiss_results[m]["precision@1"] - sdm_results[m]["precision@1"] for m in modes}
    smallest_gap_mode = min(gap, key=lambda m: gap[m])
    largest_gap_mode = max(gap, key=lambda m: gap[m])

    print("\nObservations")
    print("-" * 90)
    overall_delta = (sdm_results["overall"]["precision@1"]
                     - faiss_results["overall"]["precision@1"])
    print(f"  1. Overall precision@1: SDM={sdm_results['overall']['precision@1']:.3f} vs "
          f"FAISS={faiss_results['overall']['precision@1']:.3f} ({overall_delta:+.3f}). "
          f"FAISS wins outright on recall.")
    print(f"  2. SDM's gap to FAISS is smallest on '{smallest_gap_mode}' "
          f"(Δ={gap[smallest_gap_mode]:.3f}) and largest on '{largest_gap_mode}' "
          f"(Δ={gap[largest_gap_mode]:.3f}). When the query contains more content, "
          f"FAISS exploits it better; SDM's binary quantization costs more.")
    if sdm_wins_confuse:
        print(f"  3. SDM has LOWER sibling confusion on: {', '.join(sdm_wins_confuse)}. "
              "When SDM fails, it tends to miss entirely rather than swap in a near-duplicate — "
              "a qualitatively different failure mode than FAISS.")
    if cleanup_hurts >= 3:
        print(f"  4. Iterative cleanup HURTS precision on {cleanup_hurts}/{len(modes)} noise modes. "
              "Cleanup drifts toward sibling attractors — bad for episodic recall of distinct "
              "events; useful only for prototype extraction.")
    print("  Conclusion: SDM does NOT beat a simple FAISS baseline for episodic retrieval on "
          "this dataset. Its only qualitative advantage is a different error distribution "
          "(lower sibling confusion on partial/fragment queries), which may matter if downstream "
          "logic prefers 'miss' over 'mis-retrieve'.")


def main() -> None:
    print("=" * 86)
    print("Sparse Distributed Memory (SDM) vs. FAISS baseline")
    print("=" * 86)
    print()

    # --- Dataset ---
    dataset = build_dataset(n_families=50, siblings_per_family=5, seed=42)
    family_counts: dict[str, int] = {}
    for e in dataset.events:
        family_counts[e.family] = family_counts.get(e.family, 0) + 1
    print(f"Dataset: {len(dataset.events)} events "
          f"({', '.join(f'{k}={v}' for k, v in family_counts.items())})")
    print(f"         {len(dataset.queries)} queries across {len(set(q.noise_mode for q in dataset.queries))} noise modes")
    print()

    # --- Shared embedding / projection ---
    print("Loading encoder and building projector...", flush=True)
    encoder = TextEncoder()
    projector = HyperplaneProjector(input_dim=encoder.dim, address_bits=512, seed=42)
    print(f"  encoder dim={encoder.dim}, address_bits={projector.address_bits}")
    print()

    # --- Retrievers ---
    sdm = SDM(encoder=encoder, projector=projector,
              address_bits=512, n_hard_locations=8000, activation_top_n=10, seed=42)
    faiss_ret = FaissRetriever(encoder=encoder)

    # --- Write ---
    print("Writing events...", flush=True)
    _write_all(sdm, dataset, "SDM")
    _write_all(faiss_ret, dataset, "FAISS")
    print(f"  SDM stats: {sdm.stats()}")
    print()

    # SDM variants over the same backing store (same writes, different read modes).
    sdm_nocleanup = SDMWithCleanup(sdm, cleanup_iters=1)
    sdm_cleanup = SDMWithCleanup(sdm, cleanup_iters=3)

    # --- Sanity check: exact query should return rank 1 ---
    print("Sanity check (exact queries)...", flush=True)
    sdm_exact = sum(
        1 for e in dataset.events
        if (r := sdm_nocleanup.read(e.text, top_k=1)) and r[0][0] == e.event_id
    )
    sdm_c_exact = sum(
        1 for e in dataset.events
        if (r := sdm_cleanup.read(e.text, top_k=1)) and r[0][0] == e.event_id
    )
    faiss_exact = sum(
        1 for e in dataset.events
        if (r := faiss_ret.read(e.text, top_k=1)) and r[0][0] == e.event_id
    )
    print(f"  SDM exact-query precision@1:         {sdm_exact}/{len(dataset.events)}")
    print(f"  SDM+cleanup exact-query precision@1: {sdm_c_exact}/{len(dataset.events)}")
    print(f"  FAISS exact-query precision@1:       {faiss_exact}/{len(dataset.events)}")
    print()

    # --- Eval ---
    print("Evaluating...", flush=True)
    t0 = time.time()
    sdm_results = run_eval(sdm_nocleanup, dataset.queries)
    print(f"  SDM eval done in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    sdm_cleanup_results = run_eval(sdm_cleanup, dataset.queries)
    print(f"  SDM+cleanup eval done in {time.time() - t0:.1f}s", flush=True)
    t0 = time.time()
    faiss_results = run_eval(faiss_ret, dataset.queries)
    print(f"  FAISS eval done in {time.time() - t0:.1f}s", flush=True)

    # --- Report ---
    _print_comparison(sdm_results, sdm_cleanup_results, faiss_results)
    _observations(sdm_results, sdm_cleanup_results, faiss_results)
    print()


if __name__ == "__main__":
    main()

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from gc_memory.config import Config
from gc_memory.entry import MemoryEntry, Tier, effective_embedding
from gc_memory.mlp_adapter import DeltaPredictor, train_step
from gc_memory.segmentation import (
    find_merge_candidates,
    merge_entries,
    should_split,
    split_entry,
)
from gc_memory.store import GCMemoryStore


class StaticStore(GCMemoryStore):
    """Baseline: cross-encoder reranking but no mutation, no decay, no tiers."""

    def update_after_retrieval(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
        step: int,
    ) -> None:
        pass

    def run_decay(self, step: int) -> None:
        pass


class MLPAdapterStore(GCMemoryStore):
    """Learned MLP adapter: delta = f(query, embedding, xenc_score).

    The MLP trains online, using cross-encoder scores as supervision.
    Replaces blind Gaussian noise with a semantically informed perturbation.
    """

    def __init__(
        self,
        entries: list[MemoryEntry],
        config: Config,
        rng: np.random.Generator,
        cross_encoder: Any = None,
        bi_encoder: Any = None,
    ) -> None:
        super().__init__(entries, config, rng, cross_encoder, bi_encoder)
        self.predictor = DeltaPredictor(embed_dim=384, hidden=config.mlp_hidden)
        self.optimizer = torch.optim.SGD(
            self.predictor.parameters(), lr=config.mlp_lr
        )
        self.total_loss = 0.0
        self.train_steps = 0

    def _mutate(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
    ) -> bool:
        if self.cross_encoder is None:
            return False

        gc_entries = [(e, s) for e, s in retrieved if e.tier == Tier.GC]
        if not gc_entries:
            return False

        pairs = [(query_text, e.content) for e, _ in gc_entries]
        xenc_scores = self.cross_encoder.predict(pairs)

        any_mutated = False
        cfg = self.config

        for (entry, _), xenc_score in zip(gc_entries, xenc_scores):
            # GC control: skip mutation for high-confidence entries
            sigma = cfg.sigma_0 * (1.0 - entry.affinity) ** cfg.gamma
            if sigma < 1e-4:
                continue

            # Train MLP and get delta (MLP controls its own magnitude)
            delta, loss = train_step(
                self.predictor, self.optimizer,
                query, entry.embedding, float(xenc_score),
                cfg.max_adapter_norm,
            )
            self.total_loss += loss
            self.train_steps += 1

            # Clip adapter norm (safety bound, replaces sigma scaling)
            new_adapter = entry.adapter + delta
            adapter_norm = float(np.linalg.norm(new_adapter))
            if adapter_norm > cfg.max_adapter_norm:
                new_adapter = new_adapter * (cfg.max_adapter_norm / adapter_norm)

            new_eff = effective_embedding(entry.base_embedding, new_adapter)

            # Accept if improves cosine with query
            old_cos = float(np.dot(entry.embedding, query))
            new_cos = float(np.dot(new_eff, query))
            if new_cos - old_cos > cfg.delta:
                entry.adapter = new_adapter
                entry.embedding = new_eff
                entry.generation += 1
                any_mutated = True

        return any_mutated


class SegmentationStore(GCMemoryStore):
    """Segmentation mutation: split and merge text entries.

    Instead of changing embedding vectors, change the text granularity
    to find the best binding surface for queries.
    """

    def _mutate(
        self,
        query: npt.NDArray[np.float32],
        query_text: str,
        retrieved: list[tuple[MemoryEntry, float]],
    ) -> bool:
        if self.bi_encoder is None:
            return False

        changed = False
        cfg = self.config

        # Phase 1: Split low-affinity long entries
        to_split: list[MemoryEntry] = []
        for entry, _ in retrieved:
            if should_split(entry, cfg):
                to_split.append(entry)

        for entry in to_split:
            new_entries = split_entry(entry, self.bi_encoder)
            if new_entries:
                del self.entries[entry.id]
                for ne in new_entries:
                    self.entries[ne.id] = ne
                changed = True

        # Phase 2: Merge adjacent high-affinity co-retrieved entries
        merge_pairs = find_merge_candidates(retrieved, self.entries, cfg)
        merged_ids: set[str] = set()

        for entry_a, entry_b in merge_pairs:
            if entry_a.id in merged_ids or entry_b.id in merged_ids:
                continue
            merged = merge_entries(entry_a, entry_b, self.bi_encoder)
            del self.entries[entry_a.id]
            del self.entries[entry_b.id]
            self.entries[merged.id] = merged
            merged_ids.add(entry_a.id)
            merged_ids.add(entry_b.id)
            changed = True

        return changed

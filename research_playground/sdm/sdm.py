"""Sparse Distributed Memory (Kanerva 1988) — research prototype.

Core ideas:
- A large pool of random binary "hard locations" in {0, 1}^D covers the
  address space sparsely.
- Each hard location keeps a per-bit bipolar counter.
- Writing a memory: project its content to a binary address, activate the
  K nearest hard locations (Hamming distance), update their counters
  toward the address (bit=1 → +1, bit=0 → −1).
- Reading: activate the K nearest hard locations to the query address,
  sum counters, threshold at zero → reconstructed binary pattern.
  Rank stored memories by Hamming similarity to the reconstructed pattern.

This module is encoder-agnostic: it receives TextEncoder + HyperplaneProjector
instances, but its logic operates only on binary addresses.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from embedding import HyperplaneProjector, TextEncoder


class SDM:
    def __init__(
        self,
        encoder: TextEncoder,
        projector: HyperplaneProjector,
        address_bits: int = 512,
        n_hard_locations: int = 2000,
        activation_top_n: int = 20,
        seed: int = 42,
    ) -> None:
        assert projector.address_bits == address_bits, (
            "projector.address_bits must match SDM.address_bits"
        )
        self.encoder = encoder
        self.projector = projector
        self.address_bits = address_bits
        self.n_hard_locations = n_hard_locations
        self.activation_top_n = activation_top_n

        rng = np.random.default_rng(seed)
        # Random binary hard locations: shape (n_hard_locations, address_bits).
        self.hard_addresses: npt.NDArray[np.bool_] = (
            rng.integers(0, 2, size=(n_hard_locations, address_bits)) == 1
        )
        # Per-(location, bit) bipolar counters.
        self.counters: npt.NDArray[np.int16] = np.zeros(
            (n_hard_locations, address_bits), dtype=np.int16,
        )
        # Which memories were written through each hard location (diagnostic).
        self.location_memories: list[list[int]] = [[] for _ in range(n_hard_locations)]
        # Stored memory metadata.
        self.memory_addresses: dict[int, npt.NDArray[np.bool_]] = {}
        self.memories: dict[int, dict[str, Any]] = {}
        self._next_id: int = 0

    # ---------- internals ----------

    def _text_to_address(self, text: str) -> npt.NDArray[np.bool_]:
        dense = self.encoder.encode([text])
        return self.projector.project(dense)[0]

    def _find_activated(self, address: npt.NDArray[np.bool_]) -> npt.NDArray[np.intp]:
        """Return indices of the top-N hard locations by Hamming proximity."""
        # xor with broadcasting: (n_hard, bits) vs (bits,) → per-row popcount.
        hamming = np.sum(self.hard_addresses ^ address, axis=1)
        # Partition-select the N smallest (argpartition is O(n), not O(n log n)).
        return np.argpartition(hamming, self.activation_top_n)[: self.activation_top_n]

    def _rank_memories(
        self, reconstructed: npt.NDArray[np.bool_], top_k: int,
    ) -> list[tuple[int, float]]:
        """Rank every stored memory by Hamming similarity to the reconstructed address."""
        if not self.memory_addresses:
            return []
        ids = list(self.memory_addresses.keys())
        mem_mat = np.stack([self.memory_addresses[mid] for mid in ids])  # (M, bits)
        hamming = np.sum(mem_mat ^ reconstructed, axis=1)  # (M,)
        sim = 1.0 - hamming / self.address_bits
        order = np.argsort(-sim)  # descending
        return [(ids[i], float(sim[i])) for i in order[:top_k]]

    # ---------- public API ----------

    def write(self, memory_text: str, metadata: dict[str, Any] | None = None) -> int:
        """Store a memory. Returns its id."""
        address = self._text_to_address(memory_text)
        activated = self._find_activated(address)
        # Bipolar update: +1 where bit is 1, −1 where bit is 0.
        update = (2 * address.astype(np.int16) - 1)  # (bits,)
        self.counters[activated] += update  # broadcasts across activated rows
        memory_id = self._next_id
        self._next_id += 1
        for idx in activated:
            self.location_memories[int(idx)].append(memory_id)
        self.memory_addresses[memory_id] = address
        self.memories[memory_id] = {"text": memory_text, "metadata": metadata or {}}
        return memory_id

    def read_from_binary_address(
        self,
        address: npt.NDArray[np.bool_],
        top_k: int = 10,
    ) -> tuple[list[tuple[int, float]], npt.NDArray[np.bool_]]:
        """Read given an already-projected binary address. Returns (ranked_memories, reconstructed)."""
        activated = self._find_activated(address)
        summed = self.counters[activated].sum(axis=0)  # (bits,)
        reconstructed = summed > 0  # majority vote at each bit
        ranked = self._rank_memories(reconstructed, top_k=top_k)
        return ranked, reconstructed

    def read(
        self,
        query_text: str,
        top_k: int = 10,
        cleanup_iters: int = 1,
        convergence_bits: int = 5,
    ) -> list[tuple[int, float]]:
        """Read from text. Optional iterative cleanup: the reconstructed
        pattern is re-fed into the SDM for up to `cleanup_iters` iterations,
        stopping when the reconstruction stabilizes (< `convergence_bits`
        changed bits).

        Default: cleanup_iters=1 (no iteration beyond initial read). For
        episodic memory with near-duplicates, cleanup tends to drift into
        sibling attractors and HURTS retrieval. Cleanup helps when you want
        to recall the prototype of a family, not a specific episode.
        """
        address = self._text_to_address(query_text)
        results, reconstructed = self.read_from_binary_address(address, top_k=top_k)
        for _ in range(max(0, cleanup_iters - 1)):
            new_results, new_reconstructed = self.read_from_binary_address(
                reconstructed, top_k=top_k,
            )
            bit_change = int(np.sum(reconstructed ^ new_reconstructed))
            results = new_results
            reconstructed = new_reconstructed
            if bit_change <= convergence_bits:
                break
        return results

    # ---------- diagnostics ----------

    def stats(self) -> dict[str, Any]:
        n_mem = len(self.memories)
        counter_abs = np.abs(self.counters)
        return {
            "n_memories": n_mem,
            "n_hard_locations": self.n_hard_locations,
            "activation_top_n": self.activation_top_n,
            "address_bits": self.address_bits,
            "counter_max_abs": int(counter_abs.max()) if n_mem else 0,
            "counter_mean_abs": float(counter_abs.mean()),
            "locations_used": int(
                sum(1 for ml in self.location_memories if ml),
            ),
        }

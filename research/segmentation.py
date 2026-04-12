"""Memory segmentation mutation: split and merge text entries.

Analogous to immune affinity maturation for epitope coverage.
Instead of changing embedding vectors, change the text granularity
to find the best "binding surface" for queries.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import numpy.typing as npt

from gc_memory.config import Config
from gc_memory.entry import MemoryEntry, Tier, create_entry


# Sentence boundary regex: handles common abbreviations
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex. Returns at least one segment."""
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def should_split(entry: MemoryEntry, config: Config) -> bool:
    """Entry should split if GC-tier, low affinity, and long content."""
    return (
        entry.tier == Tier.GC
        and entry.affinity < config.split_affinity_threshold
        and len(entry.content) > config.split_min_content_len
    )


def split_entry(
    entry: MemoryEntry,
    bi_encoder: Any,
) -> list[MemoryEntry]:
    """Split entry into sentence-level sub-entries with fresh embeddings.

    Returns the new entries (does not modify the original).
    If content splits into only one sentence, returns empty list (no split).
    """
    sentences = split_sentences(entry.content)
    if len(sentences) <= 1:
        return []

    new_entries = []
    for i, sent in enumerate(sentences):
        if len(sent) < 20:  # skip very short fragments
            continue
        new_id = f"{entry.id}_s{i}"
        emb = bi_encoder.encode(sent, normalize_embeddings=True).astype(np.float32)
        new_entry = create_entry(
            new_id, sent, emb,
            session_id=entry.session_id,
            turn_idx=entry.turn_idx,
        )
        # Inherit some state from parent
        new_entry.affinity = entry.affinity
        new_entry.tier = Tier.NAIVE  # reset tier (needs to earn GC again)
        new_entries.append(new_entry)

    return new_entries


def find_merge_candidates(
    retrieved: list[tuple[MemoryEntry, float]],
    all_entries: dict[str, MemoryEntry],
    config: Config,
) -> list[tuple[MemoryEntry, MemoryEntry]]:
    """Find adjacent entry pairs that should merge.

    Criteria: same session, sequential turn indices, both high affinity,
    both retrieved in this query.
    """
    retrieved_ids = {e.id for e, _ in retrieved}
    candidates: list[tuple[MemoryEntry, MemoryEntry]] = []

    for entry, _ in retrieved:
        if entry.affinity < config.merge_affinity_threshold:
            continue
        if not entry.session_id:
            continue

        # Look for the next turn in the same session
        next_id_candidates = [
            eid for eid, e in all_entries.items()
            if e.session_id == entry.session_id
            and e.turn_idx == entry.turn_idx + 1
            and eid in retrieved_ids
            and e.affinity >= config.merge_affinity_threshold
        ]
        for next_id in next_id_candidates:
            next_entry = all_entries[next_id]
            # Avoid duplicate pairs (only add if first has lower turn_idx)
            if entry.turn_idx < next_entry.turn_idx:
                candidates.append((entry, next_entry))

    return candidates


def merge_entries(
    entry_a: MemoryEntry,
    entry_b: MemoryEntry,
    bi_encoder: Any,
) -> MemoryEntry:
    """Merge two adjacent entries into one with a fresh embedding."""
    merged_content = f"{entry_a.content} {entry_b.content}"
    merged_id = f"{entry_a.id}+{entry_b.id}"
    emb = bi_encoder.encode(merged_content, normalize_embeddings=True).astype(np.float32)
    merged = create_entry(
        merged_id, merged_content, emb,
        session_id=entry_a.session_id,
        turn_idx=entry_a.turn_idx,
    )
    # Average affinity from parents, keep higher generation
    merged.affinity = (entry_a.affinity + entry_b.affinity) / 2.0
    merged.tier = Tier.GC  # merged entries go to GC (they've proven co-relevance)
    merged.generation = max(entry_a.generation, entry_b.generation) + 1
    return merged

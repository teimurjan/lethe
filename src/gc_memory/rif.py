"""Retrieval-Induced Forgetting: principled competitor suppression.

When a memory is successfully retrieved, competing memories (those activated
but not selected) are suppressed proportionally to their competition strength.
Over time, chronic distractors get pushed down, freeing candidate slots for
previously-excluded relevant entries.

Based on Anderson's inhibition theory (1994) and the SAM competitive sampling
model (Raaijmakers & Shiffrin, 1981). First implementation in an AI memory system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RIFConfig:
    suppression_rate: float = 0.1
    reinforcement_rate: float = 0.05
    max_suppression: float = 1.0
    decay_lambda: float = 0.005
    alpha: float = 0.3  # weight of suppression in score adjustment


def competition_strength(
    initial_rank: int,
    pool_size: int,
    xenc_score: float,
) -> float:
    """Compute how strongly an entry competed but lost.

    High initial rank (retrieved early by BM25/FAISS) + low cross-encoder
    score = strong competitor (distractor). This follows the cognitive science
    finding that stronger competitors receive more suppression.

    The nonmonotonic property: very weak competitors (high rank) produce
    near-zero strength. Peak suppression hits entries that looked relevant
    on surface but weren't.
    """
    if pool_size <= 1:
        return 0.0
    rank_score = 1.0 - (initial_rank / pool_size)
    rejection = 1.0 / (1.0 + math.exp(xenc_score))  # sigmoid(-score)
    return rank_score * rejection


def apply_suppression_penalty(
    candidates: list[tuple[str, float]],
    suppression_scores: dict[str, float],
    alpha: float,
) -> list[tuple[str, float]]:
    """Adjust candidate scores by subtracting suppression penalty.

    Returns re-sorted candidates with effective scores.
    """
    adjusted = [
        (eid, score - alpha * suppression_scores.get(eid, 0.0))
        for eid, score in candidates
    ]
    adjusted.sort(key=lambda x: x[1], reverse=True)
    return adjusted


def update_suppression(
    winner_ids: set[str],
    competitor_scores: list[tuple[str, int, float]],  # (id, initial_rank, xenc_score)
    current_suppression: dict[str, float],
    pool_size: int,
    config: RIFConfig,
    current_step: int,
    last_updated: dict[str, int],
) -> dict[str, float]:
    """Apply RIF updates after a retrieval event.

    1. Decay existing suppressions based on elapsed steps
    2. Suppress competitors proportional to competition strength
    3. Reinforce winners (reduce their suppression)

    Returns updated suppression scores for affected entries.
    """
    updated: dict[str, float] = {}

    # Decay + suppress competitors
    for eid, initial_rank, xenc_score in competitor_scores:
        if eid in winner_ids:
            continue
        old = current_suppression.get(eid, 0.0)
        # Decay based on steps elapsed since last update
        elapsed = current_step - last_updated.get(eid, current_step)
        if elapsed > 0:
            old *= math.exp(-config.decay_lambda * elapsed)
        # Add competition-proportional suppression
        strength = competition_strength(initial_rank, pool_size, xenc_score)
        new = min(old + strength * config.suppression_rate, config.max_suppression)
        updated[eid] = new

    # Reinforce winners
    for eid in winner_ids:
        old = current_suppression.get(eid, 0.0)
        elapsed = current_step - last_updated.get(eid, current_step)
        if elapsed > 0:
            old *= math.exp(-config.decay_lambda * elapsed)
        updated[eid] = max(0.0, old - config.reinforcement_rate)

    return updated

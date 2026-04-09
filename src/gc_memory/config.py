from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Retrieval
    k: int = 10
    tier_weight_memory: float = 1.15
    # Affinity
    alpha: float = 0.2
    # Mutation
    sigma_0: float = 0.05
    gamma: float = 2.0
    n_mutants: int = 5
    delta: float = 0.01
    theta_anchor: float = 0.7
    # Tier transitions
    promote_naive_threshold: int = 3
    promote_memory_affinity: float = 0.75
    promote_memory_generation: int = 5
    apoptosis_affinity: float = 0.15
    apoptosis_idle_steps: int = 1000
    # Decay
    lambda_decay: float = 0.01
    decay_interval: int = 100
    # Experiment
    n_queries: int = 10_000
    hot_set_fraction: float = 0.2
    hot_set_probability: float = 0.7
    random_seed: int = 42

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Retrieval
    k: int = 10
    k_fetch: int = 30  # FAISS fetch (graph adds ON TOP of this)
    tier_weight_memory: float = 1.15
    # Graph expansion
    graph_max_neighbors: int = 30
    graph_seed_k: int = 10  # use top-k FAISS results as graph seeds
    graph_expand_per_seed: int = 10  # neighbors per seed entry
    graph_learn_top_k: int = 10  # top-K of scored candidates form clique
    graph_reinforce_weight: float = 1.0  # edge weight added per reinforcement
    # Rescue index (deep mining)
    rescue_mine_interval: int = 5  # mine every N steps
    rescue_mine_k: int = 100  # FAISS top-k to mine from
    rescue_score_threshold: float = 0.0  # xenc score to qualify for rescue
    rescue_max_size: int = 5000  # max stored rescue entries
    rescue_lookup_k: int = 20  # how many rescue entries to add per query
    rescue_similarity_threshold: float = 0.5  # min query cosine to lookup
    # Affinity
    alpha: float = 0.2
    # Cross-encoder thresholds
    xenc_relevant: float = 0.0
    xenc_irrelevant: float = -4.0
    # Tier transitions
    promote_naive_threshold: int = 3
    promote_memory_affinity: float = 0.65
    promote_memory_generation: int = 3
    apoptosis_affinity: float = 0.15
    apoptosis_idle_steps: int = 1000
    # Decay
    lambda_decay: float = 0.01
    decay_interval: int = 100
    # RIF (Retrieval-Induced Forgetting)
    rif_suppression_rate: float = 0.1
    rif_reinforcement_rate: float = 0.05
    rif_max_suppression: float = 1.0
    rif_decay_lambda: float = 0.005
    rif_alpha: float = 0.3
    # Experiment
    n_queries: int = 10_000
    hot_set_fraction: float = 0.2
    hot_set_probability: float = 0.7
    random_seed: int = 42

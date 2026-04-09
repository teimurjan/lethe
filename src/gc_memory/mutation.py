from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_sigma(affinity: float, sigma_0: float, gamma: float) -> float:
    """sigma_i = sigma_0 * (1 - affinity_i) ** gamma"""
    return sigma_0 * (1.0 - affinity) ** gamma


def generate_mutants(
    embedding: npt.NDArray[np.float32],
    affinity: float,
    n_mutants: int,
    sigma_0: float,
    gamma: float,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    """Generate n_mutants unit-normalized perturbations of embedding.

    Each mutant: (embedding + epsilon) / ||embedding + epsilon||
    where epsilon ~ Normal(0, sigma^2 * I) and sigma is affinity-adaptive.
    Returns shape (n_mutants, dim).
    """
    sigma = compute_sigma(affinity, sigma_0, gamma)
    dim = embedding.shape[0]
    noise = rng.normal(0.0, sigma, size=(n_mutants, dim)).astype(np.float32)
    mutants = embedding + noise
    norms = np.linalg.norm(mutants, axis=1, keepdims=True)
    mutants = mutants / norms
    return mutants


def select_best_mutant(
    query: npt.NDArray[np.float32],
    original: npt.NDArray[np.float32],
    original_embedding: npt.NDArray[np.float32],
    mutants: npt.NDArray[np.float32],
    delta: float,
    theta_anchor: float,
) -> npt.NDArray[np.float32] | None:
    """Select the best mutant that beats original by > delta and passes anchor check.

    Returns the accepted mutant embedding or None if no mutant qualifies.
    All inputs assumed unit-normalized, so cosine = dot product.
    """
    scores = mutants @ query
    original_score = float(np.dot(original, query))
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score - original_score <= delta:
        return None

    anchor_sim = float(np.dot(mutants[best_idx], original_embedding))
    if anchor_sim < theta_anchor:
        return None

    return mutants[best_idx].copy()

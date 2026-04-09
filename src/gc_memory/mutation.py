from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_sigma(affinity: float, sigma_0: float, gamma: float) -> float:
    """sigma_i = sigma_0 * (1 - affinity_i) ** gamma"""
    return float(sigma_0 * (1.0 - affinity) ** gamma)


def generate_adapter_mutants(
    adapter: npt.NDArray[np.float32],
    base_embedding: npt.NDArray[np.float32],
    query: npt.NDArray[np.float32],
    sigma: float,
    n_mutants: int,
    max_norm: float,
    rng: np.random.Generator,
    toward_query: bool = True,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Generate adapter mutants biased toward (or away from) the query direction.

    Instead of isotropic noise, half of the noise variance is allocated along
    the query direction (or its negative). This produces mutants that tend to
    shift the effective embedding toward/away from the query while still
    exploring other directions.

    Returns (adapter_mutants, effective_mutants) both shape (n_mutants, dim).
    """
    dim = adapter.shape[0]

    # Isotropic component
    noise = rng.normal(0.0, sigma * 0.7, size=(n_mutants, dim)).astype(np.float32)

    # Directional component along query
    direction = query - base_embedding
    dir_norm = np.linalg.norm(direction)
    if dir_norm > 0:
        direction = direction / dir_norm
    dir_noise = rng.normal(0.0, sigma * 0.7, size=(n_mutants, 1)).astype(np.float32)
    if not toward_query:
        dir_noise = -np.abs(dir_noise)
    else:
        dir_noise = np.abs(dir_noise)
    noise += dir_noise * direction

    adapter_mutants = adapter + noise

    # Clip adapter norms
    norms = np.linalg.norm(adapter_mutants, axis=1, keepdims=True)
    mask = norms > max_norm
    adapter_mutants = np.where(mask, adapter_mutants * (max_norm / norms), adapter_mutants)

    # Compute effective embeddings
    eff = base_embedding + adapter_mutants
    eff_norms = np.linalg.norm(eff, axis=1, keepdims=True)
    eff = eff / eff_norms

    return (
        np.asarray(adapter_mutants, dtype=np.float32),
        np.asarray(eff, dtype=np.float32),
    )


def select_best_adapter(
    query: npt.NDArray[np.float32],
    original_effective: npt.NDArray[np.float32],
    adapter_mutants: npt.NDArray[np.float32],
    effective_mutants: npt.NDArray[np.float32],
    delta: float,
) -> int | None:
    """Select the best adapter mutant that improves cosine with query by > delta.

    Returns index of the best mutant, or None if no mutant qualifies.
    """
    scores = effective_mutants @ query
    original_score = float(np.dot(original_effective, query))
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score - original_score <= delta:
        return None

    return best_idx

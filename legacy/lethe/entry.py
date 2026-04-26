from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt


class Tier(Enum):
    NAIVE = "naive"
    GC = "gc"
    MEMORY = "memory"
    APOPTOTIC = "apoptotic"


@dataclass
class MemoryEntry:
    id: str
    content: str
    base_embedding: npt.NDArray[np.float32]
    embedding: npt.NDArray[np.float32]
    adapter: npt.NDArray[np.float32]
    session_id: str = ""
    turn_idx: int = 0
    affinity: float = 0.5
    retrieval_count: int = 0
    generation: int = 0
    last_retrieved_step: int = 0
    tier: Tier = Tier.NAIVE
    suppression: float = 0.0


def effective_embedding(
    base: npt.NDArray[np.float32],
    adapter: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute unit-normalized effective embedding: normalize(base + adapter)."""
    combined = base + adapter
    norm = np.linalg.norm(combined)
    if norm == 0:
        return base.copy()
    return (combined / norm).astype(np.float32)


def create_entry(
    entry_id: str,
    content: str,
    embedding: npt.NDArray[np.float32],
    session_id: str = "",
    turn_idx: int = 0,
) -> MemoryEntry:
    """Create a MemoryEntry with zero adapter. Effective embedding = base embedding."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError(f"Zero-norm embedding for entry {entry_id}")
    base = (embedding / norm).astype(np.float32)
    dim = base.shape[0]
    return MemoryEntry(
        id=entry_id,
        content=content,
        base_embedding=base.copy(),
        embedding=base.copy(),
        adapter=np.zeros(dim, dtype=np.float32),
        session_id=session_id,
        turn_idx=turn_idx,
    )

"""Text → dense embedding → binary address.

Two components:
- TextEncoder: wraps a sentence-transformer model (default MiniLM-L6-v2, 384-d).
- HyperplaneProjector: maps dense vectors to binary addresses via random
  Gaussian hyperplanes (the standard LSH-for-cosine construction).

Kept deliberately separate so SDM doesn't depend on any specific encoder.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # macOS FAISS/PyTorch OMP fix

import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]


class TextEncoder:
    """Thin wrapper around a sentence-transformer producing normalized embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return (n, dim) float32 array of L2-normalized embeddings."""
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)


class HyperplaneProjector:
    """Project dense vectors to binary addresses via random Gaussian hyperplanes.

    For a vector v and hyperplane normal w, bit = sign(w·v). With enough bits
    this preserves cosine similarity as Hamming similarity:
    P(bit_matches) = 1 - theta / pi, where theta is the angle between vectors.
    """

    def __init__(self, input_dim: int, address_bits: int = 512, seed: int = 42) -> None:
        self.input_dim = input_dim
        self.address_bits = address_bits
        rng = np.random.default_rng(seed)
        # Gaussian-distributed normals → uniform over directions on the sphere.
        self.W: npt.NDArray[np.float32] = rng.standard_normal(
            (input_dim, address_bits), dtype=np.float32,
        )

    def project(self, dense: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """Project dense (n, input_dim) or (input_dim,) to binary (n, address_bits)."""
        if dense.ndim == 1:
            dense = dense[None, :]
        projected = dense @ self.W
        return (projected > 0).astype(bool)

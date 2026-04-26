"""ONNX-backed encoder adapters for the lethe CLI + plugin.

``MemoryStore`` accepts any object implementing:

    bi_encoder.encode(text_or_list, normalize_embeddings=True) -> np.ndarray
    cross_encoder.predict(pairs) -> np.ndarray

These two classes wrap `fastembed` with that interface so the torch +
sentence-transformers dependency can stay in the ``[benchmarks]`` extras
where published-number reproducibility needs it. For interactive use
(CLI, Claude Code hooks), ONNX cuts cold-start from ~18s to ~1.5s.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


class OnnxBiEncoder:
    """SentenceTransformer-compatible wrapper around ``fastembed.TextEmbedding``.

    Matches the subset of the SentenceTransformer API that MemoryStore uses:

    - ``.encode(text)`` for a single string → 1-D float32 np.ndarray
    - ``.encode([t1, t2, …])`` for a batch  → 2-D float32 np.ndarray
    - ``.get_embedding_dimension()`` → int

    The ``normalize_embeddings`` kwarg is accepted for API parity;
    fastembed already returns L2-normalized vectors for this model family.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self._dim: int | None = None

    def encode(
        self,
        text: str | Iterable[str],
        normalize_embeddings: bool = True,  # noqa: ARG002 — API compat
        **_: object,
    ) -> np.ndarray:
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        vectors = np.array(list(self._model.embed(texts)), dtype=np.float32)
        if self._dim is None and vectors.size:
            self._dim = int(vectors.shape[-1])
        return vectors[0] if single else vectors

    def get_embedding_dimension(self) -> int:
        if self._dim is None:
            _ = self.encode("probe")
        assert self._dim is not None
        return self._dim

    def get_sentence_embedding_dimension(self) -> int:  # sentence-transformers <3 alias
        return self.get_embedding_dimension()


class OnnxCrossEncoder:
    """CrossEncoder-compatible wrapper around ``fastembed.TextCrossEncoder``.

    ``predict(pairs)`` where ``pairs`` is a sequence of ``(query, passage)``
    tuples, returns a 1-D float32 np.ndarray of relevance scores.
    """

    DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name)

    def predict(self, pairs: Iterable[tuple[str, str]]) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        # fastembed's rerank API takes a single query + list of documents;
        # group consecutive pairs sharing the same query to batch efficiently.
        out = np.empty(len(pairs), dtype=np.float32)
        i = 0
        while i < len(pairs):
            query = pairs[i][0]
            j = i
            while j < len(pairs) and pairs[j][0] == query:
                j += 1
            docs = [p[1] for p in pairs[i:j]]
            scores = list(self._model.rerank(query, docs))
            out[i:j] = np.asarray(scores, dtype=np.float32)
            i = j
        return out


# Resolving the configured sentence-transformers model name against the
# fastembed catalog: a few common names need mapping since fastembed uses
# the upstream Xenova/HF id rather than the bare `cross-encoder/...` form.
_MODEL_ALIASES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "Xenova/ms-marco-MiniLM-L-6-v2",
}


def resolve_bi_encoder_name(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)


def resolve_cross_encoder_name(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)

"""Cross-project read-only retrieval via DuckDB ATTACH.

``UnionStore`` opens one in-memory DuckDB connection, ATTACHes each
registered project's ``lethe.duckdb`` as a read-only schema, and combines
per-project ``VectorIndex`` (FAISS + BM25) results via RRF before
cross-encoder reranking.

Design notes:
  - Read-only. RIF state is not updated; tier transitions don't fire.
    Mutating RIF across projects would force opening each file for write,
    which collides with concurrent hook activity. Local ``lethe search``
    continues to mutate per-project state.
  - BM25 uses per-project fan-out + RRF merge, NOT global corpus concat.
    Concat would shift per-corpus IDF and drift the benchmark numbers
    in BENCHMARKS.md. RRF is rank-based and IDF-invariant.
  - Missing / stale FAISS is tolerated per-project (BM25-only fallback).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from lethe._registry import slugify
from lethe.db import _rows_as_dicts
from lethe.reranker import Reranker
from lethe.rif import RIFConfig
from lethe.vectors import VectorIndex


@dataclass(frozen=True)
class UnionHit:
    project_slug: str
    project_root: Path
    id: str
    content: str
    score: float


@dataclass
class _ProjectHandle:
    slug: str
    root: Path
    duckdb_path: Path
    vector_index: VectorIndex
    ids: list[str]
    suppression: dict[str, float]
    content_map: dict[str, str]


class UnionStore:
    def __init__(
        self,
        roots: list[Path],
        *,
        bi_encoder: Any,
        cross_encoder: Any,
        dim: int = 384,
        k_shallow: int = 30,
        k_deep: int = 100,
        confidence_threshold: float = 4.0,
        rif_config: RIFConfig | None = None,
    ) -> None:
        self.roots = [Path(r).resolve() for r in roots]
        self.bi_encoder = bi_encoder
        self.reranker = Reranker(cross_encoder, confidence_threshold)
        self.rif = rif_config or RIFConfig()
        self.dim = dim
        self.k_shallow = k_shallow
        self.k_deep = k_deep

        self._conn = duckdb.connect(":memory:")
        self._projects: list[_ProjectHandle] = []
        for root in self.roots:
            duckdb_path = root / ".lethe" / "index" / "lethe.duckdb"
            if not duckdb_path.exists():
                continue
            try:
                handle = self._open_project(root, duckdb_path)
            except Exception as exc:  # noqa: BLE001 — tolerate one bad project
                print(f"[lethe] skipping {root}: {exc}", file=sys.stderr)
                continue
            self._projects.append(handle)

    def _open_project(self, root: Path, duckdb_path: Path) -> _ProjectHandle:
        slug = slugify(root)
        # Read-only ATTACH avoids clashing with per-project Stop hooks that
        # may currently hold a write handle.
        self._conn.execute(f"ATTACH '{duckdb_path}' AS {slug} (READ_ONLY)")

        cur = self._conn.execute(
            f"SELECT id, content, suppression FROM {slug}.entries "
            "WHERE tier != 'apoptotic'"
        )
        rows = _rows_as_dicts(cur)
        ids = [str(r["id"]) for r in rows]
        contents = [str(r["content"]) for r in rows]
        suppression = {str(r["id"]): float(r["suppression"] or 0.0) for r in rows}
        content_map = dict(zip(ids, contents))

        vi = VectorIndex(dim=self.dim)
        try:
            vi.load(root / ".lethe" / "index", ids, contents)
        except Exception as exc:  # noqa: BLE001 — stale/missing FAISS is non-fatal
            print(f"[lethe] faiss unavailable for {root}: {exc}", file=sys.stderr)
            # Still build the BM25 index from content even without FAISS.
            vi.build(ids, np.zeros((0, self.dim), dtype=np.float32), contents)

        return _ProjectHandle(
            slug=slug,
            root=root,
            duckdb_path=duckdb_path,
            vector_index=vi,
            ids=ids,
            suppression=suppression,
            content_map=content_map,
        )

    # -------- Public API --------

    def retrieve(self, query: str, k: int = 10) -> list[UnionHit]:
        if not self._projects:
            return []

        q_emb = self.bi_encoder.encode(
            query, normalize_embeddings=True
        ).astype(np.float32)

        candidates = self._gather_candidates(q_emb, query, self.k_shallow)
        hits = self._rerank_and_materialize(query, candidates, k)

        if hits and max(h.score for h in hits) < self.reranker.confidence_threshold:
            # Adaptive deep pass — widen the per-project candidate pool.
            deep = self._gather_candidates(q_emb, query, self.k_deep)
            hits = self._rerank_and_materialize(query, deep, k)
        return hits

    def stats(self) -> dict[str, Any]:
        return {
            "projects": [
                {"slug": p.slug, "root": str(p.root), "entries": len(p.ids)}
                for p in self._projects
            ],
            "total_entries": sum(len(p.ids) for p in self._projects),
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    # -------- Internals --------

    def _gather_candidates(
        self, q_emb: np.ndarray, query: str, per_project_k: int,
    ) -> list[tuple[str, str, str, float]]:
        """Return (slug, eid, content, merged_rrf_score) candidates across projects."""
        merged: dict[tuple[str, str], float] = {}

        for proj in self._projects:
            scored = proj.vector_index.search_hybrid_scored(q_emb, query, per_project_k)
            # Apply per-project suppression to each entry's RRF score.
            for rank, (eid, rrf_score) in enumerate(scored[:per_project_k]):
                penalty = self.rif.alpha * proj.suppression.get(eid, 0.0)
                # Nested RRF: contribution is 1/(rank+60), not raw score — keeps
                # projects with different corpus sizes on the same footing.
                contrib = 1.0 / (rank + 60) - penalty / max(len(self._projects), 1)
                key = (proj.slug, eid)
                merged[key] = merged.get(key, 0.0) + contrib

        top = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[: per_project_k]
        out: list[tuple[str, str, str, float]] = []
        for (slug, eid), score in top:
            proj = self._slug_to_project[slug]
            content = proj.content_map.get(eid)
            if content is None:
                continue
            out.append((slug, eid, content, score))
        return out

    def _rerank_and_materialize(
        self,
        query: str,
        candidates: list[tuple[str, str, str, float]],
        k: int,
    ) -> list[UnionHit]:
        if not candidates:
            return []
        # Cross-encoder rerank. Use a positional key so (slug, eid) collisions
        # across projects don't merge — two identical chunks in different
        # projects are legitimately two hits.
        pairs = [(query, content) for _, _, content, _ in candidates]
        if self.reranker.cross_encoder is not None:
            raw_scores = self.reranker.cross_encoder.predict(pairs)
            xenc_scores = [float(s) for s in raw_scores]
        else:
            xenc_scores = [score for *_, score in candidates]

        order = sorted(range(len(candidates)), key=lambda i: xenc_scores[i], reverse=True)
        hits: list[UnionHit] = []
        for i in order[:k]:
            slug, eid, content, _ = candidates[i]
            proj = self._slug_to_project[slug]
            hits.append(UnionHit(
                project_slug=slug,
                project_root=proj.root,
                id=eid,
                content=content,
                score=xenc_scores[i],
            ))
        return hits

    @property
    def _slug_to_project(self) -> dict[str, _ProjectHandle]:
        return {p.slug: p for p in self._projects}

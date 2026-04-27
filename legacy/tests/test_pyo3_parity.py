"""PyO3 behavioral parity: `lethe.MemoryStore` ↔ `lethe_memory.MemoryStore`.

Layer 4 of the Rust-port migration confidence ladder. Verifies that
both implementations produce equivalent *observable* behavior on the
public API (`add`, `delete`, `size`, `retrieve`, save/reopen):

- exact-content dedup decisions match
- size/lifecycle counts match
- persistence round-trips correctly
- top-K retrieval overlaps strongly on the same corpus

Numerical drift between fastembed (Python) and `ort` (Rust) means the
top-K *ids* may not be bit-identical; we assert ≥ 0.6 Jaccard overlap
on top-5 for a deterministic 30-entry corpus, which is the same
empirical floor used in the LongMemEval Layer 1 bench.

Skips automatically if `lethe_memory` is not installed
(`uv run maturin develop --release -m crates/lethe-py/Cargo.toml`).
"""
from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

lethe_memory = pytest.importorskip("lethe_memory")

from lethe.encoders import OnnxBiEncoder, OnnxCrossEncoder
from lethe.memory_store import MemoryStore as PyStore

BI = "sentence-transformers/all-MiniLM-L6-v2"
CROSS = "Xenova/ms-marco-MiniLM-L-6-v2"

CORPUS = [
    "Python is a high-level interpreted programming language.",
    "Rust is a systems programming language focused on safety.",
    "JavaScript runs in browsers and on Node.js servers.",
    "Go is a compiled language designed at Google for concurrency.",
    "Haskell is a purely functional language with lazy evaluation.",
    "Postgres is a relational database with strong ACID guarantees.",
    "Redis is an in-memory key-value store often used for caching.",
    "MongoDB is a document database storing JSON-like records.",
    "DuckDB is an embedded analytical database optimized for OLAP.",
    "SQLite is a small embedded relational database in a single file.",
    "FAISS is a library for efficient similarity search of dense vectors.",
    "BM25 is a probabilistic ranking function used by search engines.",
    "Sentence transformers produce dense embeddings for semantic search.",
    "Cross-encoders rerank candidate documents by joint query-doc scoring.",
    "RRF combines rankings from multiple retrievers into a single ordering.",
    "TF-IDF weights terms by frequency and inverse document frequency.",
    "Cosine similarity measures the angle between two vectors.",
    "Euclidean distance is the straight-line distance in vector space.",
    "Tokenization splits a string into atomic units before embedding.",
    "Stemming reduces words to a common root for keyword matching.",
    "Cats are small carnivorous mammals often kept as house pets.",
    "Dogs are loyal canine companions descended from wolves.",
    "Coffee contains caffeine and is brewed from roasted beans.",
    "Tea is an aromatic beverage prepared by steeping leaves in hot water.",
    "The Pacific Ocean is the largest body of water on Earth.",
    "Mount Everest is the highest peak above sea level on Earth.",
    "The Amazon rainforest spans nine countries in South America.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Mitochondria are the energy-producing organelles inside cells.",
    "DNA encodes the genetic instructions of living organisms.",
]

QUERIES = [
    ("Tell me about Rust as a programming language", {1}),
    ("Which database is good for analytics?", {8}),
    ("How does similarity search work?", {10, 13}),
    ("What pets do humans keep?", {20, 21}),
    ("Tell me about mountains", {25}),
]


@pytest.fixture(scope="module")
def encoders() -> tuple[OnnxBiEncoder, OnnxCrossEncoder]:
    return OnnxBiEncoder(BI), OnnxCrossEncoder(CROSS)


@pytest.fixture
def py_store(encoders, tmp_path: Path) -> Iterator[PyStore]:
    bi, xenc = encoders
    store = PyStore(
        tmp_path / "py", bi_encoder=bi, cross_encoder=xenc, dim=bi.get_embedding_dimension()
    )
    yield store
    store.save()


@pytest.fixture
def rs_store(tmp_path: Path) -> Iterator[lethe_memory.MemoryStore]:
    store = lethe_memory.MemoryStore(
        str(tmp_path / "rs"), bi_encoder=BI, cross_encoder=CROSS
    )
    yield store
    store.save()


def _seed(store, ids_returned: list[str]) -> None:
    for content in CORPUS:
        eid = store.add(content)
        assert eid is not None, f"first add of {content!r} should not dedup"
        ids_returned.append(eid)


def _retrieve_ids(store, query: str, k: int) -> list[str]:
    if isinstance(store, PyStore):
        return [eid for eid, _, _ in store.retrieve(query, k=k)]
    # lethe_memory returns Hit objects.
    return [h.id for h in store.retrieve(query, k=k)]


# --------------------------------------------------------------- 1. add

def test_add_returns_id(py_store, rs_store):
    py_id = py_store.add("Lethe is a memory store for LLM agents.")
    rs_id = rs_store.add("Lethe is a memory store for LLM agents.")
    assert isinstance(py_id, str) and len(py_id) > 0
    assert isinstance(rs_id, str) and len(rs_id) > 0


def test_add_dedups_exact_content(py_store, rs_store):
    content = "BM25 is a probabilistic ranking function."
    assert py_store.add(content) is not None
    assert py_store.add(content) is None, "Python should dedup on exact content"
    assert rs_store.add(content) is not None
    assert rs_store.add(content) is None, "Rust should dedup on exact content"


def test_add_dedups_after_reopen(encoders, tmp_path: Path):
    bi, xenc = encoders
    py_path = tmp_path / "py"
    rs_path = tmp_path / "rs"
    content = "Persistence test — DuckDB hash table preserves dedup state."

    s_py = PyStore(py_path, bi_encoder=bi, cross_encoder=xenc, dim=bi.get_embedding_dimension())
    s_rs = lethe_memory.MemoryStore(str(rs_path), bi_encoder=BI, cross_encoder=CROSS)
    assert s_py.add(content) is not None
    assert s_rs.add(content) is not None
    s_py.save()
    s_rs.save()
    del s_py, s_rs

    s_py2 = PyStore(py_path, bi_encoder=bi, cross_encoder=xenc, dim=bi.get_embedding_dimension())
    s_rs2 = lethe_memory.MemoryStore(str(rs_path), bi_encoder=BI, cross_encoder=CROSS)
    assert s_py2.add(content) is None, "Python should still dedup after reopen"
    assert s_rs2.add(content) is None, "Rust should still dedup after reopen"


# --------------------------------------------------------------- 2. size + delete

def test_size_tracks_adds_and_deletes(py_store, rs_store):
    py_ids: list[str] = []
    rs_ids: list[str] = []
    _seed(py_store, py_ids)
    _seed(rs_store, rs_ids)
    assert len(py_store.entries) == len(CORPUS)
    assert rs_store.size() == len(CORPUS)

    # Delete one — both should return True and decrement.
    assert py_store.delete(py_ids[0]) is True
    assert rs_store.delete(rs_ids[0]) is True
    assert len(py_store.entries) == len(CORPUS) - 1
    assert rs_store.size() == len(CORPUS) - 1

    # Unknown id — both return False.
    assert py_store.delete("does-not-exist") is False
    assert rs_store.delete("does-not-exist") is False


# --------------------------------------------------------------- 3. retrieve

def test_retrieve_returns_at_most_k(py_store, rs_store):
    _seed(py_store, [])
    _seed(rs_store, [])
    for k in (1, 3, 5, 10):
        assert len(py_store.retrieve("programming", k=k)) <= k
        assert len(rs_store.retrieve("programming", k=k)) <= k


def test_retrieve_finds_obvious_match(py_store, rs_store):
    _seed(py_store, [])
    _seed(rs_store, [])

    py_top = _retrieve_ids(py_store, "Rust programming language", 3)
    rs_top = _retrieve_ids(rs_store, "Rust programming language", 3)

    # Both should put the Rust corpus entry in the top 3. We can't compare
    # ids across impls (uuid's differ); we compare content via a bridge.
    py_top_content = {py_store.entries[eid].content for eid in py_top if eid in py_store.entries}
    rs_top_content = {h.content for h in rs_store.retrieve("Rust programming language", k=3)}
    assert any("Rust" in c for c in py_top_content)
    assert any("Rust" in c for c in rs_top_content)


def test_retrieve_top_k_content_overlap(py_store, rs_store):
    """Top-5 result *content* sets should overlap heavily across impls.

    IDs differ (each impl mints its own uuids), so we compare on content.
    The 0.6 Jaccard floor is the same empirical bar the LongMemEval bench
    accepts on f32 numerical drift between fastembed and `ort`.
    """
    _seed(py_store, [])
    _seed(rs_store, [])

    overlaps: list[float] = []
    for query, _gold in QUERIES:
        py_top = _retrieve_ids(py_store, query, 5)
        py_content = {py_store.entries[eid].content for eid in py_top if eid in py_store.entries}
        rs_content = {h.content for h in rs_store.retrieve(query, k=5)}
        if not py_content or not rs_content:
            continue
        overlap = len(py_content & rs_content) / len(py_content | rs_content)
        overlaps.append(overlap)

    assert overlaps, "No queries produced results from either impl"
    min_overlap = min(overlaps)
    avg_overlap = sum(overlaps) / len(overlaps)
    assert min_overlap >= 0.6, (
        f"Min top-5 content Jaccard {min_overlap:.2f} below 0.6 floor "
        f"(avg {avg_overlap:.2f}); investigate retrieval pipeline drift"
    )


# --------------------------------------------------------------- 4. persistence

def test_persistence_roundtrip(encoders, tmp_path: Path):
    bi, xenc = encoders
    py_path = tmp_path / "py"
    rs_path = tmp_path / "rs"

    s_py = PyStore(py_path, bi_encoder=bi, cross_encoder=xenc, dim=bi.get_embedding_dimension())
    s_rs = lethe_memory.MemoryStore(str(rs_path), bi_encoder=BI, cross_encoder=CROSS)
    for content in CORPUS[:10]:
        s_py.add(content)
        s_rs.add(content)
    s_py.save()
    s_rs.save()
    del s_py, s_rs

    s_py2 = PyStore(py_path, bi_encoder=bi, cross_encoder=xenc, dim=bi.get_embedding_dimension())
    s_rs2 = lethe_memory.MemoryStore(str(rs_path), bi_encoder=BI, cross_encoder=CROSS)
    assert len(s_py2.entries) == 10
    assert s_rs2.size() == 10

    # Retrieve still works and the content is recoverable.
    py_hits = s_py2.retrieve("programming language", k=3)
    rs_hits = s_rs2.retrieve("programming language", k=3)
    assert len(py_hits) > 0
    assert len(rs_hits) > 0

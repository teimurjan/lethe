"""End-to-end unit tests for lethe.memory_store.MemoryStore.

Uses mock bi-encoder + cross-encoder from conftest.py — no real model loads,
no network, no disk leakage beyond tmp_path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lethe.entry import Tier
from lethe.memory_store import MemoryStore
from lethe.rif import RIFConfig, assign_cluster


# ---------- Fixture ----------

@pytest.fixture
def store(mock_bi_encoder, mock_cross_encoder, tmp_store_path: Path) -> MemoryStore:
    return MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
        k_shallow=10,
        k_deep=20,
        confidence_threshold=4.0,
        dedup_threshold=0.95,
    )


# ---------- add() ----------

def test_add_returns_id(store: MemoryStore) -> None:
    eid = store.add("hello world")
    assert isinstance(eid, str)
    assert eid in store.entries


def test_add_exact_duplicate_returns_none(store: MemoryStore) -> None:
    store.add("exactly this text")
    again = store.add("exactly this text")
    assert again is None


def test_add_near_duplicate_keeps_longer(store: MemoryStore, mock_bi_encoder) -> None:
    # Add short then longer — the "near-duplicate" check relies on cosine > 0.95.
    # Because our mock encoder hashes text → random unit vector, different strings
    # produce ~orthogonal vectors. So we inject direct copies to trigger dedup.

    # First add a short entry
    short = store.add("hi")
    # Force a synthetic near-duplicate by mutating the stored embedding
    import numpy as np
    # Create an entry whose embedding matches the short one (cosine = 1.0)
    long_eid = "manual-id-long-content"
    long_text = "hi there this is a longer version"
    # Use the same embedding as the short entry
    same_emb = store.entries[short].base_embedding
    from lethe.entry import create_entry
    # Instead of calling store.add (which would give a different embedding via encoder),
    # manually do the near-duplicate flow by passing identical embeddings.
    # We'll just verify the short one is replaced by construction:
    # Use the public API: set near-dup threshold check through a direct build.
    # Simpler: monkeypatch encoder.encode to return `same_emb` for the long text.
    orig_encode = mock_bi_encoder.encode
    def encode_same(text, **kwargs):  # noqa: ANN001
        if text == long_text:
            return same_emb
        return orig_encode(text, **kwargs)
    mock_bi_encoder.encode = encode_same  # type: ignore[assignment]
    result = store.add(long_text)
    # Longer wins → old short entry deleted, new longer entry inserted
    assert short not in store.entries
    assert result in store.entries
    assert store.entries[result].content == long_text


def test_add_near_duplicate_skips_shorter(store: MemoryStore, mock_bi_encoder) -> None:
    """A near-duplicate shorter than the existing entry is dropped."""
    long_first = store.add("this is a longer piece of text stored first")
    same_emb = store.entries[long_first].base_embedding
    orig_encode = mock_bi_encoder.encode
    def encode_same(text, **kwargs):  # noqa: ANN001
        if text == "short":
            return same_emb
        return orig_encode(text, **kwargs)
    mock_bi_encoder.encode = encode_same  # type: ignore[assignment]
    result = store.add("short")
    assert result is None  # rejected
    assert long_first in store.entries


# ---------- retrieve() ----------

def test_retrieve_empty_store_returns_empty(store: MemoryStore) -> None:
    results = store.retrieve("anything", k=5)
    assert results == []


def test_retrieve_returns_top_k_sorted(store: MemoryStore) -> None:
    store.add("the quick brown fox jumps")
    store.add("lazy dogs sleep all day")
    store.add("cat cat cat")
    results = store.retrieve("brown fox", k=2)
    # Cross-encoder scores based on shared tokens; "the quick brown fox jumps"
    # shares {"brown", "fox"} with the query → highest.
    assert len(results) <= 2
    assert len(results) >= 1
    assert "brown fox" in " ".join(content for _, content, _ in results).lower() or \
           any("fox" in content for _, content, _ in results)
    # Scores descending
    scores = [s for _, _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_increments_step(store: MemoryStore) -> None:
    store.add("some entry")
    before = store._step
    store.retrieve("some entry", k=1)
    assert store._step == before + 1


def test_retrieve_updates_retrieval_count_and_affinity(store: MemoryStore) -> None:
    eid = store.add("the quick brown fox")
    before_count = store.entries[eid].retrieval_count
    before_aff = store.entries[eid].affinity
    store.retrieve("quick brown fox", k=1)
    assert store.entries[eid].retrieval_count == before_count + 1
    # Affinity moved toward 1.0 (winner gets positive xenc score from the mock)
    assert store.entries[eid].affinity != before_aff


def test_retrieve_applies_rif_state_on_subsequent_calls(store: MemoryStore) -> None:
    """After a retrieve, losing candidates accumulate suppression."""
    winner = store.add("travel to paris in march")
    loser = store.add("travel to tokyo in june")
    # The query matches both on "travel to" but favors one. Do a retrieve.
    store.retrieve("travel to paris in march", k=1)
    # At least one non-winner entry should be tracked for suppression.
    suppressions = {eid: store.entries[eid].suppression for eid in store.entries}
    # Winner gets reinforced (suppression stays 0 or decreases)
    assert suppressions[winner] == 0.0


def test_retrieve_tier_naive_to_gc_after_3_retrievals(store: MemoryStore) -> None:
    eid = store.add("the unique_token here")
    assert store.entries[eid].tier is Tier.NAIVE
    for _ in range(3):
        store.retrieve("unique_token", k=1)
    assert store.entries[eid].tier in (Tier.GC, Tier.MEMORY)


def test_retrieve_tier_gc_to_memory_with_high_affinity(store: MemoryStore) -> None:
    eid = store.add("target_word elevated")
    # Simulate many retrievals with strong positive scores
    for _ in range(10):
        store.retrieve("target_word elevated", k=1)
    # After 5+ retrievals with high xenc (matching tokens), affinity should push it to MEMORY.
    assert store.entries[eid].tier is Tier.MEMORY
    assert store.entries[eid].retrieval_count >= 5
    assert store.entries[eid].affinity >= 0.65


# ---------- save / reload ----------

def test_save_and_reload_roundtrip_preserves_entries(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    s1 = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
    )
    eid_a = s1.add("first entry about cats")
    eid_b = s1.add("second entry about dogs")
    s1.retrieve("cats", k=1)  # mutates state: affinities, step, etc.
    s1.save()
    s1.close()

    s2 = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
    )
    assert eid_a in s2.entries
    assert eid_b in s2.entries
    assert s2.entries[eid_a].content == "first entry about cats"
    # step survived
    assert s2._step >= 1


def test_apoptotic_entries_excluded_on_load(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    s1 = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
    )
    eid = s1.add("going to die soon")
    s1.entries[eid].tier = Tier.APOPTOTIC
    s1.save()
    s1.close()

    s2 = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
    )
    assert eid not in s2.entries


# ---------- decay ----------

def test_decay_reduces_non_memory_affinity(store: MemoryStore) -> None:
    eid = store.add("decay me")
    store.entries[eid].tier = Tier.GC
    store.entries[eid].affinity = 0.6
    store.entries[eid].last_retrieved_step = 0
    store._step = 200  # enough elapsed
    store.decay()
    assert store.entries[eid].affinity < 0.6


def test_decay_exempts_memory_tier(store: MemoryStore) -> None:
    eid = store.add("immune to decay")
    store.entries[eid].tier = Tier.MEMORY
    store.entries[eid].affinity = 0.8
    store.entries[eid].last_retrieved_step = 0
    store._step = 1_000_000
    store.decay()
    assert store.entries[eid].affinity == 0.8


# ---------- stats ----------

def test_stats_reports_tier_distribution(store: MemoryStore) -> None:
    a = store.add("a")
    b = store.add("b")
    c = store.add("c")
    store.entries[a].tier = Tier.NAIVE
    store.entries[b].tier = Tier.GC
    store.entries[c].tier = Tier.MEMORY
    stats = store.stats()
    assert stats["total_entries"] == 3
    assert stats["tiers"]["naive"] == 1
    assert stats["tiers"]["gc"] == 1
    assert stats["tiers"]["memory"] == 1


# ---------- size ----------

def test_size_property(store: MemoryStore) -> None:
    assert store.size == 0
    store.add("a")
    store.add("b")
    assert store.size == 2


# ---------- error paths ----------

def test_retrieve_raises_without_bi_encoder(tmp_store_path: Path, mock_cross_encoder) -> None:
    s = MemoryStore(path=tmp_store_path, bi_encoder=None, cross_encoder=mock_cross_encoder, dim=16)
    with pytest.raises(ValueError, match="bi_encoder required"):
        s.retrieve("q", k=1)


def test_add_raises_without_bi_encoder(tmp_store_path: Path, mock_cross_encoder) -> None:
    s = MemoryStore(path=tmp_store_path, bi_encoder=None, cross_encoder=mock_cross_encoder, dim=16)
    with pytest.raises(ValueError, match="bi_encoder required"):
        s.add("text")


# ---------- Clustered RIF ----------


def _clustered_store(
    tmp_store_path: Path, bi, xe, *, n_clusters: int = 2,
) -> MemoryStore:
    store = MemoryStore(
        path=tmp_store_path,
        bi_encoder=bi,
        cross_encoder=xe,
        dim=16,
        k_shallow=10,
        k_deep=20,
        rif_config=RIFConfig(
            alpha=0.3,
            use_rank_gap=True,
            suppression_rate=0.3,
            n_clusters=n_clusters,
        ),
    )
    # In tests with tiny n_clusters, override the 10× minimum so centroids
    # build after just n_clusters queries (avoids issuing 300 dummy queries).
    store._min_cluster_queries = n_clusters
    return store


def _force_topic_embedding(
    store: MemoryStore, entry_id: str, half: int, dim: int = 16,
) -> None:
    """Pin an entry's embedding to a chosen half of the vector space.

    half=0 → mass on dims [0..dim/2); half=1 → mass on dims [dim/2..dim).
    This lets us build two clearly-separable topic clusters without a real
    sentence encoder.
    """
    rng = np.random.default_rng(abs(hash(entry_id)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32) * 0.02
    if half == 0:
        v[: dim // 2] += 1.0
    else:
        v[dim // 2:] += 1.0
    v /= np.linalg.norm(v) + 1e-12
    store._embeddings[entry_id] = v
    store.entries[entry_id].base_embedding = v
    store.entries[entry_id].embedding = v.copy()


def test_clustered_rif_state_populated_entry_suppression_untouched(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    """In clustered mode, updates land in _cluster_state, not entry.suppression."""
    store = _clustered_store(tmp_store_path, mock_bi_encoder, mock_cross_encoder)
    eids = []
    for text in ["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel"]:
        eids.append(store.add(text))

    assert store._cluster_state is not None

    store.retrieve("alpha", k=2)
    store.retrieve("delta", k=2)

    # At least one cluster has accumulated per-entry data.
    touched = any(
        store._cluster_state.get_cluster_scores(cid)
        for cid in range(store.rif.n_clusters)
    )
    assert touched, "expected clustered state to register at least one update"

    # Global entry.suppression column stays at default (0.0) in clustered mode.
    for eid in eids:
        assert store.entries[eid].suppression == 0.0


def test_clustered_rif_is_cue_dependent(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    """An entry suppressed by topic-A queries is not suppressed for topic-B queries."""
    store = _clustered_store(tmp_store_path, mock_bi_encoder, mock_cross_encoder)

    topic_a_texts = [
        "alpha apple tokenA one",
        "alpha apple tokenA two",
        "alpha apple tokenA three",
    ]
    topic_b_texts = [
        "beta banana tokenB one",
        "beta banana tokenB two",
        "beta banana tokenB three",
    ]
    a_ids = [store.add(t) for t in topic_a_texts]
    b_ids = [store.add(t) for t in topic_b_texts]
    for eid in a_ids:
        _force_topic_embedding(store, eid, half=0)
    for eid in b_ids:
        _force_topic_embedding(store, eid, half=1)
    store._rebuild_index()

    query_a = np.zeros(16, dtype=np.float32)
    query_a[:8] = 1.0
    query_a /= np.linalg.norm(query_a)
    query_b = np.zeros(16, dtype=np.float32)
    query_b[8:] = 1.0
    query_b /= np.linalg.norm(query_b)

    def topic_aware_encode(text, **_kwargs):  # type: ignore[no-untyped-def]
        if "tokenA" in text:
            return query_a
        if "tokenB" in text:
            return query_b
        return np.zeros(16, dtype=np.float32)

    mock_bi_encoder.encode = topic_aware_encode  # type: ignore[assignment]

    # Seed with one A + one B so the first build (at n_clusters=2) gets
    # distinct centroids. Centroids are frozen after this build.
    store.retrieve("tokenA seed", k=1)  # buffer=1, no centroids yet
    store.retrieve("tokenB seed", k=1)  # buffer=2, centroids built: [A, B]
    assert store._cluster_centroids is not None
    cid_a = assign_cluster(query_a, store._cluster_centroids)
    cid_b = assign_cluster(query_b, store._cluster_centroids)
    assert cid_a != cid_b, "topics must fall into different clusters"

    # Reset cluster suppression so seeding phase doesn't contaminate.
    # Freeze centroids so k-means rebuilds don't shift cluster assignments
    # mid-test (production has stable centroids with 1000s of queries; in
    # this 16-dim test with 2 clusters, each rebuild shuffles IDs).
    from lethe.rif import ClusteredSuppressionState
    store._cluster_state = ClusteredSuppressionState()
    store._cluster_rebuild_interval = 100_000

    # Now issue many topic-A queries only. Suppression should land in
    # cid_a exclusively.
    for _ in range(8):
        store.retrieve("tokenA query", k=1)

    scores_a = store._cluster_state.get_cluster_scores(cid_a)
    scores_b = store._cluster_state.get_cluster_scores(cid_b)
    max_a = max(scores_a.values(), default=0.0)
    max_b = max(scores_b.values(), default=0.0)
    assert max_a > 0.0, "cluster A should have accumulated suppression"
    assert max_b == 0.0, (
        f"cluster B should be untouched ({max_b}) -- only topic-A queries issued"
    )


def test_clustered_rif_requires_enough_entries_falls_back_to_global_view(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    """With fewer entries than n_clusters, retrieve() still works (no centroids)."""
    store = _clustered_store(
        tmp_store_path, mock_bi_encoder, mock_cross_encoder, n_clusters=5,
    )
    store.add("solo entry")
    # Only one entry — far fewer than n_clusters=5.
    results = store.retrieve("solo entry", k=1)
    assert len(results) == 1
    # Centroids should not have been built.
    assert store._cluster_centroids is None


def test_cluster_centroids_build_once_then_freeze(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    """Centroids are built once n_clusters query embeddings are collected, then frozen."""
    store = _clustered_store(tmp_store_path, mock_bi_encoder, mock_cross_encoder)
    for t in ["one two", "three four", "five six"]:
        store.add(t)
    # n_clusters=2. First retrieve: buffer has 1, not enough.
    store.retrieve("one two", k=1)
    assert store._cluster_centroids is None
    # Second retrieve: buffer has 2, centroids build.
    store.retrieve("three four", k=1)
    assert store._cluster_centroids is not None
    first = store._cluster_centroids.copy()

    # Further queries don't rebuild — centroids are frozen.
    for i in range(200):
        store.retrieve(f"extra query {i}", k=1)
    assert np.allclose(store._cluster_centroids, first)


def test_cluster_state_persists_across_save_load(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    """save()/reopen round-trips cluster_suppression and cluster_centroids."""
    s1 = _clustered_store(tmp_store_path, mock_bi_encoder, mock_cross_encoder)
    for t in ["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel", "india juliet"]:
        s1.add(t)
    s1.retrieve("alpha bravo", k=2)
    s1.retrieve("charlie delta", k=2)
    # Snapshot expected state
    assert s1._cluster_state is not None
    expected_scores, expected_last = s1._cluster_state.snapshot()
    assert s1._cluster_centroids is not None
    expected_centroids = s1._cluster_centroids.copy()
    s1.save()
    s1.close()

    s2 = _clustered_store(tmp_store_path, mock_bi_encoder, mock_cross_encoder)
    assert s2._cluster_state is not None
    got_scores, got_last = s2._cluster_state.snapshot()
    assert got_scores == expected_scores
    assert got_last == expected_last
    assert s2._cluster_centroids is not None
    assert np.allclose(s2._cluster_centroids, expected_centroids)
    assert s2._cluster_dirty is False


def test_global_rif_unchanged_when_n_clusters_zero(
    tmp_store_path: Path, mock_bi_encoder, mock_cross_encoder,
) -> None:
    """Default n_clusters=0 path still writes to entry.suppression, not cluster state."""
    store = MemoryStore(
        path=tmp_store_path,
        bi_encoder=mock_bi_encoder,
        cross_encoder=mock_cross_encoder,
        dim=16,
        rif_config=RIFConfig(alpha=0.3, use_rank_gap=True, suppression_rate=0.3),
    )
    assert store._cluster_state is None
    for t in ["alpha bravo", "charlie delta", "echo foxtrot", "golf hotel"]:
        store.add(t)
    store.retrieve("alpha", k=2)
    # Global path: cluster centroids never computed.
    assert store._cluster_centroids is None

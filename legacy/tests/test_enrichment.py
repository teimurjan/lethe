"""Unit tests for lethe.enrichment.

Tests the Enrichment dataclass, load_enrichments JSONL parser, cost model,
and _enrich_single with a mocked AsyncAnthropic client. No network / real
API calls.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import pytest

from lethe.enrichment import (
    Enrichment,
    EnrichmentStats,
    _enrich_single,
    enrich_dataset,
    load_enrichments,
)


# ---------- Enrichment dataclass ----------

def test_as_search_text_concats_all_fields() -> None:
    e = Enrichment(
        entry_id="x",
        gist="flew to Paris",
        anticipated_queries=["when did I fly to Paris?", "paris trip date"],
        entities=["Paris"],
        temporal_markers=["March"],
    )
    text = e.as_search_text()
    assert "flew to Paris" in text
    assert "paris trip date" in text
    assert "Paris" in text
    assert "March" in text


def test_as_search_text_skips_empty_fields() -> None:
    e = Enrichment(
        entry_id="x",
        gist="a gist",
        anticipated_queries=[],  # empty
        entities=[],
        temporal_markers=[],
    )
    assert e.as_search_text() == "a gist"


# ---------- load_enrichments ----------

def test_load_enrichments_reads_jsonl_ok(tmp_path: Path) -> None:
    p = tmp_path / "out.jsonl"
    p.write_text(
        json.dumps({
            "entry_id": "a",
            "gist": "summary-a",
            "anticipated_queries": ["q1"],
            "entities": ["E"],
            "temporal_markers": [],
            "input_tokens": 100, "output_tokens": 50,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
            "error": None,
        }) + "\n"
    )
    loaded = load_enrichments(p)
    assert "a" in loaded
    assert loaded["a"].gist == "summary-a"


def test_load_enrichments_skips_error_rows(tmp_path: Path) -> None:
    p = tmp_path / "out.jsonl"
    rows = [
        {"entry_id": "a", "gist": "", "anticipated_queries": [], "entities": [],
         "temporal_markers": [], "error": "RateLimitError: 429"},
        {"entry_id": "b", "gist": "summary-b", "anticipated_queries": [],
         "entities": [], "temporal_markers": [], "error": None},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    loaded = load_enrichments(p)
    assert "a" not in loaded
    assert "b" in loaded


def test_load_enrichments_keeps_latest_successful_per_id(tmp_path: Path) -> None:
    """Retries produce multiple lines for the same entry_id; the last success wins."""
    p = tmp_path / "out.jsonl"
    rows = [
        {"entry_id": "x", "gist": "", "anticipated_queries": [], "entities": [],
         "temporal_markers": [], "error": "first try failed"},
        {"entry_id": "x", "gist": "second try worked",
         "anticipated_queries": [], "entities": [], "temporal_markers": [],
         "error": None},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    loaded = load_enrichments(p)
    assert loaded["x"].gist == "second try worked"


def test_load_enrichments_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_enrichments(tmp_path / "nonexistent.jsonl") == {}


def test_load_enrichments_tolerates_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "out.jsonl"
    p.write_text(
        "not valid json\n"
        + json.dumps({"entry_id": "b", "gist": "ok",
                      "anticipated_queries": [], "entities": [],
                      "temporal_markers": [], "error": None}) + "\n"
    )
    loaded = load_enrichments(p)
    assert set(loaded) == {"b"}


# ---------- EnrichmentStats ----------

def test_cost_estimate_per_model() -> None:
    s = EnrichmentStats(
        total_input_tokens=1_000_000,
        total_output_tokens=1_000_000,
        total_cache_read=0,
        total_cache_write=0,
    )
    assert s.est_cost_usd("claude-haiku-4-5") == pytest.approx(1.0 + 5.0)
    assert s.est_cost_usd("claude-sonnet-4-6") == pytest.approx(3.0 + 15.0)
    assert s.est_cost_usd("claude-opus-4-6") == pytest.approx(5.0 + 25.0)


def test_cost_estimate_unknown_model_defaults_to_sonnet() -> None:
    s = EnrichmentStats(total_input_tokens=1_000_000, total_output_tokens=0)
    assert s.est_cost_usd("claude-unknown") == pytest.approx(3.0)


def test_cost_estimate_includes_cache_tokens() -> None:
    s = EnrichmentStats(
        total_input_tokens=0,
        total_output_tokens=0,
        total_cache_read=1_000_000,
        total_cache_write=1_000_000,
    )
    assert s.est_cost_usd("claude-haiku-4-5") == pytest.approx(0.1 + 1.25)


# ---------- _enrich_single (mocked client) ----------

def _make_usage(input_tokens: int = 100, output_tokens: int = 50,
                cache_read: int = 0, cache_write: int = 0):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )


def _make_text_response(text: str, **usage_kwargs):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block], usage=_make_usage(**usage_kwargs))


def test_enrich_single_parses_json_response() -> None:
    mock_client = SimpleNamespace(
        messages=SimpleNamespace(
            create=AsyncMock(return_value=_make_text_response(
                json.dumps({
                    "gist": "my gist",
                    "anticipated_queries": ["q1", "q2", "q3"],
                    "entities": ["E1"],
                    "temporal_markers": ["yesterday"],
                }),
                input_tokens=120, output_tokens=60, cache_read=80, cache_write=40,
            ))
        )
    )
    out = asyncio.run(_enrich_single(
        mock_client, "id-1", "some memory", "claude-haiku-4-5", asyncio.Semaphore(1),
    ))
    assert out.error is None
    assert out.gist == "my gist"
    assert out.anticipated_queries == ["q1", "q2", "q3"]
    assert out.entities == ["E1"]
    assert out.temporal_markers == ["yesterday"]
    assert out.input_tokens == 120
    assert out.output_tokens == 60
    assert out.cache_read_tokens == 80
    assert out.cache_write_tokens == 40


def test_enrich_single_records_error_on_bad_request() -> None:
    err = anthropic.BadRequestError(
        message="schema violation",
        response=SimpleNamespace(status_code=400, headers={}, request=None),
        body=None,
    )
    mock_client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=err))
    )
    out = asyncio.run(_enrich_single(
        mock_client, "x", "text", "claude-haiku-4-5", asyncio.Semaphore(1),
    ))
    assert out.error is not None
    assert "bad_request" in out.error
    assert out.gist == ""


def test_enrich_single_records_error_on_generic_exception() -> None:
    mock_client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("boom")))
    )
    out = asyncio.run(_enrich_single(
        mock_client, "x", "text", "claude-haiku-4-5", asyncio.Semaphore(1),
    ))
    assert out.error is not None
    assert "RuntimeError" in out.error


# ---------- enrich_dataset resume ----------

def test_enrich_dataset_resume_skips_completed(tmp_path: Path, monkeypatch) -> None:
    """If the output JSONL already has an entry for an id, that id is skipped."""
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({
        "entry_id": "a",
        "gist": "already done",
        "anticipated_queries": [], "entities": [], "temporal_markers": [],
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "error": None,
    }) + "\n")

    call_count = {"n": 0}

    async def mock_create(**kwargs):  # noqa: ANN001
        call_count["n"] += 1
        return _make_text_response(json.dumps({
            "gist": "newly enriched",
            "anticipated_queries": [],
            "entities": [],
            "temporal_markers": [],
        }))

    class FakeClient:
        def __init__(self, *_, **__):
            self.messages = SimpleNamespace(create=mock_create)

    monkeypatch.setattr("lethe.enrichment.AsyncAnthropic", FakeClient)

    stats = asyncio.run(enrich_dataset(
        entries=[("a", "text-a"), ("b", "text-b")],
        output_path=out,
        model="claude-haiku-4-5",
        concurrency=1,
    ))
    assert call_count["n"] == 1  # only "b" was sent
    assert stats.total == 1
    assert stats.completed == 1
    assert stats.failed == 0

    loaded = load_enrichments(out)
    assert loaded["a"].gist == "already done"
    assert loaded["b"].gist == "newly enriched"


def test_enrich_dataset_noop_when_everything_done(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({
        "entry_id": "a",
        "gist": "done",
        "anticipated_queries": [], "entities": [], "temporal_markers": [],
        "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
        "error": None,
    }) + "\n")

    class FakeClient:
        def __init__(self, *_, **__):
            self.messages = SimpleNamespace(create=AsyncMock())

    monkeypatch.setattr("lethe.enrichment.AsyncAnthropic", FakeClient)

    stats = asyncio.run(enrich_dataset(
        entries=[("a", "text-a")],
        output_path=out,
        model="claude-haiku-4-5",
    ))
    assert stats.total == 0
    assert stats.completed == 0

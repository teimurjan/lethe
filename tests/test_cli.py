"""Tests for lethe.cli.

Avoids downloading real sentence-transformers models by monkeypatching
`_load_encoders` to return the mocks from conftest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lethe import cli


# ---------- helpers ----------

@pytest.fixture
def isolated_project(tmp_path: Path, monkeypatch) -> Path:
    """Fresh project root for a CLI invocation. Chdir into it."""
    (tmp_path / ".git").mkdir()  # pretend it's a git repo for resolve_paths
    (tmp_path / ".lethe" / "memory").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def patched_encoders(monkeypatch, mock_bi_encoder, mock_cross_encoder):
    """Replace the real sentence-transformers loader with lightweight mocks."""
    def fake_loader(_cfg):
        return mock_bi_encoder, mock_cross_encoder
    monkeypatch.setattr(cli, "_load_encoders", fake_loader)


# ---------- Paths & version ----------

def test_resolve_paths_uses_git_root(isolated_project: Path) -> None:
    paths = cli.resolve_paths()
    assert paths.root == isolated_project
    assert paths.memory == isolated_project / ".lethe" / "memory"
    assert paths.index == isolated_project / ".lethe" / "index"


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert cli.__version__ in capsys.readouterr().out


def test_main_without_command_returns_usage() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


# ---------- Config ----------

def test_config_get_default_value(isolated_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["config", "get", "top_k"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == str(cli.DEFAULT_CONFIG["top_k"])


def test_config_set_then_get_roundtrips(
    isolated_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["config", "set", "top_k", "7"])
    assert rc == 0
    capsys.readouterr()  # drain

    rc = cli.main(["config", "get", "top_k"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "7"


def test_config_set_bool(isolated_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["config", "set", "use_rank_gap", "false"])
    assert rc == 0
    cfg = cli.load_config(cli.resolve_paths())
    assert cfg["use_rank_gap"] is False


def test_config_get_unset_returns_error(
    isolated_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["config", "get", "no_such_key"])
    assert rc == 1


def test_config_get_no_key_prints_all_json(
    isolated_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["config", "get"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    for key in cli.DEFAULT_CONFIG:
        assert key in payload


# ---------- Status ----------

def test_status_uninitialized_project(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["status"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["initialized"] is False
    assert payload["total_entries"] == 0


def test_status_initialized_project(
    isolated_project: Path, patched_encoders, capsys: pytest.CaptureFixture[str]
) -> None:
    md_path = isolated_project / ".lethe" / "memory" / "today.md"
    md_path.write_text("## Heading\n- a line\n", encoding="utf-8")
    assert cli.main(["index"]) == 0
    capsys.readouterr()  # drain index output

    rc = cli.main(["status"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["initialized"] is True
    assert payload["total_entries"] >= 1
    assert "tiers" in payload


# ---------- Index / search / expand ----------

def test_index_creates_entries_and_search_finds_them(
    isolated_project: Path, patched_encoders, capsys: pytest.CaptureFixture[str]
) -> None:
    md_path = isolated_project / ".lethe" / "memory" / "today.md"
    md_path.write_text(
        "## Travel\n- I prefer window seats on flights\n\n## Work\n- I work at Google\n",
        encoding="utf-8",
    )

    assert cli.main(["index", "--json-output"]) == 0
    index_out = json.loads(capsys.readouterr().out)
    assert index_out["added"] == 2

    rc = cli.main(["search", "window seats", "--top-k", "3", "--json-output"])
    assert rc == 0
    results = json.loads(capsys.readouterr().out)
    assert isinstance(results, list)
    assert results, "expected at least one search result"
    assert all({"id", "content", "score"} <= set(r) for r in results)


def test_expand_prints_chunk(
    isolated_project: Path, patched_encoders, capsys: pytest.CaptureFixture[str]
) -> None:
    md_path = isolated_project / ".lethe" / "memory" / "today.md"
    md_path.write_text(
        "## Travel\n<!-- session:s turn:t transcript:/tmp/x -->\n- window seats\n",
        encoding="utf-8",
    )
    assert cli.main(["index", "--json-output"]) == 0
    capsys.readouterr()

    from lethe.markdown_store import MarkdownStore
    paths = cli.resolve_paths()
    chunk_id = MarkdownStore(paths.memory, paths.index).scan()[0].id

    rc = cli.main(["expand", chunk_id])
    assert rc == 0
    out = capsys.readouterr().out
    assert "window seats" in out
    assert "session:s" in out


def test_expand_missing_chunk_returns_error(
    isolated_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["expand", "deadbeef" * 2])
    assert rc == 1


# ---------- Reset ----------

def test_reset_without_confirm_fails(
    isolated_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (isolated_project / ".lethe" / "index").mkdir(parents=True, exist_ok=True)
    rc = cli.main(["reset"])
    assert rc == 1
    assert (isolated_project / ".lethe" / "index").exists()


def test_reset_yes_removes_index(
    isolated_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = isolated_project / ".lethe" / "index"
    index.mkdir(parents=True, exist_ok=True)
    (index / "lethe.db").write_text("", encoding="utf-8")

    rc = cli.main(["reset", "--yes"])
    assert rc == 0
    assert not index.exists()
    # Markdown must be preserved.
    assert (isolated_project / ".lethe" / "memory").exists()


# ---------- Enrich ----------

def test_enrich_without_api_key_is_rejected(
    isolated_project: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rc = cli.main(["enrich"])
    assert rc == 2


def test_enrich_happy_path_invokes_enrich_dataset(
    isolated_project: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """cmd_enrich should scan chunks and hand them to ``enrich_dataset``."""
    md_path = isolated_project / ".lethe" / "memory" / "today.md"
    md_path.write_text("## A\n- one\n\n## B\n- two\n", encoding="utf-8")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")

    captured: dict[str, object] = {}

    async def fake_enrich_dataset(pairs, output_path, model, concurrency, **_):
        from lethe.enrichment import EnrichmentStats
        captured["pairs"] = list(pairs)
        captured["output_path"] = output_path
        captured["model"] = model
        captured["concurrency"] = concurrency
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        stats = EnrichmentStats(total=len(pairs))
        stats.completed = len(pairs)
        return stats

    monkeypatch.setattr("lethe.enrichment.enrich_dataset", fake_enrich_dataset)

    rc = cli.main(["enrich", "--model", "claude-haiku-4-5", "--concurrency", "2"])
    assert rc == 0

    assert captured["model"] == "claude-haiku-4-5"
    assert captured["concurrency"] == 2
    assert len(captured["pairs"]) == 2  # two chunks from the seeded file

    payload = json.loads(capsys.readouterr().out)
    assert payload["completed"] == 2
    assert payload["failed"] == 0
    assert payload["output"].endswith("enrichments.jsonl")


# ---------- Config edges ----------

def test_config_set_persists_to_toml_file(isolated_project: Path) -> None:
    """After `config set`, re-reading the file produces the same value."""
    rc = cli.main(["config", "set", "bi_encoder", "custom/model"])
    assert rc == 0
    cfg_path = isolated_project / ".lethe" / "config.toml"
    contents = cfg_path.read_text(encoding="utf-8")
    assert 'bi_encoder = "custom/model"' in contents
    # Fresh load picks it up.
    assert cli.load_config(cli.resolve_paths())["bi_encoder"] == "custom/model"


def test_format_toml_value_round_trip() -> None:
    assert cli._format_toml_value(True) == "true"
    assert cli._format_toml_value(False) == "false"
    assert cli._format_toml_value(42) == "42"
    assert cli._format_toml_value(3.14) == "3.14"
    assert cli._format_toml_value('a "quoted" value') == '"a \\"quoted\\" value"'
    # Lists fall through to json.dumps as a best-effort serialization.
    assert cli._format_toml_value([1, 2]) == "[1, 2]"


def test_coerce_scalar_picks_the_right_type() -> None:
    assert cli._coerce_scalar("true") is True
    assert cli._coerce_scalar("False") is False
    assert cli._coerce_scalar("42") == 42
    assert cli._coerce_scalar("3.14") == pytest.approx(3.14)
    assert cli._coerce_scalar("hello") == "hello"


def test_load_config_with_malformed_toml_falls_back_to_defaults(
    isolated_project: Path,
) -> None:
    cfg_path = isolated_project / ".lethe" / "config.toml"
    cfg_path.write_text("this = is = not = toml", encoding="utf-8")
    cfg = cli.load_config(cli.resolve_paths())
    assert cfg == cli.DEFAULT_CONFIG


# ---------- Registry + global search ----------

def test_projects_add_list_remove_roundtrip(
    isolated_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Register current project via explicit `projects add`.
    rc = cli.main(["projects", "add"])
    assert rc == 0
    capsys.readouterr()

    rc = cli.main(["projects", "list", "--json-output"])
    assert rc == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 1
    assert listed[0]["root"] == str(isolated_project)

    rc = cli.main(["projects", "remove", str(isolated_project)])
    assert rc == 0
    capsys.readouterr()

    rc = cli.main(["projects", "list", "--json-output"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == []


def test_projects_prune_drops_missing_roots(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from lethe import _registry

    # Seed two entries: one real, one pointing at a vanished dir.
    real = tmp_path / "real"
    real.mkdir()
    gone = tmp_path / "gone"
    gone.mkdir()
    _registry.register(real)
    _registry.register(gone)
    gone.rmdir()

    rc = cli.main(["projects", "prune"])
    assert rc == 0
    capsys.readouterr()
    entries = _registry.load()
    roots = [str(e.root) for e in entries]
    assert str(real.resolve()) in roots
    assert str(gone.resolve()) not in roots


def test_index_auto_registers_by_default(
    isolated_project: Path, patched_encoders
) -> None:
    from lethe import _registry

    md = isolated_project / ".lethe" / "memory" / "today.md"
    md.write_text("# today\n\n## Session 10:00\n- a real bullet\n", encoding="utf-8")

    rc = cli.main(["index"])
    assert rc == 0
    entries = _registry.load()
    assert any(e.root == isolated_project.resolve() for e in entries)


def test_index_no_register_opts_out(
    isolated_project: Path, patched_encoders
) -> None:
    from lethe import _registry

    md = isolated_project / ".lethe" / "memory" / "today.md"
    md.write_text("# today\n\n## Session 10:00\n- a bullet\n", encoding="utf-8")

    rc = cli.main(["index", "--no-register"])
    assert rc == 0
    assert _registry.load() == []


def test_search_all_with_empty_registry_exits_nonzero(
    isolated_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["search", "anything", "--all"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no projects registered" in err


def test_lock_allows_sequential_cli_in_same_project(
    isolated_project: Path, patched_encoders
) -> None:
    """Serialized CLI calls under the same lockfile don't raise. This is a
    smoke test for the fcntl-flock wrapper — real concurrency needs a
    subprocess harness."""
    md = isolated_project / ".lethe" / "memory" / "today.md"
    md.write_text("# today\n\n## Session 10:00\n- bullet\n", encoding="utf-8")

    assert cli.main(["index"]) == 0
    assert cli.main(["status"]) == 0
    assert cli.main(["reset", "--yes"]) == 0

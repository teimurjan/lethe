# lethe — Claude Code plugin

Persistent memory across Claude Code sessions. Indexes your transcripts directly, hybrid BM25 + dense retrieval, clustered retrieval-induced forgetting.

## Install

```
/plugin marketplace add teimurjan/lethe
/plugin install lethe
```

The plugin is skills-only — it installs no hooks and writes nothing into your
repos. lethe reads the transcripts Claude Code already keeps under
`$CLAUDE_CONFIG_DIR/projects/<slug>/` (default `~/.claude`) and maintains a
global index under `~/.lethe/`:

```
~/.lethe/
├── projects.json          # registry of indexed projects (for cross-project recall)
├── config.toml            # user-editable knobs (encoder choice, etc.)
└── index/<slug>/          # per-project DuckDB index (rebuildable, safe to delete)
```

## How it works

Recall is on-demand via two skills — no background capture, no LLM
summarization:

- **`recall`** — searches the current project. `lethe search` transparently
  reindexes any new/changed transcripts before searching, so results always
  reflect your latest turns. Each hit is a raw user+assistant turn.
- **`recall-global`** — searches every registered project (cross-repo).

## CLI

Invoked by the skills; also usable directly.

```
lethe --version
lethe index                     # index this project's transcripts now (also warms a fresh clone)
lethe search "QUERY" --top-k 5
lethe search "QUERY" --json-output
lethe search "QUERY" --all      # across all registered projects
lethe expand <chunk-id>
lethe status
lethe config get KEY
lethe config set KEY VALUE
lethe reset --yes               # wipes this project's index (transcripts untouched)
```

`lethe search` (and the `recall` skill) reindex on demand, so you rarely need
`lethe index` explicitly — it's mainly for warming a fresh checkout ahead of
time or after a `reset`.

## Requirements

- `lethe` and `lethe-claude-code` binaries on PATH. Install with
  `brew tap teimurjan/lethe && brew install lethe` (macOS / Linuxbrew) or
  `cargo install lethe-cli lethe-claude-code`, or download a release
  tarball from https://github.com/teimurjan/lethe/releases.

## Reference

- Repo: <https://github.com/teimurjan/lethe>
- Retrieval methodology and benchmarks: [BENCHMARKS.md](../../BENCHMARKS.md)

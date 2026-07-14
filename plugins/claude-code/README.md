# lethe — Claude Code plugin

Persistent memory across Claude Code sessions. Indexes your transcripts directly, hybrid BM25 + dense retrieval, clustered retrieval-induced forgetting.

## Install

```
/plugin marketplace add teimurjan/lethe
/plugin install lethe
```

The plugin ships two recall skills plus one small background hook, and writes
nothing into your repos. lethe reads the transcripts Claude Code already keeps
under `$CLAUDE_CONFIG_DIR/projects/<slug>/` (default `~/.claude`) and maintains
a global index under `~/.lethe/`:

```
~/.lethe/
├── projects.json          # registry of indexed projects (for cross-project recall)
├── config.toml            # user-editable knobs (encoder choice, etc.)
└── index/<slug>/          # per-project DuckDB index (rebuildable, safe to delete)
```

## How it works

Recall is on-demand via two skills — no LLM summarization, nothing written into
your repos:

- **`recall`** — searches the current project. `lethe search` transparently
  reindexes any new/changed transcripts before searching, so results always
  reflect your latest turns. Each hit is a raw user+assistant turn.
- **`recall-global`** — searches every registered project (cross-repo).

### Background freshness

A `UserPromptSubmit` hook keeps the index current. On prompt submit it fires a
**throttled, detached** `lethe index --all && lethe dedupe --all` — at most once
every 15 minutes, backgrounded so it never blocks your prompt, single-flighted
so runs don't pile up, and a no-op if `lethe` isn't on `PATH`. Because `recall`
already reindexes the *current* project on demand, the hook's real job is
keeping **other** registered projects fresh for cross-project `recall-global`
(whose search opens them read-only and never reindexes).

Tune or disable via env: `LETHE_REFRESH_INTERVAL` (seconds, default `900`) and
`LETHE_HOME` (state dir, default `~/.lethe`). Set `LETHE_REFRESH_INTERVAL` very
high to effectively disable it.

## CLI

Invoked by the skills; also usable directly.

```
lethe --version
lethe index                     # index this project's transcripts now (also warms a fresh clone)
lethe index --all               # reindex every registered project at once
lethe search "QUERY" --top-k 5
lethe search "QUERY" --json-output
lethe search "QUERY" --all      # across all registered projects
lethe expand <chunk-id>
lethe dedupe                    # merge near-duplicate chunks in this project
lethe dedupe --dry-run          # preview the near-duplicate groups first
lethe dedupe --all              # compact every registered project
lethe status
lethe config get KEY
lethe config set KEY VALUE
lethe reset --yes               # wipes this project's index (transcripts untouched)
```

`lethe search` (and the `recall` skill) reindex on demand, so you rarely need
`lethe index` explicitly — it's mainly for warming a fresh checkout ahead of
time or after a `reset`.

## Requirements

- `lethe` on PATH. Install with `brew tap teimurjan/lethe && brew install lethe`
  (macOS / Linuxbrew), `cargo install lethe-cli`, or a release tarball from
  <https://github.com/teimurjan/lethe/releases>.

## Reference

- Repo: <https://github.com/teimurjan/lethe>
- Retrieval methodology and benchmarks: [BENCHMARKS.md](../../BENCHMARKS.md)

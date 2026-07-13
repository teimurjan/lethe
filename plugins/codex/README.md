# lethe — Codex CLI plugin

Persistent memory across [Codex CLI](https://developers.openai.com/codex) sessions. Indexes your transcripts directly, hybrid BM25 + dense retrieval, clustered retrieval-induced forgetting.

## Install

```bash
# 1. Install the lethe binaries
brew tap teimurjan/lethe && brew install lethe   # or: cargo install lethe-cli lethe-codex

# 2. Add the marketplace, then install from Codex
codex plugin marketplace add teimurjan/lethe
```

Then run `codex`, open `/plugins`, and install **lethe**. Codex reads the marketplace manifest at [`.agents/plugins/marketplace.json`](https://github.com/teimurjan/lethe/blob/main/.agents/plugins/marketplace.json) and wires the `recall` / `recall-global` skills automatically — no `config.toml` editing. `codex plugin marketplace upgrade teimurjan` pulls updates.

The plugin ships two recall skills plus one small background hook, and writes
nothing into your repos. lethe reads the Codex rollouts already kept under
`$CODEX_HOME/sessions/` (default `~/.codex`) and maintains a global index under
`~/.lethe/`:

```
~/.lethe/
├── projects.json          # registry of indexed projects (for cross-project recall)
├── config.toml            # user-editable knobs
└── index/<slug>/          # per-project DuckDB index (rebuildable, safe to delete)
```

## How it works

Recall is on-demand via two skills — no LLM summarization, nothing written into
your repos:

- **`recall`** — searches the current project. `lethe search` transparently
  reindexes any new/changed transcripts before searching, filtering the Codex
  sessions tree by the session's recorded `cwd`. Each hit is a raw
  user+assistant turn.
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

## Requirements

- `lethe`, `lethe-codex` (and `lethe-claude-code` if you also use the Claude Code plugin) binaries on PATH. Install with `brew tap teimurjan/lethe && brew install lethe` or `cargo install lethe-cli lethe-codex`.
- Codex CLI.

## Reference

- Repo: <https://github.com/teimurjan/lethe>

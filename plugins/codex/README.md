# lethe — Codex CLI plugin

Persistent memory across [Codex CLI](https://developers.openai.com/codex) sessions. Markdown-first storage, hybrid BM25 + dense retrieval, clustered retrieval-induced forgetting.

## Install

```bash
# 1. Install the lethe binaries
brew tap teimurjan/lethe && brew install lethe   # or: cargo install lethe-cli lethe-codex

# 2. Add the marketplace, then install from Codex
codex plugin marketplace add teimurjan/lethe
```

Then run `codex`, open `/plugins`, and install **lethe**. Codex reads the marketplace manifest at [`.agents/plugins/marketplace.json`](https://github.com/teimurjan/lethe/blob/main/.agents/plugins/marketplace.json) and wires the hooks (`hooks/hooks.json`) and the `recall` / `recall-global` skills automatically — no `config.toml` editing. On first run Codex asks you to review and trust the plugin's hooks before they fire. `codex plugin marketplace upgrade teimurjan` pulls updates.

After install, a `.lethe/` directory will appear in each project's git root on first use:

```
.lethe/
├── memory/            # source of truth — daily markdown files (git-diffable)
├── index/             # rebuildable DuckDB artifacts (safe to delete)
└── hooks.log          # only when LETHE_DEBUG=1
```

## How it works

| Event | Behavior |
|-------|----------|
| `SessionStart` | Injects the last ~30 lines from the 2 most recent daily files via `systemMessage`. |
| `UserPromptSubmit` | On the first prompt of a session, appends a `## Session HH:MM` heading. The `recall` (this project) and `recall-global` (all projects) skills decide when memory is relevant from their own descriptions — no per-prompt hint is injected. |
| `Stop` (async) | Summarizes the last turn via `claude -p --model haiku` (parsed from the rollout JSONL by `lethe-codex transcript`), appends bullets + a progressive-disclosure anchor to today's file, and reindexes. |

### Differences from the Claude Code plugin

- **No SessionEnd hook** in Codex CLI. Header sentinels persist for the whole session — they're tiny empty files in `<project>/.lethe/` and harmless if stale.

### First-time setup

Backfill memories from existing transcripts (Codex rollouts under
`~/.codex/sessions/`, plus Claude Code transcripts if any) before relying on
the live hooks:

```bash
lethe seed --days 7              # current project, both sources
lethe seed --days 30 --source codex --dry-run   # preview only
```

## Requirements

- `lethe`, `lethe-codex` (and `lethe-claude-code` if you also use the Claude Code plugin) binaries on PATH. Install with `brew tap teimurjan/lethe && brew install lethe` or `cargo install lethe-cli lethe-codex`.
- Codex CLI. Hooks are enabled by default; on first run Codex prompts you to review and trust the plugin's hook definitions before they fire (re-prompts whenever they change). Disable globally with `[features] hooks = false`.
- One of `codex` (preferred — uses your Codex auth) or `claude` (fallback) on PATH for the Stop hook summarizer. Without either, Stop still writes the anchor + reindexes; just no bullet summary.

## Debugging

Set `LETHE_DEBUG=1` in your shell. Hook traces land in `<project>/.lethe/hooks.log`.

## Reference

- Repo: <https://github.com/teimurjan/lethe>
- Codex hooks docs: <https://developers.openai.com/codex/hooks>

# lethe — Codex CLI plugin

Persistent memory across [Codex CLI](https://developers.openai.com/codex) sessions. Markdown-first storage, hybrid BM25 + dense retrieval, clustered retrieval-induced forgetting.

## Install

```bash
# 1. Install the lethe binary
brew tap teimurjan/lethe && brew install lethe   # or: cargo install lethe-cli

# 2. Wire the hooks into Codex
git clone https://github.com/teimurjan/lethe /tmp/lethe   # or use a release tarball
bash /tmp/lethe/plugins/codex/install.sh --auto-config
```

`install.sh` copies the hooks and skills to `~/.codex/lethe/` and (with `--auto-config`) appends a marked block to `~/.codex/config.toml`. Re-running the installer replaces the existing block, so updates are idempotent.

Without `--auto-config` the script prints the snippet for you to paste manually.

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
| `UserPromptSubmit` | On the first prompt of a session, appends a `## Session HH:MM` heading; emits a `[lethe] Memory available` hint so the agent invokes `recall` (this project) or `recall-global` (all projects). |
| `Stop` (async) | Summarizes the last turn via `claude -p --model haiku` (parsed from the rollout JSONL by `lethe-codex transcript`), appends bullets + a progressive-disclosure anchor to today's file, and reindexes. |

### Differences from the Claude Code plugin

- **No SessionEnd hook** in Codex CLI. Header sentinels persist for the whole session — they're tiny empty files in `<project>/.lethe/` and harmless if stale.

## Requirements

- `lethe`, `lethe-codex` (and `lethe-claude-code` if you also use the Claude Code plugin) binaries on PATH. Install with `brew tap teimurjan/lethe && brew install lethe` or `cargo install lethe-cli lethe-codex`.
- Codex CLI with hooks enabled (`[features].codex_hooks = true`, set automatically by `--auto-config`).
- `claude` CLI for the Stop hook summarizer (uses your existing auth — no extra API key). Without it, Stop still writes the anchor + reindexes; just no bullet summary.

## Debugging

Set `LETHE_DEBUG=1` in your shell. Hook traces land in `<project>/.lethe/hooks.log`.

## Reference

- Repo: <https://github.com/teimurjan/lethe>
- Codex hooks docs: <https://developers.openai.com/codex/hooks>

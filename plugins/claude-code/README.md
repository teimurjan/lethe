# lethe — Claude Code plugin

Persistent memory across Claude Code sessions. Markdown-first storage, hybrid BM25 + dense retrieval, clustered retrieval-induced forgetting, optional Haiku enrichment.

## Install

```
/plugin marketplace add teimurjan/lethe
/plugin install lethe
```

After restart, a `.lethe/` directory will appear in each project's git root on first use:

```
.lethe/
├── memory/            # source of truth — daily markdown files (git-diffable)
├── index/             # rebuildable SQLite + FAISS artifacts (safe to delete)
├── enrichments.jsonl  # optional LLM enrichments per chunk
└── config.toml        # user-editable knobs
```

## How it works

| Event | Behavior |
|-------|----------|
| `SessionStart` | Injects the last ~30 lines from the 2 most recent daily files as additional context. |
| `UserPromptSubmit` | On the first prompt of a session, appends a `## Session HH:MM` heading; emits a `[lethe] Memory available` hint so Claude can invoke the `recall` (this project) or `recall-global` (all projects) skill. |
| `Stop` (async) | Summarizes the latest turn via `claude -p --model haiku`, appends bullets + a progressive-disclosure anchor to today's file, and reindexes. |
| `SessionEnd` (async) | Flushes suppression state. |

## CLI

Invoked by the skill and the hooks; also usable directly.

```
lethe --version
lethe index [DIR]               # reindex markdown files (default: .lethe/memory)
lethe search "QUERY" --top-k 5
lethe search "QUERY" --json-output
lethe expand <chunk-id>
lethe status
lethe config get KEY
lethe config set KEY VALUE
lethe reset --yes               # wipes .lethe/index/; markdown preserved
lethe enrich [DIR]              # optional: Haiku enrichment (needs ANTHROPIC_API_KEY)
```

## Requirements

- `lethe` and `lethe-claude-code` binaries on PATH. Install with
  `brew tap teimurjan/lethe && brew install lethe` (macOS / Linuxbrew) or
  `cargo install lethe-cli lethe-claude-code`, or download a release
  tarball from https://github.com/teimurjan/lethe/releases.
- `claude` CLI for the Stop hook summarizer (uses your existing auth — no extra API key)

## Debugging

Set `LETHE_DEBUG=1` to write hook traces to `.lethe/hooks.log`.

## Reference

- Repo: <https://github.com/teimurjan/lethe>
- Retrieval methodology and benchmarks: [BENCHMARKS.md](../../BENCHMARKS.md)

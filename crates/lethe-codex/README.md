# lethe-codex

Codex CLI adapter binary for [lethe](https://github.com/teimurjan/lethe).
Helpers the [Codex plugin](https://github.com/teimurjan/lethe/tree/main/plugins/codex)
needs that don't belong in `lethe-core` — kept separate so the core stays
framework-agnostic.

## Install

```bash
brew tap teimurjan/lethe              # ships alongside the `lethe` binary
brew install lethe
cargo install lethe-codex             # standalone
```

## Subcommands

```bash
lethe-codex transcript <rollout.jsonl>
lethe-codex transcript <rollout.jsonl> --turn <turn-uuid>
```

`transcript` parses a Codex CLI rollout JSONL (`$CODEX_HOME/sessions/YYYY/MM/DD/rollout-*.jsonl`) and prints the last user/assistant pair (or a specific turn) as plain text suitable for piping into a downstream LLM enrichment prompt.

Output is byte-for-byte compatible with `lethe-claude-code transcript` so the same hook pipeline (`hooks/parse-transcript.sh`) works for both agents.

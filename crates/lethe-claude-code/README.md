# lethe-claude-code

Claude Code adapter binary for [lethe](https://github.com/teimurjan/lethe).
Helpers the [Claude Code plugin](https://github.com/teimurjan/lethe/tree/main/plugins/claude-code)
needs that don't belong in `lethe-core` — kept separate so the core stays
framework-agnostic.

## Install

```bash
brew tap teimurjan/lethe              # ships alongside the `lethe` binary
brew install lethe
cargo install lethe-claude-code       # standalone
```

## Subcommands

```bash
lethe-claude-code transcript <transcript.jsonl>
lethe-claude-code transcript <transcript.jsonl> --turn <user-turn-uuid>
```

`transcript` parses a Claude Code JSONL transcript and prints the
last user/assistant pair (or a specific turn) as plain text suitable
for piping into a downstream LLM enrichment prompt.

Used by the plugin's `parse-transcript.sh` hook and the `recall`
skill's progressive-disclosure step. The plugin is the only intended
consumer; it's published as a separate crate purely so the lethe core
CLI doesn't grow framework-specific surface.

## See also

- [Project landing page](https://github.com/teimurjan/lethe)
- [Claude Code plugin](https://github.com/teimurjan/lethe/tree/main/plugins/claude-code) — the consumer

License: MIT.

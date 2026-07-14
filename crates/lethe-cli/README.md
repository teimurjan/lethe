# lethe — CLI

Command-line interface for [lethe](https://github.com/teimurjan/lethe), a
self-improving memory store for LLM agents.

## Install

Homebrew installs the CLI and both coding-agent transcript adapters:

```bash
brew tap teimurjan/lethe              # macOS / Linuxbrew (recommended)
brew install lethe
```

For the standalone CLI only, use Cargo on any platform with a Rust toolchain:

```bash
cargo install lethe-cli
```

Or download a tarball from [GitHub Releases](https://github.com/teimurjan/lethe/releases).

## Use

```bash
lethe                                # no args, in a TTY → opens the TUI
lethe index                          # index this project's Claude/Codex/OMP transcripts
lethe search "your query" --top-k 5
lethe search "your query" --all      # cross-project (~/.lethe/projects.json)
lethe dedupe --dry-run               # preview near-duplicate compaction
lethe dedupe --all                   # compact every registered project
lethe projects list|add|remove|prune
lethe expand <chunk-id> [<chunk-id> ...]  # full markdown for one or more chunks
lethe status                         # diagnostic JSON
```

`lethe` with no subcommand opens the embedded TUI when stdout is a terminal;
otherwise it prints `--help` and exits 2 (so scripts get a useful exit code).

## TUI keys

`↑↓` navigate · `⏎` search/open · `Esc` back · `Ctrl+L`/`P`/`R` focus
search/projects/results · `Ctrl+Q` quit. Type anywhere to refocus the
search input.

## See also

- [Project landing page](https://github.com/teimurjan/lethe) — architecture, benchmarks, research journey
- [`lethe-core`](https://crates.io/crates/lethe-core) — embed the retrieval pipeline in your own Rust app
- [Claude Code plugin](https://github.com/teimurjan/lethe/tree/main/plugins/claude-code) — recall skills backed by transcript indexing

License: MIT.

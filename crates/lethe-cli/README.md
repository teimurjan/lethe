# lethe — CLI

Command-line interface for [lethe](https://github.com/teimurjan/lethe), a
self-improving memory store for LLM agents.

## Install

```bash
brew tap teimurjan/lethe              # macOS / Linuxbrew (recommended)
brew install lethe
cargo install lethe-cli               # any platform with a Rust toolchain
```

Or download a tarball from [GitHub Releases](https://github.com/teimurjan/lethe/releases).

## Use

```bash
lethe                                # no args, in a TTY → opens the TUI
lethe index                          # reindex .lethe/memory in the current repo
lethe search "your query" --top-k 5
lethe search "your query" --all      # cross-project (~/.lethe/projects.json)
lethe projects list|add|remove|prune
lethe expand <chunk-id>              # full markdown for one chunk
lethe status                         # diagnostic JSON
lethe migrate [--all]                # convert legacy embeddings.npz → DuckDB
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
- [Claude Code plugin](https://github.com/teimurjan/lethe/tree/main/plugins/claude-code) — agentic memory hooked into Claude Code

License: MIT.

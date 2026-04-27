# Development

## Setup

```bash
# Rust toolchain (1.94+)
rustup toolchain install stable

# Python venv for the legacy library + parity bench
uv venv --python 3.12
uv pip install -e 'legacy/[dev]'
```

The CLI is the Rust binary `lethe` (built from `crates/lethe-cli`). The Python package under `legacy/` is the original implementation, kept for the research trail and to back the parity bench. PyO3 bindings (`crates/lethe-py`) and napi-rs bindings (`crates/lethe-node`) are the supported language-binding paths going forward.

## Run tests

```bash
cargo test --workspace
cd legacy && uv run pytest tests/ -q
```

Rust: 68 unit tests, sub-second. Python: 178 production + 8 PyO3 parity = 186, ~3 minutes (the PyO3 set loads ONNX models). No network, no API keys required.

## Run the CLI locally

```bash
cargo run -p lethe-cli -- search "query"     # debug build, fast iteration
cargo install --path crates/lethe-cli        # install local build to ~/.cargo/bin
lethe                                         # opens TUI (if stdout is a terminal)
```

Common commands once installed:

```bash
lethe index                     # reindex .lethe/memory in the current repo
lethe search "query" --top-k 5
lethe search "query" --all      # cross-project via ~/.lethe/projects.json
lethe tui                       # explicit TUI (same as no-arg in a TTY)
lethe projects list
```

## Try the Claude Code plugin locally

Point Claude Code's marketplace at this checkout:

```
/plugin marketplace add /Users/you/path/to/lethe
/plugin install lethe
```

Hooks run `bash ${CLAUDE_PLUGIN_ROOT}/hooks/*.sh`; they invoke `lethe` from PATH. After `cargo install --path crates/lethe-cli`, the binary is on `~/.cargo/bin/lethe` and the hooks pick it up — no publish needed.

Turn on hook traces while iterating:

```bash
export LETHE_DEBUG=1   # writes to .lethe/hooks.log of the target repo
```

After editing `plugins/claude-code/` files (hooks, skills, manifest), run `/reload-plugins` in Claude Code.

## Commit conventions

Conventional commits. `release-please` only bumps on `feat:` / `fix:`. The workspace ships four artifacts on the same version (Rust binary via Homebrew/crates.io, `lethe-rust` wheel on PyPI, `@lethe/memory` on npm), so a `feat:` triggers releases everywhere — use it sparingly. Everything else (`chore:`, `docs:`, `refactor:`, `test:`) does not trigger a release. Breaking changes use `feat!:` or `fix!:`.

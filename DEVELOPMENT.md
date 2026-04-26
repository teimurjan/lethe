# Development

## Setup

```bash
uv venv --python 3.12
uv pip install -e 'legacy/[dev,tui]'
```

`dev` pulls pytest + mypy; `tui` pulls textual (needed only for `lethe tui`). Drop `tui` if you don't need it.

The Python implementation lives under `legacy/` after the Rust port; see `crates/` for the Rust workspace and `benchmarks/` for the Python ↔ Rust parity bench.

## Run tests

```bash
cd legacy && uv run pytest tests/ -q
```

178 Python tests + 8 PyO3 parity tests = 186, ~3 minutes (PyO3 set loads ONNX models). The pure-Python set runs in ~2 seconds. No network, no API keys required.

## Run the CLI locally

The editable install wires `lethe` to `legacy/lethe/cli.py`. Two ways to invoke it:

```bash
uv run lethe search "query"     # uses the project's .venv — always current
```

…or install the working tree as your global `lethe` so you can just type `lethe` anywhere:

```bash
uv tool install --force --editable ./legacy --with 'textual>=0.80'
lethe tui
```

`--force --editable` overwrites any previously-installed `lethe` (e.g. from `uv tool install lethe-memory`). `--with textual>=0.80` adds the `tui` extra — `uv tool install` doesn't honor `[project.optional-dependencies]` by default. Check which binary is active with `which lethe` if `tui` comes back as an invalid subcommand.

Common commands:

```bash
lethe index                     # reindex .lethe/memory in the current repo
lethe search "query" --top-k 5
lethe search "query" --all      # cross-project via ~/.lethe/projects.json
lethe tui                       # interactive browser (needs textual)
lethe projects list
```

## Try the Claude Code plugin locally

Point Claude Code's marketplace at this checkout:

```
/plugin marketplace add /Users/you/path/to/lethe
/plugin install lethe
```

Hooks run `bash ${CLAUDE_PLUGIN_ROOT}/hooks/*.sh`; they invoke `lethe` from PATH first, then fall back to `uvx --from git+... lethe`. With an editable install on PATH, your local changes hit the hooks immediately — no publish needed.

Turn on hook traces while iterating:

```bash
export LETHE_DEBUG=1   # writes to .lethe/hooks.log of the target repo
```

After editing `plugins/claude-code/` files (hooks, skills, manifest), run `/reload-plugins` in Claude Code.

## Commit conventions

Conventional commits. `release-please` only bumps on `feat:` / `fix:`, so use those sparingly — everything else (`chore:`, `docs:`, `refactor:`, `test:`) does not trigger a PyPI release. Breaking changes use `feat!:` or `fix!:`.

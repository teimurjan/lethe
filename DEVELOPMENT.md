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

Rust: 71 unit tests, sub-second (plus the CLI smoke + cross-impl npz tests). Python: 148 production + 8 PyO3 parity = 156, ~3 minutes (the PyO3 set loads ONNX models). No network, no API keys required.

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

## Building release artifacts

Hybrid: **macOS-arm64 builds locally on the maintainer's machine**,
**Linux + Windows build in CI** on native runners. ort's prebuilt
ONNX Runtime only links cleanly on the same platform it was compiled
for; cross-compiling C++ from macOS hits libstdc++/MSVC-runtime ABI
mismatches we don't want to fight, so each target builds natively.

The local macOS files go in `release_artifacts/<tag>/`; CI appends
its Linux/Windows files to the same versioned subdir. The whole
folder is committed to main via Git LFS, then attached to the
GitHub Release.

### Supported targets

| Target | Friendly | Built where |
|---|---|---|
| `aarch64-apple-darwin` | `macos-arm64` | local: `scripts/release/build.sh --tag vX.Y.Z` |
| `x86_64-unknown-linux-gnu` | `linux-x64` | CI (`ubuntu-latest`) |
| `aarch64-unknown-linux-gnu` | `linux-arm64` | CI (`ubuntu-24.04-arm`) |
| `x86_64-pc-windows-msvc` | `windows-x64` | CI (`windows-latest`) |
| `aarch64-pc-windows-msvc` | `windows-arm64` | CI (`windows-11-arm`) |

**Intel Mac (`x86_64-apple-darwin`) is not supported.** Upstream `ort`
dropped Intel macOS in rc.11 and the minimum macOS was bumped to 13.4
([changelog](https://github.com/pykeio/ort/releases/tag/v2.0.0-rc.11));
there is no version going forward that ships prebuilt ONNX Runtime
for that target.

### Release flow end-to-end

The pipeline is built so that the registry-push workflows can't fire
before the binary artifacts exist. release-please creates the GitHub
Release as a **draft** (see `"draft": true` in
`release-please-config.json`); the only thing that flips it to
published is `release.yml` after every artifact is attached.

1. Land `feat:` / `fix:` commits on `main`. `release-please.yml`
   opens a PR bumping the workspace version everywhere.
2. **Locally**, while the release-please PR is open (so you know the
   target version):
   ```bash
   scripts/release/build.sh --tag vX.Y.Z --napi --pypi
   git add release_artifacts/vX.Y.Z/
   git commit -m "chore(release): vX.Y.Z macos-arm64 artifacts"
   git push
   ```
   Bakes the macOS-arm64 binaries / `.node` / wheel into main under
   `release_artifacts/vX.Y.Z/` via Git LFS.
3. Merge the release PR. `release-please.yml` tags the merge commit
   (`vX.Y.Z`) and creates a **draft** GitHub Release. No
   `release: published` event yet → registry-push workflows do not
   fire.
4. The tag push triggers `.github/workflows/release.yml`:
   - **build** matrix on four native runners (Linux x64/arm64,
     Windows x64/arm64). No macOS — CI uses what you committed.
   - **commit** appends the matrix artifacts to the same
     `release_artifacts/vX.Y.Z/` and pushes
     `chore(release): vX.Y.Z linux+windows artifacts [skip ci]`.
   - **release** uploads everything in `release_artifacts/vX.Y.Z/`
     (your local macOS plus CI's Linux/Windows) to the draft
     Release, then runs `gh release edit --draft=false` — this is
     the moment `release: published` fires.
5. The published event fires `release-rust.yml`, `release-pypi.yml`,
   and `release-npm.yml`, which download the relevant assets and push
   to crates.io / PyPI / npm / Homebrew.

If `release.yml` fails partway (e.g., a flaky linux-arm64 runner),
the GitHub Release stays in draft state and registry-push workflows
never run — you can re-trigger `release.yml` via `workflow_dispatch`
without rolling back the version bump.

### Local build script

```bash
scripts/release/build.sh                       # loose output (gitignored, sanity check)
scripts/release/build.sh --tag vX.Y.Z          # → release_artifacts/vX.Y.Z/
scripts/release/build.sh --tag vX.Y.Z --napi   # plus .node binding
scripts/release/build.sh --tag vX.Y.Z --pypi   # plus maturin wheel
```

Without `--tag`, output lands directly in `release_artifacts/` as
loose top-level files — gitignored, useful for sanity-checking the
binary without making a release. With `--tag` the output lands in
the versioned subdir the release pipeline expects to find macOS
files in; commit it.

### Git LFS

Once cloned, run `git lfs install` once per machine. Without it,
files in `release_artifacts/<tag>/` will fetch as small text pointers rather
than the real binaries. CI's `actions/checkout@v4` with `lfs: true`
handles it automatically on workflow runs.

## Commit conventions

Conventional commits. `release-please` only bumps on `feat:` / `fix:`. The workspace ships four artifacts on the same version (Rust binary via Homebrew/crates.io, `lethe-memory` wheel on PyPI, `lethe` on npm), so a `feat:` triggers releases everywhere — use it sparingly. Everything else (`chore:`, `docs:`, `refactor:`, `test:`) does not trigger a release. Breaking changes use `feat!:` or `fix!:`.

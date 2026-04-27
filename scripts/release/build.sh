#!/usr/bin/env bash
# Local release build — macOS host only.
#
# Cross-compiling ONNX-Runtime-linked Rust binaries from macOS to
# Linux/Windows is hostile: pyke's prebuilt ORT links against GNU
# libstdc++ (Linux) or MSVC's runtime (Windows), and getting either
# from a macOS host requires standing up a full sysroot. cross's
# Docker images are linux/amd64-only on M-series; cargo-zigbuild
# links against LLVM libc++ which is ABI-incompatible with the
# prebuilt ORT; cargo-xwin's MSVC runtime doesn't ship the C++
# stdlib bits ORT references.
#
# So this script only builds the host platform locally
# (`aarch64-apple-darwin`). The other platform tarballs are built
# in GitHub Actions on native Linux/Windows runners by
# `.github/workflows/release-build.yml` and attached to the GitHub
# Release alongside the macOS tarball you upload.
#
# Usage:
#   scripts/release/build.sh                 # loose build into release_artifacts/
#   scripts/release/build.sh --tag vX.Y.Z    # build into release_artifacts/<tag>/
#                                              (the path the release pipeline expects;
#                                              git add + commit when done)
#   scripts/release/build.sh --napi          # also build the napi .node
#   scripts/release/build.sh --pypi          # also build the maturin wheel
#
# Without `--tag` the output is loose (gitignored, sanity-check only).
# With `--tag vX.Y.Z` the script writes directly into the versioned
# subdir that `release.yml`'s `commit` job will later append the
# Linux/Windows artifacts to. The CI matrix does not rebuild macOS —
# it trusts whatever you committed under `release_artifacts/<tag>/`.
#
# Files produced (suffixed with `--tag` going to subdir):
#   * `lethe-macos-arm64`                       — `lethe` binary
#   * `lethe-claude-code-macos-arm64`           — adapter binary
#   * `lethe-macos-arm64.tar.gz`                — Homebrew tarball (binaries + README + LICENSE)
#   * `lethe-macos-arm64.node`                  — napi-rs binding (--napi)
#   * `lethe_memory-*-cp39-abi3-macosx_*.whl`   — maturin wheel  (--pypi)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
TARGET_DIR="$ROOT_DIR/target"
DIST_DIR="$ROOT_DIR/release_artifacts"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "error: this script is macOS arm64 only."
  echo "       Linux/Windows builds run in GitHub Actions; see"
  echo "       .github/workflows/release-build.yml"
  exit 1
fi

TARGET="aarch64-apple-darwin"
FNAME="macos-arm64"

# Binaries shipped: cargo crate name + binary name
RUST_BINS=(
  "lethe-cli:lethe"
  "lethe-claude-code:lethe-claude-code"
)

BUILD_NAPI="false"
BUILD_PYPI="false"
TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --napi) BUILD_NAPI="true"; shift ;;
    --pypi) BUILD_PYPI="true"; shift ;;
    --tag)  TAG="$2"; shift 2 ;;
    --help|-h)
      sed -n '/^# Usage:/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

# When --tag is passed, write directly to the versioned subdir that
# release.yml expects to find macOS artifacts in. Loose files are
# gitignored; versioned subdir contents go through Git LFS.
if [[ -n "$TAG" ]]; then
  DIST_DIR="$DIST_DIR/$TAG"
fi
mkdir -p "$DIST_DIR"

rustup target add "$TARGET" >/dev/null 2>&1 || true

for entry in "${RUST_BINS[@]}"; do
  crate="${entry%%:*}"
  bin="${entry##*:}"
  echo "Building $bin for $FNAME..."
  RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release --target "$TARGET" -p "$crate"
  src="$TARGET_DIR/$TARGET/release/$bin"
  dst="$DIST_DIR/${bin}-${FNAME}"
  cp "$src" "$dst"
  chmod +x "$dst"
  echo "  -> $dst"
done

# Tarball that matches what the Homebrew formula expects.
stage_root="$(mktemp -d "${TMPDIR:-/tmp}/lethe-pkg.XXXXXX")"
stage_dir="$stage_root/lethe-${FNAME}"
mkdir -p "$stage_dir"
for entry in "${RUST_BINS[@]}"; do
  bin="${entry##*:}"
  cp "$DIST_DIR/${bin}-${FNAME}" "$stage_dir/$bin"
done
cp "$ROOT_DIR/README.md" "$ROOT_DIR/LICENSE" "$stage_dir/" 2>/dev/null || true
(cd "$stage_root" && tar -czf "$DIST_DIR/lethe-${FNAME}.tar.gz" "lethe-${FNAME}")
rm -rf "$stage_root"
echo "  -> $DIST_DIR/lethe-${FNAME}.tar.gz"

if [[ "$BUILD_NAPI" == "true" ]]; then
  echo "Building lethe-node N-API for $FNAME..."
  RUSTFLAGS="-C target-cpu=apple-m1" cargo build --release --target "$TARGET" -p lethe-node
  src="$TARGET_DIR/$TARGET/release/liblethe_node.dylib"
  [[ -f "$src" ]] || src="$TARGET_DIR/$TARGET/release/liblethe.dylib"
  dst="$DIST_DIR/lethe-${FNAME}.node"
  cp "$src" "$dst"
  echo "  -> $dst"
fi

if [[ "$BUILD_PYPI" == "true" ]]; then
  echo "Building maturin wheel..."
  maturin="$(command -v maturin || echo "$ROOT_DIR/.venv/bin/maturin")"
  if [[ ! -x "$maturin" ]]; then
    echo "  error: maturin not installed (uv pip install maturin)"
    exit 1
  fi
  (cd "$ROOT_DIR/crates/lethe-py" && "$maturin" build --release --out "$DIST_DIR" --strip)
fi

echo
echo "release_artifacts/ contents:"
ls -lh "$DIST_DIR"

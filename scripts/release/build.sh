#!/usr/bin/env bash
# Local cross-platform release build for lethe.
#
# Outputs everything needed for a GitHub Release into `dist/`:
#   * `lethe-<os>-<arch>[.exe]`              — `lethe` CLI binary (lethe-cli)
#   * `lethe-claude-code-<os>-<arch>[.exe]`  — Claude Code adapter binary
#   * `lethe-memory-*.whl` / `*.tar.gz`      — PyPI wheel(s) (lethe-py via maturin)
#   * `lethe-<os>-<arch>.node`               — npm native binding (lethe-node via napi-rs)
#   * `lethe-<os>-<arch>.tar.gz`             — packaged tarball matching the
#                                              filename Homebrew expects
#
# Usage:
#   scripts/release/build.sh --native      # host platform only (default)
#   scripts/release/build.sh --macos       # both macOS arches (arm64+x64)
#   scripts/release/build.sh --all         # everything (needs `cross` + `cargo-xwin`)
#   scripts/release/build.sh --target <T>  # single target triple
#   scripts/release/build.sh --napi        # also build N-API .node alongside
#   scripts/release/build.sh --pypi        # also build the maturin wheel
#   scripts/release/build.sh --list        # list known targets
#
# Pattern adapted from the blazediff release script. Designed to run
# on a developer's machine; CI workflows publish what this produces.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
TARGET_DIR="$ROOT_DIR/target"
DIST_DIR="$ROOT_DIR/dist"

# All targets `--all` will attempt.
ALL_TARGETS="aarch64-apple-darwin x86_64-apple-darwin aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu x86_64-pc-windows-gnu aarch64-pc-windows-msvc"

# Binaries the release ships (cargo crate name + binary name, separated by `:`).
RUST_BINS=(
  "lethe-cli:lethe"
  "lethe-claude-code:lethe-claude-code"
)

friendly_name() {
  case "$1" in
    aarch64-apple-darwin)              echo "macos-arm64" ;;
    x86_64-apple-darwin)               echo "macos-x64" ;;
    aarch64-unknown-linux-gnu)         echo "linux-arm64" ;;
    x86_64-unknown-linux-gnu)          echo "linux-x64" ;;
    aarch64-unknown-linux-musl)        echo "linux-arm64-musl" ;;
    x86_64-unknown-linux-musl)         echo "linux-x64-musl" ;;
    x86_64-pc-windows-msvc|x86_64-pc-windows-gnu)   echo "windows-x64" ;;
    aarch64-pc-windows-msvc|aarch64-pc-windows-gnu) echo "windows-arm64" ;;
    *) echo "$1" ;;
  esac
}

rustflags() {
  case "$1" in
    aarch64-apple-darwin)        echo "-C target-cpu=apple-m1" ;;
    x86_64-apple-darwin)         echo "-C target-cpu=haswell" ;;
    aarch64-unknown-linux-*)     echo "-C target-cpu=cortex-a72" ;;
    x86_64-unknown-linux-*|x86_64-pc-windows-*) echo "-C target-cpu=haswell" ;;
    *) echo "" ;;
  esac
}

check_cross() {
  command -v cross >/dev/null 2>&1 || {
    echo "error: 'cross' is required for cross-compilation"
    echo "       cargo install cross"
    exit 1
  }
}

build_rust_target() {
  local target="$1" use_cross="${2:-false}"
  local fname; fname=$(friendly_name "$target")
  local flags;  flags=$(rustflags "$target")
  local ext=""; [[ "$target" == *windows* ]] && ext=".exe"

  for entry in "${RUST_BINS[@]}"; do
    local crate="${entry%%:*}" bin="${entry##*:}"
    echo "Building $bin for $fname ($target)..."

    if [[ "$target" == "aarch64-pc-windows-msvc" ]]; then
      command -v cargo-xwin >/dev/null 2>&1 || {
        echo "  error: cargo-xwin required for $target (cargo install cargo-xwin)"
        return 1
      }
      if [[ -d "/opt/homebrew/opt/llvm/bin" ]]; then
        PATH="/opt/homebrew/opt/llvm/bin:$PATH" RUSTFLAGS="$flags" \
          cargo xwin build --release --target "$target" -p "$crate"
      else
        RUSTFLAGS="$flags" cargo xwin build --release --target "$target" -p "$crate"
      fi
    elif [[ "$use_cross" == "true" ]]; then
      RUSTFLAGS="$flags" cross build --release --target "$target" -p "$crate"
    else
      RUSTFLAGS="$flags" cargo build --release --target "$target" -p "$crate"
    fi

    local src="$TARGET_DIR/$target/release/${bin}${ext}"
    local dst="$DIST_DIR/${bin}-${fname}${ext}"
    cp "$src" "$dst"
    chmod +x "$dst"
    echo "  -> $dst"
  done

  # Tarball the per-target binaries together (matches Homebrew formula
  # expectation). Stage in a temp dir outside $DIST_DIR so we don't
  # collide with the loose `lethe-<friendly>` binary already there.
  local archive_base="lethe-${fname}"
  local stage_root
  stage_root="$(mktemp -d "${TMPDIR:-/tmp}/lethe-pkg.XXXXXX")"
  local stage_dir="$stage_root/$archive_base"
  mkdir -p "$stage_dir"
  for entry in "${RUST_BINS[@]}"; do
    local bin="${entry##*:}"
    cp "$DIST_DIR/${bin}-${fname}${ext}" "$stage_dir/${bin}${ext}"
  done
  cp "$ROOT_DIR/README.md" "$ROOT_DIR/LICENSE" "$stage_dir/" 2>/dev/null || true

  if [[ "$target" == *windows* ]]; then
    (cd "$stage_root" && zip -qr "$DIST_DIR/${archive_base}.zip" "$archive_base")
    echo "  -> $DIST_DIR/${archive_base}.zip"
  else
    (cd "$stage_root" && tar -czf "$DIST_DIR/${archive_base}.tar.gz" "$archive_base")
    echo "  -> $DIST_DIR/${archive_base}.tar.gz"
  fi
  rm -rf "$stage_root"
}

build_native() {
  local current_target
  current_target=$(rustc -vV | grep host | cut -d' ' -f2)
  build_rust_target "$current_target" false
}

build_macos() {
  rustup target add aarch64-apple-darwin x86_64-apple-darwin >/dev/null 2>&1 || true
  build_rust_target "aarch64-apple-darwin" false
  build_rust_target "x86_64-apple-darwin"  false
}

build_napi_target() {
  local target="$1" use_cross="${2:-false}"
  local fname; fname=$(friendly_name "$target")
  local flags;  flags=$(rustflags "$target")

  echo "Building lethe-node N-API for $fname ($target)..."

  if [[ "$use_cross" == "true" ]]; then
    RUSTFLAGS="$flags" cross build --release --target "$target" -p lethe-node
  else
    RUSTFLAGS="$flags" cargo build --release --target "$target" -p lethe-node
  fi

  local lib_ext lib_prefix="lib"
  case "$target" in
    *windows*) lib_ext=".dll"; lib_prefix="" ;;
    *darwin*)  lib_ext=".dylib" ;;
    *)         lib_ext=".so" ;;
  esac

  local src="$TARGET_DIR/$target/release/${lib_prefix}lethe_node${lib_ext}"
  if [[ ! -f "$src" ]]; then
    # napi-rs uses package "name" for the library; check both.
    src="$TARGET_DIR/$target/release/${lib_prefix}lethe${lib_ext}"
  fi
  local dst="$DIST_DIR/lethe-${fname}.node"
  cp "$src" "$dst"
  echo "  -> $dst"
}

build_pypi_native() {
  echo "Building maturin wheel (host)..."
  command -v maturin >/dev/null 2>&1 || {
    command -v "$ROOT_DIR/.venv/bin/maturin" >/dev/null 2>&1 || {
      echo "  error: maturin required (uv pip install maturin)"
      return 1
    }
  }
  local maturin
  maturin="$(command -v maturin || echo "$ROOT_DIR/.venv/bin/maturin")"
  (cd "$ROOT_DIR/crates/lethe-py" && "$maturin" build --release --out "$DIST_DIR" --strip)
}

usage() {
  sed -n '/^# Usage:/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
}

# --- arg parsing ---------------------------------------------------------

MODE="native"
SPECIFIC_TARGET=""
BUILD_NAPI="false"
BUILD_PYPI="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) MODE="target"; SPECIFIC_TARGET="$2"; shift 2 ;;
    --native) MODE="native"; shift ;;
    --macos)  MODE="macos";  shift ;;
    --all)    MODE="all";    shift ;;
    --napi)   BUILD_NAPI="true"; shift ;;
    --pypi)   BUILD_PYPI="true"; shift ;;
    --list)
      echo "Targets:"
      for t in $ALL_TARGETS; do echo "  $t -> $(friendly_name "$t")"; done
      exit 0 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$DIST_DIR"

case "$MODE" in
  native)
    build_native
    [[ "$BUILD_NAPI" == "true" ]] && build_napi_target "$(rustc -vV | grep host | cut -d' ' -f2)" false
    [[ "$BUILD_PYPI" == "true" ]] && build_pypi_native
    ;;
  macos)
    build_macos
    if [[ "$BUILD_NAPI" == "true" ]]; then
      build_napi_target "aarch64-apple-darwin" false || true
      build_napi_target "x86_64-apple-darwin"  false || true
    fi
    [[ "$BUILD_PYPI" == "true" ]] && build_pypi_native
    ;;
  target)
    current=$(rustc -vV | grep host | cut -d' ' -f2)
    if [[ "$SPECIFIC_TARGET" == "$current" ]]; then
      build_rust_target "$SPECIFIC_TARGET" false
      [[ "$BUILD_NAPI" == "true" ]] && build_napi_target "$SPECIFIC_TARGET" false
    elif [[ "$(uname -s)" == "Darwin" && "$SPECIFIC_TARGET" == *apple-darwin* ]]; then
      rustup target add "$SPECIFIC_TARGET" >/dev/null 2>&1 || true
      build_rust_target "$SPECIFIC_TARGET" false
      [[ "$BUILD_NAPI" == "true" ]] && build_napi_target "$SPECIFIC_TARGET" false
    else
      check_cross
      build_rust_target "$SPECIFIC_TARGET" true
      [[ "$BUILD_NAPI" == "true" ]] && build_napi_target "$SPECIFIC_TARGET" true
    fi
    [[ "$BUILD_PYPI" == "true" ]] && build_pypi_native
    ;;
  all)
    current=$(rustc -vV | grep host | cut -d' ' -f2)
    has_cross=false
    command -v cross >/dev/null 2>&1 && has_cross=true
    for t in $ALL_TARGETS; do
      if [[ "$t" == "$current" ]] || [[ "$(uname -s)" == "Darwin" && "$t" == *apple-darwin* ]]; then
        rustup target add "$t" >/dev/null 2>&1 || true
        build_rust_target "$t" false || echo "  skipped $t"
        [[ "$BUILD_NAPI" == "true" ]] && (build_napi_target "$t" false || echo "  skipped napi $t")
      elif [[ "$t" == aarch64-pc-windows-msvc ]]; then
        rustup target add "$t" >/dev/null 2>&1 || true
        build_rust_target "$t" false || echo "  skipped $t (cargo-xwin failed)"
      elif [[ "$has_cross" == "true" ]]; then
        build_rust_target "$t" true || echo "  skipped $t (cross failed)"
        [[ "$BUILD_NAPI" == "true" ]] && (build_napi_target "$t" true || echo "  skipped napi $t")
      else
        echo "  skipping $t (requires cross)"
      fi
    done
    [[ "$BUILD_PYPI" == "true" ]] && build_pypi_native
    ;;
esac

echo
echo "dist/ contents:"
ls -lh "$DIST_DIR" 2>/dev/null || echo "  (empty)"

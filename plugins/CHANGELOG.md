# Changelog

## [0.17.0](https://github.com/teimurjan/lethe/compare/lethe-plugins-v0.16.0...lethe-plugins-v0.17.0) (2026-07-13)


### ⚠ BREAKING CHANGES

* index transcripts directly, offline dedupe, background auto-index

### Features

* Index transcripts directly, offline dedupe, background auto-index ([faeef99](https://github.com/teimurjan/lethe/commit/faeef99a69e177c32a87f01dec8727ee7d368761))

## [0.16.0](https://github.com/teimurjan/lethe/compare/lethe-plugins-v0.15.0...lethe-plugins-v0.16.0) (2026-05-29)


### Features

* **codex:** Marketplace install, drop install.sh ([#43](https://github.com/teimurjan/lethe/issues/43)) ([179eb2c](https://github.com/teimurjan/lethe/commit/179eb2cb8c93d1a2067d2023bc4bcbdc83565b75))

## [0.15.0](https://github.com/teimurjan/lethe/compare/lethe-plugins-v0.14.0...lethe-plugins-v0.15.0) (2026-05-29)


### ⚠ BREAKING CHANGES

* rust port — single `lethe` binary, bindings, polyglot release ([#19](https://github.com/teimurjan/lethe/issues/19))

### Features

* Add --all global search to memory-recall skill ([3ccdafe](https://github.com/teimurjan/lethe/commit/3ccdafe9a2260e997387c0471343a78447c695a2))
* Add `lethe seed` for history backfill ([dc9b263](https://github.com/teimurjan/lethe/commit/dc9b263adbecdcbf459fd064681e963cbe498845))
* Allow multiple expand at once ([a47bcfb](https://github.com/teimurjan/lethe/commit/a47bcfb647aebb59ff9476efa9837d9a3d3d977f))
* Bg codex stop hook + unified worktree paths ([#36](https://github.com/teimurjan/lethe/issues/36)) ([a713a65](https://github.com/teimurjan/lethe/commit/a713a654bd6a6426e7e6c8895e82514acd216efc))
* **codex:** Codex CLI plugin + plugin manifest fix ([#28](https://github.com/teimurjan/lethe/issues/28)) ([0354c5b](https://github.com/teimurjan/lethe/commit/0354c5b5d2ec5a813079155629145425336795b5))
* Concurrent recall + TUI activity pane ([#34](https://github.com/teimurjan/lethe/issues/34)) ([ea330f5](https://github.com/teimurjan/lethe/commit/ea330f5cb87777db74dd069b6da382b1dc234631))
* Lethe tui, split recall skill, lazy session headers ([#7](https://github.com/teimurjan/lethe/issues/7)) ([b5799b3](https://github.com/teimurjan/lethe/commit/b5799b3089c94f48828c98990f417db63d305db0))
* Rust port — single `lethe` binary, bindings, polyglot release ([#19](https://github.com/teimurjan/lethe/issues/19)) ([8f0d1e6](https://github.com/teimurjan/lethe/commit/8f0d1e6af8cd8690cbd0b888cba43a4f72e45c90))
* Ship lethe as a Claude Code plugin ([87f1a40](https://github.com/teimurjan/lethe/commit/87f1a403e10d2549d1380e9cef83c138a9eeb4bf))
* Wire clustered RIF into production, add video demo ([d2b728b](https://github.com/teimurjan/lethe/commit/d2b728b6b90bbf27326e2e705cca0c14ce49d687))


### Bug Fixes

* **plugins:** Drop per-prompt recall hint ([#39](https://github.com/teimurjan/lethe/issues/39)) ([9ad1fb3](https://github.com/teimurjan/lethe/commit/9ad1fb35e2633066277091ef452cbd84af1fea8d))

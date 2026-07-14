# lethe — Oh My Pi plugin

Persistent memory across [Oh My Pi](https://github.com/can1357/oh-my-pi) sessions. Indexes your transcripts directly with hybrid BM25 + dense retrieval and clustered retrieval-induced forgetting.

## Install

```bash
# 1. Install lethe
brew tap teimurjan/lethe && brew install lethe   # or: cargo install lethe-cli

# 2. Add the marketplace and install the plugin
omp plugin marketplace add teimurjan/lethe
omp plugin install lethe@teimurjan
```

Restart `omp` after installation. The plugin installs the `recall` and
`recall-global` skills plus a background freshness hook. It writes nothing into
your repos. lethe reads the JSONL sessions already stored under
`$PI_CODING_AGENT_DIR/sessions` (default `~/.omp/agent/sessions`) and maintains
its rebuildable index under `~/.lethe/`.

Upgrade later with:

```bash
omp plugin marketplace update teimurjan
omp plugin upgrade lethe@teimurjan
```

## How it works

- **`recall`** searches the current project. `lethe search` first indexes any
  new or changed Oh My Pi sessions whose recorded `cwd` matches the project.
- **`recall-global`** searches every project registered by `lethe index`.
- A `before_agent_start` hook launches a throttled, detached
  `lethe index --all && lethe dedupe --all` at most once every 15 minutes. It
  keeps other registered projects fresh without delaying prompt submission.

Tune the hook with `LETHE_REFRESH_INTERVAL` (seconds, default `900`) and
`LETHE_HOME` (state directory, default `~/.lethe`). Set the interval very high
to effectively disable background refresh.

## Requirements

- `lethe` on `PATH`.
- Oh My Pi 16.5 or newer.

## Reference

- Repo: <https://github.com/teimurjan/lethe>

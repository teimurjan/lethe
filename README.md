# lethe

<div align="center">
    <img src="https://raw.githubusercontent.com/teimurjan/lethe/main/assets/logo.png" width="240" />
</div>

> *Λήθη — the ancient Greek personification of forgetfulness.*

A memory store for LLM agents that **gets better the more you use it**. Hybrid BM25 + dense retrieval, cross-encoder reranking, clustered retrieval-induced forgetting (RIF). Every retrieval teaches it which entries are chronic distractors — no fine-tuning, just bookkeeping ([Anderson, 1994](https://psycnet.apa.org/record/1994-29917-001)).

![](https://raw.githubusercontent.com/teimurjan/lethe/main/assets/demo.gif)

## Install

### 1. Install the CLI

Homebrew installs `lethe` plus the Claude Code and Codex transcript adapters:

```bash
brew tap teimurjan/lethe
brew install lethe
```

For the standalone CLI only, use Cargo on any platform with a Rust toolchain:

```bash
cargo install lethe-cli
```

Prebuilt bundles containing the CLI and both adapters are also available from
[GitHub Releases](https://github.com/teimurjan/lethe/releases).

Verify the installation, then index and search the current project's Claude
Code, Codex, and Oh My Pi transcripts:

```bash
lethe --version
lethe index
lethe search "query" --top-k 5
lethe                              # TUI
```

Transcript discovery uses `~/.claude`, `~/.codex`, and `~/.omp/agent` by
default and honors each agent's directory override environment variables.

### 2. Add a coding agent

The CLI works on its own. Install a plugin only if you want `recall` and
`recall-global` available directly inside a coding agent.

#### Claude Code

```
/plugin marketplace add teimurjan/lethe
/plugin install lethe
```

Recall via skills — lethe indexes your Claude Code transcripts directly and writes nothing into your repos. See [plugins/claude-code/README.md](https://github.com/teimurjan/lethe/blob/main/plugins/claude-code/README.md).

#### Codex CLI

```bash
codex plugin marketplace add teimurjan/lethe
```

Then run `codex`, open `/plugins`, and install **lethe** — it wires the `recall` / `recall-global` skills into Codex automatically. See [plugins/codex/README.md](https://github.com/teimurjan/lethe/blob/main/plugins/codex/README.md).

For Cargo installations, add the adapter used by your agent:

```bash
cargo install lethe-claude-code   # Claude Code
cargo install lethe-codex         # Codex CLI
```

Cross-project search and periodic compaction are also available from the CLI:

```bash
lethe search "query" --all --top-k 5
lethe dedupe --all
```

![lethe TUI](https://raw.githubusercontent.com/teimurjan/lethe/main/assets/tui.png)

### Python

```bash
pip install lethe-memory
```

```python
from lethe_memory import MemoryStore

store = MemoryStore("./my_memories")
store.add("I prefer window seats on flights", session_id="trip")
for hit in store.retrieve("travel preferences", k=5):
    print(f"[{hit.score:.1f}] {hit.content}")
store.save()
```

### Node

```bash
npm install @lethe-memory/lethe
```

```typescript
import { MemoryStore } from "@lethe-memory/lethe";
const store = new MemoryStore("./my_memories", { dim: 384 });
await store.add("first entry");
const hits = await store.retrieve("query", { k: 5 });
```

## Benchmark

LongMemEval-S, full 199,509-turn corpus, turn-level NDCG@10:

| Stage | NDCG@10 | Gain |
|---|---|---|
| Hybrid BM25 + vector (RRF) | 0.241 | baseline |
| + cross-encoder reranking | 0.382 | +59% |
| + clustered + gap RIF | 0.342 | +3.4% over RRF-truncation baseline |
| + LLM enrichment, covered queries | 0.473 | +20–25% |

**Scope.** RIF targets chronic-false-positives in long-term conversational memory. On NFCorpus (medical IR) it regresses — workload-specific by design. Full methodology + checkpoints in [BENCHMARKS.md](https://github.com/teimurjan/lethe/blob/main/BENCHMARKS.md).

## How it works

- [ARCHITECTURE.md](https://github.com/teimurjan/lethe/blob/main/ARCHITECTURE.md) — pipeline, RIF formula, storage layers.
- [RESEARCH_JOURNEY.md](https://github.com/teimurjan/lethe/blob/main/RESEARCH_JOURNEY.md) — 18 checkpoints from biology-inspired mutation through RIF to enrichment.

## License

MIT

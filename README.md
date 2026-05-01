# lethe

<div align="center">
    <img src="https://raw.githubusercontent.com/teimurjan/lethe/main/assets/logo.png" width="240" />
</div>

> *Λήθη — the ancient Greek personification of forgetfulness.*

A memory store for LLM agents that **gets better the more you use it**. Hybrid BM25 + dense retrieval, cross-encoder reranking, clustered retrieval-induced forgetting (RIF). Every retrieval teaches it which entries are chronic distractors — no fine-tuning, just bookkeeping ([Anderson, 1994](https://psycnet.apa.org/record/1994-29917-001)).

![](https://raw.githubusercontent.com/teimurjan/lethe/main/assets/demo.gif)

## Install

### Claude Code

```
/plugin marketplace add teimurjan/lethe
/plugin install lethe
```

Per-project memory in `.lethe/memory/YYYY-MM-DD.md`, recall via skill + session-start hook. See [plugins/claude-code/README.md](https://github.com/teimurjan/lethe/blob/main/plugins/claude-code/README.md).

### Codex CLI

```bash
brew tap teimurjan/lethe && brew install lethe
git clone https://github.com/teimurjan/lethe /tmp/lethe
bash /tmp/lethe/plugins/codex/install.sh --auto-config
```

See [plugins/codex/README.md](https://github.com/teimurjan/lethe/blob/main/plugins/codex/README.md).

### CLI

```bash
brew install teimurjan/lethe/lethe       # or: cargo install lethe-cli

lethe                                    # TUI
lethe search "query" --top-k 5           # one project
lethe search "query" --all --top-k 5     # all registered projects
lethe seed --days 7                      # backfill from past Claude Code / Codex transcripts
```

![lethe TUI](https://raw.githubusercontent.com/teimurjan/lethe/main/assets/tui.gif)

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

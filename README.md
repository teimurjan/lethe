# lethe

<div align="center">
    <img src="https://raw.githubusercontent.com/teimurjan/lethe/main/assets/logo.png" width="300" height="300" />
</div>

> *Λήθη: the ancient Greek personification of forgetfulness, and one of the five rivers of the underworld.*

A memory store for LLM agents that **gets better the more you use it**. Hybrid BM25 + dense retrieval, cross-encoder reranking, clustered retrieval-induced forgetting (RIF), and optional LLM enrichment at write time.

Most memory tools are static caches - you put strings in, you get strings back by similarity, and the retrieval function never changes. lethe is different: every retrieval teaches it which entries are chronic distractors for which kinds of queries, and it quietly suppresses them over time. No fine-tuning, no extra LLM calls - just bookkeeping inspired by how human memory actually works ([Anderson, 1994](https://psycnet.apa.org/record/1994-29917-001)).

## Install and quick start

### As a Claude Code plugin (recommended for daily use)

```
/plugin marketplace add teimurjan/lethe
/plugin install lethe
```

What happens after install:

- Every session is summarized into `.lethe/memory/YYYY-MM-DD.md` per project. Plain markdown, edit by hand if you want.
- Claude sees recent memory at session start and calls the `recall` skill when a past session in this project would help.
- For cross-repo context, the `recall-global` skill searches every registered project at once (uses `lethe search --all` under the hood).

Update: `brew upgrade lethe && /reload-plugins`

See [plugins/claude-code/README.md](https://github.com/teimurjan/lethe/blob/main/plugins/claude-code/README.md) for the full hook table, config knobs, and debugging.

### As a Codex CLI plugin

```bash
brew tap teimurjan/lethe && brew install lethe   # or: cargo install lethe-cli
git clone https://github.com/teimurjan/lethe /tmp/lethe
bash /tmp/lethe/plugins/codex/install.sh --auto-config
```

The installer copies hooks + skills to `~/.codex/lethe/` and patches `~/.codex/config.toml` between sentinel markers (re-runs replace the previous block). See [plugins/codex/README.md](https://github.com/teimurjan/lethe/blob/main/plugins/codex/README.md) for the hook table and current limitations (no turn summarization yet — Codex transcript format pending).

### As a CLI

```bash
brew tap teimurjan/lethe                  # one-time
brew install lethe                        # macOS / Linuxbrew
# or
cargo install lethe-cli                   # any platform with a Rust toolchain
# or download a tarball from
# https://github.com/teimurjan/lethe/releases

lethe --version
lethe                                            # no args → opens TUI
lethe index                                      # reindex .lethe/memory
lethe search "your query" --top-k 5              # single project
lethe search "your query" --all --top-k 5        # all registered projects
lethe projects list
lethe status
```

#### Interactive TUI

![lethe TUI](https://raw.githubusercontent.com/teimurjan/lethe/main/assets/tui.gif)

`lethe` with no subcommand opens the TUI when stdout is a terminal; pass `lethe tui` to force it from a script. Keys inside: `↑↓` nav, `⏎` search/open, `Esc` back, `Ctrl+Q` quit. Type anywhere to jump focus to the search box.

### As a Python binding

```bash
pip install lethe-memory
```

```python
from lethe_memory import MemoryStore

store = MemoryStore("./my_memories")  # bi-/cross-encoders default to MiniLM

store.add("I prefer window seats on flights", session_id="trip")
store.add("My wife needs aisle seats", session_id="trip")
store.add("I work at Google as a software engineer", session_id="work")

for hit in store.retrieve("What are my travel preferences?", k=5):
    print(f"  [{hit.score:.1f}] {hit.content}")

store.save()
```

### As a Node binding

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

Numbers on the full 199,509-turn LongMemEval S corpus, **turn-level retrieval, NDCG@10, no leakage**. Most memory-tool benchmarks use ~50 sessions at session granularity - a ~2000× easier task. Those 99% numbers don't translate to this setup.

| Stage | NDCG@10 | Relative gain | notes |
|---|---|---|---|
| Hybrid BM25 + vector (RRF) | 0.241 | baseline | basic retrieval (most popular) |
| + cross-encoder reranking | 0.382 | +59% | semantic reranking on the hybrid pool |
| + clustered+gap RIF (checkpoint 13) | 0.342¹ | +3.4%¹ | retrieval-induced forgetting (30 clusters, gap formula) |
| + LLM enrichment, on covered queries | 0.473² | +20–25%² | Haiku write-time enrichment on the 75 queries where the answer turn was enriched |

¹ Measured on the RIF benchmark pipeline (RRF-truncation, 500-query full eval, 5000-step burn-in; [`legacy/benchmarks/run_rif_clustered.py`](https://github.com/teimurjan/lethe/blob/main/legacy/benchmarks/run_rif_clustered.py)). Matched no-RIF baseline on this pipeline = **0.331**, so RIF delivers +3.4% / +1.1pp NDCG and +4.9% Recall@30. The +3.4% is smaller than the +6.5% measured on the previous `lower().split()` tokenizer; the stronger BM25 baseline leaves RIF less signal to recover, but the mechanism is still net-positive. Absolute NDCG under RIF moved from 0.315 to 0.342 with the tokenizer upgrade. See [BENCHMARKS.md](https://github.com/teimurjan/lethe/blob/main/BENCHMARKS.md) for the live numbers.

² Single measurement: LongMemEval S, Claude **Haiku** write-time enrichment (gist + anticipated queries + entities + temporal markers concatenated to each chunk before embed/index), evaluated on the 75/500 queries whose answer-relevant turn was in the enriched subset. Covered-bucket NDCG@10 moved 0.390 (RIF alone) → **0.473** (+21.3% rel, +8.3pp abs); diluted across all 500 queries that's +1.2pp. Measured on the previous BM25 tokenizer; on the current regex tokenizer the lift is expected to land somewhere in the 15-25% band, since better BM25 closes some of the vocabulary-mismatch gap enrichment was filling (same "smaller relative gain on a stronger baseline" effect we saw with RIF). Numbers will also vary with model choice (Haiku vs Sonnet vs Opus), corpus domain, and how well the base retriever already covers the vocabulary. Raw table: [BENCHMARKS_RIF_ENRICHED.md](https://github.com/teimurjan/lethe/blob/main/legacy/benchmarks/results/BENCHMARKS_RIF_ENRICHED.md).

**Scope.** The RIF gain is workload-specific. The mechanism targets the chronic-false-positive pattern in a single user's long-term conversation memory. On NFCorpus (a non-conversational medical IR benchmark) it doesn't transfer: three of four variants significantly regress. We diagnose this in the arXiv paper (corpus saturation + workload mismatch) and scope the claim to long-term conversational memory. Use lethe for what it's good at; don't expect it to help on general ad-hoc retrieval.

Full methodology in [BENCHMARKS.md](https://github.com/teimurjan/lethe/blob/main/BENCHMARKS.md). 18 checkpoints (11 failed or null) in [RESEARCH_JOURNEY.md](https://github.com/teimurjan/lethe/blob/main/RESEARCH_JOURNEY.md). Statistical rigor and the NFCorpus replication in [paper.tex](https://github.com/teimurjan/lethe/blob/main/paper.tex).

![](https://raw.githubusercontent.com/teimurjan/lethe/main/assets/demo.gif)

## How it works

[ARCHITECTURE.md](https://github.com/teimurjan/lethe/blob/main/ARCHITECTURE.md) - pipeline diagram, RIF formula, storage layers, entry lifecycle, cross-project search.

[RESEARCH_JOURNEY.md](https://github.com/teimurjan/lethe/blob/main/RESEARCH_JOURNEY.md) - 18 checkpoints from biology-inspired mutation (all failed) through cognitive-science RIF (+6.5% on conversational memory, does not generalize to ad-hoc IR) to LLM enrichment (+21% on covered queries) and statistical rigor with a second-dataset replication.

## License

MIT

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

Update: `uv tool install --upgrade lethe-memory && /reload-plugins`

See [plugins/claude-code/README.md](plugins/claude-code/README.md) for the full hook table, config knobs, and debugging.

### As a CLI

```bash
uv tool install lethe-memory
lethe --version

lethe index                                     # reindex .lethe/memory
lethe search "your query" --top-k 5             # single project
lethe search "your query" --all --top-k 5       # all registered projects
lethe projects list
lethe status
```

#### Interactive TUI

```bash
uv tool install --force 'lethe-memory[tui]'
# or, if lethe is already installed as a uv tool:
uv tool install --force --reinstall --with textual lethe-memory

lethe tui
```

`uv tool install` does not read `[project.optional-dependencies]` from extras syntax unless quoted; the `--with textual` form is the reliable fallback. Keys inside the TUI: `↑↓` nav, `⏎` search/open, `Esc` back, `Ctrl+Q` quit. Type anywhere to jump focus to the search box.

### As a Python library

```bash
pip install lethe-memory
```

```python
from lethe import MemoryStore
from sentence_transformers import SentenceTransformer, CrossEncoder

store = MemoryStore(
    "./my_memories",
    bi_encoder=SentenceTransformer("all-MiniLM-L6-v2"),
    cross_encoder=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2"),
)

store.add("I prefer window seats on flights", session_id="trip")
store.add("My wife needs aisle seats", session_id="trip")
store.add("I work at Google as a software engineer", session_id="work")

results = store.retrieve("What are my travel preferences?", k=5)
for entry_id, content, score in results:
    print(f"  [{score:.1f}] {content}")

store.save()
store.close()
```

## Benchmark

Numbers on the full 199,509-turn LongMemEval S corpus, **turn-level retrieval, NDCG@10, no leakage**. Most memory-tool benchmarks use ~50 sessions at session granularity - a ~2000× easier task. Those 99% numbers don't translate to this setup.

| Stage | NDCG@10 | notes |
|---|---|---|
| Hybrid BM25 + vector (RRF) | 0.217 | basic retrieval (most popular) |
| + cross-encoder reranking | 0.293 | +35% from semantic reranking |
| + clustered+gap RIF (checkpoint 13) | **0.312** | +6.5% from retrieval-induced forgetting (paired permutation p<0.002, 95% CI excludes zero) |
| + LLM enrichment, on covered queries | **0.473** | +21% on the 75 queries where the answer turn was Haiku-enriched |

**Scope.** The RIF gain is workload-specific. The mechanism targets the chronic-false-positive pattern in a single user's long-term conversation memory. On NFCorpus (a non-conversational medical IR benchmark) it doesn't transfer: three of four variants significantly regress. We diagnose this in the arXiv paper (corpus saturation + workload mismatch) and scope the claim to long-term conversational memory. Use lethe for what it's good at; don't expect it to help on general ad-hoc retrieval.

Full methodology in [BENCHMARKS.md](BENCHMARKS.md). 18 checkpoints (11 failed or null) in [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md). Statistical rigor and the NFCorpus replication in [arxiv/paper.tex](arxiv/paper.tex).

![](https://raw.githubusercontent.com/teimurjan/lethe/main/assets/demo.gif)

## How it works

[ARCHITECTURE.md](ARCHITECTURE.md) - pipeline diagram, RIF formula, storage layers, entry lifecycle, cross-project search.

[RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md) - 18 checkpoints from biology-inspired mutation (all failed) through cognitive-science RIF (+6.5% on conversational memory, does not generalize to ad-hoc IR) to LLM enrichment (+21% on covered queries) and statistical rigor with a second-dataset replication.

## License

MIT

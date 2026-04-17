# lethe

<div align="center">
    <img src="https://raw.githubusercontent.com/teimurjan/lethe/main/assets/logo.png" width="300" height="300" />
</div>

> *Λήθη: the ancient Greek personification of forgetfulness, and one of the five rivers of the underworld.*

A memory store for LLM agents that **gets better the more you use it**. Hybrid BM25 + dense retrieval, cross-encoder reranking, clustered retrieval-induced forgetting (RIF), and optional LLM enrichment at write time.

Most memory tools are static caches - you put strings in, you get strings back by similarity, and the retrieval function never changes. lethe is different: every retrieval teaches it which entries are chronic distractors for which kinds of queries, and it quietly suppresses them over time. No fine-tuning, no extra LLM calls - just bookkeeping inspired by how human memory actually works ([Anderson, 1994](https://psycnet.apa.org/record/1994-29917-001)).

## Benchmark

Numbers on the full 199,509-turn LongMemEval S corpus, **turn-level retrieval, NDCG@10, no leakage**. Most memory-tool benchmarks use ~50 sessions at session granularity - a ~2000× easier task. Those 99% numbers don't translate to this setup.

| Stage | NDCG@10 | notes |
|---|---|---|
| Hybrid BM25 + vector (RRF) | 0.217 | basic retrieval (most popular) |
| + cross-encoder reranking | 0.293 | +35% from semantic reranking |
| + clustered+gap RIF (checkpoint 13) | **0.312** | +6.5% from retrieval-induced forgetting |
| + LLM enrichment, on covered queries | **0.473** | +21% on the 75 queries where the answer turn was Haiku-enriched |

Full methodology in [BENCHMARKS.md](BENCHMARKS.md). 17 checkpoints (10 failed) in [RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md).

![](https://raw.githubusercontent.com/teimurjan/lethe/main/assets/demo.gif)

## Install and quick start

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

As a CLI: `uv tool install lethe-memory && lethe --version`

As a Claude Code plugin: `/plugin marketplace add teimurjan/lethe && /plugin install lethe`

To update: `uv tool install --upgrade lethe-memory && /reload-plugins`

See [plugins/claude-code/README.md](plugins/claude-code/README.md) for plugin details.

## How it works

[ARCHITECTURE.md](ARCHITECTURE.md) - pipeline diagram, RIF formula, storage layers, entry lifecycle, cross-project search.

[RESEARCH_JOURNEY.md](RESEARCH_JOURNEY.md) - 17 checkpoints from biology-inspired mutation (all failed) through cognitive-science RIF (+6.5%) to LLM enrichment (+21% on covered queries).

## License

MIT

# gc_mutation/

Archived research code from checkpoints 1-10 of the lethe research journey: the original germinal-center-inspired mutation experiments that didn't survive contact with real retrieval benchmarks. Kept for reproducibility, not for production use.

## What's here

| File | What it is |
|------|------------|
| `store.py` | `GCMemoryStore` — the older, fuller implementation with graph + rescue index + mutation |
| `graph.py` | Co-relevance graph: edges between co-retrieved entries, decay, expansion |
| `rescue_index.py` | Learned query → entry cache for FAISS misses |
| `baselines.py` | Static and NoGraph control variants for research A/B |
| `segmentation.py` | Text-level mutation: split / merge entries |
| `config.py` | Hyperparameters for the research experiments (not used by production `MemoryStore`) |
| `run_experiment.py` | v4 experiment harness (Static vs MLP-adapter vs segmentation on BEIR/LongMemEval) |
| `analyze.py` | Results visualization: plots and summary tables from `run_*.json` |

## Status

None of this is reachable from `legacy/lethe.MemoryStore`. The research thread concluded that the GC mutation loop doesn't improve retrieval quality (see `RESEARCH_JOURNEY.md` checkpoints 1-10). The genuinely productive threads — RIF (checkpoints 11-13) and LLM enrichment (checkpoint 17) — live in `legacy/lethe/`.

## Running

These modules depend on production primitives like `lethe.entry.MemoryEntry`. Make sure the repo is installed in editable mode:

```bash
uv pip install -e .
uv run python gc_mutation/run_experiment.py --help
```

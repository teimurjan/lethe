# gc-memory

A research prototype testing whether an adaptive mutation-selection loop over embeddings improves retrieval quality compared to a static baseline — inspired by the germinal center mechanism from adaptive immunity.

## Thesis

A vector memory store where entries undergo query-driven mutation with an affinity-adaptive rate (low-confidence entries mutate aggressively, high-confidence entries barely move) plus competitive selection against the query, should produce measurably better retrieval quality over time than a static store — without requiring external feedback signals like clicks or rewards.

## Success criteria

The experiment shows positive signal **if and only if**:

1. GC completes without tripping circuit breakers
2. GC's final NDCG@10 exceeds Static's by >= 3%
3. GC's final NDCG@10 exceeds Random's by >= 1.5%
4. GC's NDCG@10 curve is monotonically non-decreasing in a rolling 2000-step window after step 2000

If any of these fail, the result is negative. A negative result is a valid result.

## Falsification conditions

The experiment is designed to detect three critical failure modes:

- **Degenerate convergence**: all embeddings collapse toward each other (circuit breaker: diversity drops below 90% of initial)
- **Anchor drift**: mutated embeddings walk away from the content they index (circuit breaker: mean drift exceeds 0.25)
- **No-signal result**: GC ties with random-mutation control, meaning the adaptive rate adds nothing

## How to run

```bash
git clone <repo> && cd gc-memory
uv venv && uv pip install -e .
uv run python experiments/data_prep.py      # downloads NFCorpus, ~5 min
uv run python experiments/run_experiment.py  # runs all three arms, ~30-60 sec
uv run python experiments/analyze.py         # produces plots and summary table
```

## Three experimental arms

| Arm | Mutation | Decay | Tier transitions |
|-----|----------|-------|------------------|
| **Static** | None | None | None |
| **Random** | Fixed sigma=0.025 | Yes | Yes |
| **GC** | Adaptive sigma (affinity-dependent) | Yes | Yes |

## Stack

- Python 3.11+, managed with `uv`
- FAISS (faiss-cpu) for ANN
- sentence-transformers with all-MiniLM-L6-v2 (384-dim)
- numpy for mutation math
- SQLite via stdlib sqlite3 for metadata
- BEIR with NFCorpus for benchmark data
- matplotlib for result plots
- pytest for tests

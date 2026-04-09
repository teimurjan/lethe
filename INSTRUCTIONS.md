## Project: `gc-memory`

Build a research prototype of a self-refining memory store for LLM assistants, inspired by the germinal center mechanism from adaptive immunity. The goal is **not** a production library — it's a falsifiable experiment that answers: "Does an adaptive mutation-selection loop over embeddings improve retrieval quality on a stale memory workload compared to a static baseline?"

Ship something I can run end-to-end in under 60 seconds on a laptop and get a clear yes/no answer from.

## Core thesis being tested

A vector memory store where entries undergo query-driven mutation with an affinity-adaptive rate (low-confidence entries mutate aggressively, high-confidence entries barely move) plus competitive selection against the query, should produce measurably better retrieval quality over time than a static store — **without** requiring external feedback signals like clicks or rewards.

Critical failure modes to detect:
1. **Degenerate convergence** — all embeddings collapse toward each other
2. **Anchor drift** — mutated embeddings walk away from the content they index
3. **No-signal result** — GC ties with random-mutation control, meaning the adaptive rate adds nothing

The experiment must be designed to catch these, not hide them.

## Stack

- **Python 3.11+**, managed with `uv` (fast, reproducible)
- **FAISS** (`faiss-cpu`) for ANN — in-memory, rebuilt cheaply
- **sentence-transformers** with `all-MiniLM-L6-v2` (384-dim, fast, no API)
- **numpy** for the mutation math
- **SQLite** via stdlib `sqlite3` for metadata (no ORM, raw SQL)
- **BEIR** for benchmark datasets — start with `nfcorpus` (small, has near-duplicates)
- **matplotlib** for result plots — no seaborn, no plotly
- **pytest** for tests

No frameworks. No async. No config files beyond a single `config.py` with dataclass constants. One script, one run, one result.

## Repository layout

```
gc-memory/
├── pyproject.toml
├── README.md
├── src/gc_memory/
│   ├── __init__.py
│   ├── config.py          # all hyperparameters as a frozen dataclass
│   ├── entry.py           # MemoryEntry dataclass + tier enum
│   ├── store.py           # GCMemoryStore — the main class
│   ├── mutation.py        # pure functions: mutate(), select_best()
│   ├── metrics.py         # health metrics: diversity, anchor_drift, tier_distribution
│   └── baselines.py       # StaticStore, RandomMutationStore for comparison
├── experiments/
│   ├── run_experiment.py  # the headline script: runs all 3 arms, saves results
│   ├── analyze.py         # loads results JSON, produces plots
│   └── data_prep.py       # downloads and prepares NFCorpus
├── tests/
│   ├── test_mutation.py
│   ├── test_store.py
│   └── test_metrics.py
└── results/               # gitignored, populated by experiments
```

## Data model

In `entry.py`:

```python
from dataclasses import dataclass
from enum import Enum
import numpy as np

class Tier(Enum):
    NAIVE = "naive"
    GC = "gc"
    MEMORY = "memory"
    APOPTOTIC = "apoptotic"

@dataclass
class MemoryEntry:
    id: str
    content: str                    # ground truth, NEVER mutated
    embedding: np.ndarray           # unit-normalized, shape (384,)
    original_embedding: np.ndarray  # frozen at creation, used for anchor check
    affinity: float                 # EMA in [0, 1], init 0.5
    retrieval_count: int
    generation: int                 # mutation lineage depth
    last_retrieved_step: int        # step counter, not wall clock
    tier: Tier
```

Use step counters, not wall-clock time, for the experiment. Deterministic and reproducible.

## Core algorithms

Implement these exactly. The math matters — don't "improve" it without telling me.

### Mutation (in `mutation.py`)

```
sigma_i = sigma_0 * (1 - affinity_i) ** gamma
epsilon ~ Normal(0, sigma_i^2 * I)
mutant = (embedding + epsilon) / ||embedding + epsilon||
```

Generate `n_mutants = 5` candidates per GC entry per query.

### Selection

```
best_mutant = argmax over mutants of cosine(query, mutant)
accept if:
    cosine(query, best_mutant) - cosine(query, original) > delta
    AND cosine(best_mutant, original_embedding) >= theta_anchor
```

The anchor check is the single most important safety rail. Without it, mutations drift into fantasy-land. **Never skip it.** If you're tempted to skip it because "the diversity metric should catch it," you're wrong — diversity catches collective collapse, anchor catches individual drift.

### Affinity update

```
affinity_i = (1 - alpha) * affinity_i + alpha * cosine(query, embedding_i)
```

Applied to every retrieved entry after each query, with `alpha = 0.2`.

### Retrieval with tier weighting

```
effective_score = cosine(query, embedding) * tier_weight
tier_weight: naive=1.0, gc=1.0, memory=1.15, apoptotic=0.0
```

Retrieve top-10 by effective score. Apoptotic entries are never returned.

### Tier transitions

Checked after each query:
- `naive → gc` when `retrieval_count >= 3`
- `gc → memory` when `affinity >= 0.75 AND generation >= 5`
- `any → apoptotic` when `affinity < 0.15 AND (current_step - last_retrieved_step) > 1000`

Memory-tier entries are exempt from decay.

### Time decay (applied periodically, every 100 steps)

```
for entries not in memory tier and not retrieved this batch:
    affinity *= exp(-lambda * delta_steps / 100)
```

With `lambda = 0.01`.

## Configuration

Single frozen dataclass in `config.py`:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Retrieval
    k: int = 10
    tier_weight_memory: float = 1.15
    # Affinity
    alpha: float = 0.2
    # Mutation
    sigma_0: float = 0.05
    gamma: float = 2.0
    n_mutants: int = 5
    delta: float = 0.01
    theta_anchor: float = 0.7
    # Tier transitions
    promote_naive_threshold: int = 3
    promote_memory_affinity: float = 0.75
    promote_memory_generation: int = 5
    apoptosis_affinity: float = 0.15
    apoptosis_idle_steps: int = 1000
    # Decay
    lambda_decay: float = 0.01
    decay_interval: int = 100
    # Experiment
    n_queries: int = 10_000
    hot_set_fraction: float = 0.2
    hot_set_probability: float = 0.7
    random_seed: int = 42
```

Every magic number lives here. If you find yourself hardcoding a number elsewhere, move it here.

## The experiment

In `experiments/run_experiment.py` — this is the file I'll actually run.

### Setup
1. Download NFCorpus via BEIR (cache in `./data/`)
2. Embed all corpus documents with `all-MiniLM-L6-v2`, normalize to unit vectors
3. Build three stores with identical initial state:
   - **Static**: no mutation, no decay, no tier transitions (baseline)
   - **Random**: mutation with fixed `sigma = 0.025` regardless of affinity (control — tests whether the adaptive rate matters)
   - **GC**: full algorithm as specified above

### Query schedule
Generate 10,000 queries from NFCorpus's query set with Zipfian-like repetition: 20% of queries form a "hot set" that accounts for 70% of traffic. This simulates real usage where a few topics dominate. Seed the RNG from config.

### Per-query loop (same for all three arms)
```
for step in range(n_queries):
    query_text = sample_query(step)
    query_embedding = embed(query_text)
    retrieved = store.retrieve(query_embedding, k=10)
    store.update_after_retrieval(query_embedding, retrieved, step)
    if step % 100 == 0:
        run_decay(store, step)
    if step % 500 == 0:
        log_metrics(store, step)
        check_circuit_breakers(store)  # see below
```

### Metrics logged every 500 steps
- **NDCG@10** against ground-truth relevance judgments from NFCorpus
- **Recall@10**
- **Mean diversity** (average pairwise cosine distance, 1000-pair sample)
- **Anchor drift** (mean `1 - cos(embedding, original_embedding)` over 100 random entries)
- **Tier distribution** (count per tier)
- **Mean generation** (across non-naive entries)

### Circuit breakers (halt the experiment, don't hide the failure)
- Diversity drops below 90% of initial value → stop, log "DEGENERATE_CONVERGENCE"
- Anchor drift exceeds 0.25 → stop, log "ANCHOR_DRIFT_EXCEEDED"
- Mean affinity of GC tier drops below 0.1 → stop, log "AFFINITY_COLLAPSE"

When a circuit breaker fires, save all metrics up to that point and exit with a clear message. Don't try to recover. A tripped breaker **is** the result.

### Output
Save to `results/run_<timestamp>.json`:
```json
{
  "config": {...},
  "seed": 42,
  "completed": true|false,
  "halt_reason": null | "DEGENERATE_CONVERGENCE" | ...,
  "arms": {
    "static":  {"metrics_by_step": [...]},
    "random":  {"metrics_by_step": [...]},
    "gc":      {"metrics_by_step": [...]}
  }
}
```

### Analysis (`experiments/analyze.py`)
Loads the latest JSON and produces four plots saved to `results/plots/`:
1. NDCG@10 over time, three lines
2. Diversity over time, three lines
3. Anchor drift over time (only GC arm — static and random don't mutate the way GC does)
4. Tier distribution over time (only GC arm, stacked area chart)

Print a summary table at the end:
```
Arm       Final NDCG@10    Delta vs Static    Circuit Breaker
static    0.XXX            —                  —
random    0.XXX            +X.X%              —
gc        0.XXX            +X.X%              —
```

## Success criteria (write these in the README, commit them before running)

The experiment is considered to show positive signal **if and only if**:

1. GC completes without tripping circuit breakers
2. GC's final NDCG@10 exceeds Static's by ≥3%
3. GC's final NDCG@10 exceeds Random's by ≥1.5%
4. GC's NDCG@10 curve is monotonically non-decreasing in a rolling 2000-step window after step 2000

If any of these fail, the result is negative. **Do not reinterpret the numbers to find a positive story.** A negative result is a valid result and should be written up as such.

## Testing

Unit tests in `tests/` covering:
- `test_mutation.py`: mutation preserves unit norm; adaptive rate produces smaller perturbations for high affinity; `n_mutants` candidates are actually different
- `test_store.py`: retrieval returns top-k by effective score; tier transitions fire at correct thresholds; apoptotic entries are excluded; anchor constraint rejects drifted mutations
- `test_metrics.py`: diversity computation matches hand-calculated value on a 5-entry fixture; anchor drift is zero for unmutated store

Target coverage: the core algorithms in `mutation.py` and `store.py` should be 100% covered. Don't bother testing `run_experiment.py` — it's a script, not a library.

## What NOT to build

These are explicitly out of scope for v1. If Claude Code is tempted to add any of them, it should stop and ask first:

- Async, multiprocessing, or GPU support
- A REST API, CLI tool, or web UI
- Pluggable backends (no abstract base classes for "VectorStore")
- Pluggable embedding models
- Configuration via YAML/TOML files
- Docker, CI, or deployment scaffolding
- Logging frameworks (use `print` to stdout, it's fine)
- Progress bars fancier than what `tqdm` gives you
- The full DZ/LZ two-compartment architecture from the biology — we're testing the simplified version first
- Conflict detection between contradictory memories — that's a v2 feature, needs LLM calls, adds cost and complexity
- Memory-tier promotion requiring generation lineage trees — a single `generation` counter is enough

Build the smallest thing that answers the question. If it works, v2 can be more ambitious. If it doesn't work, the missing features wouldn't have saved it.

## Working style

- **Commit after each working piece.** `entry.py` + its tests → commit. `mutation.py` + its tests → commit. Don't ball everything into one mega-commit.
- **Run the tests after each change.** If tests break, fix them before moving on.
- **Use `uv` for everything.** `uv venv`, `uv pip install -e .`, `uv run pytest`, `uv run python experiments/run_experiment.py`.
- **Type hints everywhere.** `from __future__ import annotations` at the top of every file. `mypy --strict` should pass.
- **Docstrings only where non-obvious.** The formulas above are the spec; the docstring should cite which step of the spec the function implements, not re-explain the math.
- **Before writing any file, check if the skill at `/mnt/skills/public/frontend-design/SKILL.md` or other relevant skills apply.** This is a Python research project so probably not, but check.
- **No dependencies beyond what's listed in the Stack section without asking.** If you think you need `scipy` or `pandas` or `polars`, stop and justify it.

## Deliverables

When done, I should be able to:
1. `git clone <repo> && cd gc-memory && uv venv && uv pip install -e .`
2. `uv run python experiments/data_prep.py` (downloads NFCorpus, ~5 min)
3. `uv run python experiments/run_experiment.py` (runs all three arms, ~30-60 sec after data prep)
4. `uv run python experiments/analyze.py` (produces plots and summary table)
5. Read the summary and immediately know whether the thesis survived.

The README should document exactly this flow plus the success criteria plus the falsification conditions. Anyone reading the README should understand what's being tested, how to run it, and how to interpret the result — without needing to read the code.

## First step

Before writing any code, create the repo structure, `pyproject.toml`, `README.md` with the thesis and success criteria, and a minimal `config.py`. Commit that as the initial scaffold. Then implement `entry.py` + its tests and commit. Then `mutation.py` + its tests and commit. Then the rest, in whatever order makes sense, with tests and commits at each step.

Ask me before:
- Adding any dependency not in the Stack list
- Deviating from the formulas in the Core Algorithms section
- Changing any value in `config.py` (I picked those deliberately)
- Adding anything from the "What NOT to build" list
- Starting the full experiment run if any test is failing

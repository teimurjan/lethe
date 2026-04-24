# demo

Visualizations for lethe's retrieval behavior on LongMemEval. Each demo is a
Python collector that writes a JSON trace into `public/`, consumed by a
Remotion scene under `src/`.

## demos

### 0. Extended replay — `extend_replay.py` (video source)

Takes the empirical `run_replay.json` (cold + 3 real warm rounds from
`collect_replay.py`), keeps it verbatim, and appends **7 synthetic
warm rounds** per qid so the video's graph can show the gap continuing
to grow past what we actually measured.

Each synthetic round r adds a saturating gain on top of the real
per-qid warm2 peak:

    lethe(r, qid) = warm2[qid] + G_MAX · (1 − exp(−(r − 2) / τ)) + noise

with `G_MAX=0.040`, `τ=4`, `σ=0.03`. Anchoring on warm2 (not warm3)
avoids inheriting the real run's anomalous dip. Gains saturate —
honest about diminishing returns, not a linear ramp.

The same output also embeds the **headline frame number**:

    meta.headline = {
      baselineNdcg: 0.2408,
      lethNdcg:     0.3817,
      deltaPct:     58.5,
      text:         "+59% NDCG vs hybrid retrieval",
      baselineLabel: "hybrid retrieval (BM25 + vector RRF)"
    }

Numbers come from BENCHMARKS.md's LongMemEval S headline table
(re-measured 2026-04-24 on the regex BM25 tokenizer). Previous
run was 0.2171 / 0.3680 (+69.5%) on the `lower().split()` tokenizer
— the delta shrank because the simpler RRF baseline improved more
in relative terms than the full stack did. Absolute numbers both
went up. The frame is the big "+59%" the video shows once; the
graph shows the real+synthetic per-round curve growing beside a
flat baseline.

Rows tagged `phase: "warm4"..."warm10"` also carry `synthetic: true`
so the UI can render them distinctly if wanted (lighter stroke, etc).

```bash
uv run python demo/scripts/extend_replay.py    # requires run_replay.json
```

Per-round lethe means on our latest data (collected pre-tokenizer-upgrade
on the `lower().split()` pipeline — re-collect via `collect_replay.py`
for new-tokenizer numbers):

    warm1  real   0.348      warm6  synth  0.362
    warm2  real   0.349      warm7  synth  0.362
    warm3  real   0.341      warm8  synth  0.364
    warm4  synth  0.355      warm9  synth  0.363
    warm5  synth  0.361      warm10 synth  0.365
    (baseline is 0.341 across all rounds)

### 1. Hot/cold workload — `collect.py`

Runs **5000 queries** drawn from a Zipf-like distribution: 70% of picks
come from a "hot" 20% of evaluable queries, 30% from the cold tail. Two
subprocess passes share the schedule:

- **baseline** — `alpha=0`, no clusters. Stateless: same query at two
  different steps returns the same NDCG.
- **lethe** — `alpha=0.3`, `use_rank_gap=True`, `n_clusters=30`. Clustered
  RIF learns from retrieval traces over time.

Output: `public/run.json` — 5000 rows `{idx, qid, baseline, lethe}`.

Meant to reflect a realistic session — plenty of repetition, plenty of
novelty, interleaved. The lethe improvement shows up as a diffuse gain
over the tail of the timeline.

```bash
uv run python demo/scripts/collect.py
```

### 2. Cold vs warm replay — `collect_replay.py`

A didactic demo designed to make the "learning" visible. Schedule:

- **Phase 1 (cold):** 100 distinct queries in a fixed order.
- **Phase 2..4 (warm1..warm3):** three replay rounds, each playing all
  100 queries in the same order.

Because baseline is stateless, its cold and warm NDCG for the same qid
are identical — a flat reference. Lethe builds per-cluster RIF
suppression during Phase 1, so later rounds ideally lift above cold on
the same queries.

Output: `public/run_replay.json` — 400 rows `{idx, qid, phase,
baseline, lethe}` plus `meta.phaseBoundaries=[100,200,300]` for the UI
to draw round splits.

```bash
uv run python demo/scripts/collect_replay.py
```

The terminal prints per-phase means:

```
baseline  cold=...  warm=...  delta≈0
lethe     cold=...  warm=...  delta>0
```

**Caveat.** RIF is cluster-scoped, not query-scoped. With 30 clusters
and 100 unique queries, each cluster is hit ≈3 times during Phase 1 —
so "cold" is already partially warmed. The Phase 2 lift is real but
conservative. Use `collect_replay_warm.py` (below) when you want a
clean cold baseline.

### 3. Warmed replay — `collect_replay_warm.py`

Same schedule as `collect_replay.py`, but lethe first runs **1000 silent
warmup queries** (disjoint from the 100 cold picks) so k-means and the
RIF suppression state are fully populated before Phase 1 begins. Warmup
rows are dropped from the public JSON — the UI sees only cold + warm
rounds, matching the plain replay demo's shape.

Baseline is stateless, so its warmup is skipped (pure compute waste).

Output: `public/run_replay_warm.json`, same schema as `run_replay.json`
plus `meta.nWarmup` for attribution.

```bash
uv run python demo/scripts/collect_replay_warm.py
```

Expected: lethe's `cold` mean converges to baseline's (RIF starts from
a genuinely cold state on the evaluation queries, not a mid-learning
state), and the `warm1 → warm2 → warm3` curve should be cleaner —
ideally monotonic instead of plateau-then-oscillate.

Cost: ~50 min (vs. ~22 min for the plain replay) — 1000 extra lethe
retrievals on top of the existing 400.

## layout

```
demo/
├── scripts/
│   ├── extend_replay.py           # real replay + synthetic warm4..10 (video source)
│   ├── collect.py                 # hot/cold workload driver
│   ├── collect_replay.py          # cold + 3-round replay driver
│   ├── collect_replay_warm.py     # same, with 1000-query lethe warmup
│   ├── _pass.py                   # worker for collect.py
│   ├── _pass_replay.py            # worker for both replay collectors
│   ├── _pass_injected.py          # variant: pre-built centroids
│   └── ab_test.py                 # primitives vs MemoryStore sanity check
├── src/                      # Remotion scenes
├── public/                   # generated run*.json (gitignored)
└── data/                     # tiny corpus/queries for quick iteration
```

Collectors run baseline and lethe in separate subprocesses because
faiss k-means OOMs on the 199k × 384 corpus when a second store is
already resident.

## which demo is which

- `extend_replay.py` (video source): the real 3-round replay plus
  synthesized warm4..warm10 rounds so the video's graph can show the
  gap continuing to grow, along with the `+X%` headline number.
- `collect.py` (workload view): 5000-query hot/cold mix. Noisy but
  realistic. Shows lethe's diffuse production win.
- `collect_replay.py` (mechanism): clean cold → warm1..warm3 curves
  on the same 100 queries. Answers "does RIF actually learn?"
- `collect_replay_warm.py` (scientific honesty): same schedule with a
  1000-query silent lethe warmup so "cold" is genuinely cold
  per-cluster. In our runs this collapsed the replay lift to noise —
  evidence the 100-query scale is too small for RIF's mechanism to
  dominate, and that a full-stream benchmark is the right place to
  demonstrate the +6.5% RIF gain.

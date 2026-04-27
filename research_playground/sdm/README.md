# SDM Research Prototype

A from-scratch Sparse Distributed Memory (Kanerva, 1988) for agent episodic recall. The question: does SDM's distributed binary storage generalize better than embedding + ANN for partial, vague, noisy queries?

## Run

```bash
uv run python sdm/main.py
```

Outputs a noise-mode × metric comparison of SDM, SDM+cleanup, and a FAISS baseline on a synthetic episodic dataset (250 events across 4 families with 5 near-duplicate siblings each, 1000 queries across 4 noise modes).

## Files

- `sdm.py` — SDM: random hard locations in binary space, bipolar counters, top-N activation, optional iterative cleanup.
- `embedding.py` — `TextEncoder` (MiniLM-L6-v2) + `HyperplaneProjector` (random Gaussian → binary).
- `dataset.py` — synthetic trips/meetings/purchases/preferences with near-duplicate siblings.
- `eval.py` — precision@1, recall@k, sibling confusion rate.
- `main.py` — entrypoint.

## Findings (from the default config: 8000 hard locations, activation_top_n=10, 512-bit addresses)

- **FAISS wins overall precision@1** (0.62 vs SDM 0.23). Dense cosine NN is sharper than random binary projection for this dataset.
- **SDM's gap is smallest on partial/fragment queries, largest on noisy/paraphrase.** When the query preserves content, FAISS's dense geometry exploits it; SDM's binary quantization throws signal away.
- **SDM has LOWER sibling confusion** on partial/fragment: when SDM fails, it tends to miss entirely rather than swap in a near-duplicate. A qualitatively different failure mode.
- **Iterative cleanup HURTS precision** on every noise mode (≈3× drop). Cleanup is an attractor dynamic that drifts toward sibling prototypes — useful for prototype extraction, destructive for distinct-episode recall.
- Sanity check: on exact queries, SDM gets 206/250 (82%), FAISS 233/250 (93%). SDM is working correctly; the gap is architectural, not a bug.

## When might SDM still be interesting?

Not as a drop-in episodic retriever. Possibly:
- Content-addressable-memory behavior at scale (cheap bit operations vs. dense dot products).
- Biologically-plausible associative store.
- Prototype extraction (the cleanup regime we saw as harmful becomes the goal).
- Hybrid use: SDM's "miss" error mode could be combined with an LLM that asks for clarification rather than confabulating.

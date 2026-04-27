"""Run the LLM enrichment layer over a subset of LongMemEval.

Selects the "answer-relevant sessions" — i.e. every session that contains at
least one qrels-relevant turn for any query. This covers both the correct
answers and their in-session siblings (the main within-session discrimination
target), at roughly 15-25k entries.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  uv run python scripts/enrich_longmemeval.py [--model MODEL] [--max N]

Output:
  tmp_data/longmemeval_enriched.jsonl — one Enrichment per line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DATA = Path("tmp_data")

# Inside legacy/ the package is at `legacy/lethe/`; parent.parent is
# `legacy/` itself, which is on sys.path after `pip install -e legacy/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lethe.enrichment import enrich_dataset, EnrichmentStats  # noqa: E402


def select_answer_relevant_entries(
    corpus_meta: dict[str, dict],
    qrels: dict[str, dict[str, int]],
    corpus_content: dict[str, str],
) -> list[tuple[str, str]]:
    """Return (entry_id, text) for every entry in a session that contains a qrels hit."""
    # Find all sessions referenced by any qrels entry
    answer_sessions: set[str] = set()
    for qid, relmap in qrels.items():
        for eid in relmap:
            session = corpus_meta.get(eid, {}).get("session_id")
            if session:
                answer_sessions.add(session)

    # Collect all entries in those sessions
    selected: list[tuple[str, str]] = []
    for eid, meta in corpus_meta.items():
        if meta.get("session_id") not in answer_sessions:
            continue
        text = corpus_content.get(eid, "")
        if not text.strip():
            continue
        selected.append((eid, text))
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="Anthropic model (default: claude-haiku-4-5 — matches memsearch)")
    ap.add_argument("--max", type=int, default=None,
                    help="Limit number of entries to enrich (for smoke tests)")
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Max concurrent API calls (default: 5 — Haiku has 50 RPM limit)")
    ap.add_argument("--output", default="tmp_data/longmemeval_enriched.jsonl",
                    help="Output JSONL path")
    args = ap.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ERROR: set ANTHROPIC_API_KEY in the environment first.", file=sys.stderr)
        sys.exit(1)

    # Load LongMemEval metadata
    with open(DATA / "longmemeval_meta.json") as f:
        corpus_meta = json.load(f)
    with open(DATA / "longmemeval_qrels.json") as f:
        qrels = json.load(f)
    with open(DATA / "longmemeval_corpus.json") as f:
        corpus_content = json.load(f)

    entries = select_answer_relevant_entries(corpus_meta, qrels, corpus_content)
    print(f"Answer-relevant entries: {len(entries):,}")

    if args.max is not None:
        entries = entries[: args.max]
        print(f"Limited to first {len(entries):,} entries for this run.")

    output = Path(args.output)

    # Run async
    import asyncio
    stats: EnrichmentStats = asyncio.run(
        enrich_dataset(
            entries=entries,
            output_path=output,
            model=args.model,
            concurrency=args.concurrency,
        )
    )

    print()
    print("=" * 60)
    print(f"Done. ok={stats.completed} err={stats.failed}")
    print(f"  input tokens:      {stats.total_input_tokens:,}")
    print(f"  output tokens:     {stats.total_output_tokens:,}")
    print(f"  cache read tokens: {stats.total_cache_read:,}")
    print(f"  cache write tokens: {stats.total_cache_write:,}")
    print(f"  estimated cost:    ${stats.est_cost_usd(args.model):.3f}")
    print(f"  output:            {output}")


if __name__ == "__main__":
    main()

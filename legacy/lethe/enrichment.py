"""LLM enrichment layer — write-time structured extraction via Claude API.

For each memory text, ask the model to produce a structured enrichment object
(gist, anticipated queries, entities, temporal markers). Indexed alongside the
original text in BM25/vector search, this attacks three failure modes we
measured in checkpoint 13's extended-metrics report: sibling confusion
(within-session discrimination), stale-fact retrieval (temporal awareness),
and wrong-family retrieval (vocabulary mismatch).

Uses the Anthropic SDK with:
- Prompt caching on the system prompt (~30% cost reduction after first call).
- Structured JSON output via output_config.format.
- Async concurrency with asyncio semaphore.
- Resumable — reads the output JSONL on startup and skips already-enriched IDs.

Run via experiments/enrich_longmemeval.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import anthropic
from anthropic import AsyncAnthropic


# --- Enrichment schema ---

@dataclass
class Enrichment:
    """Structured enrichment for a single memory."""
    entry_id: str
    gist: str
    anticipated_queries: list[str]
    entities: list[str]
    temporal_markers: list[str]
    # Diagnostics
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    error: str | None = None

    def as_search_text(self) -> str:
        """Concat all enrichment fields into a single searchable string.
        Caller prepends or appends this to the original memory text."""
        parts = [
            self.gist,
            " ".join(self.anticipated_queries),
            " ".join(self.entities),
            " ".join(self.temporal_markers),
        ]
        return " ".join(p for p in parts if p)


ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "gist": {
            "type": "string",
            "description": "One-sentence summary of the memory's core content.",
        },
        "anticipated_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "3 distinct user questions that this memory would answer well. "
                "Use natural language; vary the vocabulary so retrieval can match "
                "paraphrased queries."
            ),
        },
        "entities": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Key named entities: people, places, products, projects, tools, "
                "events. Include the exact surface form as mentioned."
            ),
        },
        "temporal_markers": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Any temporal context: explicit dates ('2026-03-14'), relative "
                "times ('last Tuesday', 'two weeks ago'), or sequence markers "
                "('after the release'). Empty list if the memory has no "
                "temporal grounding."
            ),
        },
    },
    "required": ["gist", "anticipated_queries", "entities", "temporal_markers"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are preprocessing a memory entry for later retrieval by an AI agent.

The agent will issue natural-language queries about past conversations. Your job is to enrich this memory so retrieval can match it reliably even when:
- The query uses different vocabulary than the memory text (paraphrase mismatch).
- The query asks for a specific detail that's one of many similar entries from the same conversation.
- The query references time ('what did I say last week').

Produce a compact JSON object with four fields:

1. `gist`: a single sentence capturing the memory's core content. Be concrete and specific — don't just restate the topic; summarize what was said or what happened.

2. `anticipated_queries`: 3 distinct user questions this memory would answer well. Use varied phrasings (questions, fragments, keywords). Think about what an agent retrieving this would literally type.

3. `entities`: named entities mentioned — people, places, products, projects, tools, events, document titles, etc. Include exact surface forms.

4. `temporal_markers`: any time expressions — explicit dates, relative times ('last Tuesday', 'after release'), durations. Empty list if the memory has none.

Be concise. Do not invent content not supported by the memory. Return JSON only — no commentary, no preamble."""


# --- Client ---

@dataclass
class EnrichmentStats:
    total: int = 0
    completed: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read: int = 0
    total_cache_write: int = 0
    started_at: float = field(default_factory=time.time)

    def est_cost_usd(self, model: str) -> float:
        # Anthropic pricing (input / output / cache_read / cache_write) per 1M tokens
        pricing = {
            "claude-haiku-4-5":  (1.0, 5.0, 0.1,  1.25),
            "claude-sonnet-4-6": (3.0, 15.0, 0.3, 3.75),
            "claude-opus-4-6":   (5.0, 25.0, 0.5, 6.25),
        }
        i, o, cr, cw = pricing.get(model, (3.0, 15.0, 0.3, 3.75))
        return (
            self.total_input_tokens * i / 1e6
            + self.total_output_tokens * o / 1e6
            + self.total_cache_read * cr / 1e6
            + self.total_cache_write * cw / 1e6
        )

    def throughput(self) -> float:
        dt = max(1e-3, time.time() - self.started_at)
        return self.completed / dt


async def _enrich_single(
    client: AsyncAnthropic,
    entry_id: str,
    memory_text: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 1024,
) -> Enrichment:
    """Enrich a single memory. Returns an Enrichment (with .error populated on failure)."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                cache_control={"type": "ephemeral"},  # auto-caches system prompt
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": memory_text}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": ENRICHMENT_SCHEMA,
                    }
                },
            )
            text = next(b.text for b in response.content if b.type == "text")
            data = json.loads(text)
            return Enrichment(
                entry_id=entry_id,
                gist=data["gist"],
                anticipated_queries=data["anticipated_queries"],
                entities=data["entities"],
                temporal_markers=data["temporal_markers"],
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
            )
        except anthropic.BadRequestError as e:
            # Schema violation, content policy, etc. — log and move on.
            return Enrichment(
                entry_id=entry_id, gist="", anticipated_queries=[],
                entities=[], temporal_markers=[], error=f"bad_request: {e}",
            )
        except Exception as e:  # noqa: BLE001 — best-effort batch processing
            return Enrichment(
                entry_id=entry_id, gist="", anticipated_queries=[],
                entities=[], temporal_markers=[], error=f"{type(e).__name__}: {e}",
            )


async def enrich_dataset(
    entries: list[tuple[str, str]],  # (entry_id, memory_text)
    output_path: Path,
    model: str = "claude-haiku-4-5",
    concurrency: int = 5,
    progress_every: int = 50,
) -> EnrichmentStats:
    """Enrich a list of memories, writing JSONL incrementally. Resumable.

    - Reads output_path on startup to skip already-enriched entries.
    - Writes one Enrichment per line as each completes.
    - Returns final stats.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: gather already-completed IDs
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("error") is None and rec.get("gist"):
                        done_ids.add(rec["entry_id"])
                except json.JSONDecodeError:
                    continue
        print(f"Resume: {len(done_ids)} entries already enriched.", flush=True)

    todo = [(eid, text) for eid, text in entries if eid not in done_ids]
    stats = EnrichmentStats(total=len(todo))
    if not todo:
        print("Nothing to enrich.", flush=True)
        return stats

    print(f"Enriching {len(todo)} entries with {model} at concurrency={concurrency}...",
          flush=True)

    # max_retries=8 absorbs 429 bursts when we briefly exceed per-minute quota;
    # the SDK respects retry-after headers, so wall-clock rate converges to the limit.
    client = AsyncAnthropic(max_retries=8)
    semaphore = asyncio.Semaphore(concurrency)

    # Open output in append mode, flush after each write
    with open(output_path, "a") as fout:
        tasks = [
            asyncio.create_task(_enrich_single(client, eid, text, model, semaphore))
            for eid, text in todo
        ]
        try:
            for coro in asyncio.as_completed(tasks):
                enr = await coro
                fout.write(json.dumps(asdict(enr)) + "\n")
                fout.flush()
                if enr.error:
                    stats.failed += 1
                else:
                    stats.completed += 1
                    stats.total_input_tokens += enr.input_tokens
                    stats.total_output_tokens += enr.output_tokens
                    stats.total_cache_read += enr.cache_read_tokens
                    stats.total_cache_write += enr.cache_write_tokens

                n = stats.completed + stats.failed
                if n % progress_every == 0 or n == stats.total:
                    cost = stats.est_cost_usd(model)
                    tps = stats.throughput()
                    print(
                        f"  [{n}/{stats.total}] ok={stats.completed} err={stats.failed} "
                        f"cost=${cost:.3f} rate={tps:.1f}/s "
                        f"cache_read={stats.total_cache_read:,}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("Interrupted — partial results are saved. Re-run to resume.",
                  flush=True)
            for t in tasks:
                t.cancel()
            raise

    return stats


def load_enrichments(path: Path) -> dict[str, Enrichment]:
    """Load the JSONL written by enrich_dataset. Returns {entry_id -> Enrichment}."""
    out: dict[str, Enrichment] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Skip failures and keep the latest successful record per entry_id
            if rec.get("error") or not rec.get("gist"):
                continue
            out[rec["entry_id"]] = Enrichment(
                entry_id=rec["entry_id"],
                gist=rec["gist"],
                anticipated_queries=rec.get("anticipated_queries", []),
                entities=rec.get("entities", []),
                temporal_markers=rec.get("temporal_markers", []),
                input_tokens=rec.get("input_tokens", 0),
                output_tokens=rec.get("output_tokens", 0),
                cache_read_tokens=rec.get("cache_read_tokens", 0),
                cache_write_tokens=rec.get("cache_write_tokens", 0),
            )
    return out

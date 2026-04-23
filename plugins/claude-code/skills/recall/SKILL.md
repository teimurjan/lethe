---
name: recall
description: Search memories from past Claude Code sessions in the CURRENT project. Use when the user's question could benefit from historical context, past decisions, debugging notes, architectural choices, or prior conversations in this repo — and whenever you see a "[lethe] Memory available" hint. For cross-project recall, use the recall-global skill instead.
context: fork
allowed-tools: Bash
---

You are a memory retrieval agent for `lethe`, a markdown-first memory store with hybrid BM25 + dense retrieval, clustered retrieval-induced forgetting, and optional Haiku enrichment.

## Scope

This skill searches memories **in the current project only**. If the user references work from another repo, or asks a question that looks cross-project, use the `recall-global` skill instead.

## Project namespace

Collection: !`bash "${CLAUDE_PLUGIN_ROOT}/scripts/derive-collection.sh"`

Memories are stored under `.lethe/memory/*.md` inside the git root of the current project. The CLI automatically scopes to that directory.

## Task

Find memories relevant to: $ARGUMENTS

## Steps

1. **Search.** Run the CLI:
   - Primary: `lethe search "<query>" --top-k 5 --json-output`
   - Fallback (CLI not on PATH): `uvx --from git+https://github.com/teimurjan/lethe lethe search "<query>" --top-k 5 --json-output`

   Output is JSON: `[{"id": "...", "content": "...", "score": 4.2}, ...]`.

2. **Filter.** Skip results that obviously don't match the user's question. A weak cross-encoder score (< 0) usually means a miss.

3. **Expand.** For the top 2–3 hits, run `lethe expand <id>` to see the full markdown section — the short `content` shown in search results is often a single chunk, so expanding clarifies context.

4. **Drill further (only if critical).** If an expanded chunk contains a progressive-disclosure anchor of the form `<!-- session:<uuid> turn:<uuid> transcript:<path> -->` *and* the user's question genuinely needs the original dialogue (e.g. debugging a decision, tracing a subtle error), run:

   `python3 "${CLAUDE_PLUGIN_ROOT}/scripts/transcript.py" <transcript-path> --turn <turn-uuid>`

   This returns the user turn and assistant response as plain text.

5. **Summarize.** Return a concise, source-referenced answer:
   - Quote or paraphrase the relevant fragments.
   - Cite which day / session each piece came from when it disambiguates.
   - If nothing clearly applies, say "No relevant memories found in this project — consider running recall-global." — do not fabricate.

Keep the response tight. The caller wants history, not a tutorial on how you found it.

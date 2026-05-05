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
   - `lethe search "<query>" --top-k 5 --json-output`
   - If `lethe` is not on PATH, ask the user to install it: `brew tap teimurjan/lethe && brew install lethe` (macOS / Linuxbrew) or `cargo install lethe-cli`.

   Output is JSON: `[{"id": "...", "content": "...", "score": 4.2}, ...]`.

   By default this opens the project index read-write so retrieval-driven RIF state (suppression scores, tier promotion, retrieval counts) keeps evolving from your queries. If you hit a "Conflicting lock" error because another lethe process is writing to the same index (parallel sessions, an in-flight stop hook), retry once with `--read-only` — that opens the index read-only and skips the post-retrieve save, at the cost of not updating RIF state for that query.

2. **Filter.** Skip results that obviously don't match the user's question. A weak cross-encoder score (< 0) usually means a miss.

3. **Expand.** For the top 2–3 hits, run `lethe expand <id1> <id2> <id3>` (multi-arg, single call) to see the full markdown sections. Output is plain text with `=== <id> ===` headers between chunks. The short `content` shown in search results is often a single chunk, so expanding clarifies context.

4. **Drill further (only if critical).** If an expanded chunk contains a progressive-disclosure anchor of the form `<!-- session:<uuid> turn:<uuid> transcript:<path> -->` *and* the user's question genuinely needs the original dialogue (e.g. debugging a decision, tracing a subtle error), run:

   `lethe-claude-code transcript <transcript-path> --turn <turn-uuid>`

   This returns the user turn and assistant response as plain text.

5. **Summarize.** Return a concise, source-referenced answer:
   - Quote or paraphrase the relevant fragments.
   - Cite which day / session each piece came from when it disambiguates.
   - If nothing clearly applies, say "No relevant memories found in this project — consider running recall-global." — do not fabricate.

Keep the response tight. The caller wants history, not a tutorial on how you found it.

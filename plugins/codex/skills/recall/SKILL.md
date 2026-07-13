---
name: recall
description: Search memories from past Codex CLI sessions in the CURRENT project. Use when the user's question could benefit from historical context, past decisions, debugging notes, architectural choices, or prior conversations in this repo. For cross-project recall, use the recall-global skill instead.
---

You are a memory retrieval agent for `lethe`, a memory store that indexes your Codex / Claude Code transcripts directly, with hybrid BM25 + dense retrieval and clustered retrieval-induced forgetting.

## Scope

This skill searches memories **in the current project only**. If the user references work from another repo, or asks a question that looks cross-project, use the `recall-global` skill instead.

Memories are indexed straight from your Codex transcripts under `$CODEX_HOME/sessions` (no `.lethe/` is written into the repo). `lethe search` transparently reindexes any new or changed transcripts before searching, so recall always reflects the latest turns. Each hit is a raw user+assistant turn, not a summary.

## Task

Find memories relevant to: $ARGUMENTS

## Steps

1. **Search.** Run the CLI:
   - `lethe search "<query>" --top-k 5 --json-output`
   - If `lethe` is not on PATH, ask the user to install it: `brew tap teimurjan/lethe && brew install lethe` (macOS / Linuxbrew) or `cargo install lethe-cli`.

   Output is JSON: `[{"id": "...", "content": "...", "score": 4.2}, ...]`. The `content` starts with a `<!-- session:… turn:… transcript:… -->` anchor followed by the `USER:`/`ASSISTANT:` turn body.

   By default this opens the project index read-write so it can index new turns and let retrieval-driven RIF state (suppression scores, tier promotion, retrieval counts) evolve. If you hit a "Conflicting lock" error because another lethe process holds the writer (parallel sessions), retry once with `--read-only` — that skips the reindex + post-retrieve save and searches whatever was last indexed.

2. **Filter.** Skip results that obviously don't match the user's question. A weak cross-encoder score (< 0) usually means a miss.

3. **Expand.** For the top 2–3 hits, run `lethe expand <id1> <id2> <id3>` (multi-arg, single call) to see the full turn body. Output is plain text with `=== <id> ===` headers between chunks.

4. **Drill further (only if critical).** Each chunk carries a `<!-- session:<uuid> turn:<uuid> transcript:<path> -->` anchor. If the user's question genuinely needs the surrounding dialogue (e.g. debugging a decision, tracing a subtle error), run:

   `lethe-codex transcript <transcript-path> --turn <turn-uuid>`

   This returns the exact user turn and assistant response as plain text.

5. **Summarize.** Return a concise, source-referenced answer:
   - Quote or paraphrase the relevant fragments.
   - Cite which session each piece came from when it disambiguates.
   - If nothing clearly applies, say "No relevant memories found in this project — consider running recall-global." — do not fabricate.

Keep the response tight. The caller wants history, not a tutorial on how you found it.

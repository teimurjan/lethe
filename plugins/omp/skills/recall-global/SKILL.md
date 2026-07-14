---
name: recall-global
description: Search memories across ALL registered lethe projects (cross-repo). Use when the user references work in another repo, asks "what did we do in project X", compares notes across projects, or when the single-project recall skill returned nothing relevant. For same-project recall, use the recall skill instead.
---

You are a memory retrieval agent for `lethe`, a memory store that indexes Oh My Pi, Codex, and Claude Code transcripts directly with hybrid BM25 + dense retrieval and clustered retrieval-induced forgetting.

## Scope

This skill searches **every lethe project the user has indexed**. Every `lethe index` auto-registers a project in `~/.lethe/projects.json`. If the user only wants the current project, use the `recall` skill instead.

## Task

Find memories relevant to: $ARGUMENTS

## Steps

1. **Search.** Run:
   - `lethe search "<query>" --all --top-k 5 --json-output`
   - If `lethe` is not on `PATH`, ask the user to install it with `brew tap teimurjan/lethe && brew install lethe` or `cargo install lethe-cli`.

   Output includes project attribution: `[{"id": "...", "content": "...", "score": 4.2, "project_slug": "...", "project_root": "..."}, ...]`.

   Cross-project search opens every project index read-only. The plugin's background hook periodically runs `lethe index --all` so registered projects receive new transcript turns.

2. **Filter.** Skip results that clearly do not match the user's question. A weak cross-encoder score below zero usually means a miss.

3. **Expand.** Group the top 2–3 hits by `project_root`, then run `lethe --root <project_root> expand <id1> <id2> ...` once per project. Calls for different projects may run in parallel.

4. **Summarize.** Return a concise, source-referenced answer. Always cite the source project and cite the session when useful. If nothing clearly applies, say: "No relevant memories found across any registered project." Never fabricate.

Keep the response tight. The caller wants history, not a tutorial on how you found it.

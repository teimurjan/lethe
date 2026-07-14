//! Turn-level transcript indexing.
//!
//! Where `markdown_store` chunks curated `.lethe/memory/*.md` files, this
//! module indexes agent transcripts (Claude Code, Codex, and Oh My Pi JSONL)
//! directly.
//! Each user+assistant turn becomes one chunk carrying a progressive-
//! disclosure anchor for provenance and `lethe expand`.
//!
//! Chunk content keeps the anchor (`<!-- session:S turn:T transcript:P -->`)
//! for provenance; the bi-encoder embedding and BM25/rerank inputs run
//! through [`markdown_store::embed_content`], which strips the anchor so
//! the volatile UUIDs/paths never pollute retrieval.
//!
//! [`sync`] is **add-only** — unlike `markdown_store::reindex` it never
//! deletes entries. Transcripts are append-only and occasionally rewritten
//! (compaction); dropping turns that vanished from a rewritten file would
//! violate the "don't forget" contract, so we retain them. Pruning stale
//! turns is a `lethe reset` + rebuild operation.

use crate::markdown_store::{chunk_hash, embed_content};
use crate::memory_store::MemoryStore;

/// One indexable turn (a user prompt + the assistant reply that followed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnChunk {
    /// `chunk_hash` of the anchor-stripped body — stable across
    /// sessions/transcripts, so the same exchange dedupes to one id
    /// regardless of which transcript it was re-indexed from.
    pub id: String,
    pub session_id: String,
    /// Ordinal of the turn within its session (0-based).
    pub turn_idx: i64,
    /// Full content, anchor line first, then the `USER:`/`ASSISTANT:` body.
    pub content: String,
}

/// Build a [`TurnChunk`] from one parsed turn.
///
/// `turn_uuid` is the source transcript's user-record id and is retained in
/// the anchor for provenance.
#[must_use]
pub fn build_chunk(
    session_id: &str,
    turn_uuid: &str,
    transcript_path: &str,
    user_text: &str,
    assistant_text: &str,
    turn_idx: i64,
) -> TurnChunk {
    let content = format!(
        "<!-- session:{} turn:{} transcript:{} -->\nUSER:\n{}\n\nASSISTANT:\n{}",
        session_id,
        turn_uuid,
        transcript_path,
        user_text.trim(),
        assistant_text.trim(),
    );
    TurnChunk {
        // Hash the anchor-stripped body, not the anchored content: the
        // anchor's session/turn/transcript values are volatile, so the
        // same exchange re-indexed from a moved or recopied transcript
        // must resolve to the same id (and be skipped by `live_ids`).
        id: chunk_hash(&embed_content(&content)),
        session_id: session_id.to_owned(),
        turn_idx,
        content,
    }
}

/// Counts returned by [`sync`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SyncCounts {
    pub added: usize,
    pub unchanged: usize,
    pub total: usize,
}

/// Add every chunk in `chunks` that isn't already in `store`. Add-only:
/// never deletes. Embeds all new turns in one batched ORT call.
pub fn sync(chunks: &[TurnChunk], store: &MemoryStore) -> Result<SyncCounts, crate::Error> {
    let mut counts = SyncCounts {
        total: chunks.len(),
        ..Default::default()
    };
    store.bulk_add(|| {
        let existing = store.live_ids();
        // Chunks a prior `dedupe` merged away: skip them so re-parsing a
        // rewritten transcript can't resurrect an absorbed turn.
        let aliased = store.aliased_ids();

        // Phase 1: gather new chunks and their embed inputs (stripped).
        let mut embed_inputs: Vec<String> = Vec::new();
        let mut pending: Vec<&TurnChunk> = Vec::new();
        for chunk in chunks {
            if existing.contains(&chunk.id) || aliased.contains(&chunk.id) {
                counts.unchanged += 1;
                continue;
            }
            embed_inputs.push(embed_content(&chunk.content));
            pending.push(chunk);
        }

        // Phase 2: one batched encode, then insert full anchored content
        // with the precomputed (stripped) embedding.
        if !embed_inputs.is_empty() {
            let bi = store.bi_encoder().ok_or(crate::Error::NotInitialized(
                "bi_encoder required for transcript sync",
            ))?;
            let refs: Vec<&str> = embed_inputs.iter().map(String::as_str).collect();
            let embs = bi.encode_batch(&refs)?;
            for (i, chunk) in pending.iter().enumerate() {
                let emb = embs.row(i).to_owned();
                let inserted = store.add_with_embedding(
                    &chunk.content,
                    emb,
                    Some(&chunk.id),
                    &chunk.session_id,
                    chunk.turn_idx,
                )?;
                if inserted.is_some() {
                    counts.added += 1;
                } else {
                    counts.unchanged += 1;
                }
            }
        }
        Ok(())
    })?;
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markdown_store::parse_anchor;

    #[test]
    fn build_chunk_embeds_anchor_and_body() {
        let c = build_chunk("sess-1", "u-42", "/tmp/log.jsonl", "hello", "hi back", 0);
        // Anchor round-trips.
        let a = parse_anchor(&c.content).unwrap();
        assert_eq!(a.session, "sess-1");
        assert_eq!(a.turn, "u-42");
        assert_eq!(a.transcript, "/tmp/log.jsonl");
        // Body survives.
        assert!(c.content.contains("USER:\nhello"));
        assert!(c.content.contains("ASSISTANT:\nhi back"));
        // id is a stable 16-hex hash.
        assert_eq!(c.id.len(), 16);
        assert!(c.id.chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn embed_content_strips_anchor_keeps_bodies() {
        let c = build_chunk("s", "t", "/x", "run the tests", "done", 0);
        let stripped = embed_content(&c.content);
        assert!(!stripped.contains("<!--"));
        assert!(!stripped.contains("transcript:"));
        assert!(stripped.contains("USER:"));
        assert!(stripped.contains("run the tests"));
        assert!(stripped.contains("ASSISTANT:"));
        assert!(stripped.contains("done"));
    }

    #[test]
    fn distinct_text_gets_distinct_ids() {
        let a = build_chunk("s", "t1", "/x", "question one", "answer one", 0);
        let b = build_chunk("s", "t2", "/x", "question two", "answer two", 1);
        assert_ne!(a.id, b.id, "different exchanges must not collide");
    }

    #[test]
    fn sync_skips_aliased_chunk() {
        use crate::memory_store::{MemoryStore, StoreConfig};
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::open(
            dir.path().join("s"),
            None,
            None,
            StoreConfig {
                dim: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let chunk = build_chunk("s", "u1", "/x", "hello", "hi", 0);
        // A prior `dedupe` absorbed this chunk into some canonical id.
        store
            .with_db(|db| db.insert_alias(&chunk.id, "canonical"))
            .unwrap();
        // Re-syncing it (as a rewritten transcript would) must not
        // resurrect it — no encoder is even required since it's skipped.
        let counts = sync(std::slice::from_ref(&chunk), &store).unwrap();
        assert_eq!(counts.added, 0);
        assert_eq!(counts.unchanged, 1);
    }

    #[test]
    fn same_text_different_anchor_shares_id() {
        // The same exchange re-indexed from a moved/renamed transcript
        // (different session + transcript path) must resolve to one id so
        // the write path dedupes it via `live_ids`.
        let a = build_chunk("sess-a", "u1", "/old/path.jsonl", "hello", "hi", 0);
        let b = build_chunk("sess-b", "u2", "/new/path.jsonl", "hello", "hi", 7);
        assert_eq!(a.id, b.id, "anchor-only differences must not change the id");
    }
}

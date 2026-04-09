"""Download LongMemEval S variant, embed individual turns, save to disk."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset  # type: ignore[import-untyped]
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]


DATA_DIR = Path("data")
MODEL_NAME = "all-MiniLM-L6-v2"


def turn_id(session_id: str, turn_idx: int) -> str:
    return f"{session_id}_t{turn_idx}"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print("Loading LongMemEval S variant from HuggingFace...")
    ds = load_dataset(
        "xiaowu0162/longmemeval-cleaned",
        data_files="longmemeval_s_cleaned.json",
        split="train",
    )
    print(f"Loaded {len(ds)} questions")

    # Build corpus: individual turns, deduplicated
    corpus: dict[str, str] = {}
    for row in ds:
        for sid, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            for ti, turn in enumerate(session):
                tid = turn_id(sid, ti)
                if tid not in corpus:
                    corpus[tid] = f"{turn['role']}: {turn['content']}"

    # Build queries (need both text and embedding)
    queries: dict[str, str] = {}
    for row in ds:
        queries[row["question_id"]] = row["question"]

    # Build qrels: use has_answer labels, fallback to all turns in answer sessions
    qrels: dict[str, dict[str, int]] = {}
    for row in ds:
        qid = row["question_id"]
        relevant: dict[str, int] = {}
        for sid, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            for ti, turn in enumerate(session):
                if turn.get("has_answer") is True:
                    relevant[turn_id(sid, ti)] = 1
        if not relevant:
            for sid, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
                if sid in row["answer_session_ids"]:
                    for ti, _ in enumerate(session):
                        relevant[turn_id(sid, ti)] = 1
        qrels[qid] = relevant

    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"Corpus: {len(corpus_ids)} turns, Queries: {len(query_ids)}")

    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding {len(corpus_texts)} turns...")
    corpus_embeddings = model.encode(
        corpus_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256,
    )
    print(f"Embedding {len(query_texts)} questions...")
    query_embeddings = model.encode(
        query_texts, normalize_embeddings=True, show_progress_bar=True, batch_size=256,
    )

    np.savez(
        str(DATA_DIR / "longmemeval_prepared.npz"),
        corpus_ids=np.array(corpus_ids),
        corpus_embeddings=corpus_embeddings.astype(np.float32),
        query_ids=np.array(query_ids),
        query_embeddings=query_embeddings.astype(np.float32),
    )
    with open(DATA_DIR / "longmemeval_qrels.json", "w") as f:
        json.dump(qrels, f)
    corpus_content = {cid: text for cid, text in zip(corpus_ids, corpus_texts)}
    with open(DATA_DIR / "longmemeval_corpus.json", "w") as f:
        json.dump(corpus_content, f)
    # Save query texts for cross-encoder
    with open(DATA_DIR / "longmemeval_queries.json", "w") as f:
        json.dump(queries, f)

    print(f"Saved to {DATA_DIR}/")
    print(f"  corpus_embeddings: {corpus_embeddings.shape}")
    print(f"  query_embeddings: {query_embeddings.shape}")


if __name__ == "__main__":
    main()

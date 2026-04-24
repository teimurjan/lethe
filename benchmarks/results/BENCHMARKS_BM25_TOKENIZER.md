# BM25 Tokenizer Sweep

The shipped BM25 tokenizer was `text.lower().split()` — punctuation stays glued to words, so `"hello,"` and `"hello"` are different tokens. Replacing it with a regex word-tokenizer (`[A-Za-z0-9_]+`) gives a larger retrieval lift than any single mechanism change since clustered-RIF, at zero latency cost.

Eval: 100-query random sample (seed 0)
Pipeline: BM25 top-30 + FAISS top-30 → RRF → cross-encoder rerank; adaptive deep pass at k_deep=100
Fixed: `k_shallow=30`, `confidence_threshold=4.0`, xenc = `Xenova/ms-marco-MiniLM-L-6-v2` (ONNX, CPU)
Corpus: LongMemEval S, 199,509 turns. `nltk==3.9.4` for PorterStemmer.
Date: 2026-04-24

| Tokenizer | NDCG@10 | ΔNDCG | Recall@10 | p50 | p95 | BM25 build |
|-----------|---------|-------|-----------|-----|-----|------------|
| baseline (`lower().split()`) | 0.3022 | +0.00pp | 0.3731 | 6362ms | 9646ms | 11.2s |
| **regex words (new default)** | **0.3390** | **+3.68pp** | **0.4410** | **6084ms** | **8548ms** | **10.4s** |
| regex + stopwords removed | 0.3084 | +0.63pp | 0.4027 | 5621ms | 7585ms | 10.6s |
| regex + stop + Porter stem | 0.3153 | +1.31pp | 0.4143 | 5546ms | 7001ms | 187.8s |

## Observations

- **Punctuation is the whole story.** `lower().split()` keeps `,`, `.`, `?`, `!` glued. Query `"mongodb?"` never matches corpus `"mongodb"`. The regex variant separates tokens from punctuation and immediately picks up +3.68pp NDCG / +6.79pp Recall.
- **Stopword removal regresses.** Dropping function words ("the", "is", "of", …) erases the gain (0.3390 → 0.3084). Conversational memory has short, specific queries where function words act as syntactic anchors — removing them is net-harmful at this corpus size.
- **Porter stemming is a trap.** Build cost jumps 17× (10s → 188s), and quality tops out at +1.31pp — well below plain regex. The over-conflation cost (e.g. `generate`/`general` → `gener`) exceeds the vocabulary-compression gain on conversational text.

## Why this wasn't caught earlier

The research journey focused on mechanism changes (RIF, clusters, enrichment). Baseline tokenization was inherited from the earliest prototype. On IR benchmarks with clean-text corpora (like NFCorpus) the gap would be smaller, but LongMemEval's conversational turns are heavy on contractions, questions, and trailing punctuation.

## Shipping

Adopted regex tokenizer as the default. Build-time and retrieval-time tokenization share one helper (`tokenize_bm25`) in `lethe.vectors` so corpus and query paths can't drift.

Reproducer: `benchmarks/run_bm25_tokenizer.py`.

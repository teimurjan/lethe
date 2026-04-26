//! BM25 tokenizer — direct port of `legacy/lethe/vectors.py::tokenize_bm25`.
//!
//! The Python reference is a regex word tokenizer that lowercases input
//! and extracts `[A-Za-z0-9_]+` runs. The shipping benchmark
//! (`benchmarks/results/BENCHMARKS_BM25_TOKENIZER.md`) recorded a
//! +3.68 pp NDCG@10 / +6.79 pp Recall@10 lift over `lower().split()`.
//! See that document before considering changes.

use std::sync::LazyLock;

use regex::Regex;

static WORD_RE: LazyLock<Regex> = LazyLock::new(|| {
    // The `\w` shortcut would be Unicode-aware; the Python reference is
    // ASCII-only. Match it byte-for-byte.
    Regex::new(r"[A-Za-z0-9_]+").expect("static regex must compile")
});

/// Tokenize `text` for BM25 — lowercased ASCII word runs.
///
/// Empty input yields an empty vector (callers like `search_bm25` rely
/// on this to short-circuit on punctuation-only queries).
///
/// **Unicode**: we use `str::to_lowercase` (full Unicode case folding)
/// rather than `to_ascii_lowercase` so the byte-by-byte output matches
/// Python's `text.lower()`. The difference matters for inputs like
/// Turkish `İskender`: Python lowercases `İ` to `i + ◌̇` and the
/// downstream ASCII regex picks up an extra `i` token; if we skipped
/// Unicode case folding, that extra token would be missing and BM25
/// scores on cross-script corpora would silently drift from the
/// Python reference (caught by the LongMemEval components bench).
#[must_use]
pub fn tokenize_bm25(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let lowered = text.to_lowercase();
    WORD_RE
        .find_iter(&lowered)
        .map(|m| m.as_str().to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_punctuation_and_lowercases() {
        // Mirrors tests/test_vectors.py::test_tokenize_bm25_strips_punctuation_and_lowercases.
        assert_eq!(tokenize_bm25("MongoDB?"), vec!["mongodb"]);
        assert_eq!(tokenize_bm25("Hello, world!"), vec!["hello", "world"]);
        assert_eq!(tokenize_bm25("can't won't"), vec!["can", "t", "won", "t"]);
        assert!(tokenize_bm25("").is_empty());
    }

    #[test]
    fn punctuation_only_yields_empty() {
        for q in ["???", "...", "!!!", "   "] {
            assert!(tokenize_bm25(q).is_empty(), "expected empty for {q:?}");
        }
    }

    #[test]
    fn keeps_underscores_and_digits() {
        assert_eq!(
            tokenize_bm25("session_42 turn_idx=7"),
            vec!["session_42", "turn_idx", "7"]
        );
    }

    #[test]
    fn unicode_case_folding_matches_python_lower() {
        // Turkish `İ` (U+0130) lowercases to `i` + combining-dot
        // (U+0307) under full Unicode case folding. Python's
        // `str.lower()` does this; `str::to_ascii_lowercase` would
        // not, dropping the extra `i` token. Tested here because the
        // LongMemEval corpus contains words like "İskender" that
        // generated false-negative BM25 hits before the fix.
        assert_eq!(tokenize_bm25("İskender"), vec!["i", "skender"]);
        // Mixed-script words: the non-ASCII letter is dropped by the
        // ASCII regex, leaving the ASCII tail.
        assert_eq!(tokenize_bm25("ÄPFEL"), vec!["pfel"]);
        assert_eq!(tokenize_bm25("Café NAÏVE"), vec!["caf", "na", "ve"]);
    }
}

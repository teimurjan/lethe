//! Markdown chunker — port of `src/lethe/markdown_store.py`.
//!
//! Splits `.md` files on `##` / `###` headings (top-level `#` is a
//! file title and is NOT a chunk separator). Drops sections that are
//! "heading + anchor + blanks only" — the hook-written daily files
//! often contain those when SessionStart fires repeatedly.
//!
//! Each chunk gets a stable id = first 16 hex chars of
//! `sha256(content.strip())`. Anchor comments
//! (`<!-- session:UUID turn:UUID transcript:/path -->`) embedded at
//! the top of hook-written chunks are extracted via `parse_anchor`.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use sha2::{Digest, Sha256};

static ANCHOR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"<!--\s*session:(?P<session>\S*?)\s+turn:(?P<turn>\S*?)\s+transcript:(?P<transcript>.*?)\s*-->",
    )
    .expect("static anchor regex must compile")
});

static HEADING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(#+)\s+(.*)$").expect("static heading regex must compile"));

/// One addressable section of a markdown memory file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// First 16 hex chars of `sha256(stripped_content)`.
    pub id: String,
    pub source: PathBuf,
    pub heading: String,
    /// Raw markdown content of the chunk including the heading line.
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Anchor {
    pub session: String,
    pub turn: String,
    pub transcript: String,
}

/// 16-char hex id derived from the chunk's stripped content. Stable
/// across runs — matches `markdown_store.chunk_hash`.
#[must_use]
pub fn chunk_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.trim().as_bytes());
    let hex_digest = hex::encode(hasher.finalize());
    hex_digest[..16].to_owned()
}

/// Extract the first anchor comment from a chunk, if any.
#[must_use]
pub fn parse_anchor(chunk_text: &str) -> Option<Anchor> {
    let caps = ANCHOR_RE.captures(chunk_text)?;
    Some(Anchor {
        session: caps["session"].to_owned(),
        turn: caps["turn"].to_owned(),
        transcript: caps["transcript"].to_owned(),
    })
}

/// Strip heading lines and anchor lines from `chunk_text`.
///
/// The anchor + heading combined dominates the bi-encoder embedding
/// (UUIDs are highly self-similar across all hook-written chunks, so
/// they collapse cosine similarity above the 0.95 dedup threshold).
/// `embed_content` is what we feed to BM25 + the bi-encoder.
#[must_use]
pub fn embed_content(chunk_text: &str) -> String {
    let mut kept: Vec<&str> = Vec::with_capacity(chunk_text.len() / 64);
    for line in chunk_text.lines() {
        let stripped = line.trim();
        if stripped.starts_with('#') {
            continue;
        }
        if stripped.starts_with("<!--") && stripped.ends_with("-->") {
            continue;
        }
        kept.push(line);
    }
    kept.join("\n").trim().to_owned()
}

/// True iff `body` contains any non-heading, non-anchor line.
fn has_body(body: &str) -> bool {
    for line in body.lines() {
        let stripped = line.trim();
        if stripped.is_empty() {
            continue;
        }
        if stripped.starts_with('#') {
            continue;
        }
        if stripped.starts_with("<!--") && stripped.ends_with("-->") {
            continue;
        }
        return true;
    }
    false
}

/// Split a markdown document on `##` / `###` headings.
///
/// Top-level `#` is a file title and is NOT a separator. Heading-only
/// sections are dropped. The leading block (text before the first `##`)
/// is emitted as an anonymous chunk if it has body content.
#[must_use]
pub fn split_into_chunks(md_text: &str, source: &Path) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut current_heading = String::new();

    let flush = |buf: &Vec<&str>, current_heading: &str, chunks: &mut Vec<Chunk>| {
        if buf.is_empty() {
            return;
        }
        let body_full = buf.join("\n");
        let body = body_full.trim();
        if body.is_empty() || !has_body(body) {
            return;
        }
        chunks.push(Chunk {
            id: chunk_hash(body),
            source: source.to_path_buf(),
            heading: current_heading.to_owned(),
            content: body.to_owned(),
        });
    };

    for line in md_text.lines() {
        if let Some(caps) = HEADING_RE.captures(line) {
            // `##`+ only — single-`#` is a file title, not a separator.
            let depth = caps[1].len();
            if depth >= 2 {
                flush(&buf, &current_heading, &mut chunks);
                buf.clear();
                buf.push(line);
                current_heading.clear();
                current_heading.push_str(caps[2].trim());
                continue;
            }
        }
        buf.push(line);
    }
    flush(&buf, &current_heading, &mut chunks);
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn splits_on_h2_keeps_h1_in_body() {
        let md = "# Title\n\nintro\n\n## Section A\n\nalpha\n\n## Section B\n\nbeta\n";
        let chunks = split_into_chunks(md, &PathBuf::from("file.md"));
        // Three chunks: leading "intro" block, A, B.
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].content.contains("intro"));
        assert_eq!(chunks[1].heading, "Section A");
        assert_eq!(chunks[2].heading, "Section B");
    }

    #[test]
    fn drops_heading_only_sections() {
        let md = "## Empty\n\n## Real\nbody\n";
        let chunks = split_into_chunks(md, &PathBuf::from("file.md"));
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].heading, "Real");
    }

    #[test]
    fn anchor_extracted() {
        let chunk = "## 12:30\n<!-- session:abc turn:def transcript:/tmp/log.jsonl -->\nnotes\n";
        let a = parse_anchor(chunk).unwrap();
        assert_eq!(a.session, "abc");
        assert_eq!(a.turn, "def");
        assert_eq!(a.transcript, "/tmp/log.jsonl");
    }

    #[test]
    fn embed_content_strips_anchor_and_headings() {
        let chunk = "## 12:30\n<!-- session:s turn:t transcript:/x -->\nbody line\n- bullet\n";
        let stripped = embed_content(chunk);
        assert_eq!(stripped, "body line\n- bullet");
    }

    #[test]
    fn chunk_hash_is_stable_and_16_hex() {
        let h1 = chunk_hash("hello\nworld");
        let h2 = chunk_hash("hello\nworld");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 16);
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

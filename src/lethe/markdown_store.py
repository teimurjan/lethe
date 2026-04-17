"""Markdown-first memory layer used by the Claude Code plugin.

Responsibilities:
- Scan ``.lethe/memory/*.md`` and split each file into chunks on ``##`` / ``###``
  headings.
- Hash each chunk (SHA-256) so we can detect which chunks are new, modified, or
  removed since the last indexing pass.
- Keep a tiny file-backed map of ``chunk_id -> raw markdown text`` so ``lethe
  expand`` can show the full section for an id returned by ``lethe search``.
- Parse the progressive-disclosure anchor that hooks embed at the top of each
  generated chunk: ``<!-- session:<uuid> turn:<uuid> transcript:<path> -->``.

The module deliberately avoids any coupling to Claude Code — it just operates
on a directory of markdown files and a :class:`~lethe.memory_store.MemoryStore`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lethe.memory_store import MemoryStore


_ANCHOR_RE = re.compile(
    r"<!--\s*session:(?P<session>\S*?)\s+turn:(?P<turn>\S*?)\s+transcript:(?P<transcript>.*?)\s*-->"
)
_HEADING_RE = re.compile(r"^(##+)\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """A single addressable unit of the markdown memory."""

    id: str            # first 16 hex chars of sha256(content) — stable across runs
    source: Path       # .md file the chunk came from
    heading: str       # nearest preceding heading, empty if top of file
    content: str       # the raw markdown text of the chunk (includes heading)

    @property
    def anchor(self) -> dict[str, str] | None:
        return parse_anchor(self.content)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_hash(content: str) -> str:
    """Stable 16-char hex id derived from the chunk's normalized content."""
    normalized = content.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def parse_anchor(chunk_text: str) -> dict[str, str] | None:
    """Return ``{session, turn, transcript}`` from the first anchor in the chunk,
    or ``None`` if no anchor is present.
    """
    m = _ANCHOR_RE.search(chunk_text)
    if not m:
        return None
    return {
        "session": m.group("session"),
        "turn": m.group("turn"),
        "transcript": m.group("transcript"),
    }


def embed_content(chunk_text: str) -> str:
    """Return the chunk content with heading and anchor lines removed.

    The progressive-disclosure anchor (``<!-- session:UUID turn:UUID
    transcript:/long/path -->``) is ~150 chars of near-identical UUID noise
    that every chunk written by the Stop hook carries. Left in place it
    dominates the bi-encoder's pooled embedding, collapsing cosine
    similarities above the 0.95 dedup threshold and turning retrieval into
    "every chunk looks like every other chunk."

    The anchor is still preserved inside ``chunk_map.json`` so ``lethe
    expand`` and ``parse_anchor`` can recover it for deep-dive lookups.
    """
    kept: list[str] = []
    for line in chunk_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _has_body(body: str) -> bool:
    """True if ``body`` contains any non-heading, non-anchor text.

    Empty-section chunks (just a heading, optionally followed by an anchor
    comment or blank lines) add noise to retrieval without carrying signal —
    they often appear when SessionStart fires repeatedly or a turn produces
    no summary. Skip them at index time.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):  # any markdown heading, including `# Title`
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        return True
    return False


def split_into_chunks(md_text: str, source: Path) -> list[Chunk]:
    """Split ``md_text`` on ``##`` / ``###`` headings.

    The leading block (anything before the first ``##`` heading) is returned
    as an anonymous chunk if it contains non-whitespace text. Top-level ``#``
    headings are treated as file titles, not chunk separators. Heading-only
    sections (no bullets under them) are dropped.
    """
    lines = md_text.splitlines()
    chunks: list[Chunk] = []
    buf: list[str] = []
    current_heading = ""

    def flush() -> None:
        if not buf:
            return
        body = "\n".join(buf).strip()
        if not body or not _has_body(body):
            return
        chunks.append(
            Chunk(
                id=chunk_hash(body),
                source=source,
                heading=current_heading,
                content=body,
            )
        )

    for line in lines:
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) >= 2:  # ## or deeper, never plain #
            flush()
            buf = [line]
            current_heading = m.group(2).strip()
        else:
            buf.append(line)
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class MarkdownStore:
    """Bridges a directory of markdown files to a :class:`MemoryStore`.

    The canonical source of truth is the markdown on disk. A tiny JSON file
    (``index/chunk_map.json``) keeps the ``chunk_id -> raw content`` mapping
    so ``lethe expand`` can return the full section.
    """

    def __init__(self, memory_dir: Path, index_dir: Path) -> None:
        self.memory_dir = Path(memory_dir)
        self.index_dir = Path(index_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    # ---- Scanning ----------------------------------------------------------

    def iter_files(self) -> Iterable[Path]:
        return sorted(self.memory_dir.glob("*.md"))

    def scan(self) -> list[Chunk]:
        """Parse every markdown file in ``memory_dir`` into chunks."""
        out: list[Chunk] = []
        for f in self.iter_files():
            try:
                text = f.read_text(encoding="utf-8")
            except OSError:
                continue
            out.extend(split_into_chunks(text, f))
        return out

    # ---- Reindex -----------------------------------------------------------

    def reindex(self, store: MemoryStore) -> dict[str, int]:
        """Sync markdown chunks into ``store``.

        Adds new chunks (passing ``entry_id = chunk.id``), removes chunks that
        no longer appear in any markdown file, and leaves unchanged chunks
        alone. Returns a counts dict: ``{added, removed, unchanged, total}``.
        """
        chunks = self.scan()
        current_ids = {c.id for c in chunks}

        added = 0
        unchanged = 0
        removed = 0

        for chunk in chunks:
            if chunk.id in store.entries:
                unchanged += 1
                continue
            inserted = store.add(
                embed_content(chunk.content),
                entry_id=chunk.id,
                session_id=chunk.source.stem,
            )
            if inserted is not None:
                added += 1
            else:
                unchanged += 1

        for old_id in set(store.entries) - current_ids:
            if old_id in store.entries:
                store.db.delete_entry(old_id)
                store.entries.pop(old_id, None)
                store._embeddings.pop(old_id, None)
                removed += 1

        if removed:
            store._rebuild_index()

        return {
            "added": added,
            "removed": removed,
            "unchanged": unchanged,
            "total": len(chunks),
        }

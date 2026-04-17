"""Per-user project registry at ``~/.lethe/projects.json``.

``lethe search --all`` reads this file to know which projects to ATTACH.
``lethe index`` appends to it unless the user opts out via
``--no-register``, ``auto_register = false`` in ``config.toml``, or
``LETHE_DISABLE_GLOBAL_REGISTRY=1``.

Only absolute project root paths are stored — no contents, no hashes of
markdown. Privacy-sensitive users should disable via the env var.
"""
from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REGISTRY_VERSION = 1


def _registry_dir() -> Path:
    return Path(os.environ.get("HOME", "~")).expanduser() / ".lethe"


def _registry_path() -> Path:
    return _registry_dir() / "projects.json"


def slugify(root: Path) -> str:
    """Stable, FS + DuckDB-safe identifier for a project root.

    ``p_<sanitized-basename>_<sha1[:8]>``. Hash of the absolute path avoids
    collisions when two projects share a basename.
    """
    resolved = str(Path(root).resolve())
    name = re.sub(r"[^A-Za-z0-9]", "_", Path(resolved).name).strip("_") or "proj"
    h = hashlib.sha1(resolved.encode()).hexdigest()[:8]
    return f"p_{name}_{h}"


@dataclass(frozen=True)
class ProjectEntry:
    root: Path
    slug: str
    registered_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "slug": self.slug,
            "registered_at": self.registered_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "ProjectEntry":
        return cls(
            root=Path(d["root"]),
            slug=str(d.get("slug") or slugify(Path(d["root"]))),
            registered_at=str(d.get("registered_at", "")),
        )


def is_disabled() -> bool:
    return os.environ.get("LETHE_DISABLE_GLOBAL_REGISTRY", "").strip() not in ("", "0", "false", "False")


def load() -> list[ProjectEntry]:
    path = _registry_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    projects = data.get("projects", []) if isinstance(data, dict) else []
    out: list[ProjectEntry] = []
    for p in projects:
        if isinstance(p, dict) and "root" in p:
            out.append(ProjectEntry.from_dict(p))
    return out


def _save(entries: Iterable[ProjectEntry]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": REGISTRY_VERSION,
        "projects": [e.to_dict() for e in entries],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _with_registry_lock():
    """Return a file-lock context manager over the registry directory.

    Serializes concurrent registry mutations so two ``lethe index`` calls
    in different projects can't trample the JSON file.
    """
    from lethe._lock import acquire as _acquire

    _registry_dir().mkdir(parents=True, exist_ok=True)
    return _acquire(_registry_dir() / "registry.lock")


def register(root: Path) -> ProjectEntry:
    """Idempotently add ``root`` to the registry. Returns the entry."""
    resolved = Path(root).resolve()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with _with_registry_lock():
        entries = load()
        for existing in entries:
            if Path(existing.root).resolve() == resolved:
                return existing
        entry = ProjectEntry(root=resolved, slug=slugify(resolved), registered_at=now)
        entries.append(entry)
        _save(entries)
        return entry


def unregister(root_or_slug: str) -> bool:
    """Remove a project by root path or slug. Returns True if removed."""
    target = str(Path(root_or_slug).resolve()) if Path(root_or_slug).exists() else root_or_slug
    with _with_registry_lock():
        entries = load()
        kept = [
            e for e in entries
            if str(Path(e.root).resolve()) != target and e.slug != root_or_slug
        ]
        if len(kept) == len(entries):
            return False
        _save(kept)
        return True


def prune() -> list[ProjectEntry]:
    """Drop entries whose root directories no longer exist. Returns the kept entries."""
    with _with_registry_lock():
        entries = load()
        kept = [e for e in entries if Path(e.root).exists()]
        if len(kept) != len(entries):
            _save(kept)
        return kept


def find(name: str) -> ProjectEntry | None:
    """Find a project by slug or exact root path."""
    entries = load()
    for e in entries:
        if e.slug == name or str(e.root) == name:
            return e
    return None

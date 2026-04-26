"""fcntl-based file lock for serializing concurrent ``lethe`` CLI invocations.

Claude Code hooks (SessionStart, UserPromptSubmit, Stop, SessionEnd) fire in
quick succession and can overlap. DuckDB is single-writer-per-process across
the file, so two ``lethe index`` calls racing on ``lethe.duckdb`` would get a
lock error. This wrapper turns that race into well-mannered serialization.
"""
from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


DEFAULT_TIMEOUT_S = 30.0


class LockTimeout(RuntimeError):
    pass


@contextmanager
def acquire(lock_path: Path, timeout: float = DEFAULT_TIMEOUT_S) -> Iterator[None]:
    """Blocking ``flock`` with a timeout. Creates parent dir + lockfile if missing."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = open(lock_path, "w")
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise LockTimeout(
                        f"could not acquire {lock_path} within {timeout}s"
                    )
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            except OSError:
                # Lock file may have been rmtree'd (e.g. during `lethe reset`).
                pass
    finally:
        fd.close()

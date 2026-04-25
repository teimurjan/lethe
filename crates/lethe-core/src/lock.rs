//! File lock — port of `src/lethe/_lock.py`.
//!
//! Mirrors the polling semantics of the Python helper: try to acquire
//! an exclusive lock non-blockingly, sleep 50 ms between retries, fail
//! after `timeout` seconds.
//!
//! Backed by `fs2`'s `try_lock_exclusive` which wraps `flock(LOCK_EX | LOCK_NB)`
//! on Unix and `LockFileEx` on Windows. The lock is released on
//! `LockGuard` drop.

use std::fs::File;
use std::path::Path;
use std::time::{Duration, Instant};

use fs2::FileExt;

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Hold this guard to keep the lock; drop it to release.
#[derive(Debug)]
pub struct LockGuard {
    file: Option<File>,
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        if let Some(file) = self.file.take() {
            // Mirror Python's silent OSError swallow on unlock — the
            // lock file may have been removed by `lethe reset` while we
            // held it.
            let _ = FileExt::unlock(&file);
        }
    }
}

/// Acquire an exclusive lock on `lock_path`, polling up to `timeout`.
///
/// Creates the parent directory and the lock file if needed.
pub fn acquire(lock_path: &Path, timeout: Duration) -> Result<LockGuard, crate::Error> {
    if let Some(parent) = lock_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock_path)?;

    let deadline = Instant::now() + timeout;
    loop {
        match FileExt::try_lock_exclusive(&file) {
            Ok(()) => {
                return Ok(LockGuard { file: Some(file) });
            }
            Err(e) => {
                // fs2 returns the OS error directly; on Unix this is
                // EWOULDBLOCK / EAGAIN when the lock is held.
                if Instant::now() > deadline {
                    return Err(crate::Error::Locked(format!(
                        "could not acquire {}: {e}",
                        lock_path.display()
                    )));
                }
                std::thread::sleep(POLL_INTERVAL);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn acquires_and_releases_on_drop() {
        let dir = tempdir().unwrap();
        let lock = dir.path().join("lethe.lock");
        let guard = acquire(&lock, DEFAULT_TIMEOUT).unwrap();
        // While guard is alive, another non-blocking attempt fails.
        let f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock)
            .unwrap();
        assert!(FileExt::try_lock_exclusive(&f).is_err());
        drop(guard);
        // After release, a second acquire succeeds quickly.
        let g2 = acquire(&lock, Duration::from_millis(500)).unwrap();
        drop(g2);
    }

    #[test]
    fn timeout_returns_locked_error() {
        let dir = tempdir().unwrap();
        let lock = dir.path().join("lethe.lock");
        let _held = acquire(&lock, DEFAULT_TIMEOUT).unwrap();
        let res = acquire(&lock, Duration::from_millis(100));
        assert!(matches!(res, Err(crate::Error::Locked(_))));
    }

    #[test]
    fn creates_parent_dir() {
        let dir = tempdir().unwrap();
        let nested = dir.path().join("a/b/c").join("lethe.lock");
        let g = acquire(&nested, DEFAULT_TIMEOUT).unwrap();
        assert!(nested.exists());
        drop(g);
    }
}

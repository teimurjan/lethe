//! Per-user project registry — port of `src/lethe/_registry.py`.
//!
//! The registry lives at `~/.lethe/projects.json` and lists every
//! project (identified by absolute root path + `slug = p_<sanitized>_<sha1[:8]>`)
//! that should be searched when `lethe search --all` runs. Mutation
//! goes through a file lock at `~/.lethe/registry.lock` so concurrent
//! `lethe index` calls don't trample the JSON.

use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha1::{Digest as _, Sha1};

use crate::lock;

const REGISTRY_VERSION: u32 = 1;

/// Resolve the user-level registry directory (`$HOME/.lethe`).
pub fn registry_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("~"), PathBuf::from);
    home.join(".lethe")
}

pub fn registry_path() -> PathBuf {
    registry_dir().join("projects.json")
}

fn registry_lock_path() -> PathBuf {
    registry_dir().join("registry.lock")
}

/// `LETHE_DISABLE_GLOBAL_REGISTRY` env var honored verbatim.
#[must_use]
pub fn is_disabled() -> bool {
    match std::env::var("LETHE_DISABLE_GLOBAL_REGISTRY") {
        Ok(s) => {
            let s = s.trim();
            !matches!(s, "" | "0" | "false" | "False")
        }
        Err(_) => false,
    }
}

/// Stable, FS + DuckDB-safe identifier for a project root.
///
/// `p_<sanitized-basename>_<sha1[:8]>`. Hash is over the *resolved*
/// (canonicalized when possible) absolute path so two projects sharing
/// a basename collide-proofly. Bit-faithful to the Python helper.
pub fn slugify(root: &Path) -> String {
    let resolved = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let resolved_str = resolved.to_string_lossy().into_owned();

    let basename = resolved
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let sanitized = sanitize(&basename);

    let mut hasher = Sha1::new();
    hasher.update(resolved_str.as_bytes());
    let hash = hex::encode(hasher.finalize());

    format!("p_{}_{}", sanitized, &hash[..8])
}

fn sanitize(name: &str) -> String {
    // Equivalent of Python's `re.sub(r'[^A-Za-z0-9]', '_', name).strip('_')`,
    // falling back to "proj" when empty.
    let replaced: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let trimmed = replaced.trim_matches('_').to_owned();
    if trimmed.is_empty() {
        "proj".to_owned()
    } else {
        trimmed
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectEntry {
    pub root: PathBuf,
    pub slug: String,
    /// ISO-8601 with seconds precision (matches Python's
    /// `datetime.now(UTC).isoformat(timespec="seconds")`).
    pub registered_at: String,
}

/// `slug` derived from `root` if absent on disk; useful for legacy entries.
fn ensure_slug(entry: &mut ProjectEntry) {
    if entry.slug.is_empty() {
        entry.slug = slugify(&entry.root);
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct OnDisk {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    projects: Vec<ProjectEntry>,
}

/// Read the registry, returning an empty list on missing-or-malformed.
/// Matches Python: malformed JSON yields `[]` rather than raising.
#[must_use]
pub fn load() -> Vec<ProjectEntry> {
    let path = registry_path();
    let Ok(data) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    let mut parsed: OnDisk = serde_json::from_str(&data).unwrap_or_default();
    for p in &mut parsed.projects {
        ensure_slug(p);
    }
    parsed.projects
}

fn save_unlocked(entries: &[ProjectEntry]) -> Result<(), crate::Error> {
    let dir = registry_dir();
    std::fs::create_dir_all(&dir)?;
    let payload = OnDisk {
        version: REGISTRY_VERSION,
        projects: entries.to_vec(),
    };
    let json = serde_json::to_string_pretty(&payload)?;
    let path = registry_path();
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Idempotently add `root` to the registry, returning the entry.
pub fn register(root: &Path) -> Result<ProjectEntry, crate::Error> {
    let resolved = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    let now = current_iso8601_seconds();
    let _guard = lock::acquire(&registry_lock_path(), Duration::from_secs(30))?;
    let mut entries = load();
    for existing in &entries {
        if existing
            .root
            .canonicalize()
            .unwrap_or_else(|_| existing.root.clone())
            == resolved
        {
            return Ok(existing.clone());
        }
    }
    let entry = ProjectEntry {
        root: resolved.clone(),
        slug: slugify(&resolved),
        registered_at: now,
    };
    entries.push(entry.clone());
    save_unlocked(&entries)?;
    Ok(entry)
}

/// Remove a project by root path or slug. Returns `true` if removed.
pub fn unregister(root_or_slug: &str) -> Result<bool, crate::Error> {
    let target_path = Path::new(root_or_slug);
    let target_canon = target_path
        .canonicalize()
        .ok()
        .map(|p| p.to_string_lossy().into_owned());
    let _guard = lock::acquire(&registry_lock_path(), Duration::from_secs(30))?;
    let entries = load();
    let kept: Vec<_> = entries
        .iter()
        .filter(|e| {
            let root_str = e
                .root
                .canonicalize()
                .unwrap_or_else(|_| e.root.clone())
                .to_string_lossy()
                .into_owned();
            !(matches!(target_canon.as_deref(), Some(t) if t == root_str) || e.slug == root_or_slug)
        })
        .cloned()
        .collect();
    if kept.len() == entries.len() {
        return Ok(false);
    }
    save_unlocked(&kept)?;
    Ok(true)
}

/// Drop entries whose root no longer exists on disk. Returns kept entries.
pub fn prune() -> Result<Vec<ProjectEntry>, crate::Error> {
    let _guard = lock::acquire(&registry_lock_path(), Duration::from_secs(30))?;
    let entries = load();
    let kept: Vec<_> = entries.into_iter().filter(|e| e.root.exists()).collect();
    save_unlocked(&kept)?;
    Ok(kept)
}

/// Look up a project by exact slug or exact root path.
#[must_use]
pub fn find(name: &str) -> Option<ProjectEntry> {
    load()
        .into_iter()
        .find(|e| e.slug == name || e.root.to_string_lossy() == name)
}

fn current_iso8601_seconds() -> String {
    // Match `datetime.now(UTC).isoformat(timespec="seconds")` style:
    // YYYY-MM-DDTHH:MM:SS+00:00. Uses libc time so we don't need a
    // chrono dep just for a timestamp string.
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = unix_to_ymdhms(secs as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}+00:00")
}

/// Civil-time decomposition of a Unix timestamp (UTC). Avoids pulling
/// in `chrono` for this single use.
fn unix_to_ymdhms(t: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = t.div_euclid(86_400);
    let secs_of_day = t.rem_euclid(86_400) as u32;
    let h = secs_of_day / 3600;
    let mi = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    let (y, mo, d) = days_to_ymd(days);
    (y, mo, d, h, mi, s)
}

/// Days since 1970-01-01 → (year, month, day). Standard civil-time
/// algorithm from Howard Hinnant's date library.
#[allow(clippy::cast_possible_wrap)]
fn days_to_ymd(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Mutex, MutexGuard};
    use tempfile::tempdir;

    /// Tests in this module mutate the process-global `HOME` env var
    /// to redirect the registry away from the user's real `~/.lethe/`.
    /// `cargo test` parallelizes by default, so a shared mutex
    /// serializes them. The guard's drop order also restores the prior
    /// `HOME` value via `RestoreHome`.
    static HOME_LOCK: Mutex<()> = Mutex::new(());

    struct RestoreHome {
        prev: Option<String>,
        _guard: MutexGuard<'static, ()>,
    }

    impl Drop for RestoreHome {
        fn drop(&mut self) {
            match &self.prev {
                Some(p) => std::env::set_var("HOME", p),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    fn redirect_home_to(dir: &Path) -> RestoreHome {
        let guard = HOME_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var("HOME").ok();
        std::env::set_var("HOME", dir);
        RestoreHome {
            prev,
            _guard: guard,
        }
    }

    #[test]
    fn slugify_format_matches_python() {
        // Python: p_<sanitized>_<sha1[:8]>. Sanitize keeps alnum + _,
        // strips leading/trailing _, falls back to "proj" if empty.
        let dir = tempdir().unwrap();
        let path = dir.path();
        let slug = slugify(path);
        assert!(slug.starts_with("p_"));
        // last 8 chars are hex.
        let hash_part = &slug[slug.len() - 8..];
        assert!(hash_part.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sanitize_empty_yields_proj() {
        assert_eq!(sanitize("___"), "proj");
        assert_eq!(sanitize(""), "proj");
        assert_eq!(sanitize("a-b.c"), "a_b_c");
    }

    #[test]
    fn register_load_unregister_round_trip() {
        let home = tempdir().unwrap();
        let project = tempdir().unwrap();
        let _restore = redirect_home_to(home.path());

        let entry = register(project.path()).unwrap();
        let loaded = load();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].slug, entry.slug);

        let removed = unregister(&entry.slug).unwrap();
        assert!(removed);
        assert!(load().is_empty());
    }

    #[test]
    fn malformed_json_yields_empty() {
        let home = tempdir().unwrap();
        let _restore = redirect_home_to(home.path());

        std::fs::create_dir_all(registry_dir()).unwrap();
        std::fs::write(registry_path(), "not json").unwrap();
        assert!(load().is_empty());
    }

    #[test]
    fn prune_removes_missing_paths() {
        let home = tempdir().unwrap();
        let project_a = tempdir().unwrap();
        let _restore = redirect_home_to(home.path());

        let _a = register(project_a.path()).unwrap();
        let mut entries = load();
        entries.push(ProjectEntry {
            root: PathBuf::from("/nonexistent/does/not/exist"),
            slug: "p_dead_deadbeef".into(),
            registered_at: "2026-01-01T00:00:00+00:00".into(),
        });
        save_unlocked(&entries).unwrap();

        let kept = prune().unwrap();
        assert_eq!(kept.len(), 1);
        assert!(kept[0].root.exists());
    }
}

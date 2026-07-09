//! TUI app state.
//!
//! Interaction model (no focus panes, no Tab): typing always edits the
//! search box, and the current [`Scope`] decides what the arrow keys
//! drive. While browsing the project list (all-projects scope, no
//! results) arrows move projects, Enter opens one, and Ctrl+D deletes
//! one. Inside a project, arrows move its memory list (all memories by
//! default, or search hits), Ctrl+C copies the highlighted one, and Esc
//! goes back to the project list.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use lethe_core::db::MemoryDb;
use lethe_core::markdown_store::parse_anchor;
use lethe_core::registry::{self, ProjectEntry};

use crate::search_worker::{self, Phase, SearchOutput, SearchRequest};

/// Cap on how many memories the browse view loads per project.
const BROWSE_LIMIT: usize = 500;

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub total: usize,
    /// Memories whose source transcript is a Claude Code session.
    pub claude: usize,
    /// Memories whose source transcript is a Codex rollout.
    pub codex: usize,
    /// `(display_name, count)` sorted descending by count.
    pub by_project: Vec<(String, usize)>,
}

/// Friendly project name — the basename of its root — instead of the
/// registry slug (`p_<base>_<hash>`), which reads as noise in the UI.
pub fn project_name(root: &Path) -> String {
    root.file_name().map_or_else(
        || root.to_string_lossy().into_owned(),
        |s| s.to_string_lossy().into_owned(),
    )
}

/// Classify a memory by its anchor's transcript path. Codex rollouts are
/// named `rollout-*.jsonl` and live under `sessions/`; everything else is
/// Claude Code. `None` when the content carries no anchor.
fn source_is_codex(content: &str) -> Option<bool> {
    let a = parse_anchor(content)?;
    let base = Path::new(&a.transcript)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    Some(base.starts_with("rollout-") || a.transcript.contains("/sessions/"))
}

fn index_db_path(slug: &str) -> PathBuf {
    registry::registry_dir()
        .join("index")
        .join(slug)
        .join("lethe.duckdb")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Error,
}

#[derive(Debug, Clone)]
pub struct Toast {
    pub msg: String,
    pub kind: ToastKind,
    pub expires: Instant,
}

const TOAST_TTL: Duration = Duration::from_millis(2500);

#[derive(Debug, Clone)]
pub struct ResultRow {
    /// Source project root for cross-project hits; tags the result row
    /// with the friendly project name. `None` in single-project scope.
    pub project_root: Option<PathBuf>,
    #[allow(dead_code)]
    pub id: String,
    pub content: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub enum Scope {
    AllProjects,
    Single(ProjectEntry),
}

impl Scope {
    pub fn label(&self) -> String {
        match self {
            Scope::AllProjects => "all projects".to_owned(),
            Scope::Single(e) => project_name(&e.root),
        }
    }
}

pub struct App {
    pub scope: Scope,
    pub projects: Vec<ProjectEntry>,
    pub project_selection: usize,

    pub search_input: String,
    pub last_query: String,
    pub searching: bool,
    pub search_phase: Option<Phase>,

    pub results: Vec<ResultRow>,
    pub result_selection: usize,
    /// True when `results` is the full memory list (empty query), false
    /// when it's search hits — drives the "no memories" vs "no results"
    /// empty message and whether rows show a score.
    pub browsing: bool,

    pub detail: Option<String>,

    /// Project index pending delete confirmation (`d` pressed once).
    pub pending_delete: Option<usize>,

    /// Transient bottom-right notification (copy success, errors).
    pub toast: Option<Toast>,

    pub needs_redraw: bool,

    /// Per-project memory counts + per-source totals. `None` until the
    /// bg thread reports.
    pub stats: Option<Stats>,
    pub stats_rx: mpsc::Receiver<Stats>,

    /// Channels owned by the background search worker.
    pub search_tx: mpsc::Sender<SearchRequest>,
    pub search_rx: mpsc::Receiver<SearchOutput>,
}

impl App {
    pub fn new() -> Self {
        let projects = registry::load();
        let (search_tx, search_rx) = search_worker::spawn();
        let stats_rx = spawn_stats(projects.clone());
        Self {
            scope: Scope::AllProjects,
            projects,
            project_selection: 0,
            search_input: String::new(),
            last_query: String::new(),
            searching: false,
            search_phase: None,
            results: Vec::new(),
            result_selection: 0,
            browsing: false,
            detail: None,
            pending_delete: None,
            toast: None,
            needs_redraw: false,
            stats: None,
            stats_rx,
            search_tx,
            search_rx,
        }
    }

    /// True while browsing the project list — all-projects scope with no
    /// results on the right. Arrows move projects here; otherwise they
    /// move the memory/results list. This is the whole "focus" model.
    pub fn nav_projects(&self) -> bool {
        matches!(self.scope, Scope::AllProjects) && self.results.is_empty()
    }

    /// Back one level: leave any single-project scope and clear whatever's
    /// on the right (memories, search hits, detail), returning to the
    /// project list. A no-op when already there.
    pub fn escape(&mut self) {
        self.pending_delete = None;
        self.scope = Scope::AllProjects;
        self.results.clear();
        self.detail = None;
        self.browsing = false;
        self.search_input.clear();
        self.last_query.clear();
    }

    pub fn show_toast(&mut self, msg: impl Into<String>, kind: ToastKind) {
        self.toast = Some(Toast {
            msg: msg.into(),
            kind,
            expires: Instant::now() + TOAST_TTL,
        });
    }

    pub fn poll_toast(&mut self) {
        if let Some(t) = &self.toast {
            if Instant::now() >= t.expires {
                self.toast = None;
            }
        }
    }

    /// Arrow keys: move the project list in all-projects scope, else the
    /// memory list.
    pub fn arrow(&mut self, delta: isize) {
        self.pending_delete = None;
        if self.nav_projects() {
            if self.projects.is_empty() {
                return;
            }
            let len = self.projects.len() as isize;
            let next = (self.project_selection as isize + delta).rem_euclid(len);
            self.project_selection = next as usize;
        } else {
            if self.results.is_empty() {
                return;
            }
            let len = self.results.len() as isize;
            let next = (self.result_selection as isize + delta).rem_euclid(len);
            self.result_selection = next as usize;
            self.refresh_detail_from_highlight();
        }
    }

    /// Enter: run a search when the box has text; otherwise open the
    /// highlighted project (from the project list) or reload all its
    /// memories (already inside one).
    pub fn on_enter(&mut self) {
        self.pending_delete = None;
        if !self.search_input.trim().is_empty() {
            self.submit_search();
        } else if self.nav_projects() {
            self.open_selected_project();
        } else if matches!(self.scope, Scope::Single(_)) {
            self.load_all_memories();
        }
    }

    fn open_selected_project(&mut self) {
        let Some(p) = self.projects.get(self.project_selection).cloned() else {
            return;
        };
        self.scope = Scope::Single(p);
        self.search_input.clear();
        self.last_query.clear();
        self.detail = None;
        self.load_all_memories();
    }

    /// Load every memory of the current single-scope project (newest DB
    /// rows first, capped) into `results`. DB-only, no encoders.
    pub fn load_all_memories(&mut self) {
        let Scope::Single(entry) = &self.scope else {
            return;
        };
        self.last_query.clear();
        self.browsing = true;
        let rows = MemoryDb::open_with_mode(index_db_path(&entry.slug), true)
            .and_then(|db| db.load_all_entries())
            .map(|mut rows| {
                rows.reverse(); // newest inserts last → show them first
                rows.truncate(BROWSE_LIMIT);
                rows.into_iter()
                    .map(|r| ResultRow {
                        project_root: None,
                        id: r.id,
                        content: r.content,
                        score: 0.0,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        self.results = rows;
        self.result_selection = 0;
        if self.results.is_empty() {
            self.detail = None;
        } else {
            self.refresh_detail_from_highlight();
        }
    }

    /// Arm / confirm deletion of the highlighted project. First press
    /// arms (toast); second confirms.
    pub fn request_or_confirm_delete(&mut self) {
        if !self.nav_projects() {
            return;
        }
        let Some(p) = self.projects.get(self.project_selection) else {
            return;
        };
        let name = project_name(&p.root);
        if self.pending_delete == Some(self.project_selection) {
            self.pending_delete = None;
            self.delete_project(self.project_selection);
        } else {
            self.pending_delete = Some(self.project_selection);
            self.show_toast(
                format!("delete '{name}'? press Ctrl+D again"),
                ToastKind::Info,
            );
        }
    }

    fn delete_project(&mut self, idx: usize) {
        let Some(p) = self.projects.get(idx).cloned() else {
            return;
        };
        let name = project_name(&p.root);
        // Unregister and remove the index dir; transcripts are untouched,
        // so a later `lethe index` can rebuild it.
        let _ = registry::unregister(&p.slug);
        let dir = registry::registry_dir().join("index").join(&p.slug);
        let _ = std::fs::remove_dir_all(&dir);

        self.projects = registry::load();
        if self.projects.is_empty() {
            self.project_selection = 0;
        } else {
            self.project_selection = self.project_selection.min(self.projects.len() - 1);
        }
        // If the deleted project was open, pop back to the list.
        if matches!(&self.scope, Scope::Single(e) if e.slug == p.slug) {
            self.scope = Scope::AllProjects;
            self.results.clear();
            self.detail = None;
            self.browsing = false;
        }
        self.show_toast(format!("deleted '{name}'"), ToastKind::Info);
    }

    pub fn submit_search(&mut self) {
        let query = self.search_input.trim().to_owned();
        if query.is_empty() || self.searching {
            return;
        }
        self.last_query.clone_from(&query);
        self.searching = true;
        self.browsing = false;
        self.search_phase = None;
        self.results.clear();
        self.detail = None;
        let _ = self.search_tx.send(SearchRequest {
            query,
            scope: self.scope.clone(),
            top_k: 10,
        });
    }

    pub fn poll_stats(&mut self) {
        if self.stats.is_some() {
            return;
        }
        if let Ok(s) = self.stats_rx.try_recv() {
            self.stats = Some(s);
        }
    }

    pub fn copy_selected_to_clipboard(&mut self) {
        let Some(row) = self.results.get(self.result_selection) else {
            return;
        };
        let content = row.content.clone();
        let len = content.len();
        match arboard::Clipboard::new().and_then(|mut cb| cb.set_text(content)) {
            Ok(()) => self.show_toast(format!("✓ copied {len} chars"), ToastKind::Info),
            Err(e) => self.show_toast(format!("clipboard error: {e}"), ToastKind::Error),
        }
    }

    pub fn poll_search_results(&mut self) {
        while let Ok(output) = self.search_rx.try_recv() {
            match output {
                SearchOutput::Phase(p) => {
                    self.search_phase = Some(p);
                }
                SearchOutput::Hits(rows) => {
                    self.searching = false;
                    self.search_phase = None;
                    self.needs_redraw = true;
                    let n = rows.len();
                    self.results = rows;
                    self.result_selection = 0;
                    if n == 0 {
                        self.detail = None;
                    } else {
                        self.refresh_detail_from_highlight();
                    }
                }
                SearchOutput::Error(e) => {
                    self.searching = false;
                    self.search_phase = None;
                    self.needs_redraw = true;
                    self.show_toast(format!("error: {e}"), ToastKind::Error);
                }
            }
        }
    }

    pub fn refresh_detail_from_highlight(&mut self) {
        if let Some(row) = self.results.get(self.result_selection) {
            self.detail = Some(row.content.clone());
        }
    }
}

/// Compute per-project memory counts + per-source (Claude Code vs Codex)
/// totals off-thread. Opens each project's DuckDB file directly (no
/// embeddings, no encoders) so first paint is fast even with many
/// registered projects.
fn spawn_stats(projects: Vec<ProjectEntry>) -> mpsc::Receiver<Stats> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut by_project: Vec<(String, usize)> = Vec::with_capacity(projects.len());
        let mut total = 0usize;
        let mut claude = 0usize;
        let mut codex = 0usize;
        for p in projects {
            let db_path = index_db_path(&p.slug);
            if !db_path.exists() {
                continue;
            }
            let Ok(db) = MemoryDb::open_with_mode(&db_path, true) else {
                continue;
            };
            let Ok(rows) = db.load_all_entries() else {
                continue;
            };
            for row in &rows {
                match source_is_codex(&row.content) {
                    Some(true) => codex += 1,
                    Some(false) => claude += 1,
                    None => {}
                }
            }
            total += rows.len();
            by_project.push((project_name(&p.root), rows.len()));
        }
        by_project.sort_by(|a, b| b.1.cmp(&a.1));
        let _ = tx.send(Stats {
            total,
            claude,
            codex,
            by_project,
        });
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchored(transcript: &str) -> String {
        format!("<!-- session:s turn:t transcript:{transcript} -->\nUSER:\nq\n\nASSISTANT:\na")
    }

    #[test]
    fn classifies_codex_vs_claude() {
        assert_eq!(
            source_is_codex(&anchored(
                "/Users/x/.codex/sessions/2026/07/rollout-2026-abc.jsonl"
            )),
            Some(true)
        );
        assert_eq!(
            source_is_codex(&anchored(
                "/Users/x/.claude/projects/-Users-x-repo/6f0e.jsonl"
            )),
            Some(false)
        );
        assert_eq!(source_is_codex("no anchor here"), None);
    }

    #[test]
    fn project_name_is_basename() {
        assert_eq!(project_name(Path::new("/Users/x/Projects/lethe")), "lethe");
    }
}

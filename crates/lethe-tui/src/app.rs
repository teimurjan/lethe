//! TUI app state.
//!
//! Interaction model: Tab / Shift+Tab cycle the project sidebar and load
//! the highlighted project's memories into the right pane; ↑/↓ move the
//! highlight within that memory list. Typing edits the search box; Enter
//! searches within the current project; Esc clears the search back to all
//! memories. Ctrl+C copies the highlighted memory, Ctrl+D deletes the
//! highlighted project, Ctrl+A opens the actions menu.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use lethe_core::db::MemoryDb;
use lethe_core::maintenance::{self, StaleReason, StaleTranscript};
use lethe_core::markdown_store::parse_anchor;
use lethe_core::registry::{self, ProjectEntry};

use crate::search_worker::{self, Phase, SearchQuery, WorkerOutput, WorkerRequest};

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

/// Labels for the actions menu, in order. Index maps to `activate_action`.
pub const ACTIONS: [&str; 5] = [
    "Index this project",
    "Index all projects",
    "Delete this project",
    "Delete empty projects",
    "Clean up dead transcripts",
];

/// A destructive action awaiting confirmation.
#[derive(Debug)]
pub enum PendingAction {
    DeleteProject(ProjectEntry),
    DeleteEmpty(Vec<ProjectEntry>),
}

#[derive(Debug)]
pub struct Confirm {
    pub title: String,
    pub lines: Vec<String>,
    pub action: PendingAction,
    /// Whether a `t` (also-delete-transcripts) choice is offered.
    pub allow_transcripts: bool,
}

/// A reviewable checklist of stale transcripts to delete.
#[derive(Debug)]
pub struct CleanupList {
    pub items: Vec<StaleTranscript>,
    pub selected: Vec<bool>,
    pub cursor: usize,
}

impl CleanupList {
    fn new(items: Vec<StaleTranscript>) -> Self {
        // Pre-select only the confident "repo gone" items; leave the
        // "no memories" ones (a live repo may just have pruned its
        // transcripts) for the user to opt into.
        let selected = items
            .iter()
            .map(|s| s.reason == StaleReason::RepoGone)
            .collect();
        Self {
            items,
            selected,
            cursor: 0,
        }
    }

    fn selected_items(&self) -> Vec<StaleTranscript> {
        self.items
            .iter()
            .zip(&self.selected)
            .filter(|(_, s)| **s)
            .map(|(it, _)| it.clone())
            .collect()
    }
}

/// Modal overlay state. `None` = the normal browse view.
#[derive(Debug)]
pub enum Overlay {
    /// Actions menu with the given cursor position.
    Actions(usize),
    /// Confirm a destructive action.
    Confirm(Confirm),
    /// Reviewable multi-select list of stale transcripts.
    Cleanup(CleanupList),
    /// A background worker is running; the string is the status label.
    Busy(String),
}

pub struct App {
    pub scope: Scope,
    pub projects: Vec<ProjectEntry>,
    pub project_selection: usize,
    pub overlay: Option<Overlay>,

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

    /// Channels owned by the background worker.
    pub search_tx: mpsc::Sender<WorkerRequest>,
    pub search_rx: mpsc::Receiver<WorkerOutput>,
}

impl App {
    pub fn new() -> Self {
        let projects = registry::load();
        let (search_tx, search_rx) = search_worker::spawn();
        let stats_rx = spawn_stats(projects.clone());
        let mut app = Self {
            scope: Scope::AllProjects,
            projects,
            project_selection: 0,
            overlay: None,
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
        };
        // Show the first project's memories immediately.
        app.open_current();
        app
    }

    /// Point the scope at the highlighted project and load all its
    /// memories. Called whenever the project selection changes so the
    /// right pane always reflects the sidebar.
    fn open_current(&mut self) {
        let Some(p) = self.projects.get(self.project_selection).cloned() else {
            self.scope = Scope::AllProjects;
            self.results.clear();
            self.detail = None;
            return;
        };
        self.scope = Scope::Single(p);
        self.load_all_memories();
    }

    /// Esc: clear the search box and show the current project's full
    /// memory list again.
    pub fn escape(&mut self) {
        self.pending_delete = None;
        self.search_input.clear();
        self.last_query.clear();
        self.open_current();
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

    /// ↑/↓ — move the highlight within the current project's memory list.
    pub fn arrow(&mut self, delta: isize) {
        self.pending_delete = None;
        if self.results.is_empty() {
            return;
        }
        let len = self.results.len() as isize;
        self.result_selection = (self.result_selection as isize + delta).rem_euclid(len) as usize;
        self.refresh_detail_from_highlight();
    }

    /// Tab / Shift+Tab — cycle the project sidebar and load its memories.
    pub fn cycle_project(&mut self, delta: isize) {
        self.pending_delete = None;
        if self.projects.is_empty() {
            return;
        }
        let len = self.projects.len() as isize;
        self.project_selection = (self.project_selection as isize + delta).rem_euclid(len) as usize;
        self.search_input.clear();
        self.open_current();
    }

    /// Enter: run a search within the current project when the box has
    /// text; otherwise reload all its memories.
    pub fn on_enter(&mut self) {
        self.pending_delete = None;
        if self.search_input.trim().is_empty() {
            self.open_current();
        } else {
            self.submit_search();
        }
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
        // Reflect the new selection (or clear when nothing's left).
        self.search_input.clear();
        self.open_current();
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
        let _ = self.search_tx.send(WorkerRequest::Search(SearchQuery {
            query,
            scope: self.scope.clone(),
            top_k: 10,
        }));
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
                WorkerOutput::Phase(p) => {
                    self.search_phase = Some(p);
                    // Reflect worker progress in the Busy overlay label.
                    if matches!(self.overlay, Some(Overlay::Busy(_))) {
                        self.overlay = Some(Overlay::Busy(format!("{}…", p.label())));
                    }
                }
                WorkerOutput::Hits(rows) => {
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
                WorkerOutput::Indexed { added, projects } => {
                    self.overlay = None;
                    self.search_phase = None;
                    self.needs_redraw = true;
                    self.show_toast(
                        format!("indexed {projects} project(s), +{added} memories"),
                        ToastKind::Info,
                    );
                    self.refresh_stats();
                    self.open_current();
                }
                WorkerOutput::Scanned(items) => {
                    self.search_phase = None;
                    self.needs_redraw = true;
                    if items.is_empty() {
                        self.overlay = None;
                        self.show_toast("nothing to clean up", ToastKind::Info);
                    } else {
                        self.overlay = Some(Overlay::Cleanup(CleanupList::new(items)));
                    }
                }
                WorkerOutput::Deleted(r) => {
                    self.overlay = None;
                    self.needs_redraw = true;
                    self.show_toast(
                        format!(
                            "deleted {} transcript(s), reclaimed {}",
                            r.transcripts,
                            maintenance::human_bytes(r.bytes)
                        ),
                        ToastKind::Info,
                    );
                    self.refresh_stats();
                }
                WorkerOutput::Error(e) => {
                    self.searching = false;
                    self.search_phase = None;
                    self.overlay = None;
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

    /// Recompute per-project + per-source counts after a mutation.
    fn refresh_stats(&mut self) {
        self.stats = None;
        self.stats_rx = spawn_stats(self.projects.clone());
    }

    // -------- actions menu / overlay --------

    /// Open the actions menu (Ctrl+A).
    pub fn open_actions(&mut self) {
        self.pending_delete = None;
        self.overlay = Some(Overlay::Actions(0));
    }

    /// Route a key to the active overlay. Returns without effect when no
    /// overlay is open.
    pub fn overlay_key(&mut self, code: crossterm::event::KeyCode) {
        use crossterm::event::KeyCode;
        match &mut self.overlay {
            Some(Overlay::Actions(cursor)) => match code {
                KeyCode::Up => *cursor = (*cursor + ACTIONS.len() - 1) % ACTIONS.len(),
                KeyCode::Down => *cursor = (*cursor + 1) % ACTIONS.len(),
                KeyCode::Enter => {
                    let idx = *cursor;
                    self.activate_action(idx);
                }
                KeyCode::Esc => self.overlay = None,
                _ => {}
            },
            Some(Overlay::Confirm(c)) => match code {
                KeyCode::Char('y') => self.confirm(false),
                KeyCode::Char('t') if c.allow_transcripts => self.confirm(true),
                KeyCode::Esc | KeyCode::Char('n') => self.overlay = None,
                _ => {}
            },
            Some(Overlay::Cleanup(list)) => match code {
                KeyCode::Up => {
                    list.cursor = list.cursor.saturating_sub(1);
                }
                KeyCode::Down => {
                    list.cursor = (list.cursor + 1).min(list.items.len().saturating_sub(1));
                }
                KeyCode::Char(' ') => {
                    if let Some(sel) = list.selected.get_mut(list.cursor) {
                        *sel = !*sel;
                    }
                }
                KeyCode::Char('a') => {
                    // Toggle all: select all unless everything is already
                    // selected, in which case clear.
                    let all = list.selected.iter().all(|s| *s);
                    for s in &mut list.selected {
                        *s = !all;
                    }
                }
                KeyCode::Enter => {
                    let chosen = list.selected_items();
                    if chosen.is_empty() {
                        self.show_toast("nothing selected", ToastKind::Info);
                    } else {
                        self.start_busy("deleting");
                        let _ = self.search_tx.send(WorkerRequest::DeleteStale(chosen));
                    }
                }
                KeyCode::Esc => self.overlay = None,
                _ => {}
            },
            Some(Overlay::Busy(_)) => {
                if code == KeyCode::Esc {
                    // Dismiss the modal; the worker keeps running and its
                    // result is applied when it lands.
                    self.overlay = None;
                }
            }
            None => {}
        }
    }

    fn activate_action(&mut self, idx: usize) {
        match idx {
            0 => {
                // Index this project.
                if let Scope::Single(entry) = &self.scope {
                    let entry = entry.clone();
                    self.start_busy("indexing");
                    let _ = self.search_tx.send(WorkerRequest::Index(vec![entry]));
                }
            }
            1 => {
                // Index all registered projects.
                let all = registry::load();
                if all.is_empty() {
                    self.overlay = None;
                    self.show_toast("no registered projects", ToastKind::Info);
                } else {
                    self.start_busy("indexing");
                    let _ = self.search_tx.send(WorkerRequest::Index(all));
                }
            }
            2 => {
                // Delete this project (confirm).
                if let Some(entry) = self.projects.get(self.project_selection).cloned() {
                    let name = project_name(&entry.root);
                    self.overlay = Some(Overlay::Confirm(Confirm {
                        title: format!("Delete project '{name}'?"),
                        lines: vec![
                            "y = remove from lethe   t = also delete transcripts".into(),
                            "Transcripts are on disk under ~/.claude & ~/.codex.".into(),
                        ],
                        action: PendingAction::DeleteProject(entry),
                        allow_transcripts: true,
                    }));
                }
            }
            3 => {
                // Delete empty projects (confirm).
                let empty = maintenance::empty_projects();
                if empty.is_empty() {
                    self.overlay = None;
                    self.show_toast("no empty projects", ToastKind::Info);
                } else {
                    let mut lines: Vec<String> = empty
                        .iter()
                        .map(|e| format!("• {}", project_name(&e.root)))
                        .collect();
                    lines.push(String::new());
                    lines.push("y = remove from lethe   t = also delete transcripts".into());
                    self.overlay = Some(Overlay::Confirm(Confirm {
                        title: format!("Delete {} empty project(s)?", empty.len()),
                        lines,
                        action: PendingAction::DeleteEmpty(empty),
                        allow_transcripts: true,
                    }));
                }
            }
            4 => {
                // Scan for dead transcripts (worker); confirm on result.
                self.start_busy("scanning");
                let _ = self.search_tx.send(WorkerRequest::Scan);
            }
            _ => self.overlay = None,
        }
    }

    fn start_busy(&mut self, label: &str) {
        self.overlay = Some(Overlay::Busy(format!("{label}…")));
        self.needs_redraw = true;
    }

    /// Execute the confirmed action. `with_transcripts` reflects y vs t.
    fn confirm(&mut self, with_transcripts: bool) {
        let Some(Overlay::Confirm(c)) = self.overlay.take() else {
            return;
        };
        match c.action {
            PendingAction::DeleteProject(entry) => {
                let _ = self.search_tx.send(WorkerRequest::ReleaseCaches);
                let r = maintenance::delete_project_data(&entry, with_transcripts);
                self.after_project_delete(r);
            }
            PendingAction::DeleteEmpty(entries) => {
                let _ = self.search_tx.send(WorkerRequest::ReleaseCaches);
                let mut total = maintenance::Reclaimed::default();
                for e in &entries {
                    let r = maintenance::delete_project_data(e, with_transcripts);
                    total.projects += r.projects;
                    total.transcripts += r.transcripts;
                    total.bytes += r.bytes;
                }
                self.after_project_delete(total);
            }
        }
    }

    fn after_project_delete(&mut self, r: maintenance::Reclaimed) {
        self.projects = registry::load();
        if self.projects.is_empty() {
            self.project_selection = 0;
        } else {
            self.project_selection = self.project_selection.min(self.projects.len() - 1);
        }
        self.search_input.clear();
        self.open_current();
        self.refresh_stats();
        self.show_toast(
            format!(
                "deleted {} project(s), reclaimed {}",
                r.projects,
                maintenance::human_bytes(r.bytes)
            ),
            ToastKind::Info,
        );
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

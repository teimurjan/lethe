//! TUI app state. Holds the current scope (all projects vs. one),
//! search input, results, detail pane content, and a background
//! search worker.

use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use lethe_core::db::MemoryDb;
use lethe_core::registry::{self, ProjectEntry};

use crate::search_worker::{self, Phase, SearchOutput, SearchRequest};

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub total: usize,
    /// `(slug, count)` sorted descending by count.
    pub by_project: Vec<(String, usize)>,
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

const TOAST_TTL: Duration = Duration::from_millis(2000);

/// Which pane currently owns focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Projects,
    Search,
    Results,
}

#[derive(Debug, Clone)]
pub struct ResultRow {
    pub project_slug: Option<String>,
    /// Carried for symmetry with the Python TUI's expand handler — not
    /// rendered yet but useful when "open in editor" lands.
    #[allow(dead_code)]
    pub project_root: Option<PathBuf>,
    /// Surfaced once `lethe-tui` learns to delegate to `lethe expand`.
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
            Scope::Single(e) => e.slug.clone(),
        }
    }
}

pub struct App {
    pub focus: Focus,
    pub scope: Scope,
    pub projects: Vec<ProjectEntry>,
    pub project_selection: usize,

    pub search_input: String,
    pub last_query: String,
    pub searching: bool,
    pub search_phase: Option<Phase>,

    pub results: Vec<ResultRow>,
    pub result_selection: usize,

    pub detail: Option<String>,

    /// Transient bottom-right notification (copy success, errors).
    pub toast: Option<Toast>,

    /// Set when the worker finishes a search; consumed by the event
    /// loop to force `terminal.clear()`. Cold-load downloads may emit
    /// stray bytes (`hf-hub` / `ort` warnings) that the ratatui buffer
    /// diffing won't know to overwrite — a one-shot full redraw fixes
    /// the corrupted screen.
    pub needs_redraw: bool,

    /// Per-project memory counts. `None` until the bg thread reports.
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
            focus: Focus::Search,
            scope: Scope::AllProjects,
            projects,
            project_selection: 0,
            search_input: String::new(),
            last_query: String::new(),
            searching: false,
            search_phase: None,
            results: Vec::new(),
            result_selection: 0,
            detail: None,
            toast: None,
            needs_redraw: false,
            stats: None,
            stats_rx,
            search_tx,
            search_rx,
        }
    }

    pub fn cycle_focus(&mut self, forward: bool) {
        let order = [Focus::Search, Focus::Projects, Focus::Results];
        let idx = order.iter().position(|f| *f == self.focus).unwrap_or(0);
        let next = if forward {
            (idx + 1) % order.len()
        } else {
            (idx + order.len() - 1) % order.len()
        };
        self.focus = order[next];
    }

    pub fn escape(&mut self) {
        if self.detail.is_some() {
            self.detail = None;
            return;
        }
        if matches!(self.scope, Scope::Single(_)) {
            self.scope = Scope::AllProjects;
            self.results.clear();
            self.detail = None;
            return;
        }
        // Already at the top; refocus the search input.
        self.focus = Focus::Search;
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

    pub fn move_cursor(&mut self, delta: isize) {
        match self.focus {
            Focus::Projects => {
                if self.projects.is_empty() {
                    return;
                }
                let len = self.projects.len() as isize;
                let next = (self.project_selection as isize + delta).rem_euclid(len);
                self.project_selection = next as usize;
            }
            Focus::Results => {
                if self.results.is_empty() {
                    return;
                }
                let len = self.results.len() as isize;
                let next = (self.result_selection as isize + delta).rem_euclid(len);
                self.result_selection = next as usize;
                self.refresh_detail_from_highlight();
            }
            Focus::Search => {}
        }
    }

    pub fn enter_selected_project(&mut self) {
        if let Some(p) = self.projects.get(self.project_selection) {
            self.scope = Scope::Single(p.clone());
            self.results.clear();
            self.detail = None;
            self.focus = Focus::Search;
        }
    }

    pub fn submit_search(&mut self) {
        let query = self.search_input.trim().to_owned();
        if query.is_empty() || self.searching {
            return;
        }
        self.last_query.clone_from(&query);
        self.searching = true;
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
                    // Cold-load downloads may have leaked bytes onto
                    // the alt-screen; force a full redraw on the next
                    // tick to reset the buffer.
                    self.needs_redraw = true;
                    let n = rows.len();
                    self.results = rows;
                    self.result_selection = 0;
                    if n == 0 {
                        self.detail = None;
                    } else {
                        self.focus = Focus::Results;
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

/// Compute per-project memory counts off-thread. Opens each project's
/// DuckDB file directly (no embeddings, no encoders) so first paint is
/// fast even with many registered projects.
fn spawn_stats(projects: Vec<ProjectEntry>) -> mpsc::Receiver<Stats> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut by_project: Vec<(String, usize)> = Vec::with_capacity(projects.len());
        let mut total = 0usize;
        for p in projects {
            let db_path = p.root.join(".lethe").join("index").join("lethe.duckdb");
            if !db_path.exists() {
                continue;
            }
            let count = MemoryDb::open(&db_path)
                .and_then(|db| db.count())
                .map(|c| c.max(0) as usize)
                .unwrap_or(0);
            total += count;
            by_project.push((p.slug, count));
        }
        by_project.sort_by(|a, b| b.1.cmp(&a.1));
        let _ = tx.send(Stats { total, by_project });
    });
    rx
}

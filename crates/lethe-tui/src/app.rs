//! TUI app state. Holds the current scope (all projects vs. one),
//! search input, results, detail pane content, and a background
//! search worker.

use std::path::PathBuf;
use std::sync::mpsc;

use lethe_core::registry::{self, ProjectEntry};

use crate::search_worker::{self, SearchOutput, SearchRequest};

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

    pub results: Vec<ResultRow>,
    pub result_selection: usize,
    pub status: String,

    pub detail: Option<String>,

    /// Channels owned by the background search worker.
    pub search_tx: mpsc::Sender<SearchRequest>,
    pub search_rx: mpsc::Receiver<SearchOutput>,
}

impl App {
    pub fn new() -> Self {
        let projects = registry::load();
        let (search_tx, search_rx) = search_worker::spawn();
        Self {
            focus: Focus::Search,
            scope: Scope::AllProjects,
            projects,
            project_selection: 0,
            search_input: String::new(),
            last_query: String::new(),
            searching: false,
            results: Vec::new(),
            result_selection: 0,
            status: "type a query and press enter".to_owned(),
            detail: None,
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
            "scope: all projects".clone_into(&mut self.status);
            return;
        }
        // Already at the top; refocus the search input.
        self.focus = Focus::Search;
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
            self.status = format!("scope: {}", p.slug);
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
        self.status = format!("searching for {query:?}…");
        self.results.clear();
        self.detail = None;
        let _ = self.search_tx.send(SearchRequest {
            query,
            scope: self.scope.clone(),
            top_k: 10,
        });
    }

    pub fn poll_search_results(&mut self) {
        while let Ok(output) = self.search_rx.try_recv() {
            self.searching = false;
            match output {
                SearchOutput::Hits(rows) => {
                    let n = rows.len();
                    self.results = rows;
                    self.result_selection = 0;
                    if n == 0 {
                        "no results".clone_into(&mut self.status);
                        self.detail = None;
                    } else {
                        self.status = format!("results ({n}) — ↑/↓ to browse");
                        self.focus = Focus::Results;
                        self.refresh_detail_from_highlight();
                    }
                }
                SearchOutput::Error(e) => {
                    self.status = format!("error: {e}");
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

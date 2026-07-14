//! TUI app state — palette-first memory browser.
//!
//! The home screen is a search prompt over a global, in-memory index of
//! every project's memories (`MemIndex`, loaded off-thread). An empty query
//! shows recents grouped by project; typing runs a debounced semantic
//! search on the worker (the results pane shows a spinner until hits land).
//! Enter opens the highlighted memory in a full-screen reader. F2 opens a
//! lazygit-style settings modal (index / add / delete / remove / clean up).

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use lethe_core::db::MemoryDb;
use lethe_core::maintenance::{self, StaleReason, StaleTranscript};
use lethe_core::markdown_store::parse_anchor;
use lethe_core::registry::{self, ProjectEntry};

use crate::search_worker::{self, Phase, SearchQuery, WorkerOutput, WorkerRequest};

/// Recents: memories shown per project group on the home screen.
const RECENTS_PER_PROJECT: usize = 4;
/// How long the query must be idle before a semantic search fires.
const SEARCH_DEBOUNCE: Duration = Duration::from_millis(350);

/// Friendly project name — the basename of its root — instead of the
/// registry slug (`p_<base>_<hash>`), which reads as noise in the UI.
pub fn project_name(root: &Path) -> String {
    root.file_name().map_or_else(
        || root.to_string_lossy().into_owned(),
        |s| s.to_string_lossy().into_owned(),
    )
}

/// Agent that produced an indexed transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptSource {
    ClaudeCode,
    Codex,
    OhMyPi,
}

impl TranscriptSource {
    pub fn label(self) -> &'static str {
        match self {
            TranscriptSource::ClaudeCode => "Claude Code",
            TranscriptSource::Codex => "Codex",
            TranscriptSource::OhMyPi => "Oh My Pi",
        }
    }
}

/// Human label for the agent that produced a memory, from its anchor.
pub fn source_label(content: &str) -> Option<&'static str> {
    transcript_source(content).map(TranscriptSource::label)
}

/// Classify a memory by its anchor's transcript path. Codex rollouts have a
/// stable `rollout-` filename prefix; other transcripts under `sessions/`
/// belong to Oh My Pi. `None` means the content carries no anchor.
fn transcript_source(content: &str) -> Option<TranscriptSource> {
    let anchor = parse_anchor(content)?;
    let base = Path::new(&anchor.transcript)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if base.starts_with("rollout-") {
        Some(TranscriptSource::Codex)
    } else if anchor.transcript.contains("/sessions/") || anchor.transcript.contains("\\sessions\\")
    {
        Some(TranscriptSource::OhMyPi)
    } else {
        Some(TranscriptSource::ClaudeCode)
    }
}

fn index_db_path(slug: &str) -> PathBuf {
    registry::registry_dir()
        .join("index")
        .join(slug)
        .join("lethe.duckdb")
}

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub total: usize,
    /// `(display_name, count)` sorted descending by count.
    pub by_project: Vec<(String, usize)>,
}

/// One memory in the global in-memory index.
#[derive(Debug, Clone)]
pub struct MemItem {
    pub project_root: PathBuf,
    pub project_name: String,
    pub id: String,
    pub content: String,
    pub date_epoch: i64,
}

/// Global index of every project's memories, loaded off-thread. Powers the
/// recents view and the instant substring filter, and supplies dates for
/// semantic hits.
#[derive(Debug, Default)]
pub struct MemIndex {
    /// Sorted newest-first.
    pub items: Vec<MemItem>,
    pub stats: Stats,
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

/// A row in the home list. `score == 0` for recents/filter rows.
#[derive(Debug, Clone)]
pub struct ResultRow {
    /// `Some` for cross-project (all-scope) rows; tags the row and titles
    /// the reader. `None` when scoped to a single project.
    pub project_root: Option<PathBuf>,
    pub project_name: String,
    pub id: String,
    pub content: String,
    pub score: f32,
    pub date_epoch: i64,
}

/// What `results` currently holds — drives grouping and whether rows show a
/// relevance bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Listing {
    Recents,
    Semantic,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Home,
    Reader,
}

/// A destructive action awaiting confirmation.
#[derive(Debug)]
pub enum PendingAction {
    RemoveProject(ProjectEntry),
}

#[derive(Debug)]
pub struct Confirm {
    pub title: String,
    pub lines: Vec<String>,
    pub action: PendingAction,
    pub allow_transcripts: bool,
}

/// Delete-state-records form (project + age + typed confirmation).
#[derive(Debug, Clone)]
pub struct DeleteForm {
    /// `None` = all projects, `Some(i)` = `projects[i]`.
    pub project: Option<usize>,
    /// `0` = every record; otherwise records older than this many days.
    pub days: u32,
    /// Typed confirmation — must equal `delete` to arm.
    pub confirm: String,
}

impl DeleteForm {
    fn armed(&self) -> bool {
        self.confirm == "delete"
    }
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

/// One settings-menu entry.
pub struct SettingsItem {
    pub key: char,
    pub label: &'static str,
    pub hint: &'static str,
    pub danger: bool,
}

/// Settings menu, in display order. Index maps to `activate_setting`.
pub const SETTINGS: [SettingsItem; 9] = [
    SettingsItem {
        key: 'i',
        label: "Index this project",
        hint: "re-scan sources",
        danger: false,
    },
    SettingsItem {
        key: 'I',
        label: "Re-index all projects",
        hint: "",
        danger: false,
    },
    SettingsItem {
        key: 'r',
        label: "Rebuild index (clean)",
        hint: "wipe + re-parse",
        danger: false,
    },
    SettingsItem {
        key: 'R',
        label: "Rebuild all (clean)",
        hint: "every project",
        danger: false,
    },
    SettingsItem {
        key: 'a',
        label: "Add project path…",
        hint: "register + index",
        danger: false,
    },
    SettingsItem {
        key: 'd',
        label: "Delete state records…",
        hint: "by project / age",
        danger: true,
    },
    SettingsItem {
        key: 'x',
        label: "Remove project…",
        hint: "keeps files on disk",
        danger: true,
    },
    SettingsItem {
        key: 'c',
        label: "Clean up dead transcripts",
        hint: "reclaim disk",
        danger: false,
    },
    SettingsItem {
        key: 'q',
        label: "Quit",
        hint: "",
        danger: false,
    },
];

/// Modal overlay state. `None` = the base home/reader view.
#[derive(Debug)]
pub enum Overlay {
    Settings(usize),
    AddProject(String),
    DeleteRecords(DeleteForm),
    Confirm(Confirm),
    Cleanup(CleanupList),
    Busy(String),
}

pub struct App {
    pub mode: Mode,
    pub scope: Scope,
    pub projects: Vec<ProjectEntry>,
    /// True once the user has revealed the scope picker with Tab.
    pub scope_open: bool,
    pub overlay: Option<Overlay>,

    pub search_input: String,
    pub last_query: String,
    pub searching: bool,
    pub search_phase: Option<Phase>,
    /// Set when the query changes; fires a semantic search once idle.
    pub search_due: Option<Instant>,

    pub results: Vec<ResultRow>,
    pub result_selection: usize,
    pub listing: Listing,

    /// Full-screen reader scroll offset (lines).
    pub reader_scroll: u16,

    pub toast: Option<Toast>,
    pub needs_redraw: bool,

    /// Global memory index, loaded off-thread. `None` until it lands.
    pub mem: Option<MemIndex>,
    pub mem_rx: mpsc::Receiver<MemIndex>,

    pub search_tx: mpsc::Sender<WorkerRequest>,
    pub search_rx: mpsc::Receiver<WorkerOutput>,
}

impl App {
    pub fn new() -> Self {
        let projects = registry::load();
        let (search_tx, search_rx) = search_worker::spawn();
        let mem_rx = spawn_mem_index(projects.clone());
        Self {
            mode: Mode::Home,
            scope: Scope::AllProjects,
            projects,
            scope_open: false,
            overlay: None,
            search_input: String::new(),
            last_query: String::new(),
            searching: false,
            search_phase: None,
            search_due: None,
            results: Vec::new(),
            result_selection: 0,
            listing: Listing::Recents,
            reader_scroll: 0,
            toast: None,
            needs_redraw: false,
            mem: None,
            mem_rx,
            search_tx,
            search_rx,
        }
    }

    pub fn stats(&self) -> Option<&Stats> {
        self.mem.as_ref().map(|m| &m.stats)
    }

    pub fn selected(&self) -> Option<&ResultRow> {
        self.results.get(self.result_selection)
    }

    // -------- input / listing --------

    /// Rebuild `results` from the current query + scope. Empty query →
    /// recents; non-empty → clear the list and arm a semantic search (the
    /// results pane shows a spinner until hits land).
    fn rebuild(&mut self) {
        let Some(mem) = &self.mem else {
            self.results.clear();
            self.result_selection = 0;
            return;
        };
        self.result_selection = 0;
        if self.search_input.trim().is_empty() {
            self.results = recents(mem, &self.scope);
            self.listing = Listing::Recents;
            self.search_due = None;
            self.last_query.clear();
        } else {
            self.results.clear();
            self.listing = Listing::Semantic;
            self.search_due = Some(Instant::now() + SEARCH_DEBOUNCE);
        }
    }

    pub fn push_char(&mut self, c: char) {
        self.search_input.push(c);
        self.rebuild();
    }

    pub fn backspace(&mut self) {
        self.search_input.pop();
        self.rebuild();
    }

    /// Esc on the home screen: clear the query back to recents.
    pub fn clear_query(&mut self) {
        if self.search_input.is_empty() {
            return;
        }
        self.search_input.clear();
        self.rebuild();
    }

    pub fn arrow(&mut self, delta: isize) {
        if self.results.is_empty() {
            return;
        }
        let len = self.results.len() as isize;
        self.result_selection = (self.result_selection as isize + delta).rem_euclid(len) as usize;
    }

    /// Tab / Shift+Tab cycle the scope: all projects → each project → all.
    pub fn cycle_scope(&mut self, delta: isize) {
        if self.projects.is_empty() {
            return;
        }
        self.scope_open = true;
        // Index 0 == all projects; 1..=n == projects[i-1].
        let len = self.projects.len() as isize + 1;
        let cur = match &self.scope {
            Scope::AllProjects => 0,
            Scope::Single(e) => self
                .projects
                .iter()
                .position(|p| p.slug == e.slug)
                .map_or(0, |i| i as isize + 1),
        };
        let next = (cur + delta).rem_euclid(len);
        self.scope = if next == 0 {
            Scope::AllProjects
        } else {
            Scope::Single(self.projects[(next - 1) as usize].clone())
        };
        self.rebuild();
    }

    /// Enter on the home screen: open the highlighted memory in the reader.
    pub fn open_reader(&mut self) {
        if self.results.is_empty() {
            return;
        }
        self.reader_scroll = 0;
        self.mode = Mode::Reader;
    }

    pub fn close_reader(&mut self) {
        self.mode = Mode::Home;
    }

    /// ←/→ in the reader — page to the previous/next sibling memory.
    pub fn reader_page(&mut self, delta: isize) {
        self.arrow(delta);
        self.reader_scroll = 0;
    }

    pub fn reader_scroll(&mut self, delta: i32) {
        let next = self.reader_scroll as i32 + delta;
        self.reader_scroll = next.max(0) as u16;
    }

    /// Fire the debounced semantic search once the query has gone idle.
    pub fn poll_search_due(&mut self) {
        let Some(due) = self.search_due else { return };
        if self.searching || Instant::now() < due {
            return;
        }
        self.search_due = None;
        self.submit_search();
    }

    fn submit_search(&mut self) {
        let query = self.search_input.trim().to_owned();
        if query.is_empty() {
            return;
        }
        self.last_query.clone_from(&query);
        self.searching = true;
        self.search_phase = None;
        let _ = self.search_tx.send(WorkerRequest::Search(SearchQuery {
            query,
            scope: self.scope.clone(),
            top_k: 12,
        }));
    }

    // -------- background polling --------

    pub fn poll_mem(&mut self) {
        if self.mem.is_some() {
            return;
        }
        if let Ok(mem) = self.mem_rx.try_recv() {
            self.mem = Some(mem);
            self.rebuild();
        }
    }

    pub fn poll_search_results(&mut self) {
        while let Ok(output) = self.search_rx.try_recv() {
            match output {
                WorkerOutput::Phase(p) => {
                    self.search_phase = Some(p);
                    if matches!(self.overlay, Some(Overlay::Busy(_))) {
                        self.overlay = Some(Overlay::Busy(format!("{}…", p.label())));
                    }
                }
                WorkerOutput::Hits(rows) => {
                    self.searching = false;
                    self.search_phase = None;
                    self.needs_redraw = true;
                    // The query moved on while the search ran — discard.
                    if self.search_input.trim().is_empty() {
                        continue;
                    }
                    self.results = rows.into_iter().map(|r| self.enrich(r)).collect();
                    self.listing = Listing::Semantic;
                    self.result_selection = 0;
                }
                WorkerOutput::Indexed { added, projects } => {
                    self.overlay = None;
                    self.search_phase = None;
                    self.needs_redraw = true;
                    self.show_toast(
                        format!("indexed {projects} project(s), +{added} memories"),
                        ToastKind::Info,
                    );
                    self.refresh();
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
                    let msg = if r.records > 0 {
                        format!(
                            "deleted {} record(s), reclaimed {}",
                            r.records,
                            maintenance::human_bytes(r.bytes)
                        )
                    } else if r.projects > 0 {
                        format!(
                            "removed {} project(s), reclaimed {}",
                            r.projects,
                            maintenance::human_bytes(r.bytes)
                        )
                    } else {
                        format!(
                            "deleted {} transcript(s), reclaimed {}",
                            r.transcripts,
                            maintenance::human_bytes(r.bytes)
                        )
                    };
                    self.show_toast(msg, ToastKind::Info);
                    self.refresh();
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

    /// Fill in project name + date for a worker hit from the memory index.
    fn enrich(&self, mut r: ResultRow) -> ResultRow {
        if r.project_name.is_empty() {
            r.project_name = match &self.scope {
                Scope::Single(e) => project_name(&e.root),
                Scope::AllProjects => r
                    .project_root
                    .as_deref()
                    .map(project_name)
                    .unwrap_or_default(),
            };
        }
        if let Some(mem) = &self.mem {
            if let Some(it) = mem.items.iter().find(|it| it.id == r.id) {
                r.date_epoch = it.date_epoch;
            }
        }
        r
    }

    /// Reload projects + memory index after a mutation, keeping the current
    /// scope where possible.
    fn refresh(&mut self) {
        self.projects = registry::load();
        // A scoped-to project that just vanished falls back to all.
        if let Scope::Single(e) = &self.scope {
            if !self.projects.iter().any(|p| p.slug == e.slug) {
                self.scope = Scope::AllProjects;
            }
        }
        self.mem = None;
        self.mem_rx = spawn_mem_index(self.projects.clone());
        self.results.clear();
        self.result_selection = 0;
    }

    // -------- toasts --------

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

    // -------- settings modal --------

    pub fn open_settings(&mut self) {
        self.overlay = Some(Overlay::Settings(0));
    }

    /// Route a key to the active overlay. Returns `true` when the app wants
    /// to quit (settings → Quit).
    pub fn overlay_key(&mut self, key: crossterm::event::KeyEvent) -> bool {
        use crossterm::event::KeyCode;
        match &mut self.overlay {
            Some(Overlay::Settings(cursor)) => match key.code {
                KeyCode::Up => *cursor = (*cursor + SETTINGS.len() - 1) % SETTINGS.len(),
                KeyCode::Down => *cursor = (*cursor + 1) % SETTINGS.len(),
                KeyCode::Enter => {
                    let idx = *cursor;
                    return self.activate_setting(idx);
                }
                KeyCode::Esc => self.overlay = None,
                KeyCode::Char(c) => {
                    if let Some(i) = SETTINGS.iter().position(|s| s.key == c) {
                        return self.activate_setting(i);
                    }
                }
                _ => {}
            },
            Some(Overlay::AddProject(buf)) => match key.code {
                KeyCode::Char(c) => buf.push(c),
                KeyCode::Backspace => {
                    buf.pop();
                }
                KeyCode::Enter => self.submit_add_project(),
                KeyCode::Esc => self.overlay = None,
                _ => {}
            },
            Some(Overlay::DeleteRecords(_)) => self.delete_records_key(key),
            Some(Overlay::Confirm(c)) => match key.code {
                KeyCode::Char('y') => self.confirm(false),
                KeyCode::Char('t') if c.allow_transcripts => self.confirm(true),
                KeyCode::Esc | KeyCode::Char('n') => self.overlay = None,
                _ => {}
            },
            Some(Overlay::Cleanup(list)) => match key.code {
                KeyCode::Up => list.cursor = list.cursor.saturating_sub(1),
                KeyCode::Down => {
                    list.cursor = (list.cursor + 1).min(list.items.len().saturating_sub(1));
                }
                KeyCode::Char(' ') => {
                    if let Some(sel) = list.selected.get_mut(list.cursor) {
                        *sel = !*sel;
                    }
                }
                KeyCode::Char('a') => {
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
                if key.code == KeyCode::Esc {
                    self.overlay = None;
                }
            }
            None => {}
        }
        false
    }

    /// The project index/rebuild actions operate on: the scoped project, or
    /// — when unscoped — the repo the TUI was launched in (registered if
    /// needed), so re-adding a just-removed project is a single keystroke.
    fn target_entry(&self) -> Option<ProjectEntry> {
        match &self.scope {
            Scope::Single(e) => Some(e.clone()),
            Scope::AllProjects => current_project_root().and_then(|r| registry::register(&r).ok()),
        }
    }

    fn activate_setting(&mut self, idx: usize) -> bool {
        match SETTINGS[idx].key {
            'i' => match self.target_entry() {
                Some(entry) => {
                    self.start_busy("indexing");
                    let _ = self.search_tx.send(WorkerRequest::Index(vec![entry]));
                }
                None => {
                    self.show_toast("no project here — use Add project path", ToastKind::Info);
                }
            },
            'I' => {
                let all = registry::load();
                if all.is_empty() {
                    self.overlay = None;
                    self.show_toast("no registered projects", ToastKind::Info);
                } else {
                    self.start_busy("indexing");
                    let _ = self.search_tx.send(WorkerRequest::Index(all));
                }
            }
            'r' => match self.target_entry() {
                Some(entry) => {
                    self.start_busy("rebuilding");
                    let _ = self.search_tx.send(WorkerRequest::Rebuild(vec![entry]));
                }
                None => {
                    self.show_toast("no project here — use Add project path", ToastKind::Info);
                }
            },
            'R' => {
                let all = registry::load();
                if all.is_empty() {
                    self.overlay = None;
                    self.show_toast("no registered projects", ToastKind::Info);
                } else {
                    self.start_busy("rebuilding all");
                    let _ = self.search_tx.send(WorkerRequest::Rebuild(all));
                }
            }
            'a' => {
                // Pre-fill with the current repo — the common case is
                // re-adding the project you're standing in.
                let prefill = current_project_root()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                self.overlay = Some(Overlay::AddProject(prefill));
            }
            'd' => {
                self.overlay = Some(Overlay::DeleteRecords(DeleteForm {
                    project: match &self.scope {
                        Scope::AllProjects => None,
                        Scope::Single(e) => self.projects.iter().position(|p| p.slug == e.slug),
                    },
                    days: 30,
                    confirm: String::new(),
                }));
            }
            'x' => {
                if let Scope::Single(entry) = &self.scope {
                    let name = project_name(&entry.root);
                    self.overlay = Some(Overlay::Confirm(Confirm {
                        title: format!("Remove project '{name}'?"),
                        lines: vec![
                            "y = remove from lethe   t = also delete transcripts".into(),
                            "Transcripts live under ~/.claude & ~/.codex.".into(),
                        ],
                        action: PendingAction::RemoveProject(entry.clone()),
                        allow_transcripts: true,
                    }));
                } else {
                    self.show_toast("Tab to a project first", ToastKind::Info);
                }
            }
            'c' => {
                self.start_busy("scanning");
                let _ = self.search_tx.send(WorkerRequest::Scan);
            }
            'q' => return true,
            _ => self.overlay = None,
        }
        false
    }

    fn submit_add_project(&mut self) {
        let Some(Overlay::AddProject(buf)) = &self.overlay else {
            return;
        };
        let raw = buf.trim();
        if raw.is_empty() {
            self.overlay = None;
            return;
        }
        let path = expand_tilde(raw);
        if !path.is_dir() {
            self.show_toast(format!("not a directory: {raw}"), ToastKind::Error);
            return;
        }
        match registry::register(&path) {
            Ok(entry) => {
                self.start_busy("indexing");
                let _ = self.search_tx.send(WorkerRequest::Index(vec![entry]));
            }
            Err(e) => self.show_toast(format!("register failed: {e}"), ToastKind::Error),
        }
    }

    fn delete_records_key(&mut self, key: crossterm::event::KeyEvent) {
        use crossterm::event::KeyCode;
        let n = self.projects.len();
        let Some(Overlay::DeleteRecords(form)) = &mut self.overlay else {
            return;
        };
        match key.code {
            // Cycle project scope: all → each project → all.
            KeyCode::Left | KeyCode::BackTab => {
                form.project = cycle_project_scope(form.project, n, -1);
                form.confirm.clear();
            }
            KeyCode::Right | KeyCode::Tab => {
                form.project = cycle_project_scope(form.project, n, 1);
                form.confirm.clear();
            }
            KeyCode::Up => {
                form.days = form.days.saturating_add(1);
                form.confirm.clear();
            }
            KeyCode::Down => {
                form.days = form.days.saturating_sub(1);
                form.confirm.clear();
            }
            KeyCode::Char(c) => form.confirm.push(c),
            KeyCode::Backspace => {
                form.confirm.pop();
            }
            KeyCode::Enter => {
                if !form.armed() {
                    self.show_toast("type \"delete\" to confirm", ToastKind::Info);
                    return;
                }
                let entries: Vec<ProjectEntry> = match form.project {
                    None => self.projects.clone(),
                    Some(i) => self.projects.get(i).cloned().into_iter().collect(),
                };
                let days = form.days;
                if entries.is_empty() {
                    self.overlay = None;
                    return;
                }
                self.start_busy("deleting");
                let _ = self
                    .search_tx
                    .send(WorkerRequest::DeleteRecords { entries, days });
            }
            KeyCode::Esc => self.overlay = None,
            _ => {}
        }
    }

    fn start_busy(&mut self, label: &str) {
        self.overlay = Some(Overlay::Busy(format!("{label}…")));
        self.needs_redraw = true;
    }

    fn confirm(&mut self, with_transcripts: bool) {
        let Some(Overlay::Confirm(c)) = self.overlay.take() else {
            return;
        };
        match c.action {
            PendingAction::RemoveProject(entry) => {
                self.start_busy("removing");
                let _ = self.search_tx.send(WorkerRequest::DeleteProjects {
                    entries: vec![entry],
                    with_transcripts,
                });
            }
        }
    }
}

/// Build the recents view: memories grouped by project, projects ordered by
/// their most-recent memory, capped per project.
fn recents(mem: &MemIndex, scope: &Scope) -> Vec<ResultRow> {
    let all_scope = matches!(scope, Scope::AllProjects);
    let mut order: Vec<String> = Vec::new();
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut out = Vec::new();
    for it in mem.items.iter().filter(|it| in_scope(it, scope)) {
        let c = counts.entry(it.project_name.clone()).or_insert(0);
        if *c >= RECENTS_PER_PROJECT {
            continue;
        }
        if *c == 0 {
            order.push(it.project_name.clone());
        }
        *c += 1;
        out.push(row_from(it, all_scope));
    }
    // `items` is date-sorted, so first-seen project order is recency order;
    // group rows by project while preserving that order.
    out.sort_by_key(|r| {
        order
            .iter()
            .position(|p| *p == r.project_name)
            .unwrap_or(usize::MAX)
    });
    out
}

fn in_scope(it: &MemItem, scope: &Scope) -> bool {
    match scope {
        Scope::AllProjects => true,
        Scope::Single(e) => it.project_root == e.root,
    }
}

fn row_from(it: &MemItem, all_scope: bool) -> ResultRow {
    ResultRow {
        project_root: all_scope.then(|| it.project_root.clone()),
        project_name: it.project_name.clone(),
        id: it.id.clone(),
        content: it.content.clone(),
        score: 0.0,
        date_epoch: it.date_epoch,
    }
}

/// First non-empty, non-heading, non-role, non-anchor line — the memory's
/// "title" (usually the user's sentence).
pub fn first_line(content: &str) -> &str {
    for line in content.lines() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') || s == "USER:" || s == "ASSISTANT:" {
            continue;
        }
        if s.starts_with("<!--") && s.ends_with("-->") {
            continue;
        }
        return s;
    }
    "(empty)"
}

/// Cycle the delete-form project scope. Index `0` is "all projects";
/// `1..=n` maps to `projects[i-1]`. Wraps.
fn cycle_project_scope(cur: Option<usize>, n: usize, delta: isize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let idx = cur.map_or(0, |i| i as isize + 1);
    let next = (idx + delta).rem_euclid(n as isize + 1);
    (next != 0).then(|| (next - 1) as usize)
}

/// The project root for the directory the TUI was launched in: the nearest
/// ancestor containing a `.git`, else the cwd. Mirrors the CLI's default
/// `--root` resolution so `i`/`a` map to the same index the stop-hook uses.
fn current_project_root() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let mut cur: Option<&Path> = Some(&cwd);
    while let Some(c) = cur {
        if c.join(".git").exists() {
            return Some(c.to_path_buf());
        }
        cur = c.parent();
    }
    Some(cwd)
}

fn expand_tilde(raw: &str) -> PathBuf {
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(raw)
}

/// Load every project's memories into one in-memory index, off-thread, and
/// compute per-project/per-source stats in the same pass.
fn spawn_mem_index(projects: Vec<ProjectEntry>) -> mpsc::Receiver<MemIndex> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let mut items: Vec<MemItem> = Vec::new();
        let mut by_project: Vec<(String, usize)> = Vec::with_capacity(projects.len());
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
            let name = project_name(&p.root);
            for row in &rows {
                items.push(MemItem {
                    project_root: p.root.clone(),
                    project_name: name.clone(),
                    id: row.id.clone(),
                    content: row.content.clone(),
                    date_epoch: row.created_at_epoch,
                });
            }
            if !rows.is_empty() {
                by_project.push((name, rows.len()));
            }
        }
        items.sort_by(|a, b| b.date_epoch.cmp(&a.date_epoch));
        by_project.sort_by(|a, b| b.1.cmp(&a.1));
        let stats = Stats {
            total: items.len(),
            by_project,
        };
        let _ = tx.send(MemIndex { items, stats });
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
    fn classifies_transcript_sources() {
        assert_eq!(
            transcript_source(&anchored(
                "/Users/x/.codex/sessions/2026/07/rollout-2026-abc.jsonl"
            )),
            Some(TranscriptSource::Codex)
        );
        assert_eq!(
            transcript_source(&anchored(
                "/Users/x/.omp/agent/sessions/-Users-x-repo/2026-07-14_session.jsonl"
            )),
            Some(TranscriptSource::OhMyPi)
        );
        assert_eq!(
            transcript_source(&anchored(
                "/Users/x/.claude/projects/-Users-x-repo/6f0e.jsonl"
            )),
            Some(TranscriptSource::ClaudeCode)
        );
        assert_eq!(transcript_source("no anchor here"), None);
    }

    #[test]
    fn project_name_is_basename() {
        assert_eq!(project_name(Path::new("/Users/x/Projects/lethe")), "lethe");
    }

    #[test]
    fn first_line_skips_role_and_anchor() {
        let c = "<!-- session:s -->\nUSER:\nfix the bug\n\nASSISTANT:\ndone";
        assert_eq!(first_line(c), "fix the bug");
    }

    #[test]
    fn recents_group_by_project_and_cap() {
        let mut items = Vec::new();
        for (proj, day, n) in [("a", 100, 6), ("b", 50, 2)] {
            for i in 0..n {
                items.push(MemItem {
                    project_root: PathBuf::from(format!("/r/{proj}")),
                    project_name: proj.to_owned(),
                    id: format!("{proj}{i}"),
                    content: format!("mem {proj} {i}"),
                    date_epoch: day - i,
                });
            }
        }
        items.sort_by(|a, b| b.date_epoch.cmp(&a.date_epoch));
        let mem = MemIndex {
            items,
            stats: Stats::default(),
        };
        let rows = recents(&mem, &Scope::AllProjects);
        // Capped to 4 for "a", 2 for "b"; "a" (newer) group first.
        assert_eq!(rows.len(), RECENTS_PER_PROJECT + 2);
        assert!(rows[0].project_name == "a");
        assert!(rows.last().unwrap().project_name == "b");
    }
}

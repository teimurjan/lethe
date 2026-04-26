//! `lethe-tui` — ratatui counterpart to `legacy/lethe/tui.py`.
//!
//! Layout:
//!   ┌── lethe ─── › <scope> ─────────────────────────────┐
//!   │ Projects (N)        │ all projects ▸ █             │
//!   │ > project-a         ├─────────────────────────────┤
//!   │   project-b         │ Results ↑/↓ to browse       │
//!   │ ...                 │ ...                         │
//!   ├─────────────────────┴─────────────────────────────┤
//!   │ Detail (Esc to close)                              │
//!   ├────────────────────────────────────────────────────┤
//!   │ ↑/↓ nav · ⏎ search/open · esc back · ^q quit       │
//!   └────────────────────────────────────────────────────┘
//!
//! Behaviors mirrored from the Textual TUI:
//!   - Live-expand on result-list highlight (no Enter needed). Empty
//!     results don't blink the detail pane.
//!   - First result auto-highlighted after search; arrow keys browse.
//!   - Type-anywhere refocuses the search input.

#![allow(clippy::print_stdout)]

mod app;
mod search_worker;
mod ui;

use anyhow::Result;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io;
use std::time::Duration;

use crate::app::{App, Focus};

fn main() -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    let result = run_event_loop(&mut terminal, &mut app);

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    result
}

fn run_event_loop<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> Result<()> {
    let tick_rate = Duration::from_millis(100);
    loop {
        terminal.draw(|f| ui::draw(f, app))?;
        // Poll-based event loop so background search results can land.
        if event::poll(tick_rate)? {
            match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => {
                    if handle_key(app, k) {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }
        app.poll_search_results();
    }
}

/// Returns `true` when the app wants to exit.
fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    match (key.code, key.modifiers) {
        // Quit.
        (KeyCode::Char('q' | 'c'), KeyModifiers::CONTROL) => return true,

        // Submit search.
        (KeyCode::Enter, _) => {
            if matches!(app.focus, Focus::Search) {
                app.submit_search();
            } else if matches!(app.focus, Focus::Results) {
                // Treat enter on a result as "show detail" — already
                // shown via live-expand, this is idempotent.
                app.refresh_detail_from_highlight();
            } else if matches!(app.focus, Focus::Projects) {
                app.enter_selected_project();
            }
        }

        // Esc — close detail or pop scope or refocus search.
        (KeyCode::Esc, _) => app.escape(),

        // Tab and Shift+Tab cycle focus.
        (KeyCode::Tab, _) => app.cycle_focus(true),
        (KeyCode::BackTab, _) => app.cycle_focus(false),

        // Direct focus shortcuts.
        (KeyCode::Char('l'), KeyModifiers::CONTROL) => app.focus = Focus::Search,
        (KeyCode::Char('p'), KeyModifiers::CONTROL) => app.focus = Focus::Projects,
        (KeyCode::Char('r'), KeyModifiers::CONTROL) => app.focus = Focus::Results,

        (KeyCode::Up, _) => app.move_cursor(-1),
        (KeyCode::Down, _) => app.move_cursor(1),

        // Editing in the search input.
        (KeyCode::Backspace, _) if matches!(app.focus, Focus::Search) => {
            app.search_input.pop();
        }
        (KeyCode::Char(c), _) => {
            // Type-anywhere refocus.
            if !matches!(app.focus, Focus::Search) {
                app.focus = Focus::Search;
            }
            app.search_input.push(c);
        }

        _ => {}
    }
    false
}

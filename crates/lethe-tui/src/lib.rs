//! `lethe-tui` — palette-first memory browser.
//!
//! Library entry point: callers invoke [`run`] which owns the full terminal
//! lifecycle (`enable_raw_mode`, `EnterAlternateScreen`, event loop,
//! cleanup). The CLI's no-arg path and explicit `tui` subcommand both call
//! into this function.
//!
//! Layout (home):
//!   ◆ lethe v0.16.0                     3 projects · 128 memories
//!   ┌ Search ─────────────────────────────────────────────────┐
//!   │ ❯ auth tok█   12 matches · all projects                  │
//!   └──────────────────────────────────────────────────────────┘
//!   ┌ Results ─────────────────────────────────────────────────┐
//!   │ ▸ fix auth token refresh          · acme-api · Jul 14     │
//!   │   rotate auth tokens in CI        · infra-tools · Jul 09  │
//!   └──────────────────────────────────────────────────────────┘
//!   ↑↓ move · ↵ open · Tab scope · F2 settings · ^c copy · F10 quit
//!
//! Enter opens a full-screen reader; F2 opens a settings modal.

#![allow(clippy::print_stdout)]

mod app;
mod search_worker;
mod theme;
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

use crate::app::{App, Mode};

/// Run the TUI to completion. Owns the terminal lifecycle so callers only
/// need to handle the returned `Result`.
pub fn run() -> Result<()> {
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
        if app.needs_redraw {
            terminal.clear()?;
            app.needs_redraw = false;
        }
        terminal.draw(|f| ui::draw(f, app))?;
        if event::poll(tick_rate)? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press && handle_key(app, k) {
                    return Ok(());
                }
            }
        }
        app.poll_mem();
        app.poll_search_results();
        app.poll_search_due();
        app.poll_toast();
    }
}

/// Returns `true` when the app wants to exit.
fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    // Global quit from anywhere.
    if matches!(
        (key.code, key.modifiers),
        (KeyCode::Char('q'), KeyModifiers::CONTROL) | (KeyCode::F(10), _)
    ) {
        return true;
    }

    // Overlays capture every other key while open.
    if app.overlay.is_some() {
        return app.overlay_key(key);
    }

    match app.mode {
        Mode::Reader => handle_reader_key(app, key),
        Mode::Home => handle_home_key(app, key),
    }
    false
}

fn handle_home_key(app: &mut App, key: KeyEvent) {
    match (key.code, key.modifiers) {
        (KeyCode::F(2), _) => app.open_settings(),
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => app.copy_selected_to_clipboard(),
        (KeyCode::Enter, _) => app.open_reader(),
        (KeyCode::Esc, _) => app.clear_query(),
        (KeyCode::Tab, _) => app.cycle_scope(1),
        (KeyCode::BackTab, _) => app.cycle_scope(-1),
        (KeyCode::Up, _) => app.arrow(-1),
        (KeyCode::Down, _) => app.arrow(1),
        (KeyCode::Backspace, _) => app.backspace(),
        // Typing goes to the search box; CONTROL/ALT combos are shortcuts.
        (KeyCode::Char(c), m) if (m - KeyModifiers::SHIFT).is_empty() => app.push_char(c),
        _ => {}
    }
}

fn handle_reader_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc => app.close_reader(),
        KeyCode::Up => app.reader_scroll(-1),
        KeyCode::Down => app.reader_scroll(1),
        KeyCode::PageUp => app.reader_scroll(-10),
        KeyCode::PageDown => app.reader_scroll(10),
        KeyCode::Left => app.reader_page(-1),
        KeyCode::Right => app.reader_page(1),
        KeyCode::Char('c') => app.copy_selected_to_clipboard(),
        _ => {}
    }
}

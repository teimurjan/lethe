//! `lethe-tui` — ratatui counterpart to `research_playground/lethe_reference/lethe/tui.py`.
//!
//! Library entry point: callers invoke [`run`] which owns the full
//! terminal lifecycle (`enable_raw_mode`, `EnterAlternateScreen`,
//! event loop, cleanup). The CLI's no-arg path and explicit `tui`
//! subcommand both call into this function.
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

use crate::app::App;

/// Run the TUI to completion. Owns the terminal lifecycle so callers
/// only need to handle the returned `Result`.
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
        app.poll_stats();
        app.poll_toast();
    }
}

/// Returns `true` when the app wants to exit.
///
/// No focus panes: typing always edits the search box; arrows move the
/// project list (while browsing projects) or the memory list; Esc goes
/// back; Ctrl+C copies the highlighted memory; Ctrl+D deletes the
/// highlighted project; Ctrl+Q quits.
fn handle_key(app: &mut App, key: KeyEvent) -> bool {
    // Quit works from anywhere, including modal overlays.
    if let (KeyCode::Char('q'), KeyModifiers::CONTROL) = (key.code, key.modifiers) {
        return true;
    }
    // While an overlay (actions menu / confirm / busy) is open it captures
    // every other key, leaving the base view's bindings untouched.
    if app.overlay.is_some() {
        app.overlay_key(key.code);
        return false;
    }

    match (key.code, key.modifiers) {
        // Open the actions menu.
        (KeyCode::Char('a'), KeyModifiers::CONTROL) => app.open_actions(),

        // Copy the highlighted memory.
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => app.copy_selected_to_clipboard(),

        // Delete the highlighted project (press twice).
        (KeyCode::Char('d'), KeyModifiers::CONTROL) => app.request_or_confirm_delete(),

        (KeyCode::Enter, _) => app.on_enter(),
        (KeyCode::Esc, _) => app.escape(),

        (KeyCode::Up, _) => app.arrow(-1),
        (KeyCode::Down, _) => app.arrow(1),

        (KeyCode::Backspace, _) => {
            app.pending_delete = None;
            app.search_input.pop();
        }
        // Typing always goes to the search box. CONTROL/ALT combos are
        // terminal shortcuts (handled above or ignored), not text.
        (KeyCode::Char(c), m) if (m - KeyModifiers::SHIFT).is_empty() => {
            app.pending_delete = None;
            app.search_input.push(c);
        }

        _ => {}
    }
    false
}

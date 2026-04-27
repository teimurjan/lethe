//! `lethe tui` — launch the embedded TUI.
//!
//! TUI lives in the `lethe-tui` library crate; running it is a direct
//! function call now (used to be a subprocess to a sibling binary).

use anyhow::Result;

pub fn run() -> Result<i32> {
    lethe_tui::run()?;
    Ok(0)
}

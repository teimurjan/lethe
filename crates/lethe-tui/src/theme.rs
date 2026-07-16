//! Tokyo Night palette + shared widget styling. Colors are exact RGB so
//! the TUI matches the design mock in any terminal, independent of the
//! user's terminal theme.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders};

pub const BG: Color = Color::Rgb(0x16, 0x16, 0x1e);
pub const FG: Color = Color::Rgb(0xc8, 0xc8, 0xd0);
pub const BORDER: Color = Color::Rgb(0x3a, 0x3a, 0x4a);
/// Focus/title blue.
pub const ACCENT: Color = Color::Rgb(0x7a, 0xa2, 0xf7);
pub const DIM: Color = Color::Rgb(0x56, 0x5f, 0x89);
pub const SEL_BG: Color = Color::Rgb(0x2a, 0x2f, 0x45);
pub const SEL_FG: Color = Color::Rgb(0xe6, 0xe6, 0xf0);
pub const GREEN: Color = Color::Rgb(0x9e, 0xce, 0x6a);
pub const YELLOW: Color = Color::Rgb(0xe0, 0xaf, 0x68);
pub const RED: Color = Color::Rgb(0xf7, 0x76, 0x8e);
pub const CYAN: Color = Color::Rgb(0x7d, 0xcf, 0xff);

/// Base text style on the app background.
pub fn base() -> Style {
    Style::default().fg(FG).bg(BG)
}

pub fn dim() -> Style {
    Style::default().fg(DIM).bg(BG)
}

/// A bordered pane. Focused panes get the accent border + a `▎` title tab;
/// unfocused panes get the muted border.
pub fn pane(title: &str, focus: bool) -> Block<'_> {
    let border = if focus { ACCENT } else { BORDER };
    let title_style = Style::default()
        .fg(ACCENT)
        .bg(BG)
        .add_modifier(Modifier::BOLD);
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border).bg(BG))
        .style(Style::default().bg(BG))
        .title(Span::styled(format!(" {title} "), title_style))
}

/// The highlighted-row style used by list selections.
pub fn selection() -> Style {
    Style::default()
        .fg(SEL_FG)
        .bg(SEL_BG)
        .add_modifier(Modifier::BOLD)
}

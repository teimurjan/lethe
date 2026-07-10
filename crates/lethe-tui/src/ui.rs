//! ratatui rendering. Three-pane layout with optional bottom detail
//! panel and a footer hint row.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Stats, ToastKind};

const FOOTER: &str =
    "type search · ↑/↓ projects · ⏎ open · ^a actions · ^c copy · esc back · ^q quit";
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn draw(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3), // body
            if app.detail.is_some() {
                Constraint::Length(8)
            } else {
                Constraint::Length(0)
            },
            Constraint::Length(1), // footer
        ])
        .split(area);

    draw_body(frame, chunks[0], app);
    if app.detail.is_some() {
        draw_detail(frame, chunks[1], app);
    }
    draw_footer(frame, chunks[2]);

    // Modal overlay floats above the base view…
    if app.overlay.is_some() {
        draw_overlay(frame, area, app);
    }
    // …and a transient toast floats above everything.
    if app.toast.is_some() {
        draw_toast(frame, area, app);
    }
}

/// A `Rect` of at most `w`×`h` centered in `area`.
fn centered_rect(w: u16, h: u16, area: Rect) -> Rect {
    let w = w.min(area.width);
    let h = h.min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

fn draw_overlay(frame: &mut Frame<'_>, area: Rect, app: &App) {
    use crate::app::Overlay;
    match app.overlay.as_ref() {
        Some(Overlay::Actions(cursor)) => {
            let items: Vec<ListItem> = crate::app::ACTIONS
                .iter()
                .enumerate()
                .map(|(i, label)| {
                    let marker = if i == *cursor { "▸ " } else { "  " };
                    let style = if i == *cursor {
                        Style::default().add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Gray)
                    };
                    ListItem::new(Line::from(Span::styled(format!("{marker}{label}"), style)))
                })
                .collect();
            let w = 40u16;
            let h = crate::app::ACTIONS.len() as u16 + 2;
            let rect = centered_rect(w, h, area);
            frame.render_widget(Clear, rect);
            frame.render_widget(List::new(items).block(pane_block("Actions", true)), rect);
        }
        Some(Overlay::Confirm(c)) => {
            let mut lines: Vec<Line<'_>> = Vec::new();
            for l in &c.lines {
                lines.push(Line::from(l.clone()));
            }
            let width = c
                .lines
                .iter()
                .map(|l| l.chars().count())
                .chain(std::iter::once(c.title.chars().count()))
                .max()
                .unwrap_or(20)
                .clamp(24, 72) as u16
                + 4;
            let h = c.lines.len() as u16 + 2;
            let rect = centered_rect(width, h, area);
            frame.render_widget(Clear, rect);
            frame.render_widget(
                Paragraph::new(lines)
                    .block(pane_block(&c.title, true))
                    .wrap(Wrap { trim: false }),
                rect,
            );
        }
        Some(Overlay::Busy(label)) => {
            let body = Line::from(vec![
                Span::styled(
                    spinner_frame(),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::raw(label.clone()),
            ]);
            let w = (label.chars().count() as u16 + 8).clamp(20, 60);
            let rect = centered_rect(w, 3, area);
            frame.render_widget(Clear, rect);
            frame.render_widget(
                Paragraph::new(body).block(pane_block("Working", true)),
                rect,
            );
        }
        None => {}
    }
}

fn draw_body(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(40), Constraint::Min(20)])
        .split(area);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(4),
            Constraint::Length(8),
        ])
        .split(chunks[0]);
    draw_projects(frame, left[0], app);
    draw_sources(frame, left[1], app);
    draw_stats(frame, left[2], app);
    draw_results_pane(frame, chunks[1], app);
}

fn draw_projects(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let title = format!("Projects ({})", app.projects.len());
    // Arrows always drive this pane, so it's always the active one.
    let block = pane_block(&title, true);

    // Names get a 2-col marker prefix; trim the rest to the pane width
    // with an ellipsis so long names don't overflow or wrap.
    let name_room = (area.width.saturating_sub(2) as usize)
        .saturating_sub(2)
        .max(4);
    let items: Vec<ListItem> = app
        .projects
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let name = truncate(&crate::app::project_name(&p.root), name_room);
            let marker = if i == app.project_selection {
                "▸ "
            } else {
                "  "
            };
            ListItem::new(Line::from(format!("{marker}{name}")))
        })
        .collect();

    let mut state = ListState::default();
    if !app.projects.is_empty() {
        state.select(Some(app.project_selection.min(app.projects.len() - 1)));
    }
    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    frame.render_stateful_widget(list, area, &mut state);
}

fn draw_results_pane(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(3)])
        .split(area);
    draw_search_input(frame, chunks[0], app);
    draw_results(frame, chunks[1], app);
}

fn draw_search_input(frame: &mut Frame<'_>, area: Rect, app: &App) {
    // Typing always targets the box, so the cursor is always shown.
    let prompt = format!("{} ▸ ", app.scope.label());
    let body = Line::from(vec![
        Span::styled(prompt, Style::default().fg(Color::Cyan)),
        Span::raw(&app.search_input),
        Span::styled("█", Style::default().fg(Color::White)),
    ]);
    let block = pane_block("Search", false);
    let para = Paragraph::new(body).block(block);
    frame.render_widget(para, area);
}

fn draw_results(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let n = app.results.len();
    let noun = if app.browsing { "Memories" } else { "Results" };
    let title = if n == 0 {
        noun.to_owned()
    } else {
        format!("{noun} ({n})")
    };
    // Passive pane: memories aren't a focus target, so it's never drawn
    // as active.
    let block = pane_block(&title, false);

    if app.searching {
        let spin = spinner_frame();
        let phase_label = app
            .search_phase
            .map_or("preparing", crate::search_worker::Phase::label);
        let mut spans = vec![
            Span::styled(
                spin,
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(phase_label, Style::default().fg(Color::White)),
        ];
        if matches!(
            app.search_phase,
            Some(crate::search_worker::Phase::Searching) | None
        ) && !app.last_query.is_empty()
        {
            spans.push(Span::raw(" for "));
            spans.push(Span::styled(
                format!("\"{}\"", app.last_query),
                Style::default().fg(Color::White),
            ));
        }
        spans.push(Span::raw("…"));
        let para = Paragraph::new(Line::from(spans)).block(block);
        frame.render_widget(para, area);
        return;
    }

    if app.results.is_empty() {
        let msg = if app.projects.is_empty() {
            "no projects — run `lethe index` in a repo"
        } else if app.browsing {
            "no memories in this project"
        } else {
            "no results"
        };
        let para = Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default().fg(Color::DarkGray),
        )))
        .block(block);
        frame.render_widget(para, area);
        return;
    }

    let top_score = app.results.first().map_or(0.0, |r| r.score);
    let interior = (area.width.saturating_sub(2)) as usize;
    let items: Vec<ListItem> = app
        .results
        .iter()
        .enumerate()
        .map(|(idx, r)| result_row(idx, r, top_score, interior))
        .collect();

    // No selection highlight — the memory list is a passive display.
    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

fn draw_sources(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = pane_block("Sources", false);
    let lines: Vec<Line<'_>> = match &app.stats {
        None => vec![Line::from(Span::styled(
            "computing…",
            Style::default().fg(Color::DarkGray),
        ))],
        Some(s) => vec![
            source_line("Claude Code", s.claude),
            source_line("Codex", s.codex),
        ],
    };
    let para = Paragraph::new(lines).block(block);
    frame.render_widget(para, area);
}

fn source_line(label: &str, count: usize) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<12}"), Style::default().fg(Color::White)),
        Span::styled(
            format_count(count),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
    ])
}

fn draw_stats(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = pane_block("Stats", false);
    let lines = match &app.stats {
        None => vec![Line::from(Span::styled(
            "computing…",
            Style::default().fg(Color::DarkGray),
        ))],
        Some(s) => stats_lines(s, area.width),
    };
    let para = Paragraph::new(lines).block(block);
    frame.render_widget(para, area);
}

fn stats_lines(s: &Stats, width: u16) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::with_capacity(6);
    out.push(Line::from(vec![
        Span::styled(
            format_count(s.total),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" memories"),
    ]));
    out.push(Line::from(Span::styled(
        format!("across {} projects", s.by_project.len()),
        Style::default().fg(Color::DarkGray),
    )));
    out.push(Line::from(""));

    // Total interior width minus borders is `width - 2`. Reserve room
    // for slug (12) + space (1) + count (5) + space (1) = 19 cols, the
    // remaining is bar width.
    let interior = width.saturating_sub(2) as usize;
    let max = s.by_project.first().map_or(0, |(_, c)| *c).max(1);
    let bar_max = interior.saturating_sub(19).max(4);
    for (slug, count) in s.by_project.iter().take(3) {
        let label = truncate(slug, 12);
        let bar_len = (*count * bar_max) / max;
        let bar = "█".repeat(bar_len);
        out.push(Line::from(vec![
            Span::styled(format!("{label:<12}"), Style::default().fg(Color::White)),
            Span::raw(" "),
            Span::styled(bar, Style::default().fg(Color::Cyan)),
            Span::raw(" "),
            Span::styled(format!("{count:>4}"), Style::default().fg(Color::DarkGray)),
        ]));
    }
    out
}

fn format_count(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_owned()
    } else {
        s.chars().take(n.saturating_sub(1)).chain(['…']).collect()
    }
}

fn draw_detail(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = pane_block("Detail (Esc to close)", false);
    let lines = app.detail.as_deref().map(detail_lines).unwrap_or_default();
    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false });
    frame.render_widget(para, area);
}

/// Render a transcript-turn chunk: a metadata header parsed from the
/// `<!-- session:… -->` anchor, then the USER/ASSISTANT body with the
/// raw anchor line dropped and the role labels highlighted.
fn detail_lines(content: &str) -> Vec<Line<'static>> {
    let mut out: Vec<Line<'static>> = Vec::new();
    let dim = Style::default().fg(Color::DarkGray);
    if let Some(a) = lethe_core::markdown_store::parse_anchor(content) {
        let file = std::path::Path::new(&a.transcript).file_name().map_or_else(
            || a.transcript.clone(),
            |s| s.to_string_lossy().into_owned(),
        );
        out.push(Line::from(vec![
            Span::styled("session ", dim),
            Span::styled(truncate(&a.session, 8), Style::default().fg(Color::Cyan)),
            Span::styled("   turn ", dim),
            Span::styled(truncate(&a.turn, 8), Style::default().fg(Color::Cyan)),
        ]));
        out.push(Line::from(vec![
            Span::styled("transcript ", dim),
            Span::styled(file, dim),
        ]));
        out.push(Line::from(""));
    }
    for line in content.lines() {
        let s = line.trim_end();
        let t = s.trim();
        if t.starts_with("<!--") && t.ends_with("-->") {
            continue;
        }
        match t {
            "USER:" => out.push(Line::from(Span::styled(
                "USER",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ))),
            "ASSISTANT:" => out.push(Line::from(Span::styled(
                "ASSISTANT",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ))),
            _ => out.push(Line::from(s.to_owned())),
        }
    }
    out
}

fn draw_toast(frame: &mut Frame<'_>, full: Rect, app: &App) {
    let Some(toast) = &app.toast else { return };
    let msg = toast.msg.clone();
    let (fg, border) = match toast.kind {
        ToastKind::Info => (Color::Black, Color::Green),
        ToastKind::Error => (Color::White, Color::Red),
    };

    // Bottom-right placement with a small margin. Width = msg + padding,
    // capped to the available space.
    let inner_w = (msg.chars().count() as u16).saturating_add(2);
    let w = inner_w.saturating_add(2).min(full.width.saturating_sub(2));
    let h = 3u16;
    if full.width <= w + 2 || full.height <= h + 2 {
        return;
    }
    let area = Rect {
        x: full.x + full.width - w - 2,
        y: full.y + full.height - h - 2,
        width: w,
        height: h,
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border).add_modifier(Modifier::BOLD));
    let body = Line::from(Span::styled(
        msg,
        Style::default()
            .fg(fg)
            .bg(border)
            .add_modifier(Modifier::BOLD),
    ));
    frame.render_widget(Clear, area);
    frame.render_widget(Paragraph::new(body).block(block), area);
}

fn spinner_frame() -> &'static str {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let idx = ((ms / 80) as usize) % SPINNER.len();
    SPINNER[idx]
}

fn draw_footer(frame: &mut Frame<'_>, area: Rect) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            FOOTER,
            Style::default().fg(Color::DarkGray),
        ))),
        area,
    );
}

fn pane_block(title: &str, active: bool) -> Block<'_> {
    if active {
        let accent = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        Block::default()
            .borders(Borders::ALL)
            .border_style(accent)
            .title(Span::styled(format!("▎ {title}"), accent))
    } else {
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                format!("  {title}"),
                Style::default().fg(Color::Gray),
            ))
    }
}

// Layout: "<rank 3>" + "<bar 5>" + "  " (2) = 10 fixed left columns; then
// snippet fills the remaining width minus a right-side reservation for the
// [slug] tag (with a 2-space margin from the snippet text).
fn result_row(
    idx: usize,
    r: &crate::app::ResultRow,
    top_score: f32,
    interior: usize,
) -> ListItem<'_> {
    const LEFT_FIXED: usize = 10;
    let is_top = idx < 3;
    let snippet_style = if is_top {
        Style::default().add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    // Tag cross-project hits with the friendly project name, not the slug.
    let slug_tag = r
        .project_root
        .as_deref()
        .map(|root| format!("[{}]", truncate(&crate::app::project_name(root), 14)));
    let slug_w = slug_tag.as_ref().map_or(0, |s| s.chars().count());
    let right_reserve = if slug_w > 0 { slug_w + 2 } else { 0 };
    let snippet_room = interior
        .saturating_sub(LEFT_FIXED)
        .saturating_sub(right_reserve)
        .max(8);
    let snip = snippet(&r.content, snippet_room);
    let pad = snippet_room.saturating_sub(snip.chars().count());

    let mut spans: Vec<Span<'_>> = Vec::with_capacity(6);
    if is_top {
        spans.push(Span::styled(
            format!("{}. ", idx + 1),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    } else {
        spans.push(Span::raw("   "));
    }
    spans.push(Span::styled(
        relevance_bar(r.score, top_score),
        Style::default().fg(Color::Cyan),
    ));
    spans.push(Span::raw("  "));
    spans.push(Span::styled(snip, snippet_style));
    if let Some(tag) = slug_tag {
        spans.push(Span::raw(" ".repeat(pad + 2)));
        spans.push(Span::styled(tag, Style::default().fg(Color::Magenta)));
    }
    ListItem::new(Line::from(spans))
}

fn relevance_bar(score: f32, top: f32) -> String {
    const CELLS: usize = 5;
    let ratio = if top > f32::EPSILON {
        (score / top).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = (ratio * CELLS as f32).round() as usize;
    let filled = filled.min(CELLS);
    let mut out = String::with_capacity(CELLS * 3);
    for _ in 0..filled {
        out.push('▰');
    }
    for _ in filled..CELLS {
        out.push('▱');
    }
    out
}

fn snippet(content: &str, width: usize) -> String {
    for line in content.lines() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if s.starts_with('#') {
            continue;
        }
        if s.starts_with("<!--") && s.ends_with("-->") {
            continue;
        }
        // Transcript turns lead with role labels; skip them so the row
        // shows the actual prompt/reply text instead of "USER:".
        if s == "USER:" || s == "ASSISTANT:" {
            continue;
        }
        if s.len() > width {
            return s.chars().take(width).collect();
        }
        return s.to_owned();
    }
    "(heading only)".to_owned()
}

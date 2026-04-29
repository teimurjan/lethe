//! ratatui rendering. Three-pane layout with optional bottom detail
//! panel and a footer hint row.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus, Scope, Stats, ToastKind};

const FOOTER: &str = "↑/↓ nav · ⏎ search/open · esc back · tab focus · y copy · ^q quit";
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

    // Toast last so it floats above everything else.
    if app.toast.is_some() {
        draw_toast(frame, area, app);
    }
}

fn draw_body(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(20)])
        .split(area);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(8)])
        .split(chunks[0]);
    draw_projects(frame, left[0], app);
    draw_stats(frame, left[1], app);
    draw_results_pane(frame, chunks[1], app);
}

fn draw_projects(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let title = format!("Projects ({})", app.projects.len());
    let active = matches!(app.focus, Focus::Projects);
    let block = pane_block(&title, active);

    let current_slug = match &app.scope {
        Scope::Single(e) => Some(e.slug.as_str()),
        Scope::AllProjects => None,
    };
    let items: Vec<ListItem> = app
        .projects
        .iter()
        .map(|p| {
            let label = if Some(p.slug.as_str()) == current_slug {
                format!("▸ {}", p.slug)
            } else {
                format!("  {}", p.slug)
            };
            ListItem::new(Line::from(label))
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
    let active = matches!(app.focus, Focus::Search);
    let prompt = format!("{} ▸ ", app.scope.label());
    let body = Line::from(vec![
        Span::styled(prompt, Style::default().fg(Color::Cyan)),
        Span::raw(&app.search_input),
        Span::styled(
            if active { "█" } else { " " },
            Style::default().fg(Color::White),
        ),
    ]);
    let block = pane_block("Search", active);
    let para = Paragraph::new(body).block(block);
    frame.render_widget(para, area);
}

fn draw_results(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let n = app.results.len();
    let title = if n == 0 {
        "Results".to_owned()
    } else {
        format!("Results ({n})")
    };
    let active = matches!(app.focus, Focus::Results);
    let block = pane_block(&title, active);

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
        let msg = if app.last_query.is_empty() {
            "(type a query and press enter)"
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

    let items: Vec<ListItem> = app
        .results
        .iter()
        .map(|r| {
            let mut spans: Vec<Span<'_>> = Vec::with_capacity(4);
            if let Some(slug) = &r.project_slug {
                spans.push(Span::styled(
                    format!("[{slug}] "),
                    Style::default().fg(Color::Magenta),
                ));
            }
            spans.push(Span::styled(
                format!("{:+.2}", r.score),
                Style::default().fg(Color::Cyan),
            ));
            spans.push(Span::raw("  "));
            spans.push(Span::raw(snippet(&r.content, 200)));
            ListItem::new(Line::from(spans))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.result_selection.min(n - 1)));
    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    frame.render_stateful_widget(list, area, &mut state);
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
    let body = app.detail.clone().unwrap_or_default();
    let para = Paragraph::new(body).block(block).wrap(Wrap { trim: false });
    frame.render_widget(para, area);
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
        if s.len() > width {
            return s.chars().take(width).collect();
        }
        return s.to_owned();
    }
    "(heading only)".to_owned()
}

//! ratatui rendering. Three-pane layout with optional bottom detail
//! panel and a footer hint row.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, Focus, Scope};

const FOOTER: &str = "↑/↓ nav · ⏎ search/open · esc back · tab focus · ^q quit";

pub fn draw(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // header
            Constraint::Min(3),    // body
            if app.detail.is_some() {
                Constraint::Length(8)
            } else {
                Constraint::Length(0)
            },
            Constraint::Length(1), // footer
        ])
        .split(area);

    draw_header(frame, chunks[0], app);
    draw_body(frame, chunks[1], app);
    if app.detail.is_some() {
        draw_detail(frame, chunks[2], app);
    }
    draw_footer(frame, chunks[3]);
}

fn draw_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let scope = app.scope.label();
    let header = Line::from(vec![
        Span::styled("lethe", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  ›  "),
        Span::styled(scope, Style::default().fg(Color::Cyan)),
        Span::raw("    "),
        Span::styled(&app.status, Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(header), area);
}

fn draw_body(frame: &mut Frame<'_>, area: Rect, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(28), Constraint::Min(20)])
        .split(area);
    draw_projects(frame, chunks[0], app);
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

    if app.results.is_empty() {
        let para = Paragraph::new(Line::from(Span::styled(
            "(type a query and press enter)",
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

fn draw_detail(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Detail (Esc to close)")
        .style(Style::default().fg(Color::DarkGray));
    let body = app.detail.clone().unwrap_or_default();
    let para = Paragraph::new(body).block(block).wrap(Wrap { trim: false });
    frame.render_widget(para, area);
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
    let style = if active {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    Block::default()
        .borders(Borders::ALL)
        .title(title)
        .style(style)
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

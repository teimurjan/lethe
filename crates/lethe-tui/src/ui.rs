//! ratatui rendering for the palette-first memory TUI. The home screen is
//! a search prompt over recents/results; Enter opens a full-screen reader;
//! F2 opens a settings modal. Colors come from [`crate::theme`].

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{App, DeleteForm, Listing, Mode, ResultRow, Scope, ToastKind, SETTINGS};
use crate::search_worker::Phase;
use crate::theme;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const SPINNER: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

pub fn draw(frame: &mut Frame<'_>, app: &mut App) {
    let area = frame.area();
    // Paint the whole surface with the theme background first.
    frame.render_widget(Block::default().style(theme::base()), area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(1)])
        .split(area);

    match app.mode {
        Mode::Home => draw_home(frame, rows[0], app),
        Mode::Reader => draw_reader(frame, rows[0], app),
    }
    draw_footer(frame, rows[1], app);

    if app.overlay.is_some() {
        draw_overlay(frame, area, app);
    }
    if app.toast.is_some() {
        draw_toast(frame, area, app);
    }
}

// ------------------------------------------------------------------ home

fn draw_home(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let scope_h = if app.scope_open { 3 } else { 0 };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),       // header
            Constraint::Length(3),       // search prompt
            Constraint::Length(scope_h), // scope picker (optional)
            Constraint::Min(3),          // results
        ])
        .split(area);

    draw_header(frame, rows[0], app);
    draw_prompt(frame, rows[1], app);
    if app.scope_open {
        draw_scope(frame, rows[2], app);
    }
    draw_list(frame, rows[3], app);
}

fn draw_header(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let left = vec![
        Span::styled("◆ lethe", Style::default().fg(theme::GREEN).bg(theme::BG)),
        Span::styled(format!(" v{VERSION}"), theme::dim()),
    ];
    let right = match app.stats() {
        Some(s) => format!(
            "{} projects · {} memories",
            s.by_project.len(),
            format_count(s.total)
        ),
        None => "indexing…".to_owned(),
    };
    let left_w: usize = left.iter().map(|s| s.content.chars().count()).sum();
    let pad = (area.width as usize)
        .saturating_sub(left_w)
        .saturating_sub(right.chars().count());
    let mut spans = left;
    spans.push(Span::raw(" ".repeat(pad)));
    spans.push(Span::styled(right, theme::dim()));
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_prompt(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let mut spans = vec![
        Span::styled("❯ ", Style::default().fg(theme::GREEN).bg(theme::BG)),
        Span::styled(&app.search_input, theme::base()),
        Span::styled("█", Style::default().fg(theme::FG).bg(theme::BG)),
    ];
    let hint = if app.search_input.is_empty() {
        format!("   type to search · {}", app.scope.label())
    } else if app.searching || app.search_due.is_some() {
        format!("   {} searching…", spinner_frame())
    } else {
        format!("   {} matches · {}", app.results.len(), app.scope.label())
    };
    spans.push(Span::styled(hint, theme::dim()));
    frame.render_widget(
        Paragraph::new(Line::from(spans)).block(theme::pane("Search", true)),
        area,
    );
}

fn draw_scope(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let counts = app.stats().map(|s| &s.by_project);
    let count_of = |name: &str| -> usize {
        counts
            .and_then(|c| c.iter().find(|(n, _)| n == name).map(|(_, n)| *n))
            .unwrap_or(0)
    };
    let total: usize = app.stats().map_or(0, |s| s.total);

    let mut spans = Vec::new();
    let all_sel = matches!(app.scope, Scope::AllProjects);
    spans.push(scope_chip(&format!("all ({total})"), all_sel));
    for p in &app.projects {
        let name = crate::app::project_name(&p.root);
        let sel = matches!(&app.scope, Scope::Single(e) if e.slug == p.slug);
        spans.push(Span::styled("  ", theme::dim()));
        spans.push(scope_chip(&format!("{name} ({})", count_of(&name)), sel));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).block(theme::pane("Scope · Tab", false)),
        area,
    );
}

fn scope_chip(text: &str, selected: bool) -> Span<'static> {
    if selected {
        Span::styled(format!(" {text} "), theme::selection())
    } else {
        Span::styled(text.to_owned(), theme::dim())
    }
}

/// The recents / results pane. Recents rows are grouped under dim
/// per-project headers; search results are flat with query highlight.
fn draw_list(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let title = list_title(app);
    let block = theme::pane(&title, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if app.results.is_empty() {
        // A query is in flight (debounce armed or worker running): spinner.
        if !app.search_input.trim().is_empty() && (app.searching || app.search_due.is_some()) {
            let phase = app.search_phase.map_or("searching", Phase::label);
            let line = Line::from(vec![
                Span::styled(
                    spinner_frame(),
                    Style::default()
                        .fg(theme::CYAN)
                        .bg(theme::BG)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("  ", theme::base()),
                Span::styled(format!("{phase}…"), theme::base()),
            ]);
            frame.render_widget(Paragraph::new(line), inner);
            return;
        }
        let msg = if app.mem.is_none() {
            "loading memories…"
        } else if app.projects.is_empty() {
            "no projects — run `lethe index` in a repo"
        } else if app.search_input.is_empty() {
            "no memories yet"
        } else {
            "no matches"
        };
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(msg, theme::dim()))),
            inner,
        );
        return;
    }

    let width = inner.width as usize;
    let query = app.search_input.trim().to_lowercase();
    let top_score = app.results.first().map_or(0.0, |r| r.score);

    // Build display lines, tracking which line carries the selected memory.
    let mut lines: Vec<Line> = Vec::new();
    let mut sel_line = 0usize;
    let mut last_project: Option<&str> = None;
    for (i, r) in app.results.iter().enumerate() {
        if app.listing == Listing::Recents && last_project != Some(&r.project_name) {
            last_project = Some(&r.project_name);
            lines.push(Line::from(Span::styled(
                r.project_name.clone(),
                theme::dim().add_modifier(Modifier::BOLD),
            )));
        }
        if i == app.result_selection {
            sel_line = lines.len();
        }
        lines.push(result_line(
            r,
            i == app.result_selection,
            &query,
            top_score,
            width,
            app,
        ));
    }

    let visible = inner.height as usize;
    let offset = sel_line
        .saturating_sub(visible.saturating_sub(1))
        .min(lines.len().saturating_sub(visible));
    frame.render_widget(Paragraph::new(lines).scroll((offset as u16, 0)), inner);
}

fn list_title(app: &App) -> String {
    let n = app.results.len();
    match app.listing {
        Listing::Recents => "Recent".to_owned(),
        Listing::Semantic if app.searching || app.search_due.is_some() => {
            "Results — searching".to_owned()
        }
        Listing::Semantic => format!("Results ({n}) — ranked"),
    }
}

fn result_line(
    r: &ResultRow,
    selected: bool,
    query: &str,
    top_score: f32,
    width: usize,
    app: &App,
) -> Line<'static> {
    // Spans set foreground only; the whole line's background comes from
    // `line.style`, so the selection bar spans the full width uniformly.
    let marker = if selected { "▸ " } else { "  " };
    let mut spans: Vec<Span<'static>> = vec![Span::styled(
        marker.to_owned(),
        Style::default().fg(theme::ACCENT),
    )];

    if app.listing == Listing::Semantic {
        spans.push(Span::styled(
            format!("{} ", relevance_bar(r.score, top_score)),
            Style::default().fg(theme::CYAN),
        ));
    }

    // Right-side metadata: "· project · date" (project only in all-scope).
    let mut meta = String::new();
    if r.project_root.is_some() {
        meta.push_str(" · ");
        meta.push_str(&r.project_name);
    }
    if let Some(d) = short_date(r.date_epoch) {
        meta.push_str(" · ");
        meta.push_str(&d);
    }

    let used: usize = spans.iter().map(|s| s.content.chars().count()).sum();
    let snippet_room = width
        .saturating_sub(used)
        .saturating_sub(meta.chars().count())
        .max(8);
    let snip = truncate(crate::app::first_line(&r.content), snippet_room);

    let fg = if selected { theme::SEL_FG } else { theme::FG };
    spans.extend(highlight(&snip, query, fg));

    // Pad so the selection bar / metadata reach the right edge.
    let pad = snippet_room.saturating_sub(snip.chars().count());
    spans.push(Span::raw(" ".repeat(pad)));
    spans.push(Span::styled(meta, Style::default().fg(theme::DIM)));

    let mut line = Line::from(spans);
    line.style = if selected {
        theme::selection()
    } else {
        theme::base()
    };
    line
}

/// Split `text` into owned spans, painting case-insensitive `query` matches
/// in yellow over the base foreground `fg`.
fn highlight(text: &str, query: &str, fg: ratatui::style::Color) -> Vec<Span<'static>> {
    let base = Style::default().fg(fg);
    if query.is_empty() {
        return vec![Span::styled(text.to_owned(), base)];
    }
    let hay = text.to_lowercase();
    let hit = Style::default()
        .fg(theme::YELLOW)
        .add_modifier(Modifier::BOLD);
    let mut spans = Vec::new();
    let mut cursor = 0;
    while let Some(pos) = hay[cursor..].find(query) {
        let start = cursor + pos;
        let end = start + query.len();
        if start > cursor {
            spans.push(Span::styled(text[cursor..start].to_owned(), base));
        }
        spans.push(Span::styled(text[start..end].to_owned(), hit));
        cursor = end;
    }
    if cursor < text.len() {
        spans.push(Span::styled(text[cursor..].to_owned(), base));
    }
    spans
}

// ---------------------------------------------------------------- reader

fn draw_reader(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let Some(r) = app.selected() else {
        return;
    };
    let idx = app.result_selection + 1;
    let total = app.results.len();
    let title = format!("{} ▸ memory {idx}/{total}", r.project_name);
    let block = theme::pane(&title, true);
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let (user, assistant) = split_roles(&r.content);
    let mut lines: Vec<Line> = Vec::new();

    // User sentence pinned at top, cyan bold.
    let headline = if user.is_empty() {
        crate::app::first_line(&r.content).to_owned()
    } else {
        user.join(" ")
    };
    lines.push(Line::from(Span::styled(
        format!("\"{}\"", headline.trim()),
        Style::default()
            .fg(theme::CYAN)
            .bg(theme::BG)
            .add_modifier(Modifier::BOLD),
    )));

    // Meta line.
    let mut meta = Vec::new();
    if let Some(d) = long_date(r.date_epoch) {
        meta.push(d);
    }
    if let Some(src) = crate::app::source_label(&r.content) {
        meta.push(src.to_owned());
    }
    meta.push(format!("{} chars", r.content.len()));
    lines.push(Line::from(Span::styled(meta.join(" · "), theme::dim())));
    lines.push(Line::from(Span::styled(
        "─".repeat(inner.width as usize),
        theme::dim(),
    )));
    lines.push(Line::from(Span::styled(
        "ASSISTANT WORK",
        Style::default()
            .fg(theme::GREEN)
            .bg(theme::BG)
            .add_modifier(Modifier::BOLD),
    )));
    for l in &assistant {
        lines.push(Line::from(Span::styled(l.clone(), theme::base())));
    }

    let total_lines = lines.len();
    let body = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(1)])
        .split(inner);
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((app.reader_scroll, 0)),
        body[0],
    );

    // Scroll indicator.
    let visible = body[0].height as usize;
    let pct = if total_lines <= visible {
        100
    } else {
        (((app.reader_scroll as usize + visible).min(total_lines)) * 100 / total_lines).min(100)
    };
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!("▼ {pct}% · ↑↓ scroll · ←→ prev/next"),
            theme::dim(),
        ))),
        body[1],
    );
}

/// Split a transcript chunk into its USER and ASSISTANT bodies, dropping the
/// anchor line and role markers.
fn split_roles(content: &str) -> (Vec<String>, Vec<String>) {
    let mut user = Vec::new();
    let mut assistant = Vec::new();
    let mut target = &mut user;
    for line in content.lines() {
        let t = line.trim();
        if t.starts_with("<!--") && t.ends_with("-->") {
            continue;
        }
        match t {
            "USER:" => target = &mut user,
            "ASSISTANT:" => target = &mut assistant,
            _ => target.push(line.trim_end().to_owned()),
        }
    }
    (user, assistant)
}

// -------------------------------------------------------------- overlays

fn draw_overlay(frame: &mut Frame<'_>, area: Rect, app: &App) {
    use crate::app::Overlay;
    // Dim the base view behind the modal.
    match app.overlay.as_ref() {
        Some(Overlay::Settings(cursor)) => draw_settings(frame, area, *cursor),
        Some(Overlay::AddProject(buf)) => draw_add_project(frame, area, buf),
        Some(Overlay::DeleteRecords(form)) => draw_delete_records(frame, area, form, app),
        Some(Overlay::Confirm(c)) => draw_confirm(frame, area, c),
        Some(Overlay::Cleanup(list)) => draw_cleanup(frame, area, list),
        Some(Overlay::Busy(label)) => draw_busy(frame, area, label),
        None => {}
    }
}

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

fn draw_settings(frame: &mut Frame<'_>, area: Rect, cursor: usize) {
    let mut lines: Vec<Line> = Vec::new();
    for (i, item) in SETTINGS.iter().enumerate() {
        let selected = i == cursor;
        let key_style = if item.danger {
            Style::default().fg(theme::RED).bg(sel_bg(selected))
        } else {
            Style::default().fg(theme::GREEN).bg(sel_bg(selected))
        };
        let label_style = row_style(selected);
        let mut spans = vec![
            Span::styled(if selected { "▸ " } else { "  " }, label_style),
            Span::styled(format!("{} ", item.key), key_style),
            Span::styled(format!("{:<26}", item.label), label_style),
        ];
        if !item.hint.is_empty() {
            spans.push(Span::styled(item.hint.to_owned(), theme::dim()));
        }
        lines.push(Line::from(spans));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "↑↓ / letter · ↵ select · Esc close",
        theme::dim(),
    )));
    let rect = centered_rect(56, lines.len() as u16 + 2, area);
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines).block(modal_block("Settings", theme::ACCENT)),
        rect,
    );
}

fn draw_add_project(frame: &mut Frame<'_>, area: Rect, buf: &str) {
    let lines = vec![
        Line::from(Span::styled("Path to a repo to index:", theme::base())),
        Line::from(""),
        Line::from(vec![
            Span::styled("❯ ", Style::default().fg(theme::GREEN).bg(theme::BG)),
            Span::styled(buf.to_owned(), theme::base()),
            Span::styled("█", Style::default().fg(theme::FG).bg(theme::BG)),
        ]),
        Line::from(""),
        Line::from(Span::styled("↵ add & index · Esc cancel", theme::dim())),
    ];
    let rect = centered_rect(58, lines.len() as u16 + 2, area);
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines).block(modal_block("Add project", theme::ACCENT)),
        rect,
    );
}

fn draw_delete_records(frame: &mut Frame<'_>, area: Rect, form: &DeleteForm, app: &App) {
    let project = match form.project {
        None => "all".to_owned(),
        Some(i) => app
            .projects
            .get(i)
            .map_or_else(|| "?".to_owned(), |p| crate::app::project_name(&p.root)),
    };
    let matches = count_matches(app, form);
    let days_label = if form.days == 0 {
        "everything".to_owned()
    } else {
        format!("{} days", form.days)
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("Project    ", theme::base()),
            Span::styled(format!(" {project} ▾ "), theme::selection()),
            Span::styled("  ←→ all / pick", theme::dim()),
        ]),
        Line::from(vec![
            Span::styled("Older than ", theme::base()),
            Span::styled(
                format!(" {days_label} "),
                Style::default().fg(theme::CYAN).bg(theme::BG),
            ),
            Span::styled("  ↑↓ · 0 = everything", theme::dim()),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            format!("⚠ {matches} record(s) match — cannot be undone"),
            Style::default().fg(theme::YELLOW).bg(theme::BG),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("type ", theme::base()),
            Span::styled("delete", Style::default().fg(theme::RED).bg(theme::BG)),
            Span::styled(" to confirm: ", theme::base()),
            Span::styled(
                form.confirm.clone(),
                Style::default().fg(theme::CYAN).bg(theme::BG),
            ),
            Span::styled("█", Style::default().fg(theme::FG).bg(theme::BG)),
        ]),
        Line::from(""),
        Line::from(Span::styled("↵ confirm · Esc cancel", theme::dim())),
    ];
    let rect = centered_rect(56, lines.len() as u16 + 2, area);
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines).block(modal_block("Delete state records", theme::RED)),
        rect,
    );
}

/// Live count of records the delete form would remove, from the in-memory
/// index (no DB round-trip).
fn count_matches(app: &App, form: &DeleteForm) -> usize {
    let Some(mem) = &app.mem else { return 0 };
    let root = form
        .project
        .and_then(|i| app.projects.get(i))
        .map(|p| p.root.clone());
    let cutoff = if form.days == 0 {
        i64::MAX
    } else {
        now_epoch().saturating_sub(i64::from(form.days) * 86_400)
    };
    mem.items
        .iter()
        .filter(|it| root.as_ref().is_none_or(|r| &it.project_root == r))
        .filter(|it| form.days == 0 || it.date_epoch < cutoff)
        .count()
}

fn draw_confirm(frame: &mut Frame<'_>, area: Rect, c: &crate::app::Confirm) {
    let lines: Vec<Line> = c.lines.iter().map(|l| Line::from(l.clone())).collect();
    let width = c
        .lines
        .iter()
        .map(|l| l.chars().count())
        .chain(std::iter::once(c.title.chars().count()))
        .max()
        .unwrap_or(20)
        .clamp(24, 72) as u16
        + 4;
    let rect = centered_rect(width, c.lines.len() as u16 + 2, area);
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines)
            .block(modal_block(&c.title, theme::RED))
            .wrap(Wrap { trim: false }),
        rect,
    );
}

fn draw_cleanup(frame: &mut Frame<'_>, area: Rect, list: &crate::app::CleanupList) {
    let sel = list.selected.iter().filter(|s| **s).count();
    let bytes: u64 = list
        .items
        .iter()
        .zip(&list.selected)
        .filter(|(_, s)| **s)
        .map(|(it, _)| it.bytes)
        .sum();
    let title = format!(
        "Clean up — {sel}/{} selected, {}",
        list.items.len(),
        lethe_core::maintenance::human_bytes(bytes)
    );

    let rect = centered_rect(78, (list.items.len() as u16 + 4).min(area.height), area);
    let inner_rows = rect.height.saturating_sub(2) as usize;
    let body_rows = inner_rows.saturating_sub(1).max(1);
    let start = list
        .cursor
        .saturating_sub(body_rows.saturating_sub(1))
        .min(list.items.len().saturating_sub(body_rows));
    let mut lines: Vec<Line> = Vec::new();
    for (i, it) in list.items.iter().enumerate().skip(start).take(body_rows) {
        let checked = list.selected.get(i).copied().unwrap_or(false);
        let name = it.path.file_name().map_or_else(
            || it.path.to_string_lossy().into_owned(),
            |n| n.to_string_lossy().into_owned(),
        );
        let row = format!(
            "{} {:<11} {:>9}  {}",
            if checked { "[x]" } else { "[ ]" },
            it.reason.label(),
            lethe_core::maintenance::human_bytes(it.bytes),
            name,
        );
        let style = if i == list.cursor {
            theme::selection()
        } else if checked {
            theme::base()
        } else {
            theme::dim()
        };
        lines.push(Line::from(Span::styled(row, style)));
    }
    lines.push(Line::from(Span::styled(
        "space toggle · a all · ↵ delete selected · esc cancel",
        theme::dim(),
    )));
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(lines).block(modal_block(&title, theme::ACCENT)),
        rect,
    );
}

fn draw_busy(frame: &mut Frame<'_>, area: Rect, label: &str) {
    let body = Line::from(vec![
        Span::styled(
            spinner_frame(),
            Style::default()
                .fg(theme::CYAN)
                .bg(theme::BG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", theme::base()),
        Span::styled(label.to_owned(), theme::base()),
    ]);
    let w = (label.chars().count() as u16 + 8).clamp(20, 60);
    let rect = centered_rect(w, 3, area);
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(body).block(modal_block("Working", theme::ACCENT)),
        rect,
    );
}

fn modal_block(title: &str, accent: ratatui::style::Color) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(accent).bg(theme::BG))
        .style(Style::default().bg(theme::BG))
        .title(Span::styled(
            format!(" {title} "),
            Style::default()
                .fg(accent)
                .bg(theme::BG)
                .add_modifier(Modifier::BOLD),
        ))
}

// ---------------------------------------------------------------- footer

fn draw_footer(frame: &mut Frame<'_>, area: Rect, app: &App) {
    let keys: &[(&str, &str)] = match app.mode {
        Mode::Home => &[
            ("↑↓", "move"),
            ("↵", "open"),
            ("Tab", "scope"),
            ("F2", "settings"),
            ("^c", "copy"),
            ("F10", "quit"),
        ],
        Mode::Reader => &[
            ("↑↓", "scroll"),
            ("←→", "prev/next"),
            ("c", "copy"),
            ("Esc", "back"),
        ],
    };
    let mut spans = Vec::new();
    for (i, (k, label)) in keys.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled("   ", theme::dim()));
        }
        spans.push(Span::styled(
            format!(" {k} "),
            Style::default().fg(theme::FG).bg(theme::SEL_BG),
        ));
        spans.push(Span::styled(format!(" {label}"), theme::dim()));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_toast(frame: &mut Frame<'_>, full: Rect, app: &App) {
    let Some(toast) = &app.toast else { return };
    let msg = toast.msg.clone();
    let border = match toast.kind {
        ToastKind::Info => theme::GREEN,
        ToastKind::Error => theme::RED,
    };

    let inner_w = (msg.chars().count() as u16).saturating_add(2);
    let w = inner_w.saturating_add(2).min(full.width.saturating_sub(2));
    let h = 3u16;
    if full.width <= w + 2 || full.height <= h + 2 {
        return;
    }
    let rect = Rect {
        x: full.x + full.width - w - 2,
        y: full.y + full.height - h - 2,
        width: w,
        height: h,
    };
    frame.render_widget(Clear, rect);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default()
                .fg(theme::BG)
                .bg(border)
                .add_modifier(Modifier::BOLD),
        )))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(border).bg(theme::BG))
                .style(Style::default().bg(theme::BG)),
        ),
        rect,
    );
}

// ----------------------------------------------------------------- utils

fn sel_bg(selected: bool) -> ratatui::style::Color {
    if selected {
        theme::SEL_BG
    } else {
        theme::BG
    }
}

fn row_style(selected: bool) -> Style {
    if selected {
        theme::selection()
    } else {
        Style::default().fg(theme::FG).bg(theme::BG)
    }
}

fn relevance_bar(score: f32, top: f32) -> String {
    const CELLS: usize = 5;
    let ratio = if top > f32::EPSILON {
        (score / top).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = ((ratio * CELLS as f32).round() as usize).min(CELLS);
    let mut out = String::with_capacity(CELLS * 3);
    for _ in 0..filled {
        out.push('▰');
    }
    for _ in filled..CELLS {
        out.push('▱');
    }
    out
}

fn spinner_frame() -> &'static str {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    SPINNER[((ms / 80) as usize) % SPINNER.len()]
}

fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Civil (year, month, day) from unix seconds — Hinnant's algorithm.
fn ymd(epoch: i64) -> (i64, u32, u32) {
    let days = epoch.div_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

/// `Jul 14` — `None` for the sentinel `0` epoch (unknown date).
fn short_date(epoch: i64) -> Option<String> {
    if epoch <= 0 {
        return None;
    }
    let (_, m, d) = ymd(epoch);
    Some(format!("{} {d:02}", MONTHS[(m as usize - 1) % 12]))
}

/// `Jul 14 2026`.
fn long_date(epoch: i64) -> Option<String> {
    if epoch <= 0 {
        return None;
    }
    let (y, m, d) = ymd(epoch);
    Some(format!("{} {d:02} {y}", MONTHS[(m as usize - 1) % 12]))
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

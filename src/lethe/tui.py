"""Interactive TUI for browsing registered lethe projects and searching.

Install with the optional extra:

    uv pip install -e '.[tui]'
    lethe tui

Layout::

    ┌─ lethe ──────────── › all projects ──────────────────────┐
    │ Projects (7)          │ Results                           │
    │ > lethe               │ (type a query and press enter)    │
    │   claude-code         │                                   │
    │   agents-repo         │                                   │
    ├───────────────────────┴───────────────────────────────────┤
    │ detail (shown on Enter on a result; Esc to close)         │
    ├────────────────────────────────────────────────────────── │
    │ all projects ▸ █                                          │
    │ ↑/↓ nav · ⏎ search/open · esc back · tab focus · ^q quit  │
    └──────────────────────────────────────────────────────────┘

Scope is either ``all projects`` at the top level (queries hit every
registered project via :class:`~lethe.union_store.UnionStore`) or a single
project when the user ``Enter``s a row in the Projects list.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, Static

from lethe import _registry
from lethe._registry import ProjectEntry


@dataclass
class Scope:
    """Current search scope: either all projects, or a single project."""

    project: ProjectEntry | None  # None → all registered projects

    @property
    def short(self) -> str:
        return self.project.slug if self.project else "all projects"

    @property
    def breadcrumb(self) -> str:
        return f"› {self.project.slug}" if self.project else "› all projects"


@dataclass
class SearchHit:
    """Unified shape for single-project and cross-project results."""

    id: str
    content: str
    score: float
    project_slug: str | None  # None when inside a single-project scope

    def snippet(self, width: int = 200) -> str:
        for line in self.content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("<!--") and stripped.endswith("-->"):
                continue
            return stripped[:width]
        return "(heading only)"


class ProjectItem(ListItem):
    """ListItem wrapper that carries its ProjectEntry for event handling."""

    def __init__(self, entry: ProjectEntry, *, is_current: bool = False) -> None:
        label = f"▸ {entry.slug}" if is_current else f"  {entry.slug}"
        static = Static(label, markup=False)
        if is_current:
            static.add_class("current-scope")
        super().__init__(static)
        self.entry = entry


class ResultItem(ListItem):
    def __init__(self, hit: SearchHit) -> None:
        score = f"[cyan]{hit.score:+.2f}[/]"
        tag = f"[magenta]{hit.project_slug}[/] " if hit.project_slug else ""
        line = f"{tag}{score}  {hit.snippet(200)}"
        super().__init__(Static(line))
        self.hit = hit


class LetheApp(App[None]):
    """Browse projects, search within or across them."""

    CSS = """
    Screen { layout: vertical; }

    #body {
        height: 1fr;
        min-height: 5;
    }

    #projects-pane {
        width: 32;
        background: $surface;
        border-right: solid $primary-darken-2;
    }
    #results-pane {
        width: 1fr;
        background: $surface;
    }

    .pane-title {
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $accent;
        text-style: bold;
    }

    #projects, #results {
        height: 1fr;
        background: $surface;
        border: none;
        padding: 0;
    }
    #projects > ListItem, #results > ListItem {
        padding: 0 1;
    }
    .current-scope { color: $accent; text-style: bold; }
    .dim { color: $text-muted; }

    #detail-wrap {
        height: auto;
        max-height: 14;
        display: none;
        background: $panel;
        border-top: solid $primary-darken-2;
    }
    #detail-title {
        height: 1;
        padding: 0 1;
        background: $boost;
        color: $accent;
        text-style: bold;
    }
    #detail {
        padding: 0 1;
        color: $text;
    }

    #search-row {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $primary-darken-2;
    }
    #search-label { width: auto; color: $accent; padding: 1 1 0 0; }
    #search-input { width: 1fr; }
    #search-input:focus { border: tall $accent; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("escape", "escape", "Back/close"),
        Binding("tab", "focus_next", "Next pane", show=False),
        Binding("shift+tab", "focus_prev", "Prev pane", show=False),
        Binding("ctrl+l", "focus_search", "Search"),
        Binding("ctrl+p", "focus_projects", "Projects"),
        Binding("ctrl+r", "focus_results", "Results"),
    ]

    def __init__(self, projects: list[ProjectEntry]) -> None:
        super().__init__()
        self._projects = projects
        self._scope = Scope(project=None)
        self._searching = False

    # ---- layout -----------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)

        with Horizontal(id="body"):
            with Vertical(id="projects-pane"):
                yield Static(self._projects_title(), id="projects-title", classes="pane-title")
                yield ListView(id="projects")
            with Vertical(id="results-pane"):
                with Horizontal(id="search-row"):
                    yield Label(f"{self._scope.short} ▸", id="search-label")
                    yield Input(
                        placeholder="type a query and press enter…",
                        id="search-input",
                        select_on_focus=False,
                    )
                yield Static("Results", id="results-title", classes="pane-title")
                yield ListView(id="results")

        with Vertical(id="detail-wrap"):
            yield Static("Detail  (Esc to close)", id="detail-title")
            with VerticalScroll():
                yield Static("", id="detail")

        yield Footer()

    def on_mount(self) -> None:
        self.title = "lethe"
        self.sub_title = self._scope.breadcrumb
        self._populate_projects()
        self._show_results_placeholder()
        self.query_one("#search-input", Input).focus()

    # ---- helpers ----------------------------------------------------------

    def _projects_title(self) -> str:
        n = len(self._projects)
        return f"Projects ({n})" if n else "Projects (none registered)"

    def _populate_projects(self) -> None:
        view = self.query_one("#projects", ListView)
        view.clear()
        if not self._projects:
            view.append(ListItem(Static("(run `lethe index` in a repo)", classes="dim")))
            return
        current_slug = self._scope.project.slug if self._scope.project else None
        for entry in self._projects:
            view.append(ProjectItem(entry, is_current=(entry.slug == current_slug)))

    def _show_results_placeholder(self, text: str = "(type a query and press enter)") -> None:
        view = self.query_one("#results", ListView)
        view.clear()
        view.append(ListItem(Static(text, classes="dim")))

    def _set_scope(self, scope: Scope) -> None:
        self._scope = scope
        self.sub_title = scope.breadcrumb
        self.query_one("#search-label", Label).update(f"{scope.short} ▸")
        self.query_one("#results-title", Static).update("Results")
        self._populate_projects()
        self._show_results_placeholder()
        self._hide_detail()

    def _hide_detail(self) -> None:
        wrap = self.query_one("#detail-wrap")
        self.query_one("#detail", Static).update("")
        wrap.styles.display = "none"

    def _show_detail(self, text: str) -> None:
        self.query_one("#detail", Static).update(text)
        self.query_one("#detail-wrap").styles.display = "block"

    # ---- actions ----------------------------------------------------------

    def action_escape(self) -> None:
        wrap = self.query_one("#detail-wrap")
        if wrap.styles.display != "none":
            self._hide_detail()
            return
        if self._scope.project is not None:
            self._set_scope(Scope(project=None))
            return
        # Top-level + no detail — refocus the search input as a gentle reset.
        self.query_one("#search-input", Input).focus()

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_focus_projects(self) -> None:
        self.query_one("#projects", ListView).focus()

    def action_focus_results(self) -> None:
        self.query_one("#results", ListView).focus()

    # ---- type-to-search: any printable key while focus is elsewhere hops
    # focus to the input and appends the character. Arrows/Tab/Enter/Esc
    # and modifier combos are left alone so list nav + bindings still work.

    def on_key(self, event: Key) -> None:
        input_widget = self.query_one("#search-input", Input)
        if self.focused is input_widget:
            return
        char = event.character
        if not char or len(char) != 1 or not char.isprintable():
            return
        input_widget.value = input_widget.value + char
        input_widget.cursor_position = len(input_widget.value)
        input_widget.focus()
        event.stop()
        event.prevent_default()

    # ---- event handlers ---------------------------------------------------

    @on(ListView.Selected, "#projects")
    def _on_project_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ProjectItem):
            self._set_scope(Scope(project=item.entry))
            self.query_one("#search-input", Input).focus()

    @on(ListView.Selected, "#results")
    def _on_result_selected(self, event: ListView.Selected) -> None:
        item = event.item
        if isinstance(item, ResultItem):
            self.run_worker(self._expand_result(item.hit), exclusive=True, group="expand")

    @on(Input.Submitted, "#search-input")
    def _on_search_submit(self, event: Input.Submitted) -> None:
        query = (event.value or "").strip()
        if not query:
            return
        if self._searching:
            return
        self.run_worker(self._run_search(query), exclusive=True, group="search")

    # ---- workers ----------------------------------------------------------

    async def _run_search(self, query: str) -> None:
        self._searching = True
        results_view = self.query_one("#results", ListView)
        results_view.clear()
        self.query_one("#results-title", Static).update(f"Results — searching for “{query[:40]}”…")
        results_view.loading = True
        self._hide_detail()
        try:
            scope = self._scope
            hits = await asyncio.to_thread(_retrieve, scope, query, 10)
        except Exception as exc:  # pragma: no cover — surface errors in UI
            results_view.loading = False
            results_view.clear()
            results_view.append(ListItem(Static(f"error: {exc}", classes="dim")))
            self.query_one("#results-title", Static).update("Results — error")
            self._searching = False
            return

        results_view.loading = False
        results_view.clear()
        if not hits:
            results_view.append(ListItem(Static("(no results)", classes="dim")))
            self.query_one("#results-title", Static).update("Results (0)")
        else:
            for hit in hits:
                results_view.append(ResultItem(hit))
            self.query_one("#results-title", Static).update(f"Results ({len(hits)}) — ⏎ to expand")
        self._searching = False

    async def _expand_result(self, hit: SearchHit) -> None:
        self._show_detail("loading…")
        try:
            content = await asyncio.to_thread(_expand, self._scope, hit)
        except Exception as exc:  # pragma: no cover
            self._show_detail(f"error: {exc}")
            return
        self._show_detail(content or "(empty)")


# --- retrieval helpers (no UI deps — safe to call from asyncio.to_thread) ---


def _retrieve(scope: Scope, query: str, k: int) -> list[SearchHit]:
    if scope.project is None:
        return _retrieve_union(query, k)
    return _retrieve_single(scope.project, query, k)


def _retrieve_single(entry: ProjectEntry, query: str, k: int) -> list[SearchHit]:
    from lethe.cli import _open_store, load_config, resolve_paths

    paths = resolve_paths(str(entry.root))
    cfg = load_config(paths)
    store = _open_store(paths, cfg, need_encoders=True)
    try:
        rows = store.retrieve(query, k=k)
        store.save()
    finally:
        store.close()
    return [
        SearchHit(id=eid, content=content, score=float(score), project_slug=None)
        for eid, content, score in rows
    ]


def _retrieve_union(query: str, k: int) -> list[SearchHit]:
    from lethe.cli import load_config, resolve_paths
    from lethe.encoders import (
        OnnxBiEncoder,
        OnnxCrossEncoder,
        resolve_bi_encoder_name,
        resolve_cross_encoder_name,
    )
    from lethe.rif import RIFConfig
    from lethe.union_store import UnionStore

    entries = _registry.load()
    roots = [e.root for e in entries]
    if not roots:
        return []

    cfg_source = next(
        (r for r in roots if (r / ".lethe" / "config.toml").exists()),
        roots[0],
    )
    cfg = load_config(resolve_paths(str(cfg_source)))
    bi = OnnxBiEncoder(resolve_bi_encoder_name(cfg["bi_encoder"]))
    xenc = OnnxCrossEncoder(resolve_cross_encoder_name(cfg["cross_encoder"]))
    rif = RIFConfig(
        n_clusters=int(cfg.get("n_clusters", 30)),
        use_rank_gap=bool(cfg.get("use_rank_gap", True)),
    )
    union = UnionStore(
        roots,
        bi_encoder=bi,
        cross_encoder=xenc,
        dim=bi.get_embedding_dimension(),
        rif_config=rif,
    )
    try:
        hits = union.retrieve(query, k=k)
    finally:
        union.close()
    return [
        SearchHit(
            id=h.id,
            content=h.content,
            score=float(h.score),
            project_slug=h.project_slug,
        )
        for h in hits
    ]


def _expand(scope: Scope, hit: SearchHit) -> str:
    """Fetch the full markdown section behind a result row."""
    from lethe.db import MemoryDB

    root = scope.project.root if scope.project else _root_for_hit(hit)
    if root is None:
        return hit.content
    db_path = Path(root) / ".lethe" / "index" / "lethe.duckdb"
    if not db_path.exists():
        return hit.content
    db = MemoryDB(db_path)
    try:
        return db.get_content(hit.id) or hit.content
    finally:
        db.close()


def _root_for_hit(hit: SearchHit) -> Path | None:
    if not hit.project_slug:
        return None
    for entry in _registry.load():
        if entry.slug == hit.project_slug:
            return entry.root
    return None


# --- entry point -------------------------------------------------------------


def run() -> int:
    entries = _registry.load()
    LetheApp(entries).run()
    return 0

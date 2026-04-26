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
from typing import Any

from rich.markup import escape as _escape_markup
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
        # Styling via markup is fine for score/tag (we control them), but
        # hit.snippet() is user content — escape it so stray `[...]` doesn't
        # get interpreted (or crash rendering on malformed markup).
        score = f"[cyan]{hit.score:+.2f}[/]"
        tag = f"[magenta]{_escape_markup(hit.project_slug)}[/] " if hit.project_slug else ""
        line = f"{tag}{score}  {_escape_markup(hit.snippet(200))}"
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

    /* search-row needs height 4 (not 3): Input renders with a tall
       border (height 3), and border-bottom on the row eats 1 more row.
       At height 3, the Input's bottom edge overlaps the row's border
       and the input text gets clipped. */
    #search-row {
        height: 4;
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

        # Retrieval objects are expensive to build (ONNX model load, DuckDB
        # ATTACH). Cache across searches and close in on_unmount so the UI
        # feels snappy on repeated queries within a scope.
        self._bi_encoder: Any = None
        self._cross_encoder: Any = None
        self._union_store: Any = None
        self._project_stores: dict[str, Any] = {}

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

    def on_unmount(self) -> None:
        self._close_union_store()
        self._close_project_stores()
        self._bi_encoder = None
        self._cross_encoder = None

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
        # DuckDB's per-process unique file-handle lock means a `.duckdb`
        # file can be held by either the UnionStore (via ATTACH) OR a
        # per-project MemoryStore, but not both at once. Release the
        # handles owned by the outgoing scope before we ever build the
        # incoming one, otherwise the next retrieve() explodes with
        # "Unique file handle conflict".
        old_is_top = self._scope.project is None
        new_is_top = scope.project is None
        if old_is_top and not new_is_top:
            self._close_union_store()
        elif not old_is_top and new_is_top:
            self._close_project_stores()

        self._scope = scope
        self.sub_title = scope.breadcrumb
        self.query_one("#search-label", Label).update(f"{scope.short} ▸")
        self.query_one("#results-title", Static).update("Results")
        self._populate_projects()
        self._show_results_placeholder()
        self._hide_detail()

    def _close_union_store(self) -> None:
        if self._union_store is None:
            return
        try:
            self._union_store.close()
        except Exception:
            pass
        self._union_store = None

    def _close_project_stores(self) -> None:
        for store in list(self._project_stores.values()):
            try:
                store.close()
            except Exception:
                pass
        self._project_stores.clear()

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

    @on(ListView.Highlighted, "#results")
    def _on_result_highlighted(self, event: ListView.Highlighted) -> None:
        # Live-expand on cursor move — the content is already in memory
        # (see _expand), so this is cheap and avoids the two-step
        # "highlight then press Enter" interaction. Highlighted fires
        # with event.item = None during list rebuild/clear — ignore
        # those so we don't blink the detail pane shut between a search
        # completing and the user arrowing.
        item = event.item
        if isinstance(item, ResultItem):
            self._show_detail(_expand(self._scope, item.hit) or "(empty)")

    @on(ListView.Selected, "#results")
    def _on_result_selected(self, event: ListView.Selected) -> None:
        # Enter is redundant with the highlight-driven expand above, but
        # handle it idempotently so hitting it isn't surprising.
        item = event.item
        if isinstance(item, ResultItem):
            self._show_detail(_expand(self._scope, item.hit) or "(empty)")

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
            hits = await asyncio.to_thread(self._retrieve, scope, query, 10)
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
            self.query_one("#results-title", Static).update(f"Results ({len(hits)}) — ↑/↓ to browse")
            # Hand focus to the results list so arrow keys work immediately.
            # Any printable key hops back to the input via on_key.
            results_view.index = 0
            results_view.focus()
            # Textual doesn't fire Highlighted for a programmatic
            # index=0 on a freshly-populated list, so expand the first
            # hit explicitly to match the live-on-highlight behavior.
            self._show_detail(hits[0].content or "(empty)")
        self._searching = False

    # ---- retrieval (blocking; called via asyncio.to_thread) --------------
    #
    # Instance-cached so repeated searches within a scope don't rebuild the
    # ONNX encoders or re-ATTACH DuckDB files on every query. All resources
    # are released in on_unmount.

    def _retrieve(self, scope: Scope, query: str, k: int) -> list[SearchHit]:
        if scope.project is None:
            return self._retrieve_union(query, k)
        return self._retrieve_single(scope.project, query, k)

    def _ensure_encoders(self, cfg: dict) -> None:
        if self._bi_encoder is not None and self._cross_encoder is not None:
            return
        from lethe.encoders import (
            OnnxBiEncoder,
            OnnxCrossEncoder,
            resolve_bi_encoder_name,
            resolve_cross_encoder_name,
        )
        self._bi_encoder = OnnxBiEncoder(resolve_bi_encoder_name(cfg["bi_encoder"]))
        self._cross_encoder = OnnxCrossEncoder(resolve_cross_encoder_name(cfg["cross_encoder"]))

    def _retrieve_single(self, entry: ProjectEntry, query: str, k: int) -> list[SearchHit]:
        from lethe.cli import load_config, resolve_paths
        from lethe.memory_store import MemoryStore
        from lethe.rif import RIFConfig

        store = self._project_stores.get(entry.slug)
        if store is None:
            paths = resolve_paths(str(entry.root))
            cfg = load_config(paths)
            self._ensure_encoders(cfg)
            rif = RIFConfig(
                n_clusters=int(cfg.get("n_clusters", 30)),
                use_rank_gap=bool(cfg.get("use_rank_gap", True)),
            )
            store = MemoryStore(
                paths.index,
                bi_encoder=self._bi_encoder,
                cross_encoder=self._cross_encoder,
                dim=self._bi_encoder.get_embedding_dimension(),
                rif_config=rif,
            )
            self._project_stores[entry.slug] = store
        rows = store.retrieve(query, k=k)
        # Persist RIF suppression updates — cheap, and guarantees state
        # survives if the user quits without cleanly unmounting.
        store.save()
        return [
            SearchHit(id=eid, content=content, score=float(score), project_slug=None)
            for eid, content, score in rows
        ]

    def _retrieve_union(self, query: str, k: int) -> list[SearchHit]:
        from lethe.cli import load_config, resolve_paths
        from lethe.rif import RIFConfig
        from lethe.union_store import UnionStore

        if self._union_store is None:
            entries = _registry.load()
            roots = [e.root for e in entries]
            if not roots:
                return []
            cfg_source = next(
                (r for r in roots if (r / ".lethe" / "config.toml").exists()),
                roots[0],
            )
            cfg = load_config(resolve_paths(str(cfg_source)))
            self._ensure_encoders(cfg)
            rif = RIFConfig(
                n_clusters=int(cfg.get("n_clusters", 30)),
                use_rank_gap=bool(cfg.get("use_rank_gap", True)),
            )
            self._union_store = UnionStore(
                roots,
                bi_encoder=self._bi_encoder,
                cross_encoder=self._cross_encoder,
                dim=self._bi_encoder.get_embedding_dimension(),
                rif_config=rif,
            )
        hits = self._union_store.retrieve(query, k=k)
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
    """Return the full markdown section behind a result row.

    ``hit.content`` already holds the full section (both ``MemoryStore``
    and ``UnionStore`` load it into memory up front), so this is just an
    indirection hook. Opening a fresh DuckDB connection here used to
    collide with the UnionStore's ATTACH on the same file —
    DuckDB rejects a direct ``connect(path)`` while the file is ATTACHed
    under another alias in the same process — so don't.
    """
    del scope  # kept for API symmetry; no longer needed
    return hit.content


# --- entry point -------------------------------------------------------------


def run() -> int:
    entries = _registry.load()
    LetheApp(entries).run()
    return 0

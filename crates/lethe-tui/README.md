# lethe-tui

Ratatui-based terminal UI for [lethe](https://github.com/teimurjan/lethe).
Shipped as a library so the `lethe` CLI binary can embed it directly without
spawning a sibling process.

```
cargo add lethe-tui
```

## Use

```rust
fn main() -> anyhow::Result<()> {
    lethe_tui::run()?;
    Ok(())
}
```

`run()` owns the full terminal lifecycle: enables raw mode, enters the
alt-screen, runs the event loop, then restores the terminal on exit.
Callers don't need to touch `crossterm` themselves.

## Behaviour

Mirrors the original Textual TUI:

- Live-expand on result-list highlight (no Enter needed)
- First result auto-highlighted after search
- Type-anywhere refocuses the search input
- Ctrl+Q / Ctrl+C quits

## See also

- [Project landing page](https://github.com/teimurjan/lethe)
- [`lethe-cli`](https://crates.io/crates/lethe-cli) — the CLI that embeds this

License: MIT.

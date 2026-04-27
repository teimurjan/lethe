//! End-to-end smoke test for the `lethe` binary. Runs via
//! `cargo test -p lethe-cli`. The `CARGO_BIN_EXE_lethe` env var is
//! injected by Cargo and points at the built binary.

use std::process::Command;

fn lethe() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lethe"))
}

#[test]
fn version_flag_prints_binary_name() {
    let out = lethe().arg("--version").output().expect("spawn lethe");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    assert!(stdout.starts_with("lethe "), "got: {stdout:?}");
}

#[test]
fn help_flag_documents_tui_default() {
    let out = lethe().arg("--help").output().expect("spawn lethe");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    assert!(
        stdout.contains("Persistent memory store"),
        "missing about line: {stdout}"
    );
    assert!(
        stdout.contains("Run with no arguments to launch the interactive TUI"),
        "long_about should mention TUI default: {stdout}"
    );
}

#[test]
fn no_args_in_non_tty_prints_help_and_exits_two() {
    // Cargo's test harness gives us a non-TTY stdout, so the
    // dispatcher should print help and exit 2 instead of trying to
    // open the TUI.
    let out = lethe().output().expect("spawn lethe");
    let code = out.status.code().expect("exit code");
    assert_eq!(
        code,
        2,
        "expected exit 2 for non-TTY no-arg invocation; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf8");
    assert!(
        stdout.contains("Usage: lethe"),
        "should print clap help: {stdout}"
    );
}

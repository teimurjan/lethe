//! End-to-end smoke test for the `lethe` binary. Runs via
//! `cargo test -p lethe-cli`. The `CARGO_BIN_EXE_lethe` env var is
//! injected by Cargo and points at the built binary.

use std::process::Command;

fn lethe() -> Command {
    Command::new(env!("CARGO_BIN_EXE_lethe"))
}

#[test]
fn cleanup_deletes_dead_transcript_folder() {
    // Point HOME / CLAUDE_CONFIG_DIR / CODEX_HOME at temp dirs, plant a
    // Claude project folder whose recorded cwd doesn't exist, and confirm
    // `cleanup --yes` removes it while leaving the store isolated.
    let tmp = std::env::temp_dir().join(format!("lethe-cleanup-smoke-{}", std::process::id()));
    let home = tmp.join("home");
    let cfg = tmp.join("cfg");
    let dead = cfg.join("projects").join("-gone-repo");
    std::fs::create_dir_all(&dead).unwrap();
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(
        dead.join("s.jsonl"),
        "{\"type\":\"user\",\"uuid\":\"u1\",\"sessionId\":\"a\",\"cwd\":\"/no/such/repo/xyz\",\
         \"message\":{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}\n\
         {\"type\":\"assistant\",\"sessionId\":\"a\",\
         \"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"yo\"}]}}\n",
    )
    .unwrap();

    let run = |args: &[&str]| {
        lethe()
            .args(args)
            .env("HOME", &home)
            .env("CLAUDE_CONFIG_DIR", &cfg)
            .env("CODEX_HOME", tmp.join("codex"))
            .output()
            .expect("spawn lethe")
    };

    // Dry run: reports the candidate, does not delete.
    let dry = run(&["cleanup"]);
    assert!(dry.status.success());
    assert!(dead.exists(), "dry run must not delete");
    assert!(String::from_utf8_lossy(&dry.stdout).contains("repo gone"));

    // --yes: deletes it.
    let del = run(&["cleanup", "--yes"]);
    assert!(del.status.success());
    assert!(!dead.exists(), "--yes should delete the dead folder");

    std::fs::remove_dir_all(&tmp).ok();
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

//! Cross-impl smoke: ensure Rust reads a Python-written `embeddings.npz`
//! and Python reads a Rust-written one. Skips when `python3` isn't on
//! PATH so the test stays optional.

use std::process::Command;
use tempfile::tempdir;

/// Locate a Python that has `numpy` installed. The repo's `.venv` is
/// preferred; system pythons are tried as fallback. Returns `None`
/// when no suitable interpreter is found, in which case the test
/// short-circuits with a `skipping` log instead of failing.
fn python() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Some(repo) = std::env::var_os("CARGO_MANIFEST_DIR") {
        // crates/lethe-core/Cargo.toml → repo root is two parents up.
        let root = std::path::Path::new(&repo)
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());
        if let Some(r) = root {
            candidates.push(r.join(".venv/bin/python").to_string_lossy().into_owned());
        }
    }
    candidates.extend(["python3".to_owned(), "python".to_owned()]);
    for cand in candidates {
        let ok = Command::new(&cand)
            .args(["-c", "import numpy"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(cand);
        }
    }
    None
}

#[test]
fn rust_reads_python_written_npz() {
    let Some(py) = python() else {
        eprintln!("skipping: no python on PATH");
        return;
    };
    let dir = tempdir().unwrap();
    let path = dir.path().join("embeddings.npz");

    let out = Command::new(&py)
        .arg("-c")
        .arg(format!(
            "import numpy as np\n\
             ids = np.array(['abc123', 'longer_id_xyz', 'short'])\n\
             embs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)\n\
             np.savez(r'{}', ids=ids, embeddings=embs)\n",
            path.display()
        ))
        .output()
        .expect("python runs");
    assert!(
        out.status.success(),
        "python failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let map = lethe_core::npz::load_embeddings(&path).expect("rust read");
    assert_eq!(map.len(), 3);
    assert_eq!(map["abc123"].to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(map["longer_id_xyz"].to_vec(), vec![4.0, 5.0, 6.0]);
    assert_eq!(map["short"].to_vec(), vec![7.0, 8.0, 9.0]);
}

#[test]
fn python_reads_rust_written_npz() {
    let Some(py) = python() else {
        eprintln!("skipping: no python on PATH");
        return;
    };
    let dir = tempdir().unwrap();
    let path = dir.path().join("embeddings.npz");

    let mut input = std::collections::HashMap::new();
    input.insert(
        "alpha".to_owned(),
        ndarray::Array1::from_vec(vec![1.0_f32, 2.0, 3.0]),
    );
    input.insert(
        "beta_with_longer_id".to_owned(),
        ndarray::Array1::from_vec(vec![4.0_f32, 5.0, 6.0]),
    );
    lethe_core::npz::save_embeddings(&path, &input).expect("rust write");

    let script = format!(
        "import numpy as np; \
         data = np.load(r'{}', allow_pickle=False); \
         ids = list(data['ids']); \
         embs = data['embeddings']; \
         print('ids:', ids); \
         print('shape:', embs.shape); \
         print('dtype:', embs.dtype); \
         [print(k + ':', list(v)) for k, v in sorted(zip(ids, embs))]",
        path.display()
    );
    let out = Command::new(&py)
        .arg("-c")
        .arg(&script)
        .output()
        .expect("python runs");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "python failed: stderr={stderr} stdout={stdout}"
    );
    // Accept both ['alpha', 'beta_...'] and any unicode dtype variant;
    // checking that python sees the same ids and shape is enough.
    assert!(stdout.contains("alpha:"), "python stdout: {stdout}");
    assert!(
        stdout.contains("beta_with_longer_id:"),
        "python stdout: {stdout}"
    );
    assert!(stdout.contains("shape: (2, 3)"), "python stdout: {stdout}");
    assert!(stdout.contains("dtype: float32"), "python stdout: {stdout}");
}

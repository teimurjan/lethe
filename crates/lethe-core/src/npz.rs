//! Read and write `.lethe/index/embeddings.npz` in numpy's npz format.
//!
//! The Python `MemoryStore` writes embeddings via
//! `np.savez(path, ids=np.array(ids), embeddings=embs)`. We mirror that
//! exact layout so the Python and Rust implementations share storage —
//! a Rust-written index opens cleanly under Python and vice versa.
//!
//! Layout:
//!   * `ids`        — 1-D `<UN` (UCS-4 fixed-width strings), length = N
//!   * `embeddings` — 2-D `<f4` (little-endian f32), shape `[N, dim]`

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use ndarray::Array1;
use npyz::npz::{NpzArchive, NpzWriter};
use npyz::zip::write::FileOptions;
use npyz::{DType, WriterBuilder};

use crate::Error;

/// Load `embeddings.npz` if present. Returns an empty map when the
/// file doesn't exist (e.g. a freshly created store).
pub fn load_embeddings(path: &Path) -> Result<HashMap<String, Array1<f32>>, Error> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let mut archive = NpzArchive::open(path).map_err(io_to_err)?;

    let ids = {
        let entry = archive
            .by_name("ids")
            .map_err(io_to_err)?
            .ok_or_else(|| invalid("npz missing `ids` array"))?;
        let v: Vec<String> = entry.into_vec().map_err(io_to_err)?;
        v
    };

    let (n, dim, flat) = {
        let entry = archive
            .by_name("embeddings")
            .map_err(io_to_err)?
            .ok_or_else(|| invalid("npz missing `embeddings` array"))?;
        let shape = entry.shape().to_vec();
        if shape.len() != 2 {
            return Err(invalid("embeddings array is not 2-D"));
        }
        let n = shape[0] as usize;
        let dim = shape[1] as usize;
        let flat: Vec<f32> = entry.into_vec().map_err(io_to_err)?;
        (n, dim, flat)
    };

    if ids.len() != n {
        return Err(invalid("ids length != embeddings rows"));
    }

    let mut out = HashMap::with_capacity(n);
    for (i, id) in ids.into_iter().enumerate() {
        let start = i * dim;
        let end = start + dim;
        out.insert(id, Array1::from_vec(flat[start..end].to_vec()));
    }
    Ok(out)
}

/// Write `embeddings.npz` from an `id → vector` map. Atomic via a
/// `.tmp` sibling rename so a crash mid-save can't truncate the file.
pub fn save_embeddings(path: &Path, embs: &HashMap<String, Array1<f32>>) -> Result<(), Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("npz.tmp");

    let ids: Vec<&str> = embs.keys().map(String::as_str).collect();
    let dim = embs
        .values()
        .next()
        .map(|v| v.len())
        .ok_or_else(|| invalid("save_embeddings called with empty map"))?;

    // `<UN` widths must accommodate the longest id. Round up to the
    // next multiple of 16 so we don't churn the dtype string on every
    // save (most ids are 12-char UUID prefixes).
    let max_chars = ids
        .iter()
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(16)
        .max(16);
    let width = max_chars.div_ceil(16) * 16;

    {
        let file = File::create(&tmp)?;
        let mut npz = NpzWriter::new(BufWriter::new(file));

        // ids array — `<UN` (UCS-4 fixed-width).
        let ids_dtype = parse_dtype(&format!("<U{width}"))?;
        let mut ids_writer = npz
            .array("ids", FileOptions::default())
            .map_err(io_to_err)?
            .dtype(ids_dtype)
            .shape(&[ids.len() as u64])
            .begin_nd()
            .map_err(io_to_err)?;
        for id in &ids {
            ids_writer.push(*id).map_err(io_to_err)?;
        }
        ids_writer.finish().map_err(io_to_err)?;

        // embeddings array — `<f4`, shape (N, dim), C-order.
        let f32_dtype = parse_dtype("<f4")?;
        let mut emb_writer = npz
            .array("embeddings", FileOptions::default())
            .map_err(io_to_err)?
            .dtype(f32_dtype)
            .shape(&[ids.len() as u64, dim as u64])
            .begin_nd()
            .map_err(io_to_err)?;
        for id in &ids {
            for v in embs[*id].iter() {
                emb_writer.push(v).map_err(io_to_err)?;
            }
        }
        emb_writer.finish().map_err(io_to_err)?;
        drop(npz); // close the zip stream
    }

    std::fs::rename(&tmp, path)?;
    Ok(())
}

// --------------------------------------------------------------- helpers

fn parse_dtype(s: &str) -> Result<DType, Error> {
    DType::parse(&format!("'{s}'")).map_err(|e| invalid(&e.to_string()))
}

fn invalid(msg: &str) -> Error {
    Error::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        msg.to_owned(),
    ))
}

fn io_to_err(e: impl std::fmt::Display) -> Error {
    Error::Io(std::io::Error::other(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trip_two_entries() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("embeddings.npz");

        let mut input = HashMap::new();
        input.insert("a".to_owned(), Array1::from_vec(vec![1.0_f32, 2.0, 3.0]));
        input.insert("bb".to_owned(), Array1::from_vec(vec![4.0_f32, 5.0, 6.0]));

        save_embeddings(&path, &input).unwrap();
        let out = load_embeddings(&path).unwrap();

        assert_eq!(out.len(), 2);
        assert_eq!(out["a"].to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(out["bb"].to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn missing_file_yields_empty_map() {
        let dir = tempdir().unwrap();
        let out = load_embeddings(&dir.path().join("absent.npz")).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn wide_ids_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("embeddings.npz");

        let id = "a".repeat(40); // > 16, < 64
        let mut input = HashMap::new();
        input.insert(id.clone(), Array1::from_vec(vec![0.5_f32; 4]));

        save_embeddings(&path, &input).unwrap();
        let out = load_embeddings(&path).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out.contains_key(&id));
    }
}

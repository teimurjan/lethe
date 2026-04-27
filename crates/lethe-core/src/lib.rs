//! Core retrieval library for `lethe`.
//!
//! See [the porting plan](../../../.claude/plans/functional-herding-boot.md)
//! for the contract this crate honors against the Python reference at
//! `legacy/lethe/`.

#![deny(rust_2018_idioms)]
#![warn(clippy::all)]

pub mod bm25;
pub mod db;
pub mod dedup;
pub mod encoders;
pub mod entry;
pub mod error;
pub mod faiss_flat;
pub mod kmeans;
pub mod lock;
pub mod markdown_store;
pub mod memory_store;
pub mod npz;
pub mod registry;
pub mod reranker;
pub mod rif;
pub mod rrf;
pub mod tokenize;
pub mod union_store;

pub use error::{Error, Result};

/// Crate version baked from Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the workspace this crate ships from. Used by version banners
/// in the CLI/TUI binaries.
#[must_use]
pub const fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_non_empty() {
        assert!(!version().is_empty());
    }
}

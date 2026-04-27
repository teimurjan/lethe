//! PyO3 bindings: import as `lethe_memory` after `maturin develop`.
//!
//! Surfaces the same shape as `lethe.MemoryStore` (Python) so you can
//! A/B compare the two implementations. The GIL is released around
//! all native work (encode + retrieve + save) via
//! `Python::allow_threads` so multi-threaded Python callers see real
//! parallelism on the Rust side.

#![allow(unsafe_code)] // pyo3 uses unsafe internally; binding crates opt in.
#![allow(missing_debug_implementations)]
// PyO3 derive types don't need it.
// `?` on `PyResult` triggers `useless_conversion` because the
// pyo3-derive expansion returns `PyResult<T>` from the closure. Both
// are `Result<T, PyErr>`, so the conversion is genuinely identity.
#![allow(clippy::useless_conversion)]

use std::sync::Arc;

use lethe_core::encoders::{BiEncoder as CoreBi, CrossEncoder as CoreCross};
use lethe_core::memory_store::{MemoryStore as CoreStore, StoreConfig};
use lethe_core::rif::RifConfig;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

#[pyclass(name = "RIFConfig")]
#[derive(Clone)]
struct PyRifConfig {
    inner: RifConfig,
}

#[pymethods]
impl PyRifConfig {
    #[new]
    #[pyo3(signature = (
        suppression_rate = 0.1,
        reinforcement_rate = 0.05,
        max_suppression = 1.0,
        decay_lambda = 0.005,
        alpha = 0.3,
        n_clusters = 0,
        use_rank_gap = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        suppression_rate: f32,
        reinforcement_rate: f32,
        max_suppression: f32,
        decay_lambda: f32,
        alpha: f32,
        n_clusters: u32,
        use_rank_gap: bool,
    ) -> Self {
        Self {
            inner: RifConfig {
                suppression_rate,
                reinforcement_rate,
                max_suppression,
                decay_lambda,
                alpha,
                n_clusters,
                use_rank_gap,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RIFConfig(alpha={}, n_clusters={}, use_rank_gap={})",
            self.inner.alpha, self.inner.n_clusters, self.inner.use_rank_gap
        )
    }
}

#[pyclass(name = "Hit")]
#[derive(Clone)]
struct PyHit {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    content: String,
    #[pyo3(get)]
    score: f32,
}

#[pyclass(name = "MemoryStore")]
struct PyMemoryStore {
    inner: CoreStore,
}

fn map_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pymethods]
impl PyMemoryStore {
    /// Open or create a store at `path`. `bi_encoder` and
    /// `cross_encoder` are HuggingFace repo names; pass `None` for the
    /// defaults (`Xenova/all-MiniLM-L6-v2` and
    /// `Xenova/ms-marco-MiniLM-L-6-v2`).
    #[new]
    #[pyo3(signature = (
        path,
        bi_encoder = None,
        cross_encoder = None,
        rif_config = None,
        k_shallow = 30,
        k_deep = 100,
        confidence_threshold = 4.0,
        dedup_threshold = 0.95,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        path: String,
        bi_encoder: Option<String>,
        cross_encoder: Option<String>,
        rif_config: Option<PyRifConfig>,
        k_shallow: usize,
        k_deep: usize,
        confidence_threshold: f32,
        dedup_threshold: f32,
    ) -> PyResult<Self> {
        py.allow_threads(|| {
            let bi = match bi_encoder {
                Some(name) => Some(Arc::new(CoreBi::from_repo(&name).map_err(map_err)?)),
                None => Some(Arc::new(
                    CoreBi::from_repo("Xenova/all-MiniLM-L6-v2").map_err(map_err)?,
                )),
            };
            let cross = match cross_encoder {
                Some(name) => Some(Arc::new(CoreCross::from_repo(&name).map_err(map_err)?)),
                None => Some(Arc::new(
                    CoreCross::from_repo("Xenova/ms-marco-MiniLM-L-6-v2").map_err(map_err)?,
                )),
            };
            let dim = bi.as_ref().map_or(384, |b| b.dim());
            let cfg = StoreConfig {
                dim,
                k_shallow,
                k_deep,
                confidence_threshold,
                dedup_threshold,
                rif: rif_config.map_or_else(RifConfig::default, |c| c.inner),
            };
            let store = CoreStore::open(&path, bi, cross, cfg).map_err(map_err)?;
            Ok(Self { inner: store })
        })
    }

    /// Add a memory entry. Returns the id, or `None` when deduped.
    #[pyo3(signature = (content, entry_id = None, session_id = "", turn_idx = 0))]
    fn add(
        &self,
        py: Python<'_>,
        content: String,
        entry_id: Option<String>,
        session_id: &str,
        turn_idx: i64,
    ) -> PyResult<Option<String>> {
        py.allow_threads(|| {
            self.inner
                .add(&content, entry_id.as_deref(), session_id, turn_idx)
                .map_err(map_err)
        })
    }

    /// Retrieve top-k. Returns a list of `Hit` (id, content, score).
    #[pyo3(signature = (query, k = 5))]
    fn retrieve(&self, py: Python<'_>, query: String, k: usize) -> PyResult<Vec<PyHit>> {
        py.allow_threads(|| {
            let hits = self.inner.retrieve(&query, k).map_err(map_err)?;
            Ok(hits
                .into_iter()
                .map(|h| PyHit {
                    id: h.id,
                    content: h.content,
                    score: h.score,
                })
                .collect())
        })
    }

    fn save(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.save().map_err(map_err))
    }

    fn delete(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        py.allow_threads(|| self.inner.delete(&id).map_err(map_err))
    }

    fn size(&self) -> usize {
        self.inner.size()
    }
}

/// Module entry — register the classes.
#[pymodule]
fn lethe_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", lethe_core::version())?;
    m.add_class::<PyRifConfig>()?;
    m.add_class::<PyHit>()?;
    m.add_class::<PyMemoryStore>()?;
    Ok(())
}

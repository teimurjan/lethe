//! BM25 build + score benchmarks at 1k / 10k / 100k corpus sizes.
//!
//! Useful for tracking the linear-with-N cost that dominates retrieve
//! latency on power-user corpora. The Python `BM25Okapi.get_scores`
//! reference at N=200k is ~1.5–3 s; this bench is the apples-to-apples
//! Rust number.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lethe_core::bm25::BM25Okapi;
use lethe_core::tokenize::tokenize_bm25;

fn synthetic_corpus(n: usize) -> Vec<Vec<String>> {
    (0..n)
        .map(|i| {
            tokenize_bm25(&format!(
                "entry {i} alpha beta gamma session_{} turn_{}",
                i % 17,
                i % 31
            ))
        })
        .collect()
}

fn bench_build(c: &mut Criterion) {
    let mut g = c.benchmark_group("bm25_build");
    for &n in &[1_000_usize, 10_000, 50_000] {
        let docs = synthetic_corpus(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &docs, |b, docs| {
            b.iter(|| BM25Okapi::new(docs));
        });
    }
    g.finish();
}

fn bench_score(c: &mut Criterion) {
    let mut g = c.benchmark_group("bm25_score");
    for &n in &[1_000_usize, 10_000, 50_000] {
        let docs = synthetic_corpus(n);
        let bm = BM25Okapi::new(&docs);
        let q = tokenize_bm25("alpha session_3 turn_5");
        g.bench_with_input(BenchmarkId::from_parameter(n), &(bm, q), |b, (bm, q)| {
            b.iter(|| bm.get_scores(q));
        });
    }
    g.finish();
}

criterion_group!(benches, bench_build, bench_score);
criterion_main!(benches);

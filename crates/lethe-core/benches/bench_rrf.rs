//! RRF merge of two ~100-id lists. The Rust path is purely a HashMap
//! sum; this just confirms it's noise vs. the rest of the pipeline.

use criterion::{criterion_group, criterion_main, Criterion};
use lethe_core::rrf::rrf_merge;

fn bench_merge(c: &mut Criterion) {
    let l1: Vec<String> = (0..100).map(|i| format!("a{i}")).collect();
    let l2: Vec<String> = (50..150).map(|i| format!("a{i}")).collect();
    c.bench_function("rrf_merge_100x100", |b| {
        b.iter(|| rrf_merge(&[l1.clone(), l2.clone()]));
    });
}

criterion_group!(benches, bench_merge);
criterion_main!(benches);

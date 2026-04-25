//! Dense top-k benchmark on `(N, dim)` IndexFlatIP-equivalent.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use lethe_core::faiss_flat::FlatIp;
use ndarray::Array2;

fn random_unit_rows(n: usize, dim: usize) -> Array2<f32> {
    let mut rows = Array2::<f32>::zeros((n, dim));
    let mut state = 0xC0FFEE_u64;
    for v in rows.iter_mut() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *v = ((state >> 32) as u32 as f32 / u32::MAX as f32) - 0.5;
    }
    for mut row in rows.axis_iter_mut(ndarray::Axis(0)) {
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            row.mapv_inplace(|x| x / n);
        }
    }
    rows
}

fn bench_search(c: &mut Criterion) {
    let dim = 384;
    let mut g = c.benchmark_group("flat_ip_search");
    for &n in &[1_000_usize, 10_000, 100_000] {
        let rows = random_unit_rows(n, dim);
        let mut idx = FlatIp::new(dim);
        idx.add_batch(rows.view()).unwrap();
        let query = rows.row(0).to_owned();
        g.bench_with_input(
            BenchmarkId::from_parameter(n),
            &(idx, query),
            |b, (i, q)| {
                b.iter(|| i.search(q.view(), 30).unwrap());
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);

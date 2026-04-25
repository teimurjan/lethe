//! Single RIF update event on a 30-candidate pool. Mirrors the cost
//! paid per query in retrieve()'s RIF-update step.

use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::{HashMap, HashSet};

use lethe_core::rif::{update_suppression, CompetitorRow, RifConfig};

fn bench_update(c: &mut Criterion) {
    let cfg = RifConfig {
        use_rank_gap: true,
        ..RifConfig::default()
    };
    let competitors: Vec<CompetitorRow> = (0..30)
        .map(|i| CompetitorRow {
            eid: format!("e{i}"),
            initial_rank: i,
            xenc_rank: 29 - i,
            xenc_score: -1.0,
        })
        .collect();
    let suppression: HashMap<String, f32> = HashMap::new();
    let last_updated: HashMap<String, i64> = HashMap::new();
    let mut winners: HashSet<String> = HashSet::new();
    winners.insert("e0".to_owned());
    winners.insert("e1".to_owned());
    c.bench_function("rif_update_30_candidates", |b| {
        b.iter(|| {
            update_suppression(
                &winners,
                &competitors,
                &suppression,
                30,
                &cfg,
                100,
                &last_updated,
            )
        });
    });
}

criterion_group!(benches, bench_update);
criterion_main!(benches);

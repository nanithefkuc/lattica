//! Scalar-versus-dispatched real-vector transform measurements.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lattica::kernel::{internals, transform_batch_soa};

fn transform_benchmark(c: &mut Criterion) {
    const DIMENSION: usize = 16;
    let mut group = c.benchmark_group("real_transform_batch_soa_16");
    let matrix: Vec<f64> = (0..DIMENSION * DIMENSION)
        .map(|i| (f64::from(u32::try_from(i % 97).unwrap()) - 48.0) / 64.0)
        .collect();
    for vectors in [8usize, 64, 257] {
        let inputs: Vec<f64> = (0..DIMENSION * vectors)
            .map(|i| (f64::from(u32::try_from(i % 113).unwrap()) - 56.0) / 32.0)
            .collect();
        let mut outputs = vec![0.0; DIMENSION * vectors];
        group.throughput(Throughput::Elements(
            u64::try_from(DIMENSION * DIMENSION * vectors).unwrap(),
        ));
        group.bench_with_input(BenchmarkId::new("scalar", vectors), &vectors, |b, _| {
            b.iter(|| {
                internals::transform_batch_soa_scalar(
                    &matrix,
                    DIMENSION,
                    vectors,
                    &inputs,
                    &mut outputs,
                );
            });
        });
        group.bench_with_input(BenchmarkId::new("dispatched", vectors), &vectors, |b, _| {
            b.iter(|| {
                transform_batch_soa(
                    &matrix,
                    DIMENSION,
                    DIMENSION,
                    vectors,
                    &inputs,
                    &mut outputs,
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, transform_benchmark);
criterion_main!(benches);

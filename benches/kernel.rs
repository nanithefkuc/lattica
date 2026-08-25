//! Scalar-versus-dispatched real-vector transform measurements.
//!
//! The 16-output group reproduces the dispatched shape's crossover evidence.
//! The 24-output groups separate coordinate arithmetic from layout work: the
//! arithmetic groups time transforms over structure-of-arrays batches
//! directly, the conversion case times an array-of-structures transpose
//! alone, and the pipeline group adds that transpose to the fastest
//! candidates so an array-of-structures consumer can judge its end-to-end
//! position.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lattica::kernel::{internals, transform_batch, transform_batch_soa};
use std::hint::black_box;

/// Transposes an array-of-structures batch into structure-of-arrays planes.
fn aos_to_soa(aos: &[f64], rows: usize, vectors: usize) -> Vec<f64> {
    let mut soa = vec![0.0; rows * vectors];
    for vector in 0..vectors {
        for row in 0..rows {
            soa[row * vectors + vector] = aos[vector * rows + row];
        }
    }
    soa
}

fn matrix(rows: usize) -> Vec<f64> {
    (0..rows * rows)
        .map(|i| (f64::from(u32::try_from(i % 97).unwrap()) - 48.0) / 64.0)
        .collect()
}

fn aos_inputs(rows: usize, vectors: usize) -> Vec<f64> {
    (0..rows * vectors)
        .map(|i| (f64::from(u32::try_from(i % 113).unwrap()) - 56.0) / 32.0)
        .collect()
}

fn soa_inputs(rows: usize, vectors: usize) -> Vec<f64> {
    let flat = aos_inputs(rows, vectors);
    aos_to_soa(&flat, rows, vectors)
}

fn elements(rows: usize, vectors: usize) -> u64 {
    u64::try_from(rows * rows * vectors).unwrap()
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod dispatched {
    use lattica::kernel::internals;

    pub type Kernel = fn(archmage::X64V3Token, &[f64], usize, &[f64], &mut [f64]);

    pub fn token() -> archmage::X64V3Token {
        use archmage::SimdToken;
        archmage::X64V3Token::summon().expect("dispatch benchmarks require an x86 v3 host")
    }

    pub fn generic(
        token: archmage::X64V3Token,
        matrix: &[f64],
        vectors: usize,
        inputs: &[f64],
        outputs: &mut [f64],
    ) {
        internals::transform_batch_soa_avx2_generic(token, matrix, 24, vectors, inputs, outputs);
    }

    /// Candidate kernels over the exact 24-by-24 shape, including the two
    /// rejected block sizes retained as measurement evidence.
    pub fn twentyfour() -> Vec<(&'static str, Kernel)> {
        vec![
            ("avx2_generic", generic),
            (
                "avx2_fixed_block6",
                internals::transform_batch_soa_fixed_24_block6,
            ),
            (
                "avx2_fixed_block8",
                internals::transform_batch_soa_fixed_24_block8,
            ),
            (
                "avx2_fixed_block12",
                internals::transform_batch_soa_fixed_24_block12,
            ),
        ]
    }
}

fn sixteen_output_benchmark(c: &mut Criterion) {
    const DIMENSION: usize = 16;
    let mut group = c.benchmark_group("real_transform_batch_soa_16");
    let matrix = matrix(DIMENSION);
    for vectors in [8usize, 64, 257] {
        let inputs = soa_inputs(DIMENSION, vectors);
        let mut outputs = vec![0.0; DIMENSION * vectors];
        group.throughput(Throughput::Elements(elements(DIMENSION, vectors)));
        group.bench_with_input(BenchmarkId::new("scalar", vectors), &vectors, |b, _| {
            b.iter(|| {
                internals::transform_batch_soa_scalar(
                    &matrix,
                    DIMENSION,
                    vectors,
                    &inputs,
                    &mut outputs,
                );
                black_box(&outputs);
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
                black_box(&outputs);
            });
        });
    }
    group.finish();
}

fn twentyfour_output_arithmetic_benchmark(c: &mut Criterion) {
    const DIMENSION: usize = 24;
    let mut group = c.benchmark_group("real_transform_soa_24");
    let matrix = matrix(DIMENSION);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let token = dispatched::token();
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let kernels = dispatched::twentyfour();

    for vectors in [1usize, 4, 8, 16, 32, 64, 128, 257] {
        let inputs = soa_inputs(DIMENSION, vectors);
        let mut outputs = vec![0.0; DIMENSION * vectors];
        group.throughput(Throughput::Elements(elements(DIMENSION, vectors)));
        group.bench_with_input(BenchmarkId::new("scalar", vectors), &vectors, |b, _| {
            b.iter(|| {
                internals::transform_batch_soa_scalar(
                    &matrix,
                    DIMENSION,
                    vectors,
                    &inputs,
                    &mut outputs,
                );
                black_box(&outputs);
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
                black_box(&outputs);
            });
        });

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        for (name, kernel) in &kernels {
            group.bench_with_input(BenchmarkId::new(*name, vectors), &vectors, |b, _| {
                b.iter(|| {
                    kernel(token, &matrix, vectors, &inputs, &mut outputs);
                    black_box(&outputs);
                });
            });
        }
    }
    group.finish();
}

fn twentyfour_output_pipeline_benchmark(c: &mut Criterion) {
    const DIMENSION: usize = 24;
    let mut group = c.benchmark_group("real_transform_pipeline_24");
    let matrix = matrix(DIMENSION);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    let token = dispatched::token();

    for vectors in [8usize, 32, 64, 128, 257] {
        let flat = aos_inputs(DIMENSION, vectors);
        let mut outputs = vec![0.0; DIMENSION * vectors];
        group.throughput(Throughput::Elements(elements(DIMENSION, vectors)));
        group.bench_with_input(BenchmarkId::new("aos_scalar", vectors), &vectors, |b, _| {
            b.iter(|| {
                transform_batch(&matrix, DIMENSION, DIMENSION, &flat, &mut outputs).unwrap();
                black_box(&outputs);
            });
        });
        group.bench_with_input(
            BenchmarkId::new("convert_only", vectors),
            &vectors,
            |b, _| {
                b.iter(|| aos_to_soa(&flat, DIMENSION, vectors));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("convert_then_scalar", vectors),
            &vectors,
            |b, _| {
                b.iter(|| {
                    let inputs = aos_to_soa(&flat, DIMENSION, vectors);
                    internals::transform_batch_soa_scalar(
                        &matrix,
                        DIMENSION,
                        vectors,
                        &inputs,
                        &mut outputs,
                    );
                    black_box(&outputs);
                });
            },
        );

        #[cfg(all(feature = "simd", target_arch = "x86_64"))]
        for name in ["convert_then_avx2_generic", "convert_then_avx2_fixed"] {
            group.bench_with_input(BenchmarkId::new(name, vectors), &vectors, |b, _| {
                b.iter(|| {
                    let inputs = aos_to_soa(&flat, DIMENSION, vectors);
                    if name.ends_with("generic") {
                        dispatched::generic(token, &matrix, vectors, &inputs, &mut outputs);
                    } else {
                        internals::transform_batch_soa_fixed_24_block12(
                            token,
                            &matrix,
                            vectors,
                            &inputs,
                            &mut outputs,
                        );
                    }
                    black_box(&outputs);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    sixteen_output_benchmark,
    twentyfour_output_arithmetic_benchmark,
    twentyfour_output_pipeline_benchmark,
);
criterion_main!(benches);

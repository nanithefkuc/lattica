//! Workload-separated optimization measurements and correctness fingerprints.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lattica::Basis;
use lattica::int::{IntMatrix, adjugate, hnf, hnf_mod_det, invariant_factors};
use lattica::reduce::{Delta, is_reduced, lll, lll_profiled};

const DIMENSIONS: [usize; 3] = [8, 16, 24];
const SAMPLES: usize = 11;

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measured<R>(mut body: impl FnMut() -> R) -> Duration {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        black_box(body());
        samples.push(start.elapsed());
    }
    median(samples)
}

fn skew_basis(dimension: usize, case: usize, shear_bits: u32) -> Vec<i128> {
    let mut basis = vec![0i128; dimension * dimension];
    for row in 0..dimension {
        basis[row * dimension + row] = if row % 3 == 0 { 2 } else { 1 };
        if row > 0 {
            basis[row * dimension + row - 1] = if row % 2 == 0 { -1 } else { 1 };
        }
    }
    let mut rng = Rng(0x4C41_5454_4943_4100 ^ u64::try_from(case).unwrap());
    let mask = (1u64 << shear_bits.min(12)) - 1;
    for _ in 0..dimension.min(6) {
        let target = usize::try_from(rng.next()).unwrap() % dimension;
        let mut source = usize::try_from(rng.next()).unwrap() % dimension;
        if source == target {
            source = (source + 1) % dimension;
        }
        let magnitude = i128::from((rng.next() & mask).max(1));
        let factor = if rng.next() & 1 == 0 {
            magnitude
        } else {
            -magnitude
        };
        let source_row = basis[source * dimension..(source + 1) * dimension].to_vec();
        for column in 0..dimension {
            basis[target * dimension + column] += factor * source_row[column];
        }
    }
    basis
}

fn benchmark_lll() {
    println!("metric,dimension,geometry,value,fingerprint");
    for dimension in DIMENSIONS {
        for shear_bits in [2, 4, 6] {
            let bases: Vec<_> = (0..16)
                .map(|case| {
                    let rows = skew_basis(dimension, case, shear_bits);
                    Basis::from_rows(dimension, dimension, &rows)
                        .unwrap()
                        .gram()
                        .unwrap()
                })
                .collect();
            let fingerprint: i128 = bases
                .iter()
                .enumerate()
                .map(|(index, gram)| i128::try_from(index + 1).unwrap() * gram.det().unwrap())
                .sum();

            let elapsed = measured(|| {
                for gram in &bases {
                    black_box(lll(black_box(gram), Delta::STRONG).unwrap());
                }
            });
            let (_, stats) = lll_profiled(&bases[0], Delta::STRONG).unwrap();
            let certificate = measured(|| {
                for gram in &bases {
                    let reduced = lll(gram, Delta::STRONG).unwrap();
                    black_box(is_reduced(&reduced.gram, Delta::STRONG).unwrap());
                    black_box(reduced.gram.det().unwrap());
                    black_box(reduced.transform.det().unwrap());
                }
            });
            let basis_count = f64::from(u32::try_from(bases.len()).unwrap());
            let per_basis = elapsed.as_secs_f64() * 1e9 / basis_count;
            let certificate_ns = certificate.as_secs_f64() * 1e9 / basis_count;
            println!("lll_ns,{dimension},shear_{shear_bits},{per_basis:.2},{fingerprint}");
            println!(
                "lll_certificate_ns,{dimension},shear_{shear_bits},{certificate_ns:.2},{fingerprint}"
            );
            println!(
                "lll_factorizations,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.factorizations
            );
            println!(
                "lll_size_reductions,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.size_reductions
            );
            println!(
                "lll_size_reduction_checks,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.size_reduction_checks
            );
            println!(
                "lll_zero_quotients,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.zero_quotients
            );
            println!(
                "lll_quotient_divisions,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.quotient_divisions
            );
            println!(
                "lll_swaps,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.swaps
            );
            println!(
                "lll_swap_update_terms,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.swap_update_terms
            );
            println!(
                "lll_iterations,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.iterations
            );
            println!(
                "lll_gram_copies,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.gram_copies
            );
            println!(
                "lll_checked_updates,{dimension},shear_{shear_bits},{},{fingerprint}",
                stats.checked_updates
            );
        }
    }
}

fn algebra_matrix(dimension: usize) -> IntMatrix<i128> {
    let mut data = vec![0i128; dimension * dimension];
    for row in 0..dimension {
        data[row * dimension + row] = 1;
        if row > 0 {
            data[row * dimension + row - 1] = if row % 2 == 0 { -1 } else { 1 };
        }
    }
    IntMatrix::from_rows(dimension, dimension, &data).unwrap()
}

fn benchmark_algebra() {
    for dimension in DIMENSIONS {
        let matrix = algebra_matrix(dimension);
        let gram = Basis::from_rows(dimension, dimension, matrix.as_slice())
            .unwrap()
            .gram()
            .unwrap();
        let fingerprint = gram.det().unwrap();
        let det_time = measured(|| black_box(matrix.det().unwrap()));
        let pd_time = measured(|| black_box(gram.is_positive_definite().unwrap()));
        let adj_time = measured(|| black_box(adjugate(&matrix).unwrap()));
        let hnf_time = measured(|| black_box(hnf(&matrix).unwrap()));
        let hnf_mod_time = measured(|| black_box(hnf_mod_det(&matrix).unwrap()));
        let snf_time = measured(|| black_box(invariant_factors(&matrix).unwrap()));
        for (name, duration) in [
            ("det", det_time),
            ("positive_definite", pd_time),
            ("adjugate", adj_time),
            ("hnf", hnf_time),
            ("hnf_mod_det", hnf_mod_time),
            ("snf", snf_time),
        ] {
            println!(
                "algebra_ns,{dimension},{name},{:.2},{fingerprint}",
                duration.as_secs_f64() * 1e9
            );
        }
    }
}

fn main() {
    benchmark_lll();
    benchmark_algebra();
}

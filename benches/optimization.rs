//! Workload-separated optimization measurements and correctness fingerprints.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lattica::Basis;
use lattica::int::{IntMatrix, adjugate, hnf, hnf_mod_det, invariant_factors};
use lattica::quant::{
    AmbientScratch, An, BarnesWall16, Dn, DnPlus, EnumerationScratch, Enumerator, Leech24,
    Quantizer, Scratch, Zn, e8 as e8_quantizer, nearest_batch,
};
use lattica::reduce::{Delta, is_reduced, lll_profiled};

const DIMENSIONS: [usize; 3] = [8, 16, 24];
const SAMPLES: usize = 11;
const NODE_BUDGET: u64 = 1 << 28;

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
        let factor = if rng.next() & 1 == 0 { magnitude } else { -magnitude };
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
                .map(|(index, gram)| {
                    i128::try_from(index + 1).unwrap() * gram.det().unwrap()
                })
                .sum();

            let elapsed = measured(|| {
                for gram in &bases {
                    black_box(lll_profiled(black_box(gram), Delta::STRONG).unwrap());
                }
            });
            let (_, stats) = lll_profiled(&bases[0], Delta::STRONG).unwrap();
            let certificate = measured(|| {
                for gram in &bases {
                    let (reduced, _) = lll_profiled(gram, Delta::STRONG).unwrap();
                    black_box(is_reduced(&reduced.gram, Delta::STRONG).unwrap());
                    black_box(reduced.gram.det().unwrap());
                    black_box(reduced.transform.det().unwrap());
                }
            });
            let per_basis = elapsed.as_secs_f64() * 1e9 / bases.len() as f64;
            let certificate_ns = certificate.as_secs_f64() * 1e9 / bases.len() as f64;
            println!("lll_ns,{dimension},shear_{shear_bits},{per_basis:.2},{fingerprint}");
            println!("lll_certificate_ns,{dimension},shear_{shear_bits},{certificate_ns:.2},{fingerprint}");
            println!("lll_factorizations,{dimension},shear_{shear_bits},{},{fingerprint}", stats.factorizations);
            println!("lll_size_reductions,{dimension},shear_{shear_bits},{},{fingerprint}", stats.size_reductions);
            println!("lll_swaps,{dimension},shear_{shear_bits},{},{fingerprint}", stats.swaps);
            println!("lll_iterations,{dimension},shear_{shear_bits},{},{fingerprint}", stats.iterations);
            println!("lll_gram_copies,{dimension},shear_{shear_bits},{},{fingerprint}", stats.gram_copies);
            println!("lll_checked_updates,{dimension},shear_{shear_bits},{},{fingerprint}", stats.checked_updates);
        }
    }
}

fn cvp_gram(dimension: usize) -> lattica::Gram<i128> {
    let mut rows = vec![0i128; dimension * dimension];
    for row in 0..dimension {
        rows[row * dimension + row] = 2;
        if row + 1 < dimension {
            rows[row * dimension + row + 1] = 1;
        }
    }
    Basis::from_rows(dimension, dimension, &rows)
        .unwrap()
        .gram()
        .unwrap()
}

fn benchmark_cvp() {
    for dimension in DIMENSIONS {
        let gram = cvp_gram(dimension);
        let prepare = measured(|| black_box(Enumerator::new(black_box(&gram)).unwrap()));
        println!("cvp_prepare_ns,{dimension},single,{:.2},{}", prepare.as_secs_f64() * 1e9, gram.det().unwrap());
        let enumerator = Enumerator::new(&gram).unwrap();
        let mut scratch = EnumerationScratch::new();
        let mut out = vec![0i64; dimension];
        for (class, offset) in [("easy", 0.0625), ("median", 0.4375), ("boundary", 0.5)] {
            let target: Vec<f64> = (0..dimension)
                .map(|index| if index % 2 == 0 { offset } else { -offset })
                .collect();
            let expected = enumerator
                .nearest_ml(&target, &mut out, NODE_BUDGET, &mut scratch)
                .unwrap();
            let fingerprint: i128 = out
                .iter()
                .enumerate()
                .map(|(index, value)| i128::try_from(index + 1).unwrap() * i128::from(*value))
                .sum();
            let elapsed = measured(|| {
                black_box(
                    enumerator
                        .nearest_ml(
                            black_box(&target),
                            black_box(&mut out),
                            NODE_BUDGET,
                            black_box(&mut scratch),
                        )
                        .unwrap(),
                );
            });
            let ns = elapsed.as_secs_f64() * 1e9;
            println!("cvp_nodes,{dimension},{class},{expected},{fingerprint}");
            println!("cvp_warm_ns,{dimension},{class},{ns:.2},{fingerprint}");
            println!("cvp_ns_per_node,{dimension},{class},{:.2},{fingerprint}", ns / expected.max(1) as f64);
            println!("cvp_max_depth,{dimension},{class},{dimension},{fingerprint}");
        }
    }
}

fn benchmark_named() {
    let bw_setup = measured(|| black_box(BarnesWall16::new().unwrap()));
    let leech_setup = measured(|| black_box(Leech24::new().unwrap()));
    println!("named_setup_ns,16,bw16,{:.2},16", bw_setup.as_secs_f64() * 1e9);
    println!("named_setup_ns,24,leech,{:.2},24", leech_setup.as_secs_f64() * 1e9);

    let bw = BarnesWall16::new().unwrap();
    let leech = Leech24::new().unwrap();
    let mut scratch = AmbientScratch::new();
    let mut out16 = [0i64; 16];
    let mut out24 = [0i64; 24];
    let ambient16 = [0.31; 16];
    let ambient24 = [0.31; 24];
    let total16 = measured(|| {
        black_box(bw.nearest(&ambient16, &mut out16, NODE_BUDGET, &mut scratch).unwrap());
    });
    let total24 = measured(|| {
        black_box(leech.nearest(&ambient24, &mut out24, NODE_BUDGET, &mut scratch).unwrap());
    });
    println!("named_total_ns,16,bw16,{:.2},{}", total16.as_secs_f64() * 1e9, out16.iter().sum::<i64>());
    println!("named_total_ns,24,leech,{:.2},{}", total24.as_secs_f64() * 1e9, out24.iter().sum::<i64>());
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
            println!("algebra_ns,{dimension},{name},{:.2},{fingerprint}", duration.as_secs_f64() * 1e9);
        }
    }
}

fn benchmark_quantizers() {
    let quantizers: Vec<(&str, Box<dyn Quantizer>)> = vec![
        ("zn24", Box::new(Zn::new(24).unwrap())),
        ("dn24", Box::new(Dn::new(24).unwrap())),
        ("an23", Box::new(An::new(23).unwrap())),
        ("dnplus24", Box::new(DnPlus::new(24).unwrap())),
        ("e8", Box::new(e8_quantizer())),
    ];
    for (name, quantizer) in quantizers {
        for vectors in [1usize, 8, 64, 257] {
            let dimension = quantizer.dim();
            let input: Vec<f64> = (0..dimension * vectors)
                .map(|index| (index % 31) as f64 / 16.0 - 0.9375)
                .collect();
            let mut output = vec![0i64; input.len()];
            let mut scratch = Scratch::new(dimension);
            let duration = measured(|| {
                nearest_batch(
                    quantizer.as_ref(),
                    black_box(&input),
                    black_box(&mut output),
                    black_box(&mut scratch),
                )
                .unwrap();
            });
            let fingerprint: i128 = output.iter().map(|value| i128::from(*value)).sum();
            println!("quantizer_batch_ns,{dimension},{name}_{vectors},{:.2},{fingerprint}", duration.as_secs_f64() * 1e9);
        }
    }
}

fn main() {
    benchmark_lll();
    benchmark_cvp();
    benchmark_named();
    benchmark_algebra();
    benchmark_quantizers();
}

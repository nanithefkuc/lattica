//! In-process comparison harness for operations shared with fplll.
//!
//! LLL reduction only; the closest-vector comparison lives in `lattice-engine`
//! with the decoders. The fplll side of the comparison is
//! `fplll_compare.cpp` in this directory; build and run it separately at a
//! pinned fplll version. This harness prints the same CSV shape so the two
//! outputs can be diffed directly.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lattica::basis::Basis;
use lattica::reduce::{Delta, lll};

const DIMENSIONS: [usize; 3] = [8, 16, 24];
const LLL_CASES: usize = 16;
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

    fn index(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }
}

fn canonical_basis(dimension: usize) -> Vec<i128> {
    let mut basis = vec![0i128; dimension * dimension];
    for row in 0..dimension {
        basis[row * dimension + row] = 2;
        if row > 0 {
            basis[row * dimension + row - 1] = 1;
        }
        if row > 1 {
            basis[row * dimension + row - 2] = -1;
        }
    }
    basis
}

fn skew_basis(dimension: usize, case: usize) -> Vec<i128> {
    let mut basis = canonical_basis(dimension);
    let mut rng = Rng(0xd1b5_4a32_d192_ed03
        ^ u64::try_from(dimension).unwrap()
        ^ (u64::try_from(case).unwrap() << 32));
    for _ in 0..2 * dimension {
        let destination = rng.index(dimension);
        let mut source = rng.index(dimension - 1);
        if source >= destination {
            source += 1;
        }
        let sign = if rng.next() & 1 == 0 { -1 } else { 1 };
        let destination_start = destination * dimension;
        let source_start = source * dimension;
        let acceptable = (0..dimension).all(|column| {
            let candidate = basis[destination_start + column] + sign * basis[source_start + column];
            candidate.abs() <= 256
        });
        if acceptable {
            for column in 0..dimension {
                basis[destination_start + column] += sign * basis[source_start + column];
            }
        }
    }
    basis
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn per_operation(duration: Duration, operations: usize) -> f64 {
    duration.as_secs_f64() * 1e9 / f64::from(u32::try_from(operations).unwrap())
}

fn basis_checksum(bases: &[Vec<i128>]) -> i128 {
    bases
        .iter()
        .flat_map(|basis| basis.iter())
        .enumerate()
        .map(|(index, &entry)| i128::try_from(index + 1).unwrap() * entry)
        .sum()
}

fn benchmark_lll(dimension: usize) -> Result<(f64, i128), Box<dyn std::error::Error>> {
    let bases: Vec<Vec<i128>> = (0..LLL_CASES)
        .map(|case| skew_basis(dimension, case))
        .collect();
    let grams = bases
        .iter()
        .map(|rows| Basis::from_rows(dimension, dimension, rows)?.gram())
        .collect::<Result<Vec<_>, _>>()?;

    for gram in &grams {
        black_box(lll(gram, Delta::STRONG)?);
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        for gram in &grams {
            let reduced = lll(gram, Delta::STRONG)?;
            black_box(reduced.transform.as_slice()[0]);
        }
        samples.push(start.elapsed());
    }
    Ok((
        per_operation(median(samples), LLL_CASES),
        basis_checksum(&bases),
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("library,operation,dimension,median_ns,fingerprint");
    for dimension in DIMENSIONS {
        let (nanoseconds, fingerprint) = benchmark_lll(dimension)?;
        println!("lattica,lll,{dimension},{nanoseconds:.2},{fingerprint}");
    }
    Ok(())
}

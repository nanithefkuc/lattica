//! In-process comparison harness for operations shared with fplll.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lattica::basis::Basis;
use lattica::quant::{EnumerationScratch, Enumerator};
use lattica::reduce::{Delta, is_reduced, lll};

const DIMENSIONS: [usize; 3] = [8, 16, 24];
const LLL_CASES: usize = 16;
const TARGETS: usize = 128;
const SAMPLES: usize = 11;
const NODE_BUDGET: u64 = 1 << 30;
const CVP_SCALE: i128 = 1_009;

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

    fn signed(&mut self, bound: i64) -> i64 {
        let width = u64::try_from(2 * bound + 1).unwrap();
        i64::try_from(self.next() % width).unwrap() - bound
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

fn cvp_basis(dimension: usize) -> Vec<i128> {
    let mut basis = vec![0i128; dimension * dimension];
    for row in 0..dimension {
        basis[row * dimension + row] = 2;
        if row + 1 < dimension {
            basis[row * dimension + row + 1] = 1;
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

fn targets(basis: &[i128], dimension: usize) -> Vec<Vec<i128>> {
    let mut rng = Rng(0xa076_1d64_78bd_642f ^ u64::try_from(dimension).unwrap());
    let mut targets = Vec::with_capacity(TARGETS);
    for _ in 0..TARGETS {
        let lattice_coordinates: Vec<i64> = (0..dimension).map(|_| rng.signed(8)).collect();
        let mut target = vec![0i128; dimension];
        for (row, &coordinate) in lattice_coordinates.iter().enumerate() {
            for column in 0..dimension {
                target[column] +=
                    i128::from(coordinate) * basis[row * dimension + column] * CVP_SCALE;
            }
        }
        for entry in &mut target {
            *entry += i128::from(rng.signed(1_000));
        }
        targets.push(target);
    }
    targets
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn per_operation(duration: Duration, operations: usize) -> f64 {
    duration.as_secs_f64() * 1e9 / f64::from(u32::try_from(operations).unwrap())
}

fn bounded_f64(value: i128) -> f64 {
    f64::from(i32::try_from(value).unwrap())
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

struct CvpCase {
    gram: lattica::Gram<i128>,
    reduced_basis: Basis<i128>,
    ambient_targets: Vec<Vec<i128>>,
    target_coordinates: Vec<Vec<f64>>,
}

fn prepare_cvp(dimension: usize) -> Result<CvpCase, Box<dyn std::error::Error>> {
    let rows = cvp_basis(dimension);
    let basis = Basis::from_rows(dimension, dimension, &rows)?;
    let gram = basis.gram()?;
    assert!(is_reduced(&gram, Delta::STRONG)?);
    let ambient_targets = targets(&rows, dimension);
    let mut target_coordinates = Vec::with_capacity(TARGETS);
    for target in &ambient_targets {
        let mut coordinate = vec![0.0; dimension];
        for column in 0..dimension {
            let mut value = bounded_f64(target[column]) / bounded_f64(CVP_SCALE);
            for (row, &coefficient) in coordinate.iter().take(column).enumerate() {
                value -= coefficient * bounded_f64(basis.as_matrix().get(row, column));
            }
            coordinate[column] = value / bounded_f64(basis.as_matrix().get(column, column));
        }
        target_coordinates.push(coordinate);
    }
    Ok(CvpCase {
        gram,
        reduced_basis: basis,
        ambient_targets,
        target_coordinates,
    })
}

fn cvp_fingerprints(case: &CvpCase, outputs: &[Vec<i64>]) -> (i128, i128, i128) {
    let mut target_checksum = 0i128;
    let mut point_checksum = 0i128;
    let mut distance_checksum = 0i128;
    let dimension = case.gram.dim();
    for (target_index, point) in outputs.iter().enumerate() {
        let mut ambient = vec![0i128; dimension];
        for (row, &coefficient) in point.iter().enumerate() {
            for (column, slot) in ambient.iter_mut().enumerate() {
                *slot += i128::from(coefficient) * case.reduced_basis.as_matrix().get(row, column);
            }
        }
        let mut distance = 0i128;
        for (column, value) in ambient.into_iter().enumerate() {
            let weight = i128::try_from(1 + target_index * dimension + column).unwrap();
            point_checksum += weight * value;
            target_checksum += weight * case.ambient_targets[target_index][column];
            let residual = case.ambient_targets[target_index][column] - value * CVP_SCALE;
            distance += residual * residual;
        }
        distance_checksum += i128::try_from(target_index + 1).unwrap() * distance;
    }
    (target_checksum, point_checksum, distance_checksum)
}

struct CvpMeasurement {
    cold_ns: f64,
    warm_ns: f64,
    target_fingerprint: i128,
    point_fingerprint: i128,
    distance_fingerprint: i128,
}

fn benchmark_cvp(dimension: usize) -> Result<CvpMeasurement, Box<dyn std::error::Error>> {
    let case = prepare_cvp(dimension)?;
    let enumerator = Enumerator::new(&case.gram)?;
    let mut scratch = EnumerationScratch::new();
    let mut outputs = vec![vec![0i64; dimension]; TARGETS];

    for (target, output) in case.target_coordinates.iter().zip(&mut outputs) {
        enumerator.nearest_ml(target, output, NODE_BUDGET, &mut scratch)?;
    }
    let (target_fingerprint, point_fingerprint, distance_fingerprint) =
        cvp_fingerprints(&case, &outputs);

    let mut warm_samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        for (target, output) in case.target_coordinates.iter().zip(&mut outputs) {
            enumerator.nearest_ml(target, output, NODE_BUDGET, &mut scratch)?;
            black_box(&output[0]);
        }
        warm_samples.push(start.elapsed());
    }

    let mut cold_samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        for (target, output) in case.target_coordinates.iter().zip(&mut outputs) {
            let cold_enumerator = Enumerator::new(&case.gram)?;
            let mut cold_scratch = EnumerationScratch::new();
            cold_enumerator.nearest_ml(target, output, NODE_BUDGET, &mut cold_scratch)?;
            black_box(&output[0]);
        }
        cold_samples.push(start.elapsed());
    }

    Ok(CvpMeasurement {
        cold_ns: per_operation(median(cold_samples), TARGETS),
        warm_ns: per_operation(median(warm_samples), TARGETS),
        target_fingerprint,
        point_fingerprint,
        distance_fingerprint,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "library,operation,dimension,median_ns,target_fingerprint,point_fingerprint,distance_fingerprint"
    );
    for dimension in DIMENSIONS {
        let (nanoseconds, fingerprint) = benchmark_lll(dimension)?;
        println!("lattica,lll,{dimension},{nanoseconds:.2},{fingerprint},0,0");

        let cvp = benchmark_cvp(dimension)?;
        println!(
            "lattica,cvp_cold,{dimension},{:.2},{},{},{}",
            cvp.cold_ns, cvp.target_fingerprint, cvp.point_fingerprint, cvp.distance_fingerprint
        );
        println!(
            "lattica,cvp_warm,{dimension},{:.2},{},{},{}",
            cvp.warm_ns, cvp.target_fingerprint, cvp.point_fingerprint, cvp.distance_fingerprint
        );
    }
    Ok(())
}

//! Acceptance tests: `mod Λ`, nesting, and Construction A.
//!
//! The `mod Λ` identities are asserted *exactly*. Query points are dyadic and
//! lattice points are integers or half-integers, so every intermediate is
//! exactly representable and an approximate comparison would only hide bugs.

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]

use std::num::NonZeroU32;

use lattica::construct::{CodeMembership, ConstructionA, construction_a_basis};
use lattica::error::DecodeError;
use lattica::int::IntMatrix;
use lattica::named::{d_n, d_n_basis, e8 as e8_gram, e8_generator, zn, zn_basis};
use lattica::nested::Nested;
use lattica::quant::{
    Dn, Quantizer, Scaled, Scratch, Zn, e8 as e8_decoder, mod_lattice, mod_lattice_dithered,
};
#[cfg(miri)]
const ML_CASES: usize = 3;
#[cfg(not(miri))]
const ML_CASES: usize = 3_000;
#[cfg(miri)]
const DIFFERENTIAL_CASES: usize = 5;
#[cfg(not(miri))]
const DIFFERENTIAL_CASES: usize = 5_000;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// A multiple of `2^-8` in `[-8, 8)`.
    fn dyadic(&mut self) -> f64 {
        (f64::from(u32::try_from(self.next() % 4096).unwrap()) - 2048.0) / 256.0
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

// ---------------------------------------------------------------- mod lattice

#[test]
fn reduction_is_idempotent_and_lands_in_the_voronoi_region() {
    let mut rng = Rng(0xA1B2_C3D4_E5F6_0718);
    let q = e8_decoder();
    let mut scratch = Scratch::new(8);
    let (mut once, mut twice) = ([0.0f64; 8], [0.0f64; 8]);
    let mut point = [0i64; 8];

    for _ in 0..2000 {
        let x: Vec<f64> = (0..8).map(|_| rng.dyadic()).collect();
        mod_lattice(&q, &x, &mut once, &mut scratch).unwrap();
        let saved = once;
        mod_lattice(&q, &saved, &mut twice, &mut scratch).unwrap();
        assert_eq!(once, twice, "(x mod L) mod L != x mod L");

        q.nearest(&once, &mut point, &mut scratch).unwrap();
        assert_eq!(point, [0i64; 8], "the residual left the Voronoi region");
    }
}

#[test]
fn reduction_is_invariant_under_adding_a_lattice_vector() {
    // For an *integral* lattice vector the invariance is exact: rounding is
    // unchanged by an integer shift, so the residuals agree coordinate for
    // coordinate.
    let mut rng = Rng(0xB2C3_D4E5_F607_1829);
    let q = Dn::new(8).unwrap();
    let mut scratch = Scratch::new(8);
    let (mut plain, mut shifted) = ([0.0f64; 8], [0.0f64; 8]);

    for _ in 0..1000 {
        let x: Vec<f64> = (0..8).map(|_| rng.dyadic()).collect();
        // An element of D_8: integral with an even coordinate sum.
        let mut lambda: Vec<f64> = (0..8)
            .map(|_| f64::from(i32::try_from(rng.next() % 7).unwrap()) - 3.0)
            .collect();
        if lambda.iter().sum::<f64>() % 2.0 != 0.0 {
            lambda[0] += 1.0;
        }
        let moved: Vec<f64> = x.iter().zip(&lambda).map(|(a, b)| a + b).collect();

        mod_lattice(&q, &x, &mut plain, &mut scratch).unwrap();
        mod_lattice(&q, &moved, &mut shifted, &mut scratch).unwrap();
        assert_eq!(plain, shifted, "(x + lambda) mod L != x mod L");
    }
}

#[test]
fn distance_to_the_lattice_is_translation_invariant() {
    // The general statement, and the strongest one available. Translating by a
    // lattice vector cannot change the distance to the lattice -- but it *can*
    // change which of several equidistant nearest points is chosen, because
    // the tie rules are index-based and the D_n^+ coset preference is not
    // symmetric under a glue-vector shift. So the invariant is asserted on the
    // norm, which is a property of the lattice, rather than on the point,
    // which is a property of the specification. Same shape of limitation as
    // the negation caveat in the quantizer tests.
    let mut rng = Rng(0xB2C3_D4E5_F607_182A);
    let q = e8_decoder();
    let basis = e8_generator();
    let mut scratch = Scratch::new(8);
    let (mut plain, mut shifted) = ([0.0f64; 8], [0.0f64; 8]);
    let mut boundary_hits = 0usize;

    for _ in 0..2000 {
        let x: Vec<f64> = (0..8).map(|_| rng.dyadic()).collect();
        let mut lambda = [0.0f64; 8];
        for row in &basis {
            let c = f64::from(i32::try_from(rng.next() % 5).unwrap()) - 2.0;
            for (dst, &b) in lambda.iter_mut().zip(row) {
                *dst += c * b;
            }
        }
        let moved: Vec<f64> = x.iter().zip(&lambda).map(|(a, b)| a + b).collect();

        mod_lattice(&q, &x, &mut plain, &mut scratch).unwrap();
        mod_lattice(&q, &moved, &mut shifted, &mut scratch).unwrap();

        let a: f64 = plain.iter().map(|v| v * v).sum();
        let b: f64 = shifted.iter().map(|v| v * v).sum();
        assert_eq!(a, b, "distance to the lattice changed under translation");

        if plain != shifted {
            // The two residuals must then differ by a lattice vector.
            let diff: Vec<f64> = plain.iter().zip(&shifted).map(|(p, q)| p - q).collect();
            let mut point = [0i64; 8];
            let mut residue = [0.0f64; 8];
            mod_lattice(&q, &diff, &mut residue, &mut scratch).unwrap();
            q.nearest(&diff, &mut point, &mut scratch).unwrap();
            assert!(
                residue.iter().all(|v| v.abs() < 1e-12),
                "the two residuals differ by a non-lattice vector"
            );
            boundary_hits += 1;
        }
    }
    // The caveat is real, not theoretical: it fires on this input set.
    assert!(boundary_hits > 0, "no boundary case was exercised");
}

#[test]
fn reduction_distributes_over_addition() {
    // ((a mod L) + b) mod L == (a + b) mod L, because Q is Λ-periodic.
    let mut rng = Rng(0xC3D4_E5F6_0718_293A);
    let q = Dn::new(6).unwrap();
    let mut scratch = Scratch::new(6);
    let (mut folded, mut direct, mut partial) = ([0.0f64; 6], [0.0f64; 6], [0.0f64; 6]);

    for _ in 0..2000 {
        let a: Vec<f64> = (0..6).map(|_| rng.dyadic()).collect();
        let b: Vec<f64> = (0..6).map(|_| rng.dyadic()).collect();

        mod_lattice(&q, &a, &mut partial, &mut scratch).unwrap();
        let mixed: Vec<f64> = partial.iter().zip(&b).map(|(p, v)| p + v).collect();
        mod_lattice(&q, &mixed, &mut folded, &mut scratch).unwrap();

        let sum: Vec<f64> = a.iter().zip(&b).map(|(p, v)| p + v).collect();
        mod_lattice(&q, &sum, &mut direct, &mut scratch).unwrap();

        assert_eq!(folded, direct, "mod is not distributive");
    }
}

#[test]
fn dithering_round_trips() {
    // ((x + d) mod L) - d recovers the plain residual shifted by the dither,
    // and is exactly x mod L when d is itself a lattice point.
    let mut rng = Rng(0xD4E5_F607_1829_3A4B);
    let q = Zn::new(5).unwrap();
    let mut scratch = Scratch::new(5);
    let (mut plain, mut dithered) = ([0.0f64; 5], [0.0f64; 5]);

    for _ in 0..1000 {
        let x: Vec<f64> = (0..5).map(|_| rng.dyadic()).collect();
        let lattice_dither: Vec<f64> = (0..5)
            .map(|_| f64::from(i32::try_from(rng.next() % 7).unwrap()) - 3.0)
            .collect();

        mod_lattice(&q, &x, &mut plain, &mut scratch).unwrap();
        mod_lattice_dithered(&q, &x, &lattice_dither, &mut dithered, &mut scratch).unwrap();
        // With a lattice-valued dither the two agree up to a lattice vector:
        // exactly equal away from a Voronoi boundary, and equidistant on one.
        let mut energy_a = 0.0;
        let mut energy_b = 0.0;
        for i in 0..5 {
            let folded = dithered[i] + lattice_dither[i];
            let gap = folded - plain[i];
            assert_eq!(gap, gap.round(), "residuals differ by a non-lattice vector");
            energy_a += folded * folded;
            energy_b += plain[i] * plain[i];
        }
        assert_eq!(energy_a, energy_b, "dithering changed the distance");
    }
}

// --------------------------------------------------------------------- nesting

#[test]
fn the_coset_count_is_the_covolume_ratio_exactly() {
    for n in [2usize, 4, 8] {
        for factor in 2..=4i64 {
            let mut transform = IntMatrix::<i64>::zeros(n, n).unwrap();
            for i in 0..n {
                transform.set(i, i, factor);
            }
            let pair = Nested::new(zn::<i64>(n).unwrap(), transform).unwrap();

            let coding = pair.coding().det().unwrap();
            let shaping = pair.shaping_gram().unwrap().det().unwrap();
            // index^2 = det(Gram_s) / det(Gram_c): the ratio of *covolumes*.
            assert_eq!(pair.index() * pair.index(), shaping / coding);

            let reps = pair.coset_representatives().unwrap();
            assert_eq!(i64::try_from(reps.len()).unwrap(), pair.index());
            let mut seen = reps.clone();
            seen.sort();
            seen.dedup();
            assert_eq!(seen.len(), reps.len(), "representatives collide");
        }
    }
}

#[test]
fn representatives_are_distinct_modulo_the_sublattice() {
    // Two representatives may not differ by a sublattice vector. With
    // Λ_s = M·Λ_c that means no two may agree coordinatewise modulo M.
    let m = 3i64;
    let n = 4usize;
    let mut transform = IntMatrix::<i64>::zeros(n, n).unwrap();
    for i in 0..n {
        transform.set(i, i, m);
    }
    let pair = Nested::new(d_n::<i64>(n).unwrap(), transform).unwrap();
    let reps = pair.coset_representatives().unwrap();
    assert_eq!(reps.len(), 81);

    let mut residues: Vec<Vec<i64>> = reps
        .iter()
        .map(|r| r.iter().map(|v| v.rem_euclid(m)).collect())
        .collect();
    residues.sort();
    residues.dedup();
    assert_eq!(residues.len(), 81, "two cosets coincide");
}

#[test]
fn a_non_nested_pair_is_rejected() {
    let zn4 = zn_basis::<i64>(4).unwrap();
    let dn4 = d_n_basis::<i64>(4).unwrap();
    // D_4 is inside Z^4 at index 2.
    assert_eq!(Nested::from_bases(&zn4, &dn4).unwrap().index(), 2);
    // Z^4 is not inside D_4.
    assert!(Nested::from_bases(&dn4, &zn4).is_err());
}

// ---------------------------------------------------------------- shaping gain

/// The shaping gain of a lattice's Voronoi region over a cube of equal volume,
/// in decibels, measured directly through `mod_lattice`.
fn shaping_gain_db<Q: Quantizer>(
    coding: &Q,
    basis: &[[f64; 8]; 8],
    factor: i64,
    samples: usize,
    rng: &mut Rng,
) -> (f64, f64) {
    let dim = coding.dim();
    let shaping = Scaled::new(coding, factor).unwrap();
    let mut scratch = Scratch::new(dim);
    let mut residual = vec![0.0f64; dim];

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..samples {
        // Uniform over the fundamental parallelepiped of factor*Λ, which
        // `mod_lattice` folds onto the Voronoi region.
        let mut raw = vec![0.0f64; dim];
        for row in basis {
            let u = rng.unit() * factor as f64;
            for (dst, &b) in raw.iter_mut().zip(row) {
                *dst += u * b;
            }
        }
        mod_lattice(&shaping, &raw, &mut residual, &mut scratch).unwrap();
        let energy: f64 = residual.iter().map(|v| v * v).sum();
        sum += energy;
        sum_sq += energy * energy;
    }

    let count = samples as f64;
    let mean = sum / count;
    let power = mean / dim as f64;
    let variance = (sum_sq / count - mean * mean).max(0.0);
    let standard_error = (variance / count).sqrt() / dim as f64;

    let cube = (factor * factor) as f64 / 12.0;
    let gain = 10.0 * (cube / power).log10();
    let error = 10.0 / std::f64::consts::LN_10 * standard_error / power;
    (gain, error)
}

#[test]
fn e8_reproduces_its_published_shaping_gain() {
    // THE RELEASE GATE. G(E_8) = 0.0716821 against G(cube) = 1/12 gives
    // 10*log10((1/12)/0.0716821) = 0.6539 dB. The tolerance is five standard
    // errors of the sample, so it cannot be widened to pass.
    let mut rng = Rng(0x0E80_0E80_0E80_0E81);
    let (gain, error) = shaping_gain_db(&e8_decoder(), &e8_generator(), 4, 200_000, &mut rng);
    assert!(
        (gain - 0.6539).abs() <= 5.0 * error,
        "E_8 shaping gain {gain:.4} dB +/- {error:.4}, published 0.6539 dB"
    );
    // Sanity: it must be well under the ultimate 1.5329 dB and well over zero.
    assert!(gain > 0.4 && gain < 1.0);
}

// ------------------------------------------------------------- Construction A

/// The single parity check code over `Z_2`, length 4. Construction A over it is
/// `D_4`, which the crate can also build two other ways.
struct ParityCheck;

impl CodeMembership for ParityCheck {
    fn modulus(&self) -> NonZeroU32 {
        NonZeroU32::new(2).unwrap()
    }
    fn length(&self) -> usize {
        4
    }
    fn cardinality(&self) -> u64 {
        8
    }
    fn contains(&self, residues: &[u32]) -> bool {
        residues.iter().sum::<u32>() % 2 == 0
    }
    fn decode_costs(&self, costs: &[f64], out: &mut [u32]) -> Result<(), DecodeError> {
        // Exhaustive: 16 words, keep the even-weight one of least cost.
        let mut best = f64::INFINITY;
        let mut best_word = 0usize;
        for word in 0..16usize {
            if (word.count_ones() % 2) != 0 {
                continue;
            }
            let total: f64 = (0..4).map(|i| costs[i * 2 + ((word >> i) & 1)]).sum();
            if total < best {
                best = total;
                best_word = word;
            }
        }
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = u32::try_from((best_word >> i) & 1).unwrap();
        }
        Ok(())
    }
}

/// The ternary repetition code of length 3: `{(a, a, a)}` over `Z_3`.
struct Repetition3;

impl CodeMembership for Repetition3 {
    fn modulus(&self) -> NonZeroU32 {
        NonZeroU32::new(3).unwrap()
    }
    fn length(&self) -> usize {
        3
    }
    fn cardinality(&self) -> u64 {
        3
    }
    fn contains(&self, residues: &[u32]) -> bool {
        residues.iter().all(|&r| r == residues[0])
    }
    fn decode_costs(&self, costs: &[f64], out: &mut [u32]) -> Result<(), DecodeError> {
        let mut best = f64::INFINITY;
        let mut best_symbol = 0u32;
        for s in 0..3usize {
            let total: f64 = (0..3).map(|i| costs[i * 3 + s]).sum();
            if total < best {
                best = total;
                best_symbol = u32::try_from(s).unwrap();
            }
        }
        out.fill(best_symbol);
        Ok(())
    }
}

#[test]
fn construction_a_covolume_is_q_to_the_redundancy() {
    let parity = ConstructionA::new(ParityCheck).unwrap();
    // q^n / |C| = 2^4 / 8 = 2 = q^(n-k) with k = 3.
    assert_eq!(parity.covolume().unwrap(), 2);

    let repetition = ConstructionA::new(Repetition3).unwrap();
    // 3^3 / 3 = 9 = q^(n-k) with k = 1.
    assert_eq!(repetition.covolume().unwrap(), 9);
}

#[test]
fn construction_a_over_the_parity_code_is_the_checkerboard_lattice() {
    let lattice = ConstructionA::new(ParityCheck).unwrap();
    // Membership agrees with D_4's definition.
    for a in -3..=3i64 {
        for b in -3..=3i64 {
            for c in -3..=3i64 {
                for d in -3..=3i64 {
                    let point = [a, b, c, d];
                    let want = (a + b + c + d) % 2 == 0;
                    assert_eq!(lattice.contains(&point).unwrap(), want);
                }
            }
        }
    }
    // And so does the generator-matrix route.
    let generator =
        IntMatrix::<i64>::from_rows(3, 4, &[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]).unwrap();
    let basis = construction_a_basis(2i64, &generator).unwrap();
    assert_eq!(
        basis.gram().unwrap().det().unwrap(),
        d_n::<i64>(4).unwrap().det().unwrap()
    );
}

fn scaled_coordinates(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&coordinate| (coordinate * 256.0) as i64)
        .collect()
}

fn scaled_squared_distance(x: &[i64], point: &[i64]) -> i128 {
    x.iter()
        .zip(point)
        .map(|(&coordinate, &lattice_coordinate)| {
            let residual = i128::from(coordinate) - 256 * i128::from(lattice_coordinate);
            residual * residual
        })
        .sum()
}

/// Minimal squared distance from `x` to the lattice, by exhaustive search over
/// a box that provably contains the answer.
fn brute_force_min<C: CodeMembership>(lattice: &ConstructionA<C>, x: &[f64]) -> i128 {
    let n = x.len();
    // Rounding to the nearest integer point and repairing it into the lattice
    // gives some valid upper bound; a box of half-width `n` around `x` is
    // vastly more than enough for the toy geometries here.
    let lo: Vec<i64> = x.iter().map(|v| (v - 3.0).ceil() as i64).collect();
    let hi: Vec<i64> = x.iter().map(|v| (v + 3.0).floor() as i64).collect();
    let scaled = scaled_coordinates(x);

    let mut best = i128::MAX;
    let mut v = lo.clone();
    loop {
        if lattice.contains(&v).unwrap() {
            let d = scaled_squared_distance(&scaled, &v);
            if d < best {
                best = d;
            }
        }
        let mut i = 0;
        while i < n {
            v[i] += 1;
            if v[i] <= hi[i] {
                break;
            }
            v[i] = lo[i];
            i += 1;
        }
        if i == n {
            break;
        }
    }
    best
}

#[test]
fn construction_a_decoding_is_maximum_likelihood() {
    // The soft-cost seam is what makes this exact. A hard-decision seam would
    // land on a nearby lattice point that is not always the nearest, and the
    // only symptom would be a slightly worse error rate.
    let mut rng = Rng(0xE5F6_0718_293A_4B5C);

    let parity = ConstructionA::new(ParityCheck).unwrap();
    let mut scratch = Scratch::new(4);
    let mut out = [0i64; 4];
    for _ in 0..ML_CASES {
        let x: Vec<f64> = (0..4).map(|_| rng.dyadic()).collect();
        parity.nearest(&x, &mut out, &mut scratch).unwrap();
        assert!(parity.contains(&out).unwrap());
        let got = scaled_squared_distance(&scaled_coordinates(&x), &out);
        assert_eq!(got, brute_force_min(&parity, &x), "parity: {x:?}");
    }

    let repetition = ConstructionA::new(Repetition3).unwrap();
    let mut out = [0i64; 3];
    for _ in 0..ML_CASES {
        let x: Vec<f64> = (0..3).map(|_| rng.dyadic()).collect();
        repetition.nearest(&x, &mut out, &mut scratch).unwrap();
        assert!(repetition.contains(&out).unwrap());
        let got = scaled_squared_distance(&scaled_coordinates(&x), &out);
        assert_eq!(got, brute_force_min(&repetition, &x), "repetition: {x:?}");
    }
}

#[test]
fn construction_a_over_the_parity_code_matches_the_closed_form_decoder() {
    // Two decoders with nothing in common -- a soft-decision code search over
    // Z_2 versus the Conway-Sloane f/g construction -- for the same lattice.
    let mut rng = Rng(0xF607_1829_3A4B_5C6D);
    let coded = ConstructionA::new(ParityCheck).unwrap();
    let closed = Dn::new(4).unwrap();
    let mut scratch = Scratch::new(4);
    let (mut a, mut b) = ([0i64; 4], [0i64; 4]);

    for _ in 0..DIFFERENTIAL_CASES {
        let x: Vec<f64> = (0..4).map(|_| rng.dyadic()).collect();
        coded.nearest(&x, &mut a, &mut scratch).unwrap();
        closed.nearest(&x, &mut b, &mut scratch).unwrap();
        let scaled = scaled_coordinates(&x);
        let da = scaled_squared_distance(&scaled, &a);
        let db = scaled_squared_distance(&scaled, &b);
        assert_eq!(da, db, "decoders disagree on distance for {x:?}");
    }
}

#[test]
fn construction_a_rejects_bad_geometry() {
    let lattice = ConstructionA::new(ParityCheck).unwrap();
    assert!(lattice.contains(&[1, 2, 3]).is_err());
    let mut scratch = Scratch::new(4);
    let mut out = [0i64; 4];
    assert!(
        lattice
            .nearest(&[1.0, f64::NAN, 0.0, 0.0], &mut out, &mut scratch)
            .is_err()
    );
}

#[test]
fn a_nested_pair_built_from_construction_a_has_the_expected_index() {
    // D_4 (Construction A over the parity code) inside Z^4.
    let generator =
        IntMatrix::<i64>::from_rows(3, 4, &[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]).unwrap();
    let shaping = construction_a_basis(2i64, &generator).unwrap();
    let coding = zn_basis::<i64>(4).unwrap();
    let pair = Nested::from_bases(&coding, &shaping).unwrap();
    assert_eq!(pair.index(), 2);
    assert_eq!(pair.coset_representatives().unwrap().len(), 2);
    assert_eq!(
        pair.shaping_gram().unwrap().det().unwrap(),
        e8_gram::<i64>().unwrap().det().unwrap() * 4
    );
}

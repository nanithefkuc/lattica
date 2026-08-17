//! Acceptance tests: GSO, LLL, and Gauss.
//!
//! # The oracle is a certificate, not an expected output
//!
//! LLL has no canonical result — the output depends on `δ`, on the pivot order,
//! on the implementation. Comparing against a stored answer would test the
//! implementation against itself. What *is* canonical is the definition:
//!
//! ```text
//! size-reduced        |μ_{i,j}| ≤ ½
//! Lovász at δ         δ‖b*_{k-1}‖² ≤ ‖b*_k‖² + μ²_{k,k-1}‖b*_{k-1}‖²
//! lattice preserved   U·G·Uᵀ == G_out  and  |det U| == 1
//! ```
//!
//! Those four facts, all checked in exact integers, are a complete proof that
//! the output is an LLL-reduced basis of the input lattice, independent of how
//! it was produced. That is a stronger test than any fixture.

#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::many_single_char_names
)]

use lattica::basis::Gram;
use lattica::gso::Gso;
use lattica::int::IntMatrix;
use lattica::named::{a_n, d_n, e8, zn};
use lattica::reduce::{Delta, Reduced, gauss, is_reduced, lll, lll_deep};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn signed(&mut self, bound: i64) -> i64 {
        i64::try_from(self.next() % u64::try_from(2 * bound + 1).unwrap()).unwrap() - bound
    }
}

/// A random positive-definite Gram matrix, from a random full-rank basis.
fn random_gram(rng: &mut Rng, n: usize, bound: i64) -> Option<Gram<i64>> {
    let data: Vec<i64> = (0..n * n).map(|_| rng.signed(bound)).collect();
    let basis = lattica::Basis::from_rows(n, n, &data).ok()?;
    let gram = basis.gram().ok()?;
    if gram.det().ok()? <= 0 {
        return None;
    }
    Gso::new(&gram).ok()?;
    Some(gram)
}

/// The full certificate: every claim LLL makes, verified in integers.
fn check_certificate(original: &Gram<i64>, reduced: &Reduced<i64>, delta: Delta) {
    let n = original.dim();

    // 1. The transform is unimodular, so the lattice is unchanged.
    assert_eq!(
        reduced.transform.det().unwrap().abs(),
        1,
        "transform is not unimodular"
    );

    // 2. And it actually produced this Gram matrix: U G Uᵀ == G_out.
    let congruence = reduced
        .transform
        .mul(original.as_matrix())
        .unwrap()
        .mul(&reduced.transform.transpose().unwrap())
        .unwrap();
    assert_eq!(&congruence, reduced.gram.as_matrix(), "U G U^T != G_out");

    // 3. Size-reduced and Lovász, re-derived independently of the reduction.
    assert!(
        is_reduced(&reduced.gram, delta).unwrap(),
        "output is not LLL-reduced"
    );

    // 4. The determinant is an invariant of the lattice.
    assert_eq!(reduced.gram.det().unwrap(), original.det().unwrap());
    assert_eq!(reduced.gram.dim(), n);
}

#[test]
fn lll_output_always_carries_a_valid_certificate() {
    let mut rng = Rng(0x11FF_22EE_33DD_44CC);
    let mut checked = 0;
    for n in 2..=6 {
        for _ in 0..40 {
            let Some(gram) = random_gram(&mut rng, n, 6) else {
                continue;
            };
            for delta in [Delta::LLL, Delta::STRONG, Delta::new(1, 2).unwrap()] {
                let Ok(reduced) = lll(&gram, delta) else {
                    continue;
                };
                check_certificate(&gram, &reduced, delta);
                checked += 1;
            }
        }
    }
    assert!(checked > 200, "only {checked} reductions were exercised");
}

#[test]
fn deep_insertion_carries_the_same_certificate() {
    let mut rng = Rng(0x22EE_33DD_44CC_55BB);
    let mut checked = 0;
    for n in 2..=5 {
        for _ in 0..25 {
            let Some(gram) = random_gram(&mut rng, n, 5) else {
                continue;
            };
            let Ok(reduced) = lll_deep(&gram, Delta::LLL) else {
                continue;
            };
            check_certificate(&gram, &reduced, Delta::LLL);
            checked += 1;
        }
    }
    assert!(
        checked > 50,
        "only {checked} deep reductions were exercised"
    );
}

#[test]
fn the_named_lattices_reduce_and_keep_their_invariants() {
    for gram in [
        zn::<i64>(6).unwrap(),
        a_n::<i64>(7).unwrap(),
        d_n::<i64>(7).unwrap(),
        e8::<i64>().unwrap(),
    ] {
        for delta in [Delta::LLL, Delta::STRONG] {
            let reduced = lll(&gram, delta).unwrap();
            check_certificate(&gram, &reduced, delta);
            // Reduction cannot change the minimal distance or the kissing
            // number -- they are properties of the lattice, not the basis.
            let before = lattica::census(&gram, lattica::shortvec::DEFAULT_NODE_BUDGET).unwrap();
            let after =
                lattica::census(&reduced.gram, lattica::shortvec::DEFAULT_NODE_BUDGET).unwrap();
            assert_eq!(before.min_norm_sq, after.min_norm_sq);
            assert_eq!(before.kissing_number, after.kissing_number);
        }
    }
}

#[test]
fn reduction_never_increases_the_orthogonality_defect() {
    // The defect is Π‖b_i‖ / covolume, and the covolume is invariant, so
    // comparing Π G_ii is exact and sufficient.
    let mut rng = Rng(0x33DD_44CC_55BB_66AA);
    let mut improved = 0;
    for n in 2..=5 {
        for _ in 0..40 {
            let Some(gram) = random_gram(&mut rng, n, 5) else {
                continue;
            };
            let Ok(reduced) = lll(&gram, Delta::STRONG) else {
                continue;
            };
            let before: i128 = (0..n).map(|i| i128::from(gram.entry(i, i))).product();
            let after: i128 = (0..n)
                .map(|i| i128::from(reduced.gram.entry(i, i)))
                .product();
            assert!(after <= before, "defect grew: {before} -> {after}");
            if after < before {
                improved += 1;
            }
        }
    }
    assert!(
        improved > 20,
        "reduction never improved anything ({improved})"
    );
}

#[test]
fn reduction_is_a_fixpoint() {
    let mut rng = Rng(0x44CC_55BB_66AA_7799);
    for n in 2..=5 {
        for _ in 0..25 {
            let Some(gram) = random_gram(&mut rng, n, 5) else {
                continue;
            };
            let Ok(once) = lll(&gram, Delta::LLL) else {
                continue;
            };
            let twice = lll(&once.gram, Delta::LLL).unwrap();
            assert_eq!(once.gram, twice.gram, "reduction is not idempotent");
            assert_eq!(twice.transform, IntMatrix::identity(n).unwrap());
        }
    }
}

#[test]
fn a_skewed_basis_reduces_to_a_canonically_short_one() {
    // There is no "the" LLL-reduced basis to compare against: the output
    // depends on delta, on the descent order, and on sign conventions. An
    // independent reference reduction of this very input returns (0,1,0),
    // (1,0,1), (-2,0,1) while this crate returns (-1,0,2) for the third
    // vector -- both perfectly valid, differing by a unimodular change of
    // basis. Asserting either as canonical would be asserting an accident.
    //
    // So this asserts what *is* canonical: the certificate, the determinant,
    // and that the first vector attains the lattice minimum, which is a
    // property of the lattice and is computed here by the independently
    // validated short-vector enumeration.
    let start = lattica::Basis::<i64>::from_rows(3, 3, &[1, 1, 1, -1, 0, 2, 3, 5, 6]).unwrap();
    let gram = start.gram().unwrap();
    let reduced = lll(&gram, Delta::LLL).unwrap();
    check_certificate(&gram, &reduced, Delta::LLL);

    let census = lattica::census(&reduced.gram, lattica::shortvec::DEFAULT_NODE_BUDGET).unwrap();
    assert_eq!(
        reduced.gram.entry(0, 0),
        census.min_norm_sq.unwrap(),
        "the first reduced vector is not a shortest one"
    );

    // Applying the transform to the ambient basis must reproduce the reduced
    // Gram matrix -- the two representations cannot drift apart.
    let applied = reduced.apply(&start).unwrap();
    assert_eq!(applied.gram().unwrap(), reduced.gram);
}

#[test]
fn gauss_returns_a_provably_shortest_pair() {
    // In two dimensions reduction is optimal, not heuristic: the first vector
    // must attain the lattice minimum and the second the successive minimum.
    let mut rng = Rng(0x55BB_66AA_7799_8811);
    for _ in 0..200 {
        let Some(gram) = random_gram(&mut rng, 2, 7) else {
            continue;
        };
        let reduced = gauss(&gram).unwrap();
        check_certificate(&gram, &reduced, Delta::LLL);

        let census = lattica::census(&gram, lattica::shortvec::DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(
            reduced.gram.entry(0, 0),
            census.min_norm_sq.unwrap(),
            "first vector is not a shortest one"
        );
        assert!(reduced.gram.entry(1, 1) >= reduced.gram.entry(0, 0));
    }
}

#[test]
fn an_overflowing_reduction_reports_and_leaves_the_input_alone() {
    // Entries near the width limit make the fraction-free minors overflow.
    let big = i64::MAX / 3;
    let gram = Gram::<i64>::from_rows(2, &[big, big / 2, big / 2, big]).unwrap();
    let before = gram.clone();
    let result = lll(&gram, Delta::LLL);
    assert!(result.is_err(), "expected the width budget to be hit");
    assert_eq!(gram, before, "the input was modified");
}

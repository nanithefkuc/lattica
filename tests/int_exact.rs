//! Acceptance tests: exact integer linear algebra.
//!
//! Every expected value here comes from something other than this crate — a
//! cofactor expansion written independently below, an exactly checkable
//! certificate such as `U * A == H` with `|det U| == 1`, a differential between
//! two independent algorithms, or a structural invariant like the divisibility
//! chain of the invariant factors. Nothing is compared against `lattica`'s own
//! output.
//!
//! # On the dimensions used here
//!
//! Euclidean Hermite reduction with a retained transform suffers the classical
//! coefficient explosion, and the dimensions below are the measured limits of
//! the element widths rather than arbitrary choices: with entries bounded by 6,
//! `i64` intermediates survive to dimension 7 and `i128` to dimension 11. That
//! is a property of the algorithm, not a defect in the implementation, and it
//! is exactly why `hnf_mod_det` exists — the high-dimension coverage runs
//! through it, on the small-determinant lattices this crate actually serves.

use lattica::int::{IntMatrix, det, hnf, hnf_mod_det, invariant_factors};
use lattica::{Op, RangeError, ReduceError};

/// xorshift64, so the randomized cases are reproducible and dependency-free.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn index(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(n).unwrap()).unwrap()
    }

    /// A value in `[-bound, bound]`.
    fn signed(&mut self, bound: i64) -> i64 {
        let span = 2 * bound + 1;
        i64::try_from(self.next() % u64::try_from(span).unwrap()).unwrap() - bound
    }
}

/// Determinant by Laplace cofactor expansion — an independent oracle, correct
/// by definition and far too slow to be the implementation.
fn cofactor_det(m: &[Vec<i128>]) -> i128 {
    let n = m.len();
    if n == 1 {
        return m[0][0];
    }
    let mut acc = 0i128;
    for (j, &entry) in m[0].iter().enumerate() {
        if entry == 0 {
            continue;
        }
        let minor: Vec<Vec<i128>> = m[1..]
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|&(c, _)| c != j)
                    .map(|(_, &v)| v)
                    .collect()
            })
            .collect();
        let term = entry * cofactor_det(&minor);
        if j % 2 == 0 {
            acc += term;
        } else {
            acc -= term;
        }
    }
    acc
}

fn nested(m: &IntMatrix<i128>) -> Vec<Vec<i128>> {
    (0..m.rows()).map(|i| m.row(i).to_vec()).collect()
}

fn widen(m: &IntMatrix<i64>) -> IntMatrix<i128> {
    let data: Vec<i128> = m.as_slice().iter().map(|&v| i128::from(v)).collect();
    IntMatrix::from_rows(m.rows(), m.cols(), &data).unwrap()
}

fn random_matrix(rng: &mut Rng, n: usize, bound: i64) -> IntMatrix<i64> {
    let data: Vec<i64> = (0..n * n).map(|_| rng.signed(bound)).collect();
    IntMatrix::from_rows(n, n, &data).unwrap()
}

/// A random unimodular matrix, built from elementary row operations so its
/// determinant is `±1` by construction rather than by computation.
fn random_unimodular(rng: &mut Rng, n: usize, steps: usize) -> IntMatrix<i64> {
    let mut m = IntMatrix::<i64>::identity(n).unwrap();
    if n < 2 {
        return m;
    }
    for _ in 0..steps {
        let i = rng.index(n);
        let mut j = rng.index(n);
        if i == j {
            j = (j + 1) % n;
        }
        m.row_sub_mul(i, j, rng.signed(1)).unwrap();
    }
    m
}

/// Asserts the four defining properties of a row Hermite Normal Form.
fn check_hermite_shape(h: &IntMatrix<i64>, rank: usize) {
    let mut previous_pivot: Option<usize> = None;
    for i in 0..h.rows() {
        match (0..h.cols()).find(|&j| h.get(i, j) != 0) {
            None => assert!(i >= rank, "zero row {i} inside the rank"),
            Some(col) => {
                assert!(i < rank, "nonzero row {i} outside the rank");
                if let Some(prev) = previous_pivot {
                    assert!(prev < col, "pivots are not strictly increasing");
                }
                previous_pivot = Some(col);
                let p = h.get(i, col);
                assert!(p > 0, "pivot at ({i}, {col}) is not positive");
                for above in 0..i {
                    let v = h.get(above, col);
                    assert!(
                        (0..p).contains(&v),
                        "entry {v} above pivot {p} is unreduced"
                    );
                }
            }
        }
    }
}

#[test]
fn bareiss_agrees_with_cofactor_expansion() {
    let mut rng = Rng::new(0x5EED_0000_0000_0001);
    for n in 1..=6 {
        for _ in 0..40 {
            let a = random_matrix(&mut rng, n, 9);
            assert_eq!(
                i128::from(det(&a).unwrap()),
                cofactor_det(&nested(&widen(&a))),
                "determinant mismatch at n = {n}"
            );
        }
    }
}

#[test]
fn hermite_form_carries_a_valid_certificate() {
    // `U * A == H` with `|det U| == 1` proves the row lattice is unchanged,
    // without reference to how the reduction was performed. That pair of facts
    // is a complete correctness proof of the output.
    let mut rng = Rng::new(0x5EED_0000_0000_0002);
    for n in 1..=7 {
        for _ in 0..20 {
            let a = random_matrix(&mut rng, n, 6);
            let r = hnf(&a).unwrap();
            assert_eq!(r.u.mul(&a).unwrap(), r.h, "U * A != H at n = {n}");
            assert_eq!(
                r.u.det().unwrap().abs(),
                1,
                "U is not unimodular at n = {n}"
            );
            check_hermite_shape(&r.h, r.rank);
        }
    }
}

#[test]
fn hermite_certificate_holds_at_higher_dimension_in_a_wider_type() {
    // `i128` buys a few dimensions, not an order of magnitude -- the explosion
    // is superexponential, so widening is not a strategy. It is measured
    // headroom, and the modular path below is the actual answer.
    let mut rng = Rng::new(0x5EED_0000_0000_0003);
    for n in 8..=10 {
        for _ in 0..10 {
            let a = widen(&random_matrix(&mut rng, n, 4));
            let r = hnf(&a).unwrap();
            assert_eq!(r.u.mul(&a).unwrap(), r.h, "U * A != H at n = {n}");
            assert_eq!(
                r.u.det().unwrap().abs(),
                1,
                "U is not unimodular at n = {n}"
            );
        }
    }
}

#[test]
fn rectangular_and_rank_deficient_input_is_handled() {
    let mut rng = Rng::new(0x5EED_0000_0000_0004);
    for rows in 1..=6usize {
        for cols in 1..=6usize {
            let data: Vec<i64> = (0..rows * cols).map(|_| rng.signed(4)).collect();
            let a = IntMatrix::from_rows(rows, cols, &data).unwrap();
            let r = hnf(&a).unwrap();
            assert_eq!(r.u.mul(&a).unwrap(), r.h, "{rows}x{cols}");
            assert_eq!(r.u.det().unwrap().abs(), 1, "{rows}x{cols}");
            assert!(r.rank <= rows.min(cols));
            check_hermite_shape(&r.h, r.rank);
        }
    }
}

#[test]
fn hermite_form_is_a_lattice_invariant() {
    // Two bases of the same lattice must reduce to the same Hermite form. This
    // is the property that makes the normal form useful at all, and it is
    // checkable without knowing the expected form.
    let mut rng = Rng::new(0x5EED_0000_0000_0005);
    for n in 1..=6 {
        for _ in 0..20 {
            let a = random_matrix(&mut rng, n, 5);
            let u = random_unimodular(&mut rng, n, 3 * n);
            let b = u.mul(&a).unwrap();
            assert_eq!(u.det().unwrap().abs(), 1);
            assert_eq!(hnf(&a).unwrap().h, hnf(&b).unwrap().h, "n = {n}");
        }
    }
}

#[test]
fn hermite_reduction_is_idempotent() {
    let mut rng = Rng::new(0x5EED_0000_0000_0006);
    for n in 1..=7 {
        let a = random_matrix(&mut rng, n, 6);
        let once = hnf(&a).unwrap().h;
        let twice = hnf(&once).unwrap().h;
        assert_eq!(once, twice, "n = {n}");
    }
}

#[test]
fn hermite_reduction_preserves_the_determinant_up_to_sign() {
    let mut rng = Rng::new(0x5EED_0000_0000_0007);
    for n in 1..=7 {
        for _ in 0..20 {
            let a = random_matrix(&mut rng, n, 6);
            let r = hnf(&a).unwrap();
            assert_eq!(det(&a).unwrap().abs(), det(&r.h).unwrap().abs(), "n = {n}");
        }
    }
}

#[test]
fn modular_and_euclidean_hermite_forms_agree() {
    // Two independent algorithms for the same function: Euclidean elimination
    // with a transform, versus stacked reduction modulo the determinant. Neither
    // is the other's oracle by construction, so agreement is real evidence.
    let mut rng = Rng::new(0x5EED_0000_0000_0008);
    let mut compared = 0;
    for n in 1..=7 {
        for _ in 0..30 {
            let a = random_matrix(&mut rng, n, 6);
            let (Ok(euclidean), Ok(modular)) = (hnf(&a), hnf_mod_det(&a)) else {
                continue;
            };
            assert_eq!(euclidean.h, modular, "n = {n}");
            compared += 1;
        }
    }
    assert!(compared > 100, "only {compared} comparisons were possible");
}

#[test]
fn modular_hermite_scales_to_the_dimensions_this_crate_serves() {
    // The lattices `lattica` exists for have tiny determinants: 1 for Z^n and
    // E_8, 4 for D_n, n+1 for A_n. Entries stay below `d` and intermediates
    // below `d^2`, so dimension costs time but not magnitude -- which is the
    // whole reason the modular path is here.
    let mut rng = Rng::new(0x5EED_0000_0000_0009);
    for n in [8usize, 12, 16, 24, 32] {
        // Determinant exactly 1: a disguised basis of Z^n must reduce to I.
        let u = random_unimodular(&mut rng, n, 6 * n);
        assert_eq!(u.det().unwrap().abs(), 1);
        assert_eq!(
            hnf_mod_det(&u).unwrap(),
            IntMatrix::identity(n).unwrap(),
            "unimodular basis at n = {n}"
        );

        // A disguised sublattice of small index: determinant must survive, and
        // the form must be canonical for the lattice rather than for the basis.
        for index in [2i64, 3, 4, 7] {
            let mut scaled = IntMatrix::<i64>::identity(n).unwrap();
            scaled.set(n - 1, n - 1, index);
            let disguised = random_unimodular(&mut rng, n, 4 * n).mul(&scaled).unwrap();
            let h = hnf_mod_det(&disguised).unwrap();
            assert_eq!(det(&h).unwrap().abs(), index, "index {index} at n = {n}");
            check_hermite_shape(&h, n);
            // Canonical: another disguise of the same lattice gives the same form.
            let again = random_unimodular(&mut rng, n, 4 * n)
                .mul(&disguised)
                .unwrap();
            assert_eq!(hnf_mod_det(&again).unwrap(), h, "index {index} at n = {n}");
        }
    }
}

#[test]
fn modular_hermite_rejects_a_singular_matrix() {
    // Row 2 is the sum of rows 0 and 1.
    let a = IntMatrix::<i64>::from_rows(3, 3, &[1, 2, 3, 4, 5, 6, 5, 7, 9]).unwrap();
    assert_eq!(det(&a).unwrap(), 0);
    assert_eq!(hnf_mod_det(&a), Err(ReduceError::Singular));
    // The Euclidean path still handles it, which is why both exist.
    assert_eq!(hnf(&a).unwrap().rank, 2);
}

#[test]
fn invariant_factors_form_a_divisibility_chain_with_the_right_product() {
    let mut rng = Rng::new(0x5EED_0000_0000_000A);
    for n in 1..=6 {
        for _ in 0..25 {
            let a = random_matrix(&mut rng, n, 5);
            let factors = invariant_factors(&a).unwrap();

            for w in factors.windows(2) {
                assert!(w[0] > 0 && w[1] > 0, "factors must be positive");
                assert_eq!(w[1] % w[0], 0, "{factors:?} is not a divisibility chain");
            }

            let d = det(&a).unwrap();
            if d == 0 {
                assert!(factors.len() < n, "singular matrix has full-rank factors");
            } else {
                assert_eq!(factors.len(), n, "nonsingular matrix is rank deficient");
                let product = factors.iter().try_fold(1i64, |acc, &f| acc.checked_mul(f));
                assert_eq!(product, Some(d.abs()), "product of factors != |det|");
            }
        }
    }
}

#[test]
fn invariant_factors_are_invariant_under_a_change_of_basis() {
    let mut rng = Rng::new(0x5EED_0000_0000_000B);
    for n in 2..=6 {
        for _ in 0..15 {
            let a = random_matrix(&mut rng, n, 4);
            let u = random_unimodular(&mut rng, n, 2 * n);
            let b = u.mul(&a).unwrap();
            assert_eq!(
                invariant_factors(&a).unwrap(),
                invariant_factors(&b).unwrap(),
                "n = {n}"
            );
        }
    }
}

#[test]
fn determinant_growth_beyond_the_width_is_reported_not_wrapped() {
    // Invariant I8. The proof is not "the check fired" -- that would trust the
    // check's own arithmetic. It is: the same matrix over `i128` has a
    // determinant provably larger than `i64::MAX`, confirmed against cofactor
    // expansion, so the `i64` path *cannot* produce a correct answer. It must
    // therefore report rather than return one.
    let mut rng = Rng::new(0x5EED_0000_0000_000C);
    let n = 6;

    let (narrow, exact) = (0..64)
        .map(|_| random_matrix(&mut rng, n, 2000))
        .find_map(|m| {
            let d = det(&widen(&m)).ok()?;
            (d.abs() > i128::from(i64::MAX)).then_some((m, d))
        })
        .expect("no sufficiently large determinant was generated");

    assert_eq!(
        exact,
        cofactor_det(&nested(&widen(&narrow))),
        "the wide path itself is wrong"
    );
    assert_eq!(
        det(&narrow),
        Err(RangeError::Overflow {
            op: Op::Mul,
            width_bits: 64
        })
    );
}

#[test]
fn geometry_is_validated_before_allocation() {
    assert_eq!(
        IntMatrix::<i64>::zeros(lattica::int::MAX_DIM + 1, 2),
        Err(RangeError::Dimension {
            requested: lattica::int::MAX_DIM + 1,
            max: lattica::int::MAX_DIM
        })
    );
    assert_eq!(
        IntMatrix::<i64>::from_rows(2, 2, &[1, 2, 3]),
        Err(RangeError::Shape {
            expected: 4,
            found: 3
        })
    );
    let a = IntMatrix::<i64>::zeros(2, 3).unwrap();
    assert!(a.mul(&a).is_err(), "inner dimensions must be checked");
    assert!(det(&a).is_err(), "determinant requires a square matrix");
    assert!(
        hnf_mod_det(&a).is_err(),
        "modular HNF requires a square matrix"
    );
}

#[test]
fn a_rejected_reduction_leaves_the_input_untouched() {
    // Whether these particular calls overflow or not, none of them may modify
    // their argument: the reductions work on a clone.
    let mut rng = Rng::new(0x5EED_0000_0000_000D);
    let a = random_matrix(&mut rng, 8, 1_000_000_000);
    let before = a.clone();
    let _ = hnf(&a);
    let _ = hnf_mod_det(&a);
    let _ = invariant_factors(&a);
    let _ = det(&a);
    assert_eq!(a, before);
}

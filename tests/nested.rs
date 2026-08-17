//! Acceptance tests: nested lattice pairs and Construction A/D generators.
//!
//! The `Nested` identities are asserted *exactly* in integers; the coset
//! counts and representative distinctness come from the Hermite normal form.
//! The shaping and `mod Λ` tests moved to `lattice-engine` with the decoders.

#![allow(clippy::as_conversions, clippy::cast_precision_loss)]

use lattica::construct::construction_a_basis;
use lattica::int::IntMatrix;
use lattica::named::{d_n, d_n_basis, e8, zn, zn_basis};
use lattica::nested::Nested;

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

// --------------------------------------------------------- construction generators

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
        e8::<i64>().unwrap().det().unwrap() * 4
    );
}

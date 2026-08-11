//! The classical lattices, as Gram matrices.
//!
//! Each constructor returns a [`Gram`], because that is what every metric
//! question in this crate is asked of. Where the lattice also has an integral
//! ambient basis — `Z^n`, `A_n`, `D_n` do; `E_8` does not — a `*_basis`
//! constructor returns it, and the two agree by construction:
//! `basis.gram() == gram()`. That agreement is a cross-check, not a definition,
//! since the Gram matrices here are written from the Dynkin diagrams and the
//! bases from their ambient coordinates.
//!
//! No constructor hardcodes a determinant, a minimal norm, or a kissing number.
//! Those are properties to be *computed* — see [`crate::shortvec`] — and
//! hardcoding them would make the acceptance tests circular.

use super::basis::{Basis, Gram};
use super::error::LatticeError;
use super::int::Int;

/// The integer lattice `Z^n`.
///
/// Determinant 1, minimal squared norm 1, kissing number `2n`.
///
/// # Errors
///
/// [`LatticeError::Degenerate`] if `n` is zero, and [`LatticeError::Range`] if
/// `n` exceeds the matrix dimension limit.
pub fn zn<T: Int>(n: usize) -> Result<Gram<T>, LatticeError> {
    if n == 0 {
        return Err(LatticeError::Degenerate);
    }
    let mut data = vec![T::ZERO; n * n];
    for i in 0..n {
        data[i * n + i] = T::ONE;
    }
    Gram::from_rows(n, &data)
}

/// The standard basis of `Z^n`.
///
/// # Errors
///
/// As [`zn`].
pub fn zn_basis<T: Int>(n: usize) -> Result<Basis<T>, LatticeError> {
    if n == 0 {
        return Err(LatticeError::Degenerate);
    }
    let mut data = vec![T::ZERO; n * n];
    for i in 0..n {
        data[i * n + i] = T::ONE;
    }
    Ok(Basis::from_rows(n, n, &data)?)
}

/// The root lattice `A_n`, of rank `n`.
///
/// `A_n = {x ∈ Z^(n+1) : Σ x_i = 0}`. Determinant `n+1`, minimal squared norm
/// 2, kissing number `n(n+1)`. Its Gram matrix is the `A_n` Cartan matrix: a
/// path of `n` nodes.
///
/// # Errors
///
/// [`LatticeError::Degenerate`] if `n` is zero, and [`LatticeError::Range`] on
/// a dimension that is too large.
pub fn a_n<T: Int>(n: usize) -> Result<Gram<T>, LatticeError> {
    if n == 0 {
        return Err(LatticeError::Degenerate);
    }
    cartan(n, &path_edges(n))
}

/// The standard basis of `A_n`, as `n` vectors in `Z^(n+1)`.
///
/// Row `i` is `e_i - e_{i+1}`.
///
/// # Errors
///
/// As [`a_n`].
pub fn a_n_basis<T: Int>(n: usize) -> Result<Basis<T>, LatticeError> {
    if n == 0 {
        return Err(LatticeError::Degenerate);
    }
    let ambient = n + 1;
    let mut data = vec![T::ZERO; n * ambient];
    for i in 0..n {
        data[i * ambient + i] = T::ONE;
        data[i * ambient + i + 1] = T::ZERO.try_sub(T::ONE)?;
    }
    Ok(Basis::from_rows(n, ambient, &data)?)
}

/// The checkerboard lattice `D_n`, of rank `n`, for `n >= 3`.
///
/// `D_n = {x ∈ Z^n : Σ x_i even}`. Determinant 4, minimal squared norm 2,
/// kissing number `2n(n-1)`. Its Gram matrix is the `D_n` Cartan matrix: a path
/// of `n-1` nodes with the last node forked off the second-to-last.
///
/// `D_3` is isomorphic to `A_3` and `D_2` to `A_1 ⊕ A_1`; the lower cases are
/// rejected rather than silently aliased, since the Dynkin construction below
/// does not describe them.
///
/// # Errors
///
/// [`LatticeError::Degenerate`] if `n < 3`, and [`LatticeError::Range`] on a
/// dimension that is too large.
pub fn d_n<T: Int>(n: usize) -> Result<Gram<T>, LatticeError> {
    if n < 3 {
        return Err(LatticeError::Degenerate);
    }
    let mut edges = path_edges(n - 1);
    // The fork: node n-1 attaches to node n-3, not to the end of the path.
    edges.push((n - 3, n - 1));
    cartan(n, &edges)
}

/// The standard basis of `D_n`, as `n` vectors in `Z^n`.
///
/// Rows `0..n-1` are `e_i - e_{i+1}`; the last row is `e_{n-2} + e_{n-1}`.
///
/// # Errors
///
/// As [`d_n`].
pub fn d_n_basis<T: Int>(n: usize) -> Result<Basis<T>, LatticeError> {
    if n < 3 {
        return Err(LatticeError::Degenerate);
    }
    let minus_one = T::ZERO.try_sub(T::ONE)?;
    let mut data = vec![T::ZERO; n * n];
    for i in 0..n - 1 {
        data[i * n + i] = T::ONE;
        data[i * n + i + 1] = minus_one;
    }
    data[(n - 1) * n + (n - 2)] = T::ONE;
    data[(n - 1) * n + (n - 1)] = T::ONE;
    Ok(Basis::from_rows(n, n, &data)?)
}

/// The exceptional lattice `E_8`, as a Gram matrix.
///
/// For the *decoder* see [`crate::quant::e8`]; this is the algebraic side.
///
/// Determinant 1, minimal squared norm 2, kissing number 240. It is the
/// smallest even unimodular lattice, and the densest packing in eight
/// dimensions.
///
/// `E_8` has no integral basis in `Z^8` — half its vectors have half-integer
/// coordinates — so there is no `e8_basis`. Its Gram matrix is integral all the
/// same, which is exactly the reason this crate works in coordinates rather
/// than in ambient space.
///
/// The Gram matrix is the `E_8` Cartan matrix in Bourbaki labelling: a path
/// `1–3–4–5–6–7–8` with node 2 attached to node 4.
///
/// # Errors
///
/// [`LatticeError::Range`] cannot occur at this fixed dimension, but the
/// allocation path is shared with the checked constructors.
pub fn e8<T: Int>() -> Result<Gram<T>, LatticeError> {
    // Bourbaki indices are 1-based; these are the same edges, zero-based.
    let edges = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3)];
    cartan(8, &edges)
}

/// A real generator matrix for `E_8`, one basis vector per row.
///
/// `E_8` is the one named lattice with no integral ambient basis — half its
/// vectors have half-integer coordinates — so this is `f64` where the others
/// return [`Basis`]. Its determinant is 1 and its Gram matrix is integral and
/// even, which is the whole reason the rest of the crate works in coordinates
/// rather than in ambient space.
///
/// Consumers doing ambient geometry — shaping a constellation, generating a
/// dither, plotting — need this; consumers asking metric questions should use
/// [`e8`] and stay exact.
///
/// The rows are `2e_0`, then `e_i - e_{i-1}` for `i` in `1..7`, then the glue
/// vector `(½, …, ½)`.
#[must_use]
pub fn e8_generator() -> [[f64; 8]; 8] {
    let mut rows = [[0.0f64; 8]; 8];
    rows[0][0] = 2.0;
    for i in 1..7 {
        rows[i][i] = 1.0;
        rows[i][i - 1] = -1.0;
    }
    rows[7] = [0.5; 8];
    rows
}

/// The 16-dimensional Barnes–Wall lattice `BW_16`.
///
/// Its Gram determinant is 256, minimal squared norm is 4, and kissing number
/// is 4320. The constructor derives the Gram matrix from the published
/// half-integral generator; none of those invariants is stored.
///
/// # Errors
///
/// The fixed generator is valid for every supported integer width. Errors are
/// reported through the shared checked construction path.
pub fn bw16<T: Int>() -> Result<Gram<T>, LatticeError> {
    gram_from_scaled_generator(16, &BW16_NUMERATORS, 4)
}

/// The 24-dimensional Leech lattice `Λ_24`.
///
/// Its Gram determinant is 1, minimal squared norm is 4, and kissing number is
/// 196560. The Gram matrix is derived from a published integer numerator
/// matrix divided by `sqrt(8)`.
///
/// # Errors
///
/// The fixed generator is valid for every supported integer width. Errors are
/// reported through the shared checked construction path.
pub fn leech24<T: Int>() -> Result<Gram<T>, LatticeError> {
    gram_from_scaled_generator(24, &LEECH24_NUMERATORS, 8)
}

fn gram_from_scaled_generator<T: Int>(
    dimension: usize,
    numerators: &[i8],
    denominator_sq: i128,
) -> Result<Gram<T>, LatticeError> {
    let mut data = vec![T::ZERO; dimension * dimension];
    for i in 0..dimension {
        for j in 0..dimension {
            let mut inner = 0i128;
            for k in 0..dimension {
                let product = i128::from(numerators[i * dimension + k])
                    .checked_mul(i128::from(numerators[j * dimension + k]))
                    .ok_or(crate::error::RangeError::Overflow {
                        op: crate::error::Op::Mul,
                        width_bits: 128,
                    })?;
                inner = inner
                    .checked_add(product)
                    .ok_or(crate::error::RangeError::Overflow {
                        op: crate::error::Op::Add,
                        width_bits: 128,
                    })?;
            }
            if inner % denominator_sq != 0 {
                return Err(LatticeError::Degenerate);
            }
            data[i * dimension + j] = T::narrow(inner / denominator_sq)?;
        }
    }
    Gram::from_rows(dimension, &data)
}

pub(crate) const BW16_NUMERATORS: [i8; 16 * 16] = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0,
    0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2,
    0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2,
    0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
];

pub(crate) const LEECH24_NUMERATORS: [i8; 24 * 24] = [
    8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0,
    0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0,
    2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2,
    2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0,
    2, 0, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0,
    0, 2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0,
    2, 0, 2, 0, 2, 0, 2, 0, -3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,
];

/// Builds a simply-laced Cartan matrix: 2 on the diagonal, -1 for each edge.
fn cartan<T: Int>(n: usize, edges: &[(usize, usize)]) -> Result<Gram<T>, LatticeError> {
    let two = T::ONE.try_add(T::ONE)?;
    let minus_one = T::ZERO.try_sub(T::ONE)?;
    let mut data = vec![T::ZERO; n * n];
    for i in 0..n {
        data[i * n + i] = two;
    }
    for &(i, j) in edges {
        if i >= n || j >= n || i == j {
            return Err(LatticeError::Degenerate);
        }
        data[i * n + j] = minus_one;
        data[j * n + i] = minus_one;
    }
    Gram::from_rows(n, &data)
}

/// Edges of a path on `n` nodes: `0–1–2–…–(n-1)`.
fn path_edges(n: usize) -> Vec<(usize, usize)> {
    (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect()
}

#[cfg(test)]
mod tests {
    use super::{a_n, a_n_basis, d_n, d_n_basis, e8, e8_generator, zn, zn_basis};
    use crate::basis::Gram;
    use crate::error::LatticeError;

    #[test]
    fn degenerate_parameters_are_rejected() {
        assert_eq!(zn::<i64>(0), Err(LatticeError::Degenerate));
        assert_eq!(a_n::<i64>(0), Err(LatticeError::Degenerate));
        assert_eq!(d_n::<i64>(2), Err(LatticeError::Degenerate));
        assert_eq!(d_n_basis::<i64>(2), Err(LatticeError::Degenerate));
    }

    #[test]
    fn ambient_bases_reproduce_the_dynkin_gram_matrices() {
        // Two independent constructions: ambient coordinates versus the Cartan
        // matrix read off the Dynkin diagram.
        for n in 1..=10 {
            assert_eq!(zn_basis::<i64>(n).unwrap().gram().unwrap(), zn(n).unwrap());
            assert_eq!(
                a_n_basis::<i64>(n).unwrap().gram().unwrap(),
                a_n(n).unwrap()
            );
        }
        for n in 3..=10 {
            assert_eq!(
                d_n_basis::<i64>(n).unwrap().gram().unwrap(),
                d_n(n).unwrap()
            );
        }
    }

    #[test]
    fn bases_have_full_rank() {
        for n in 3..=8 {
            assert_eq!(d_n_basis::<i64>(n).unwrap().rank().unwrap(), n);
            assert_eq!(a_n_basis::<i64>(n).unwrap().rank().unwrap(), n);
        }
    }

    #[test]
    fn e8_is_even_and_unimodular() {
        let g = e8::<i64>().unwrap();
        assert_eq!(g.dim(), 8);
        assert_eq!(g.det().unwrap(), 1);
        // Even: every diagonal entry is even, so every vector has even norm.
        for i in 0..8 {
            assert_eq!(g.entry(i, i) % 2, 0);
        }
        assert!(g.is_positive_definite().unwrap());
    }

    #[test]
    fn all_named_lattices_are_positive_definite() {
        for n in 1..=8 {
            assert!(zn::<i64>(n).unwrap().is_positive_definite().unwrap());
            assert!(a_n::<i64>(n).unwrap().is_positive_definite().unwrap());
        }
        for n in 3..=8 {
            assert!(d_n::<i64>(n).unwrap().is_positive_definite().unwrap());
        }
    }

    #[test]
    #[allow(clippy::as_conversions)]
    fn the_e8_generator_matrix_really_generates_e8() {
        // Two unrelated routes to the same lattice: an ambient generator
        // matrix with half-integer entries, and a Cartan matrix read off the
        // Dynkin diagram. Their Gram matrices differ (different bases), but
        // every invariant must agree -- and in dimension 8 an even unimodular
        // lattice is unique, so agreement here is conclusive.
        let rows = e8_generator();
        let mut data = [0i64; 64];
        for i in 0..8 {
            for j in 0..8 {
                let entry: f64 = (0..8).map(|k| rows[i][k] * rows[j][k]).sum();
                assert!(
                    (entry - entry.round()).abs() < 1e-12,
                    "E_8 must be an integral lattice"
                );
                #[allow(clippy::cast_possible_truncation)]
                {
                    data[i * 8 + j] = entry.round() as i64;
                }
            }
        }
        let gram = Gram::from_rows(8, &data).unwrap();
        assert_eq!(gram.det().unwrap(), 1, "unimodular");
        for i in 0..8 {
            assert_eq!(gram.entry(i, i) % 2, 0, "even");
        }
        let census = crate::shortvec::census(&gram, crate::shortvec::DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(census.min_norm_sq, Some(2));
        assert_eq!(census.kissing_number, 240);
    }

    #[test]
    fn a_one_is_the_scaled_integer_lattice() {
        assert_eq!(a_n::<i64>(1).unwrap(), Gram::from_rows(1, &[2]).unwrap());
    }
}

//! Construction A and D: generator matrices for lattices built from codes.
//!
//! # Code-free arithmetic
//!
//! These are the *generator* sides of the constructions: given a code's
//! generator matrix over `Z_q`, compute the lattice's integral basis by
//! Hermite reduction of a stacked integer matrix. No code abstraction appears
//! here — the matrix arrives as numbers, which is why this crate stays
//! independent of every field and graph layer.
//!
//! The decodable side — the `CodeMembership` seam and Construction A
//! decoding — lives in `lattice-engine`, together with every other decision
//! on a real received vector. Construction D's multistage decoder belongs to
//! the consumer that owns the code family; nothing here consumes one.

use crate::basis::Basis;
use crate::error::LatticeError;
use crate::int::{Int, IntMatrix, hnf};

/// The generator matrix of the Construction A lattice `q·Z^n + lift(C)`.
///
/// `generator` holds the code's generator matrix over `Z_q`, one codeword basis
/// per row, with entries already reduced into `[0, q)`. The lattice basis is
/// the Hermite normal form of that matrix stacked over `q·I`, which is a
/// generating set by definition and full rank because of the `q·I` rows.
///
/// # Errors
///
/// [`LatticeError::BadModulus`] for `q < 2`, [`LatticeError::Degenerate`] if
/// the geometry is empty, and [`LatticeError::Range`] on overflow.
pub fn construction_a_basis<T: Int>(
    q: T,
    generator: &IntMatrix<T>,
) -> Result<Basis<T>, LatticeError> {
    let n = generator.cols();
    let k = generator.rows();
    if n == 0 || q <= T::ONE {
        return Err(LatticeError::Degenerate);
    }
    let mut stacked = IntMatrix::<T>::zeros(k + n, n)?;
    for i in 0..k {
        for j in 0..n {
            stacked.set(i, j, generator.get(i, j));
        }
    }
    for i in 0..n {
        stacked.set(k + i, i, q);
    }
    let reduced = hnf(&stacked)?;
    if reduced.rank != n {
        return Err(LatticeError::Degenerate);
    }
    let mut rows = IntMatrix::<T>::zeros(n, n)?;
    for i in 0..n {
        for j in 0..n {
            rows.set(i, j, reduced.h.get(i, j));
        }
    }
    Ok(Basis::from_rows(n, n, rows.as_slice())?)
}

/// The generator matrix of a Construction D lattice.
///
/// `levels[j]` is the generator matrix of the level-`j` code, over `Z_base`,
/// and the lattice is the integer span of
///
/// ```text
/// { base^j * g : g a row of levels[j] }  union  { base^a * e_i }
/// ```
///
/// with `a = levels.len()`. The caller is responsible for the chain being
/// nested, `C_0 ⊆ C_1 ⊆ … ⊆ C_{a-1}`; this function computes the span of what
/// it is given, which is what a generator-based construction means.
///
/// # Errors
///
/// [`LatticeError::Degenerate`] for an empty level list, a zero length, or a
/// base below 2; [`LatticeError::Range`] on overflow.
pub fn construction_d_basis<T: Int>(
    base: T,
    levels: &[IntMatrix<T>],
) -> Result<Basis<T>, LatticeError> {
    if levels.is_empty() || base <= T::ONE {
        return Err(LatticeError::Degenerate);
    }
    let n = levels[0].cols();
    if n == 0 || levels.iter().any(|m| m.cols() != n) {
        return Err(LatticeError::Degenerate);
    }

    let total_rows = levels.iter().map(IntMatrix::rows).sum::<usize>() + n;
    let mut stacked = IntMatrix::<T>::zeros(total_rows, n)?;

    let mut row = 0usize;
    let mut weight = T::ONE;
    for level in levels {
        for i in 0..level.rows() {
            for j in 0..n {
                stacked.set(row, j, level.get(i, j).try_mul(weight)?);
            }
            row += 1;
        }
        weight = weight.try_mul(base)?;
    }
    // `weight` is now base^a.
    for i in 0..n {
        stacked.set(row + i, i, weight);
    }

    let reduced = hnf(&stacked)?;
    if reduced.rank != n {
        return Err(LatticeError::Degenerate);
    }
    let mut rows = IntMatrix::<T>::zeros(n, n)?;
    for i in 0..n {
        for j in 0..n {
            rows.set(i, j, reduced.h.get(i, j));
        }
    }
    Ok(Basis::from_rows(n, n, rows.as_slice())?)
}

#[cfg(test)]
mod tests {
    use super::{construction_a_basis, construction_d_basis};
    use crate::int::IntMatrix;
    use crate::named::d_n;

    #[test]
    fn the_single_parity_check_code_gives_the_checkerboard_lattice() {
        // C = {x in F_2^4 : sum even} has generator rows (1,1,0,0), (0,1,1,0),
        // (0,0,1,1), and q*Z^n + lift(C) is exactly D_4.
        let generator =
            IntMatrix::<i64>::from_rows(3, 4, &[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]).unwrap();
        let basis = construction_a_basis(2i64, &generator).unwrap();
        let gram = basis.gram().unwrap();
        // Covolume q^(n-k) = 2^1 = 2, so det Gram = 4 -- the same as D_4.
        assert_eq!(gram.det().unwrap(), 4);
        assert_eq!(gram.det().unwrap(), d_n::<i64>(4).unwrap().det().unwrap());
    }

    #[test]
    fn the_whole_space_gives_the_integer_lattice() {
        let generator = IntMatrix::<i64>::identity(3).unwrap();
        let basis = construction_a_basis(5i64, &generator).unwrap();
        assert_eq!(basis.gram().unwrap().det().unwrap(), 1);
    }

    #[test]
    fn the_trivial_code_gives_the_scaled_integer_lattice() {
        let generator = IntMatrix::<i64>::zeros(1, 3).unwrap();
        let basis = construction_a_basis(3i64, &generator).unwrap();
        // 3Z^3 has covolume 27, so det Gram = 27^2 = 729.
        assert_eq!(basis.gram().unwrap().det().unwrap(), 729);
    }

    #[test]
    fn construction_d_with_one_trivial_level_is_the_scaled_lattice() {
        let levels = [IntMatrix::<i64>::zeros(1, 3).unwrap()];
        let basis = construction_d_basis(2i64, &levels).unwrap();
        // 2Z^3: covolume 8, det Gram 64.
        assert_eq!(basis.gram().unwrap().det().unwrap(), 64);
    }

    #[test]
    fn construction_d_at_one_level_agrees_with_construction_a() {
        let generator =
            IntMatrix::<i64>::from_rows(3, 4, &[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]).unwrap();
        let a = construction_a_basis(2i64, &generator).unwrap();
        let d = construction_d_basis(2i64, core::slice::from_ref(&generator)).unwrap();
        assert_eq!(a, d);
    }

    #[test]
    fn degenerate_parameters_are_rejected() {
        let empty: [IntMatrix<i64>; 0] = [];
        assert!(construction_d_basis(2i64, &empty).is_err());
        assert!(construction_d_basis(1i64, &[IntMatrix::<i64>::identity(2).unwrap()]).is_err());
        assert!(construction_a_basis(1i64, &IntMatrix::<i64>::identity(2).unwrap()).is_err());
    }
}

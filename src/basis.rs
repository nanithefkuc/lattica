//! Lattice representation: integral generator matrices and Gram matrices.
//!
//! # Coordinates, not ambient points
//!
//! A lattice vector is stored as its **integer coordinate vector** `c` relative
//! to a basis, never as an ambient point. Every metric quantity then comes from
//! the Gram matrix: `<x, y> = c G dᵀ` and `‖x‖² = c G cᵀ`.
//!
//! This is not a stylistic choice. `E_8` and `D_n^+` have half-integer ambient
//! coordinates, so an ambient representation would force rationals or a scale
//! factor into the middle of the crate — and with them, a rounding question in
//! every norm. Their *Gram* matrices are integral, because they are integral
//! lattices: inner products of lattice vectors are integers even when the
//! coordinates are not. Working in coordinates keeps invariant I1 intact for
//! every lattice, not just the ones that happen to sit inside `Z^n`.
//!
//! [`Basis`] exists for the lattices that do have an integral ambient basis. It
//! is a way to *construct* a Gram matrix, and a way to cross-check one.

use super::error::{LatticeError, RangeError, ReduceError};
use super::gso::Gso;
use super::int::{Int, IntMatrix, det, hnf};

/// An integral generator matrix: one lattice basis vector per row, in ambient
/// `Z^m` coordinates.
///
/// Only lattices that embed in `Z^m` have one. `E_8` does not; construct it
/// from its Gram matrix instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Basis<T: Int> {
    rows: IntMatrix<T>,
}

impl<T: Int> Basis<T> {
    /// Wraps a generator matrix whose rows are basis vectors.
    ///
    /// # Errors
    ///
    /// [`RangeError::Shape`] if `data` is not `count * ambient` long, and
    /// [`RangeError::Dimension`] if either dimension is too large.
    pub fn from_rows(count: usize, ambient: usize, data: &[T]) -> Result<Self, RangeError> {
        Ok(Self {
            rows: IntMatrix::from_rows(count, ambient, data)?,
        })
    }

    /// Number of basis vectors.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.rows.rows()
    }

    /// Dimension of the ambient space.
    #[must_use]
    pub const fn ambient_dim(&self) -> usize {
        self.rows.cols()
    }

    /// The generator matrix.
    #[must_use]
    pub const fn as_matrix(&self) -> &IntMatrix<T> {
        &self.rows
    }

    /// The rank of the lattice these vectors generate.
    ///
    /// Equal to [`count`](Basis::count) exactly when the rows are independent.
    ///
    /// # Errors
    ///
    /// [`RangeError`] if Hermite reduction exceeds the element width.
    pub fn rank(&self) -> Result<usize, RangeError> {
        Ok(hnf(&self.rows)?.rank)
    }

    /// The Gram matrix `B Bᵀ`.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if an inner product exceeds the element width;
    /// [`LatticeError::Degenerate`] cannot arise here, since `B Bᵀ` is
    /// symmetric by construction.
    pub fn gram(&self) -> Result<Gram<T>, RangeError> {
        let count = self.count();
        let ambient = self.ambient_dim();
        let mut product = IntMatrix::zeros(count, count)?;
        for i in 0..count {
            for j in 0..=i {
                let mut inner = T::ZERO;
                for k in 0..ambient {
                    inner = inner.try_add(self.rows.get(i, k).try_mul(self.rows.get(j, k))?)?;
                }
                product.set(i, j, inner);
                if i != j {
                    product.set(j, i, inner);
                }
            }
        }
        Gram::new(product).map_err(|error| match error {
            LatticeError::Range(range) => range,
            _ => unreachable!("mirrored inner products are symmetric"),
        })
    }
}

/// A Gram matrix: the symmetric integer matrix of pairwise inner products of a
/// basis.
///
/// This is the crate's working representation of a lattice. It determines every
/// metric property — determinant, norms, minimal distance, kissing number — and
/// it is integral for every integral lattice, including those whose ambient
/// coordinates are not.
///
/// Construction validates squareness and symmetry. Positive-definiteness is
/// *not* checked here, because doing so costs a full fraction-free
/// factorization; it is established where it is needed, by
/// [`is_positive_definite`](Gram::is_positive_definite) or implicitly by
/// [`crate::shortvec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gram<T: Int> {
    m: IntMatrix<T>,
}

impl<T: Int> Gram<T> {
    /// Wraps a square symmetric integer matrix.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Range`] with [`RangeError::Shape`] if the matrix is not
    /// square, and [`LatticeError::Degenerate`] if it is not symmetric.
    pub fn new(m: IntMatrix<T>) -> Result<Self, LatticeError> {
        if !m.is_square() {
            return Err(RangeError::Shape {
                expected: m.rows(),
                found: m.cols(),
            }
            .into());
        }
        for i in 0..m.rows() {
            for j in 0..i {
                if m.get(i, j) != m.get(j, i) {
                    return Err(LatticeError::Degenerate);
                }
            }
        }
        Ok(Self { m })
    }

    /// Builds a Gram matrix from row-major data.
    ///
    /// # Errors
    ///
    /// As [`new`](Gram::new), plus [`RangeError::Shape`] if `data` is not
    /// `n * n` long.
    pub fn from_rows(n: usize, data: &[T]) -> Result<Self, LatticeError> {
        Self::new(IntMatrix::from_rows(n, n, data)?)
    }

    /// The dimension, which is the rank of the lattice.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.m.rows()
    }

    /// The entry at `(i, j)`, the inner product of basis vectors `i` and `j`.
    ///
    /// # Panics
    ///
    /// If either index is out of bounds.
    #[must_use]
    pub fn entry(&self, i: usize, j: usize) -> T {
        self.m.get(i, j)
    }

    /// The underlying matrix.
    #[must_use]
    pub const fn as_matrix(&self) -> &IntMatrix<T> {
        &self.m
    }

    /// The lattice determinant, meaning `det G`.
    ///
    /// This is the square of the covolume, and it is the quantity tabulated as
    /// "determinant" for the classical lattices: 1 for `Z^n` and `E_8`, 4 for
    /// `D_n`, `n+1` for `A_n`.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if an intermediate exceeds the element width.
    pub fn det(&self) -> Result<T, RangeError> {
        det(&self.m)
    }

    /// The squared norm `c G cᵀ` of the lattice vector with coordinates `c`.
    ///
    /// # Errors
    ///
    /// [`RangeError::Shape`] if `c` has the wrong length, and
    /// [`RangeError::Overflow`] on accumulation overflow.
    pub fn norm_sq(&self, c: &[T]) -> Result<T, RangeError> {
        self.inner(c, c)
    }

    /// The inner product `a G bᵀ`.
    ///
    /// # Errors
    ///
    /// [`RangeError::Shape`] if either operand has the wrong length, and
    /// [`RangeError::Overflow`] on accumulation overflow.
    pub fn inner(&self, a: &[T], b: &[T]) -> Result<T, RangeError> {
        let n = self.dim();
        if a.len() != n || b.len() != n {
            return Err(RangeError::Shape {
                expected: n,
                found: if a.len() == n { b.len() } else { a.len() },
            });
        }
        let mut total = T::ZERO;
        for (i, &ai) in a.iter().enumerate() {
            if ai.is_zero() {
                continue;
            }
            let mut row = T::ZERO;
            for (j, &bj) in b.iter().enumerate() {
                if bj.is_zero() {
                    continue;
                }
                row = row.try_add(self.m.get(i, j).try_mul(bj)?)?;
            }
            total = total.try_add(ai.try_mul(row)?)?;
        }
        Ok(total)
    }

    /// Returns `true` if every leading principal minor is positive, which for a
    /// symmetric matrix is exactly positive-definiteness (Sylvester).
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if a minor exceeds the element width.
    pub fn is_positive_definite(&self) -> Result<bool, RangeError> {
        match Gso::new(self) {
            Ok(_) => Ok(true),
            Err(ReduceError::NotFullRank { .. } | ReduceError::Singular) => Ok(false),
            Err(ReduceError::Range(error)) => Err(error),
            Err(ReduceError::BudgetExhausted { .. }) => {
                unreachable!("factorization has no iterative search")
            }
        }
    }

    /// The adjugate, satisfying `adj(G) · G == det(G) · I`.
    ///
    /// The dual lattice has Gram matrix `G⁻¹ = adj(G) / det(G)`. Returning the
    /// adjugate rather than the inverse keeps the dual exactly representable:
    /// the pair `(adj(G), det(G))` carries the same information with no
    /// rationals, and the identities that characterise duality —
    /// `det(adj G) = det(G)^(n-1)` and `adj(adj G) = det(G)^(n-2) · G` — are
    /// then checkable in integers.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if a cofactor exceeds the element width.
    pub fn adjugate(&self) -> Result<Self, LatticeError> {
        Self::new(crate::int::adjugate(&self.m)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{Basis, Gram};
    use crate::error::LatticeError;
    use crate::int::IntMatrix;

    #[test]
    fn a_non_symmetric_matrix_is_not_a_gram_matrix() {
        let m = IntMatrix::<i64>::from_rows(2, 2, &[1, 2, 3, 4]).unwrap();
        assert_eq!(Gram::new(m), Err(LatticeError::Degenerate));
    }

    #[test]
    fn a_non_square_matrix_is_not_a_gram_matrix() {
        let m = IntMatrix::<i64>::zeros(2, 3).unwrap();
        assert!(Gram::new(m).is_err());
    }

    #[test]
    fn gram_of_the_standard_basis_is_the_identity() {
        let b = Basis::<i64>::from_rows(3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1]).unwrap();
        let g = b.gram().unwrap();
        assert_eq!(g.as_matrix(), &IntMatrix::identity(3).unwrap());
        assert_eq!(g.det().unwrap(), 1);
        assert_eq!(b.rank().unwrap(), 3);
    }

    #[test]
    fn norms_and_inner_products_come_from_the_gram_matrix() {
        // A basis of two vectors at 60 degrees with squared length 2: the
        // hexagonal lattice A_2.
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        assert_eq!(g.norm_sq(&[1, 0]).unwrap(), 2);
        assert_eq!(g.norm_sq(&[1, 1]).unwrap(), 2);
        assert_eq!(g.norm_sq(&[1, -1]).unwrap(), 6);
        assert_eq!(g.inner(&[1, 0], &[0, 1]).unwrap(), -1);
        assert_eq!(g.det().unwrap(), 3);
    }

    #[test]
    fn length_mismatches_are_rejected() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        assert!(g.norm_sq(&[1]).is_err());
        assert!(g.inner(&[1, 0], &[0]).is_err());
    }

    #[test]
    fn adjugate_times_the_matrix_is_the_determinant() {
        let g = Gram::<i64>::from_rows(3, &[2, -1, 0, -1, 2, -1, 0, -1, 2]).unwrap();
        let d = g.det().unwrap();
        let product = g
            .adjugate()
            .unwrap()
            .as_matrix()
            .mul(g.as_matrix())
            .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product.get(i, j), if i == j { d } else { 0 });
            }
        }
    }

    #[test]
    fn positive_definiteness_follows_sylvester() {
        let good = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        assert!(good.is_positive_definite().unwrap());
        // Indefinite: determinant is negative.
        let bad = Gram::<i64>::from_rows(2, &[1, 2, 2, 1]).unwrap();
        assert!(!bad.is_positive_definite().unwrap());
    }
}

//! Gram–Schmidt orthogonalization, without leaving the integers.
//!
//! # Why this is exact
//!
//! Textbook Gram–Schmidt produces rationals: `μ_{i,j}` and `‖b*_i‖²` are
//! quotients, and computing them in floating point puts an approximation
//! underneath every basis reduction that follows. The fraction-free
//! formulation keeps two integer arrays instead:
//!
//! ```text
//! d[k]      the k-by-k leading principal minor of G, with d[0] = 1
//! λ[i][j]   an integer, for j < i
//!
//! ‖b*_i‖² = d[i+1] / d[i]          μ_{i,j} = λ[i][j] / d[j+1]
//! ```
//!
//! Both are exact rationals with *known* denominators, so every test a
//! reduction needs — size reduction, the Lovász condition, deep insertion —
//! becomes an integer comparison after clearing them. Nothing rounds.
//!
//! The arrays come from Bareiss elimination on `G`: the diagonal of the
//! fraction-free upper-triangular form is `d[1..=n]`, and the off-diagonal
//! entries are the `λ`. That is the same factorization
//! [`crate::shortvec`] enumerates with, and it is computed here so there is one
//! implementation of it rather than two.
//!
//! # Positive-definiteness comes free
//!
//! Sylvester's criterion is "every leading principal minor is positive", and
//! this computes exactly those. A Gram matrix that is not positive definite
//! does not describe a lattice, and [`Gso::new`] rejects it.

// Converting exact integers into `f64` is the defining act of these modules:
// the lattice is integral, the target is real, and the two must meet. Every
// cast is on a Gram entry, a minor, or a coefficient already validated to be
// finite; where a minor could exceed 2^53 the loss is in a reported quantity,
// never in a decision the exact path also makes.
#![allow(clippy::as_conversions, clippy::cast_precision_loss)]

use crate::basis::Gram;
use crate::error::{RangeError, ReduceError};
use crate::int::Int;

/// The fraction-free Gram–Schmidt data of a positive-definite Gram matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gso<T: Int> {
    n: usize,
    /// Bareiss upper-triangular form, row-major. `upper[j * n + i]` is
    /// `λ[i][j]` for `j < i`, and `upper[k * n + k]` is `d[k+1]`.
    upper: Vec<T>,
    /// Leading principal minors, `minors[0] = 1` through `minors[n]`.
    minors: Vec<T>,
}

impl<T: Int> Gso<T> {
    /// Computes the orthogonalization.
    ///
    /// # Errors
    ///
    /// [`ReduceError::NotFullRank`] if some leading principal minor is not
    /// positive, which by Sylvester means the matrix is not positive definite
    /// and so is not the Gram matrix of a basis; [`ReduceError::Range`] if an
    /// intermediate exceeds the element width.
    pub fn new(gram: &Gram<T>) -> Result<Self, ReduceError> {
        let n = gram.dim();
        let mut work = gram.as_matrix().as_slice().to_vec();
        let mut minors = Vec::with_capacity(n + 1);
        minors.push(T::ONE);

        let mut previous = T::ONE;
        for k in 0..n {
            let pivot = work[k * n + k];
            if pivot <= T::ZERO {
                return Err(ReduceError::NotFullRank {
                    rank: k,
                    required: n,
                });
            }
            for i in k + 1..n {
                let leading = work[i * n + k];
                for j in k + 1..n {
                    let cross = work[i * n + j]
                        .try_mul(pivot)?
                        .try_sub(leading.try_mul(work[k * n + j])?)?;
                    work[i * n + j] = cross.try_div_exact(previous)?;
                }
                work[i * n + k] = T::ZERO;
            }
            previous = pivot;
            minors.push(pivot);
        }

        Ok(Self {
            n,
            upper: work,
            minors,
        })
    }

    /// The dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.n
    }

    /// The `k`-by-`k` leading principal minor, with `minor(0) == 1`.
    ///
    /// # Panics
    ///
    /// If `k > dim()`.
    #[must_use]
    pub fn minor(&self, k: usize) -> T {
        self.minors[k]
    }

    /// The integer `λ[i][j]`, for `j < i`.
    ///
    /// # Panics
    ///
    /// If the indices are out of range or `j >= i`.
    #[must_use]
    pub fn lambda(&self, i: usize, j: usize) -> T {
        assert!(j < i && i < self.n, "lambda is defined for j < i");
        self.upper[j * self.n + i]
    }

    /// `‖b*_i‖²` as the exact fraction `(numerator, denominator)`, namely
    /// `(d[i+1], d[i])`.
    ///
    /// # Panics
    ///
    /// If `i >= dim()`.
    #[must_use]
    pub fn norm_sq(&self, i: usize) -> (T, T) {
        (self.minors[i + 1], self.minors[i])
    }

    /// `μ_{i,j}` as an `f64`, for reporting and for the real-valued paths.
    ///
    /// The exact value is `lambda(i, j) / minor(j + 1)`; prefer those when the
    /// answer must be exact.
    ///
    /// # Panics
    ///
    /// If `j >= i` or the indices are out of range.
    #[must_use]
    pub fn mu(&self, i: usize, j: usize) -> f64 {
        let numerator = self.lambda(i, j).widen() as f64;
        let denominator = self.minors[j + 1].widen() as f64;
        numerator / denominator
    }

    /// Is every `|μ_{i,j}| ≤ ½`?
    ///
    /// Checked as `2·|λ[i][j]| ≤ d[j+1]`, so no rounding is involved.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if a doubling overflows.
    pub fn is_size_reduced(&self) -> Result<bool, RangeError> {
        for i in 1..self.n {
            for j in 0..i {
                let doubled = self
                    .lambda(i, j)
                    .try_abs()?
                    .try_add(self.lambda(i, j).try_abs()?)?;
                if doubled > self.minors[j + 1] {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// The product of the squared basis-vector norms, `Π G_ii`.
    ///
    /// Divided by `det G` this is the squared orthogonality defect. Reduction
    /// must never increase it, and comparing the products alone suffices since
    /// the determinant is invariant.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if the product does not fit.
    pub fn diagonal_product(gram: &Gram<T>) -> Result<T, RangeError> {
        let mut total = T::ONE;
        for i in 0..gram.dim() {
            total = total.try_mul(gram.entry(i, i))?;
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::Gso;
    use crate::basis::Gram;
    use crate::named::{a_n, d_n, e8, zn};

    #[test]
    fn the_integer_lattice_is_already_orthogonal() {
        let g = zn::<i64>(5).unwrap();
        let gso = Gso::new(&g).unwrap();
        for k in 0..=5 {
            assert_eq!(gso.minor(k), 1);
        }
        for i in 0..5 {
            assert_eq!(gso.norm_sq(i), (1, 1));
        }
        for i in 1..5 {
            for j in 0..i {
                assert_eq!(gso.lambda(i, j), 0);
            }
        }
        assert!(gso.is_size_reduced().unwrap());
    }

    #[test]
    fn minors_are_the_leading_determinants() {
        // A_n's leading k-by-k minor is k+1, which is also det(A_k).
        let g = a_n::<i64>(6).unwrap();
        let gso = Gso::new(&g).unwrap();
        assert_eq!(gso.minor(0), 1);
        for k in 1..=6 {
            assert_eq!(gso.minor(k), i64::try_from(k).unwrap() + 1);
        }
        assert_eq!(gso.minor(6), g.det().unwrap());
    }

    #[test]
    fn squared_norms_multiply_to_the_determinant() {
        // Π ‖b*_i‖² = Π d[i+1]/d[i] telescopes to d[n] = det G.
        for g in [
            zn::<i64>(4).unwrap(),
            a_n::<i64>(5).unwrap(),
            d_n::<i64>(6).unwrap(),
            e8::<i64>().unwrap(),
        ] {
            let gso = Gso::new(&g).unwrap();
            let n = g.dim();
            assert_eq!(gso.minor(0), 1);
            assert_eq!(gso.minor(n), g.det().unwrap());
        }
    }

    #[test]
    fn mu_matches_the_ratio_it_is_defined_as() {
        let g = a_n::<i64>(4).unwrap();
        let gso = Gso::new(&g).unwrap();
        for i in 1..4 {
            for j in 0..i {
                let expected = gso.lambda(i, j) as f64 / gso.minor(j + 1) as f64;
                assert!((gso.mu(i, j) - expected).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn a_form_that_is_not_positive_definite_is_rejected() {
        let g = Gram::<i64>::from_rows(2, &[1, 2, 2, 1]).unwrap();
        assert!(Gso::new(&g).is_err());
    }

    #[test]
    fn the_root_bases_are_not_size_reduced() {
        // A tempting assumption -- every off-diagonal entry is 0 or -1 against
        // a diagonal of 2 -- and false. A_2 happens to satisfy |mu| <= 1/2
        // exactly, but A_6 reaches mu = -5/6 and E_8 reaches -4/3. The Cartan
        // bases are short, not size-reduced, and those are different
        // properties. This is what gives LLL something to do on them.
        assert!(
            Gso::new(&a_n::<i64>(2).unwrap())
                .unwrap()
                .is_size_reduced()
                .unwrap()
        );
        for g in [
            a_n::<i64>(6).unwrap(),
            d_n::<i64>(6).unwrap(),
            e8::<i64>().unwrap(),
        ] {
            assert!(!Gso::new(&g).unwrap().is_size_reduced().unwrap());
        }
    }
}

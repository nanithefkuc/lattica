//! Nested lattice pairs `Λ_s ⊆ Λ_c`, their index, and their cosets.
//!
//! # Nesting is a matrix, not a search
//!
//! A sublattice of `Λ_c` is exactly an integral matrix `T` whose rows are its
//! generators *written in `Λ_c`'s basis*. In that representation inclusion is
//! not a property to be tested — it is a consequence of `T` being integral —
//! the index is `|det T|`, and the coset representatives fall out of the
//! Hermite normal form. Everything is exact integer arithmetic, and no ambient
//! coordinates appear.
//!
//! [`Nested::from_bases`] exists for callers who have two ambient bases and
//! want the inclusion *checked*: it solves `B_s = T · B_c` exactly, verifies
//! that `T` reconstructs the shaping basis, and reports
//! [`LatticeError::NotNested`] when it does not.
//!
//! # A note on determinants
//!
//! With `det Λ` meaning the Gram determinant, `det Λ_s = (det T)² · det Λ_c`,
//! so the index is `sqrt(det Λ_s / det Λ_c)` — the ratio of *covolumes*, not of
//! determinants. It is easy to write the unsquared version by accident; the
//! test suite checks the correct one.

use crate::basis::{Basis, Gram};
use crate::error::{LatticeError, RangeError};
use crate::int::{Int, IntMatrix, adjugate, hnf_form};

/// A pair `Λ_s ⊆ Λ_c` of a lattice and a finite-index sublattice.
///
/// The coding lattice `Λ_c` is the fine one and carries the code; the shaping
/// lattice `Λ_s` is the coarse sublattice and bounds the constellation. Their
/// quotient is the codebook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Nested<T: Int> {
    coding: Gram<T>,
    transform: IntMatrix<T>,
    index: T,
    radices: Vec<T>,
}

impl<T: Int> Nested<T> {
    /// Builds the pair from the coding lattice and the sublattice's generators
    /// in coding coordinates.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Range`] if `transform` is not square or disagrees with
    /// the lattice dimension, and [`LatticeError::Degenerate`] if it is
    /// singular, which would make the sublattice lower-rank rather than
    /// finite-index.
    pub fn new(coding: Gram<T>, transform: IntMatrix<T>) -> Result<Self, LatticeError> {
        let n = coding.dim();
        if transform.rows() != n || transform.cols() != n {
            return Err(RangeError::Shape {
                expected: n,
                found: transform.rows(),
            }
            .into());
        }
        // The Hermite form of `T` is upper triangular with positive diagonal,
        // and the box `0 <= a_i < H_ii` is a complete set of coset
        // representatives for `Z^n / rowspan(H)`. Those diagonal entries are
        // the mixed radices an encoder counts in — and, for a square
        // full-rank form, their product is `|det T|`, the index. One
        // certificate-free elimination serves both; a zero product is exactly
        // singularity.
        let (form, _rank) = hnf_form(&transform)?;
        let mut radices = Vec::with_capacity(n);
        let mut index = T::ONE;
        for i in 0..n {
            let radix = form.get(i, i);
            index = index.try_mul(radix)?;
            radices.push(radix);
        }
        if index.is_zero() {
            return Err(LatticeError::Degenerate);
        }

        Ok(Self {
            coding,
            transform,
            index,
            radices,
        })
    }

    /// Builds the pair from two ambient bases, **checking** that the second is
    /// a sublattice of the first.
    ///
    /// Solves `B_s = T · B_c` as `T = (B_s B_cᵀ) · adj(G_c) / det(G_c)`, which
    /// works for a non-square coding basis — `A_n` has one — and is exact.
    ///
    /// # Errors
    ///
    /// [`LatticeError::NotNested`] if the solution is not integral or does not
    /// reconstruct the shaping basis, meaning some generator of the second
    /// lattice is not in the first;
    /// [`LatticeError::Degenerate`] for a rank-deficient coding basis; and
    /// [`LatticeError::Range`] on a shape mismatch or an overflow.
    pub fn from_bases(coding: &Basis<T>, shaping: &Basis<T>) -> Result<Self, LatticeError> {
        if coding.ambient_dim() != shaping.ambient_dim() {
            return Err(RangeError::Shape {
                expected: coding.ambient_dim(),
                found: shaping.ambient_dim(),
            }
            .into());
        }
        let gram = coding.gram()?;
        let determinant = gram.det()?;
        if determinant.is_zero() {
            return Err(LatticeError::Degenerate);
        }
        let cofactors = adjugate(gram.as_matrix())?;
        let cross = shaping.as_matrix().mul(&coding.as_matrix().transpose()?)?;
        let scaled = cross.mul(&cofactors)?;

        let n = coding.count();
        let mut transform = IntMatrix::<T>::zeros(shaping.count(), n)?;
        for i in 0..shaping.count() {
            for j in 0..n {
                let value = scaled
                    .get(i, j)
                    .try_div_exact(determinant)
                    .map_err(|_| LatticeError::NotNested)?;
                transform.set(i, j, value);
            }
        }
        // Integral projected coordinates do not prove inclusion: a shaping
        // vector with a component outside the coding span still projects to
        // integral coordinates. Reconstruct the shaping basis exactly and
        // compare, so this type only ever describes a genuine sublattice.
        if &transform.mul(coding.as_matrix())? != shaping.as_matrix() {
            return Err(LatticeError::NotNested);
        }
        Self::new(gram, transform)
    }

    /// The coding lattice's Gram matrix.
    #[must_use]
    pub const fn coding(&self) -> &Gram<T> {
        &self.coding
    }

    /// The sublattice's generators in coding coordinates.
    #[must_use]
    pub const fn transform(&self) -> &IntMatrix<T> {
        &self.transform
    }

    /// The index `|Λ_c / Λ_s|`, which is the codebook size.
    #[must_use]
    pub const fn index(&self) -> T {
        self.index
    }

    /// The mixed radices the coset index is written in: the Hermite diagonal of
    /// the transform, whose product is [`index`](Nested::index).
    #[must_use]
    pub fn radices(&self) -> &[T] {
        &self.radices
    }

    /// The shaping lattice's Gram matrix, `T G Tᵀ`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Range`] on overflow.
    pub fn shaping_gram(&self) -> Result<Gram<T>, LatticeError> {
        let product = self
            .transform
            .mul(self.coding.as_matrix())?
            .mul(&self.transform.transpose()?)?;
        Gram::new(product)
    }

    /// Writes the `which`-th coset representative, in coding coordinates.
    ///
    /// Representatives are enumerated in mixed radix over
    /// [`radices`](Nested::radices), so this is a constant-time map from a
    /// message index to a codeword and never enumerates the codebook.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Range`] if `which` is at least the index, or if `out`
    /// is the wrong length.
    pub fn coset_representative(&self, which: u64, out: &mut [T]) -> Result<(), LatticeError> {
        let n = self.radices.len();
        if out.len() != n {
            return Err(RangeError::Shape {
                expected: n,
                found: out.len(),
            }
            .into());
        }
        // First pass: prove the index is in range before anything is written,
        // so a rejected call leaves `out` exactly as the caller left it.
        let mut remaining = which;
        for &radix in &self.radices {
            remaining = mixed_radix_digit(radix, remaining)?.1;
        }
        if remaining != 0 {
            return Err(RangeError::Dimension {
                requested: usize::try_from(which).unwrap_or(usize::MAX),
                max: usize::try_from(self.index.widen()).unwrap_or(usize::MAX),
            }
            .into());
        }
        // Second pass: write the digits.
        let mut remaining = which;
        for (slot, &radix) in out.iter_mut().zip(&self.radices) {
            let (digit, next) = mixed_radix_digit(radix, remaining)?;
            *slot = digit;
            remaining = next;
        }
        Ok(())
    }

    /// Every coset representative, in coding coordinates.
    ///
    /// Allocates the whole codebook; prefer
    /// [`coset_representative`](Nested::coset_representative) in an encoder.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Range`] if the index does not fit in memory.
    pub fn coset_representatives(&self) -> Result<Vec<Vec<T>>, LatticeError> {
        let count = u64::try_from(self.index.widen()).map_err(|_| RangeError::Overflow {
            op: crate::error::Op::Mul,
            width_bits: 64,
        })?;
        let n = self.radices.len();
        let too_large = || RangeError::Overflow {
            op: crate::error::Op::Mul,
            width_bits: usize::BITS,
        };
        let mut all = Vec::new();
        let mut buffer = Vec::new();
        // Every allocation below is fallible and maps to the same range
        // error: the outer reservation, the scratch buffer, and each
        // representative row. A codebook that cannot fit in memory is the
        // documented error, never an aborting allocation, and a reserved
        // `Vec` never reallocates while it fills within its reservation.
        all.try_reserve(usize::try_from(count).map_err(|_| too_large())?)
            .map_err(|_| too_large())?;
        buffer.try_reserve(n).map_err(|_| too_large())?;
        buffer.resize(n, T::ZERO);
        for which in 0..count {
            self.coset_representative(which, &mut buffer)?;
            let mut row = Vec::new();
            row.try_reserve(n).map_err(|_| too_large())?;
            row.extend_from_slice(&buffer);
            all.push(row);
        }
        Ok(all)
    }
}

/// One mixed-radix digit of an index: `(digit, remaining / radix)`.
///
/// A radix wider than `u64` can never be reached by a `u64` index — the digit
/// is the whole remainder and nothing can follow it — so wide radices are
/// handled directly rather than rejected. The digit is nonnegative and
/// strictly below `radix`, so narrowing it into `T` cannot lose information.
fn mixed_radix_digit<T: Int>(radix: T, remaining: u64) -> Result<(T, u64), RangeError> {
    let Some(modulus) = u64::try_from(radix.widen()).ok() else {
        return Ok((T::narrow(i128::from(remaining))?, 0));
    };
    let digit = T::narrow(i128::from(remaining % modulus))?;
    Ok((digit, remaining / modulus))
}

#[cfg(test)]
mod tests {
    use super::Nested;
    use crate::basis::Basis;
    use crate::basis::Gram;
    use crate::error::LatticeError;
    use crate::int::IntMatrix;
    use crate::named::{d_n_basis, e8, zn, zn_basis};

    fn scaled_transform(n: usize, factor: i64) -> IntMatrix<i64> {
        let mut m = IntMatrix::<i64>::zeros(n, n).unwrap();
        for i in 0..n {
            m.set(i, i, factor);
        }
        m
    }

    #[test]
    fn a_self_similar_pair_has_index_equal_to_the_scaling_power() {
        for n in 1..=6 {
            for factor in 2..=4i64 {
                let pair = Nested::new(zn(n).unwrap(), scaled_transform(n, factor)).unwrap();
                assert_eq!(pair.index(), factor.pow(u32::try_from(n).unwrap()));
                assert_eq!(pair.radices(), vec![factor; n]);
            }
        }
    }

    #[test]
    fn the_index_is_the_ratio_of_covolumes_not_determinants() {
        // det(Gram_s) = (det T)^2 * det(Gram_c), so the index is the square
        // root of the determinant ratio. Writing it unsquared is the easy
        // mistake, and this is what catches it.
        let pair = Nested::new(e8().unwrap(), scaled_transform(8, 3)).unwrap();
        let coding = pair.coding().det().unwrap();
        let shaping = pair.shaping_gram().unwrap().det().unwrap();
        let ratio = shaping / coding;
        assert_eq!(pair.index() * pair.index(), ratio);
        assert_eq!(pair.index(), 3i64.pow(8));
    }

    #[test]
    fn coset_representatives_are_distinct_and_exactly_index_many() {
        let pair = Nested::new(zn(3).unwrap(), scaled_transform(3, 3)).unwrap();
        let reps = pair.coset_representatives().unwrap();
        assert_eq!(reps.len(), 27);
        let mut sorted = reps.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 27, "representatives collide");
        for r in &reps {
            assert!(r.iter().all(|&c| (0..3).contains(&c)));
        }
    }

    #[test]
    fn a_singular_transform_is_not_a_finite_index_sublattice() {
        let mut t = IntMatrix::<i64>::zeros(2, 2).unwrap();
        t.set(0, 0, 1);
        t.set(1, 0, 2);
        assert_eq!(
            Nested::new(zn(2).unwrap(), t),
            Err(LatticeError::Degenerate)
        );
    }

    #[test]
    fn inclusion_is_checked_when_building_from_bases() {
        // D_n sits inside Z^n with index 2.
        let coding = zn_basis::<i64>(4).unwrap();
        let shaping = d_n_basis::<i64>(4).unwrap();
        let pair = Nested::from_bases(&coding, &shaping).unwrap();
        assert_eq!(pair.index(), 2);

        // The reverse is not an inclusion: Z^4 is not inside D_4.
        assert_eq!(
            Nested::from_bases(&shaping, &coding),
            Err(LatticeError::NotNested)
        );
    }

    #[test]
    fn shape_mismatches_are_rejected() {
        let t = IntMatrix::<i64>::zeros(2, 2).unwrap();
        assert!(Nested::new(zn(3).unwrap(), t).is_err());
        let pair = Nested::new(zn(2).unwrap(), scaled_transform(2, 2)).unwrap();
        let mut out = [0i64; 3];
        assert!(pair.coset_representative(0, &mut out).is_err());
    }

    #[test]
    fn an_out_of_range_message_is_rejected() {
        let pair = Nested::new(zn(2).unwrap(), scaled_transform(2, 2)).unwrap();
        let mut out = [0i64; 2];
        assert!(pair.coset_representative(3, &mut out).is_ok());
        assert!(pair.coset_representative(4, &mut out).is_err());
    }

    #[test]
    fn a_non_diagonal_sublattice_still_works() {
        // Generators (1, 2) and (0, 5): index 5, and the Hermite radices are
        // (1, 5) rather than the naive (1, 2)-ish reading of the rows.
        let t = IntMatrix::<i64>::from_rows(2, 2, &[1, 2, 0, 5]).unwrap();
        let pair = Nested::new(zn(2).unwrap(), t).unwrap();
        assert_eq!(pair.index(), 5);
        let reps = pair.coset_representatives().unwrap();
        assert_eq!(reps.len(), 5);
        // T Tᵀ with T = [[1,2],[0,5]], and its determinant is the index squared.
        let shaping = pair.shaping_gram().unwrap();
        assert_eq!(Gram::from_rows(2, &[5, 10, 10, 25]).unwrap(), shaping);
        assert_eq!(shaping.det().unwrap(), 25);
    }

    #[test]
    fn a_shaping_vector_outside_the_coding_span_is_rejected() {
        // The coding basis spans only the first coordinate, while the shaping
        // generator has a second component outside that span. Its projection
        // is integral, so before the reconstruction check this pair was
        // accepted with index 1.
        let coding = Basis::<i64>::from_rows(1, 2, &[1, 0]).unwrap();
        let shaping = Basis::<i64>::from_rows(1, 2, &[1, 1]).unwrap();
        assert_eq!(
            Nested::from_bases(&coding, &shaping),
            Err(LatticeError::NotNested)
        );
    }

    #[test]
    fn a_rejected_coset_index_leaves_the_output_untouched() {
        let pair = Nested::new(zn(2).unwrap(), scaled_transform(2, 2)).unwrap();
        let mut out = [7i64; 2];
        assert!(pair.coset_representative(4, &mut out).is_err());
        assert_eq!(out, [7, 7]);
    }

    #[test]
    fn wide_radices_reach_small_indices_exactly() {
        // A radix wider than `u64` used to overflow the radix conversion even
        // for indices that never touch it.
        let transform = IntMatrix::<i128>::from_rows(2, 2, &[2, 0, 0, 1i128 << 65]).unwrap();
        let pair = Nested::new(zn(2).unwrap(), transform).unwrap();
        let mut out = [0i128; 2];
        pair.coset_representative(1, &mut out).unwrap();
        assert_eq!(out, [1, 0]);
        pair.coset_representative(2, &mut out).unwrap();
        assert_eq!(out, [0, 1]);
    }

    #[test]
    fn the_u64_index_boundary_decomposes_exactly() {
        // index = 4 · 2^62 = 2^64: every `u64` index is in range, and the
        // boundary value itself splits into full digits.
        let transform = IntMatrix::<i128>::from_rows(2, 2, &[4, 0, 0, 1i128 << 62]).unwrap();
        let pair = Nested::new(zn(2).unwrap(), transform).unwrap();
        let mut out = [0i128; 2];
        pair.coset_representative(u64::MAX, &mut out).unwrap();
        assert_eq!(out, [3, (1i128 << 62) - 1]);
    }

    #[test]
    fn an_unrepresentable_codebook_is_an_error_not_an_abort() {
        let transform = IntMatrix::<i128>::from_rows(1, 1, &[1i128 << 63]).unwrap();
        let pair = Nested::new(zn(1).unwrap(), transform).unwrap();
        assert!(matches!(
            pair.coset_representatives(),
            Err(LatticeError::Range(_))
        ));
    }
}

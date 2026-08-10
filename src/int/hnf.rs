//! Row-style Hermite Normal Form with a retained unimodular transform.

use super::{Int, IntMatrix, det, div_nearest};
use crate::error::{RangeError, ReduceError};

/// A Hermite Normal Form together with the transform that produced it.
///
/// The identity `u * a == h` holds exactly and `|det u| == 1`, so the row space
/// over the integers — the lattice — is provably unchanged. That pair of facts
/// is a complete correctness certificate for the reduction, checkable without
/// reference to how it was computed, which is why the transform is a product
/// rather than an option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hnf<T: Int> {
    /// The reduced matrix in row Hermite Normal Form.
    pub h: IntMatrix<T>,
    /// The unimodular transform, with `u * a == h`.
    pub u: IntMatrix<T>,
    /// Number of nonzero rows, which is the rank of the input.
    pub rank: usize,
}

/// Reduces an integer matrix to row Hermite Normal Form.
///
/// The result satisfies:
///
/// - rows `0..rank` are nonzero and rows `rank..` are zero;
/// - pivots move strictly right as the row index increases;
/// - every pivot is strictly positive;
/// - every entry *above* a pivot lies in `[0, pivot)`.
///
/// Elimination is Euclidean rather than Bézout-based: the pivot is repeatedly
/// replaced by the smallest remainder in its column, using
/// nearest-integer quotients. Each round at least halves the pivot magnitude,
/// which keeps intermediates far smaller than the textbook two-by-two
/// extended-gcd formulation on the same input.
///
/// # Errors
///
/// [`RangeError::Overflow`] if an intermediate exceeds the element width.
/// Hermite reduction is subject to coefficient explosion — entries can grow far
/// beyond those of the input even when the final form is small — so this is a
/// real outcome on adversarial input, not a formality. The input is never
/// modified.
///
/// # Examples
///
/// ```
/// use lattica::int::{IntMatrix, hnf};
///
/// // Two bases of the same lattice reduce to the same Hermite form.
/// let a = IntMatrix::<i64>::from_rows(2, 2, &[2, 0, 0, 3])?;
/// let b = IntMatrix::<i64>::from_rows(2, 2, &[2, 3, 2, 6])?;
/// assert_eq!(hnf(&a)?.h, hnf(&b)?.h);
/// # Ok::<(), lattica::RangeError>(())
/// ```
pub fn hnf<T: Int>(a: &IntMatrix<T>) -> Result<Hnf<T>, RangeError> {
    let rows = a.rows();
    let cols = a.cols();

    let mut h = a.clone();
    let mut u = IntMatrix::<T>::identity(rows)?;
    let mut pivot_row = 0usize;

    for col in 0..cols {
        if pivot_row >= rows {
            break;
        }

        // Drive the column below `pivot_row` to a single nonzero entry. Each
        // pass either finishes the column or strictly reduces the pivot.
        loop {
            let Some(smallest) = smallest_nonzero(&h, pivot_row, rows, col)? else {
                break;
            };
            h.swap_rows(pivot_row, smallest);
            u.swap_rows(pivot_row, smallest);

            let pivot = h.get(pivot_row, col);
            let mut cleared = true;
            for i in pivot_row + 1..rows {
                let entry = h.get(i, col);
                if entry.is_zero() {
                    continue;
                }
                let q = div_nearest(entry, pivot)?;
                h.row_sub_mul(i, pivot_row, q)?;
                u.row_sub_mul(i, pivot_row, q)?;
                if !h.get(i, col).is_zero() {
                    cleared = false;
                }
            }
            if cleared {
                break;
            }
        }

        if h.get(pivot_row, col).is_zero() {
            continue;
        }

        if h.get(pivot_row, col).is_negative() {
            h.negate_row(pivot_row)?;
            u.negate_row(pivot_row)?;
        }

        // Reduce above the pivot into `[0, pivot)`. Rows above have zeros in
        // every earlier pivot column, so this cannot disturb them.
        let pivot = h.get(pivot_row, col);
        for i in 0..pivot_row {
            let entry = h.get(i, col);
            if entry.is_zero() {
                continue;
            }
            let q = entry.try_div_floor(pivot)?;
            h.row_sub_mul(i, pivot_row, q)?;
            u.row_sub_mul(i, pivot_row, q)?;
        }

        pivot_row += 1;
    }

    Ok(Hnf {
        h,
        u,
        rank: pivot_row,
    })
}

/// Index of the row in `start..end` whose entry in `col` is nonzero and
/// smallest in absolute value.
fn smallest_nonzero<T: Int>(
    m: &IntMatrix<T>,
    start: usize,
    end: usize,
    col: usize,
) -> Result<Option<usize>, RangeError> {
    let mut best: Option<(usize, T)> = None;
    for i in start..end {
        let v = m.get(i, col);
        if v.is_zero() {
            continue;
        }
        let magnitude = v.try_abs()?;
        if best.is_none_or(|(_, b)| magnitude < b) {
            best = Some((i, magnitude));
        }
    }
    Ok(best.map(|(i, _)| i))
}

/// Hermite Normal Form of a square nonsingular matrix, computed modulo its
/// determinant.
///
/// Returns `h` only — there is no transform on this path, which is the whole
/// point.
///
/// # Why this exists next to [`hnf`]
///
/// Euclidean Hermite reduction suffers a well-known coefficient explosion: on a
/// random 8-by-8 basis with entries below 10, intermediates already exceed
/// `i64`, and by dimension 12 they exceed `i128` — even though the *answer*
/// stays small. Carrying the unimodular transform is what forces those
/// intermediates to be materialized.
///
/// Dropping the transform admits a much better algorithm. Since
/// `d·Z^n ⊆ Λ` whenever `d` is the lattice determinant, the rows of `d·I` are
/// themselves lattice vectors, so *every intermediate entry may be reduced
/// modulo `d` at any time* without changing the lattice. Entries stay in
/// `[0, d)` and intermediates below `d²`, independent of dimension. For the
/// lattices this crate exists to serve — `Z^n`, `D_n`, `A_n`, `E_8`, with
/// determinants of 1, 4, `n+1`, and 1 — that bound is trivial at any dimension.
///
/// This is the Domich–Kannan–Trotter modular approach, in the stacked form
/// `[A; d·I]`.
///
/// # Errors
///
/// - [`ReduceError::Singular`] if the matrix is not invertible over the
///   rationals, since there is then no determinant to reduce by.
/// - [`ReduceError::Range`] carrying [`RangeError::Shape`] if the matrix is not
///   square, or [`RangeError::Overflow`] if `d²` does not fit the element
///   width. That is a far weaker requirement than [`hnf`]'s.
///
/// # Examples
///
/// ```
/// use lattica::int::{IntMatrix, hnf_mod_det};
///
/// // The D_2 lattice: {x in Z^2 : x0 + x1 even}, determinant 2.
/// let a = IntMatrix::<i64>::from_rows(2, 2, &[1, 1, 1, -1])?;
/// let h = hnf_mod_det(&a)?;
/// assert_eq!(h.row(0), &[1, 1]);
/// assert_eq!(h.row(1), &[0, 2]);
/// # Ok::<(), lattica::ReduceError>(())
/// ```
pub fn hnf_mod_det<T: Int>(a: &IntMatrix<T>) -> Result<IntMatrix<T>, ReduceError> {
    let dim = a.rows();
    if a.cols() != dim {
        return Err(RangeError::Shape {
            expected: dim,
            found: a.cols(),
        }
        .into());
    }
    if dim == 0 {
        return Ok(a.clone());
    }

    let modulus = det(a)?.try_abs().map_err(ReduceError::from)?;
    if modulus.is_zero() {
        return Err(ReduceError::Singular);
    }

    // Stack `A` over `d·I` where `d` is the determinant. The added rows are
    // lattice vectors, so they change nothing, and their presence is what
    // licenses reduction modulo `d`.
    let mut work = IntMatrix::<T>::zeros(2 * dim, dim).map_err(ReduceError::from)?;
    // Those rows carry `d` itself, not `d mod d`: a zero row would contribute
    // nothing, and `d` is precisely the generator being added.
    for i in 0..dim {
        for j in 0..dim {
            work.set(i, j, reduce_mod(a.get(i, j), modulus)?);
        }
        work.set(dim + i, i, modulus);
    }

    for col in 0..dim {
        loop {
            let Some(smallest) = smallest_nonzero(&work, col, 2 * dim, col)? else {
                // Unreachable while `d·e_col` is in the lattice; a zero column
                // would mean the determinant lied.
                return Err(ReduceError::Singular);
            };
            work.swap_rows(col, smallest);

            let pivot = work.get(col, col);
            let mut cleared = true;
            for i in col + 1..2 * dim {
                let entry = work.get(i, col);
                if entry.is_zero() {
                    continue;
                }
                // Entries are non-negative here, so truncation is the floor and
                // the remainder descends in `[0, pivot)`.
                let q = entry.try_div_trunc(pivot)?;
                work.row_sub_mul(i, col, q)?;
                reduce_row_mod(&mut work, i, modulus)?;
                if !work.get(i, col).is_zero() {
                    cleared = false;
                }
            }
            if cleared {
                break;
            }
        }

        let pivot = work.get(col, col);
        for i in 0..col {
            let entry = work.get(i, col);
            if entry.is_zero() {
                continue;
            }
            let q = entry.try_div_floor(pivot)?;
            work.row_sub_mul(i, col, q)?;
            reduce_row_mod(&mut work, i, modulus)?;
        }
    }

    let mut h = IntMatrix::<T>::zeros(dim, dim).map_err(ReduceError::from)?;
    for i in 0..dim {
        for j in 0..dim {
            h.set(i, j, work.get(i, j));
        }
    }
    Ok(h)
}

/// The representative of `v` modulo `d` in `[0, d)`.
fn reduce_mod<T: Int>(v: T, d: T) -> Result<T, RangeError> {
    let r = v.try_rem_trunc(d)?;
    if r.is_negative() { r.try_add(d) } else { Ok(r) }
}

/// Reduces every entry of one row into `[0, d)`.
fn reduce_row_mod<T: Int>(m: &mut IntMatrix<T>, row: usize, d: T) -> Result<(), RangeError> {
    for j in 0..m.cols() {
        let v = m.get(row, j);
        if v.is_negative() || v >= d {
            m.set(row, j, reduce_mod(v, d)?);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::hnf;
    use crate::int::IntMatrix;

    #[test]
    fn identity_is_its_own_hermite_form() {
        let a = IntMatrix::<i64>::identity(4).unwrap();
        let r = hnf(&a).unwrap();
        assert_eq!(r.h, a);
        assert_eq!(r.rank, 4);
    }

    #[test]
    fn a_unimodular_basis_reduces_to_the_identity() {
        // det = 1, so this generates all of Z^2.
        let a = IntMatrix::<i64>::from_rows(2, 2, &[3, 5, 1, 2]).unwrap();
        let r = hnf(&a).unwrap();
        assert_eq!(r.h, IntMatrix::identity(2).unwrap());
        assert_eq!(r.u.mul(&a).unwrap(), r.h);
        assert_eq!(r.u.det().unwrap().abs(), 1);
    }

    #[test]
    fn rank_deficient_input_leaves_trailing_zero_rows() {
        // Row 2 is row 0 plus row 1.
        let a = IntMatrix::<i64>::from_rows(3, 3, &[1, 2, 3, 4, 5, 6, 5, 7, 9]).unwrap();
        let r = hnf(&a).unwrap();
        assert_eq!(r.rank, 2);
        assert_eq!(r.h.row(2), &[0, 0, 0]);
        assert_eq!(r.u.mul(&a).unwrap(), r.h);
    }

    #[test]
    fn entries_above_a_pivot_are_reduced_into_range() {
        let a = IntMatrix::<i64>::from_rows(2, 2, &[1, 17, 0, 5]).unwrap();
        let r = hnf(&a).unwrap();
        // Pivots are (0,0) = 1 and (1,1) = 5; 17 reduces to 17 mod 5 = 2.
        assert_eq!(r.h.row(0), &[1, 2]);
        assert_eq!(r.h.row(1), &[0, 5]);
    }
}

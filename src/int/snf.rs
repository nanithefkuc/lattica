//! Smith Normal Form: the invariant factors of an integer matrix.

use super::{Int, IntMatrix, div_nearest, gcd, lcm};
use crate::error::RangeError;

/// The nonzero invariant factors of an integer matrix, in divisibility order.
///
/// Returns `d_1 | d_2 | ... | d_r`, all strictly positive, where `r` is the
/// rank. For a square nonsingular matrix the product of the factors is
/// `|det|`, and for a lattice basis the factors describe the quotient group
/// `Z^n / Λ` — which is exactly the structure a nested-lattice construction
/// needs to enumerate cosets.
///
/// The transforms are not returned. Nothing in this crate consumes them, and
/// computing them roughly doubles the intermediate growth of an algorithm that
/// is already the most explosive one here.
///
/// # Errors
///
/// [`RangeError::Overflow`] if an intermediate exceeds the element width.
/// Diagonalization alternates row and column elimination, so intermediates can
/// exceed those of Hermite reduction on the same input. The input is never
/// modified.
///
/// # Examples
///
/// ```
/// use lattica::int::{IntMatrix, invariant_factors};
///
/// // Z^2 / Λ is cyclic of order 6 here, not Z/2 x Z/3.
/// let m = IntMatrix::<i64>::from_rows(2, 2, &[2, 0, 0, 3])?;
/// assert_eq!(invariant_factors(&m)?, [1, 6]);
/// # Ok::<(), lattica::RangeError>(())
/// ```
pub fn invariant_factors<T: Int>(a: &IntMatrix<T>) -> Result<Vec<T>, RangeError> {
    let rows = a.rows();
    let cols = a.cols();
    let mut m = a.clone();
    let mut factors: Vec<T> = Vec::new();

    for t in 0..rows.min(cols) {
        let Some((pi, pj)) = smallest_nonzero(&m, t, rows, cols)? else {
            break;
        };
        m.swap_rows(t, pi);
        m.swap_cols(t, pj);

        // Alternate clearing the pivot column and the pivot row. Clearing one
        // can dirty the other, so this repeats; it terminates because every
        // pass that fails to clear replaces the pivot with a strictly smaller
        // magnitude, and a positive integer cannot descend forever.
        loop {
            if clear_below(&mut m, t, rows)? {
                continue;
            }
            if clear_right(&mut m, t, cols)? {
                continue;
            }
            break;
        }

        if m.get(t, t).is_negative() {
            m.negate_row(t)?;
        }
        factors.push(m.get(t, t));
    }

    enforce_divisibility(&mut factors)?;
    Ok(factors)
}

/// Eliminates the pivot column below `t`. Returns `true` if the pivot changed,
/// meaning the caller must restart.
fn clear_below<T: Int>(m: &mut IntMatrix<T>, t: usize, rows: usize) -> Result<bool, RangeError> {
    for i in t + 1..rows {
        let entry = m.get(i, t);
        if entry.is_zero() {
            continue;
        }
        let q = div_nearest(entry, m.get(t, t))?;
        m.row_sub_mul(i, t, q)?;
        if !m.get(i, t).is_zero() {
            m.swap_rows(t, i);
            return Ok(true);
        }
    }
    Ok(false)
}

/// Eliminates the pivot row right of `t`. Returns `true` if the pivot changed.
fn clear_right<T: Int>(m: &mut IntMatrix<T>, t: usize, cols: usize) -> Result<bool, RangeError> {
    for j in t + 1..cols {
        let entry = m.get(t, j);
        if entry.is_zero() {
            continue;
        }
        let q = div_nearest(entry, m.get(t, t))?;
        m.col_sub_mul(j, t, q)?;
        if !m.get(t, j).is_zero() {
            m.swap_cols(t, j);
            return Ok(true);
        }
    }
    Ok(false)
}

/// Position of the nonzero entry of least absolute value in the trailing
/// submatrix anchored at `(start, start)`.
fn smallest_nonzero<T: Int>(
    m: &IntMatrix<T>,
    start: usize,
    rows: usize,
    cols: usize,
) -> Result<Option<(usize, usize)>, RangeError> {
    let mut best: Option<((usize, usize), T)> = None;
    for i in start..rows {
        for j in start..cols {
            let v = m.get(i, j);
            if v.is_zero() {
                continue;
            }
            let magnitude = v.try_abs()?;
            if best.is_none_or(|(_, b)| magnitude < b) {
                best = Some(((i, j), magnitude));
            }
        }
    }
    Ok(best.map(|(pos, _)| pos))
}

/// Rewrites a diagonal into divisibility order.
///
/// Diagonalization gives *a* diagonal form, not *the* Smith form: the entries
/// need not divide one another. Replacing an adjacent pair `(a, b)` with
/// `(gcd(a, b), lcm(a, b))` preserves both the product and the underlying
/// group, and repeating to a fixpoint yields the invariant factors.
fn enforce_divisibility<T: Int>(factors: &mut [T]) -> Result<(), RangeError> {
    if factors.len() < 2 {
        return Ok(());
    }
    loop {
        let mut changed = false;
        for i in 0..factors.len() - 1 {
            let (a, b) = (factors[i], factors[i + 1]);
            if b.try_rem_trunc(a)?.is_zero() {
                continue;
            }
            factors[i] = gcd(a, b)?;
            factors[i + 1] = lcm(a, b)?;
            changed = true;
        }
        if !changed {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::invariant_factors;
    use crate::int::IntMatrix;

    #[test]
    fn identity_has_all_unit_factors() {
        let m = IntMatrix::<i64>::identity(3).unwrap();
        assert_eq!(invariant_factors(&m).unwrap(), vec![1, 1, 1]);
    }

    #[test]
    fn coprime_diagonal_becomes_cyclic() {
        // Z/2 x Z/3 is cyclic of order 6, so the Smith form is diag(1, 6).
        let m = IntMatrix::<i64>::from_rows(2, 2, &[2, 0, 0, 3]).unwrap();
        assert_eq!(invariant_factors(&m).unwrap(), vec![1, 6]);
    }

    #[test]
    fn non_coprime_diagonal_is_already_in_order() {
        let m = IntMatrix::<i64>::from_rows(2, 2, &[2, 0, 0, 4]).unwrap();
        assert_eq!(invariant_factors(&m).unwrap(), vec![2, 4]);
    }

    #[test]
    fn rank_deficient_input_yields_fewer_factors() {
        let m = IntMatrix::<i64>::from_rows(3, 3, &[1, 2, 3, 4, 5, 6, 5, 7, 9]).unwrap();
        assert_eq!(invariant_factors(&m).unwrap().len(), 2);
    }

    #[test]
    fn textbook_three_by_three() {
        // Invariant factors from the minor-gcd characterisation, computed by
        // hand: D1 = 2, D2 = 12, D3 = |det| = 144, so d = (2, 6, 12).
        let m = IntMatrix::<i64>::from_rows(3, 3, &[2, 4, 4, -6, 6, 12, 10, -4, -16]).unwrap();
        assert_eq!(invariant_factors(&m).unwrap(), vec![2, 6, 12]);
        assert_eq!(m.det().unwrap(), -144);
    }
}

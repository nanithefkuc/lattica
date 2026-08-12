//! Exact determinant by fraction-free Gaussian elimination.

use super::{Int, IntMatrix};
use crate::error::RangeError;

/// The exact determinant of a square integer matrix.
///
/// Uses Bareiss's multistep integer-preserving elimination. Every division in
/// the inner loop is exact by Sylvester's determinant identity — the quotient
/// is itself a minor of the original matrix — so the whole computation stays in
/// the integers with no rational intermediates and no growth beyond the minors
/// themselves.
///
/// The determinant of the empty matrix is `1`, the empty product.
///
/// # Errors
///
/// - [`RangeError::Shape`] if the matrix is not square.
/// - [`RangeError::Overflow`] if an intermediate exceeds the element width.
///   Intermediates are minors of the input, so this depends on entry magnitude
///   as much as on dimension. Widen `T` or reduce the basis first; there is no
///   arbitrary-precision fallback by design.
///
/// # Examples
///
/// ```
/// use lattica::int::{IntMatrix, det};
///
/// let m = IntMatrix::<i64>::from_rows(3, 3, &[2, 0, 0, 0, 3, 0, 0, 0, 5])?;
/// assert_eq!(det(&m)?, 30);
/// # Ok::<(), lattica::RangeError>(())
/// ```
pub fn det<T: Int>(a: &IntMatrix<T>) -> Result<T, RangeError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(RangeError::Shape {
            expected: n,
            found: a.cols(),
        });
    }
    if n == 0 {
        return Ok(T::ONE);
    }

    let mut m: Vec<T> = a.as_slice().to_vec();
    let mut negated = false;
    let mut prev = T::ONE;

    for k in 0..n - 1 {
        if m[k * n + k].is_zero() {
            let Some(swap) = (k + 1..n).find(|&i| !m[i * n + k].is_zero()) else {
                // A zero column below the diagonal means a zero determinant;
                // the remaining elimination would divide by that zero pivot.
                return Ok(T::ZERO);
            };
            for j in 0..n {
                m.swap(k * n + j, swap * n + j);
            }
            negated = !negated;
        }

        let pivot = m[k * n + k];
        for i in k + 1..n {
            let leading = m[i * n + k];
            for j in k + 1..n {
                let cross = m[i * n + j]
                    .try_mul(pivot)?
                    .try_sub(leading.try_mul(m[k * n + j])?)?;
                m[i * n + j] = cross.try_div_exact(prev)?;
            }
            m[i * n + k] = T::ZERO;
        }
        prev = pivot;
    }

    let d = m[(n - 1) * n + (n - 1)];
    if negated { d.try_neg() } else { Ok(d) }
}

/// The adjugate: the transpose of the cofactor matrix, satisfying
/// `adj(A) · A == det(A) · I`.
///
/// For an invertible `A` this is `det(A) · A⁻¹`, which is how an exact inverse
/// is represented in this crate: the pair `(adj(A), det(A))` carries the same
/// information as `A⁻¹` with no rationals, so identities that would otherwise
/// need fractions stay checkable in integers.
///
/// Uses fraction-free Gauss–Jordan elimination, sharing one decomposition
/// across every identity right-hand side. Singular inputs and fixed-width
/// recurrences that exceed their budget fall back to direct cofactors so the
/// accepted input domain does not shrink.
///
/// # Errors
///
/// [`RangeError::Shape`] if the matrix is not square, and
/// [`RangeError::Overflow`] if a cofactor exceeds the element width.
pub fn adjugate<T: Int>(a: &IntMatrix<T>) -> Result<IntMatrix<T>, RangeError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(RangeError::Shape {
            expected: n,
            found: a.cols(),
        });
    }
    if n == 0 {
        return IntMatrix::<T>::zeros(0, 0);
    }
    if n == 1 {
        return IntMatrix::<T>::identity(1);
    }
    match fraction_free_adjugate(a) {
        Ok(Some(out)) => Ok(out),
        Ok(None) | Err(_) => adjugate_cofactors(a),
    }
}

fn fraction_free_adjugate<T: Int>(
    a: &IntMatrix<T>,
) -> Result<Option<IntMatrix<T>>, RangeError> {
    let n = a.rows();
    let mut left = a.clone();
    let mut right = IntMatrix::<T>::identity(n)?;
    let mut previous = T::ONE;
    let mut negated = false;

    for k in 0..n {
        let Some(pivot_row) = (k..n).find(|&row| !left.get(row, k).is_zero()) else {
            return Ok(None);
        };
        if pivot_row != k {
            left.swap_rows(k, pivot_row);
            right.swap_rows(k, pivot_row);
            negated = !negated;
        }
        let pivot = left.get(k, k);
        for row in 0..n {
            if row == k {
                continue;
            }
            let leading = left.get(row, k);
            for column in 0..n {
                if column != k {
                    let value = left
                        .get(row, column)
                        .try_mul(pivot)?
                        .try_sub(leading.try_mul(left.get(k, column))?)?
                        .try_div_exact(previous)?;
                    left.set(row, column, value);
                }
                let value = right
                    .get(row, column)
                    .try_mul(pivot)?
                    .try_sub(leading.try_mul(right.get(k, column))?)?
                    .try_div_exact(previous)?;
                right.set(row, column, value);
            }
            left.set(row, k, T::ZERO);
        }
        previous = pivot;
    }

    if negated {
        for row in 0..n {
            right.negate_row(row)?;
        }
    }
    Ok(Some(right))
}

fn adjugate_cofactors<T: Int>(a: &IntMatrix<T>) -> Result<IntMatrix<T>, RangeError> {
    let n = a.rows();
    let mut out = IntMatrix::<T>::zeros(n, n)?;
    let mut minor = IntMatrix::<T>::zeros(n - 1, n - 1)?;
    for i in 0..n {
        for j in 0..n {
            // Entry (i, j) of the adjugate is the (j, i) cofactor.
            for (r, row) in (0..n).filter(|&r| r != j).enumerate() {
                for (c, col) in (0..n).filter(|&c| c != i).enumerate() {
                    minor.set(r, c, a.get(row, col));
                }
            }
            let value = det(&minor)?;
            out.set(
                i,
                j,
                if (i + j) % 2 == 0 {
                    value
                } else {
                    value.try_neg()?
                },
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::det;
    use crate::int::IntMatrix;

    #[test]
    fn empty_determinant_is_the_empty_product() {
        let m = IntMatrix::<i64>::zeros(0, 0).unwrap();
        assert_eq!(det(&m).unwrap(), 1);
    }

    #[test]
    fn known_small_determinants() {
        let m = IntMatrix::<i64>::from_rows(2, 2, &[1, 2, 3, 4]).unwrap();
        assert_eq!(det(&m).unwrap(), -2);

        // Singular: the third row is the sum of the first two.
        let m = IntMatrix::<i64>::from_rows(3, 3, &[1, 2, 3, 4, 5, 6, 5, 7, 9]).unwrap();
        assert_eq!(det(&m).unwrap(), 0);

        // Requires a pivot swap on the very first column.
        let m = IntMatrix::<i64>::from_rows(3, 3, &[0, 1, 0, 1, 0, 0, 0, 0, 1]).unwrap();
        assert_eq!(det(&m).unwrap(), -1);
    }

    #[test]
    fn rejects_a_non_square_matrix() {
        let m = IntMatrix::<i64>::zeros(2, 3).unwrap();
        assert!(det(&m).is_err());
    }
}

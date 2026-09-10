//! A dense row-major integer matrix with checked geometry.

use super::Int;
use crate::error::RangeError;

/// The largest matrix dimension this crate will allocate.
///
/// Exact integer elimination is superlinear in both time and intermediate
/// magnitude, so a very large integer matrix is a mistake rather than a
/// workload. Lattice bases of interest here are far smaller: named lattices
/// reach dimension 24, and an LDLC parity matrix is handled through its sparse
/// support rather than as a dense matrix.
pub const MAX_DIM: usize = 1024;

/// A dense, row-major matrix of exact integers.
///
/// Every constructor validates its geometry against [`MAX_DIM`] before
/// allocating, and every arithmetic method is checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntMatrix<T: Int> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Int> IntMatrix<T> {
    fn check_dims(rows: usize, cols: usize) -> Result<usize, RangeError> {
        if rows > MAX_DIM {
            return Err(RangeError::Dimension {
                requested: rows,
                max: MAX_DIM,
            });
        }
        if cols > MAX_DIM {
            return Err(RangeError::Dimension {
                requested: cols,
                max: MAX_DIM,
            });
        }
        // Both factors are bounded by MAX_DIM, so the product cannot overflow
        // usize on any target this crate supports (16-bit is not one).
        Ok(rows * cols)
    }

    /// Creates a `rows` by `cols` matrix of zeros.
    ///
    /// # Errors
    ///
    /// [`RangeError::Dimension`] if either dimension exceeds [`MAX_DIM`].
    pub fn zeros(rows: usize, cols: usize) -> Result<Self, RangeError> {
        let len = Self::check_dims(rows, cols)?;
        Ok(Self {
            rows,
            cols,
            data: vec![T::ZERO; len],
        })
    }

    /// Creates the `n` by `n` identity matrix.
    ///
    /// # Errors
    ///
    /// [`RangeError::Dimension`] if `n` exceeds [`MAX_DIM`].
    pub fn identity(n: usize) -> Result<Self, RangeError> {
        let mut m = Self::zeros(n, n)?;
        for i in 0..n {
            m.data[i * n + i] = T::ONE;
        }
        Ok(m)
    }

    /// Creates a matrix from row-major data.
    ///
    /// # Errors
    ///
    /// [`RangeError::Dimension`] if either dimension exceeds [`MAX_DIM`], and
    /// [`RangeError::Shape`] if `data` is not exactly `rows * cols` long.
    pub fn from_rows(rows: usize, cols: usize, data: &[T]) -> Result<Self, RangeError> {
        let len = Self::check_dims(rows, cols)?;
        if data.len() != len {
            return Err(RangeError::Shape {
                expected: len,
                found: data.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            data: data.to_vec(),
        })
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Returns `true` if the matrix is square.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The entry at `(row, col)`.
    ///
    /// # Panics
    ///
    /// If either index is out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> T {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Sets the entry at `(row, col)`.
    ///
    /// # Panics
    ///
    /// If either index is out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Borrows one row.
    ///
    /// # Panics
    ///
    /// If `row` is out of bounds.
    #[must_use]
    pub fn row(&self, row: usize) -> &[T] {
        assert!(row < self.rows, "row index out of bounds");
        &self.data[row * self.cols..(row + 1) * self.cols]
    }

    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [T] {
        assert!(row < self.rows, "row index out of bounds");
        &mut self.data[row * self.cols..(row + 1) * self.cols]
    }

    pub(crate) fn copy_column_from_slice(&mut self, col: usize, values: &[T]) {
        assert!(col < self.cols, "column index out of bounds");
        assert_eq!(values.len(), self.rows, "column length mismatch");
        for (row, &value) in values.iter().enumerate() {
            self.data[row * self.cols + col] = value;
        }
    }

    /// Borrows the whole buffer in row-major order.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Exchanges two rows. A no-op when they are the same row.
    ///
    /// # Panics
    ///
    /// If either index is out of bounds.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        assert!(a < self.rows && b < self.rows, "row index out of bounds");
        if a == b {
            return;
        }
        for j in 0..self.cols {
            self.data.swap(a * self.cols + j, b * self.cols + j);
        }
    }

    /// Exchanges two columns. A no-op when they are the same column.
    ///
    /// # Panics
    ///
    /// If either index is out of bounds.
    pub fn swap_cols(&mut self, a: usize, b: usize) {
        assert!(a < self.cols && b < self.cols, "column index out of bounds");
        if a == b {
            return;
        }
        for i in 0..self.rows {
            self.data.swap(i * self.cols + a, i * self.cols + b);
        }
    }

    /// `row[target] -= factor * row[source]`.
    ///
    /// Leaves the matrix untouched when `factor` is zero or the rows coincide.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if any entry overflows. The update is
    /// transactional: a rejected call restores every entry it had already
    /// written, so the matrix is exactly as it was before the call.
    ///
    /// # Panics
    ///
    /// If either index is out of bounds.
    pub fn row_sub_mul(
        &mut self,
        target: usize,
        source: usize,
        factor: T,
    ) -> Result<(), RangeError> {
        assert!(
            target < self.rows && source < self.rows,
            "row index out of bounds"
        );
        if factor.is_zero() || target == source {
            return Ok(());
        }
        for column in 0..self.cols {
            let s = self.data[source * self.cols + column];
            if s.is_zero() {
                continue;
            }
            let product = match factor.try_mul(s) {
                Ok(product) => product,
                Err(error) => {
                    self.undo_row_sub_mul(target, source, factor, column)?;
                    return Err(error);
                }
            };
            let t = self.data[target * self.cols + column];
            match t.try_sub(product) {
                Ok(updated) => self.data[target * self.cols + column] = updated,
                Err(error) => {
                    self.undo_row_sub_mul(target, source, factor, column)?;
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Restores `row[target] += factor * row[source]` over the columns before
    /// `exclusive_end` after a failed update.
    ///
    /// The inverse cannot overflow: every restored entry was representable
    /// before the update, the source row is unchanged, and checked integer
    /// arithmetic is an exact bijection on the values involved.
    fn undo_row_sub_mul(
        &mut self,
        target: usize,
        source: usize,
        factor: T,
        exclusive_end: usize,
    ) -> Result<(), RangeError> {
        for column in 0..exclusive_end {
            let s = self.data[source * self.cols + column];
            if s.is_zero() {
                continue;
            }
            let updated = self.data[target * self.cols + column];
            self.data[target * self.cols + column] = updated.try_add(factor.try_mul(s)?)?;
        }
        Ok(())
    }

    /// `col[target] -= factor * col[source]`.
    ///
    /// Leaves the matrix untouched when `factor` is zero or the columns
    /// coincide.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if any entry overflows. The update is
    /// transactional: a rejected call restores every entry it had already
    /// written, so the matrix is exactly as it was before the call.
    ///
    /// # Panics
    ///
    /// If either index is out of bounds.
    pub fn col_sub_mul(
        &mut self,
        target: usize,
        source: usize,
        factor: T,
    ) -> Result<(), RangeError> {
        assert!(
            target < self.cols && source < self.cols,
            "column index out of bounds"
        );
        if factor.is_zero() || target == source {
            return Ok(());
        }
        for row in 0..self.rows {
            let s = self.data[row * self.cols + source];
            if s.is_zero() {
                continue;
            }
            let product = match factor.try_mul(s) {
                Ok(product) => product,
                Err(error) => {
                    self.undo_col_sub_mul(target, source, factor, row)?;
                    return Err(error);
                }
            };
            let t = self.data[row * self.cols + target];
            match t.try_sub(product) {
                Ok(updated) => self.data[row * self.cols + target] = updated,
                Err(error) => {
                    self.undo_col_sub_mul(target, source, factor, row)?;
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Restores `col[target] += factor * col[source]` over the rows before
    /// `exclusive_end` after a failed update; see [`Self::undo_row_sub_mul`]
    /// for why the inverse cannot overflow.
    fn undo_col_sub_mul(
        &mut self,
        target: usize,
        source: usize,
        factor: T,
        exclusive_end: usize,
    ) -> Result<(), RangeError> {
        for row in 0..exclusive_end {
            let s = self.data[row * self.cols + source];
            if s.is_zero() {
                continue;
            }
            let updated = self.data[row * self.cols + target];
            self.data[row * self.cols + target] = updated.try_add(factor.try_mul(s)?)?;
        }
        Ok(())
    }

    /// Negates every entry of one row.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if an entry is the type minimum. The update is
    /// transactional: a rejected call restores every entry it had already
    /// negated.
    ///
    /// # Panics
    ///
    /// If `row` is out of bounds.
    pub fn negate_row(&mut self, row: usize) -> Result<(), RangeError> {
        assert!(row < self.rows, "row index out of bounds");
        for column in 0..self.cols {
            match self.data[row * self.cols + column].try_neg() {
                Ok(negated) => self.data[row * self.cols + column] = negated,
                Err(error) => {
                    self.undo_negate_row(row, column)?;
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Restores the negated prefix before `exclusive_end`. The inverse of a
    /// succeeded negation is always representable.
    fn undo_negate_row(&mut self, row: usize, exclusive_end: usize) -> Result<(), RangeError> {
        for column in 0..exclusive_end {
            self.data[row * self.cols + column] = self.data[row * self.cols + column].try_neg()?;
        }
        Ok(())
    }

    /// Negates every entry of one column.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if an entry is the type minimum. The update is
    /// transactional: a rejected call restores every entry it had already
    /// negated.
    ///
    /// # Panics
    ///
    /// If `col` is out of bounds.
    pub fn negate_col(&mut self, col: usize) -> Result<(), RangeError> {
        assert!(col < self.cols, "column index out of bounds");
        for row in 0..self.rows {
            match self.data[row * self.cols + col].try_neg() {
                Ok(negated) => self.data[row * self.cols + col] = negated,
                Err(error) => {
                    self.undo_negate_col(col, row)?;
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Restores the negated prefix before `exclusive_end`; see
    /// [`Self::undo_negate_row`] for why the inverse always succeeds.
    fn undo_negate_col(&mut self, col: usize, exclusive_end: usize) -> Result<(), RangeError> {
        for row in 0..exclusive_end {
            self.data[row * self.cols + col] = self.data[row * self.cols + col].try_neg()?;
        }
        Ok(())
    }

    /// Matrix product `self * rhs`.
    ///
    /// # Errors
    ///
    /// [`RangeError::Shape`] if the inner dimensions disagree, and
    /// [`RangeError::Overflow`] if any accumulation overflows.
    pub fn mul(&self, rhs: &Self) -> Result<Self, RangeError> {
        if self.cols != rhs.rows {
            return Err(RangeError::Shape {
                expected: self.cols,
                found: rhs.rows,
            });
        }
        let mut out = Self::zeros(self.rows, rhs.cols)?;
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k];
                if a.is_zero() {
                    continue;
                }
                for j in 0..rhs.cols {
                    let b = rhs.data[k * rhs.cols + j];
                    if b.is_zero() {
                        continue;
                    }
                    let acc = out.data[i * out.cols + j];
                    out.data[i * out.cols + j] = acc.try_add(a.try_mul(b)?)?;
                }
            }
        }
        Ok(out)
    }

    /// The transpose.
    ///
    /// # Errors
    ///
    /// [`RangeError::Dimension`] cannot occur for an existing matrix, but the
    /// allocation path is shared with the checked constructors.
    pub fn transpose(&self) -> Result<Self, RangeError> {
        let mut out = Self::zeros(self.cols, self.rows)?;
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[j * out.cols + i] = self.data[i * self.cols + j];
            }
        }
        Ok(out)
    }

    /// The exact determinant, by fraction-free elimination.
    ///
    /// See [`det`](super::det) for the algorithm and its error conditions.
    ///
    /// # Errors
    ///
    /// [`RangeError::Shape`] if the matrix is not square, and
    /// [`RangeError::Overflow`] if an intermediate exceeds the element width.
    pub fn det(&self) -> Result<T, RangeError> {
        super::det(self)
    }
}

#[cfg(test)]
mod tests {
    use super::IntMatrix;

    #[test]
    fn overflowing_row_updates_are_transactional() {
        // Column 0 succeeds (0 - (-1) = 1); column 1 overflows (MAX + 1).
        let mut m = IntMatrix::<i32>::from_rows(2, 2, &[0, i32::MAX, 1, 1]).unwrap();
        let before = m.clone();
        assert!(m.row_sub_mul(0, 1, -1).is_err());
        assert_eq!(m, before);
    }

    #[test]
    fn overflowing_products_are_transactional() {
        // The product `factor * s` overflows before any subtraction runs.
        let mut m = IntMatrix::<i32>::from_rows(2, 2, &[1, 1, 2, 1]).unwrap();
        let before = m.clone();
        assert!(m.row_sub_mul(0, 1, i32::MAX).is_err());
        assert_eq!(m, before);

        let mut m = IntMatrix::<i32>::from_rows(2, 2, &[1, 2, 1, 1]).unwrap();
        let before = m.clone();
        assert!(m.col_sub_mul(0, 1, i32::MAX).is_err());
        assert_eq!(m, before);
    }

    #[test]
    fn overflowing_column_updates_are_transactional() {
        let mut m = IntMatrix::<i32>::from_rows(2, 2, &[0, 1, i32::MAX, 1]).unwrap();
        let before = m.clone();
        assert!(m.col_sub_mul(0, 1, -1).is_err());
        assert_eq!(m, before);
    }

    #[test]
    fn overflowing_row_negations_are_transactional() {
        let mut m = IntMatrix::<i32>::from_rows(1, 2, &[-1, i32::MIN]).unwrap();
        let before = m.clone();
        assert!(m.negate_row(0).is_err());
        assert_eq!(m, before);
    }

    #[test]
    fn overflowing_column_negations_are_transactional() {
        let mut m = IntMatrix::<i32>::from_rows(2, 2, &[-1, 0, i32::MIN, 0]).unwrap();
        let before = m.clone();
        assert!(m.negate_col(0).is_err());
        assert_eq!(m, before);
    }

    #[test]
    fn successful_updates_are_unchanged_by_the_transactional_path() {
        let mut m = IntMatrix::<i64>::from_rows(2, 2, &[10, 4, 3, 2]).unwrap();
        m.row_sub_mul(0, 1, 2).unwrap();
        assert_eq!(m.row(0), &[4, 0]);
        m.col_sub_mul(1, 0, -1).unwrap();
        assert_eq!(m.row(0), &[4, 4]);
        m.negate_row(1).unwrap();
        assert_eq!(m.row(1), &[-3, -5]);
        m.negate_col(0).unwrap();
        assert_eq!(m.row(0), &[-4, 4]);
    }
}

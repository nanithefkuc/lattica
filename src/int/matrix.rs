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
    /// [`RangeError::Overflow`] if any entry overflows. The matrix is then left
    /// partially updated; callers that need atomicity work on a clone, which is
    /// what the reduction routines in this module do.
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
        for j in 0..self.cols {
            let s = self.data[source * self.cols + j];
            if s.is_zero() {
                continue;
            }
            let t = self.data[target * self.cols + j];
            self.data[target * self.cols + j] = t.try_sub(factor.try_mul(s)?)?;
        }
        Ok(())
    }

    /// `col[target] -= factor * col[source]`.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if any entry overflows.
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
        for i in 0..self.rows {
            let s = self.data[i * self.cols + source];
            if s.is_zero() {
                continue;
            }
            let t = self.data[i * self.cols + target];
            self.data[i * self.cols + target] = t.try_sub(factor.try_mul(s)?)?;
        }
        Ok(())
    }

    /// Negates every entry of one row.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if an entry is the type minimum.
    ///
    /// # Panics
    ///
    /// If `row` is out of bounds.
    pub fn negate_row(&mut self, row: usize) -> Result<(), RangeError> {
        assert!(row < self.rows, "row index out of bounds");
        for j in 0..self.cols {
            let v = self.data[row * self.cols + j];
            self.data[row * self.cols + j] = v.try_neg()?;
        }
        Ok(())
    }

    /// Negates every entry of one column.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if an entry is the type minimum.
    ///
    /// # Panics
    ///
    /// If `col` is out of bounds.
    pub fn negate_col(&mut self, col: usize) -> Result<(), RangeError> {
        assert!(col < self.cols, "column index out of bounds");
        for i in 0..self.rows {
            let v = self.data[i * self.cols + col];
            self.data[i * self.cols + col] = v.try_neg()?;
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

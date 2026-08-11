//! Runtime-dispatched kernels over real-vector layouts owned by `lattica`.
//!
//! The matrix-vector primitive walks a column-major transform: each input
//! coordinate updates a contiguous run of outputs. SIMD lanes therefore hold
//! independent outputs while every lane accumulates input coordinates in the
//! same order as the scalar reference. The dispatched result is bit-identical,
//! not merely numerically close.

use crate::error::RangeError;

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
mod x86;

#[cfg(feature = "simd")]
use simdispatch::{Backend, Selection};

#[cfg(feature = "simd")]
const LATTICA_TIERS: &[Backend] = &[Backend::V3GfniCrypto, Backend::V3, Backend::Scalar];

/// Applies a column-major dense transform, `out = input * matrix`.
///
/// `matrix` contains `rows` contiguous rows of `cols` output coefficients.
/// The output is initialized by this call. Single-vector geometry deliberately
/// uses the scalar kernel: measurements show dispatch overhead dominates at
/// the dimensions served by this crate. Use [`transform_batch`] to amortize
/// dispatch over many vectors.
///
/// # Errors
///
/// [`RangeError::Shape`] if the matrix or either vector has the wrong length.
pub fn transform(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    input: &[f64],
    out: &mut [f64],
) -> Result<(), RangeError> {
    validate_matrix(matrix, rows, cols)?;
    if input.len() != rows {
        return Err(RangeError::Shape {
            expected: rows,
            found: input.len(),
        });
    }
    if out.len() != cols {
        return Err(RangeError::Shape {
            expected: cols,
            found: out.len(),
        });
    }
    transform_scalar(matrix, cols, input, out);
    Ok(())
}

/// Applies one column-major transform to a flat batch of input vectors.
///
/// `inputs` is strided by `rows`; `outputs` is strided by `cols`. This
/// array-of-structures adapter remains scalar because its short strided vectors
/// do not amortize dispatch. [`transform_batch_soa`] is the dispatched
/// primitive for consumers that keep batches hot.
///
/// # Errors
///
/// [`RangeError::Shape`] for zero geometry, a malformed matrix, incomplete
/// input vectors, or an output length that does not match the input count.
pub fn transform_batch(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    inputs: &[f64],
    outputs: &mut [f64],
) -> Result<(), RangeError> {
    validate_matrix(matrix, rows, cols)?;
    if !inputs.len().is_multiple_of(rows) {
        return Err(RangeError::Shape {
            expected: inputs.len().div_ceil(rows) * rows,
            found: inputs.len(),
        });
    }
    let vectors = inputs.len() / rows;
    let output_len = vectors.checked_mul(cols).ok_or(RangeError::Shape {
        expected: usize::MAX,
        found: outputs.len(),
    })?;
    if outputs.len() != output_len {
        return Err(RangeError::Shape {
            expected: output_len,
            found: outputs.len(),
        });
    }

    transform_batch_scalar(matrix, rows, cols, inputs, outputs);
    Ok(())
}

/// Applies a transform to a structure-of-arrays batch.
///
/// `inputs` contains `rows` planes of `vectors` values; `outputs` contains
/// `cols` planes. SIMD lanes are independent vectors, while every lane
/// accumulates rows in scalar order. The result is therefore bit-identical to
/// the portable reference across lane boundaries and ragged tails.
///
/// # Errors
///
/// [`RangeError::Shape`] for zero matrix geometry or buffer lengths that do not
/// equal `rows * vectors` and `cols * vectors`.
pub fn transform_batch_soa(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    vectors: usize,
    inputs: &[f64],
    outputs: &mut [f64],
) -> Result<(), RangeError> {
    validate_matrix(matrix, rows, cols)?;
    let input_len = rows.checked_mul(vectors).ok_or(RangeError::Shape {
        expected: usize::MAX,
        found: inputs.len(),
    })?;
    if inputs.len() != input_len {
        return Err(RangeError::Shape {
            expected: input_len,
            found: inputs.len(),
        });
    }
    let output_len = cols.checked_mul(vectors).ok_or(RangeError::Shape {
        expected: usize::MAX,
        found: outputs.len(),
    })?;
    if outputs.len() != output_len {
        return Err(RangeError::Shape {
            expected: output_len,
            found: outputs.len(),
        });
    }

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if vectors >= 64 && cols == 16 {
        use archmage::SimdToken;
        if matches!(backend(), Backend::V3GfniCrypto | Backend::V3) {
            // Selection remains simdispatch's single source of policy. Summon
            // only materializes archmage's safe capability token for the tier
            // already selected; it never chooses or upgrades a backend.
            if let Some(token) = archmage::X64V3Token::summon() {
                x86::transform_batch_soa_avx2(token, matrix, cols, vectors, inputs, outputs);
                return Ok(());
            }
        }
    }

    transform_batch_soa_scalar(matrix, cols, vectors, inputs, outputs);
    Ok(())
}

fn validate_matrix(matrix: &[f64], rows: usize, cols: usize) -> Result<(), RangeError> {
    if rows == 0 || cols == 0 {
        return Err(RangeError::Shape {
            expected: 1,
            found: 0,
        });
    }
    let matrix_len = rows.checked_mul(cols).ok_or(RangeError::Shape {
        expected: usize::MAX,
        found: matrix.len(),
    })?;
    if matrix.len() != matrix_len {
        return Err(RangeError::Shape {
            expected: matrix_len,
            found: matrix.len(),
        });
    }
    Ok(())
}

#[cfg(feature = "simd")]
fn backend() -> Backend {
    use std::sync::LazyLock;
    static BACKEND: LazyLock<Backend> = LazyLock::new(|| {
        Selection::new("SIMD_BACKEND")
            .supports(LATTICA_TIERS)
            .resolve()
    });
    *BACKEND
}

fn transform_scalar(matrix: &[f64], cols: usize, input: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for (row, &value) in input.iter().enumerate() {
        let coefficients = &matrix[row * cols..(row + 1) * cols];
        for (slot, &coefficient) in out.iter_mut().zip(coefficients) {
            *slot += value * coefficient;
        }
    }
}

fn transform_batch_scalar(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    inputs: &[f64],
    outputs: &mut [f64],
) {
    for (input, out) in inputs
        .chunks_exact(rows)
        .zip(outputs.chunks_exact_mut(cols))
    {
        transform_scalar(matrix, cols, input, out);
    }
}

fn transform_batch_soa_scalar(
    matrix: &[f64],
    cols: usize,
    vectors: usize,
    inputs: &[f64],
    outputs: &mut [f64],
) {
    for column in 0..cols {
        let out = &mut outputs[column * vectors..(column + 1) * vectors];
        out.fill(0.0);
        for (row, input) in inputs.chunks_exact(vectors).enumerate() {
            let coefficient = matrix[row * cols + column];
            for (slot, &value) in out.iter_mut().zip(input) {
                *slot += coefficient * value;
            }
        }
    }
}

/// Unstable implementation access for differential tests and benchmarks.
#[cfg(feature = "internals")]
pub mod internals {
    /// Portable scalar reference for [`super::transform`].
    pub fn transform_scalar(matrix: &[f64], cols: usize, input: &[f64], out: &mut [f64]) {
        super::transform_scalar(matrix, cols, input, out);
    }

    /// Portable scalar reference for [`super::transform_batch`].
    pub fn transform_batch_scalar(
        matrix: &[f64],
        rows: usize,
        cols: usize,
        inputs: &[f64],
        outputs: &mut [f64],
    ) {
        super::transform_batch_scalar(matrix, rows, cols, inputs, outputs);
    }

    /// Portable scalar reference for [`super::transform_batch_soa`].
    pub fn transform_batch_soa_scalar(
        matrix: &[f64],
        cols: usize,
        vectors: usize,
        inputs: &[f64],
        outputs: &mut [f64],
    ) {
        super::transform_batch_soa_scalar(matrix, cols, vectors, inputs, outputs);
    }
}

#[cfg(test)]
#[allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::float_cmp
)]
mod tests {
    use super::{
        transform, transform_batch, transform_batch_scalar, transform_batch_soa,
        transform_batch_soa_scalar, transform_scalar,
    };

    #[test]
    fn dispatched_transform_is_bit_identical_across_boundaries() {
        for rows in 1..=9 {
            for cols in 1..=11 {
                let matrix: Vec<f64> = (0..rows * cols)
                    .map(|i| (f64::from(i as u32) - 17.0) / 32.0)
                    .collect();
                let input: Vec<f64> = (0..rows)
                    .map(|i| (f64::from(i as u32) - 3.0) / 16.0)
                    .collect();
                let mut want = vec![0.0; cols];
                let mut got = vec![0.0; cols];
                transform_scalar(&matrix, cols, &input, &mut want);
                transform(&matrix, rows, cols, &input, &mut got).unwrap();
                assert_eq!(got, want, "{rows}x{cols}");
            }
        }
    }

    #[test]
    fn dispatched_batches_match_scalar_with_ragged_tails() {
        for rows in 1..=9 {
            for cols in 1..=11 {
                let vectors = 13;
                let matrix: Vec<f64> = (0..rows * cols)
                    .map(|i| (f64::from(i as u32) - 17.0) / 32.0)
                    .collect();
                let inputs: Vec<f64> = (0..rows * vectors)
                    .map(|i| (f64::from(i as u32) - 29.0) / 64.0)
                    .collect();
                let mut want = vec![0.0; cols * vectors];
                let mut got = vec![0.0; cols * vectors];
                transform_batch_scalar(&matrix, rows, cols, &inputs, &mut want);
                transform_batch(&matrix, rows, cols, &inputs, &mut got).unwrap();
                assert_eq!(got, want, "{rows}x{cols}, {vectors} vectors");
            }
        }
    }

    #[test]
    fn dispatched_soa_batches_match_scalar_across_lane_boundaries() {
        for rows in 1..=9 {
            for cols in 1..=16 {
                for vectors in (1..=17).chain([64, 65]) {
                    let matrix: Vec<f64> = (0..rows * cols)
                        .map(|i| (f64::from(i as u32) - 17.0) / 32.0)
                        .collect();
                    let inputs: Vec<f64> = (0..rows * vectors)
                        .map(|i| (f64::from(i as u32) - 29.0) / 64.0)
                        .collect();
                    let mut want = vec![0.0; cols * vectors];
                    let mut got = vec![0.0; cols * vectors];
                    transform_batch_soa_scalar(&matrix, cols, vectors, &inputs, &mut want);
                    transform_batch_soa(&matrix, rows, cols, vectors, &inputs, &mut got).unwrap();
                    assert_eq!(got, want, "{rows}x{cols}, {vectors} vectors");
                }
            }
        }
    }

    #[test]
    fn bad_geometry_does_not_mutate_output() {
        let mut out = [7.0; 3];
        assert!(transform(&[1.0; 5], 2, 3, &[1.0; 2], &mut out).is_err());
        assert_eq!(out, [7.0; 3]);
        let mut batch_out = [7.0; 6];
        assert!(transform_batch(&[1.0; 6], 2, 3, &[1.0; 3], &mut batch_out).is_err());
        assert_eq!(batch_out, [7.0; 6]);
        let mut soa_out = [7.0; 6];
        assert!(transform_batch_soa(&[1.0; 6], 2, 3, 2, &[1.0; 3], &mut soa_out).is_err());
        assert_eq!(soa_out, [7.0; 6]);
    }
}

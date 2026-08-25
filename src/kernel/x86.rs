use archmage::prelude::*;

/// Four independent vectors per SIMD register. There is deliberately no FMA:
/// separate multiply and add preserve the scalar operation sequence.
#[allow(clippy::used_underscore_binding)]
#[arcane(import_intrinsics)]
pub fn transform_batch_soa_avx2(
    _token: X64V3Token,
    matrix: &[f64],
    cols: usize,
    vectors: usize,
    inputs: &[f64],
    outputs: &mut [f64],
) {
    let vector_end = vectors / 4 * 4;
    for column in 0..cols {
        let out = &mut outputs[column * vectors..(column + 1) * vectors];
        out.fill(0.0);
        for (row, input) in inputs.chunks_exact(vectors).enumerate() {
            let coefficient = _mm256_set1_pd(matrix[row * cols + column]);
            for (destination, values) in out[..vector_end]
                .chunks_exact_mut(4)
                .zip(input[..vector_end].chunks_exact(4))
            {
                let destination: &mut [f64; 4] = destination.try_into().unwrap();
                let values: &[f64; 4] = values.try_into().unwrap();
                let updated = _mm256_add_pd(
                    _mm256_loadu_pd(destination),
                    _mm256_mul_pd(coefficient, _mm256_loadu_pd(values)),
                );
                _mm256_storeu_pd(destination, updated);
            }
            let scalar = matrix[row * cols + column];
            for lane in vector_end..vectors {
                out[lane] += scalar * input[lane];
            }
        }
    }
}

/// Fixed 24-by-24 geometry with register-carried output accumulators.
///
/// Output columns advance in blocks of `BLOCK`, so every loaded input chunk
/// feeds `BLOCK` output planes and each accumulator lives entirely in
/// registers across all twenty-four rows. Every lane still accumulates rows
/// in ascending order with separate multiply and add, which is the exact
/// scalar operation sequence; the ragged tail reuses the scalar expression
/// order directly.
macro_rules! fixed_24_kernel {
    ($name:ident, $block:expr) => {
        /// Fixed-geometry 24-by-24 `SoA` kernel; see the family comment above
        /// for the exactness argument.
        #[allow(clippy::used_underscore_binding)]
        #[arcane(import_intrinsics)]
        pub fn $name(
            _token: X64V3Token,
            matrix: &[f64],
            vectors: usize,
            inputs: &[f64],
            outputs: &mut [f64],
        ) {
            const ROWS: usize = 24;
            const COLS: usize = 24;
            const BLOCK: usize = $block;
            debug_assert_eq!(matrix.len(), ROWS * COLS);
            let vector_end = vectors / 4 * 4;
            let mut block = 0;
            while block < COLS {
                for offset in (0..vector_end).step_by(4) {
                    let mut acc = [_mm256_setzero_pd(); BLOCK];
                    for row in 0..ROWS {
                        let values: &[f64; 4] =
                            inputs[row * vectors + offset..][..4].try_into().unwrap();
                        let loaded = _mm256_loadu_pd(values);
                        let coefficients = &matrix[row * COLS + block..][..BLOCK];
                        for (slot, coefficient) in acc.iter_mut().zip(coefficients) {
                            let scaled = _mm256_broadcast_sd(coefficient);
                            *slot = _mm256_add_pd(*slot, _mm256_mul_pd(scaled, loaded));
                        }
                    }
                    for (plane, value) in acc.iter().enumerate() {
                        let destination: &mut [f64; 4] = (&mut outputs
                            [(block + plane) * vectors + offset..][..4])
                            .try_into()
                            .unwrap();
                        _mm256_storeu_pd(destination, *value);
                    }
                }
                block += BLOCK;
            }
            for column in 0..COLS {
                let out = &mut outputs[column * vectors..(column + 1) * vectors];
                for lane in vector_end..vectors {
                    let mut sum = 0.0;
                    for row in 0..ROWS {
                        sum += matrix[row * COLS + column] * inputs[row * vectors + lane];
                    }
                    out[lane] = sum;
                }
            }
        }
    };
}

fixed_24_kernel!(transform_batch_soa_fixed_24_block6, 6);
fixed_24_kernel!(transform_batch_soa_fixed_24_block8, 8);
fixed_24_kernel!(transform_batch_soa_fixed_24_block12, 12);

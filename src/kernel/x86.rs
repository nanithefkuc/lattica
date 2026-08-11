use archmage::prelude::*;

/// Four independent vectors per SIMD register. There is deliberately no FMA:
/// separate multiply and add preserve the scalar operation sequence.
#[allow(clippy::used_underscore_binding)]
#[arcane(import_intrinsics)]
pub(super) fn transform_batch_soa_avx2(
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

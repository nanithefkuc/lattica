//! Exact Voronoi-relevant vectors in low dimension.
//!
//! A nonzero lattice vector `v` is Voronoi-relevant exactly when `v` and `-v`
//! are the only shortest vectors in the coset `v + 2Λ`. The implementation
//! enumerates one exact ball large enough to contain a representative of every
//! parity coset, then applies that characterization without floating point.

use crate::basis::Gram;
use crate::error::{DecodeError, RangeError};
use crate::int::Int;
use crate::shortvec::for_each_short;

/// Largest supported dimension for relevant-vector enumeration.
///
/// The algorithm stores one state per coset of `2Λ`, so its unavoidable state
/// is exponential in the dimension. This API is intentionally for low-
/// dimensional oracle and facet work, not for high-dimensional decoding.
pub const MAX_RELEVANT_DIM: usize = 16;

#[derive(Default)]
struct CosetMinimum {
    norm_sq: Option<i128>,
    vectors: Vec<Vec<i128>>,
}

/// Enumerates every Voronoi-relevant vector of `gram`.
///
/// Each vector and its negation are returned separately, matching the usual
/// facet-count convention. Results are in lexicographic coordinate order.
/// Every comparison is exact integer arithmetic.
///
/// # Errors
///
/// - [`RangeError::Dimension`] above [`MAX_RELEVANT_DIM`];
/// - [`DecodeError::NotInLattice`] if `gram` is not positive definite;
/// - [`DecodeError::EnumerationBudget`] if `node_budget` is exhausted;
/// - [`DecodeError::Range`] if an exact intermediate exceeds `i128`.
pub fn relevant_vectors<T: Int>(
    gram: &Gram<T>,
    node_budget: u64,
) -> Result<Vec<Vec<i128>>, DecodeError> {
    let n = gram.dim();
    if n > MAX_RELEVANT_DIM {
        return Err(RangeError::Dimension {
            requested: n,
            max: MAX_RELEVANT_DIM,
        }
        .into());
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    let coset_count = 1usize << n;
    let mut representative = vec![T::ZERO; n];
    let mut radius_sq = 0i128;
    for mask in 1..coset_count {
        for (i, value) in representative.iter_mut().enumerate() {
            *value = if mask & (1 << i) == 0 {
                T::ZERO
            } else {
                T::ONE
            };
        }
        radius_sq = radius_sq.max(gram.norm_sq(&representative)?.widen());
    }

    let mut minima: Vec<CosetMinimum> = (0..coset_count).map(|_| CosetMinimum::default()).collect();
    for_each_short(gram, radius_sq, node_budget, |coordinates, norm_sq| {
        let mask = parity_mask(coordinates);
        let state = &mut minima[mask];
        match state.norm_sq {
            None => {
                state.norm_sq = Some(norm_sq);
                state.vectors.push(coordinates.to_vec());
            }
            Some(current) if norm_sq < current => {
                state.norm_sq = Some(norm_sq);
                state.vectors.clear();
                state.vectors.push(coordinates.to_vec());
            }
            Some(current) if norm_sq == current => {
                state.vectors.push(coordinates.to_vec());
            }
            Some(_) => {}
        }
    })?;

    let mut relevant = Vec::new();
    for state in minima.into_iter().skip(1) {
        if state.vectors.len() != 2 {
            continue;
        }
        let a = &state.vectors[0];
        let b = &state.vectors[1];
        if a.iter().zip(b).all(|(&x, &y)| x.checked_neg() == Some(y)) {
            relevant.extend(state.vectors);
        }
    }
    relevant.sort();
    Ok(relevant)
}

fn parity_mask(coordinates: &[i128]) -> usize {
    coordinates
        .iter()
        .enumerate()
        .fold(0usize, |mask, (i, &value)| {
            if value & 1 == 0 {
                mask
            } else {
                mask | (1 << i)
            }
        })
}

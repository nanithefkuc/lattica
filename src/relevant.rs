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
#[cfg(feature = "internals")]
use std::time::Instant;

/// Largest supported dimension for relevant-vector enumeration.
///
/// The algorithm stores one state per coset of `2Λ`, so its unavoidable state
/// is exponential in the dimension. This API is intentionally for low-
/// dimensional oracle and facet work, not for high-dimensional decoding.
pub const MAX_RELEVANT_DIM: usize = 16;

/// Flat per-coset minima: one best norm, an arrival count capped past two,
/// and up to two coordinate blocks for the opposite-pair check.
///
/// A coset is Voronoi-relevant exactly when its minimum is attained by
/// precisely two vectors and they are negatives. Ties beyond two prove the
/// coset irrelevant, so nothing past the second block is ever stored.
#[derive(Default)]
struct CosetMinima {
    n: usize,
    norms: Vec<Option<i128>>,
    counts: Vec<u32>,
    blocks: Vec<i128>,
}

impl CosetMinima {
    fn new(cosets: usize, n: usize) -> Self {
        Self {
            n,
            norms: vec![None; cosets],
            counts: vec![0; cosets],
            blocks: vec![0; 2 * cosets * n],
        }
    }

    fn block(&self, mask: usize, slot: usize) -> &[i128] {
        let start = (mask * 2 + slot) * self.n;
        &self.blocks[start..start + self.n]
    }

    fn block_mut(&mut self, mask: usize, slot: usize) -> &mut [i128] {
        let start = (mask * 2 + slot) * self.n;
        &mut self.blocks[start..start + self.n]
    }

    fn offer<S: CosetSink>(
        &mut self,
        mask: usize,
        coordinates: &[i128],
        norm_sq: i128,
        sink: &mut S,
    ) {
        match self.norms[mask] {
            None => {
                self.norms[mask] = Some(norm_sq);
                self.counts[mask] = 1;
                self.block_mut(mask, 0).copy_from_slice(coordinates);
            }
            Some(current) if norm_sq < current => {
                self.norms[mask] = Some(norm_sq);
                self.counts[mask] = 1;
                self.block_mut(mask, 0).copy_from_slice(coordinates);
                sink.reset();
            }
            Some(current) if norm_sq == current => {
                sink.tie();
                let count = self.counts[mask];
                if count < 2 {
                    let slot = usize::try_from(count).unwrap_or(2);
                    self.block_mut(mask, slot).copy_from_slice(coordinates);
                    self.counts[mask] = count + 1;
                }
            }
            Some(_) => {}
        }
    }
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
    let (coset_count, radius_sq) = radius_for_parity_ball(gram)?;
    if coset_count == 0 {
        return Ok(Vec::new());
    }
    let mut minima = CosetMinima::new(coset_count, gram.dim());
    collect_coset_minima_with(gram, radius_sq, node_budget, &mut minima, &mut NoSink)?;
    Ok(materialize_relevant(&minima))
}

/// Computes the parity-coset count and the smallest radius whose ball holds a
/// representative of every coset: the largest norm among the 0/1 vectors.
fn radius_for_parity_ball<T: Int>(gram: &Gram<T>) -> Result<(usize, i128), DecodeError> {
    let n = gram.dim();
    if n > MAX_RELEVANT_DIM {
        return Err(RangeError::Dimension {
            requested: n,
            max: MAX_RELEVANT_DIM,
        }
        .into());
    }
    if n == 0 {
        return Ok((0, 0));
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
    Ok((coset_count, radius_sq))
}

fn collect_coset_minima_with<T: Int, S: CosetSink>(
    gram: &Gram<T>,
    radius_sq: i128,
    node_budget: u64,
    minima: &mut CosetMinima,
    sink: &mut S,
) -> Result<(), DecodeError> {
    for_each_short(gram, radius_sq, node_budget, |coordinates, norm_sq| {
        sink.emission();
        let mask = parity_mask(coordinates);
        minima.offer(mask, coordinates, norm_sq, sink);
    })?;
    Ok(())
}

fn materialize_relevant(minima: &CosetMinima) -> Vec<Vec<i128>> {
    let mut relevant = Vec::new();
    for mask in 1..minima.norms.len() {
        if minima.counts[mask] != 2 {
            continue;
        }
        let a = minima.block(mask, 0);
        let b = minima.block(mask, 1);
        if a.iter().zip(b).all(|(&x, &y)| x.checked_neg() == Some(y)) {
            relevant.push(a.to_vec());
            relevant.push(b.to_vec());
        }
    }
    relevant.sort();
    relevant
}

/// Unstable benchmark counters and stage timings for relevant-vector
/// enumeration.
///
/// Available only with `internals`; not a compatibility promise. The
/// enumerated result matches [`relevant_vectors`] exactly.
#[cfg(feature = "internals")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RelevantStats {
    /// Parity-coset representatives evaluated for the radius.
    pub masks: u64,
    /// Short vectors seen by the coset pass.
    pub emissions: u64,
    /// Strictly-better coset minima replaced.
    pub coset_resets: u64,
    /// Equal-minimum vectors stored beyond the first.
    pub ties_stored: u64,
    /// Voronoi-relevant vectors materialized.
    pub output_len: u64,
    /// Nanoseconds forming the radius over the parity representatives.
    pub setup_ns: u64,
    /// Nanoseconds enumerating and classifying short vectors.
    pub walk_ns: u64,
    /// Nanoseconds collecting opposite pairs and sorting.
    pub finalize_ns: u64,
}

/// Sink receiving the classification events of the coset pass.
trait CosetSink {
    fn emission(&mut self) {}
    fn reset(&mut self) {}
    fn tie(&mut self) {}
}

struct NoSink;

impl CosetSink for NoSink {}

#[cfg(feature = "internals")]
#[derive(Default)]
struct CountingSink {
    emissions: u64,
    resets: u64,
    ties: u64,
}

#[cfg(feature = "internals")]
impl CosetSink for CountingSink {
    fn emission(&mut self) {
        self.emissions += 1;
    }

    fn reset(&mut self) {
        self.resets += 1;
    }

    fn tie(&mut self) {
        self.ties += 1;
    }
}

/// Enumerates relevant vectors while returning unstable benchmark counters
/// and stage timings.
///
/// # Errors
///
/// As [`relevant_vectors`].
#[cfg(feature = "internals")]
pub fn relevant_vectors_profiled<T: Int>(
    gram: &Gram<T>,
    node_budget: u64,
) -> Result<(Vec<Vec<i128>>, RelevantStats), DecodeError> {
    let mut stats = RelevantStats::default();
    let setup_start = Instant::now();
    let (coset_count, radius_sq) = radius_for_parity_ball(gram)?;
    stats.setup_ns = u64::try_from(setup_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    stats.masks = u64::try_from(coset_count.saturating_sub(1)).unwrap_or(u64::MAX);
    if coset_count == 0 {
        return Ok((Vec::new(), stats));
    }

    let mut minima = CosetMinima::new(coset_count, gram.dim());
    let mut sink = CountingSink::default();
    let walk_start = Instant::now();
    collect_coset_minima_with(gram, radius_sq, node_budget, &mut minima, &mut sink)?;
    stats.walk_ns = u64::try_from(walk_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    stats.emissions = sink.emissions;
    stats.coset_resets = sink.resets;
    stats.ties_stored = sink.ties;

    let finalize_start = Instant::now();
    let relevant = materialize_relevant(&minima);
    stats.finalize_ns = u64::try_from(finalize_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    stats.output_len = u64::try_from(relevant.len()).unwrap_or(u64::MAX);
    Ok((relevant, stats))
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

#[cfg(test)]
mod tests {
    use super::{MAX_RELEVANT_DIM, relevant_vectors};
    use crate::basis::Gram;
    use crate::named::{a_n, d_n, e8, zn};

    #[test]
    fn the_cubic_lattice_has_only_its_axes() {
        let g = zn::<i64>(3).unwrap();
        let v = relevant_vectors(&g, 1 << 20).unwrap();
        assert_eq!(v.len(), 6);
        assert!(v.contains(&vec![1, 0, 0]));
        assert!(v.contains(&vec![0, 0, -1]));
    }

    #[test]
    fn the_hexagonal_lattice_has_six_relevant_vectors() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        let v = relevant_vectors(&g, 1 << 20).unwrap();
        assert_eq!(v.len(), 6);
    }

    #[test]
    fn root_lattices_keep_their_published_facet_counts() {
        // Z^n: ±e_i. A_n and D_n: their roots. E8: its 240 roots.
        let cases: Vec<(Gram<i64>, u64)> = vec![
            (zn::<i64>(6).unwrap(), 12),
            (a_n::<i64>(7).unwrap(), 56),
            (d_n::<i64>(8).unwrap(), 112),
            (e8::<i64>().unwrap(), 240),
        ];
        for (gram, expected) in cases {
            assert_eq!(
                u64::try_from(relevant_vectors(&gram, 1 << 24).unwrap().len()).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn every_output_pair_is_opposite_and_sorted() {
        let g = d_n::<i64>(6).unwrap();
        let v = relevant_vectors(&g, 1 << 24).unwrap();
        // Lexicographic order.
        assert!(v.windows(2).all(|pair| pair[0] < pair[1]));
        // Each vector's negation is present exactly once more.
        for vector in &v {
            let negated: Vec<i128> = vector.iter().map(|&x| -x).collect();
            assert_eq!(v.iter().filter(|c| **c == negated).count(), 1);
        }
    }

    #[test]
    fn the_dimension_cap_holds() {
        let g = zn::<i64>(17).unwrap();
        assert!(matches!(
            relevant_vectors(&g, 1 << 20),
            Err(crate::error::DecodeError::Range(
                crate::error::RangeError::Dimension {
                    requested: 17,
                    max: MAX_RELEVANT_DIM
                }
            ))
        ));
    }
}

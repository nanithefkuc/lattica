//! Exact Voronoi-relevant vectors in low dimension.
//!
//! A nonzero lattice vector `v` is Voronoi-relevant exactly when `v` and `-v`
//! are the only shortest vectors in the coset `v + 2Λ`. The implementation
//! enumerates one exact ball large enough to contain a representative of every
//! parity coset, then applies that characterization without floating point.

use crate::basis::Gram;
use crate::error::{EnumerationError, RangeError};
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
/// - [`EnumerationError::NotALattice`] if `gram` is not positive definite;
/// - [`EnumerationError::EnumerationBudget`] if `node_budget` is exhausted;
/// - [`EnumerationError::Range`] if an exact intermediate exceeds `i128`.
pub fn relevant_vectors<T: Int>(
    gram: &Gram<T>,
    node_budget: u64,
) -> Result<Vec<Vec<i128>>, EnumerationError> {
    check_dimension(gram.dim())?;
    let components = orthogonal_components(gram);
    if components.len() <= 1 {
        return Ok(relevant_connected(gram, node_budget)?.0);
    }

    // An orthogonal direct sum's Voronoi cell is the product of the cells,
    // and a product's facets are exactly the factors' facets: a vector with
    // components in two summands is never relevant, because flipping the
    // sign of one component gives a distinct vector of the same norm in the
    // same coset. The relevant set is the union of the components', embedded
    // into the full coordinates, with every component's walk charged against
    // the one aggregate budget.
    let dimension = gram.dim();
    let mut remaining = node_budget;
    let mut all = Vec::new();
    for component in &components {
        let block = component_gram(gram, component);
        let (vectors, nodes) = relevant_connected(&block, remaining)?;
        remaining -= nodes;
        for vector in vectors {
            let mut embedded = vec![0i128; dimension];
            for (position, &index) in component.iter().enumerate() {
                embedded[index] = vector[position];
            }
            all.push(embedded);
        }
    }
    all.sort();
    Ok(all)
}

/// The connected components of the Gram matrix's off-diagonal support: the
/// maximal orthogonal direct-sum decomposition of the lattice.
fn orthogonal_components<T: Int>(gram: &Gram<T>) -> Vec<Vec<usize>> {
    let n = gram.dim();
    let mut seen = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut component = vec![start];
        let mut cursor = 0;
        while cursor < component.len() {
            let i = component[cursor];
            cursor += 1;
            let mut newly_seen = Vec::new();
            for (j, &visited) in seen.iter().enumerate() {
                if !visited && j != i && !gram.entry(i, j).is_zero() {
                    newly_seen.push(j);
                }
            }
            for j in newly_seen {
                seen[j] = true;
                component.push(j);
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

/// The Gram matrix of one component's sublattice, in the component's sorted
/// index order.
fn component_gram<T: Int>(gram: &Gram<T>, component: &[usize]) -> Gram<T> {
    let k = component.len();
    let mut data = vec![T::ZERO; k * k];
    for (r, &i) in component.iter().enumerate() {
        for (c, &j) in component.iter().enumerate() {
            data[r * k + c] = gram.entry(i, j);
        }
    }
    // A principal submatrix of a Gram matrix is square and symmetric, and
    // `k <= MAX_RELEVANT_DIM` is far below the dimension limit, so the
    // checked constructor cannot reject it.
    Gram::from_rows(k, &data).expect("a principal submatrix of a Gram matrix")
}

/// The connected case: one parity-coset classification walk over the whole
/// form. Returns the relevant vectors and the nodes the walk spent.
fn relevant_connected<T: Int>(
    gram: &Gram<T>,
    node_budget: u64,
) -> Result<(Vec<Vec<i128>>, u64), EnumerationError> {
    let (coset_count, radius_sq) = radius_for_parity_ball(gram)?;
    if coset_count == 0 {
        return Ok((Vec::new(), 0));
    }
    let mut minima = CosetMinima::new(coset_count, gram.dim());
    let nodes = collect_coset_minima_with(gram, radius_sq, node_budget, &mut minima, &mut NoSink)?;
    Ok((materialize_relevant(&minima), nodes))
}

/// Computes the parity-coset count and the smallest radius whose ball holds a
/// representative of every coset: the largest norm among the 0/1 vectors.
fn radius_for_parity_ball<T: Int>(gram: &Gram<T>) -> Result<(usize, i128), EnumerationError> {
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
) -> Result<u64, EnumerationError> {
    let nodes = for_each_short(gram, radius_sq, node_budget, |coordinates, norm_sq| {
        sink.emission();
        let mask = parity_mask(coordinates);
        minima.offer(mask, coordinates, norm_sq, sink);
    })?;
    Ok(nodes)
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
) -> Result<(Vec<Vec<i128>>, RelevantStats), EnumerationError> {
    check_dimension(gram.dim())?;
    let components = orthogonal_components(gram);
    if components.len() <= 1 {
        let (vectors, stats, _nodes) = relevant_connected_profiled(gram, node_budget)?;
        return Ok((vectors, stats));
    }

    // Same decomposition as `relevant_vectors`, with each component's
    // counters summed so the profiled totals describe the whole call.
    let dimension = gram.dim();
    let mut remaining = node_budget;
    let mut total = RelevantStats::default();
    let mut all = Vec::new();
    for component in &components {
        let block = component_gram(gram, component);
        let (vectors, stats, nodes) = relevant_connected_profiled(&block, remaining)?;
        remaining -= nodes;
        total.setup_ns = total.setup_ns.saturating_add(stats.setup_ns);
        total.walk_ns = total.walk_ns.saturating_add(stats.walk_ns);
        total.finalize_ns = total.finalize_ns.saturating_add(stats.finalize_ns);
        total.masks = total.masks.saturating_add(stats.masks);
        total.emissions = total.emissions.saturating_add(stats.emissions);
        total.coset_resets = total.coset_resets.saturating_add(stats.coset_resets);
        total.ties_stored = total.ties_stored.saturating_add(stats.ties_stored);
        for vector in vectors {
            let mut embedded = vec![0i128; dimension];
            for (position, &index) in component.iter().enumerate() {
                embedded[index] = vector[position];
            }
            all.push(embedded);
        }
    }
    let finalize_start = Instant::now();
    all.sort();
    total.finalize_ns = total
        .finalize_ns
        .saturating_add(u64::try_from(finalize_start.elapsed().as_nanos()).unwrap_or(u64::MAX));
    total.output_len = u64::try_from(all.len()).unwrap_or(u64::MAX);
    Ok((all, total))
}

/// The profiled connected case: the original single-walk stage split, plus
/// the walk's node count so the decomposed path can charge one budget.
#[cfg(feature = "internals")]
fn relevant_connected_profiled<T: Int>(
    gram: &Gram<T>,
    node_budget: u64,
) -> Result<(Vec<Vec<i128>>, RelevantStats, u64), EnumerationError> {
    let mut stats = RelevantStats::default();
    let setup_start = Instant::now();
    let (coset_count, radius_sq) = radius_for_parity_ball(gram)?;
    stats.setup_ns = u64::try_from(setup_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    stats.masks = u64::try_from(coset_count.saturating_sub(1)).unwrap_or(u64::MAX);
    if coset_count == 0 {
        return Ok((Vec::new(), stats, 0));
    }

    let mut minima = CosetMinima::new(coset_count, gram.dim());
    let mut sink = CountingSink::default();
    let walk_start = Instant::now();
    let nodes = collect_coset_minima_with(gram, radius_sq, node_budget, &mut minima, &mut sink)?;
    stats.walk_ns = u64::try_from(walk_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    stats.emissions = sink.emissions;
    stats.coset_resets = sink.resets;
    stats.ties_stored = sink.ties;

    let finalize_start = Instant::now();
    let relevant = materialize_relevant(&minima);
    stats.finalize_ns = u64::try_from(finalize_start.elapsed().as_nanos()).unwrap_or(u64::MAX);
    stats.output_len = u64::try_from(relevant.len()).unwrap_or(u64::MAX);
    Ok((relevant, stats, nodes))
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

/// The cap applies to the *lattice* dimension, before any decomposition: a
/// seventeen-dimensional diagonal Gram is still over the budget even though
/// every component is one-dimensional.
fn check_dimension(n: usize) -> Result<(), EnumerationError> {
    if n > MAX_RELEVANT_DIM {
        return Err(RangeError::Dimension {
            requested: n,
            max: MAX_RELEVANT_DIM,
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "internals")]
    use super::relevant_vectors_profiled;
    use super::{MAX_RELEVANT_DIM, relevant_vectors};
    use crate::basis::Gram;
    use crate::error::EnumerationError;
    use crate::named::{a_n, d_n, e8, zn};
    use crate::shortvec::DEFAULT_NODE_BUDGET;

    #[test]
    fn a_zero_dimensional_lattice_has_no_relevant_vectors() {
        let empty = Gram::<i64>::from_rows(0, &[]).unwrap();
        assert_eq!(
            relevant_vectors(&empty, 1 << 8).unwrap(),
            Vec::<Vec<i128>>::new()
        );
        #[cfg(feature = "internals")]
        {
            let (v, stats) = relevant_vectors_profiled(&empty, 1 << 8).unwrap();
            assert!(v.is_empty());
            assert_eq!(stats.masks, 0);
        }
    }

    /// The profiled path returns the same vectors as the public one, with
    /// counters that partition the walk.
    #[cfg(feature = "internals")]
    #[test]
    fn profiled_counters_match_the_public_path() {
        let g = d_n::<i64>(6).unwrap();
        let plain = relevant_vectors(&g, 1 << 24).unwrap();
        let (profiled, stats) = relevant_vectors_profiled(&g, 1 << 24).unwrap();
        assert_eq!(plain, profiled);
        assert_eq!(stats.output_len, u64::try_from(profiled.len()).unwrap());
        assert_eq!(stats.masks, 63);
        assert!(stats.emissions >= stats.output_len);
        assert!(stats.coset_resets > 0);
        assert!(stats.walk_ns > 0 && stats.setup_ns > 0);
        // Emissions exceed stored ties because irrelevant minima are dropped
        // and strictly-worse vectors change nothing.
        assert!(stats.ties_stored > 0);
    }

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
            Err(crate::error::EnumerationError::Range(
                crate::error::RangeError::Dimension {
                    requested: 17,
                    max: MAX_RELEVANT_DIM
                }
            ))
        ));
    }
    #[test]
    fn an_orthogonal_sum_exposes_each_components_facets() {
        // Z^6 is six one-dimensional summands; its Voronoi cell is a cube,
        // whose facets are exactly the ±e_i pairs.
        let cube = zn::<i64>(6).unwrap();
        let vectors = relevant_vectors(&cube, DEFAULT_NODE_BUDGET).unwrap();
        let mut expected = Vec::new();
        for i in 0..6 {
            let mut plus = vec![0i128; 6];
            plus[i] = 1;
            let mut minus = vec![0i128; 6];
            minus[i] = -1;
            expected.push(minus);
            expected.push(plus);
        }
        expected.sort();
        assert_eq!(vectors, expected);

        // Two hexagonal summands: the cell is the product of two hexagons,
        // so the facet count doubles and every facet lives in exactly one
        // summand's coordinates.
        let mut data = [0i64; 16];
        let hex = [2i64, -1, -1, 2];
        data[0] = hex[0];
        data[1] = hex[1];
        data[4] = hex[1];
        data[5] = hex[0];
        data[10] = hex[0];
        data[11] = hex[1];
        data[14] = hex[1];
        data[15] = hex[0];
        let pair = Gram::<i64>::from_rows(4, &data).unwrap();
        let vectors = relevant_vectors(&pair, DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(vectors.len(), 12);
        for vector in &vectors {
            let first = vector[0] != 0 || vector[1] != 0;
            let second = vector[2] != 0 || vector[3] != 0;
            assert!(first ^ second, "facet spans both summands: {vector:?}");
        }
    }

    #[test]
    fn decomposed_walks_share_one_budget() {
        let cube = zn::<i64>(12).unwrap();
        assert!(matches!(
            relevant_vectors(&cube, 1),
            Err(EnumerationError::EnumerationBudget { .. })
        ));
    }
}

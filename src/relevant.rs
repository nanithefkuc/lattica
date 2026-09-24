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
///
/// That capping relies on the walk's emission order: vectors arrive in
/// ascending lexicographic order over `(c_{n-1}, …, c_0)`, and negation
/// reverses it, so with four or more minima the first two arrivals are
/// never opposite. A walk that reorders emissions must revisit this.
struct CosetMinima {
    n: usize,
    cosets: usize,
    norms: Vec<Option<i128>>,
    counts: Vec<u32>,
    blocks: Vec<i128>,
}

impl CosetMinima {
    fn new(cosets: usize, n: usize) -> Self {
        Self {
            n,
            cosets,
            norms: vec![None; cosets],
            counts: vec![0; cosets],
            blocks: vec![0; 2 * cosets * n],
        }
    }

    /// Rewinds the leading entries for a component walk: the first `cosets`
    /// counts are cleared and the first `cosets` norms forgotten, so no
    /// earlier component's minima leak into this one. Blocks need no
    /// clearing: [`offer`](Self::offer) writes a block before raising its
    /// count, and [`materialize_relevant`] only reads blocks whose count
    /// reached two.
    ///
    /// The buffers keep their capacity across calls; the caller sizes them
    /// for the whole lattice, so every component fits.
    ///
    /// Only the reusable scratch calls this.
    // Called only by the reusable scratch behind the `internals` facade.
    #[allow(dead_code)]
    fn reset(&mut self, cosets: usize, n: usize) {
        self.n = n;
        self.cosets = cosets;
        self.norms[..cosets].fill(None);
        self.counts[..cosets].fill(0);
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
    let mut seen = Vec::new();
    let mut newly_seen = Vec::new();
    let mut components = Vec::new();
    orthogonal_components_into(gram, &mut seen, &mut newly_seen, &mut components);
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
    let mut gather = Vec::new();
    let mut all = Vec::new();
    for component in &components {
        let block = component_gram_into(gram, component, &mut gather);
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
    // Component vectors are distinct, so the stable order and the unstable
    // order coincide; the unstable sort only drops the merge allocation.
    all.sort_unstable();
    Ok(all)
}

/// Unstable relevant-vector surface: reusable buffers and benchmark counters.
/// Reachable externally only through the `internals` facade.
// Unstable items are reachable only through the `internals` facade, so the
// library target without that feature reports them as unused.
#[allow(dead_code)]
pub(crate) mod unstable {
    use super::{
        CosetMinima, CosetSink, EnumerationError, Gram, Int, NoSink, RangeError, check_dimension,
        collect_coset_minima_with, component_gram_into, materialize_relevant,
        orthogonal_components_into, parity_mask, radius_for_parity_ball,
        radius_for_parity_ball_into,
    };
    use std::time::Instant;
    /// Reusable relevant-vector buffers for one lattice dimension.
    ///
    /// The scratch holds an [`EnumerationScratch`](crate::internals::shortvec::EnumerationScratch)
    /// for the coset walks, the per-coset minima sized for the whole lattice,
    /// and the decomposition and probe buffers. Each call re-walks its Gram,
    /// so the scratch carries no lattice between calls — only the allocation.
    ///
    /// A rejected call leaves the scratch reusable, though not untouched: the
    /// decomposition and probe buffers are rewritten before any fallible check
    /// past the dimension match, and a failed walk can leave coordinates behind.
    /// None of that state is observable — only [`dim`](Self::dim) is exposed —
    /// and no later call can read it: the minima are rewound on entry to every
    /// component walk, and every coordinates read follows a write on the
    /// current walk.
    ///
    /// Reachable only through the `internals` facade; not a compatibility promise.
    pub struct RelevantScratch<T: Int> {
        dim: usize,
        enumeration: crate::shortvec::unstable::EnumerationScratch,
        minima: CosetMinima,
        seen: Vec<bool>,
        newly_seen: Vec<usize>,
        components: Vec<Vec<usize>>,
        gather: Vec<T>,
        probe: Vec<T>,
    }

    impl<T: Int> RelevantScratch<T> {
        /// Allocates relevant-vector buffers for `dimension`.
        ///
        /// # Errors
        ///
        /// [`RangeError::Dimension`] above [`MAX_RELEVANT_DIM`](super::MAX_RELEVANT_DIM).
        pub fn new(dimension: usize) -> Result<Self, EnumerationError> {
            check_dimension(dimension)?;
            Ok(Self {
                dim: dimension,
                enumeration: crate::shortvec::unstable::EnumerationScratch::new(dimension)?,
                minima: CosetMinima::new(1 << dimension, dimension),
                seen: Vec::new(),
                newly_seen: Vec::new(),
                components: Vec::new(),
                gather: Vec::new(),
                probe: Vec::new(),
            })
        }

        /// The dimension this scratch was sized for.
        #[must_use]
        pub const fn dim(&self) -> usize {
            self.dim
        }

        /// Enumerates relevant vectors over reused buffers, identical to
        /// [`relevant_vectors`](super::relevant_vectors) on the input.
        ///
        /// # Errors
        ///
        /// [`RangeError::Shape`] if `gram.dim()` does not equal
        /// [`Self::dim`]; otherwise as [`relevant_vectors`](super::relevant_vectors).
        pub fn relevant_vectors(
            &mut self,
            gram: &Gram<T>,
            node_budget: u64,
        ) -> Result<Vec<Vec<i128>>, EnumerationError> {
            if gram.dim() != self.dim {
                return Err(RangeError::Shape {
                    expected: self.dim,
                    found: gram.dim(),
                }
                .into());
            }
            let n = self.dim;
            orthogonal_components_into(
                gram,
                &mut self.seen,
                &mut self.newly_seen,
                &mut self.components,
            );
            if self.components.len() <= 1 {
                let (coset_count, radius_sq) = radius_for_parity_ball_into(gram, &mut self.probe)?;
                if coset_count == 0 {
                    return Ok(Vec::new());
                }
                self.minima.reset(coset_count, n);
                let minima = &mut self.minima;
                self.enumeration.for_each_prefix(
                    gram,
                    radius_sq,
                    node_budget,
                    |coordinates, norm_sq| {
                        minima.offer(parity_mask(coordinates), coordinates, norm_sq, &mut NoSink);
                    },
                )?;
                return Ok(materialize_relevant(&self.minima));
            }
            let mut remaining = node_budget;
            let mut all = Vec::new();
            for index in 0..self.components.len() {
                let block = component_gram_into(gram, &self.components[index], &mut self.gather);
                let k = block.dim();
                let (coset_count, radius_sq) =
                    radius_for_parity_ball_into(&block, &mut self.probe)?;
                if coset_count == 0 {
                    continue;
                }
                self.minima.reset(coset_count, k);
                let minima = &mut self.minima;
                let nodes = self.enumeration.for_each_prefix(
                    &block,
                    radius_sq,
                    remaining,
                    |coordinates, norm_sq| {
                        minima.offer(parity_mask(coordinates), coordinates, norm_sq, &mut NoSink);
                    },
                )?;
                remaining -= nodes;
                let vectors = materialize_relevant(&self.minima);
                for vector in vectors {
                    let mut embedded = vec![0i128; n];
                    for (position, &coordinate) in self.components[index].iter().enumerate() {
                        embedded[coordinate] = vector[position];
                    }
                    all.push(embedded);
                }
            }
            all.sort_unstable();
            Ok(all)
        }
    }
    /// Unstable benchmark counters and stage timings for relevant-vector
    /// enumeration.
    ///
    /// Reachable only through the `internals` facade; not a compatibility promise. The
    /// enumerated result matches [`relevant_vectors`](super::relevant_vectors) exactly.
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
    #[derive(Default)]
    struct CountingSink {
        emissions: u64,
        resets: u64,
        ties: u64,
    }

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
    /// As [`relevant_vectors`](super::relevant_vectors).
    pub fn relevant_vectors_profiled<T: Int>(
        gram: &Gram<T>,
        node_budget: u64,
    ) -> Result<(Vec<Vec<i128>>, RelevantStats), EnumerationError> {
        check_dimension(gram.dim())?;
        let mut seen = Vec::new();
        let mut newly_seen = Vec::new();
        let mut components = Vec::new();
        orthogonal_components_into(gram, &mut seen, &mut newly_seen, &mut components);
        if components.len() <= 1 {
            let (vectors, stats, _nodes) = relevant_connected_profiled(gram, node_budget)?;
            return Ok((vectors, stats));
        }

        // Same decomposition as `relevant_vectors`, with each component's
        // counters summed so the profiled totals describe the whole call.
        let dimension = gram.dim();
        let mut remaining = node_budget;
        let mut total = RelevantStats::default();
        let mut gather = Vec::new();
        let mut all = Vec::new();
        for component in &components {
            let block = component_gram_into(gram, component, &mut gather);
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
        all.sort_unstable();
        total.finalize_ns = total
            .finalize_ns
            .saturating_add(u64::try_from(finalize_start.elapsed().as_nanos()).unwrap_or(u64::MAX));
        total.output_len = u64::try_from(all.len()).unwrap_or(u64::MAX);
        Ok((all, total))
    }
    /// The profiled connected case: the original single-walk stage split, plus
    /// the walk's node count so the decomposed path can charge one budget.
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
        let nodes =
            collect_coset_minima_with(gram, radius_sq, node_budget, &mut minima, &mut sink)?;
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
}

/// The component decomposition over caller-owned index buffers: `seen` and
/// `newly_seen` are reused across calls, and finished components accumulate
/// in `components`, whose inner vectors are cleared and refilled rather
/// than reallocated.
///
/// Only the leading entries are used; the caller sizes every buffer at
/// least for the lattice dimension.
fn orthogonal_components_into<T: Int>(
    gram: &Gram<T>,
    seen: &mut Vec<bool>,
    newly_seen: &mut Vec<usize>,
    components: &mut Vec<Vec<usize>>,
) {
    let n = gram.dim();
    seen.clear();
    seen.resize(n, false);
    newly_seen.clear();
    let mut count = 0;
    for start in 0..n {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        if count == components.len() {
            components.push(Vec::new());
        }
        let current = &mut components[count];
        current.clear();
        current.push(start);
        let mut cursor = 0;
        while cursor < current.len() {
            let i = current[cursor];
            cursor += 1;
            newly_seen.clear();
            for (j, &visited) in seen.iter().enumerate() {
                if !visited && j != i && !gram.entry(i, j).is_zero() {
                    newly_seen.push(j);
                }
            }
            for &j in newly_seen.iter() {
                seen[j] = true;
                current.push(j);
            }
        }
        current.sort_unstable();
        count += 1;
    }
    components.truncate(count);
}

/// The component Gram gathered into a caller-owned buffer: `gather` is
/// refilled rather than reallocated. The returned [`Gram`] still copies
/// once through its checked constructor, which is unavoidable without
/// changing what a Gram owns.
fn component_gram_into<T: Int>(
    gram: &Gram<T>,
    component: &[usize],
    gather: &mut Vec<T>,
) -> Gram<T> {
    let k = component.len();
    gather.clear();
    gather.resize(k * k, T::ZERO);
    for (r, &i) in component.iter().enumerate() {
        for (c, &j) in component.iter().enumerate() {
            gather[r * k + c] = gram.entry(i, j);
        }
    }
    // As in `component_gram`: a principal submatrix is square and symmetric,
    // and `k <= MAX_RELEVANT_DIM` is far below the dimension limit.
    Gram::from_rows(k, gather).expect("a principal submatrix of a Gram matrix")
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
    let mut representative = Vec::new();
    radius_for_parity_ball_into(gram, &mut representative)
}

/// The parity-coset count and radius over a caller-owned probe: the 0/1
/// representative vector is refilled rather than reallocated.
fn radius_for_parity_ball_into<T: Int>(
    gram: &Gram<T>,
    representative: &mut Vec<T>,
) -> Result<(usize, i128), EnumerationError> {
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
    representative.clear();
    representative.resize(n, T::ZERO);
    let mut radius_sq = 0i128;
    for mask in 1..coset_count {
        for (i, value) in representative.iter_mut().enumerate() {
            *value = if mask & (1 << i) == 0 {
                T::ZERO
            } else {
                T::ONE
            };
        }
        radius_sq = radius_sq.max(gram.norm_sq(representative)?.widen());
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
    for mask in 1..minima.cosets {
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
    // Relevant vectors are distinct, so the stable order and the unstable
    // order coincide; the unstable sort only drops the merge allocation.
    relevant.sort_unstable();
    relevant
}

/// Sink receiving the classification events of the coset pass.
trait CosetSink {
    fn emission(&mut self) {}
    fn reset(&mut self) {}
    fn tie(&mut self) {}
}

struct NoSink;

impl CosetSink for NoSink {}

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
    use super::unstable::relevant_vectors_profiled;
    use super::{MAX_RELEVANT_DIM, relevant_vectors};
    use crate::basis::Gram;
    use crate::error::EnumerationError;
    use crate::error::RangeError;
    use crate::named::{a_n, d_n, e8, zn};
    use crate::shortvec::DEFAULT_NODE_BUDGET;

    #[test]
    fn a_zero_dimensional_lattice_has_no_relevant_vectors() {
        let empty = Gram::<i64>::from_rows(0, &[]).unwrap();
        assert_eq!(
            relevant_vectors(&empty, 1 << 8).unwrap(),
            Vec::<Vec<i128>>::new()
        );
        let (v, stats) = relevant_vectors_profiled(&empty, 1 << 8).unwrap();
        assert!(v.is_empty());
        assert_eq!(stats.masks, 0);
    }

    /// The profiled path returns the same vectors as the public one, with
    /// counters that partition the walk.
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

    /// The reusable scratch returns the one-shot's vectors twice in a row
    /// over the same buffers, on connected and decomposed lattices alike.
    #[test]
    fn scratch_matches_the_one_shot() {
        use super::unstable::RelevantScratch;
        use crate::named::{a_n, d_n, e8, zn};
        for gram in [
            zn::<i64>(4).unwrap(),
            a_n::<i64>(4).unwrap(),
            d_n::<i64>(4).unwrap(),
            e8::<i64>().unwrap(),
        ] {
            let plain = relevant_vectors(&gram, 1 << 20).unwrap();
            let mut scratch = RelevantScratch::new(gram.dim()).unwrap();
            for _ in 0..2 {
                assert_eq!(scratch.relevant_vectors(&gram, 1 << 20).unwrap(), plain);
            }
        }

        // A dimension mismatch is rejected before any walk runs.
        let mut scratch = RelevantScratch::<i64>::new(4).unwrap();
        assert!(matches!(
            scratch.relevant_vectors(&zn(2).unwrap(), 1 << 20),
            Err(EnumerationError::Range(RangeError::Shape { .. }))
        ));
    }
}

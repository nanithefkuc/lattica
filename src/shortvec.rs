//! Exact enumeration of short lattice vectors over an integral Gram matrix.
//!
//! # What makes this exact
//!
//! Enumeration needs a triangular factorization to prune with, and the obvious
//! one — a floating-point Cholesky — makes completeness a numerical argument
//! rather than a proof: a bound that is slightly too tight silently drops
//! vectors, and a dropped vector is a wrong kissing number that looks like a
//! plausible one.
//!
//! Instead this uses the fraction-free (Bareiss) factorization, which is
//! integral. Writing `D_k` for the `k`-by-`k` leading principal minor of `G`
//! with `D_0 = 1`, and `U` for the Bareiss upper-triangular form, the quadratic
//! form separates exactly:
//!
//! ```text
//! c G cᵀ  =  Σ_k  S_k² / (D_k · D_{k+1}),    S_k = Σ_{j ≥ k} U[k][j] · c_j
//! ```
//!
//! with every `S_k`, `U[k][j]` and `D_k` an integer. Scaling by
//! `L = lcm_k(D_k · D_{k+1})` clears the denominators, so the pruning test at
//! every node is a comparison between two integers. There is no rounding
//! anywhere, and completeness is a consequence of the arithmetic rather than of
//! an error budget.
//!
//! Positive-definiteness falls out of the same factorization: `G` is positive
//! definite exactly when every `D_k` is positive (Sylvester), which the
//! factorization computes on the way past.
//!
//! # Relationship to the decoder
//!
//! This is not the decoder. It is exact, complete, integral, and enumerates a
//! *coordinate* ball; the Schnorr–Euchner enumeration that decoding needs works
//! over a real basis against a received vector and trades completeness for a
//! bounded node count. They solve different problems, and this one is the
//! oracle the other will be tested against.

use crate::basis::Gram;
use crate::error::{EnumerationError, Op, RangeError};
use crate::int::Int;

/// Default ceiling on nodes visited by one enumeration.
///
/// Generous enough that no well-conditioned lattice of interest approaches it,
/// low enough that a mistaken radius fails in seconds instead of hanging
/// (invariant I6).
pub const DEFAULT_NODE_BUDGET: u64 = 1 << 28;

/// A count of the short vectors of a lattice.
///
/// "Short" means nonzero with squared norm at most the enumeration radius. Each
/// vector and its negation are counted separately, matching the usual
/// convention for the kissing number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Census<T: Int> {
    /// Smallest nonzero squared norm found, or `None` for the zero lattice.
    pub min_norm_sq: Option<T>,
    /// Number of vectors attaining `min_norm_sq`: the kissing number.
    pub kissing_number: u64,
    /// Number of nonzero vectors within the enumeration radius.
    pub total: u64,
    /// Nodes visited, for cost reporting.
    pub nodes: u64,
}

/// Enumerates every nonzero lattice vector of squared norm at most
/// `radius_sq`, calling `visit` with its coordinate vector and exact squared
/// norm.
///
/// The enumeration is complete: no vector within the radius is skipped.
///
/// # Errors
/// - [`EnumerationError::NotALattice`] if `G` is not positive definite, which
///   means it does not describe a lattice.
/// - [`EnumerationError::InvalidRadius`] if `radius_sq` is negative.
/// - [`EnumerationError::EnumerationBudget`] if the node budget is exhausted.
/// - [`EnumerationError::Range`] if an intermediate exceeds `i128`.
///
/// # Examples
///
/// ```
/// use lattica::basis::Gram;
/// use lattica::shortvec::{DEFAULT_NODE_BUDGET, for_each_short};
///
/// let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
/// let mut count = 0;
/// for_each_short(&g, 2, DEFAULT_NODE_BUDGET, |_coords, norm_sq| {
///     assert_eq!(norm_sq, 2);
///     count += 1;
/// }).unwrap();
/// assert_eq!(count, 6);
/// ```
pub fn for_each_short<T, F>(
    gram: &Gram<T>,
    radius_sq: i128,
    budget: u64,
    visit: F,
) -> Result<u64, EnumerationError>
where
    T: Int,
    F: FnMut(&[i128], i128),
{
    for_each_short_observed(gram, radius_sq, budget, visit, &mut Unobserved)
}

trait EnumerationObserver {
    fn node(&mut self) {}
    fn leaf(&mut self) {}
    fn tail_term(&mut self) {}
    fn leaf_norm(&mut self) {}
}

struct Unobserved;

impl EnumerationObserver for Unobserved {}

impl<O: EnumerationObserver> EnumerationObserver for &mut O {
    fn node(&mut self) {
        (**self).node();
    }

    fn leaf(&mut self) {
        (**self).leaf();
    }

    fn tail_term(&mut self) {
        (**self).tail_term();
    }

    fn leaf_norm(&mut self) {
        (**self).leaf_norm();
    }
}

/// Reusable enumeration buffers: the widened Gram factored in place, the
/// cleared-denominator weights, and the depth-first coordinates.
struct Buffers {
    upper: Vec<i128>,
    weights: Vec<i128>,
    coords: Vec<i128>,
}

impl Buffers {
    fn new(n: usize) -> Self {
        Self {
            upper: vec![0i128; n * n],
            weights: vec![0i128; n],
            coords: vec![0i128; n],
        }
    }
}

fn for_each_short_observed<T, F, O>(
    gram: &Gram<T>,
    radius_sq: i128,
    budget: u64,
    visit: F,
    observer: &mut O,
) -> Result<u64, EnumerationError>
where
    T: Int,
    F: FnMut(&[i128], i128),
    O: EnumerationObserver,
{
    let n = gram.dim();
    if n == 0 {
        return Ok(0);
    }
    if radius_sq < 0 {
        // A negative squared radius is caller error. Answering it with an
        // empty enumeration would present nonsense as a proved fact.
        return Err(EnumerationError::InvalidRadius { radius_sq });
    }
    let mut buffers = Buffers::new(n);
    enumerate_with(&mut buffers, gram, radius_sq, budget, visit, observer)
}

/// Depth-first enumeration over caller-owned buffers, sized at least for
/// `gram.dim()`: only the leading entries are read and written.
///
/// The Gram's dimension, the empty lattice, and the radius are validated by
/// the caller before any buffer is written.
fn enumerate_with<T, F, O>(
    buffers: &mut Buffers,
    gram: &Gram<T>,
    radius_sq: i128,
    budget: u64,
    visit: F,
    observer: &mut O,
) -> Result<u64, EnumerationError>
where
    T: Int,
    F: FnMut(&[i128], i128),
    O: EnumerationObserver,
{
    let n = gram.dim();
    if n == 0 {
        return Ok(0);
    }
    let scale = factor_into(&mut buffers.upper, &mut buffers.weights, gram, n)?;
    let limit = mul(radius_sq, scale)?;

    let Buffers {
        upper,
        weights,
        coords,
    } = buffers;
    let mut walk = Walk {
        upper,
        weights,
        scale,
        n,
        limit,
        coords: &mut coords[..],
        budget,
        nodes: 0,
        observer,
        visit,
    };
    walk.descend(n, 0, 0)?;
    Ok(walk.nodes)
}
/// Finds the minimal squared norm and the kissing number of a lattice.
///
/// The enumeration radius is the smallest diagonal entry of `G`, which is the
/// squared norm of an actual basis vector and therefore an upper bound on the
/// minimal distance. Nothing is supplied by the caller and nothing is assumed
/// about the lattice: the constants come out of the enumeration.
///
/// # Errors
///
/// As [`for_each_short`].
///
/// # Panics
///
/// Never: the only `expect` is guarded by the zero-dimension early return
/// immediately above it.
pub fn census<T: Int>(gram: &Gram<T>, budget: u64) -> Result<Census<T>, EnumerationError> {
    census_core(&mut Buffers::new(gram.dim()), gram, budget, &mut Unobserved)
}

/// The census tally over caller-owned buffers: the diagonal-radius prologue
/// shared by [`census`], [`census_profiled`], and the reusable scratch.
///
/// Radius selection reads only the Gram; buffers are first written by the
/// factorization inside, which is also where a non-lattice is rejected.
fn census_core<T: Int, O: EnumerationObserver>(
    buffers: &mut Buffers,
    gram: &Gram<T>,
    budget: u64,
    observer: &mut O,
) -> Result<Census<T>, EnumerationError> {
    let n = gram.dim();
    if n == 0 {
        return Ok(Census {
            min_norm_sq: None,
            kissing_number: 0,
            total: 0,
            nodes: 0,
        });
    }

    let radius_sq = (0..n)
        .map(|i| gram.entry(i, i).widen())
        .min()
        .expect("dimension is nonzero");

    // Each diagonal entry is the squared norm of a basis vector, so a
    // positive-definite form has a strictly positive minimum. A nonpositive
    // one proves the input is not a lattice before a walk could report a
    // vacuous census for it.
    if radius_sq <= 0 {
        return Err(EnumerationError::NotALattice);
    }

    let mut best = i128::MAX;
    let mut at_best = 0u64;
    let mut total = 0u64;
    let nodes = enumerate_with(
        buffers,
        gram,
        radius_sq,
        budget,
        |_, norm_sq| {
            total += 1;
            if norm_sq < best {
                best = norm_sq;
                at_best = 1;
            } else if norm_sq == best {
                at_best += 1;
            }
        },
        observer,
    )?;

    let min_norm_sq = if total == 0 {
        None
    } else {
        Some(T::narrow(best)?)
    };
    Ok(Census {
        min_norm_sq,
        kissing_number: if total == 0 { 0 } else { at_best },
        total,
        nodes,
    })
}

/// Factors the widened Gram into caller-owned buffers: `upper` holds the
/// Gram on entry and the Bareiss upper-triangular form on success, and
/// `weights` holds the cleared denominators. Returns the denominator scale.
///
/// Only the leading `n`-by-`n` entries of `upper` and the leading `n`
/// entries of `weights` are written; the caller sizes them at least so.
fn factor_into<T: Int>(
    upper: &mut [i128],
    weights: &mut [i128],
    gram: &Gram<T>,
    n: usize,
) -> Result<i128, EnumerationError> {
    for i in 0..n {
        for j in 0..n {
            upper[i * n + j] = gram.entry(i, j).widen();
        }
    }
    let mut prev = 1i128;
    for k in 0..n {
        let pivot = upper[k * n + k];
        // Sylvester: a symmetric matrix is positive definite exactly when
        // every leading principal minor is positive. Bareiss produces them
        // on the diagonal, so the check is free here.
        if pivot <= 0 {
            return Err(EnumerationError::NotALattice);
        }
        for i in k + 1..n {
            let leading = upper[i * n + k];
            for j in k + 1..n {
                let cross = sub(
                    mul(upper[i * n + j], pivot)?,
                    mul(leading, upper[k * n + j])?,
                )?;
                upper[i * n + j] = exact_div(cross, prev)?;
            }
            upper[i * n + k] = 0;
        }
        prev = pivot;
    }

    // Denominators D_k * D_{k+1}, with D_0 = 1, formed in the weights
    // buffer and turned into cleared weights in place.
    let mut previous_minor = 1i128;
    for k in 0..n {
        let minor = upper[k * n + k];
        weights[k] = mul(previous_minor, minor)?;
        previous_minor = minor;
    }

    let mut scale = 1i128;
    for &denominator in weights.iter().take(n) {
        scale = lcm(scale, denominator)?;
    }
    for weight in weights.iter_mut().take(n) {
        *weight = exact_div(scale, *weight)?;
    }
    Ok(scale)
}

/// Depth-first traversal state.
struct Walk<'a, F, O> {
    upper: &'a [i128],
    weights: &'a [i128],
    scale: i128,
    n: usize,
    limit: i128,
    coords: &'a mut [i128],
    budget: u64,
    nodes: u64,
    observer: O,
    visit: F,
}

impl<F: FnMut(&[i128], i128), O: EnumerationObserver> Walk<'_, F, O> {
    /// Chooses coordinate `remaining - 1`, with `acc` the scaled partial norm
    /// and `tail` the exact suffix sum of the coordinates already fixed.
    ///
    /// The caller forms that suffix sum once for the whole sibling group, so
    /// no node recomputes an dot product that its parent could amortize.
    fn descend(&mut self, remaining: usize, acc: i128, tail: i128) -> Result<(), EnumerationError> {
        self.nodes += 1;
        self.observer.node();
        if self.nodes > self.budget {
            return Err(EnumerationError::EnumerationBudget { nodes: self.nodes });
        }

        if remaining == 0 {
            self.observer.leaf();
            // Only the leading entries belong to this walk: oversized
            // scratch buffers keep stale coordinates past them.
            let coords = &self.coords[..self.n];
            if coords.iter().all(|&c| c == 0) {
                return Ok(());
            }
            // Every level contributed `S_k² · weights[k]` on the way down,
            // so `acc` now holds exactly `c G cᵀ · scale`.
            self.observer.leaf_norm();
            let norm_sq = exact_div(acc, self.scale)?;
            (self.visit)(coords, norm_sq);
            return Ok(());
        }
        let k = remaining - 1;
        let n = self.n;
        let diagonal = self.upper[k * n + k];

        // S_k² · weights[k] ≤ limit - acc, so |S_k| ≤ isqrt((limit - acc) / w).
        let room = sub(self.limit, acc)?;
        if room < 0 {
            return Ok(());
        }
        let bound = isqrt(room / self.weights[k]);

        let lo = ceil_div(sub(neg(bound)?, tail)?, diagonal);
        let hi = floor_div(sub(bound, tail)?, diagonal);

        // Every child at level `k - 1` needs the row-`k - 1` suffix sum over
        // the coordinates above `k`, which are frozen throughout this value
        // loop. Form it once; each child's own tail is then one rank-one
        // update with its chosen value.
        let child_base = if k > 0 {
            let mut base = 0i128;
            for j in k + 1..n {
                if self.coords[j] != 0 {
                    self.observer.tail_term();
                    base = add(base, mul(self.upper[(k - 1) * n + j], self.coords[j])?)?;
                }
            }
            base
        } else {
            0
        };
        let child_diagonal = if k > 0 {
            self.upper[(k - 1) * n + k]
        } else {
            0
        };

        let mut value = lo;
        while value <= hi {
            let s = add(mul(diagonal, value)?, tail)?;
            let next = add(acc, mul(mul(s, s)?, self.weights[k])?)?;
            if next <= self.limit {
                self.coords[k] = value;
                let child_tail = add(mul(child_diagonal, value)?, child_base)?;
                self.descend(k, next, child_tail)?;
            }
            value += 1;
        }
        self.coords[k] = 0;
        Ok(())
    }
}

fn overflow(op: Op) -> RangeError {
    RangeError::Overflow {
        op,
        width_bits: 128,
    }
}

fn add(a: i128, b: i128) -> Result<i128, RangeError> {
    a.checked_add(b).ok_or_else(|| overflow(Op::Add))
}

fn sub(a: i128, b: i128) -> Result<i128, RangeError> {
    a.checked_sub(b).ok_or_else(|| overflow(Op::Sub))
}

fn mul(a: i128, b: i128) -> Result<i128, RangeError> {
    a.checked_mul(b).ok_or_else(|| overflow(Op::Mul))
}

fn neg(a: i128) -> Result<i128, RangeError> {
    a.checked_neg().ok_or_else(|| overflow(Op::Neg))
}

fn exact_div(a: i128, b: i128) -> Result<i128, RangeError> {
    if b == 0 {
        return Err(overflow(Op::Div));
    }
    if a % b != 0 {
        return Err(RangeError::InexactDivision);
    }
    Ok(a / b)
}

fn lcm(a: i128, b: i128) -> Result<i128, RangeError> {
    if a == 0 || b == 0 {
        return Ok(0);
    }
    let (left, right) = (a.abs(), b.abs());
    let mut x = left;
    let mut y = right;
    while y != 0 {
        let remainder = x % y;
        x = y;
        y = remainder;
    }
    mul(left / x, right)
}

fn isqrt(value: i128) -> i128 {
    let magnitude = u128::try_from(value).unwrap_or(0);
    i128::try_from(magnitude.isqrt()).unwrap_or(i128::MAX)
}

fn floor_div(a: i128, b: i128) -> i128 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

fn ceil_div(a: i128, b: i128) -> i128 {
    -floor_div(-a, b)
}
/// Unstable short-vector enumeration surface: benchmark counters and reusable
/// buffers. Reachable externally only through the `internals` facade.
// Unstable items are reachable only through the `internals` facade, so the
// library target without that feature reports them as unused.
#[allow(dead_code)]
pub(crate) mod unstable {
    use super::{
        Buffers, Census, EnumerationError, EnumerationObserver, Gram, Int, RangeError, Unobserved,
        census_core, enumerate_with, for_each_short_observed,
    };
    /// Enumerates while returning unstable benchmark counters.
    ///
    /// Reachable only through the `internals` facade; counters are not a compatibility promise.
    /// The returned node count matches [`for_each_short`](super::for_each_short); allocation counts are
    /// measured externally by the benchmark harness, not here.
    ///
    /// # Errors
    ///
    /// As [`for_each_short`](super::for_each_short).
    pub fn for_each_short_profiled<T, F>(
        gram: &Gram<T>,
        radius_sq: i128,
        budget: u64,
        visit: F,
    ) -> Result<(u64, EnumerationStats), EnumerationError>
    where
        T: Int,
        F: FnMut(&[i128], i128),
    {
        let mut stats = EnumerationStats::default();
        let nodes = for_each_short_observed(gram, radius_sq, budget, visit, &mut stats)?;
        Ok((nodes, stats))
    }
    /// Internal operation counters for exact short-vector enumeration benchmarks.
    ///
    /// Reachable only through the `internals` facade; counters are not a compatibility promise.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct EnumerationStats {
        /// Depth-first nodes visited: every recursion entry of the walk.
        pub nodes: u64,
        /// Complete coordinate assignments evaluated, including the all-zero one.
        pub leaves: u64,
        /// Exact multiply-add terms formed by the amortized per-level tail sums.
        pub tail_terms: u64,
        /// Exact norms derived at leaves from the accumulated scaled sum.
        pub leaf_norms: u64,
    }

    impl EnumerationObserver for EnumerationStats {
        fn node(&mut self) {
            self.nodes += 1;
        }

        fn leaf(&mut self) {
            self.leaves += 1;
        }

        fn tail_term(&mut self) {
            self.tail_terms += 1;
        }

        fn leaf_norm(&mut self) {
            self.leaf_norms += 1;
        }
    }
    /// Counts short vectors while returning unstable benchmark counters.
    ///
    /// Reachable only through the `internals` facade; counters are not a compatibility promise.
    /// The [`Census`] matches [`census`](super::census) exactly on the same input.
    ///
    /// # Errors
    ///
    /// As [`census`](super::census).
    ///
    /// # Panics
    ///
    /// Never: the only `expect` is guarded by the zero-dimension early return
    /// immediately above it.
    pub fn census_profiled<T: Int>(
        gram: &Gram<T>,
        budget: u64,
    ) -> Result<(Census<T>, EnumerationStats), EnumerationError> {
        let mut buffers = Buffers::new(gram.dim());
        let mut stats = EnumerationStats::default();
        let census = census_core(&mut buffers, gram, budget, &mut stats)?;
        Ok((census, stats))
    }
    /// Reusable exact-enumeration buffers for one dimension.
    ///
    /// The scratch holds the factored form's working storage across calls: the
    /// widened Gram factored in place, the cleared-denominator weights, and the
    /// depth-first coordinates. Each call re-factors its Gram, so the scratch
    /// carries no lattice between calls — only the allocation.
    ///
    /// A rejected call leaves the scratch reusable, though not untouched: the
    /// factorization's positive-definiteness test, fallible narrowing, i128
    /// overflow, and budget exhaustion can all fire after buffers are written.
    /// None of that state is observable — only [`dim`](Self::dim) is exposed —
    /// and no later call can read it: every coordinates read follows a write
    /// on the current walk.
    ///
    /// Reachable only through the `internals` facade; not a compatibility promise.
    pub struct EnumerationScratch {
        dim: usize,
        buffers: Buffers,
    }

    impl EnumerationScratch {
        /// Allocates enumeration buffers for `dimension`.
        ///
        /// # Errors
        ///
        /// [`RangeError::Dimension`] if `dimension` exceeds the crate's maximum
        /// matrix dimension.
        pub fn new(dimension: usize) -> Result<Self, EnumerationError> {
            if dimension > crate::int::MAX_DIM {
                return Err(RangeError::Dimension {
                    requested: dimension,
                    max: crate::int::MAX_DIM,
                }
                .into());
            }
            Ok(Self {
                dim: dimension,
                buffers: Buffers::new(dimension),
            })
        }

        /// The dimension this scratch was sized for.
        #[must_use]
        pub const fn dim(&self) -> usize {
            self.dim
        }

        /// Enumerates short vectors over reused buffers, identical to
        /// [`for_each_short`](super::for_each_short) on the input.
        ///
        /// # Errors
        ///
        /// [`RangeError::Shape`] if `gram.dim()` does not equal
        /// [`Self::dim`]; otherwise as [`for_each_short`](super::for_each_short).
        pub fn for_each<T, F>(
            &mut self,
            gram: &Gram<T>,
            radius_sq: i128,
            budget: u64,
            visit: F,
        ) -> Result<u64, EnumerationError>
        where
            T: Int,
            F: FnMut(&[i128], i128),
        {
            if gram.dim() != self.dim {
                return Err(RangeError::Shape {
                    expected: self.dim,
                    found: gram.dim(),
                }
                .into());
            }
            if gram.dim() == 0 {
                return Ok(0);
            }
            if radius_sq < 0 {
                return Err(EnumerationError::InvalidRadius { radius_sq });
            }
            enumerate_with(
                &mut self.buffers,
                gram,
                radius_sq,
                budget,
                visit,
                &mut Unobserved,
            )
        }

        /// Counts short vectors over reused buffers, identical to [`census`](super::census) on
        /// the input.
        ///
        /// # Errors
        ///
        /// [`RangeError::Shape`] if `gram.dim()` does not equal
        /// [`Self::dim`]; otherwise as [`census`](super::census).
        pub fn census<T: Int>(
            &mut self,
            gram: &Gram<T>,
            budget: u64,
        ) -> Result<Census<T>, EnumerationError> {
            if gram.dim() != self.dim {
                return Err(RangeError::Shape {
                    expected: self.dim,
                    found: gram.dim(),
                }
                .into());
            }
            census_core(&mut self.buffers, gram, budget, &mut Unobserved)
        }

        /// Enumerates over the leading entries of oversized buffers: the
        /// component case of a decomposed lattice, where the stride is the
        /// component dimension rather than the scratch dimension.
        ///
        /// Crate-internal: the caller guarantees `gram.dim()` fits the buffers.
        /// Only the relevant-vector scratch calls this, which sizes its buffers
        /// for the whole lattice.
        ///
        /// # Errors
        ///
        /// As [`for_each`](EnumerationScratch::for_each).
        pub(crate) fn for_each_prefix<T, F>(
            &mut self,
            gram: &Gram<T>,
            radius_sq: i128,
            budget: u64,
            visit: F,
        ) -> Result<u64, EnumerationError>
        where
            T: Int,
            F: FnMut(&[i128], i128),
        {
            if gram.dim() > self.dim {
                return Err(RangeError::Shape {
                    expected: self.dim,
                    found: gram.dim(),
                }
                .into());
            }
            if radius_sq < 0 {
                return Err(EnumerationError::InvalidRadius { radius_sq });
            }
            enumerate_with(
                &mut self.buffers,
                gram,
                radius_sq,
                budget,
                visit,
                &mut Unobserved,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_NODE_BUDGET, census, for_each_short};
    use crate::basis::Gram;
    use crate::error::EnumerationError;
    use crate::error::RangeError;

    #[test]
    fn the_integer_lattice_has_two_minimal_vectors_per_axis() {
        for n in 1..=6 {
            let mut data = vec![0i64; n * n];
            for i in 0..n {
                data[i * n + i] = 1;
            }
            let g = Gram::from_rows(n, &data).unwrap();
            let c = census(&g, DEFAULT_NODE_BUDGET).unwrap();
            assert_eq!(c.min_norm_sq, Some(1));
            assert_eq!(c.kissing_number, 2 * u64::try_from(n).unwrap());
        }
    }

    #[test]
    fn hexagonal_lattice_has_six_minimal_vectors() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        let c = census(&g, DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(c.min_norm_sq, Some(2));
        assert_eq!(c.kissing_number, 6);
        assert_eq!(c.total, 6);
    }

    #[test]
    fn every_emitted_vector_has_the_norm_it_reports() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        for_each_short(&g, 6, DEFAULT_NODE_BUDGET, |coords, norm_sq| {
            let narrow: Vec<i64> = coords.iter().map(|&v| i64::try_from(v).unwrap()).collect();
            assert_eq!(g.norm_sq(&narrow).unwrap(), i64::try_from(norm_sq).unwrap());
            assert!(norm_sq > 0 && norm_sq <= 6);
        })
        .unwrap();
    }

    #[test]
    fn an_indefinite_form_is_not_a_lattice() {
        let g = Gram::<i64>::from_rows(2, &[1, 2, 2, 1]).unwrap();
        assert_eq!(
            census(&g, DEFAULT_NODE_BUDGET),
            Err(EnumerationError::NotALattice)
        );
    }

    #[test]
    fn a_nonpositive_diagonal_is_not_a_vacuous_census() {
        // Before the guard, the negative minimum diagonal flowed through as a
        // negative radius and produced `Ok` with an empty census.
        let negative = Gram::<i64>::from_rows(1, &[-1]).unwrap();
        assert_eq!(
            census(&negative, DEFAULT_NODE_BUDGET),
            Err(EnumerationError::NotALattice)
        );
        let zero = Gram::<i64>::from_rows(2, &[0, 0, 0, 1]).unwrap();
        assert_eq!(
            census(&zero, DEFAULT_NODE_BUDGET),
            Err(EnumerationError::NotALattice)
        );
    }

    #[test]
    fn a_negative_radius_is_an_error_not_an_empty_enumeration() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        assert_eq!(
            for_each_short(&g, -1, DEFAULT_NODE_BUDGET, |_, _| {}),
            Err(EnumerationError::InvalidRadius { radius_sq: -1 })
        );
    }

    #[test]
    fn the_node_budget_is_enforced() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        let r = for_each_short(&g, 10_000, 16, |_, _| {});
        assert!(matches!(r, Err(EnumerationError::EnumerationBudget { .. })));
    }

    #[test]
    fn a_zero_radius_finds_nothing() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        let mut seen = 0;
        for_each_short(&g, 0, DEFAULT_NODE_BUDGET, |_, _| seen += 1).unwrap();
        assert_eq!(seen, 0);
    }

    #[test]
    fn profiled_counters_partition_the_walk() {
        use super::unstable::{EnumerationStats, census_profiled, for_each_short_profiled};
        use crate::named::{e8, zn};

        // A zero-dimensional Gram short-circuits every entry point.
        let empty = Gram::<i64>::from_rows(0, &[]).unwrap();
        let public = census(&empty, DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(public.total, 0);
        assert_eq!(public.min_norm_sq, None);
        let (census, stats) = census_profiled(&empty, DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(census.total, 0);
        assert_eq!(stats, EnumerationStats::default());
        assert_eq!(
            for_each_short_profiled(&empty, 4, 8, |_, _| {}).unwrap(),
            (0, stats)
        );

        // Z^4's census runs at the minimal diagonal, radius one: exactly the
        // axis vectors ±e_i survive.
        let g = zn::<i64>(4).unwrap();
        let (census, stats) = census_profiled(&g, DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(census.total, 8);
        assert_eq!(census.kissing_number, 8);
        assert_eq!(census.nodes, stats.nodes);
        // One leaf per emitted vector plus the rejected all-zero assignment,
        // and one carried norm per emitted vector.
        assert_eq!(stats.leaves, census.total + 1);
        assert_eq!(stats.leaf_norms, census.total);
        assert!(stats.nodes > stats.leaves);
        assert!(stats.tail_terms > 0);

        // E8's root shell: the published kissing number, recovered.
        let (census, stats) = census_profiled(&e8::<i64>().unwrap(), DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(census.kissing_number, 240);
        assert_eq!(stats.leaf_norms, census.total);
        assert_eq!(stats.leaves, census.total + 1);

        // The radius-controlled profiled path reports the same partition.
        let (nodes, stats) =
            for_each_short_profiled(&e8::<i64>().unwrap(), 2, DEFAULT_NODE_BUDGET, |_, _| {})
                .unwrap();
        assert_eq!(nodes, stats.nodes);
        assert_eq!(stats.leaves, 241);
    }

    #[test]
    fn carried_norms_match_the_direct_quadratic_form() {
        use super::{Unobserved, for_each_short_observed};
        use crate::named::{a_n, d_n};

        // The direct `O(n²)` evaluation through `Gram::norm_sq` is the oracle
        // for every carried norm, across lattices whose factorizations clear
        // denominators differently. Radius three times the minimal diagonal
        // reaches beyond the root shell.
        for gram in [
            crate::named::zn::<i64>(5).unwrap(),
            a_n::<i64>(7).unwrap(),
            d_n::<i64>(6).unwrap(),
            crate::named::e8::<i64>().unwrap(),
            Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap(),
        ] {
            let n = gram.dim();
            let radius_sq = 3
                * (0..n)
                    .map(|i| <i64 as crate::Int>::widen(gram.entry(i, i)))
                    .min()
                    .unwrap();
            let mut checked = 0u64;
            for_each_short_observed(
                &gram,
                radius_sq,
                DEFAULT_NODE_BUDGET,
                |coords, norm_sq| {
                    let narrow: Vec<i64> =
                        coords.iter().map(|&v| i64::try_from(v).unwrap()).collect();
                    assert_eq!(
                        Ok(i64::try_from(norm_sq).unwrap()),
                        gram.norm_sq(&narrow),
                        "coords {coords:?}"
                    );
                    checked += 1;
                },
                &mut Unobserved,
            )
            .unwrap();
            assert!(checked > 10, "only {checked} vectors were exercised");
        }
    }

    /// The reusable scratch visits the same vectors with the same norms and
    /// counts the same census as the one-shots, twice in a row over the
    /// same buffers.
    #[test]
    fn scratch_matches_the_one_shots() {
        use super::unstable::EnumerationScratch;
        use crate::named::{d_n, e8, zn};

        let cases: Vec<(Gram<i64>, i128)> = vec![
            (zn(4).unwrap(), 2),
            (e8().unwrap(), 2),
            (d_n(6).unwrap(), 4),
        ];
        for (gram, radius) in &cases {
            let mut plain = Vec::new();
            for_each_short(gram, *radius, DEFAULT_NODE_BUDGET, |coords, norm| {
                plain.push((coords.to_vec(), norm));
            })
            .unwrap();
            let mut scratch = EnumerationScratch::new(gram.dim()).unwrap();
            for _ in 0..2 {
                let mut reused = Vec::new();
                scratch
                    .for_each(gram, *radius, DEFAULT_NODE_BUDGET, |coords, norm| {
                        reused.push((coords.to_vec(), norm));
                    })
                    .unwrap();
                assert_eq!(reused, plain);
                assert_eq!(
                    scratch.census(gram, DEFAULT_NODE_BUDGET).unwrap(),
                    census(gram, DEFAULT_NODE_BUDGET).unwrap()
                );
            }
        }

        // A dimension mismatch and a negative radius are rejected before
        // any walk runs.
        let mut scratch = EnumerationScratch::new(4).unwrap();
        let small = zn::<i64>(2).unwrap();
        assert!(matches!(
            scratch.for_each(&small, 2, DEFAULT_NODE_BUDGET, |_, _| {}),
            Err(EnumerationError::Range(RangeError::Shape { .. }))
        ));
        assert!(matches!(
            scratch.census(&small, DEFAULT_NODE_BUDGET),
            Err(EnumerationError::Range(RangeError::Shape { .. }))
        ));
        let big = zn::<i64>(4).unwrap();
        assert!(matches!(
            scratch.for_each(&big, -1, DEFAULT_NODE_BUDGET, |_, _| {}),
            Err(EnumerationError::InvalidRadius { .. })
        ));
    }
}

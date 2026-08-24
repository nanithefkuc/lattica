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
use crate::error::{DecodeError, Op, RangeError};
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
///
/// - [`DecodeError::NotInLattice`] if `G` is not positive definite, which means
///   it does not describe a lattice.
/// - [`DecodeError::EnumerationBudget`] if the node budget is exhausted.
/// - [`DecodeError::Range`] if an intermediate exceeds `i128`.
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
) -> Result<u64, DecodeError>
where
    T: Int,
    F: FnMut(&[i128], i128),
{
    for_each_short_observed(gram, radius_sq, budget, visit, &mut Unobserved)
}

/// Enumerates while returning unstable benchmark counters.
///
/// Available only with `internals`; counters are not a compatibility promise.
/// The returned node count matches [`for_each_short`]; allocation counts are
/// measured externally by the benchmark harness, not here.
///
/// # Errors
///
/// As [`for_each_short`].
#[cfg(feature = "internals")]
pub fn for_each_short_profiled<T, F>(
    gram: &Gram<T>,
    radius_sq: i128,
    budget: u64,
    visit: F,
) -> Result<(u64, EnumerationStats), DecodeError>
where
    T: Int,
    F: FnMut(&[i128], i128),
{
    let mut stats = EnumerationStats::default();
    let nodes = for_each_short_observed(gram, radius_sq, budget, visit, &mut stats)?;
    Ok((nodes, stats))
}

trait EnumerationObserver {
    fn node(&mut self) {}
    fn leaf(&mut self) {}
    fn tail_term(&mut self) {}
    fn direct_norm(&mut self) {}
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

    fn direct_norm(&mut self) {
        (**self).direct_norm();
    }
}

/// Internal operation counters for exact short-vector enumeration benchmarks.
///
/// Available only with `internals`; counters are not a compatibility promise.
#[cfg(feature = "internals")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnumerationStats {
    /// Depth-first nodes visited: every [`Walk::descend`](struct@Walk) entry.
    pub nodes: u64,
    /// Complete coordinate assignments evaluated, including the all-zero one.
    pub leaves: u64,
    /// Exact multiply-add terms formed by per-node tail sums.
    pub tail_terms: u64,
    /// Direct `c G cᵀ` recomputations at emitted vectors.
    pub direct_norms: u64,
}

#[cfg(feature = "internals")]
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

    fn direct_norm(&mut self) {
        self.direct_norms += 1;
    }
}

fn for_each_short_observed<T, F, O>(
    gram: &Gram<T>,
    radius_sq: i128,
    budget: u64,
    visit: F,
    observer: &mut O,
) -> Result<u64, DecodeError>
where
    T: Int,
    F: FnMut(&[i128], i128),
    O: EnumerationObserver,
{
    let n = gram.dim();
    if n == 0 || radius_sq < 0 {
        return Ok(0);
    }

    let widened: Vec<i128> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| gram.entry(i, j).widen())
        .collect();

    let factored = Factorization::new(&widened, n)?;
    let limit = mul(radius_sq, factored.scale)?;

    let mut walk = Walk {
        f: &factored,
        gram: &widened,
        n,
        limit,
        coords: vec![0i128; n],
        budget,
        nodes: 0,
        observer,
        visit,
    };
    walk.descend(n, 0)?;
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
pub fn census<T: Int>(gram: &Gram<T>, budget: u64) -> Result<Census<T>, DecodeError> {
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

    let mut best = i128::MAX;
    let mut at_best = 0u64;
    let mut total = 0u64;
    let nodes = for_each_short(gram, radius_sq, budget, |_, norm_sq| {
        total += 1;
        if norm_sq < best {
            best = norm_sq;
            at_best = 1;
        } else if norm_sq == best {
            at_best += 1;
        }
    })?;

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

/// Counts short vectors while returning unstable benchmark counters.
///
/// Available only with `internals`; counters are not a compatibility promise.
/// The [`Census`] matches [`census`] exactly on the same input.
///
/// # Errors
///
/// As [`census`].
///
/// # Panics
///
/// Never: the only `expect` is guarded by the zero-dimension early return
/// immediately above it.
#[cfg(feature = "internals")]
pub fn census_profiled<T: Int>(
    gram: &Gram<T>,
    budget: u64,
) -> Result<(Census<T>, EnumerationStats), DecodeError> {
    let n = gram.dim();
    if n == 0 {
        return Ok((
            Census {
                min_norm_sq: None,
                kissing_number: 0,
                total: 0,
                nodes: 0,
            },
            EnumerationStats::default(),
        ));
    }

    let radius_sq = (0..n)
        .map(|i| gram.entry(i, i).widen())
        .min()
        .expect("dimension is nonzero");

    let mut best = i128::MAX;
    let mut at_best = 0u64;
    let mut total = 0u64;
    let mut stats = EnumerationStats::default();
    let nodes = for_each_short_observed(
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
        &mut stats,
    )?;

    let min_norm_sq = if total == 0 {
        None
    } else {
        Some(T::narrow(best)?)
    };
    Ok((
        Census {
            min_norm_sq,
            kissing_number: if total == 0 { 0 } else { at_best },
            total,
            nodes,
        },
        stats,
    ))
}

/// The fraction-free triangular factorization and its cleared denominators.
struct Factorization {
    /// Bareiss upper-triangular form, row-major, `n` by `n`.
    upper: Vec<i128>,
    /// `weights[k] = scale / (D_k · D_{k+1})`.
    weights: Vec<i128>,
    /// `lcm_k(D_k · D_{k+1})`.
    scale: i128,
}

impl Factorization {
    fn new(gram: &[i128], n: usize) -> Result<Self, DecodeError> {
        let mut m = gram.to_vec();
        let mut prev = 1i128;
        for k in 0..n {
            let pivot = m[k * n + k];
            // Sylvester: a symmetric matrix is positive definite exactly when
            // every leading principal minor is positive. Bareiss produces them
            // on the diagonal, so the check is free here.
            if pivot <= 0 {
                return Err(DecodeError::NotInLattice);
            }
            for i in k + 1..n {
                let leading = m[i * n + k];
                for j in k + 1..n {
                    let cross = sub(mul(m[i * n + j], pivot)?, mul(leading, m[k * n + j])?)?;
                    m[i * n + j] = exact_div(cross, prev)?;
                }
                m[i * n + k] = 0;
            }
            prev = pivot;
        }

        // Denominators D_k * D_{k+1}, with D_0 = 1.
        let mut denominators = Vec::with_capacity(n);
        let mut previous_minor = 1i128;
        for k in 0..n {
            let minor = m[k * n + k];
            denominators.push(mul(previous_minor, minor)?);
            previous_minor = minor;
        }

        let mut scale = 1i128;
        for &d in &denominators {
            scale = lcm(scale, d)?;
        }
        let weights = denominators
            .iter()
            .map(|&d| exact_div(scale, d))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            upper: m,
            weights,
            scale,
        })
    }
}

/// Depth-first traversal state.
struct Walk<'a, F, O> {
    f: &'a Factorization,
    gram: &'a [i128],
    n: usize,
    limit: i128,
    coords: Vec<i128>,
    budget: u64,
    nodes: u64,
    observer: O,
    visit: F,
}

impl<F: FnMut(&[i128], i128), O: EnumerationObserver> Walk<'_, F, O> {
    /// Chooses coordinate `remaining - 1`, with `acc` the scaled partial norm
    /// contributed by the coordinates already fixed.
    fn descend(&mut self, remaining: usize, acc: i128) -> Result<(), DecodeError> {
        self.nodes += 1;
        self.observer.node();
        if self.nodes > self.budget {
            return Err(DecodeError::EnumerationBudget { nodes: self.nodes });
        }

        if remaining == 0 {
            self.observer.leaf();
            if self.coords.iter().all(|&c| c == 0) {
                return Ok(());
            }
            self.observer.direct_norm();
            let norm_sq = self.exact_norm()?;
            (self.visit)(&self.coords, norm_sq);
            return Ok(());
        }

        let k = remaining - 1;
        let n = self.n;
        let diagonal = self.f.upper[k * n + k];

        // The part of S_k already determined by the coordinates above k.
        let mut tail = 0i128;
        for j in k + 1..n {
            if self.coords[j] != 0 {
                self.observer.tail_term();
                tail = add(tail, mul(self.f.upper[k * n + j], self.coords[j])?)?;
            }
        }

        // S_k² · weights[k] ≤ limit - acc, so |S_k| ≤ isqrt((limit - acc) / w).
        let room = sub(self.limit, acc)?;
        if room < 0 {
            return Ok(());
        }
        let bound = isqrt(room / self.f.weights[k]);

        let lo = ceil_div(sub(neg(bound)?, tail)?, diagonal);
        let hi = floor_div(sub(bound, tail)?, diagonal);

        let mut value = lo;
        while value <= hi {
            let s = add(mul(diagonal, value)?, tail)?;
            let next = add(acc, mul(mul(s, s)?, self.f.weights[k])?)?;
            if next <= self.limit {
                self.coords[k] = value;
                self.descend(k, next)?;
            }
            value += 1;
        }
        self.coords[k] = 0;
        Ok(())
    }

    fn exact_norm(&self) -> Result<i128, RangeError> {
        let n = self.n;
        let mut total = 0i128;
        for i in 0..n {
            let ci = self.coords[i];
            if ci == 0 {
                continue;
            }
            let mut row = 0i128;
            for j in 0..n {
                let cj = self.coords[j];
                if cj != 0 {
                    row = add(row, mul(self.gram[i * n + j], cj)?)?;
                }
            }
            total = add(total, mul(ci, row)?)?;
        }
        Ok(total)
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

#[cfg(test)]
mod tests {
    use super::{DEFAULT_NODE_BUDGET, census, for_each_short};
    use crate::basis::Gram;
    use crate::error::DecodeError;

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
            Err(DecodeError::NotInLattice)
        );
    }

    #[test]
    fn the_node_budget_is_enforced() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        let r = for_each_short(&g, 10_000, 16, |_, _| {});
        assert!(matches!(r, Err(DecodeError::EnumerationBudget { .. })));
    }

    #[test]
    fn a_zero_radius_finds_nothing() {
        let g = Gram::<i64>::from_rows(2, &[2, -1, -1, 2]).unwrap();
        let mut seen = 0;
        for_each_short(&g, 0, DEFAULT_NODE_BUDGET, |_, _| seen += 1).unwrap();
        assert_eq!(seen, 0);
    }

    #[cfg(feature = "internals")]
    #[test]
    fn profiled_counters_partition_the_walk() {
        use super::census_profiled;
        use crate::named::{e8, zn};

        // Z^4's census runs at the minimal diagonal, radius one: exactly the
        // axis vectors ±e_i survive.
        let g = zn::<i64>(4).unwrap();
        let (census, stats) = census_profiled(&g, DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(census.total, 8);
        assert_eq!(census.kissing_number, 8);
        assert_eq!(census.nodes, stats.nodes);
        // One leaf per emitted vector plus the rejected all-zero assignment,
        // and one direct norm recomputation per emitted vector.
        assert_eq!(stats.leaves, census.total + 1);
        assert_eq!(stats.direct_norms, census.total);
        assert!(stats.nodes > stats.leaves);
        assert!(stats.tail_terms > 0);

        // E8's root shell: the published kissing number, recovered.
        let (census, stats) = census_profiled(&e8::<i64>().unwrap(), DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(census.kissing_number, 240);
        assert_eq!(stats.direct_norms, census.total);
        assert_eq!(stats.leaves, census.total + 1);
    }
}

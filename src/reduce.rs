//! Basis reduction: Lagrange–Gauss in the plane, and LLL in general.
//!
//! # Exact, including the parameter
//!
//! Reduction runs on the Gram matrix and never touches ambient coordinates, so
//! the whole computation is integer arithmetic on a [`Gram`] plus a unimodular
//! transform. Even `δ` is exact: [`Delta`] is a rational, so the Lovász test
//!
//! ```text
//! δ ‖b*_{k-1}‖²  ≤  ‖b*_k‖² + μ²_{k,k-1} ‖b*_{k-1}‖²
//! ```
//!
//! clears its denominators into
//!
//! ```text
//! δ_num · d[k]²  ≤  δ_den · (d[k+1]·d[k-1] + λ[k][k-1]²)
//! ```
//!
//! and becomes a comparison between two integers. A floating-point `δ` would
//! make the reduced-basis *predicate* approximate, which is worse than it
//! sounds: the certificate a caller checks would then differ from the one the
//! algorithm enforced.
//!
//! # Exact incremental state
//!
//! Reduction is per-code setup, but it still keeps the symmetric Gram matrix
//! and fraction-free orthogonalization synchronized after every elementary
//! basis operation. Ordinary LLL delays reductions that cannot affect the
//! next Lovász test until that test passes, and proves zero quotients with an
//! exact magnitude comparison before entering checked integer division. Every
//! quotient and decision remains exact.

// The `Delta` bounds check widens two `u32`s; nothing else here leaves the
// integers.
#![allow(clippy::as_conversions)]

use crate::basis::{Basis, Gram};
use crate::error::{LatticeError, RangeError, ReduceError};
use crate::gso::Gso;
use crate::int::{Int, IntMatrix, div_nearest, gcd};

/// The LLL reduction parameter, as an exact rational in `(1/4, 1)`.
///
/// Larger values give better bases and take longer. `3/4` is Lenstra, Lenstra
/// and Lovász's original choice; `99/100` is the usual practical default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Delta {
    numerator: u32,
    denominator: u32,
}

impl Delta {
    /// The original LLL parameter, `3/4`.
    pub const LLL: Self = Self {
        numerator: 3,
        denominator: 4,
    };

    /// The usual practical parameter, `99/100`.
    pub const STRONG: Self = Self {
        numerator: 99,
        denominator: 100,
    };

    /// Builds `numerator / denominator`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Degenerate`] unless `1/4 < δ < 1`. Below `1/4` the
    /// algorithm proves nothing; at `1` it need not terminate.
    pub const fn new(numerator: u32, denominator: u32) -> Result<Self, LatticeError> {
        if denominator == 0
            || numerator >= denominator
            || 4 * (numerator as u64) <= denominator as u64
        {
            return Err(LatticeError::Degenerate);
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    /// The value as an `f64`, for reporting.
    #[must_use]
    pub fn value(self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

/// A reduced Gram matrix and the transform that produced it.
///
/// `transform * original * transformᵀ == gram`, and `|det transform| == 1`.
/// Those two facts are a complete correctness certificate: the lattice is
/// provably the one that went in, checkable without knowing how the reduction
/// ran.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reduced<T: Int> {
    /// The reduced Gram matrix.
    pub gram: Gram<T>,
    /// The unimodular transform, acting on basis vectors as rows.
    pub transform: IntMatrix<T>,
}

impl<T: Int> Reduced<T> {
    /// Applies the transform to an ambient basis, giving the reduced one.
    ///
    /// # Errors
    ///
    /// [`RangeError`] on a shape mismatch or overflow.
    pub fn apply(&self, basis: &Basis<T>) -> Result<Basis<T>, RangeError> {
        let product = self.transform.mul(basis.as_matrix())?;
        Basis::from_rows(product.rows(), product.cols(), product.as_slice())
    }
}

/// Reduces a Gram matrix with LLL.
///
/// The result is size-reduced (`|μ_{i,j}| ≤ ½`) and satisfies the Lovász
/// condition at `delta`. Both are verifiable afterwards with
/// [`is_reduced`], in integers, without trusting this function.
///
/// # Errors
///
/// [`ReduceError::NotFullRank`] if the input is not positive definite, and
/// [`ReduceError::Range`] if an intermediate exceeds the element width. The
/// input is never modified.
///
/// # Examples
///
/// ```
/// use lattica::basis::Gram;
/// use lattica::reduce::{Delta, is_reduced, lll};
///
/// // A badly skewed basis of a lattice of determinant 1.
/// let g = Gram::<i64>::from_rows(2, &[10, 3, 3, 1])?;
/// let reduced = lll(&g, Delta::LLL)?;
///
/// assert!(is_reduced(&reduced.gram, Delta::LLL)?);
/// assert_eq!(reduced.gram.det()?, g.det()?);
/// assert_eq!(reduced.transform.det()?.abs(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn lll<T: Int>(gram: &Gram<T>, delta: Delta) -> Result<Reduced<T>, ReduceError> {
    reduce_with(gram, delta, false)
}

/// Reduces a Gram matrix with deep-insertion LLL.
///
/// Where [`lll`] only ever swaps adjacent vectors, this moves `b_k` all the way
/// down to the first position where it shortens the projected basis. The result
/// is a better basis for the same certificate, at a cost: the insertion test
/// needs the norm of `b_k` projected past the first `i` vectors, which is a
/// ratio of Gram *sub*determinants, so each candidate position costs a
/// determinant. Reduction is precompute; this trade is available when the basis
/// is worth it.
///
/// The output satisfies the same [`is_reduced`] predicate as [`lll`].
///
/// # Errors
///
/// As [`lll`].
pub fn lll_deep<T: Int>(gram: &Gram<T>, delta: Delta) -> Result<Reduced<T>, ReduceError> {
    reduce_with(gram, delta, true)
}

/// Reduces with LLL while returning unstable benchmark counters.
///
/// Available only with `internals`; counters are not a compatibility promise.
///
/// # Errors
///
/// As [`lll`].
#[cfg(feature = "internals")]
pub fn lll_profiled<T: Int>(
    gram: &Gram<T>,
    delta: Delta,
) -> Result<(Reduced<T>, ReductionStats), ReduceError> {
    let mut stats = ReductionStats::default();
    let reduced = reduce_observed(gram, delta, false, &mut stats)?;
    Ok((reduced, stats))
}

/// Reduces with deep-insertion LLL while returning unstable benchmark counters.
///
/// Available only with `internals`; counters are not a compatibility promise.
///
/// # Errors
///
/// As [`lll_deep`].
#[cfg(feature = "internals")]
pub fn lll_deep_profiled<T: Int>(
    gram: &Gram<T>,
    delta: Delta,
) -> Result<(Reduced<T>, ReductionStats), ReduceError> {
    let mut stats = ReductionStats::default();
    let reduced = reduce_observed(gram, delta, true, &mut stats)?;
    Ok((reduced, stats))
}

/// Lagrange–Gauss reduction of a rank-two lattice.
///
/// Returns a basis of two shortest possible vectors: in the plane, unlike in
/// general, reduction is not a heuristic — the output is provably optimal.
///
/// # Errors
///
/// [`ReduceError::Range`] with [`RangeError::Shape`] if the Gram matrix is not
/// two-dimensional, [`ReduceError::NotFullRank`] if it is not positive
/// definite, and [`ReduceError::Range`] on overflow.
pub fn gauss<T: Int>(gram: &Gram<T>) -> Result<Reduced<T>, ReduceError> {
    if gram.dim() != 2 {
        return Err(RangeError::Shape {
            expected: 2,
            found: gram.dim(),
        }
        .into());
    }
    // Without this the loop happily "reduces" an indefinite form forever.
    Gso::new(gram)?;
    let mut state = State::new(gram)?;

    let mut steps = 0u64;
    loop {
        guard(&mut steps, BUDGET)?;
        if state.gram.get(0, 0) > state.gram.get(1, 1) {
            state.swap(0, 1);
        }
        let norm = state.gram.get(0, 0);
        if norm.is_zero() {
            return Err(ReduceError::NotFullRank {
                rank: 0,
                required: 2,
            });
        }
        let quotient = div_nearest(state.gram.get(0, 1), norm)?;
        if quotient.is_zero() {
            break;
        }
        state.subtract(1, 0, quotient)?;
    }
    // Leave the shorter vector first.
    if state.gram.get(0, 0) > state.gram.get(1, 1) {
        state.swap(0, 1);
    }
    Ok(state.finish())
}

/// Checks the reduced-basis certificate: size-reduced, and Lovász at `delta`.
///
/// This is deliberately independent of the reduction itself — it re-derives the
/// orthogonalization from the Gram matrix and tests the two defining
/// inequalities in integers. Passing it is a complete proof that the output is
/// LLL-reduced, whatever path produced it.
///
/// # Errors
///
/// As [`Gso::new`], plus [`ReduceError::Range`] on overflow.
pub fn is_reduced<T: Int>(gram: &Gram<T>, delta: Delta) -> Result<bool, ReduceError> {
    let gso = Gso::new(gram)?;
    if !gso.is_size_reduced()? {
        return Ok(false);
    }
    for k in 1..gram.dim() {
        if !lovasz_holds(&gso, k, delta)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// `δ·d[k]² ≤ d[k+1]·d[k-1] + λ[k][k-1]²`, with `δ` cleared.
fn lovasz_holds<T: Int>(gso: &Gso<T>, k: usize, delta: Delta) -> Result<bool, RangeError> {
    let lambda = gso.lambda(k, k - 1);
    let right = gso
        .minor(k + 1)
        .try_mul(gso.minor(k - 1))?
        .try_add(lambda.try_mul(lambda)?)?;
    let left = gso.minor(k).try_mul(gso.minor(k))?;
    let scaled_left = left.try_mul(T::from_i8(1).try_mul(narrow(delta.numerator)?)?)?;
    let scaled_right = right.try_mul(narrow(delta.denominator)?)?;
    Ok(scaled_left <= scaled_right)
}

fn narrow<T: Int>(v: u32) -> Result<T, RangeError> {
    T::narrow(i128::from(v))
}

/// Iteration ceiling for the reduction loops.
///
/// LLL terminates by a potential argument, so this should never fire. It exists
/// because a *bug* in the descent must surface as an error rather than as a
/// hang, and because invariant I6 asks every search in this crate to be
/// bounded. It cost nothing to add and caught a wrong deep-insertion condition
/// during development.
const BUDGET: u64 = 1 << 22;

fn guard(steps: &mut u64, budget: u64) -> Result<(), ReduceError> {
    *steps += 1;
    if *steps > budget {
        return Err(ReduceError::BudgetExhausted { steps: *steps });
    }
    Ok(())
}

trait ReductionObserver {
    fn gram_copy(&mut self) {}
    fn factorization(&mut self, _dimension: usize) {}
    fn iteration(&mut self) {}
    fn quotient_check(&mut self, _zero_proved: bool) {}
    fn size_reduction(&mut self, _checked_updates: u64) {}
    fn swaps(&mut self, _count: u64, _update_terms: u64, _checked_updates: u64) {}
    fn deep_insertion(&mut self) {}
    fn deep_predicate_term(&mut self, _scale_rescaled: bool, _exact_divisions: u64) {}
    fn deep_denominator<T: Int>(&mut self, _denominator: T, _scale: T) {}
}

struct Unobserved;

impl ReductionObserver for Unobserved {}

/// Internal operation counters for exact basis reduction benchmarks.
///
/// `checked_updates` counts checked integer operations in factorization and
/// elementary state recurrences. Quotient and adjacent-swap counters expose
/// the work that those recurrences do not distinguish.
#[cfg(feature = "internals")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReductionStats {
    /// Full exact factorizations.
    pub factorizations: u64,
    /// Completed reduction-loop iterations.
    pub iterations: u64,
    /// Size-reduction coefficients examined.
    pub size_reduction_checks: u64,
    /// Coefficients proved to have a zero nearest quotient without division.
    pub zero_quotients: u64,
    /// Coefficients that entered checked nearest division.
    pub quotient_divisions: u64,
    /// Nonzero size reductions.
    pub size_reductions: u64,
    /// Adjacent basis swaps.
    pub swaps: u64,
    /// Later GSO coefficients updated across adjacent swaps.
    pub swap_update_terms: u64,
    /// Deep insertions.
    pub deep_insertions: u64,
    /// Exact suffix-sum terms formed by deep-insertion predicates.
    pub deep_predicate_terms: u64,
    /// Deep-predicate suffix sums rescaled to a larger common denominator.
    pub deep_scale_rescalings: u64,
    /// Exact divisions used to form deep-predicate weights and rescalings.
    pub deep_exact_divisions: u64,
    /// Largest deep-predicate denominator width in bits.
    pub deep_max_denominator_bits: u64,
    /// Largest deep-predicate common-scale width in bits.
    pub deep_max_scale_bits: u64,
    /// Full Gram buffers copied.
    pub gram_copies: u64,
    /// Checked operations in factorization and elementary updates.
    pub checked_updates: u64,
}

#[cfg(feature = "internals")]
impl ReductionObserver for ReductionStats {
    fn gram_copy(&mut self) {
        self.gram_copies += 1;
    }

    fn factorization(&mut self, dimension: usize) {
        self.factorizations += 1;
        for remaining in (0..dimension).rev() {
            let entries = u64::try_from(remaining * remaining).unwrap_or(u64::MAX);
            self.checked_updates = self.checked_updates.saturating_add(4 * entries);
        }
    }

    fn iteration(&mut self) {
        self.iterations += 1;
    }

    fn quotient_check(&mut self, zero_proved: bool) {
        self.size_reduction_checks += 1;
        if zero_proved {
            self.zero_quotients += 1;
        } else {
            self.quotient_divisions += 1;
        }
    }

    fn size_reduction(&mut self, checked_updates: u64) {
        self.size_reductions += 1;
        self.checked_updates += checked_updates;
    }

    fn swaps(&mut self, count: u64, update_terms: u64, checked_updates: u64) {
        self.swaps += count;
        self.swap_update_terms += update_terms;
        self.checked_updates += checked_updates;
    }

    fn deep_insertion(&mut self) {
        self.deep_insertions += 1;
    }

    fn deep_predicate_term(&mut self, scale_rescaled: bool, exact_divisions: u64) {
        self.deep_predicate_terms += 1;
        self.deep_scale_rescalings += u64::from(scale_rescaled);
        self.deep_exact_divisions += exact_divisions;
    }

    fn deep_denominator<T: Int>(&mut self, denominator: T, scale: T) {
        let denominator_bits = u64::from(i128::BITS - denominator.widen().leading_zeros());
        let scale_bits = u64::from(i128::BITS - scale.widen().leading_zeros());
        self.deep_max_denominator_bits = self.deep_max_denominator_bits.max(denominator_bits);
        self.deep_max_scale_bits = self.deep_max_scale_bits.max(scale_bits);
    }
}

/// Symmetric Gram matrix plus its accumulated transform.
///
/// Row scratch makes a checked congruence operation transactional without
/// cloning either matrix. Symmetry is preserved by construction and validated
/// once when the public result is materialized.
struct State<T: Int> {
    gram: IntMatrix<T>,
    transform: IntMatrix<T>,
    gram_row: Vec<T>,
    transform_row: Vec<T>,
}

impl<T: Int> State<T> {
    fn new(gram: &Gram<T>) -> Result<Self, ReduceError> {
        let n = gram.dim();
        Ok(Self {
            gram: gram.as_matrix().clone(),
            transform: IntMatrix::identity(n)?,
            gram_row: vec![T::ZERO; n],
            transform_row: vec![T::ZERO; n],
        })
    }

    fn gso(&self) -> Result<Gso<T>, ReduceError> {
        Gso::from_symmetric_matrix(&self.gram)
    }

    fn swap(&mut self, i: usize, j: usize) {
        self.gram.swap_rows(i, j);
        self.gram.swap_cols(i, j);
        self.transform.swap_rows(i, j);
    }

    /// `b_target -= factor * b_source`, applied to both the Gram matrix and the
    /// transform.
    fn subtract(&mut self, target: usize, source: usize, factor: T) -> Result<u64, RangeError> {
        if factor.is_zero() {
            return Ok(0);
        }
        let mut checked_updates = 0u64;

        // Preflight every checked expression into scratch. The two immutable
        // rows establish the geometry once, so the inner loop carries no
        // matrix index arithmetic or repeated bounds assertions.
        for ((output, &target_value), &source_value) in self
            .gram_row
            .iter_mut()
            .zip(self.gram.row(target))
            .zip(self.gram.row(source))
        {
            *output = if source_value.is_zero() {
                target_value
            } else {
                checked_updates += 2;
                target_value.try_sub(factor.try_mul(source_value)?)?
            };
        }
        for ((output, &target_value), &source_value) in self
            .transform_row
            .iter_mut()
            .zip(self.transform.row(target))
            .zip(self.transform.row(source))
        {
            *output = if source_value.is_zero() {
                target_value
            } else {
                checked_updates += 2;
                target_value.try_sub(factor.try_mul(source_value)?)?
            };
        }

        // Match the previous row-then-column congruence arithmetic. The
        // diagonal sees the already-updated target/source entry.
        let row_diagonal = self.gram_row[target];
        let row_source = self.gram_row[source];
        self.gram_row[target] = if row_source.is_zero() {
            row_diagonal
        } else {
            checked_updates += 2;
            row_diagonal.try_sub(factor.try_mul(row_source)?)?
        };

        // No checked operation remains, so committing the preflighted rows and
        // mirrored column cannot fail or expose a partial state.
        self.gram.row_mut(target).copy_from_slice(&self.gram_row);
        self.gram.copy_column_from_slice(target, &self.gram_row);
        self.transform
            .row_mut(target)
            .copy_from_slice(&self.transform_row);
        Ok(checked_updates)
    }

    /// Moves row `from` down to position `to`, shifting the rest up while
    /// updating the exact factorization after every adjacent swap.
    fn rotate(&mut self, from: usize, to: usize, gso: &mut Gso<T>) -> Result<(), RangeError> {
        for k in (to..from).rev() {
            self.swap(k, k + 1);
            gso.swap_adjacent(k + 1)?;
        }
        Ok(())
    }

    fn finish(self) -> Reduced<T> {
        Reduced {
            gram: Gram::new(self.gram).expect("reduction preserves symmetry"),
            transform: self.transform,
        }
    }
}

fn reduce_with<T: Int>(
    gram: &Gram<T>,
    delta: Delta,
    deep: bool,
) -> Result<Reduced<T>, ReduceError> {
    reduce_observed(gram, delta, deep, &mut Unobserved)
}
fn size_reduce_pair<T: Int, O: ReductionObserver>(
    state: &mut State<T>,
    gso: &mut Gso<T>,
    target: usize,
    source: usize,
    observer: &mut O,
) -> Result<(), ReduceError> {
    let coefficient = gso.lambda(target, source);
    let denominator = gso.minor(source + 1);
    if let Ok(magnitude) = coefficient.try_abs() {
        // `2 * magnitude <= denominator` without overflowing the product.
        // In that case the nearest quotient is zero, including exact ties.
        if magnitude <= denominator.try_sub(magnitude)? {
            observer.quotient_check(true);
            return Ok(());
        }
    }
    observer.quotient_check(false);
    let quotient = div_nearest(coefficient, denominator)?;
    let state_updates = state.subtract(target, source, quotient)?;
    gso.size_reduce(target, source, quotient)?;
    observer.size_reduction(state_updates + 2 * u64::try_from(source + 1).unwrap_or(u64::MAX));
    Ok(())
}

fn reduce_observed<T: Int, O: ReductionObserver>(
    gram: &Gram<T>,
    delta: Delta,
    deep: bool,
    observer: &mut O,
) -> Result<Reduced<T>, ReduceError> {
    let n = gram.dim();
    let mut state = State::new(gram)?;
    observer.gram_copy();
    let mut gso = state.gso()?;
    observer.factorization(n);
    if n < 2 {
        return Ok(state.finish());
    }
    let mut steps = 0u64;
    let mut k = 1usize;
    while k < n {
        guard(&mut steps, BUDGET)?;
        observer.iteration();

        if deep {
            // Deep insertion needs every projected coefficient of b_k.
            for j in (0..k).rev() {
                size_reduce_pair(&mut state, &mut gso, k, j, observer)?;
            }
            if let Some(target) = deep_insertion_point(&gso, k, delta, observer)? {
                state.rotate(k, target, &mut gso)?;
                observer.deep_insertion();
                let mut checked_updates = 0u64;
                let mut update_terms = 0u64;
                for swapped in (target + 1)..=k {
                    let later = u64::try_from(n - swapped - 1).unwrap_or(u64::MAX);
                    checked_updates += 4 + 8 * later;
                    update_terms += 2 * later;
                }
                observer.swaps(
                    u64::try_from(k - target).unwrap_or(u64::MAX),
                    update_terms,
                    checked_updates,
                );
                k = target.max(1);
                continue;
            }
            k += 1;
            continue;
        }

        // The Lovász test only depends on μ[k][k-1]. Delay reductions against
        // earlier vectors until the test passes, avoiding work that a swap
        // would immediately invalidate.
        size_reduce_pair(&mut state, &mut gso, k, k - 1, observer)?;
        if lovasz_holds(&gso, k, delta)? {
            for j in (0..k - 1).rev() {
                size_reduce_pair(&mut state, &mut gso, k, j, observer)?;
            }
            k += 1;
        } else {
            state.swap(k - 1, k);
            gso.swap_adjacent(k)?;
            let later = u64::try_from(n - k - 1).unwrap_or(u64::MAX);
            observer.swaps(1, 2 * later, 4 + 8 * later);
            k = (k - 1).max(1);
        }
    }
    Ok(state.finish())
}

/// The first position `i < k` where inserting `b_k` shortens the projected
/// basis, if any.
///
/// The test is `‖π_i(b_k)‖² < δ ‖b*_i‖²`, where `π_i` projects orthogonally to
/// the span of the **first `i`** basis vectors — not to a middle block, which
/// is a tempting and wrong reading that makes the search cycle. In
/// fraction-free terms
///
/// ```text
/// ‖π_i(b_k)‖²  =  Σ_{j=i..k}  λ̃[k][j]² / (d[j]·d[j+1]),   λ̃[k][k] := d[k+1]
/// ```
///
/// so clearing the denominators with `L = lcm_j(d[j]·d[j+1])` turns the whole
/// test into one integer comparison. At `i = k-1` it reduces to the ordinary
/// Lovász condition, which is why deep insertion subsumes the plain swap.
fn deep_insertion_point<T: Int, O: ReductionObserver>(
    gso: &Gso<T>,
    k: usize,
    delta: Delta,
    observer: &mut O,
) -> Result<Option<usize>, RangeError> {
    let mut scale = gso.minor(k).try_mul(gso.minor(k + 1))?;
    let last = gso.minor(k + 1);
    let mut sum = last.try_mul(last)?;
    observer.deep_predicate_term(false, 0);
    let mut target = None;

    for i in (0..k).rev() {
        let denominator = gso.minor(i).try_mul(gso.minor(i + 1))?;
        // The carried suffix scale usually already contains this denominator.
        // Reuse that exact quotient instead of recomputing the same gcd and
        // lcm. Otherwise, for g = gcd(scale, denominator),
        //
        // L / scale       = denominator / g
        // L / denominator = scale / g
        //
        // and L = (scale / g) * denominator, the same checked multiplication
        // order as `lcm(scale, denominator)`.
        let (next_scale, scale_rescaled, rescale_factor, weight) =
            match scale.try_div_exact(denominator) {
                Ok(weight) => (scale, false, T::ONE, weight),
                Err(RangeError::InexactDivision) => {
                    let common = gcd(scale, denominator)?;
                    let rescale_factor = denominator.try_div_exact(common)?;
                    let weight = scale.try_div_exact(common)?;
                    (weight.try_mul(denominator)?, true, rescale_factor, weight)
                }
                Err(error) => return Err(error),
            };
        if scale_rescaled {
            sum = sum.try_mul(rescale_factor)?;
        }
        let numerator = gso.lambda(k, i);
        sum = sum.try_add(numerator.try_mul(numerator)?.try_mul(weight)?)?;
        scale = next_scale;
        observer.deep_denominator(denominator, scale);
        observer.deep_predicate_term(scale_rescaled, 1 + u64::from(scale_rescaled));
        let left = narrow::<T>(delta.denominator)?
            .try_mul(gso.minor(i))?
            .try_mul(sum)?;
        let right = narrow::<T>(delta.numerator)?
            .try_mul(gso.minor(i + 1))?
            .try_mul(scale)?;
        if left < right {
            target = Some(i);
        }
    }
    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::{
        Delta, State, Unobserved, deep_insertion_point, div_nearest, gauss, is_reduced, lll,
        lll_deep, narrow,
    };
    #[cfg(feature = "internals")]
    use super::{lll_deep_profiled, lll_profiled};
    use crate::basis::{Basis, Gram};
    use crate::error::{LatticeError, RangeError};
    use crate::gso::Gso;
    use crate::int::{Int, IntMatrix, lcm};
    use crate::named::{a_n, d_n, e8};

    #[test]
    fn delta_must_lie_strictly_between_a_quarter_and_one() {
        assert!(Delta::new(1, 2).is_ok());
        assert_eq!(Delta::new(1, 4), Err(LatticeError::Degenerate));
        assert_eq!(Delta::new(1, 1), Err(LatticeError::Degenerate));
        assert_eq!(Delta::new(5, 4), Err(LatticeError::Degenerate));
        assert_eq!(Delta::new(1, 0), Err(LatticeError::Degenerate));
        assert!((Delta::LLL.value() - 0.75).abs() < 1e-15);
    }

    #[test]
    fn reduction_size_reduces_the_root_bases() {
        // The Cartan bases are short but not size-reduced; LLL fixes that
        // without changing the lattice.
        for g in [
            a_n::<i64>(6).unwrap(),
            d_n::<i64>(6).unwrap(),
            e8::<i64>().unwrap(),
        ] {
            let reduced = lll(&g, Delta::LLL).unwrap();
            assert!(is_reduced(&reduced.gram, Delta::LLL).unwrap());
            assert_eq!(reduced.gram.det().unwrap(), g.det().unwrap());
            assert_eq!(reduced.transform.det().unwrap().abs(), 1);
        }
    }

    #[test]
    fn a_skewed_two_dimensional_basis_reduces_to_the_short_one() {
        // Determinant 1, so the lattice is Z^2 in disguise and the reduced
        // Gram matrix must be the identity.
        let g = Gram::<i64>::from_rows(2, &[10, 3, 3, 1]).unwrap();
        let reduced = gauss(&g).unwrap();
        assert_eq!(reduced.gram, Gram::from_rows(2, &[1, 0, 0, 1]).unwrap());
        assert_eq!(reduced.transform.det().unwrap().abs(), 1);
    }

    #[test]
    fn gauss_and_lll_agree_in_the_plane() {
        for entries in [
            [10i64, 3, 3, 1],
            [4, 2, 2, 3],
            [17, 5, 5, 2],
            [2, -1, -1, 2],
        ] {
            let g = Gram::<i64>::from_rows(2, &entries).unwrap();
            let a = gauss(&g).unwrap();
            let b = lll(&g, Delta::STRONG).unwrap();
            assert_eq!(a.gram.entry(0, 0), b.gram.entry(0, 0));
            assert_eq!(a.gram.det().unwrap(), b.gram.det().unwrap());
        }
    }

    #[test]
    fn reduction_is_a_fixpoint_on_an_already_reduced_basis() {
        let g = e8::<i64>().unwrap();
        let once = lll(&g, Delta::LLL).unwrap();
        let twice = lll(&once.gram, Delta::LLL).unwrap();
        assert_eq!(once.gram, twice.gram);
        assert_eq!(twice.transform, crate::int::IntMatrix::identity(8).unwrap());
    }

    #[test]
    fn deep_insertion_satisfies_the_same_certificate() {
        for g in [
            a_n::<i64>(5).unwrap(),
            d_n::<i64>(5).unwrap(),
            e8::<i64>().unwrap(),
        ] {
            let reduced = lll_deep(&g, Delta::LLL).unwrap();
            assert!(is_reduced(&reduced.gram, Delta::LLL).unwrap());
            assert_eq!(reduced.gram.det().unwrap(), g.det().unwrap());
            assert_eq!(reduced.transform.det().unwrap().abs(), 1);
        }
    }

    #[test]
    fn one_dimensional_input_is_already_reduced() {
        let g = Gram::<i64>::from_rows(1, &[7]).unwrap();
        let reduced = lll(&g, Delta::LLL).unwrap();
        assert_eq!(reduced.gram, g);
    }

    #[test]
    fn a_non_positive_definite_form_is_rejected() {
        let g = Gram::<i64>::from_rows(2, &[1, 2, 2, 1]).unwrap();
        assert!(lll(&g, Delta::LLL).is_err());
        assert!(gauss(&g).is_err());
    }

    #[test]
    fn gauss_requires_two_dimensions() {
        assert!(gauss(&a_n::<i64>(3).unwrap()).is_err());
    }
    fn state_subtraction_matches_congruence<T: Int>() {
        let entries = [5i8, 1, 1, 1, 4, 1, 1, 1, 3].map(T::from_i8);
        let gram = Gram::from_rows(3, &entries).unwrap();
        let factor = T::from_i8(2);
        let mut state = State::new(&gram).unwrap();
        state.subtract(2, 0, factor).unwrap();
        let reduced = state.finish();

        let mut transform = IntMatrix::identity(3).unwrap();
        transform.row_sub_mul(2, 0, factor).unwrap();
        let expected = transform
            .mul(gram.as_matrix())
            .unwrap()
            .mul(&transform.transpose().unwrap())
            .unwrap();
        assert_eq!(reduced.gram.as_matrix(), &expected);
        assert_eq!(reduced.transform, transform);
    }

    #[test]
    fn state_subtraction_matches_exact_congruence_at_every_width() {
        state_subtraction_matches_congruence::<i32>();
        state_subtraction_matches_congruence::<i64>();
        state_subtraction_matches_congruence::<i128>();
    }

    fn overflowing_state_subtraction_is_atomic<T: Int>(factor: T) {
        let entries = [1i8, 0, 0, 1].map(T::from_i8);
        let gram = Gram::from_rows(2, &entries).unwrap();
        let mut state = State::new(&gram).unwrap();
        let gram_before = state.gram.clone();
        let transform_before = state.transform.clone();

        assert!(state.subtract(1, 0, factor).is_err());
        assert_eq!(state.gram, gram_before);
        assert_eq!(state.transform, transform_before);
    }

    #[test]
    fn overflowing_state_subtraction_is_atomic_at_every_width() {
        overflowing_state_subtraction_is_atomic::<i32>(i32::MAX);
        overflowing_state_subtraction_is_atomic::<i64>(i64::MAX);
        overflowing_state_subtraction_is_atomic::<i128>(i128::MAX);
    }

    fn deep_insertion_point_lcm<T: Int>(
        gso: &Gso<T>,
        k: usize,
        delta: Delta,
    ) -> Result<Option<usize>, RangeError> {
        let mut scale = gso.minor(k).try_mul(gso.minor(k + 1))?;
        let last = gso.minor(k + 1);
        let mut sum = last.try_mul(last)?;
        let mut target = None;

        for i in (0..k).rev() {
            let denominator = gso.minor(i).try_mul(gso.minor(i + 1))?;
            let next_scale = lcm(scale, denominator)?;
            if next_scale != scale {
                sum = sum.try_mul(next_scale.try_div_exact(scale)?)?;
            }
            let weight = next_scale.try_div_exact(denominator)?;
            let numerator = gso.lambda(k, i);
            sum = sum.try_add(numerator.try_mul(numerator)?.try_mul(weight)?)?;
            scale = next_scale;

            let left = narrow::<T>(delta.denominator)?
                .try_mul(gso.minor(i))?
                .try_mul(sum)?;
            let right = narrow::<T>(delta.numerator)?
                .try_mul(gso.minor(i + 1))?
                .try_mul(scale)?;
            if left < right {
                target = Some(i);
            }
        }
        Ok(target)
    }

    fn suffix_reuse_matches_lcm_oracle<T: Int>() {
        let mut rng = 0x4445_4550_4C4C_4C00u64;
        for n in 2..=6 {
            for _ in 0..32 {
                let mut entries = vec![T::ZERO; n * n];
                for row in 0..n {
                    entries[row * n + row] = T::ONE;
                    for column in 0..row {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        let value = i8::try_from(rng % 5).unwrap() - 2;
                        entries[row * n + column] = T::from_i8(value);
                    }
                }
                let gram = Basis::from_rows(n, n, &entries).unwrap().gram().unwrap();
                let gso = Gso::new(&gram).unwrap();
                for delta in [Delta::LLL, Delta::STRONG] {
                    for k in 1..n {
                        let expected = deep_insertion_point_lcm(&gso, k, delta);
                        let actual = deep_insertion_point(&gso, k, delta, &mut Unobserved);
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }

    #[test]
    fn suffix_reuse_matches_lcm_oracle_at_every_width() {
        suffix_reuse_matches_lcm_oracle::<i32>();
        suffix_reuse_matches_lcm_oracle::<i64>();
        suffix_reuse_matches_lcm_oracle::<i128>();
    }

    #[test]
    fn signed_minimum_quotient_stays_on_the_checked_route() {
        assert!(i32::MIN.try_abs().is_err());
        assert_eq!(div_nearest(i32::MIN, i32::MAX), Ok(-1));
        assert!(i64::MIN.try_abs().is_err());
        assert_eq!(div_nearest(i64::MIN, i64::MAX), Ok(-1));
        assert!(i128::MIN.try_abs().is_err());
        assert_eq!(div_nearest(i128::MIN, i128::MAX), Ok(-1));
    }

    #[cfg(feature = "internals")]
    #[test]
    fn profiled_quotient_counters_partition_checks() {
        let basis = Basis::<i64>::from_rows(3, 3, &[1, 10, 0, 0, 1, 10, 0, 0, 1]).unwrap();
        let gram = basis.gram().unwrap();
        let (_, stats) = lll_profiled(&gram, Delta::STRONG).unwrap();

        assert_eq!(
            stats.size_reduction_checks,
            stats.zero_quotients + stats.quotient_divisions
        );
        assert!(stats.zero_quotients > 0);
        assert!(stats.quotient_divisions >= stats.size_reductions);
        assert!(stats.swap_update_terms > 0);
    }

    #[cfg(feature = "internals")]
    #[test]
    fn profiled_deep_counters_account_for_suffix_arithmetic() {
        let basis =
            Basis::<i64>::from_rows(4, 4, &[2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
                .unwrap();
        let gram = basis.gram().unwrap();
        let (_, stats) = lll_deep_profiled(&gram, Delta::STRONG).unwrap();

        assert!(stats.deep_insertions > 0);
        assert_eq!(
            stats.size_reduction_checks,
            stats.zero_quotients + stats.quotient_divisions
        );
        assert!(stats.zero_quotients > 0);
        assert!(stats.swaps >= stats.deep_insertions);
        assert_eq!(
            stats.deep_exact_divisions + stats.iterations,
            stats.deep_predicate_terms + stats.deep_scale_rescalings
        );
        assert!(stats.deep_max_denominator_bits > 0);
        assert!(stats.deep_max_scale_bits >= stats.deep_max_denominator_bits);
    }
}

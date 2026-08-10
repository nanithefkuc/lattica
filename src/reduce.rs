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
//! # This is precompute
//!
//! Reduction is per-code setup, not per-symbol work, so it is written for
//! obviousness over speed. The orthogonalization is recomputed after each
//! basis change rather than updated in place; the update rules are a classical
//! source of subtle bugs and buy nothing at the dimensions this crate serves.

// The `Delta` bounds check widens two `u32`s; nothing else here leaves the
// integers.
#![allow(clippy::as_conversions)]

use crate::basis::{Basis, Gram};
use crate::error::{LatticeError, RangeError, ReduceError};
use crate::gso::Gso;
use crate::int::{Int, IntMatrix, div_nearest, lcm};

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
        if state.gram.entry(0, 0) > state.gram.entry(1, 1) {
            state.swap(0, 1);
        }
        let norm = state.gram.entry(0, 0);
        if norm.is_zero() {
            return Err(ReduceError::NotFullRank {
                rank: 0,
                required: 2,
            });
        }
        let quotient = div_nearest(state.gram.entry(0, 1), norm)?;
        if quotient.is_zero() {
            break;
        }
        state.subtract(1, 0, quotient)?;
    }
    // Leave the shorter vector first.
    if state.gram.entry(0, 0) > state.gram.entry(1, 1) {
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

/// Gram matrix plus accumulated transform, with the row operations that keep
/// them in step.
struct State<T: Int> {
    gram: Gram<T>,
    transform: IntMatrix<T>,
}

impl<T: Int> State<T> {
    fn new(gram: &Gram<T>) -> Result<Self, ReduceError> {
        let n = gram.dim();
        Ok(Self {
            gram: gram.clone(),
            transform: IntMatrix::identity(n)?,
        })
    }

    fn swap(&mut self, i: usize, j: usize) {
        let mut m = self.gram.as_matrix().clone();
        m.swap_rows(i, j);
        m.swap_cols(i, j);
        self.gram = Gram::new(m).expect("a symmetric permutation stays symmetric");
        self.transform.swap_rows(i, j);
    }

    /// `b_target -= factor * b_source`, applied to both the Gram matrix and the
    /// transform.
    fn subtract(&mut self, target: usize, source: usize, factor: T) -> Result<(), RangeError> {
        if factor.is_zero() {
            return Ok(());
        }
        let mut m = self.gram.as_matrix().clone();
        m.row_sub_mul(target, source, factor)?;
        m.col_sub_mul(target, source, factor)?;
        self.gram = Gram::new(m).expect("a congruence keeps symmetry");
        self.transform.row_sub_mul(target, source, factor)
    }

    /// Moves row `from` down to position `to`, shifting the rest up.
    fn rotate(&mut self, from: usize, to: usize) {
        for k in (to..from).rev() {
            self.swap(k, k + 1);
        }
    }

    fn finish(self) -> Reduced<T> {
        Reduced {
            gram: self.gram,
            transform: self.transform,
        }
    }
}

fn reduce_with<T: Int>(
    gram: &Gram<T>,
    delta: Delta,
    deep: bool,
) -> Result<Reduced<T>, ReduceError> {
    let n = gram.dim();
    let mut state = State::new(gram)?;
    if n < 2 {
        Gso::new(&state.gram)?;
        return Ok(state.finish());
    }

    Gso::new(&state.gram)?;
    let mut steps = 0u64;
    let mut k = 1usize;
    while k < n {
        guard(&mut steps, BUDGET)?;
        let mut gso = Gso::new(&state.gram)?;

        // Size-reduce b_k against every earlier vector, largest index first.
        // The lambda update after each step keeps the remaining quotients
        // correct without a full recomputation.
        for j in (0..k).rev() {
            let quotient = div_nearest(gso.lambda(k, j), gso.minor(j + 1))?;
            if quotient.is_zero() {
                continue;
            }
            state.subtract(k, j, quotient)?;
            gso = Gso::new(&state.gram)?;
        }

        if deep {
            if let Some(target) = deep_insertion_point(&gso, k, delta)? {
                state.rotate(k, target);
                k = target.max(1);
                continue;
            }
            k += 1;
        } else if lovasz_holds(&gso, k, delta)? {
            k += 1;
        } else {
            state.swap(k - 1, k);
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
fn deep_insertion_point<T: Int>(
    gso: &Gso<T>,
    k: usize,
    delta: Delta,
) -> Result<Option<usize>, RangeError> {
    for i in 0..k {
        let mut scale = T::ONE;
        for j in i..=k {
            scale = lcm(scale, gso.minor(j).try_mul(gso.minor(j + 1))?)?;
        }
        let mut sum = T::ZERO;
        for j in i..=k {
            let numerator = if j == k {
                gso.minor(k + 1)
            } else {
                gso.lambda(k, j)
            };
            let denominator = gso.minor(j).try_mul(gso.minor(j + 1))?;
            let weight = scale.try_div_exact(denominator)?;
            sum = sum.try_add(numerator.try_mul(numerator)?.try_mul(weight)?)?;
        }
        let left = narrow::<T>(delta.denominator)?
            .try_mul(gso.minor(i))?
            .try_mul(sum)?;
        let right = narrow::<T>(delta.numerator)?
            .try_mul(gso.minor(i + 1))?
            .try_mul(scale)?;
        if left < right {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::{Delta, gauss, is_reduced, lll, lll_deep};
    use crate::basis::Gram;
    use crate::error::LatticeError;
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
}

//! Exact fixed-width integer arithmetic and integer linear algebra.
//!
//! # Why there are no operators
//!
//! [`Int`] deliberately does not require `Add`, `Sub`, or `Mul`. In release
//! builds those wrap silently, and a wrapped intermediate in a determinant or a
//! Hermite normal form yields a *plausible* wrong answer: the shape is right,
//! the unimodularity check may even pass, and the lattice it describes is not
//! the one the caller has. Every arithmetic method here returns
//! `Result<_, RangeError>` so that outcome is unrepresentable rather than
//! merely discouraged.
//!
//! The cost is verbose implementation code. That is the correct trade for a
//! layer whose entire value is being exactly right.
//!
//! # Growth
//!
//! Fraction-free elimination and Hermite reduction grow intermediates roughly
//! with the subdeterminants of the input, so entry magnitude — not dimension
//! alone — decides whether a problem fits. Nothing here silently promotes to a
//! wider type or falls back to floating point; an overflow is a
//! [`RangeError::Overflow`] naming the width it exceeded.

mod det;
mod hnf;
mod matrix;
mod snf;

pub use det::{adjugate, det};
pub use hnf::{Hnf, hnf, hnf_mod_det};
pub use matrix::{IntMatrix, MAX_DIM};
pub use snf::invariant_factors;

use crate::error::{Op, RangeError};

mod sealed {
    /// Prevents third-party implementations of [`super::Int`].
    pub trait Sealed {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for i128 {}
}

/// A signed fixed-width integer usable as an exact lattice coordinate.
///
/// Implemented for `i32`, `i64`, and `i128`, and sealed: this is a closed set
/// describing the widths this crate supports, not an extensible numeric tower.
///
/// Every fallible operation returns [`RangeError`] rather than wrapping,
/// saturating, or panicking. See the [module documentation](self) for why.
pub trait Int:
    sealed::Sealed
    + Copy
    + Eq
    + Ord
    + core::fmt::Debug
    + core::fmt::Display
    + core::hash::Hash
    + Send
    + Sync
    + 'static
{
    /// The additive identity.
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
    /// Width of this type in bits.
    const WIDTH_BITS: u32;

    /// Converts from a small signed constant.
    fn from_i8(value: i8) -> Self;

    /// Widens to `i128` without loss.
    fn widen(self) -> i128;

    /// Narrows from `i128`, or reports the overflow.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] if `value` does not fit this type.
    fn narrow(value: i128) -> Result<Self, RangeError>;

    /// Checked addition.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] on overflow.
    fn try_add(self, rhs: Self) -> Result<Self, RangeError>;

    /// Checked subtraction.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] on overflow.
    fn try_sub(self, rhs: Self) -> Result<Self, RangeError>;

    /// Checked multiplication.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] on overflow.
    fn try_mul(self, rhs: Self) -> Result<Self, RangeError>;

    /// Checked negation.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] for the type minimum, which has no positive
    /// counterpart.
    fn try_neg(self) -> Result<Self, RangeError>;

    /// Checked absolute value.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] for the type minimum.
    fn try_abs(self) -> Result<Self, RangeError>;

    /// Truncating division, rounding the quotient toward zero.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] on division by zero or on `MIN / -1`.
    fn try_div_trunc(self, rhs: Self) -> Result<Self, RangeError>;

    /// Remainder of [`try_div_trunc`](Int::try_div_trunc), taking the sign of
    /// the dividend.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] on division by zero or on `MIN % -1`.
    fn try_rem_trunc(self, rhs: Self) -> Result<Self, RangeError>;

    /// Division rounding the quotient toward negative infinity, for either
    /// sign of divisor.
    ///
    /// With a positive divisor this is the map that puts the corresponding
    /// remainder in `[0, rhs)`, which is what Hermite reduction above a pivot
    /// requires.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] on division by zero or on `MIN / -1`.
    fn try_div_floor(self, rhs: Self) -> Result<Self, RangeError>;

    /// Division that is required to leave no remainder.
    ///
    /// # Errors
    ///
    /// [`RangeError::Overflow`] on division by zero, and
    /// [`RangeError::InexactDivision`] if the division is not exact.
    fn try_div_exact(self, rhs: Self) -> Result<Self, RangeError>;

    /// Returns `true` if this is zero.
    fn is_zero(self) -> bool;

    /// Returns `true` if this is strictly negative.
    fn is_negative(self) -> bool;
}

macro_rules! impl_int {
    ($($t:ty),* $(,)?) => {$(
        impl Int for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const WIDTH_BITS: u32 = <$t>::BITS;

            #[inline]
            fn from_i8(value: i8) -> Self {
                Self::from(value)
            }

            #[inline]
            fn widen(self) -> i128 {
                i128::from(self)
            }

            #[inline]
            fn narrow(value: i128) -> Result<Self, RangeError> {
                Self::try_from(value).map_err(|_| RangeError::Overflow {
                    op: Op::Mul,
                    width_bits: <$t>::BITS,
                })
            }

            #[inline]
            fn try_add(self, rhs: Self) -> Result<Self, RangeError> {
                self.checked_add(rhs).ok_or(RangeError::Overflow {
                    op: Op::Add,
                    width_bits: <$t>::BITS,
                })
            }

            #[inline]
            fn try_sub(self, rhs: Self) -> Result<Self, RangeError> {
                self.checked_sub(rhs).ok_or(RangeError::Overflow {
                    op: Op::Sub,
                    width_bits: <$t>::BITS,
                })
            }

            #[inline]
            fn try_mul(self, rhs: Self) -> Result<Self, RangeError> {
                self.checked_mul(rhs).ok_or(RangeError::Overflow {
                    op: Op::Mul,
                    width_bits: <$t>::BITS,
                })
            }

            #[inline]
            fn try_neg(self) -> Result<Self, RangeError> {
                self.checked_neg().ok_or(RangeError::Overflow {
                    op: Op::Neg,
                    width_bits: <$t>::BITS,
                })
            }

            #[inline]
            fn try_abs(self) -> Result<Self, RangeError> {
                self.checked_abs().ok_or(RangeError::Overflow {
                    op: Op::Neg,
                    width_bits: <$t>::BITS,
                })
            }

            #[inline]
            fn try_div_trunc(self, rhs: Self) -> Result<Self, RangeError> {
                self.checked_div(rhs).ok_or(RangeError::Overflow {
                    op: Op::Div,
                    width_bits: <$t>::BITS,
                })
            }

            #[inline]
            fn try_rem_trunc(self, rhs: Self) -> Result<Self, RangeError> {
                self.checked_rem(rhs).ok_or(RangeError::Overflow {
                    op: Op::Div,
                    width_bits: <$t>::BITS,
                })
            }

            #[inline]
            fn try_div_floor(self, rhs: Self) -> Result<Self, RangeError> {
                let q = self.try_div_trunc(rhs)?;
                let r = self.try_rem_trunc(rhs)?;
                // Truncation rounds toward zero; when the remainder is nonzero
                // and disagrees in sign with the divisor, that is one step
                // above the floor.
                if r != 0 && (r < 0) != (rhs < 0) {
                    q.try_sub(1)
                } else {
                    Ok(q)
                }
            }

            #[inline]
            fn try_div_exact(self, rhs: Self) -> Result<Self, RangeError> {
                if self.try_rem_trunc(rhs)? != 0 {
                    return Err(RangeError::InexactDivision);
                }
                self.try_div_trunc(rhs)
            }

            #[inline]
            fn is_zero(self) -> bool {
                self == 0
            }

            #[inline]
            fn is_negative(self) -> bool {
                self < 0
            }
        }
    )*};
}

impl_int!(i32, i64, i128);

/// Greatest common divisor, always non-negative.
///
/// `gcd(0, 0)` is `0`.
///
/// # Errors
///
/// [`RangeError::Overflow`] if an intermediate absolute value does not fit,
/// which can only happen for the type minimum.
///
/// # Examples
///
/// ```
/// use lattica::int::gcd;
///
/// assert_eq!(gcd(-12i64, 18)?, 6);
/// assert_eq!(gcd(0i64, 0)?, 0);
/// # Ok::<(), lattica::RangeError>(())
/// ```
pub fn gcd<T: Int>(a: T, b: T) -> Result<T, RangeError> {
    let mut a = a.try_abs()?;
    let mut b = b.try_abs()?;
    while !b.is_zero() {
        let r = a.try_rem_trunc(b)?;
        a = b;
        b = r;
    }
    Ok(a)
}

/// Least common multiple, always non-negative.
///
/// `lcm(x, 0)` is `0`.
///
/// # Errors
///
/// [`RangeError::Overflow`] if the result does not fit.
pub fn lcm<T: Int>(a: T, b: T) -> Result<T, RangeError> {
    if a.is_zero() || b.is_zero() {
        return Ok(T::ZERO);
    }
    let g = gcd(a, b)?;
    a.try_div_exact(g)?.try_mul(b)?.try_abs()
}

/// Division rounding the quotient to the nearest integer, ties away from zero.
///
/// # Errors
///
/// [`RangeError::Overflow`] on division by zero or on an overflowing quotient.
///
/// The remainder `a - q * b` then satisfies `2|r| <= |b|`, which is what makes
/// the Euclidean elimination in [`hnf`] and [`invariant_factors`] halve their
/// pivot each round instead of stepping down by one. Growth control, not
/// aesthetics: it is the difference between terminating and overflowing.
pub fn div_nearest<T: Int>(a: T, b: T) -> Result<T, RangeError> {
    let q = a.try_div_trunc(b)?;
    let r = a.try_sub(q.try_mul(b)?)?;
    if r.is_zero() {
        return Ok(q);
    }
    let abs_r = r.try_abs()?;
    let abs_b = b.try_abs()?;
    // `2 * abs_r > abs_b` without forming the product, which could overflow.
    if abs_r > abs_b.try_sub(abs_r)? {
        if r.is_negative() == b.is_negative() {
            q.try_add(T::ONE)
        } else {
            q.try_sub(T::ONE)
        }
    } else {
        Ok(q)
    }
}

/// Output of the extended Euclidean algorithm.
///
/// The identity `g == s * a + t * b` holds exactly, with `g >= 0`. That makes
/// the pair `(s, t)` usable to build a unimodular elimination step, which is
/// how [`hnf`] keeps its transform provably determinant-preserving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Xgcd<T: Int> {
    /// The greatest common divisor, non-negative.
    pub g: T,
    /// Bézout coefficient of the first argument.
    pub s: T,
    /// Bézout coefficient of the second argument.
    pub t: T,
}

/// Extended Euclidean algorithm.
///
/// # Errors
///
/// [`RangeError::Overflow`] if an intermediate does not fit.
///
/// # Examples
///
/// ```
/// use lattica::int::xgcd;
///
/// let r = xgcd(240i64, 46)?;
/// assert_eq!(r.g, 2);
/// assert_eq!(r.s * 240 + r.t * 46, r.g);
/// # Ok::<(), lattica::RangeError>(())
/// ```
// `g`, `s`, `t`, `q`, `r` are the canonical names for the extended Euclidean
// quantities; expanding them would obscure the algorithm, not clarify it.
#[allow(clippy::many_single_char_names)]
pub fn xgcd<T: Int>(a: T, b: T) -> Result<Xgcd<T>, RangeError> {
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (T::ONE, T::ZERO);
    let (mut old_t, mut t) = (T::ZERO, T::ONE);

    while !r.is_zero() {
        let q = old_r.try_div_trunc(r)?;

        let next_r = old_r.try_sub(q.try_mul(r)?)?;
        old_r = r;
        r = next_r;

        let next_s = old_s.try_sub(q.try_mul(s)?)?;
        old_s = s;
        s = next_s;

        let next_t = old_t.try_sub(q.try_mul(t)?)?;
        old_t = t;
        t = next_t;
    }

    if old_r.is_negative() {
        old_r = old_r.try_neg()?;
        old_s = old_s.try_neg()?;
        old_t = old_t.try_neg()?;
    }

    Ok(Xgcd {
        g: old_r,
        s: old_s,
        t: old_t,
    })
}

#[cfg(test)]
mod tests {
    // The oracle here is `f64::floor` on values far below 2^53, where the
    // conversion is exact in both directions.
    #![allow(clippy::as_conversions, clippy::cast_possible_truncation)]
    use super::{Int, gcd, lcm, xgcd};
    use crate::error::{Op, RangeError};

    #[test]
    fn floor_division_matches_the_mathematical_floor() {
        // Truncation and floor disagree exactly when the remainder is nonzero
        // and the operands differ in sign.
        for a in -20i64..=20 {
            for b in -20i64..=20 {
                if b == 0 {
                    continue;
                }
                let q = a.try_div_floor(b).unwrap();
                #[allow(clippy::cast_precision_loss)]
                let expected = ((a as f64) / (b as f64)).floor() as i64;
                assert_eq!(q, expected, "floor({a}/{b})");
                // The defining property: a - q*b lands in [0, |b|) for b > 0.
                if b > 0 {
                    let r = a - q * b;
                    assert!((0..b).contains(&r), "remainder {r} for {a}/{b}");
                }
            }
        }
    }

    #[test]
    fn exact_division_rejects_a_remainder() {
        assert_eq!(12i64.try_div_exact(4), Ok(3));
        assert_eq!(12i64.try_div_exact(5), Err(RangeError::InexactDivision));
    }

    #[test]
    fn type_minimum_has_no_absolute_value() {
        assert_eq!(
            i32::MIN.try_abs(),
            Err(RangeError::Overflow {
                op: Op::Neg,
                width_bits: 32
            })
        );
        assert_eq!(
            i64::MAX.try_add(1),
            Err(RangeError::Overflow {
                op: Op::Add,
                width_bits: 64
            })
        );
    }

    #[test]
    fn division_by_zero_is_an_error_not_a_panic() {
        assert!(1i64.try_div_trunc(0).is_err());
        assert!(1i64.try_rem_trunc(0).is_err());
        assert!(1i64.try_div_floor(0).is_err());
    }

    #[test]
    fn gcd_is_non_negative_and_divides_both() {
        for a in -30i64..=30 {
            for b in -30i64..=30 {
                let g = gcd(a, b).unwrap();
                assert!(g >= 0);
                if a == 0 && b == 0 {
                    assert_eq!(g, 0);
                    continue;
                }
                assert!(g > 0);
                assert_eq!(a % g, 0);
                assert_eq!(b % g, 0);
            }
        }
    }

    #[test]
    fn bezout_identity_holds_exactly() {
        for a in -40i64..=40 {
            for b in -40i64..=40 {
                let r = xgcd(a, b).unwrap();
                assert!(r.g >= 0);
                assert_eq!(r.s * a + r.t * b, r.g, "xgcd({a}, {b})");
                assert_eq!(r.g, gcd(a, b).unwrap());
            }
        }
    }

    #[test]
    fn lcm_agrees_with_the_product_over_the_gcd() {
        for a in 1i64..=40 {
            for b in 1i64..=40 {
                assert_eq!(lcm(a, b).unwrap(), a / gcd(a, b).unwrap() * b);
            }
        }
        assert_eq!(lcm(7i64, 0).unwrap(), 0);
    }
}

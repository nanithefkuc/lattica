//! The ring `Z_q`, and the bridge between residues and lattice coordinates.
//!
//! This is deliberately a **ring**, not a field. Construction A lifts a
//! codeword to `Λ = q·Z^n + lift(C)` and shaping reduces modulo a power of two;
//! neither needs multiplicative inverses, and requiring `q` prime would rule
//! out the power-of-two moduli that shaping actually uses. Multiplicative field
//! structure belongs in `fff`, on the consumer's side of the boundary.
//!
//! # The two representatives
//!
//! A residue class has a canonical representative in `[0, q)` and a *centered*
//! one in `[-q/2, q/2)`. Lattice work needs the centered form — it is the
//! coset representative of least magnitude, so it is the one that makes
//! `q·Z^n + lift(c)` land near the origin — while modular arithmetic is easier
//! in `[0, q)`. Both are provided and the boundary between them is explicit,
//! because a disagreement about which one is meant is a decode failure that
//! reproduces on exactly one side of a link.

// Modular reduction is width-juggling by nature: every operation here widens to
// `u64`/`u128` to hold a product, then narrows a value already proven to be
// below `q`. Forcing `TryFrom` plus an `expect` into that path would add a
// branch to the hottest arithmetic in the crate without catching anything --
// the truncation, wrap, and sign-loss lints stay on, and each remaining cast
// carries a proof in a comment. The rest of the crate keeps the strict rule.
#![allow(clippy::as_conversions)]
use core::num::NonZeroU32;

use crate::error::LatticeError;

/// The ring of integers modulo `q`.
///
/// Construction precomputes a Barrett reciprocal, so reduction of a wide value
/// costs a multiply-high and a subtract rather than a hardware division. When
/// `q` is a power of two the reciprocal is bypassed for a mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Zq {
    q: u32,
    /// `floor(2^64 / q)`, unused on the power-of-two path.
    reciprocal: u64,
    /// `q - 1` when `q` is a power of two, otherwise zero.
    mask: u32,
    is_power_of_two: bool,
    /// The centering threshold, `(q + 1) / 2`.
    threshold: u32,
}

impl Zq {
    /// Creates the ring `Z_q`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::BadModulus`] if `q` is 1, which has no meaningful
    /// centered representative and is never a useful coding modulus.
    pub fn new(q: NonZeroU32) -> Result<Self, LatticeError> {
        let q = q.get();
        if q < 2 {
            return Err(LatticeError::BadModulus);
        }
        Ok(Self {
            q,
            reciprocal: u64::MAX / u64::from(q),
            mask: if q.is_power_of_two() { q - 1 } else { 0 },
            is_power_of_two: q.is_power_of_two(),
            // `ceil(q / 2)` without forming `q + 1`, which overflows at
            // `q == u32::MAX`.
            threshold: (q >> 1) + (q & 1),
        })
    }

    /// The modulus.
    #[must_use]
    pub const fn modulus(&self) -> u32 {
        self.q
    }

    /// Reduces an unsigned value to `[0, q)`.
    // Casts: the mask and the Barrett remainder are both strictly below `q`,
    // which is a `u32`, so narrowing cannot lose information.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn reduce_u64(&self, x: u64) -> u32 {
        if self.is_power_of_two {
            return (x as u32) & self.mask;
        }
        // Barrett: `hi` under-approximates `x / q`, so the remainder is
        // non-negative and small; the loop is the exact correction.
        let hi = ((x as u128 * self.reciprocal as u128) >> 64) as u64;
        let mut r = x - hi * self.q as u64;
        while r >= self.q as u64 {
            r -= self.q as u64;
        }
        r as u32
    }

    /// Reduces a signed value to `[0, q)`.
    // Cast: reinterpreting `i64` as `u64` then negating recovers the exact
    // magnitude for every input including `i64::MIN`.
    #[allow(clippy::cast_sign_loss)]
    #[must_use]
    pub const fn reduce_i64(&self, x: i64) -> u32 {
        if x >= 0 {
            return self.reduce_u64(x as u64);
        }
        let magnitude = (x as u64).wrapping_neg();
        let r = self.reduce_u64(magnitude);
        if r == 0 { 0 } else { self.q - r }
    }

    /// The centered representative of a residue, in `[-q/2, q/2)`.
    ///
    /// # Panics
    ///
    /// Debug builds assert that `r` is already reduced.
    // Casts: the result lies in `[-q/2, q/2)` with `q <= u32::MAX`, so its
    // magnitude is at most `2^31`, which `i32` represents.
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn center(&self, r: u32) -> i32 {
        debug_assert!(r < self.q, "residue is not reduced");
        if r < self.threshold {
            r as i32
        } else {
            (r as i64 - self.q as i64) as i32
        }
    }

    /// Lifts a residue to the integer of least magnitude in its class.
    ///
    /// Identical to [`center`](Zq::center); the two names exist because the
    /// operations mean different things at the call site. `center` normalizes a
    /// residue, while `lift` crosses from `Z_q` into `Z` on the Construction A
    /// path, and a reader should be able to tell which is happening.
    #[must_use]
    pub const fn lift(&self, r: u32) -> i32 {
        self.center(r)
    }

    /// Modular addition of two reduced residues.
    // Cast: the sum of two values below `q` is below `2q`, and one conditional
    // subtraction brings it below `q`.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use]
    pub const fn add(&self, a: u32, b: u32) -> u32 {
        debug_assert!(a < self.q && b < self.q, "operand is not reduced");
        let s = a as u64 + b as u64;
        if s >= self.q as u64 {
            (s - self.q as u64) as u32
        } else {
            s as u32
        }
    }

    /// Modular subtraction of two reduced residues.
    #[must_use]
    pub const fn sub(&self, a: u32, b: u32) -> u32 {
        debug_assert!(a < self.q && b < self.q, "operand is not reduced");
        if a >= b { a - b } else { self.q - (b - a) }
    }

    /// Modular negation of a reduced residue.
    #[must_use]
    pub const fn neg(&self, a: u32) -> u32 {
        debug_assert!(a < self.q, "operand is not reduced");
        if a == 0 { 0 } else { self.q - a }
    }

    /// Modular multiplication of two reduced residues.
    #[must_use]
    pub const fn mul(&self, a: u32, b: u32) -> u32 {
        debug_assert!(a < self.q && b < self.q, "operand is not reduced");
        self.reduce_u64(a as u64 * b as u64)
    }

    /// Reduces a slice of signed values into residues.
    ///
    /// # Errors
    ///
    /// [`LatticeError::BadSupport`] if the buffers differ in length.
    pub fn reduce_slice(&self, src: &[i64], dst: &mut [u32]) -> Result<(), LatticeError> {
        if src.len() != dst.len() {
            return Err(LatticeError::BadSupport);
        }
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = self.reduce_i64(s);
        }
        Ok(())
    }

    /// Lifts a slice of residues to centered integers.
    ///
    /// # Errors
    ///
    /// [`LatticeError::BadSupport`] if the buffers differ in length.
    pub fn lift_slice(&self, src: &[u32], dst: &mut [i32]) -> Result<(), LatticeError> {
        if src.len() != dst.len() {
            return Err(LatticeError::BadSupport);
        }
        for (d, &s) in dst.iter_mut().zip(src) {
            *d = self.lift(s);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Oracle arithmetic: every cast narrows a value already reduced below `q`.
    #![allow(clippy::cast_possible_truncation)]
    use super::Zq;
    use crate::error::LatticeError;
    use core::num::NonZeroU32;

    fn zq(q: u32) -> Zq {
        Zq::new(NonZeroU32::new(q).unwrap()).unwrap()
    }

    #[test]
    fn modulus_one_is_rejected() {
        assert_eq!(
            Zq::new(NonZeroU32::new(1).unwrap()),
            Err(LatticeError::BadModulus)
        );
    }

    #[test]
    fn reduction_matches_the_remainder_operator_exhaustively() {
        for q in 2..=64u32 {
            let r = zq(q);
            for x in 0..(4 * u64::from(q)) {
                assert_eq!(r.reduce_u64(x), (x % u64::from(q)) as u32, "{x} mod {q}");
            }
            for x in -(4 * i64::from(q))..(4 * i64::from(q)) {
                assert_eq!(
                    r.reduce_i64(x),
                    x.rem_euclid(i64::from(q)) as u32,
                    "{x} mod {q}"
                );
            }
        }
    }

    #[test]
    fn barrett_agrees_near_the_type_maximum() {
        // The correction loop is the fragile part of Barrett reduction, and it
        // is only exercised by wide inputs.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        for q in [3u32, 7, 255, 65_521, 1 << 20, u32::MAX - 4, u32::MAX] {
            let r = zq(q);
            for _ in 0..2_000 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                assert_eq!(r.reduce_u64(state), (state % u64::from(q)) as u32);
            }
            assert_eq!(r.reduce_u64(u64::MAX), (u64::MAX % u64::from(q)) as u32);
            assert_eq!(r.reduce_u64(0), 0);
        }
    }

    #[test]
    fn centered_representatives_span_the_half_open_interval() {
        for q in 2..=64u32 {
            let r = zq(q);
            let lo = -i64::from(q) / 2;
            let hi = i64::from(q) - i64::from(q) / 2;
            for x in 0..q {
                let c = i64::from(r.center(x));
                assert!((lo..hi).contains(&c), "center({x}) = {c} for q = {q}");
                // The defining property: it is the same class.
                assert_eq!(r.reduce_i64(c), x);
            }
        }
    }

    #[test]
    fn lift_round_trips_through_reduction() {
        for q in 2..=64u32 {
            let r = zq(q);
            for x in -200i64..=200 {
                let residue = r.reduce_i64(x);
                let lifted = i64::from(r.lift(residue));
                assert_eq!((x - lifted).rem_euclid(i64::from(q)), 0);
                assert_eq!(r.reduce_i64(lifted), residue);
            }
        }
    }

    #[test]
    fn ring_operations_agree_with_integer_arithmetic() {
        for q in 2..=32u32 {
            let r = zq(q);
            for a in 0..q {
                for b in 0..q {
                    let (a64, b64, q64) = (u64::from(a), u64::from(b), u64::from(q));
                    assert_eq!(u64::from(r.add(a, b)), (a64 + b64) % q64);
                    assert_eq!(u64::from(r.mul(a, b)), (a64 * b64) % q64);
                    assert_eq!(
                        i64::from(r.sub(a, b)),
                        (i64::from(a) - i64::from(b)).rem_euclid(i64::from(q))
                    );
                }
                assert_eq!(r.add(a, r.neg(a)), 0);
            }
        }
    }

    #[test]
    fn power_of_two_path_matches_the_general_one() {
        for shift in 1..20u32 {
            let q = 1u32 << shift;
            let r = zq(q);
            assert!(r.modulus().is_power_of_two());
            for x in [0u64, 1, 12_345, u64::from(q) - 1, u64::from(q), u64::MAX] {
                assert_eq!(r.reduce_u64(x), (x % u64::from(q)) as u32);
            }
        }
    }

    #[test]
    fn slice_helpers_reject_mismatched_lengths() {
        let r = zq(7);
        let mut out = [0u32; 2];
        assert_eq!(
            r.reduce_slice(&[1, 2, 3], &mut out),
            Err(LatticeError::BadSupport)
        );
        // The rejected call wrote nothing.
        assert_eq!(out, [0, 0]);
    }
    #[test]
    fn slice_helpers_reduce_and_lift_each_coordinate() {
        let ring = zq(7);
        let mut residues = [0u32; 5];
        ring.reduce_slice(&[-8, -1, 0, 6, 8], &mut residues)
            .unwrap();
        assert_eq!(residues, [6, 6, 0, 6, 1]);

        let mut lifted = [0i32; 5];
        ring.lift_slice(&residues, &mut lifted).unwrap();
        assert_eq!(lifted, [-1, -1, 0, -1, 1]);
    }
}

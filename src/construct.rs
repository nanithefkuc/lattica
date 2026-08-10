//! Construction A and D: lattices built from linear codes.
//!
//! # The seam
//!
//! `lattica` never owns a code. Construction A needs one, so it
//! arrives through [`CodeMembership`]: the caller answers membership and
//! decoding questions about residues, and `lattica` supplies the lattice built
//! around those answers. No field type crosses the boundary, which is why this
//! crate does not depend on `fff`.
//!
//! ## Why the seam carries costs, not residues
//!
//! The plan specified a hard-decision seam — `decode(&self, residues)`. That
//! turns out not to be enough. The nearest point of `Λ = qZ^n + lift(C)` to a
//! real `x` is
//!
//! ```text
//! min over c in C of  sum_i dist(x_i, lift(c_i) + qZ)^2
//! ```
//!
//! which is a *soft* decoding problem with a per-symbol metric. Handing the
//! code only a rounded residue vector throws away exactly the information that
//! decides the answer, and yields a bounded-distance decoder wearing a
//! maximum-likelihood label — a correctness bug whose only symptom is a
//! slightly worse error rate. So [`CodeMembership::decode_costs`] takes the
//! metric, and [`ConstructionA`] is maximum-likelihood whenever the caller's
//! decoder is.
//!
//! # Construction D
//!
//! [`construction_d_basis`] builds the generator matrix from a chain of codes.
//! The *decoder* is not here: multistage decoding is defined by the order the
//! chain is peeled in, which is a property of the code family rather than of
//! the lattice, and belongs with the consumer. Nothing in this crate consumes
//! it yet, and shipping a stub would be worse than shipping nothing.

// Construction A converts between residues, real coordinates, and integer
// lattice points on every symbol. Each cast is on a value bounded by the
// modulus or by an already-validated coordinate.
#![allow(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use core::num::NonZeroU32;

use crate::basis::Basis;
use crate::error::{DecodeError, LatticeError, RangeError};
use crate::int::{Int, IntMatrix, hnf};
use crate::quant::{Quantizer, Scratch};
use crate::zq::Zq;

/// A linear code over `Z_q`, supplied by the caller.
///
/// Implementors own the code, its field or ring, and its decoder. `lattica`
/// only asks questions.
pub trait CodeMembership {
    /// The modulus `q`.
    fn modulus(&self) -> NonZeroU32;

    /// The code length `n`, which is the lattice dimension.
    fn length(&self) -> usize;

    /// The number of codewords, used to compute the lattice covolume.
    fn cardinality(&self) -> u64;

    /// Is this residue vector a codeword?
    fn contains(&self, residues: &[u32]) -> bool;

    /// Writes the codeword minimizing `Σ_i costs[i * q + c_i]`.
    ///
    /// `costs` is row-major with `q` entries per coordinate. An implementation
    /// that minimizes exactly makes [`ConstructionA`] a maximum-likelihood
    /// decoder; one that does not makes it bounded-distance, and should say so.
    ///
    /// # Errors
    ///
    /// Implementation-defined; [`DecodeError::LengthMismatch`] when the buffers
    /// do not match the code's geometry.
    fn decode_costs(&self, costs: &[f64], out: &mut [u32]) -> Result<(), DecodeError>;
}

/// The Construction A lattice `Λ = q·Z^n + lift(C)`.
///
/// Its covolume is `q^n / |C|`, so a `[n, k]` code over `Z_q` gives `q^(n-k)`.
#[derive(Debug, Clone)]
pub struct ConstructionA<C> {
    code: C,
    zq: Zq,
}

impl<C: CodeMembership> ConstructionA<C> {
    /// Wraps a code as a lattice.
    ///
    /// # Errors
    ///
    /// [`LatticeError::BadModulus`] for `q < 2`, and
    /// [`LatticeError::Degenerate`] for a zero-length code.
    pub fn new(code: C) -> Result<Self, LatticeError> {
        if code.length() == 0 {
            return Err(LatticeError::Degenerate);
        }
        let zq = Zq::new(code.modulus())?;
        Ok(Self { code, zq })
    }

    /// The code this lattice is built from.
    pub const fn code(&self) -> &C {
        &self.code
    }

    /// The covolume `q^n / |C|`, the volume of a fundamental region.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Range`] if `q^n` overflows `u128`, or if the cardinality
    /// does not divide it — which means the caller's `cardinality` is wrong.
    pub fn covolume(&self) -> Result<u128, LatticeError> {
        let q = u128::from(self.zq.modulus());
        let mut total: u128 = 1;
        for _ in 0..self.code.length() {
            total = total.checked_mul(q).ok_or(RangeError::Overflow {
                op: crate::error::Op::Mul,
                width_bits: 128,
            })?;
        }
        let size = u128::from(self.code.cardinality());
        if size == 0 || !total.is_multiple_of(size) {
            return Err(LatticeError::Degenerate);
        }
        Ok(total / size)
    }

    /// Is the integer point `x` in the lattice?
    ///
    /// # Errors
    ///
    /// [`LatticeError::BadSupport`] if `x` is not the code's length.
    pub fn contains(&self, x: &[i64]) -> Result<bool, LatticeError> {
        if x.len() != self.code.length() {
            return Err(LatticeError::BadSupport);
        }
        let mut residues = vec![0u32; x.len()];
        for (dst, &v) in residues.iter_mut().zip(x) {
            *dst = self.zq.reduce_i64(v);
        }
        Ok(self.code.contains(&residues))
    }
}

impl<C: CodeMembership> Quantizer for ConstructionA<C> {
    fn dim(&self) -> usize {
        self.code.length()
    }

    fn scale(&self) -> i64 {
        1
    }

    fn nearest(
        &self,
        x: &[f64],
        out: &mut [i64],
        scratch: &mut Scratch,
    ) -> Result<(), DecodeError> {
        let n = self.code.length();
        crate::quant::validate(x, out, n)?;
        let q = self.zq.modulus();
        let symbols = usize::try_from(q).map_err(|_| DecodeError::NotInLattice)?;
        scratch.ensure(n);
        scratch.ensure_costs(n, symbols);

        // For each coordinate and each residue class, the nearest integer in
        // that class and the squared error it costs. The lattice's own `out`
        // doubles as storage for the winning representative, so the cost table
        // is the only extra state.
        let mut costs = core::mem::take(&mut scratch.costs);
        #[allow(clippy::cast_precision_loss)]
        let modulus = f64::from(q);
        for (i, &xi) in x.iter().enumerate() {
            for s in 0..symbols {
                #[allow(clippy::cast_precision_loss)]
                let lifted = f64::from(self.zq.lift(u32::try_from(s).unwrap_or(0)));
                let steps = round_away((xi - lifted) / modulus);
                #[allow(clippy::cast_precision_loss)]
                let candidate = lifted + steps as f64 * modulus;
                let d = xi - candidate;
                costs[i * symbols + s] = d * d;
            }
        }

        let mut chosen = core::mem::take(&mut scratch.symbols);
        let result = self
            .code
            .decode_costs(&costs[..n * symbols], &mut chosen[..n]);
        if result.is_ok() {
            for (i, (&xi, slot)) in x.iter().zip(out.iter_mut()).enumerate() {
                let lifted = f64::from(self.zq.lift(chosen[i]));
                let steps = round_away((xi - lifted) / modulus);
                #[allow(clippy::cast_possible_truncation)]
                let value = lifted as i64 + steps * i64::from(q);
                *slot = value;
            }
        }
        scratch.costs = costs;
        scratch.symbols = chosen;
        result
    }
}

/// Nearest integer, ties away from zero.
#[allow(clippy::cast_possible_truncation)]
fn round_away(v: f64) -> i64 {
    let truncated = v as i64;
    let fraction = v - truncated as f64;
    if fraction >= 0.5 {
        truncated + 1
    } else if fraction <= -0.5 {
        truncated - 1
    } else {
        truncated
    }
}

/// The generator matrix of the Construction A lattice `q·Z^n + lift(C)`.
///
/// `generator` holds the code's generator matrix over `Z_q`, one codeword basis
/// per row, with entries already reduced into `[0, q)`. The lattice basis is
/// the Hermite normal form of that matrix stacked over `q·I`, which is a
/// generating set by definition and full rank because of the `q·I` rows.
///
/// # Errors
///
/// [`LatticeError::BadModulus`] for `q < 2`, [`LatticeError::Degenerate`] if
/// the geometry is empty, and [`LatticeError::Range`] on overflow.
pub fn construction_a_basis<T: Int>(
    q: T,
    generator: &IntMatrix<T>,
) -> Result<Basis<T>, LatticeError> {
    let n = generator.cols();
    let k = generator.rows();
    if n == 0 || q <= T::ONE {
        return Err(LatticeError::Degenerate);
    }
    let mut stacked = IntMatrix::<T>::zeros(k + n, n)?;
    for i in 0..k {
        for j in 0..n {
            stacked.set(i, j, generator.get(i, j));
        }
    }
    for i in 0..n {
        stacked.set(k + i, i, q);
    }
    let reduced = hnf(&stacked)?;
    if reduced.rank != n {
        return Err(LatticeError::Degenerate);
    }
    let mut rows = IntMatrix::<T>::zeros(n, n)?;
    for i in 0..n {
        for j in 0..n {
            rows.set(i, j, reduced.h.get(i, j));
        }
    }
    Ok(Basis::from_rows(n, n, rows.as_slice())?)
}

/// The generator matrix of a Construction D lattice.
///
/// `levels[j]` is the generator matrix of the level-`j` code, over `Z_base`,
/// and the lattice is the integer span of
///
/// ```text
/// { base^j * g : g a row of levels[j] }  union  { base^a * e_i }
/// ```
///
/// with `a = levels.len()`. The caller is responsible for the chain being
/// nested, `C_0 ⊆ C_1 ⊆ … ⊆ C_{a-1}`; this function computes the span of what
/// it is given, which is what a generator-based construction means.
///
/// # Errors
///
/// [`LatticeError::Degenerate`] for an empty level list, a zero length, or a
/// base below 2; [`LatticeError::Range`] on overflow.
pub fn construction_d_basis<T: Int>(
    base: T,
    levels: &[IntMatrix<T>],
) -> Result<Basis<T>, LatticeError> {
    if levels.is_empty() || base <= T::ONE {
        return Err(LatticeError::Degenerate);
    }
    let n = levels[0].cols();
    if n == 0 || levels.iter().any(|m| m.cols() != n) {
        return Err(LatticeError::Degenerate);
    }

    let total_rows = levels.iter().map(IntMatrix::rows).sum::<usize>() + n;
    let mut stacked = IntMatrix::<T>::zeros(total_rows, n)?;

    let mut row = 0usize;
    let mut weight = T::ONE;
    for level in levels {
        for i in 0..level.rows() {
            for j in 0..n {
                stacked.set(row, j, level.get(i, j).try_mul(weight)?);
            }
            row += 1;
        }
        weight = weight.try_mul(base)?;
    }
    // `weight` is now base^a.
    for i in 0..n {
        stacked.set(row + i, i, weight);
    }

    let reduced = hnf(&stacked)?;
    if reduced.rank != n {
        return Err(LatticeError::Degenerate);
    }
    let mut rows = IntMatrix::<T>::zeros(n, n)?;
    for i in 0..n {
        for j in 0..n {
            rows.set(i, j, reduced.h.get(i, j));
        }
    }
    Ok(Basis::from_rows(n, n, rows.as_slice())?)
}

#[cfg(test)]
mod tests {
    use super::{construction_a_basis, construction_d_basis};
    use crate::int::IntMatrix;
    use crate::named::d_n;

    #[test]
    fn the_single_parity_check_code_gives_the_checkerboard_lattice() {
        // C = {x in F_2^4 : sum even} has generator rows (1,1,0,0), (0,1,1,0),
        // (0,0,1,1), and q*Z^n + lift(C) is exactly D_4.
        let generator =
            IntMatrix::<i64>::from_rows(3, 4, &[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]).unwrap();
        let basis = construction_a_basis(2i64, &generator).unwrap();
        let gram = basis.gram().unwrap();
        // Covolume q^(n-k) = 2^1 = 2, so det Gram = 4 -- the same as D_4.
        assert_eq!(gram.det().unwrap(), 4);
        assert_eq!(gram.det().unwrap(), d_n::<i64>(4).unwrap().det().unwrap());
    }

    #[test]
    fn the_whole_space_gives_the_integer_lattice() {
        let generator = IntMatrix::<i64>::identity(3).unwrap();
        let basis = construction_a_basis(5i64, &generator).unwrap();
        assert_eq!(basis.gram().unwrap().det().unwrap(), 1);
    }

    #[test]
    fn the_trivial_code_gives_the_scaled_integer_lattice() {
        let generator = IntMatrix::<i64>::zeros(1, 3).unwrap();
        let basis = construction_a_basis(3i64, &generator).unwrap();
        // 3Z^3 has covolume 27, so det Gram = 27^2 = 729.
        assert_eq!(basis.gram().unwrap().det().unwrap(), 729);
    }

    #[test]
    fn construction_d_with_one_trivial_level_is_the_scaled_lattice() {
        let levels = [IntMatrix::<i64>::zeros(1, 3).unwrap()];
        let basis = construction_d_basis(2i64, &levels).unwrap();
        // 2Z^3: covolume 8, det Gram 64.
        assert_eq!(basis.gram().unwrap().det().unwrap(), 64);
    }

    #[test]
    fn construction_d_at_one_level_agrees_with_construction_a() {
        let generator =
            IntMatrix::<i64>::from_rows(3, 4, &[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1]).unwrap();
        let a = construction_a_basis(2i64, &generator).unwrap();
        let d = construction_d_basis(2i64, core::slice::from_ref(&generator)).unwrap();
        assert_eq!(a, d);
    }

    #[test]
    fn degenerate_parameters_are_rejected() {
        let empty: [IntMatrix<i64>; 0] = [];
        assert!(construction_d_basis(2i64, &empty).is_err());
        assert!(construction_d_basis(1i64, &[IntMatrix::<i64>::identity(2).unwrap()]).is_err());
        assert!(construction_a_basis(1i64, &IntMatrix::<i64>::identity(2).unwrap()).is_err());
    }
}

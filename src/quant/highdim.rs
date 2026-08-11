//! Maximum-likelihood decoders for `BW_16` and the Leech lattice.
//!
//! Both named decoders use the same proved Schnorr–Euchner core as a deliberate
//! correctness choice. Their published generators are separate, but a
//! bounded-distance Barnes–Wall recursion and a hexacode/MOG Leech decoder have
//! different failure regions outside their guaranteed radii. Reusing exhaustive
//! enumeration instead gives the stronger contract: success is the global
//! nearest point; exhaustion is an error, never an approximate answer.

#![allow(clippy::as_conversions, clippy::cast_precision_loss)]

use crate::basis::Gram;
use crate::error::{DecodeError, LatticeError, Op, RangeError, ReduceError};
use crate::int::{Int, adjugate};
use crate::named::{BW16_NUMERATORS, LEECH24_NUMERATORS, bw16, leech24};
use crate::quant::COORD_LIMIT;
use crate::quant::enumerate::{EnumerationScratch, Enumerator};

/// Reusable buffers for the `BW_16` and Leech ambient decoders.
#[derive(Debug, Clone, Default)]
pub struct AmbientScratch {
    coefficients: Vec<f64>,
    coordinates: Vec<i64>,
    numerators: Vec<i64>,
    enumeration: EnumerationScratch,
}

impl AmbientScratch {
    /// Creates empty scratch that grows on first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            coefficients: Vec::new(),
            coordinates: Vec::new(),
            numerators: Vec::new(),
            enumeration: EnumerationScratch::new(),
        }
    }

    /// Reserves buffers for `dimension` coordinates.
    pub fn reserve(&mut self, dimension: usize) {
        if self.coefficients.len() < dimension {
            self.coefficients.resize(dimension, 0.0);
            self.coordinates.resize(dimension, 0);
            self.numerators.resize(dimension, 0);
            self.enumeration.reserve(dimension);
        }
    }
}

#[derive(Debug, Clone)]
struct AmbientCore<const N: usize> {
    enumerator: Enumerator<i64>,
    /// Column-major transform from ambient coordinates to basis coefficients.
    dual: Vec<f64>,
    numerators: &'static [i8],
    denominator_sq: i64,
}

impl<const N: usize> AmbientCore<N> {
    fn new(
        gram: &Gram<i64>,
        numerators: &'static [i8],
        denominator_sq: i64,
    ) -> Result<Self, LatticeError> {
        let determinant = gram.det()?;
        if determinant == 0 {
            return Err(LatticeError::Degenerate);
        }
        let cofactors = adjugate(gram.as_matrix())?;
        let scale = (denominator_sq as f64).sqrt();
        let mut dual = vec![0.0; N * N];
        for j in 0..N {
            for k in 0..N {
                let mut numerator = 0i128;
                for i in 0..N {
                    let product = cofactors
                        .get(i, j)
                        .widen()
                        .checked_mul(i128::from(numerators[i * N + k]))
                        .ok_or_else(|| range_overflow(Op::Mul))?;
                    numerator = numerator
                        .checked_add(product)
                        .ok_or_else(|| range_overflow(Op::Add))?;
                }
                dual[k * N + j] = numerator as f64 / (determinant as f64 * scale);
            }
        }
        let enumerator = Enumerator::new(gram).map_err(reduce_to_lattice)?;
        Ok(Self {
            enumerator,
            dual,
            numerators,
            denominator_sq,
        })
    }

    fn nearest(
        &self,
        ambient: &[f64],
        numerator_out: &mut [i64],
        node_budget: u64,
        scratch: &mut AmbientScratch,
    ) -> Result<u64, DecodeError> {
        validate_ambient::<N>(ambient, numerator_out)?;
        scratch.reserve(N);
        crate::kernel::transform(&self.dual, N, N, ambient, &mut scratch.coefficients[..N])?;
        for (j, &coefficient) in scratch.coefficients[..N].iter().enumerate() {
            if !coefficient.is_finite() || coefficient.abs() > COORD_LIMIT {
                return Err(DecodeError::NonFinite { index: j });
            }
        }

        let nodes = self.enumerator.nearest_ml(
            &scratch.coefficients[..N],
            &mut scratch.coordinates[..N],
            node_budget,
            &mut scratch.enumeration,
        )?;
        coordinates_to_numerators::<N>(
            &scratch.coordinates[..N],
            self.numerators,
            &mut scratch.numerators[..N],
        )?;
        numerator_out.copy_from_slice(&scratch.numerators[..N]);
        Ok(nodes)
    }

    fn nearest_coefficients(
        &self,
        target: &[f64],
        out: &mut [i64],
        node_budget: u64,
        scratch: &mut AmbientScratch,
    ) -> Result<u64, DecodeError> {
        scratch.reserve(N);
        self.enumerator
            .nearest_ml(target, out, node_budget, &mut scratch.enumeration)
    }
}

/// Maximum-likelihood decoder for the 16-dimensional Barnes–Wall lattice.
///
/// Ambient outputs are exact numerators: divide every returned coordinate by
/// 2. The exhaustive search starts at a Babai radius and either proves the
/// global nearest point or reports [`DecodeError::BudgetExhausted`].
#[derive(Debug, Clone)]
pub struct BarnesWall16 {
    core: AmbientCore<16>,
}

impl BarnesWall16 {
    /// Prepares the decoder from the published `BW_16` generator.
    ///
    /// # Errors
    ///
    /// Only if exact construction of the fixed Gram matrix exceeds the selected
    /// width or fails its positive-definiteness check.
    pub fn new() -> Result<Self, LatticeError> {
        Ok(Self {
            core: AmbientCore::new(&bw16::<i64>()?, &BW16_NUMERATORS, 4)?,
        })
    }

    /// Squared denominator of the exact ambient output coordinates.
    #[must_use]
    pub const fn coordinate_denominator_sq(&self) -> i64 {
        self.core.denominator_sq
    }

    /// Writes twice the ambient coordinates of the nearest lattice point.
    ///
    /// # Errors
    ///
    /// [`DecodeError`] for invalid geometry, arithmetic overflow, or budget
    /// exhaustion. The output is unchanged on error.
    pub fn nearest(
        &self,
        ambient: &[f64],
        numerator_out: &mut [i64],
        node_budget: u64,
        scratch: &mut AmbientScratch,
    ) -> Result<u64, DecodeError> {
        self.core
            .nearest(ambient, numerator_out, node_budget, scratch)
    }

    /// Decodes a real basis-coordinate target to integer basis coordinates.
    ///
    /// # Errors
    ///
    /// As [`Enumerator::nearest_ml`].
    pub fn nearest_coefficients(
        &self,
        target: &[f64],
        out: &mut [i64],
        node_budget: u64,
        scratch: &mut AmbientScratch,
    ) -> Result<u64, DecodeError> {
        self.core
            .nearest_coefficients(target, out, node_budget, scratch)
    }
}

/// Maximum-likelihood decoder for the 24-dimensional Leech lattice.
///
/// Ambient outputs are exact numerators in the published algebraic scaling:
/// divide every returned coordinate by `sqrt(8)`. A successful exhaustive
/// search is maximum-likelihood; the decoder never labels a bounded-distance
/// candidate as ML.
#[derive(Debug, Clone)]
pub struct Leech24 {
    core: AmbientCore<24>,
}

impl Leech24 {
    /// Prepares the decoder from the published Leech generator.
    ///
    /// # Errors
    ///
    /// Only if exact construction of the fixed Gram matrix exceeds the selected
    /// width or fails its positive-definiteness check.
    pub fn new() -> Result<Self, LatticeError> {
        Ok(Self {
            core: AmbientCore::new(&leech24::<i64>()?, &LEECH24_NUMERATORS, 8)?,
        })
    }

    /// Squared denominator of the exact ambient output coordinates.
    #[must_use]
    pub const fn coordinate_denominator_sq(&self) -> i64 {
        self.core.denominator_sq
    }

    /// Writes `sqrt(8)` times the ambient coordinates of the nearest point.
    ///
    /// # Errors
    ///
    /// [`DecodeError`] for invalid geometry, arithmetic overflow, or budget
    /// exhaustion. The output is unchanged on error.
    pub fn nearest(
        &self,
        ambient: &[f64],
        numerator_out: &mut [i64],
        node_budget: u64,
        scratch: &mut AmbientScratch,
    ) -> Result<u64, DecodeError> {
        self.core
            .nearest(ambient, numerator_out, node_budget, scratch)
    }

    /// Decodes a real basis-coordinate target to integer basis coordinates.
    ///
    /// # Errors
    ///
    /// As [`Enumerator::nearest_ml`].
    pub fn nearest_coefficients(
        &self,
        target: &[f64],
        out: &mut [i64],
        node_budget: u64,
        scratch: &mut AmbientScratch,
    ) -> Result<u64, DecodeError> {
        self.core
            .nearest_coefficients(target, out, node_budget, scratch)
    }
}

fn validate_ambient<const N: usize>(
    ambient: &[f64],
    numerator_out: &[i64],
) -> Result<(), DecodeError> {
    if ambient.len() != N || numerator_out.len() != N {
        return Err(DecodeError::LengthMismatch {
            expected: N,
            found: if ambient.len() == N {
                numerator_out.len()
            } else {
                ambient.len()
            },
        });
    }
    for (index, &value) in ambient.iter().enumerate() {
        if !value.is_finite() || value.abs() > COORD_LIMIT {
            return Err(DecodeError::NonFinite { index });
        }
    }
    Ok(())
}

fn coordinates_to_numerators<const N: usize>(
    coordinates: &[i64],
    generator: &[i8],
    out: &mut [i64],
) -> Result<(), DecodeError> {
    for k in 0..N {
        let mut total = 0i128;
        for i in 0..N {
            let product = i128::from(coordinates[i])
                .checked_mul(i128::from(generator[i * N + k]))
                .ok_or_else(|| overflow(Op::Mul))?;
            total = total
                .checked_add(product)
                .ok_or_else(|| overflow(Op::Add))?;
        }
        out[k] = i64::try_from(total).map_err(|_| overflow(Op::Add))?;
    }
    Ok(())
}

const fn range_overflow(op: Op) -> RangeError {
    RangeError::Overflow { op, width_bits: 64 }
}

const fn overflow(op: Op) -> DecodeError {
    DecodeError::Range(range_overflow(op))
}

const fn reduce_to_lattice(error: ReduceError) -> LatticeError {
    match error {
        ReduceError::Range(range) => LatticeError::Range(range),
        ReduceError::Singular
        | ReduceError::NotFullRank { .. }
        | ReduceError::BudgetExhausted { .. } => LatticeError::Degenerate,
    }
}

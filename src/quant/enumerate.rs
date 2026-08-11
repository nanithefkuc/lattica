//! Budgeted Schnorr–Euchner enumeration for general integral lattices.
//!
//! Targets and answers are basis-coordinate vectors. The target is real; the
//! answer is integral. Search walks the exact Gram–Schmidt triangular system
//! from the last coordinate to the first, visits children in zig-zag order,
//! and shrinks the radius whenever it finds a better point.

#![allow(clippy::as_conversions, clippy::cast_precision_loss)]

use core::cmp::Ordering;

use crate::basis::Gram;
use crate::error::{DecodeError, ReduceError};
use crate::gso::Gso;
use crate::int::Int;
use crate::quant::COORD_LIMIT;

/// Reusable buffers for nearest-point enumeration.
///
/// Keep one per decoding worker. After it has seen the largest dimension used
/// by that worker, [`Enumerator::nearest`] performs no allocation.
#[derive(Debug, Clone, Default)]
pub struct EnumerationScratch {
    point: Vec<i64>,
    best: Vec<i64>,
    coefficients: Vec<f64>,
}

impl EnumerationScratch {
    /// Creates empty scratch space that grows on first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            point: Vec::new(),
            best: Vec::new(),
            coefficients: Vec::new(),
        }
    }

    /// Reserves space for `dimension` coordinates.
    pub fn reserve(&mut self, dimension: usize) {
        if self.point.len() < dimension {
            self.point.resize(dimension, 0);
            self.best.resize(dimension, 0);
            self.coefficients.resize(dimension, 0.0);
        }
    }
}

/// One point returned by radius-list enumeration.
#[derive(Debug, Clone, PartialEq)]
pub struct ListPoint {
    point: Vec<i64>,
    distance_sq: f64,
}

impl ListPoint {
    /// Integer basis coordinates of the lattice point.
    #[must_use]
    pub fn point(&self) -> &[i64] {
        &self.point
    }

    /// Squared distance from the target, in the lattice metric.
    #[must_use]
    pub const fn distance_sq(&self) -> f64 {
        self.distance_sq
    }
}

/// Prepared Schnorr–Euchner decoder for one Gram matrix.
///
/// Construction performs the fraction-free Gram–Schmidt factorization once.
/// Reuse the result across every target for the same basis.
#[derive(Debug, Clone)]
pub struct Enumerator<T: Int> {
    gso: Gso<T>,
}

impl<T: Int> Enumerator<T> {
    /// Prepares enumeration for `gram`.
    ///
    /// # Errors
    ///
    /// As [`Gso::new`]: the Gram matrix must be positive definite and its exact
    /// factorization must fit the selected integer width.
    pub fn new(gram: &Gram<T>) -> Result<Self, ReduceError> {
        Ok(Self {
            gso: Gso::new(gram)?,
        })
    }

    /// Lattice dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.gso.dim()
    }

    /// Finds the nearest lattice point inside `radius_sq`.
    ///
    /// `target` and `out` are basis-coordinate vectors. Equal-distance points
    /// are resolved by lexicographically smallest integer coordinates. This
    /// total tie rule is independent of traversal order.
    ///
    /// The output is written only after the search proves its answer. Budget
    /// exhaustion and an empty radius leave it unchanged.
    ///
    /// # Errors
    ///
    /// - [`DecodeError::LengthMismatch`] for the wrong dimension;
    /// - [`DecodeError::NonFinite`] for a non-finite target;
    /// - [`DecodeError::InvalidRadius`] for a negative or non-finite radius;
    /// - [`DecodeError::OutsideRadius`] when the ball contains no lattice point;
    /// - [`DecodeError::BudgetExhausted`] when `node_budget` is reached before
    ///   the search is complete.
    pub fn nearest(
        &self,
        target: &[f64],
        out: &mut [i64],
        radius_sq: f64,
        node_budget: u64,
        scratch: &mut EnumerationScratch,
    ) -> Result<u64, DecodeError> {
        validate(self.dim(), target, out.len(), radius_sq)?;
        scratch.reserve(self.dim());

        let mut search = Search {
            gso: &self.gso,
            target,
            point: &mut scratch.point[..self.dim()],
            best: &mut scratch.best[..self.dim()],
            radius_sq,
            node_budget,
            nodes: 0,
            found: false,
        };
        search.nearest_level(self.dim(), 0.0)?;
        if !search.found {
            return Err(DecodeError::OutsideRadius { radius_sq });
        }
        out.copy_from_slice(search.best);
        Ok(search.nodes)
    }

    /// Finds the unrestricted nearest lattice point.
    ///
    /// Babai nearest-plane supplies a finite radius containing a lattice point;
    /// Schnorr–Euchner then proves the maximum-likelihood answer inside that
    /// radius. The Babai point is only a bound and is never returned unless the
    /// exhaustive search proves it.
    ///
    /// # Errors
    ///
    /// As [`nearest`](Self::nearest), except the radius cannot be empty because
    /// it is derived from an actual lattice point.
    pub fn nearest_ml(
        &self,
        target: &[f64],
        out: &mut [i64],
        node_budget: u64,
        scratch: &mut EnumerationScratch,
    ) -> Result<u64, DecodeError> {
        validate(self.dim(), target, out.len(), 0.0)?;
        scratch.reserve(self.dim());
        scratch.coefficients[..self.dim()].copy_from_slice(target);
        crate::quant::babai::nearest_plane(
            &self.gso,
            &mut scratch.coefficients[..self.dim()],
            &mut scratch.best[..self.dim()],
        )?;
        let candidate_sq = triangular_distance(&self.gso, target, &scratch.best[..self.dim()])?;
        // The same positive terms are accumulated along a recursive path in
        // `nearest`; allow one small rounding envelope so the Babai point that
        // established the radius cannot fall a few ulps outside its own ball.
        let envelope = f64::EPSILON * candidate_sq.abs().max(1.0) * (4 * self.dim()) as f64;
        self.nearest(target, out, candidate_sq + envelope, node_budget, scratch)
    }

    /// Returns every lattice point inside `radius_sq`.
    ///
    /// Results are ordered by increasing squared distance, then by
    /// lexicographic integer coordinates. Unlike nearest-point mode, the radius
    /// is fixed and never shrinks.
    ///
    /// # Errors
    ///
    /// As [`nearest`](Self::nearest), except an empty ball returns an empty
    /// list rather than [`DecodeError::OutsideRadius`].
    pub fn list(
        &self,
        target: &[f64],
        radius_sq: f64,
        node_budget: u64,
        scratch: &mut EnumerationScratch,
    ) -> Result<Vec<ListPoint>, DecodeError> {
        validate(self.dim(), target, self.dim(), radius_sq)?;
        scratch.reserve(self.dim());

        let mut list = Vec::new();
        let mut walk = ListSearch {
            gso: &self.gso,
            target,
            point: &mut scratch.point[..self.dim()],
            radius_sq,
            node_budget,
            nodes: 0,
            list: &mut list,
        };
        walk.level(self.dim(), 0.0)?;
        list.sort_by(|a, b| {
            a.distance_sq
                .total_cmp(&b.distance_sq)
                .then_with(|| a.point.cmp(&b.point))
        });
        Ok(list)
    }
}

fn validate(
    dimension: usize,
    target: &[f64],
    output_len: usize,
    radius_sq: f64,
) -> Result<(), DecodeError> {
    if target.len() != dimension || output_len != dimension {
        return Err(DecodeError::LengthMismatch {
            expected: dimension,
            found: if target.len() == dimension {
                output_len
            } else {
                target.len()
            },
        });
    }
    for (index, &value) in target.iter().enumerate() {
        if !value.is_finite() || value.abs() > COORD_LIMIT {
            return Err(DecodeError::NonFinite { index });
        }
    }
    if !radius_sq.is_finite() || radius_sq < 0.0 {
        return Err(DecodeError::InvalidRadius { radius_sq });
    }
    Ok(())
}

struct Search<'a, T: Int> {
    gso: &'a Gso<T>,
    target: &'a [f64],
    point: &'a mut [i64],
    best: &'a mut [i64],
    radius_sq: f64,
    node_budget: u64,
    nodes: u64,
    found: bool,
}

impl<T: Int> Search<'_, T> {
    fn nearest_level(&mut self, depth: usize, partial: f64) -> Result<(), DecodeError> {
        if depth == 0 {
            let replace = !self.found
                || partial < self.radius_sq
                || (partial.total_cmp(&self.radius_sq) == Ordering::Equal
                    && self.point < self.best);
            if replace {
                self.best.copy_from_slice(self.point);
                self.radius_sq = partial;
                self.found = true;
            }
            return Ok(());
        }

        let level = depth - 1;
        let center = center(self.gso, self.target, self.point, level)?;
        let weight = weight(self.gso, level);
        let Some(children) = Children::new(center, self.radius_sq - partial, weight) else {
            return Ok(());
        };
        for value in children {
            if self.nodes >= self.node_budget {
                return Err(DecodeError::BudgetExhausted {
                    nodes: self.nodes,
                    radius_sq: self.radius_sq,
                });
            }
            self.nodes += 1;
            self.point[level] = value;
            let delta = center - value as f64;
            let next = partial + weight * delta * delta;
            if next <= self.radius_sq {
                self.nearest_level(level, next)?;
            }
        }
        Ok(())
    }
}

struct ListSearch<'a, T: Int> {
    gso: &'a Gso<T>,
    target: &'a [f64],
    point: &'a mut [i64],
    radius_sq: f64,
    node_budget: u64,
    nodes: u64,
    list: &'a mut Vec<ListPoint>,
}

impl<T: Int> ListSearch<'_, T> {
    fn level(&mut self, depth: usize, partial: f64) -> Result<(), DecodeError> {
        if depth == 0 {
            self.list.push(ListPoint {
                point: self.point.to_vec(),
                distance_sq: if partial == 0.0 { 0.0 } else { partial },
            });
            return Ok(());
        }

        let level = depth - 1;
        let center = center(self.gso, self.target, self.point, level)?;
        let weight = weight(self.gso, level);
        let Some(children) = Children::new(center, self.radius_sq - partial, weight) else {
            return Ok(());
        };
        for value in children {
            if self.nodes >= self.node_budget {
                return Err(DecodeError::BudgetExhausted {
                    nodes: self.nodes,
                    radius_sq: self.radius_sq,
                });
            }
            self.nodes += 1;
            self.point[level] = value;
            let delta = center - value as f64;
            let next = partial + weight * delta * delta;
            if next <= self.radius_sq {
                self.level(level, next)?;
            }
        }
        Ok(())
    }
}

fn triangular_distance<T: Int>(
    gso: &Gso<T>,
    target: &[f64],
    point: &[i64],
) -> Result<f64, DecodeError> {
    let mut total = 0.0;
    for level in (0..gso.dim()).rev() {
        let value = center(gso, target, point, level)?;
        let delta = value - point[level] as f64;
        total += weight(gso, level) * delta * delta;
    }
    if total.is_finite() {
        Ok(total)
    } else {
        Err(DecodeError::InvalidRadius { radius_sq: total })
    }
}

fn center<T: Int>(
    gso: &Gso<T>,
    target: &[f64],
    point: &[i64],
    level: usize,
) -> Result<f64, DecodeError> {
    let mut value = target[level];
    for j in level + 1..gso.dim() {
        value += (target[j] - point[j] as f64) * gso.mu(j, level);
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(DecodeError::NonFinite { index: level })
    }
}

fn weight<T: Int>(gso: &Gso<T>, level: usize) -> f64 {
    let (numerator, denominator) = gso.norm_sq(level);
    numerator.widen() as f64 / denominator.widen() as f64
}

struct Children {
    center: f64,
    low: i64,
    high: i64,
    nearest: i64,
    step: i64,
    emitted_nearest: bool,
}

impl Children {
    #[allow(clippy::cast_possible_truncation)]
    fn new(center: f64, remaining: f64, weight: f64) -> Option<Self> {
        if remaining < 0.0 || !remaining.is_finite() || weight <= 0.0 || !weight.is_finite() {
            return None;
        }
        let bound = (remaining / weight).sqrt();
        let low_f = (center - bound).ceil();
        let high_f = (center + bound).floor();
        if low_f > high_f || low_f < i64::MIN as f64 || high_f > i64::MAX as f64 {
            return None;
        }
        let low = low_f as i64;
        let high = high_f as i64;
        let nearest = round_away(center).clamp(low, high);
        Some(Self {
            center,
            low,
            high,
            nearest,
            step: 1,
            emitted_nearest: false,
        })
    }
}

impl Iterator for Children {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.emitted_nearest {
            self.emitted_nearest = true;
            return Some(self.nearest);
        }
        loop {
            let magnitude = (self.step + 1) / 2;
            let positive_first = self.center >= self.nearest as f64;
            let positive = if self.step % 2 == 1 {
                positive_first
            } else {
                !positive_first
            };
            self.step = self.step.checked_add(1)?;
            let candidate = if positive {
                self.nearest.checked_add(magnitude)?
            } else {
                self.nearest.checked_sub(magnitude)?
            };
            if candidate >= self.low && candidate <= self.high {
                return Some(candidate);
            }
            if self.nearest.saturating_sub(magnitude) < self.low
                && self.nearest.saturating_add(magnitude) > self.high
            {
                return None;
            }
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn round_away(value: f64) -> i64 {
    let truncated = value as i64;
    let fraction = value - truncated as f64;
    if fraction >= 0.5 {
        truncated + 1
    } else if fraction <= -0.5 {
        truncated - 1
    } else {
        truncated
    }
}

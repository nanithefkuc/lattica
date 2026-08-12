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

const ITERATIVE_DIMENSION: usize = 24;

/// Reusable buffers for nearest-point enumeration.
///
/// Keep one per decoding worker. After it has seen the largest dimension used
/// by that worker, [`Enumerator::nearest`] performs no allocation.
#[derive(Debug, Clone, Default)]
pub struct EnumerationScratch {
    point: Vec<i64>,
    best: Vec<i64>,
    coefficients: Vec<f64>,
    centers: Vec<f64>,
    partials: Vec<f64>,
    children: Vec<Children>,
}

impl EnumerationScratch {
    /// Creates empty scratch space that grows on first use.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            point: Vec::new(),
            best: Vec::new(),
            coefficients: Vec::new(),
            centers: Vec::new(),
            partials: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Reserves space for `dimension` coordinates.
    pub fn reserve(&mut self, dimension: usize) {
        self.reserve_core(dimension);
        self.reserve_state(dimension);
    }

    fn reserve_core(&mut self, dimension: usize) {
        if self.point.len() < dimension {
            self.point.resize(dimension, 0);
            self.best.resize(dimension, 0);
            self.coefficients.resize(dimension, 0.0);
        }
    }

    fn reserve_state(&mut self, dimension: usize) {
        if self.centers.len() < dimension {
            self.centers.resize(dimension, 0.0);
            self.partials.resize(dimension, 0.0);
            self.children.resize(dimension, Children::empty());
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
    mu: Vec<f64>,
    weights: Vec<f64>,
}

impl<T: Int> Enumerator<T> {
    /// Prepares enumeration for `gram`.
    ///
    /// # Errors
    ///
    /// As [`Gso::new`]: the Gram matrix must be positive definite and its exact
    /// factorization must fit the selected integer width.
    pub fn new(gram: &Gram<T>) -> Result<Self, ReduceError> {
        let gso = Gso::new(gram)?;
        let n = gso.dim();
        let mut mu = vec![0.0; n * n];
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let (numerator, denominator) = gso.norm_sq(i);
            weights.push(numerator.widen() as f64 / denominator.widen() as f64);
            for j in 0..i {
                mu[i * n + j] = gso.mu(i, j);
            }
        }
        Ok(Self { gso, mu, weights })
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
        scratch.reserve_core(self.dim());
        if self.dim() < ITERATIVE_DIMENSION {
            let mut search = RecursiveSearch {
                mu: &self.mu,
                weights: &self.weights,
                target,
                point: &mut scratch.point[..self.dim()],
                best: &mut scratch.best[..self.dim()],
                radius_sq,
                node_budget,
                nodes: 0,
                found: false,
            };
            search.run(self.dim(), 0.0)?;
            if !search.found {
                return Err(DecodeError::OutsideRadius { radius_sq });
            }
            out.copy_from_slice(search.best);
            return Ok(search.nodes);
        }
        scratch.reserve_state(self.dim());

        let mut search = Search {
            mu: &self.mu,
            weights: &self.weights,
            target,
            point: &mut scratch.point[..self.dim()],
            best: &mut scratch.best[..self.dim()],
            centers: &mut scratch.centers[..self.dim()],
            partials: &mut scratch.partials[..self.dim()],
            children: &mut scratch.children[..self.dim()],
            radius_sq,
            node_budget,
            nodes: 0,
            found: false,
        };
        search.run()?;
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
        scratch.reserve_core(self.dim());
        scratch.coefficients[..self.dim()].copy_from_slice(target);
        crate::quant::babai::nearest_plane(
            &self.gso,
            &mut scratch.coefficients[..self.dim()],
            &mut scratch.best[..self.dim()],
        )?;
        let candidate_sq = triangular_distance(
            &self.mu,
            &self.weights,
            target,
            &scratch.best[..self.dim()],
        )?;
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
        scratch.reserve_core(self.dim());
        if self.dim() < ITERATIVE_DIMENSION {
            let mut list = Vec::new();
            let mut walk = RecursiveListSearch {
                mu: &self.mu,
                weights: &self.weights,
                target,
                point: &mut scratch.point[..self.dim()],
                radius_sq,
                node_budget,
                nodes: 0,
                list: &mut list,
            };
            walk.run(self.dim(), 0.0)?;
            list.sort_by(|a, b| {
                a.distance_sq
                    .total_cmp(&b.distance_sq)
                    .then_with(|| a.point.cmp(&b.point))
            });
            return Ok(list);
        }
        scratch.reserve_state(self.dim());

        let mut list = Vec::new();
        let mut walk = ListSearch {
            mu: &self.mu,
            weights: &self.weights,
            target,
            point: &mut scratch.point[..self.dim()],
            centers: &mut scratch.centers[..self.dim()],
            partials: &mut scratch.partials[..self.dim()],
            children: &mut scratch.children[..self.dim()],
            radius_sq,
            node_budget,
            nodes: 0,
            list: &mut list,
        };
        walk.run()?;
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

struct RecursiveSearch<'a> {
    mu: &'a [f64],
    weights: &'a [f64],
    target: &'a [f64],
    point: &'a mut [i64],
    best: &'a mut [i64],
    radius_sq: f64,
    node_budget: u64,
    nodes: u64,
    found: bool,
}

impl RecursiveSearch<'_> {
    fn consider(&mut self, distance_sq: f64) {
        let replace = !self.found
            || distance_sq < self.radius_sq
            || (distance_sq.total_cmp(&self.radius_sq) == Ordering::Equal
                && self.point < self.best);
        if replace {
            self.best.copy_from_slice(self.point);
            self.radius_sq = distance_sq;
            self.found = true;
        }
    }

    fn run(&mut self, depth: usize, partial: f64) -> Result<(), DecodeError> {
        if depth == 0 {
            self.consider(partial);
            return Ok(());
        }
        let level = depth - 1;
        let value = center(self.mu, self.target, self.point, level)?;
        let Some(children) =
            Children::new(value, self.radius_sq - partial, self.weights[level])
        else {
            return Ok(());
        };
        for child in children {
            if self.nodes >= self.node_budget {
                return Err(DecodeError::BudgetExhausted {
                    nodes: self.nodes,
                    radius_sq: self.radius_sq,
                });
            }
            self.nodes += 1;
            self.point[level] = child;
            let delta = value - child as f64;
            let next = partial + self.weights[level] * delta * delta;
            if next <= self.radius_sq {
                self.run(level, next)?;
            }
        }
        Ok(())
    }
}

struct RecursiveListSearch<'a> {
    mu: &'a [f64],
    weights: &'a [f64],
    target: &'a [f64],
    point: &'a mut [i64],
    radius_sq: f64,
    node_budget: u64,
    nodes: u64,
    list: &'a mut Vec<ListPoint>,
}

impl RecursiveListSearch<'_> {
    fn run(&mut self, depth: usize, partial: f64) -> Result<(), DecodeError> {
        if depth == 0 {
            self.list.push(ListPoint {
                point: self.point.to_vec(),
                distance_sq: if partial == 0.0 { 0.0 } else { partial },
            });
            return Ok(());
        }
        let level = depth - 1;
        let value = center(self.mu, self.target, self.point, level)?;
        let Some(children) =
            Children::new(value, self.radius_sq - partial, self.weights[level])
        else {
            return Ok(());
        };
        for child in children {
            if self.nodes >= self.node_budget {
                return Err(DecodeError::BudgetExhausted {
                    nodes: self.nodes,
                    radius_sq: self.radius_sq,
                });
            }
            self.nodes += 1;
            self.point[level] = child;
            let delta = value - child as f64;
            let next = partial + self.weights[level] * delta * delta;
            if next <= self.radius_sq {
                self.run(level, next)?;
            }
        }
        Ok(())
    }
}

struct Search<'a> {
    mu: &'a [f64],
    weights: &'a [f64],
    target: &'a [f64],
    point: &'a mut [i64],
    best: &'a mut [i64],
    centers: &'a mut [f64],
    partials: &'a mut [f64],
    children: &'a mut [Children],
    radius_sq: f64,
    node_budget: u64,
    nodes: u64,
    found: bool,
}

impl Search<'_> {
    fn enter(&mut self, level: usize, partial: f64) {
        let value = self.centers[level];
        self.partials[level] = partial;
        self.children[level] =
            Children::new(value, self.radius_sq - partial, self.weights[level])
                .unwrap_or_else(Children::empty);
    }

    fn consider(&mut self, distance_sq: f64) {
        let replace = !self.found
            || distance_sq < self.radius_sq
            || (distance_sq.total_cmp(&self.radius_sq) == Ordering::Equal
                && self.point < self.best);
        if replace {
            self.best.copy_from_slice(self.point);
            self.radius_sq = distance_sq;
            self.found = true;
        }
    }


    fn run(&mut self) -> Result<(), DecodeError> {
        let n = self.target.len();
        if n == 0 {
            self.consider(0.0);
            return Ok(());
        }
        initialize_centers(self.mu, self.target, self.point, self.centers)?;
        let mut level = n - 1;
        self.enter(level, 0.0);
        loop {
            let Some(value) = self.children[level].next() else {
                if level == n - 1 {
                    return Ok(());
                }
                level += 1;
                continue;
            };
            if self.nodes >= self.node_budget {
                return Err(DecodeError::BudgetExhausted {
                    nodes: self.nodes,
                    radius_sq: self.radius_sq,
                });
            }
            self.nodes += 1;
            update_centers(self.mu, self.point, self.centers, level, value);
            let delta = self.centers[level] - value as f64;
            let next = self.partials[level] + self.weights[level] * delta * delta;
            if next > self.radius_sq {
                continue;
            }
            if level == 0 {
                self.consider(next);
            } else {
                level -= 1;
                self.enter(level, next);
            }
        }
    }
}

struct ListSearch<'a> {
    mu: &'a [f64],
    weights: &'a [f64],
    target: &'a [f64],
    point: &'a mut [i64],
    centers: &'a mut [f64],
    partials: &'a mut [f64],
    children: &'a mut [Children],
    radius_sq: f64,
    node_budget: u64,
    nodes: u64,
    list: &'a mut Vec<ListPoint>,
}

impl ListSearch<'_> {
    fn enter(&mut self, level: usize, partial: f64) {
        let value = self.centers[level];
        self.partials[level] = partial;
        self.children[level] =
            Children::new(value, self.radius_sq - partial, self.weights[level])
                .unwrap_or_else(Children::empty);
    }


    fn run(&mut self) -> Result<(), DecodeError> {
        let n = self.target.len();
        if n == 0 {
            self.list.push(ListPoint {
                point: Vec::new(),
                distance_sq: 0.0,
            });
            return Ok(());
        }
        initialize_centers(self.mu, self.target, self.point, self.centers)?;
        let mut level = n - 1;
        self.enter(level, 0.0);
        loop {
            let Some(value) = self.children[level].next() else {
                if level == n - 1 {
                    return Ok(());
                }
                level += 1;
                continue;
            };
            if self.nodes >= self.node_budget {
                return Err(DecodeError::BudgetExhausted {
                    nodes: self.nodes,
                    radius_sq: self.radius_sq,
                });
            }
            self.nodes += 1;
            update_centers(self.mu, self.point, self.centers, level, value);
            let delta = self.centers[level] - value as f64;
            let next = self.partials[level] + self.weights[level] * delta * delta;
            if next > self.radius_sq {
                continue;
            }
            if level == 0 {
                self.list.push(ListPoint {
                    point: self.point.to_vec(),
                    distance_sq: if next == 0.0 { 0.0 } else { next },
                });
            } else {
                level -= 1;
                self.enter(level, next);
            }
        }
    }
}

fn initialize_centers(
    mu: &[f64],
    target: &[f64],
    point: &mut [i64],
    centers: &mut [f64],
) -> Result<(), DecodeError> {
    point.fill(0);
    centers.copy_from_slice(target);
    for higher in 1..target.len() {
        let residual = target[higher];
        let row = &mu[higher * target.len()..(higher + 1) * target.len()];
        for lower in 0..higher {
            centers[lower] += residual * row[lower];
        }
    }
    if let Some(index) = centers.iter().position(|value| !value.is_finite()) {
        return Err(DecodeError::NonFinite { index });
    }
    Ok(())
}

fn update_centers(
    mu: &[f64],
    point: &mut [i64],
    centers: &mut [f64],
    level: usize,
    value: i64,
) {
    let old = point[level];
    if old == value {
        return;
    }
    let n = point.len();
    let change = old as f64 - value as f64;
    for lower in 0..level {
        centers[lower] += change * mu[level * n + lower];
    }
    point[level] = value;
}

fn triangular_distance(
    mu: &[f64],
    weights: &[f64],
    target: &[f64],
    point: &[i64],
) -> Result<f64, DecodeError> {
    let mut total = 0.0;
    for level in (0..target.len()).rev() {
        let value = center(mu, target, point, level)?;
        let delta = value - point[level] as f64;
        total += weights[level] * delta * delta;
    }
    if total.is_finite() {
        Ok(total)
    } else {
        Err(DecodeError::InvalidRadius { radius_sq: total })
    }
}

fn center(
    mu: &[f64],
    target: &[f64],
    point: &[i64],
    level: usize,
) -> Result<f64, DecodeError> {
    let n = target.len();
    let mut value = target[level];
    for j in level + 1..n {
        value += (target[j] - point[j] as f64) * mu[j * n + level];
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(DecodeError::NonFinite { index: level })
    }
}

#[derive(Debug, Clone)]
struct Children {
    center: f64,
    remaining: f64,
    weight: f64,
    nearest: i64,
    step: i64,
    emitted_nearest: bool,
    positive_done: bool,
    negative_done: bool,
}

impl Children {
    const fn empty() -> Self {
        Self {
            center: 0.0,
            remaining: 0.0,
            weight: 1.0,
            nearest: 0,
            step: 1,
            emitted_nearest: true,
            positive_done: true,
            negative_done: true,
        }
    }

    fn new(center: f64, remaining: f64, weight: f64) -> Option<Self> {
        if remaining < 0.0 || !remaining.is_finite() || weight <= 0.0 || !weight.is_finite() {
            return None;
        }
        let nearest = round_away(center);
        let delta = center - nearest as f64;
        if weight * delta * delta > remaining {
            return None;
        }
        Some(Self {
            center,
            remaining,
            weight,
            nearest,
            step: 1,
            emitted_nearest: false,
            positive_done: false,
            negative_done: false,
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
        while !self.positive_done || !self.negative_done {
            let magnitude = self.step.checked_add(1)? / 2;
            let positive_first = self.center >= self.nearest as f64;
            let positive = if self.step % 2 == 1 {
                positive_first
            } else {
                !positive_first
            };
            self.step = self.step.checked_add(1)?;
            if (positive && self.positive_done) || (!positive && self.negative_done) {
                continue;
            }
            let candidate = if positive {
                self.nearest.checked_add(magnitude)
            } else {
                self.nearest.checked_sub(magnitude)
            };
            let Some(candidate) = candidate else {
                if positive {
                    self.positive_done = true;
                } else {
                    self.negative_done = true;
                }
                continue;
            };
            let delta = self.center - candidate as f64;
            if self.weight * delta * delta <= self.remaining {
                return Some(candidate);
            }
            if positive {
                self.positive_done = true;
            } else {
                self.negative_done = true;
            }
        }
        None
    }
}

#[allow(clippy::cast_possible_truncation)]
fn round_away(value: f64) -> i64 {
    let truncated = value as i64;
    if truncated == i64::MAX || truncated == i64::MIN {
        return truncated;
    }
    let fraction = value - truncated as f64;
    if fraction >= 0.5 {
        truncated + 1
    } else if fraction <= -0.5 {
        truncated - 1
    } else {
        truncated
    }
}

//! Shared arithmetic for point lattices in `Z^n` and `R^n`, underneath
//! lattice-based erasure and error-correcting codes.
//!
//! The guiding rule: *given a lattice, do arithmetic on it fast and exactly;
//! never construct the combinatorial object that defines it.* Codes, sparsity
//! patterns, and graphs arrive through seams — they are never generated here.
//!
//! # Exactness
//!
//! Every operation on an integral lattice is exact integer arithmetic with
//! checked overflow at the boundaries. This is enforced structurally rather
//! than by convention: [`Int`] exposes no `Add`/`Sub`/`Mul` operators, only
//! [`try_add`](Int::try_add), [`try_sub`](Int::try_sub),
//! [`try_mul`](Int::try_mul) and friends returning [`RangeError`]. A wrapped
//! determinant would produce a wrong basis that passes every downstream shape
//! check, so wrapping is made unrepresentable instead of merely discouraged.
//!
//! There is no arbitrary-precision fallback. A geometry whose intermediate
//! values exceed the chosen width is rejected with the bound it violated.
//!
//! ```
//! use lattica::int::{IntMatrix, hnf};
//!
//! // A basis of the same lattice as the identity, in disguise.
//! let a = IntMatrix::<i64>::from_rows(2, 2, &[3, 5, 1, 2])?;
//! let reduced = hnf(&a)?;
//!
//! // The Hermite normal form recovers the canonical basis, and the transform
//! // that produced it is unimodular -- so the lattice is provably unchanged.
//! assert_eq!(reduced.h.row(0), &[1, 0]);
//! assert_eq!(reduced.h.row(1), &[0, 1]);
//! assert_eq!(reduced.u.det()?.abs(), 1);
//! # Ok::<(), lattica::RangeError>(())
//! ```
//!
//! # Representation
//!
//! A lattice vector is an integer **coordinate** vector, and every metric
//! quantity comes from the [`Gram`] matrix: `‖x‖² = c G cᵀ`. `E_8` has
//! half-integer ambient coordinates but an integral Gram matrix, so working in
//! coordinates keeps every lattice on the exact integer path rather than only
//! those that sit inside `Z^n`.
//!
//! ```
//! use lattica::named::e8;
//! use lattica::shortvec::{DEFAULT_NODE_BUDGET, census};
//!
//! let g = e8::<i64>().unwrap();
//! let c = census(&g, DEFAULT_NODE_BUDGET).unwrap();
//!
//! // Recovered by enumeration, not stored: E_8 is unimodular, even, and has
//! // 240 minimal vectors.
//! assert_eq!(g.det().unwrap(), 1);
//! assert_eq!(c.min_norm_sq, Some(2));
//! assert_eq!(c.kissing_number, 240);
//! ```
//!
//! # Status
//!
//! Implemented: the scalar contract, exact integer linear algebra, the `Z_q`
//! ring, lattice representation, named lattices through `BW_16` and `Λ_24`,
//! exact short-vector enumeration, closed-form and maximum-likelihood
//! quantizers, `mod Λ`, nested pairs, Construction A/D, fraction-free GSO,
//! LLL, Babai, budgeted Schnorr–Euchner nearest/list enumeration,
//! low-dimensional Voronoi-relevant vectors, and dispatched real-vector batch
//! transforms.
//!
//! `cargo run --release --example e8_awgn` runs a nested `E_8` lattice code
//! over a simulated AWGN channel and checks that it reproduces the published
//! shaping gain of the `E_8` Voronoi region.

#![forbid(unsafe_code)]
pub mod basis;
pub mod construct;
pub mod error;
pub mod gso;
pub mod int;
pub mod kernel;
pub mod named;
pub mod nested;
pub mod quant;
pub mod reduce;
pub mod shortvec;
pub mod zq;

pub use basis::{Basis, Gram};
pub use error::{DecodeError, LatticeError, Op, RangeError, ReduceError};
pub use int::Int;
pub use nested::Nested;
pub use quant::{Quantizer, Scratch, mod_lattice};
pub use reduce::{Delta, Reduced, lll};
pub use shortvec::{Census, census};
pub use zq::Zq;

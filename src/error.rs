//! Error types.
//!
//! One enum per concern and no stringly-typed variants: an error carries the
//! quantity that was violated so a caller can act on it rather than log it.
//!
//! Validation happens before mutation everywhere in this crate. A call that
//! returns an error leaves every output buffer and all internal state exactly
//! as it was.

use core::fmt;

/// The arithmetic operation that overflowed.
///
/// Carried by [`RangeError::Overflow`] so a caller can tell an addition
/// boundary from a determinant blow-up without parsing a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Op {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Negation, or absolute value of the type minimum.
    Neg,
    /// Division, including division by zero.
    Div,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "addition",
            Self::Sub => "subtraction",
            Self::Mul => "multiplication",
            Self::Neg => "negation",
            Self::Div => "division",
        };
        f.write_str(s)
    }
}

/// A fixed-width budget was exceeded.
///
/// This crate has no arbitrary-precision fallback by design: a geometry that
/// needs one is out of range for a real-time forward error correction code and
/// must fail loudly rather than get slow. Every variant names the bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RangeError {
    /// An operation on the integer path overflowed its fixed width.
    Overflow {
        /// Which operation overflowed.
        op: Op,
        /// Width of the integer type in bits.
        width_bits: u32,
    },
    /// A division was expected to be exact and was not.
    ///
    /// Reachable only through an internal invariant violation — the
    /// fraction-free algorithms in [`crate::int`] divide only where Sylvester's
    /// identity guarantees exactness — so it indicates a bug, not bad input.
    InexactDivision,
    /// A dimension exceeded the configured maximum.
    Dimension {
        /// The dimension that was requested.
        requested: usize,
        /// The largest supported dimension.
        max: usize,
    },
    /// A matrix shape did not match the data supplied for it, or two operands
    /// had incompatible shapes.
    Shape {
        /// Number of elements expected.
        expected: usize,
        /// Number of elements supplied.
        found: usize,
    },
}

impl fmt::Display for RangeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow { op, width_bits } => {
                write!(f, "{op} overflowed {width_bits}-bit integer")
            }
            Self::InexactDivision => f.write_str("expected an exact division"),
            Self::Dimension { requested, max } => {
                write!(f, "dimension {requested} exceeds the maximum of {max}")
            }
            Self::Shape { expected, found } => {
                write!(f, "expected {expected} elements, found {found}")
            }
        }
    }
}
/// Failure while enumerating lattice vectors or proving a lattice fact.
///
/// This is the exact, integral side of the crate: complete enumeration over a
/// coordinate ball and structural classification. The decode vocabulary —
/// radius-shrinking search over a real basis, quantization, `mod Λ` — makes
/// decisions about real targets and belongs to the layer above, so none of it
/// appears here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EnumerationError {
    /// The supplied form is not the Gram matrix of a lattice.
    ///
    /// Enumeration needs a positive-definite form; anything else does not
    /// describe a lattice, and answering a query about it would present
    /// nonsense as a proved fact.
    NotALattice,
    /// An exact enumeration exceeded its node budget.
    ///
    /// The walk is complete and integral: there is no radius to report, only
    /// the node count at which the budget ran out.
    EnumerationBudget {
        /// Nodes visited before the budget was reached.
        nodes: u64,
    },
    /// The enumeration radius was negative.
    ///
    /// A squared radius is a squared length; a negative one is caller error,
    /// not an empty ball.
    InvalidRadius {
        /// The rejected squared radius.
        radius_sq: i128,
    },
    /// The integer path overflowed while forming the answer.
    Range(RangeError),
}

impl From<RangeError> for EnumerationError {
    fn from(e: RangeError) -> Self {
        Self::Range(e)
    }
}

impl fmt::Display for EnumerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotALattice => f.write_str("form is not positive definite"),
            Self::EnumerationBudget { nodes } => {
                write!(f, "enumeration budget exhausted after {nodes} nodes")
            }
            Self::InvalidRadius { radius_sq } => {
                write!(f, "invalid squared enumeration radius {radius_sq}")
            }
            Self::Range(e) => write!(f, "{e}"),
        }
    }
}

/// Failure during basis reduction or orthogonalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReduceError {
    /// The matrix was singular where an invertible one was required.
    Singular,
    /// The basis did not have full rank.
    NotFullRank {
        /// The rank actually found.
        rank: usize,
        /// The rank required.
        required: usize,
    },
    /// A reduction loop exceeded its iteration ceiling.
    ///
    /// LLL terminates by a potential argument, so this indicates a bug in the
    /// descent rather than a hard input. It exists so that such a bug is an
    /// error instead of a hang.
    BudgetExhausted {
        /// Iterations performed before the ceiling was reached.
        steps: u64,
    },
    /// The exact integer path exceeded its width budget.
    Range(RangeError),
}

impl From<RangeError> for ReduceError {
    fn from(e: RangeError) -> Self {
        Self::Range(e)
    }
}

impl fmt::Display for ReduceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Singular => f.write_str("matrix is singular"),
            Self::NotFullRank { rank, required } => {
                write!(f, "rank {rank} is below the required {required}")
            }
            Self::BudgetExhausted { steps } => {
                write!(f, "reduction did not converge after {steps} steps")
            }
            Self::Range(e) => write!(f, "{e}"),
        }
    }
}

/// Construction-time validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LatticeError {
    /// A modulus was zero, or otherwise outside the supported range.
    BadModulus,
    /// A supplied support was malformed: unsorted, out of range, duplicated,
    /// or of inconsistent degree.
    BadSupport,
    /// A nested pair failed its inclusion check: the fine lattice is not a
    /// sublattice of the coarse one.
    NotNested,
    /// A basis was degenerate for the requested construction.
    Degenerate,
    /// The exact integer path exceeded its width budget.
    Range(RangeError),
}

impl From<RangeError> for LatticeError {
    fn from(e: RangeError) -> Self {
        Self::Range(e)
    }
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadModulus => f.write_str("modulus is out of range"),
            Self::BadSupport => f.write_str("support is malformed"),
            Self::NotNested => f.write_str("lattices are not nested"),
            Self::Degenerate => f.write_str("basis is degenerate"),
            Self::Range(e) => write!(f, "{e}"),
        }
    }
}

mod std_impls {
    use super::{EnumerationError, LatticeError, RangeError, ReduceError};

    impl std::error::Error for RangeError {}

    impl std::error::Error for EnumerationError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Range(e) => Some(e),
                _ => None,
            }
        }
    }

    impl std::error::Error for ReduceError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Range(e) => Some(e),
                _ => None,
            }
        }
    }

    impl std::error::Error for LatticeError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Range(e) => Some(e),
                _ => None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EnumerationError, LatticeError, Op, RangeError, ReduceError};
    use std::error::Error as _;

    fn overflow() -> RangeError {
        RangeError::Overflow {
            op: Op::Mul,
            width_bits: 64,
        }
    }

    #[test]
    fn operation_and_range_displays_preserve_details() {
        for (op, expected) in [
            (Op::Add, "addition"),
            (Op::Sub, "subtraction"),
            (Op::Mul, "multiplication"),
            (Op::Neg, "negation"),
            (Op::Div, "division"),
        ] {
            assert_eq!(op.to_string(), expected);
        }

        assert_eq!(
            overflow().to_string(),
            "multiplication overflowed 64-bit integer"
        );
        assert_eq!(
            RangeError::InexactDivision.to_string(),
            "expected an exact division"
        );
        assert_eq!(
            RangeError::Dimension {
                requested: 25,
                max: 24
            }
            .to_string(),
            "dimension 25 exceeds the maximum of 24"
        );
        assert_eq!(
            RangeError::Shape {
                expected: 8,
                found: 7
            }
            .to_string(),
            "expected 8 elements, found 7"
        );
        assert!(overflow().source().is_none());
    }

    #[test]
    fn enumeration_displays_and_sources_preserve_details() {
        // The vocabulary carries no float payload, so it is totally ordered.
        fn assert_total<E: Eq>() {}

        for (error, expected) in [
            (
                EnumerationError::NotALattice,
                "form is not positive definite",
            ),
            (
                EnumerationError::EnumerationBudget { nodes: 9 },
                "enumeration budget exhausted after 9 nodes",
            ),
            (
                EnumerationError::InvalidRadius { radius_sq: -3 },
                "invalid squared enumeration radius -3",
            ),
            (
                EnumerationError::from(overflow()),
                "multiplication overflowed 64-bit integer",
            ),
        ] {
            assert_eq!(error.to_string(), expected);
        }
        assert!(EnumerationError::from(overflow()).source().is_some());
        assert!(EnumerationError::NotALattice.source().is_none());
        assert_total::<EnumerationError>();
    }

    #[test]
    fn reduction_and_lattice_displays_preserve_details() {
        for (error, expected) in [
            (ReduceError::Singular, "matrix is singular"),
            (
                ReduceError::NotFullRank {
                    rank: 2,
                    required: 3,
                },
                "rank 2 is below the required 3",
            ),
            (
                ReduceError::BudgetExhausted { steps: 17 },
                "reduction did not converge after 17 steps",
            ),
            (
                ReduceError::from(overflow()),
                "multiplication overflowed 64-bit integer",
            ),
        ] {
            assert_eq!(error.to_string(), expected);
        }
        assert!(ReduceError::from(overflow()).source().is_some());
        assert!(ReduceError::Singular.source().is_none());

        for (error, expected) in [
            (LatticeError::BadModulus, "modulus is out of range"),
            (LatticeError::BadSupport, "support is malformed"),
            (LatticeError::NotNested, "lattices are not nested"),
            (LatticeError::Degenerate, "basis is degenerate"),
            (
                LatticeError::from(overflow()),
                "multiplication overflowed 64-bit integer",
            ),
        ] {
            assert_eq!(error.to_string(), expected);
        }
        assert!(LatticeError::from(overflow()).source().is_some());
        assert!(LatticeError::Degenerate.source().is_none());
    }
}

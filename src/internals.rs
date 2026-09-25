//! Unstable implementation APIs for experimentation and downstream tuning.
//!
//! Exempt from this crate's compatibility guarantees; anything here may
//! change or vanish in any release. Production consumers use the supported
//! public API at the crate root and in its public modules.

/// Exact integer linear-algebra implementation details.
pub mod int {
    pub use crate::int::det::{AdjugatePath, adjugate_profiled};
}

/// Dispatched real-vector kernels and their portable references.
pub mod kernel {
    pub use crate::kernel::portable::{
        transform_batch_scalar, transform_batch_soa_scalar, transform_scalar,
    };
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    pub use crate::kernel::x86::{
        transform_batch_soa_avx2, transform_batch_soa_fixed_16_block8,
        transform_batch_soa_fixed_24_block6, transform_batch_soa_fixed_24_block8,
        transform_batch_soa_fixed_24_block12,
    };
}

/// Basis-reduction implementation details and benchmark counters.
pub mod reduce {
    pub use crate::reduce::unstable::{
        ReductionStats, ReductionWorkspace, lll_deep_profiled, lll_profiled,
    };
}

/// Voronoi-relevant enumeration implementation details.
pub mod relevant {
    pub use crate::relevant::unstable::{
        RelevantScratch, RelevantStats, relevant_vectors_profiled,
    };
}

/// Short-vector enumeration implementation details.
pub mod shortvec {
    pub use crate::shortvec::unstable::{
        EnumerationScratch, EnumerationStats, census_profiled, for_each_short_profiled,
    };
}

/// Residue-composition operations, outside the lattice path.
pub mod zq {
    pub use crate::zq::composition::ZqComposition;
}

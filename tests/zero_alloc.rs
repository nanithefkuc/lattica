//! Invariant: the hot paths allocate nothing in steady state.
//!
//! The decode paths moved to `lattice-engine` with their counting-allocator
//! suite. What remains hot here is the dispatched real-vector kernel and the
//! materializing structural enumerations; this measures them directly.
//!
//! The counter is thread-local rather than global: the test harness allocates
//! on its own threads, and a global count would measure that noise instead of
//! the code under test. `Cell` with a `const` initializer keeps the counting
//! path itself allocation-free, so it cannot recurse into the allocator.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

// SAFETY: every method forwards to `System` unchanged; the only addition is a
// thread-local increment that performs no allocation of its own.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(layout) }
    }
}

fn bump() {
    let _ = ALLOCATIONS.try_with(|c| c.set(c.get() + 1));
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Runs `body` and returns how many allocations it made on this thread.
fn allocations_during<F: FnOnce()>(body: F) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    body();
    ALLOCATIONS.with(Cell::get) - before
}

#[test]
fn steady_state_transform_batches_allocate_nothing() {
    use lattica::kernel::transform_batch_soa;

    const DIMENSION: usize = 16;
    const VECTORS: usize = 257;
    // A column-major transform: the dispatched 16-output shape.
    let mut matrix = [0.0f64; DIMENSION * DIMENSION];
    for (index, slot) in matrix.iter_mut().enumerate() {
        *slot = f64::from(u32::try_from(index % 7).unwrap()) / 8.0 - 0.375;
    }
    let mut inputs = vec![0.0f64; DIMENSION * VECTORS];
    for (index, slot) in inputs.iter_mut().enumerate() {
        *slot = f64::from(u32::try_from(index % 31).unwrap()) / 16.0 - 0.9375;
    }
    let mut outputs = vec![0.0f64; DIMENSION * VECTORS];

    transform_batch_soa(
        &matrix,
        DIMENSION,
        DIMENSION,
        VECTORS,
        &inputs,
        &mut outputs,
    )
    .unwrap();
    let allocations = allocations_during(|| {
        transform_batch_soa(
            &matrix,
            DIMENSION,
            DIMENSION,
            VECTORS,
            &inputs,
            &mut outputs,
        )
        .unwrap();
    });
    assert_eq!(
        allocations, 0,
        "warm batch transform allocated {allocations} times"
    );
}

#[test]
fn materializing_enumerations_allocate_as_promised() {
    use lattica::named::d_n;
    use lattica::relevant::relevant_vectors;
    use lattica::shortvec::census;

    let gram = d_n::<i64>(4).unwrap();

    let relevant_allocations = allocations_during(|| {
        assert!(!relevant_vectors(&gram, 1 << 20).unwrap().is_empty());
    });
    assert!(
        relevant_allocations > 0,
        "relevant-vector materialization did not allocate"
    );

    let census_allocations = allocations_during(|| {
        assert!(census(&gram, 1 << 20).unwrap().min_norm_sq.is_some());
    });
    assert!(census_allocations > 0, "cold exact census did not allocate");
}

/// A deterministic skewed positive-definite Gram matrix of one dimension.
#[cfg(feature = "internals")]
fn skewed_gram(dimension: usize) -> lattica::basis::Gram<i64> {
    let mut rng = 0x5EED_5EED_0000u64 ^ u64::try_from(dimension).unwrap();
    let mut entries = vec![0i64; dimension * dimension];
    for row in 0..dimension {
        entries[row * dimension + row] = 1;
        for column in 0..row {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            entries[row * dimension + column] = i64::try_from(rng % 7).unwrap() - 3;
        }
    }
    lattica::Basis::from_rows(dimension, dimension, &entries)
        .unwrap()
        .gram()
        .unwrap()
}

#[test]
#[cfg(feature = "internals")]
fn prepared_reduction_allocates_only_its_results() {
    use lattica::reduce::{Delta, ReductionWorkspace, lll, lll_deep};

    const DIMENSION: usize = 12;
    let gram = skewed_gram(DIMENSION);
    let mut workspace = ReductionWorkspace::<i64>::new(DIMENSION).unwrap();

    // Warm every path once, then count a steady-state call. The only allowed
    // allocations are the returned Gram matrix buffer and transform buffer.
    // The one-shot results double as the differential oracle.
    let ordinary_expected = lll(&gram, Delta::STRONG).unwrap();
    let deep_expected = lll_deep(&gram, Delta::STRONG).unwrap();
    drop(workspace.reduce(&gram, Delta::STRONG).unwrap());
    let ordinary_allocations = allocations_during(|| {
        assert_eq!(
            workspace.reduce(&gram, Delta::STRONG).unwrap().gram,
            ordinary_expected.gram
        );
    });
    assert_eq!(
        ordinary_allocations, 2,
        "warm prepared reduction allocated {ordinary_allocations} times"
    );

    drop(workspace.reduce_deep(&gram, Delta::STRONG).unwrap());
    let deep_allocations = allocations_during(|| {
        assert_eq!(
            workspace.reduce_deep(&gram, Delta::STRONG).unwrap().gram,
            deep_expected.gram
        );
    });
    assert_eq!(
        deep_allocations, 2,
        "warm prepared deep reduction allocated {deep_allocations} times"
    );

    // The one-shot public path still pays for its own setup state, which is
    // precisely what the prepared form removes.
    let one_shot_allocations = allocations_during(|| {
        drop(lll(&gram, Delta::STRONG).unwrap());
    });
    assert!(
        one_shot_allocations > 2,
        "one-shot reduction allocated {one_shot_allocations} times"
    );
}

//! Invariant: the prepared steady-state paths allocate only as promised.
//!
//! The reusable workspaces and scratch buffers behind the `internals` facade
//! exist so repeated same-dimension callers stop paying per-call setup. This
//! measures those prepared paths directly against their one-shot oracles.

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

/// A deterministic skewed positive-definite Gram matrix of one dimension.
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
fn prepared_reduction_allocates_only_its_results() {
    use lattica::internals::reduce::ReductionWorkspace;
    use lattica::reduce::{Delta, lll, lll_deep};

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

#[test]
fn prepared_enumeration_allocates_nothing_steady_state() {
    use lattica::internals::shortvec::EnumerationScratch;
    use lattica::shortvec::{census, for_each_short};

    const DIMENSION: usize = 8;
    const BUDGET: u64 = 1 << 20;
    let gram = skewed_gram(DIMENSION);
    let mut scratch = EnumerationScratch::new(DIMENSION).unwrap();

    // Warm every path once, then count steady-state calls. Neither call
    // owns heap output, so both must allocate nothing. The one-shots
    // double as the differential oracle.
    let expected_census = census(&gram, BUDGET).unwrap();
    let _ = scratch.census(&gram, BUDGET).unwrap();
    let census_allocations = allocations_during(|| {
        assert_eq!(scratch.census(&gram, BUDGET).unwrap(), expected_census);
    });
    assert_eq!(
        census_allocations, 0,
        "warm prepared census allocated {census_allocations} times"
    );

    let expected_nodes = for_each_short(&gram, 4, BUDGET, |_, _| {}).unwrap();
    let _ = scratch.for_each(&gram, 4, BUDGET, |_, _| {}).unwrap();
    let for_each_allocations = allocations_during(|| {
        assert_eq!(
            scratch.for_each(&gram, 4, BUDGET, |_, _| {}).unwrap(),
            expected_nodes
        );
    });
    assert_eq!(
        for_each_allocations, 0,
        "warm prepared enumeration allocated {for_each_allocations} times"
    );
}

#[test]
fn prepared_relevant_vectors_reuse_buffers() {
    use lattica::internals::relevant::RelevantScratch;
    use lattica::named::d_n;
    use lattica::relevant::relevant_vectors;

    const BUDGET: u64 = 1 << 20;
    let gram = d_n::<i64>(4).unwrap();
    let mut scratch = RelevantScratch::<i64>::new(4).unwrap();

    // The one-shot result is the oracle; two warm steady-state calls must
    // agree with it and with each other while allocating strictly less
    // than the one-shot, which rebuilds every buffer per call.
    let expected = relevant_vectors(&gram, BUDGET).unwrap();
    drop(scratch.relevant_vectors(&gram, BUDGET).unwrap());
    let first = allocations_during(|| {
        assert_eq!(scratch.relevant_vectors(&gram, BUDGET).unwrap(), expected);
    });
    let second = allocations_during(|| {
        assert_eq!(scratch.relevant_vectors(&gram, BUDGET).unwrap(), expected);
    });
    assert_eq!(
        first, second,
        "steady-state relevant calls allocated {first} then {second} times"
    );
    let one_shot = allocations_during(|| {
        drop(relevant_vectors(&gram, BUDGET).unwrap());
    });
    assert!(
        first < one_shot,
        "prepared relevant allocated {first} times against one-shot {one_shot}"
    );
}

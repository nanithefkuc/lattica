//! The codebook materializer must survive an exhausted allocator.
//!
//! `Nested::coset_representatives` promises a `LatticeError::Range` when the
//! codebook does not fit in memory. That contract covers *every* allocation
//! on the path — the outer vector, the scratch buffer, and each
//! representative row — not just the outer reservation. This binary runs
//! alone with a budgeted global allocator (its own process, so the budget
//! cannot disturb any other test) and denies allocation outright to prove
//! the failure surfaces as the documented error rather than an abort.

#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use lattica::Nested;
use lattica::error::LatticeError;
use lattica::int::IntMatrix;
use lattica::named::zn;

/// Largest admitted allocation; `usize::MAX` admits everything.
static BUDGET: AtomicUsize = AtomicUsize::new(usize::MAX);

struct Budgeted;

// SAFETY: every method forwards to `System` unchanged; the only addition is
// a size comparison against an atomic that itself allocates nothing.
unsafe impl GlobalAlloc for Budgeted {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > BUDGET.load(Ordering::Relaxed) {
            return core::ptr::null_mut();
        }
        // SAFETY: forwarding to `System` with the layout we were given.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarding the pointer and layout `System` produced.
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static GLOBAL: Budgeted = Budgeted;

#[test]
fn an_exhausted_allocator_is_the_documented_error() {
    let transform = IntMatrix::<i64>::from_rows(2, 2, &[2, 0, 0, 2]).unwrap();
    let pair = Nested::new(zn(2).unwrap(), transform).unwrap();
    assert_eq!(pair.index(), 4);

    // Deny the fallible reservations and every row allocation. A single
    // admitted byte keeps the harness's own bookkeeping allocations alive;
    // denying allocation outright lets a harness thread race the restore and
    // abort instead of producing the documented error.
    BUDGET.store(1, Ordering::Relaxed);
    let result = pair.coset_representatives();
    BUDGET.store(usize::MAX, Ordering::Relaxed);

    assert!(matches!(result, Err(LatticeError::Range(_))));
}

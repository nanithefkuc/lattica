//! Invariant I5: the hot path allocates nothing in steady state.
//!
//! A decoder that allocates per received vector is not a decoder for a
//! real-time codec, and the cost is invisible in a throughput benchmark on an
//! unloaded machine. This measures it directly.
//!
//! The counter is thread-local rather than global: the test harness allocates
//! on its own threads, and a global count would measure that noise instead of
//! the decoder. `Cell` with a `const` initializer keeps the counting path
//! itself allocation-free, so it cannot recurse into the allocator.

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

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump();
        unsafe { System.realloc(ptr, layout, new_size) }
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

use lattica::quant::{An, Dn, DnPlus, Quantizer, Scratch, Zn, e8, nearest_batch};

/// Deterministic dyadic coordinates, so the measurement is reproducible.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (f64::from(u32::try_from(self.0 % 8192).unwrap()) - 4096.0) / 1024.0
    }
}

fn check_batch(name: &str, q: &dyn Quantizer, vectors: usize) {
    let dim = q.dim();
    let mut rng = Rng(0xC0FF_EE00_1234_5678);
    let points: Vec<f64> = (0..dim * vectors).map(|_| rng.next()).collect();
    let mut out = vec![0i64; dim * vectors];
    let mut scratch = Scratch::new(dim);

    // Warm every buffer to its high-water mark before measuring.
    nearest_batch(q, &points[..dim], &mut out[..dim], &mut scratch).unwrap();

    let allocations = allocations_during(|| {
        nearest_batch(q, &points, &mut out, &mut scratch).unwrap();
    });
    assert_eq!(
        allocations, 0,
        "{name} allocated {allocations} times over {vectors} vectors"
    );
}

#[test]
fn steady_state_batch_decoding_allocates_nothing() {
    // Roughly one million decoded coordinates per family.
    check_batch("Z^24", &Zn::new(24).unwrap(), 40_000);
    check_batch("D_24", &Dn::new(24).unwrap(), 40_000);
    check_batch("A_23", &An::new(23).unwrap(), 40_000);
    check_batch("D_24^+", &DnPlus::new(24).unwrap(), 40_000);
    check_batch("E_8", &e8(), 125_000);
}

#[test]
fn a_cold_scratch_grows_once_and_never_again() {
    let q = Dn::new(16).unwrap();
    let mut scratch = Scratch::default();
    let mut out = [0i64; 16];
    let x = [0.3f64; 16];

    // The first call is allowed to allocate; it is the only one that may.
    let cold = allocations_during(|| {
        q.nearest(&x, &mut out, &mut scratch).unwrap();
    });
    assert!(cold > 0, "expected the cold path to allocate its buffers");

    let warm = allocations_during(|| {
        for _ in 0..10_000 {
            q.nearest(&x, &mut out, &mut scratch).unwrap();
        }
    });
    assert_eq!(warm, 0, "warm decoding allocated {warm} times");
}

#[test]
fn error_paths_do_not_allocate_either() {
    let q = Dn::new(8).unwrap();
    let mut scratch = Scratch::new(8);
    let mut out = [0i64; 8];
    let bad = [f64::NAN; 8];
    let short = [0.0f64; 4];

    // Warm up.
    q.nearest(&[0.5f64; 8], &mut out, &mut scratch).unwrap();

    let allocations = allocations_during(|| {
        for _ in 0..1000 {
            assert!(q.nearest(&bad, &mut out, &mut scratch).is_err());
            assert!(q.nearest(&short, &mut out, &mut scratch).is_err());
        }
    });
    assert_eq!(allocations, 0, "rejection allocated {allocations} times");
}

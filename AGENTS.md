# Repository Guidelines

Working rules for changing `lattica`. Item and module rustdoc owns API
contracts, the crate-level page is `README.md` included from `src/lib.rs` so
scope and usage are written once, `BENCHMARKS.md` owns public measurements,
and `CHANGELOG.md` owns release changes and migration guidance.

## Required workflow

Work from the crate root and use `just` for routine commands:

```sh
just doctor            # required tools and toolchains
just test [ARGS]       # focused tests on the host's selected backend
just test-tiers        # requested backend sweep
just features          # no-default, default, all-features
just lint              # formatting and clippy at both feature ends
just fmt               # format in place
just doc               # rustdoc with warnings denied
just msrv              # all features and targets on the minimum toolchain
just cover             # merged per-tier coverage, 95% minimum
just validate          # complete pull-request gate
```

Run a focused regression before the complete gate and `just validate` before
submitting a change. Validation includes lint, dependencies, documentation,
feature tests, backend tests, unsafe checks, and coverage; it does not include
benchmarks or the separate MSRV check. Do not replace a supported recipe with
a bare Cargo command. Fix the recipe when its behavior is insufficient.

`justfile` is a shared, byte-identical command surface; do not edit the vendored
copy. Crate-specific values and recipes belong in `crate.just`. Keep its
`MSRV`, `TIERS`, `MIRI`, and `COV_IGNORE` consistent with the contracts below.
The crate uses edition 2024 and MSRV Rust 1.89.

## Scope and dependencies

`lattica` owns point-lattice arithmetic: exact integer linear algebra,
Gram-based reduction, nested cosets, structural enumeration, and real-vector
transforms over its own layouts. It computes lattice facts, not nearest-point
decisions. Quantization, target-centered decoding, code and graph generation,
and lattice cryptography stay outside this crate. Construction A/D receives
caller-supplied generator matrices; it does not generate the defining code.

Runtime dependencies are limited to exact-pinned `simdispatch` and `archmage`,
both optional under the default `simd` feature. `criterion` is dev-only.
Disabling default features removes dispatch dependencies, not `std`: this
crate does not support `no_std`. Do not add another arithmetic domain or a
runtime competitor dependency.

## Exact arithmetic and representation

- `Int` exposes checked operations returning `RangeError`, not unchecked
  `Add`, `Sub`, or `Mul` operators. Preserve checked intermediate arithmetic;
  a wrapped value can describe the wrong lattice while passing shape checks.
- Integral answers remain exact. Reject values outside the fixed-width budget;
  do not add arbitrary-precision fallback or silently downgrade to floating
  point. Real generators and real-vector kernels do not relax this boundary.
- A lattice vector is an integer coordinate vector. Metric quantities use the
  Gram matrix, `c G cᵀ`, rather than assuming integral ambient coordinates.
- Named generators may use published coefficients. Determinants, minimal
  norms, and kissing numbers must be computed from the construction, not stored
  as answers that make verification circular.
- Reduction operates on the Gram matrix with an exact rational `Delta` and
  fraction-free GSO. Preserve the same reduced-basis predicate used by
  `reduce::is_reduced`; floating-point approximations must not decide it.
- Validate shape, index, and arithmetic bounds before committing mutations.
  A rejected mutable operation preserves its output and state; an overflow
  discovered partway through a loop is not permission to leave partial writes.
- Reusable preparation and mutable execution workspace are distinct contracts.
  State what construction and application allocate. Prepared state must not
  bypass one-shot validation or silently narrow its accepted inputs.

## Modules and experimental APIs

Keep the module-named file style: `int.rs` beside `int/`, `kernel.rs` beside
`kernel/`. Parent modules hold documentation, declarations, re-exports, and
genuinely shared items; split implementations along a named responsibility.
Extend an existing algorithm or family instead of adding a parallel engine
for the same operation.

`src/internals.rs` is the sole feature-gated, re-export-only facade.
`internals = []` changes reachability, never implementation compilation or
dependency activation, and carries no compatibility guarantee. Keep portable
oracles, profiling counters, and experimental kernels behind that boundary.
Benchmarks and integration targets that use it declare `required-features`;
private unit tests do not require the facade.

## Real-vector kernels and safety

`lattica` owns transform-buffer layouts. Preserve the documented coefficient
order and the distinction between strided vector batches and plane-major
structure-of-arrays batches. Layout conversion must not change arithmetic.

Dispatched results are bit-identical to the portable reference. Each output
accumulates input rows in scalar order, using separate multiplication and
addition. Do not introduce FMA, `mul_add`, reassociation, or a reordered
reduction in `kernel`.

`simdispatch` owns CPU detection, ordering, and the process-startup,
downgrade-only `SIMD_BACKEND` request. Resolve once through
`Selection::supports(LATTICA_TIERS)` and cache the result. Do not call the
unrestricted `simdispatch::backend()`, add local detection, or add another
override. Listed tiers must have an architecture-gated dispatch arm; tiers
sharing an implementation may share an arm.

Architecture entries use safe Archmage capability-token boundaries. Library
code retains `#![forbid(unsafe_code)]`; do not weaken it to introduce raw
intrinsics. Counting-allocator adapters in integration tests are outside that
attribute, so the library prohibition is not a claim that the repository has
no unsafe code. Changes to those adapters require narrow per-item allowances
and local SINCE–THUS proofs of their allocator obligations.

`TIERS` is `v3 scalar`. `V3GfniCrypto` shares the V3 dispatch arm and is
covered by direct kernel tests; it stays out of the sweep so every swept tier
resolves to a distinct arm on hosts without that tier, where a forced request
is silently ignored. A requested tier may be unsupported, and a selected
backend may still use the portable path for a particular geometry. Inspect the
resolved backend and exercise the kernel directly before claiming ISA coverage.
`MIRI` is empty: `just unsafe-check` reports a skip. `COV_IGNORE` is empty:
every library line counts toward the coverage gate.

## Verification

Use the built-in Rust harness and deterministic input helpers already present
in the relevant suite. Public contracts belong in integration tests; private
state belongs in local unit tests. Test observable values, error variants,
boundaries, and state preservation, not source text, forwarding, field copies,
or exact diagnostic wording.

An algorithm's expected answer must come from an independent oracle. Prefer an
exact certificate, then bounded brute force or a published constant. Use a
statistical check only with a fixed seed and a derived tolerance. LLL has no
canonical output basis: verify unimodularity, the Gram congruence, and the
reduced predicate rather than pinning one valid basis.

Preserve frozen representation and ordering fixtures. A changed expected value
requires evidence of a corrected oracle or an explicit contract change; do not
regenerate expectations from the implementation under test.

For kernels, compare directly against portable references across lane, tail,
batch, and geometry boundaries, including empty batches and scalar fallbacks.
Then run the backend sweep for routing. Allocation promises require counting-
allocator coverage: distinguish allocation-free transforms and warmed scratch
from materializing operations that allocate their results.

## Measurements and review

Use an identified CPU through `FEC_GOLDEN_CORE` and the benchmark recipes:

```sh
FEC_GOLDEN_CORE=<cpu> just bench-save kernel
FEC_GOLDEN_CORE=<cpu> just bench kernel
FEC_GOLDEN_CORE=<cpu> just bench optimization
FEC_GOLDEN_CORE=<cpu> just bench-fplll
```

`NAME` selects a benchmark target; the Criterion baseline is named `before`.
The `kernel` target uses Criterion. `optimization` emits custom CSV and needs
captured, interleaved before/after runs rather than Criterion baseline flags.
The crate-specific `bench-fplll` recipe runs only the Rust comparison binary;
it does not build or run the native comparison side. Build instructions and
comparison contracts belong in `BENCHMARKS.md`, not this manual.

A performance change requires independent correctness checks and paired
measurements in one session, with an unchanged control and matched geometry.
Keep input generation and reusable preparation outside timing; include result
allocation when it belongs to the measured public call. Record the actual
operation backend, affinity, toolchain, flags, warmup, and aggregation. Do not
change a dispatch threshold from reasoning or cross-session numbers alone.

Publish only public-operation and competitor measurements in `BENCHMARKS.md`.
Add rows to existing tables, keep units in headers, and put reproduction details
below them. Internal timings and experiment journals stay in ignored storage.
Do not put measurements or result commentary in source comments or commit
messages, or cite private planning material in public files.

Write documentation in third person and present tense, with one point per
paragraph. Public items need summary sentences and complete error, panic,
layout, and ownership contracts. Update `CHANGELOG.md` for user-visible changes;
breaking entries include migration guidance under `Unreleased`.

Review exported-symbol changes across all callers. Migrate callers and remove
obsolete aliases, wrappers, comments, and re-exports in the same change. Keep
commit subjects near ten words, shaped `lattica: short verb phrase`.

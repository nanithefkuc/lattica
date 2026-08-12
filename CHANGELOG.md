# Changelog

All notable changes to `lattica` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0]

Initial release: the shared arithmetic layer for point lattices in `Z^n` and
`R^n`. Given a lattice, it does arithmetic on it exactly and fast; it never
constructs the combinatorial object that defines the lattice. It is not a codec,
not a field library, and not a lattice-cryptography library.

### Added

- **Exact fixed-width integer substrate.** The `Int` scalar contract exposes
  only checked `try_add`/`try_sub`/`try_mul` and friends returning
  `Result<_, RangeError>`; there are no wrapping operators. Integer linear
  algebra over lattice bases: Hermite Normal Form, Smith Normal Form, unimodular
  transforms, and integer determinant, all exact with checked overflow at the
  boundaries.
- **`Z_q` ring.** Vector reduction, the centered representative, the lift
  `Z_q -> Z`, and the power-of-two `q` path, for shaping and Construction-A
  lifting.
- **Lattice representation.** `Gram` and `Basis` types, with determinant/volume,
  rank, and the Gram matrix `G = B^T B`. A lattice vector is an integer
  coordinate vector; every metric quantity comes from the Gram matrix, so `E_8`
  stays on the exact integer path despite its half-integer ambient coordinates.
- **Named lattices** as first-class constructions: `Z^n`, `A_n`, `D_n`, and
  `E_8`, provided as both Gram matrices and generator bases.
- **Exact short-vector enumeration** (`census`) with an explicit node budget,
  which *recovers* the classical determinants, minimal norms, kissing numbers,
  and theta series by computation rather than storing them.
- **Closed-form Conway–Sloane nearest-point decoders** for `Z^n`, `A_n`, `D_n`,
  and `D_n^+` (`E_8` at `n = 8`). Each is `O(n)` per point and uses only add,
  subtract, compare, and round, so two peers on different architectures agree at
  a Voronoi boundary. Tie-breaking is specified and fixture-tested.
- **`mod Λ` operator** with dithered modulo, and the shaping map to the Voronoi
  region.
- **Nested lattice pairs** (`Nested`) with the quotient index and coset
  representative enumeration; Construction A and Construction D.
- **Fraction-free Gram–Schmidt orthogonalization** (GSO), retaining the GSO
  coefficients and `‖b_i*‖²` for the quantizers.
- **Basis reduction:** Lagrange–Gauss (dim 2), LLL, and deep-insertion LLL over
  the integral Gram matrix with an exact rational `δ`, plus size reduction.
- **Babai** rounding and nearest-plane over a reduced basis.
- **Budgeted exact decoding.** Prepared Schnorr–Euchner nearest-point and list
  enumeration over general integral Gram matrices, with zig-zag child order,
  radius shrinking, deterministic node exhaustion, and lexicographic
  coordinate ties. List results are sorted by distance and the same total tie
  rule. Exact parity-coset enumeration recovers low-dimensional
  Voronoi-relevant vectors.
- **Barnes–Wall and Leech lattices.** Published algebraic generators produce
  exact integral Gram matrices for `BW_16` and `Λ_24`; maximum-likelihood
  ambient and coefficient decoders use the proved enumeration core. Ambient
  answers remain exact numerators over 2 and `sqrt(8)`, respectively.
- **Dispatched real-vector batches.** A structure-of-arrays transform preserves
  scalar accumulation order across AVX2 lanes and ragged tails. Stack-wide
  backend selection comes from `simdispatch`; unmeasured and losing shapes stay
  scalar. The scalar references are exposed by `internals`.
- **Competitor benchmark.** A pinned, in-process fplll 5.5.0 harness compares
  LLL and public CVP calls on identical deterministic inputs, with input,
  output, and distance fingerprints. The record includes reproducible fplll
  `CVPM_PROVED` correctness and Babai-cycle failures discovered while making
  the comparison fair.
- **`e8_awgn` example**, a nested `E_8` lattice code over a simulated AWGN
  channel that reproduces the published `0.6539 dB` shaping gain of the `E_8`
  Voronoi region, used as the release gate.

### Changed

- Exact LLL and deep-insertion reduction now update the symmetric Gram matrix
  and fraction-free GSO state transactionally after size reductions and
  adjacent swaps. The 16-basis comparison corpus improved by 11.5x, 48.9x, and
  112.8x at dimensions 8, 16, and 24 while retaining exact certificates,
  checked overflow, and one factorization.
- General CVP enumeration caches real triangular coefficients, uses
  square-root-free zig-zag stepping and reusable iterative state at dimension
  24 and above, and accepts independently validated initial candidates.
  `PreparedEnumerator` adds strongly reduced-basis search with exact coordinate
  mapping back to the caller's basis; named Barnes–Wall and Leech decoders use
  that prepared form.
- Positive-definiteness uses one fraction-free factorization, adjugates share
  one fraction-free solve with a cofactor fallback for the accepted overflow
  domain, triangular determinants use their diagonal product, and symmetric
  Gram construction computes one triangle.
- The optimization benchmark reports operation counts, proof-tree nodes,
  nanoseconds per node, exact-algebra costs, batch geometry, allocation counts,
  and deterministic correctness fingerprints. Build-option and rejected
  crossover measurements are recorded in `BENCHMARKS.md`.

### Notes

- Edition 2024 and MSRV 1.89. Runtime dependencies are limited to optional
  `simdispatch` and `archmage` under the default `simd` feature. `no_std` is not
  supported: real-basis reduction needs `sqrt` and friends, and a `libm`
  dependency would cost more than `std` does.
- `#![forbid(unsafe_code)]` at the crate root, including the SIMD module;
  archmage supplies safe capability-token boundaries and memory operations.
- Features: `simd` (default; measured AVX2 SoA batches with scalar fallback)
  and `internals` (unstable scalar references and implementation APIs).
- General-basis decoding requires both an explicit squared radius and a node
  budget. Radius exhaustion is distinct from node exhaustion, so a bounded
  search never presents an unproved candidate as a nearest point.
- Equal-distance general-basis candidates use lexicographic basis coordinates.
  This total order was chosen because it is independent of traversal and does
  not change when a later kernel changes child scheduling.
- The named high-dimensional decoders use exhaustive maximum-likelihood
  enumeration rather than bounded-distance Barnes–Wall and hexacode shortcuts.
  This costs a mandatory node budget but gives one honest contract: success is
  globally nearest, while exhaustion is explicit. Measured behavior beyond the
  unique decoding radius is in `BENCHMARKS.md`.
- SIMD dispatch is limited to 16-output SoA batches of at least 64 vectors.
  Smaller batches, AoS input, and other dimensions measured flat or slower and
  remain scalar rather than gaining a speculative crossover.

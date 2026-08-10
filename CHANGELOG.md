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
- **`e8_awgn` example**, a nested `E_8` lattice code over a simulated AWGN
  channel that reproduces the published `0.6539 dB` shaping gain of the `E_8`
  Voronoi region, used as the release gate.

### Notes

- Edition 2024, MSRV 1.89, and no runtime dependencies. `no_std` is not
  supported: real-basis reduction needs `sqrt` and friends, and a `libm`
  dependency would cost more than `std` does.
- `#![forbid(unsafe_code)]` at the crate root.
- Features: `simd` (default; currently selects the portable scalar kernels) and
  `internals` (unstable implementation APIs, exempt from compatibility
  guarantees).
- Not yet implemented: Schnorr–Euchner enumeration and list decoding.

> [!WARNING]
> This library was made with the help of AI. While the library has tests
to check for regressions, things may break. Audit the code yourself, or with
your own agent before using.

# lattica

`lattica` is the shared arithmetic layer for point lattices in `Z^n` and
`R^n`, underneath lattice-based erasure and error-correcting codes. It is
deliberately not a codec, not a field library, and not a cryptographic lattice
library.

It provides:

- The classical named lattices as first-class constructions — `Z^n`, `A_n`,
  `D_n`, `E_8`, `BW_16`, and `Λ_24` — plus nested pairs `Λ_s ⊆ Λ_c` with
  coset enumeration, and Construction A / Construction D over a
  caller-supplied code membership seam.
- Exact integer and modular substrate: fixed-width integer linear algebra
  (Hermite and Smith normal forms, integer determinant, unimodular
  transforms) and `Z_q` ring arithmetic with centered representatives and lift
  `Z_q → Z`.
- Offline reduction and orthogonalization: fraction-free Gram–Schmidt (GSO
  coefficients and `‖b_i*‖²` retained), Lagrange–Gauss, LLL and deep-insertion
  LLL with an exact rational `δ`, and size reduction.
- Hot-path quantization and decoding: Babai rounding and nearest-plane,
  budgeted Schnorr–Euchner nearest-point and list enumeration, exact
  maximum-likelihood `BW_16` and `Λ_24` decoders, Voronoi-relevant vectors in
  low dimension, the closed-form Conway–Sloane decoders, and `mod Λ`
  (`x - Q_Λ(x)`) with dithering and Voronoi-region shaping.

Design is split along an exactness seam. Every operation on an integral
lattice — membership, quantization of an integral point, `mod Λ`, coset
extraction, HNF, SNF, `det`, LLL — is exact integer arithmetic with checked
overflow at the boundaries; `f64` appears only on the genuinely real path (the
received vector), where the closed-form decoders use only add, subtract,
compare, and round-to-integer so identical inputs give identical outputs on
every platform. `#![forbid(unsafe_code)]` at the root.

## Implementation status

Working end to end for the lattices it names. A nested `E_8` lattice code built
from this crate reproduces the published `0.65 dB` shaping gain of the `E_8`
Voronoi region — see `cargo run --release --example e8_awgn`.

- **Implemented:** exact fixed-width integer arithmetic; Hermite and Smith
  normal forms; the `Z_q` ring; `Gram`/`Basis` representation; the named
  lattices `Z^n`/`A_n`/`D_n`/`E_8`/`BW_16`/`Λ_24`; exact short-vector
  enumeration (which *recovers* the classical determinants, minimal norms,
  kissing numbers and theta series rather than storing them); the closed-form
  Conway–Sloane decoders; exact maximum-likelihood `BW_16` and `Λ_24` decoding;
  `mod Λ` with dithering; nested pairs with coset enumeration; Construction
  A/D; fraction-free Gram–Schmidt; LLL and deep-insertion LLL with an exact
  rational `δ`; Lagrange–Gauss; Babai rounding and nearest-plane; budgeted
  Schnorr–Euchner nearest/list enumeration; exact low-dimensional
  Voronoi-relevant vectors; and a runtime-dispatched real-vector batch kernel.

## Usage

The MSRV is Rust 1.89.

`lattica` is distributed through git only; it is not published to
[crates.io](https://crates.io).

```toml
[dependencies]
lattica = { git = "https://github.com/nanithefkuc/lattica" }
```

### Features

| Feature | Result |
| --- | --- |
| default (`simd`) | `simdispatch`-selected AVX2 for measured 16-output structure-of-arrays batches; portable scalar fallback everywhere else |
| `internals` | unstable scalar references and implementation APIs, exempt from compatibility guarantees |

## Building

`lattica` builds on stable Rust (edition 2024, MSRV 1.89) with no target-feature
flags. The default `simd` feature resolves the stack-wide `SIMD_BACKEND` through
`simdispatch`; disabling it removes runtime dispatch and `archmage`:

```sh
cargo build                     # default: simd
cargo build --features internals
cargo test
```

Kernel crossovers, high-dimensional decoder measurements, and the reproducible
fplll and FLINT comparisons are recorded in [`BENCHMARKS.md`](BENCHMARKS.md).

`lattica` is **not** `no_std`: real-basis reduction needs `sqrt` and friends,
and a `libm` dependency would cost more than `std` does.

## License

MIT.

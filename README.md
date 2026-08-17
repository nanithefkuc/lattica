> [!WARNING]
> This library was made with the help of AI. While the library has tests
> to check for regressions, things can break. Audit the code yourself, or with
> your own agent before using.

# lattica

`lattica` is the shared arithmetic layer for point lattices in `Z^n` and
`R^n`, underneath lattice-based erasure and error-correcting codes. It is
deliberately not a codec, not a field library, and not a cryptographic lattice
library.

Deciding lattice points from real targets — quantization, sphere enumeration,
maximum-likelihood decoding, `mod Λ` — lives one layer up in
[`lattice-engine`](https://github.com/nanithefkuc/lattice-engine). This crate
computes *facts* about a lattice; it never selects a point.

It provides:

- The classical named lattices as first-class constructions — `Z^n`, `A_n`,
  `D_n`, `E_8`, `BW_16`, and `Λ_24` — plus nested pairs `Λ_s ⊆ Λ_c` with
  coset enumeration, and the code-free Construction A / Construction D
  generator constructions over caller-supplied generator matrices.
- Exact integer and modular substrate: fixed-width integer linear algebra
  (Hermite and Smith normal forms, integer determinant, unimodular
  transforms) and `Z_q` ring arithmetic with centered representatives and lift
  `Z_q → Z`.
- Offline reduction and orthogonalization: fraction-free Gram–Schmidt (GSO
  coefficients and `‖b_i*‖²` retained), Lagrange–Gauss, LLL and deep-insertion
  LLL with an exact rational `δ`, and size reduction.
- Structural enumeration: exact short-vector enumeration that *recovers* the
  classical determinants, minimal norms, kissing numbers, and theta series,
  and exact Voronoi-relevant vectors in low dimension.
- Dispatched real-vector batch transforms over `lattica`-owned layouts.

Design is split along an exactness seam. Every operation on an integral
lattice — membership, coset extraction, HNF, SNF, `det`, LLL, generator
constructions — is exact integer arithmetic with checked overflow at the
boundaries; `f64` appears only in the real-vector kernels (where dispatched
results are bit-identical to the scalar reference, with no FMA or
reassociation) and in published real generators such as `e8_generator`.
`#![forbid(unsafe_code)]` at the root.

## Implementation status

Working end to end for the lattices it names; the decode side of the stack is
`lattice-engine`'s.

- **Implemented:** exact fixed-width integer arithmetic; Hermite and Smith
  normal forms; the `Z_q` ring; `Gram`/`Basis` representation; the named
  lattices `Z^n`/`A_n`/`D_n`/`E_8`/`BW_16`/`Λ_24` with public generator
  numerators; exact short-vector enumeration; exact low-dimensional
  Voronoi-relevant vectors; nested pairs with coset enumeration;
  Construction A/D generator constructions; fraction-free Gram–Schmidt; LLL
  and deep-insertion LLL with an exact rational `δ`; Lagrange–Gauss; and a
  runtime-dispatched real-vector batch kernel.

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

Kernel crossovers and the reproducible fplll and FLINT comparisons are
recorded in [`BENCHMARKS.md`](BENCHMARKS.md).

`lattica` is **not** `no_std`: real-basis reduction needs `sqrt` and friends,
and a `libm` dependency would cost more than `std` does.

## License

MIT.

> [!WARNING]
> This library was made with the help of AI. Audit the code yourself, or with
> your own agent before using.

> [!WARNING]
> `lattica` makes no constant-time guarantee. Handling of secrets in 
> lattice-based cryptography requires a separate audit and is not supported by 
> `lattica` natively.

# lattica

`lattica` provides arithmetic for point lattices in `Z^n` and `R^n`. It supports
exact integer linear algebra, basis reduction, nested cosets, structural
enumeration, and real-vector transforms underneath lattice-based coding.

| Main property | What it provides |
| --- | --- |
| Exact arithmetic | Fixed-width integers with checked overflow, Hermite and Smith normal forms, determinants, and unimodular transforms. |
| Gram-based representation | Integer coordinates and exact metric queries, including lattices without an integral ambient basis. |
| Basis reduction | Fraction-free Gram–Schmidt, Lagrange–Gauss, LLL, and deep-insertion LLL with an exact rational parameter. |
| Lattice constructions | Named lattices, nested pairs, and Construction A/D over caller-supplied generators. |
| Structural enumeration | Short vectors, shell counts, minimal norms, and low-dimensional Voronoi-relevant vectors. |
| Real-vector batches | Runtime-dispatched transforms with bit-identical portable results and allocation-free execution. |

`lattica` computes facts about a lattice; it does not select the nearest point
to a real target. Quantization, target-centered decoding, code and graph
generation, and lattice cryptography are outside its scope. It is not a codec
or a finite-field library.

## Installation

The minimum supported Rust version is 1.89, edition 2024. Add as a cargo
dependency:

```toml
[dependencies]
lattica = { version = "1.0.0" }
```

Set `default-features = false` to use portable kernels without the SIMD
dependencies. The crate requires `std` in every feature configuration.

## Quick start

Construct the integral Gram matrix of E8 and recover its minimal vectors by
exact enumeration:

```rust
use lattica::named::e8;
use lattica::shortvec::{DEFAULT_NODE_BUDGET, census};

let gram = e8::<i64>().unwrap();
let result = census(&gram, DEFAULT_NODE_BUDGET).unwrap();

assert_eq!(gram.det().unwrap(), 1);
assert_eq!(result.min_norm_sq, Some(2));
assert_eq!(result.kissing_number, 240);
```

The determinant, minimal norm, and kissing number are computed from the
construction rather than stored answers. Enumeration takes an explicit node
budget and reports exhaustion instead of treating a partial search as complete.

## Exact lattice operations

| Surface | Contract |
| --- | --- |
| `Int`, `int::IntMatrix` | Checked integer arithmetic and matrix operations, including HNF, SNF, and determinant. |
| `Gram`, `Basis` | Gram matrices and integral generator bases, with rank and metric queries. |
| `gso`, `reduce` | Fraction-free orthogonalization and exact basis reduction. |
| `named` | `Z^n`, `A_n`, `D_n`, `E_8`, `BW_16`, and `Λ_24`, including published generator numerators. |
| `Nested` | Inclusion checks, quotient index, and coset representatives for nested lattice pairs. |
| `construct` | Construction A/D generators from supplied code-generator matrices. |
| `Zq` | Modular reduction, centered representatives, and lifting residues to integers. |
| `shortvec`, `relevant` | Exact structural enumeration with explicit work budgets. |

A lattice vector is an integer coordinate vector. Its squared norm is
`c G cᵀ`, where `G` is the Gram matrix; ambient coordinates need not be
integral. Reduction uses an exact rational `Delta`, not a floating-point
approximation. A reduced basis is not unique: its unimodular transform and
reduced-basis predicate provide the certificate.

Integer operations return a range error when the chosen width cannot hold an
intermediate result. There is no wrapping, arbitrary-precision fallback, or
floating-point approximation of an integral answer. Shape and range errors
leave mutable outputs unchanged.

## Real-vector transforms

`kernel::transform` applies one dense transform. `transform_batch` accepts
strided vectors; `transform_batch_soa` accepts coordinate planes. All write
caller-owned output buffers and validate geometry before writing.

```rust
use lattica::kernel::transform;

// Each input coordinate contributes one contiguous row of output coefficients.
let matrix = [1.0, 2.0, 3.0, 4.0];
let input = [5.0, 6.0];
let mut output = [0.0; 2];

transform(&matrix, 2, 2, &input, &mut output).unwrap();
assert_eq!(output, [23.0, 34.0]);
```

The dispatched kernels accumulate input rows in scalar order with separate
multiplication and addition. No FMA or reassociation is used, so SIMD results
are bit-identical to the portable reference. Buffer layouts and scratch
requirements belong to each operation's API documentation.

## Features and backends

| Feature | Effect |
| --- | --- |
| default | enables `simd` |
| `simd` | enables `simdispatch` selection and Archmage-backed x86-64 AVX2 kernels |
| `internals` | re-export-only facade of portable references, profiling, workspace, and experimental APIs |

Nothing behind `internals` is a compatibility promise; those APIs may change
or disappear in any release. Disabling SIMD leaves the exact arithmetic and
portable real-vector surface available, but does not enable `no_std`.

[`simdispatch`](https://github.com/nanithefkuc/simdispatch) owns CPU detection
and the process-startup, downgrade-only `SIMD_BACKEND` override. AVX2 dispatch
is geometry-dependent; other targets and shapes use portable kernels. No
architecture-specific compiler flags are required.

The library forbids unsafe Rust. Safe memory access and exact arithmetic do
not imply constant-time execution; lattice cryptography is not a supported use.

## Performance

[BENCHMARKS.md](BENCHMARKS.md) records public-operation measurements, dispatch
geometry, and competitor comparisons, including their input contracts and
reproduction details.

## Building

From the repository, use `just` for build and verification commands:

```sh
just build       # release build with all features
just features    # no-default, default, and all-feature tests
just test
just test-tiers   # requested backend sweep
just doc
just validate    # complete gate, including coverage
```

## License

MIT — see [LICENSE](https://github.com/nanithefkuc/lattica/blob/main/LICENSE).

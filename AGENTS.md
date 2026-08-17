# AGENTS.md

Working rules for `lattica`: the operational summary of how the crate is built
and tested.

## What this crate is

Shared arithmetic for point lattices in `Z^n` and `R^n`, underneath
`latticode` and `gldlc`. **Given a lattice, do arithmetic on it fast and
exactly; never construct the combinatorial object that defines it.** Deciding
lattice points from real targets — quantization, enumeration, ML decoding,
`mod Λ` — is `lattice-engine`'s, one layer up; nothing here selects a point.

Not a codec. Not a field library — that is `fff`/`fgf`. Not a graph library —
that is `sgraph`. Not lattice cryptography, ever.

## Hard rules

1. **Only dispatch dependencies.** Runtime dependencies are limited to
   `simdispatch` for the stack-wide backend policy and `archmage` for safe
   capability tokens, both optional under `simd`. Adding `fgf`, `sgraph`,
   `butterfly-fft`, `gfm`, or `lattice-engine` inverts the stack's layering;
   CI fails the build if you try.
2. **No `unsafe`.** Forbidden at the crate root.
3. **Checked arithmetic only on the integer path.** `Int` deliberately exposes
   no `Add`/`Sub`/`Mul` operators — only `try_add`, `try_sub`, `try_mul`, and
   friends returning `Result<_, RangeError>`. A silent wrap in release mode
   would produce a wrong basis that passes every downstream shape check. Do not
   "simplify" this by adding operator impls.
4. **No floating point in an integer answer.** If the input is integral, the
   output is exact. `f64` appears only in the real-vector kernels and the
   published real generators.
5. **No FMA, no reassociation in the dispatched kernels.** SIMD lanes are
   independent vectors and every lane accumulates rows in scalar order, so the
   dispatched result is bit-identical to the scalar reference. Never add
   `mul_add` or a reassociated reduction to `kernel/`.
6. **Validate before mutating.** A rejected call leaves every output buffer and
   all internal state exactly as it was.
7. **No bignum.** Fixed-width with a checked magnitude budget. An out-of-range
   geometry is a loud `RangeError`, never a silent precision downgrade.

## Testing

A test whose expected value came from this crate's own output is not a test.
Every algorithm has an independent oracle, and that oracle is the authority on
what counts.

Preferred oracles, in order: an exactly checkable certificate (`U·B == H` with
`|det U| == 1`), a differential against brute force, a published constant, and
only then a statistical check with a seed and a derived tolerance.

Fixtures are format, not test data. Moving a file is fine; changing an
expected value is a format break that needs a versioned algorithm.

## Commands

```sh
cargo test --all-features
cargo test --no-default-features
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

## Implementation status

Implemented and gated by the release check: the `Int` contract, exact integer
linear algebra, the `Z_q` ring, `Gram`/`Basis`, named lattices through `BW_16`
and `Λ_24`, exact short-vector and Voronoi-relevant enumeration, nested pairs,
code-free Construction A/D generator constructions, fraction-free GSO, LLL,
and dispatched real-vector batch transforms.

Three rules specific to what is already here:

- **A lattice vector is an integer coordinate vector.** Metric questions go
  through the Gram matrix, never through ambient coordinates. `E_8` has no
  integral ambient basis and does not need one.
- **Named constructors never store a constant.** Determinants, minimal norms,
  and kissing numbers are computed. Hardcoding one makes the acceptance tests
  circular, which is worse than having no test.
- **Reduction runs on the Gram matrix and `δ` is an exact rational.** Do not
  introduce an `f64` δ or an `f64` Cholesky: the reduced-basis predicate must
  be the same one `reduce::is_reduced` checks, or the certificate means
  nothing.
- **There is no canonical LLL output.** Two correct implementations disagree on
  which reduced basis they return. Test the certificate, never a stored basis.

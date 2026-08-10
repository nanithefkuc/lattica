# AGENTS.md

Working rules for `lattica`: the operational summary of how the crate is built
and tested.

## What this crate is

Shared arithmetic for point lattices in `Z^n` and `R^n`, underneath `latticode`
and `gldlc`. **Given a lattice, do arithmetic on it fast and exactly; never
construct the combinatorial object that defines it.**

Not a codec. Not a field library — that is `fff`/`fgf`. Not a graph library —
that is `sgraph`. Not lattice cryptography, ever.

## Hard rules

1. **No dependencies.** `lattica` has none today and will only ever gain
   `archmage`. Adding `fff`, `fgf`, `sgraph`, `cafft`, or `gfm` inverts
   the stack's layering; CI fails the build if you try.
2. **No `unsafe`.** Forbidden at the crate root.
3. **Checked arithmetic only on the integer path.** `Int` deliberately exposes
   no `Add`/`Sub`/`Mul` operators — only `try_add`, `try_sub`, `try_mul`, and
   friends returning `Result<_, RangeError>`. A silent wrap in release mode
   would produce a wrong basis that passes every downstream shape check. Do not
   "simplify" this by adding operator impls.
4. **No floating point in an integer answer.** If the input is integral, the
   output is exact. `f64` appears only where the input is a genuinely real
   received vector.
5. **No FMA, no reassociation, no transcendentals on a decision path.** The
   closed-form decoders must compile to add/sub/compare/round so that two peers
   on different architectures agree at a Voronoi boundary. Never add `mul_add`
   to `quant/closed.rs`.
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

Fixtures under `tests/data/` are format, not test data. Moving a file is fine;
changing an expected value is a format break that needs a versioned algorithm.

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
linear algebra, the `Z_q` ring, `Gram`/`Basis`, the named lattices, exact
short-vector enumeration, the closed-form quantizers, `mod Λ`, nested pairs,
Construction A/D, the fraction-free GSO, LLL, and Babai. Not yet implemented:
Schnorr–Euchner enumeration and list decoding.

Two rules specific to what is already here:

- **A lattice vector is an integer coordinate vector.** Metric questions go
  through the Gram matrix, never through ambient coordinates. `E_8` has no
  integral ambient basis and does not need one.
- **Named constructors never store a constant.** Determinants, minimal norms,
  and kissing numbers are computed. Hardcoding one makes the acceptance tests
  circular, which is worse than having no test.
- **The decoders use only add, subtract, compare, and round.** That operation
  set is why two peers on different architectures agree at a Voronoi boundary.
  `f64::round` is deliberately *not* used — it is `std`-only, and the hand
  written `round_away` states the tie rule directly. Never introduce `mul_add`,
  a transcendental, or a reassociation into `quant/closed.rs`.
- **`tests/data/ties.txt` is format.** Changing an expected value there is a
  wire break requiring a versioned decoder, not an edited line.
- **Ties break translation and negation symmetry, and that is inherent.**
  `Q(-x) = -Q(x)` and `(x + λ) mod Λ = x mod Λ` hold away from Voronoi
  boundaries and cannot hold on them: the tie set is symmetric and any rule
  must pick one side. Assert on the *distance*, which is always invariant, not
  on the point. Do not "fix" this.
- **The nesting index is `sqrt(det Λ_s / det Λ_c)`.** `det` here is the Gram
  determinant, so the ratio is squared. Writing it unsquared is the standing
  mistake in this crate's history; a test exists solely to catch it.
- **`cargo run --release --example e8_awgn` is the release gate.** If the
  measured shaping gain leaves 0.6539 dB, something is broken.
- **Reduction runs on the Gram matrix and `δ` is an exact rational.** Do not
  introduce an `f64` δ or an `f64` Cholesky: the reduced-basis predicate must
  be the same one `reduce::is_reduced` checks, or the certificate means
  nothing.
- **There is no canonical LLL output.** Two correct implementations disagree on
  which reduced basis they return. Test the certificate, never a stored basis.

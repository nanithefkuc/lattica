# Benchmarks and decoder measurements

Performance thresholds and decoder behavior are recorded here rather than in
API documentation. Re-run the named harness before changing either policy.

## Real-transform dispatch

Command:

```sh
taskset -c 2 cargo bench --bench kernel --features internals
```

Measured 2026-08-12 on an Intel Core Ultra 7 258V, Linux x86-64, with
`rustc 1.93.0`. Criterion used 100 samples after its standard three-second
warm-up. Scalar and dispatched cases ran in the same pinned-core harness. The
shape is a 16-by-16 column-major transform over a structure-of-arrays batch.

| Vectors | Scalar median | AVX2 dispatched median | Ratio |
| ---: | ---: | ---: | ---: |
| 8 | 399.52 ns | 424.39 ns | 0.94x |
| 64 | 2.0753 µs | 1.7743 µs | 1.17x |
| 257 | 8.1951 µs | 6.2963 µs | 1.30x |

Decision: dispatch only the 16-output SoA shape at 64 vectors or more. Eight
vectors lose to token and call overhead. Array-of-structures batches and single
transforms also lost in measurement, so they remain scalar. The 24-output and
31-output experiments varied from a small win to a regression between pinned
runs; neither is dispatched. This conservative cutoff avoids turning a feature
flag into a performance regression.

The AVX2 kernel uses separate multiply and add instructions, not FMA. SIMD lanes
are independent received vectors and each lane accumulates rows in scalar order;
differential tests require bit-identical output across 1–65 vectors, odd row
counts, and scalar tails.

## Barnes–Wall and Leech decoding beyond packing radius

Command:

```sh
cargo run --release --example highdim_ml
```

Measured 2026-08-12 on the same machine and toolchain. For each radius, 2,000
deterministic directions were sampled uniformly from a normalized cube vector
on the ambient sphere. The transmitted point was zero. A word error means the
maximum-likelihood point was nonzero; budget exhaustion is reported separately.
The node budget was `2^24` per point.

| Lattice | Radius | Word errors | Budget exhausted |
| --- | ---: | ---: | ---: |
| `BW_16` | 0.95 | 0 / 2000 | 0 / 2000 |
| `BW_16` | 1.05 | 0 / 2000 | 0 / 2000 |
| `BW_16` | 1.25 | 738 / 2000 | 0 / 2000 |
| `BW_16` | 1.50 | 2000 / 2000 | 0 / 2000 |
| `Λ_24` | 0.95 | 0 / 2000 | 0 / 2000 |
| `Λ_24` | 1.05 | 0 / 2000 | 0 / 2000 |
| `Λ_24` | 1.25 | 385 / 2000 | 0 / 2000 |
| `Λ_24` | 1.50 | 2000 / 2000 | 0 / 2000 |

Both lattices have minimal squared norm 4, so radius 1 is the guaranteed unique
decoding radius. The implementation deliberately uses maximum-likelihood
Schnorr–Euchner search instead of a bounded-distance recursion or hexacode
shortcut: success is the globally nearest point, and an insufficient node
budget is an error. This avoids a second, lattice-specific failure region. The
out-of-radius table measures channel word errors, not hidden algorithm errors.

Ambient outputs retain exact algebraic scaling. `BW_16` returns numerators over
2; `Λ_24` returns numerators over `sqrt(8)`. Returning approximate `f64` lattice
points was rejected because it would turn a discrete answer into a rounding
question.

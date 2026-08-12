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

## Optimization corpus

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-12 on the same Intel Core Ultra 7 258V and `rustc 1.93.0`.
The harness separates LLL operations and certificate work; CVP preparation,
nodes, and nanoseconds per node; named-decoder setup and total latency; exact
algebra operations; and closed-form batch quantizers. Every row carries a
geometry name and deterministic correctness fingerprint.

The incremental exact LLL path changed the 16-basis comparison corpus from
`86.622 µs`, `3.3341 ms`, and `28.167 ms` per basis at dimensions 8, 16, and 24
to `7.561 µs`, `68.165 µs`, and `249.814 µs`. The speedups are `11.5x`, `48.9x`,
and `112.8x`; every measured reduction uses one factorization and one Gram copy.
An `f64` estimate layer was not added: after incremental exact updates, the
largest comparison case is `112.8x` faster while retaining a single exact
factorization. The remaining checked updates are not the dominant measured
cost, so approximate scheduling would add a second state without a supported
crossover.
Warm CVP on the same comparison corpus improved from `1.512 µs`, `16.540 µs`,
and `120.107 µs` to `0.635 µs`, `4.795 µs`, and `24.699 µs`, with unchanged
target, point, and distance fingerprints.

Strong reduced-basis preconditioning is the selected proof-tree optimization.
On the named target `[0.31; n]`, `BW_16` visits 18 nodes and `Λ_24` visits
19,202 nodes; the full 2,000-word radius sweep above reports no budget
exhaustion. Stronger floating lower bounds and deterministic multi-start Babai
candidates were therefore not added: neither has a measured exhaustion case to
solve, and both would add per-node or per-word work to the default path.
Single-word subtree scheduling remains deferred; independent received words are
the deterministic parallel boundary.

The specialized Barnes–Wall recursion and Leech hexacode candidate engines were
not selected after this measurement. Preconditioned exhaustive search already
meets the ML sweep without exhaustion, so a second candidate implementation
would add tables, scratch, and a new membership proof without a failing workload
to recover. Decoder construction remains setup work: measured medians are
`69.6 µs` for `BW_16` and `177.7 µs` for `Λ_24`. Precomputed dual tables were
also rejected because they would replace independently checked exact
construction for an unmeasured cold-path saving.

The exact-algebra corpus identified structural cases worth selecting. For a
24-dimensional unit lower-bidiagonal matrix, triangular determinant selection
measures `0.39 µs`; one fraction-free adjugate solve measures `0.235 ms`; and
the one-factorization positive-definiteness check measures `0.030 ms`.

A five-run `perf stat` of the complete pinned corpus reported medians of
`951,444,719` P-core cycles, `5,515,777,396` P-core instructions,
`490,275,530` P-core branches, `2,776,408` P-core branch misses, and `7,052`
P-core cache misses. Elapsed task-clock dispersion was `0.93%`. Generated
release assembly was inspected after `cargo rustc --release --features
internals --lib -- --emit=asm`; the hot enumeration loop remains scalar and
contains no square-root call after child construction was rewritten.

## fplll comparison

[`fplll`](https://github.com/fplll/fplll) overlaps with this crate at LLL
reduction and general CVP enumeration. It does not provide a comparable API for
the named-lattice decoders or the real-vector batch transform, so those are not
forced into this comparison.

The harness pins fplll 5.5.0 at commit
`a8dedce384689047daba154bd50d6215e35bf03b`, the latest stable fplll release
available when measured. fplll was built statically with GCC 16.1.1,
`-O3 -march=native -DNDEBUG`, GMP 6.3.0, and MPFR 4.2.2. `lattica` used the
Cargo release profile. Both executables took the median of 11 in-process
samples on CPU 2. External process startup is excluded.

From the crate root, the complete setup and measurement are:

```sh
git clone --depth 1 --branch 5.5.0 \
  https://github.com/fplll/fplll.git target/fplll-5.5.0
cd target/fplll-5.5.0
./autogen.sh
./configure --disable-shared CXXFLAGS="-O3 -march=native -DNDEBUG"
make -j
cd ../..

c++ -O3 -march=native -DNDEBUG -std=c++17 \
  -Itarget/fplll-5.5.0 benches/fplll_compare.cpp \
  target/fplll-5.5.0/fplll/.libs/libfplll.a \
  -lmpfr -lgmp -lpthread -o target/fplll-compare

taskset -c 2 target/fplll-compare
taskset -c 2 cargo bench --bench fplll_compare
```

The fplll checkout and linked executable stay under ignored `target/`; fplll is
not a dependency of `lattica`.

### LLL

Each dimension uses the same 16 deterministic integral bases. They begin as a
small banded determinant basis and receive `2n` bounded unimodular row shears.
Both libraries use `δ = 0.99`; `lattica` uses exact `i128` Gram arithmetic and
fplll uses its `mpz_t` `LM_WRAPPER` path. The matching input fingerprints in the
harness output guard the duplicated Rust and C++ generators.

This measures each library's public reduction boundary, not identical internal
work. `lattica` starts with its canonical Gram representation. fplll starts with
its canonical ambient basis representation and copies it because its API
reduces in place. Input construction is outside both timers.

| Dimension | `lattica` median | fplll median | fplll speedup |
| ---: | ---: | ---: | ---: |
| 8 | 86.622 µs | 14.070 µs | 6.16x |
| 16 | 3.3341 ms | 46.553 µs | 71.6x |
| 24 | 28.167 ms | 162.673 µs | 173x |

This is the expected loss from `lattica`'s current reduction design:
fraction-free GSO is recomputed after every basis change, while fplll is a
mature, update-oriented reduction library. Reduction remains an offline setup
operation here, but this is the clearest measured optimization gap.

### CVP

The CVP corpus uses one exactly `δ = 0.99` LLL-reduced upper-bidiagonal basis per
dimension and 128 deterministic targets. The diagonal is 2 and the
superdiagonal is 1. Targets have denominator 1009; fplll receives the equivalent
integer problem with both basis and target scaled by 1009. This avoids giving
either implementation a rounding-boundary corpus while respecting fplll's
integer-target API.

`lattica cold` constructs an `Enumerator` and fresh scratch for every query.
`lattica warm` reuses both. fplll's public `closest_vector` API rebuilds its
floating GSO and working vectors on every query. fplll's `CVPM_FAST` mode is not
a guaranteed solver; it is included as a throughput ceiling because its result
fingerprints match `lattica` on this corpus. The documented guaranteed
`CVPM_PROVED` mode is reported separately.

| Dimension | `lattica` cold | `lattica` warm | fplll `FAST` | fplll `PROVED` |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2.498 µs | 1.512 µs | 21.065 µs | 49.516 µs |
| 16 | 25.184 µs | 16.540 µs | 57.760 µs | 102.130 µs |
| 24 | 149.392 µs | 120.107 µs | 128.278 µs | 202.037 µs |

Against fplll `FAST`, warm `lattica` is 13.9x faster at dimension 8, 3.49x at
16, and 1.07x at 24. Cold `lattica` is 8.43x and 2.29x faster at 8 and 16;
fplll is 1.16x faster at 24. These are throughput comparisons, not equivalent
guarantees: `lattica` retains exact pruning plus an explicit node budget, while
fplll `FAST` does not promise the closest point.

The output fingerprints expose a correctness problem in fplll 5.5.0
`CVPM_PROVED`. `FAST` and `lattica` reported identical target, point, and
distance fingerprints. `PROVED` reported different point and larger distance
fingerprints:

| Dimension | Target | `lattica` / `FAST` point | `PROVED` point | `lattica` / `FAST` distance | `PROVED` distance |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | -356691156 | -364109 | -363165 | 20641984069 | 20757042357 |
| 16 | 13549574 | 15984 | 12831 | 42334915811 | 42670101575 |
| 24 | -217921229 | -153113 | -140791 | 61652689299 | 61930640547 |

These fingerprints are weighted integer checksums, not distances or benchmark
scores. The harness prints them so a timing run cannot silently compare
different input or output sequences.

One independently checkable miss is frozen in
`benches/data/fplll_proved_miss.txt`:

```sh
target/fplll-5.5.0/fplll/fplll -a cvp \
  < benches/data/fplll_proved_miss.txt
```

fplll returns
`[-26234, -1009, 14126, -2018, 18162, 11099, -13117, -2018]`, at squared
distance 3602525. The lattice point
`[-26234, -1009, 14126, -2018, 18162, 12108, -14126, -4036]` has squared
distance 3053629 to the same target. A lower-distance lattice point is enough
to disprove the claimed closest result; no assumption about `lattica` is needed
for that check. Consequently, the `PROVED` timings above are diagnostic and are
not used for a speedup claim.

Both failures also reproduce on fplll master commit
`1987472ec5ca19107f93c3891c53db3363c8a78d`, checked on 2026-08-12. Timings
remain pinned to the stable release; this master check only establishes that
the findings were not already fixed after 5.5.0.

### fplll Babai cycle

The first attempted corpus also exposed a non-terminating fplll CVP prepass.
The minimal input is retained in `benches/data/fplll_babai_cycle.txt`:

```sh
timeout 1 target/fplll-5.5.0/fplll/fplll -a cvp \
  < benches/data/fplll_babai_cycle.txt
```

The process emits `warning: possible infinite loop in Babai's algorithm` and
does not terminate before the timeout. Instrumenting the pinned source showed
the residual alternating between
`[1, 0, 1, -1, 1, -1, 1, 0]` and
`[-1, 1, -1, 0, -1, 0, -1, 0]`; the rounded Babai coefficients alternate with
opposite signs, so neither iteration satisfies the stop condition.

The relevant fplll loop
[only warns at power-of-two iteration counts and has no hard limit](https://github.com/fplll/fplll/blob/5.5.0/fplll/svpcvp.cpp#L571-L595).
The comparison corpus uses denominator-1009 targets in general position so the
timing harness terminates, but the reproducer remains part of the benchmark
record rather than being hidden by that dataset choice.

# Benchmarks

Performance thresholds are recorded here rather than in API documentation.
Re-run the named harness before changing either policy. Decoder measurements
moved to `lattice-engine`'s `BENCHMARKS.md` with the decoders.

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

## Optimization corpus

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-12 on the same Intel Core Ultra 7 258V and `rustc 1.93.0`,
before the decoder extraction; the corpus then also included the decode
groups that now live in `lattice-engine`. The harness separates LLL
operations and certificate work from exact algebra operations. Every row
carries a geometry name and deterministic correctness fingerprint.

The incremental exact LLL path changed the 16-basis comparison corpus from
`86.622 µs`, `3.3341 ms`, and `28.167 ms` per basis at dimensions 8, 16, and 24
to `7.561 µs`, `68.165 µs`, and `249.814 µs`. The speedups are `11.5x`, `48.9x`,
and `112.8x`; every measured reduction uses one factorization and one Gram copy.
An `f64` estimate layer was not added: after incremental exact updates, the
largest comparison case is `112.8x` faster while retaining a single exact
factorization. The remaining checked updates are not the dominant measured
cost, so approximate scheduling would add a second state without a supported
crossover.
The exact-algebra corpus identified structural cases worth selecting. For a
24-dimensional unit lower-bidiagonal matrix, triangular determinant selection
measures `0.39 µs`; one fraction-free adjugate solve measures `0.235 ms`; and
the one-factorization positive-definiteness check measures `0.030 ms`.
Canonical HNF and SNF were left on their existing exact paths. At dimension 24
the measured medians are `2.32 µs` for classical HNF, `3.94 µs` for
determinant-modular HNF, and `3.56 µs` for invariant factors; no repeated
consumer workload selects among them. The same evidence rejects a generic
matrix-storage rewrite: half-matrix Gram construction removes the proved
duplicate work without adding a second dense layout or column cache.

A five-run `perf stat` of the complete pinned corpus reported medians of
`951,444,719` P-core cycles, `5,515,777,396` P-core instructions,
`490,275,530` P-core branches, `2,776,408` P-core branch misses, and `7,052`
P-core cache misses. Elapsed task-clock dispersion was `0.93%`. Generated
release assembly was inspected after `cargo rustc --release --features
internals --lib -- --emit=asm`; the hot enumeration loop remains scalar and
contains no square-root call after child construction was rewritten.

The final SIMD decision remains one 16-output SoA transform at 64 vectors or
more. A fresh pinned run measured scalar/dispatched medians of `525/528 ns`,
`1.724/1.461 µs`, and `8.740/6.716 µs` at 8, 64, and 257 vectors. The 24- and
31-output prototypes remain unselected because repeated runs did not establish
a stable crossover. Closed-form quantizer batches, metric kernels, and checked
integer algebra also remain scalar: the corpus identifies no layout-preserving
consumer or repeatable gain that would justify a second semantic
implementation. Integer SIMD is specifically incompatible with the existing
per-operation overflow boundary.

Repeated local-workload optimization also stopped at the evidence boundary.
List and relevant-vector calls allocate output-proportionally, exact census
allocates cold factorization state, and Construction-A membership currently
allocates one residue buffer per call. No crate consumer repeats these setup or
oracle operations in a hot path, so prepared public APIs and extra scratch
types were not added speculatively. The selected scalar improvements are the
measured reduction/enumeration changes, total unstable list ordering, shared
exact algebra, and triangular Gram work above.

Consumer build options were measured separately with the comparison harness.
`-C target-cpu=native` produced LLL medians of `6.792/69.315/247.868 µs` and
warm CVP medians of `0.631/4.880/26.036 µs` at dimensions 8/16/24. Fat LTO with
one codegen unit produced `6.687/70.948/247.537 µs` and
`0.582/4.658/25.085 µs`. Neither option improves every core geometry relative
to the ordinary release results, so these remain final-binary choices rather
than library profile settings.

## Further exact-arithmetic pass

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
taskset -c 2 cargo bench --bench fplll_compare
```

Measured 2026-08-13 on the same machine and toolchain. A `perf record` run
identified initial fraction-free GSO construction and determinant certificates
as the largest remaining avoidable costs. Symmetric Bareiss factorization now
computes each trailing entry once and mirrors it; determinant detection scans
each off-diagonal pair once and skips identity Bareiss row updates.

| Operation | Dimension | Before | After | Reduction |
| --- | ---: | ---: | ---: | ---: |
| comparison LLL | 8 | 7.561 µs | 6.608 µs | 12.6% |
| comparison LLL | 16 | 68.165 µs | 64.895 µs | 4.8% |
| comparison LLL | 24 | 249.814 µs | 238.518 µs | 4.5% |
| CVP preparation | 8 | 1.259 µs | 1.098 µs | 12.8% |
| CVP preparation | 16 | 9.873 µs | 6.558 µs | 33.6% |
| CVP preparation | 24 | 31.329 µs | 21.134 µs | 32.6% |
| positive-definiteness | 8 | 1.151 µs | 0.784 µs | 31.9% |
| positive-definiteness | 16 | 9.556 µs | 5.285 µs | 44.7% |
| positive-definiteness | 24 | 32.582 µs | 17.839 µs | 45.2% |
| triangular determinant | 8 | 67 ns | 59 ns | 11.9% |
| triangular determinant | 16 | 183 ns | 147 ns | 19.7% |
| triangular determinant | 24 | 427 ns | 279 ns | 34.7% |

All comparison fingerprints and reduction operation counts were unchanged.
The post-change profile is dominated by checked `i128` division, LLL updates,
and determinant certificates. Those are semantic work under the crate's exact
fixed-width contract; replacing them with approximate scheduling or unchecked
arithmetic would change the contract rather than optimize it.

## Lazy exact size reduction

Command:

```sh
taskset -c 2 cargo bench --bench fplll_compare
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 on the same machine. The comparison A/B used 101 internal
samples to reduce short-run dispersion; the table reports the baseline and the
median of three post-change runs. Ordinary LLL now reduces `b_k` against
`b_{k-1}` before the Lovász test and delays quotients against earlier vectors
until that test passes. Those earlier coefficients cannot affect the test, and
a failed test immediately swaps the vectors, so eager evaluation was exact work
whose result was about to be invalidated.

| Dimension | Before | After | Reduction |
| ---: | ---: | ---: | ---: |
| 8 | 6.400 µs | 5.497 µs | 14.1% |
| 16 | 68.863 µs | 57.216 µs | 16.9% |
| 24 | 245.678 µs | 207.225 µs | 15.7% |

Across the nine shear geometries in the optimization corpus, reduction time
fell by 8.3% to 14.5%. Factorization, nonzero size-reduction, swap, iteration,
Gram-copy, and checked-update counts were unchanged for every geometry. The
independent randomized certificate suite continued to prove size reduction,
the Lovász condition, unimodularity, and `U G U^T == G_reduced`; deep-insertion
LLL retains its eager pass because its insertion predicate consumes every
projected coefficient.


## Exact zero-quotient filter

Command:

```sh
taskset -c 2 cargo bench --bench fplll_compare
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 immediately after the lazy-reduction pass. The comparison
A/B again used 101 internal samples; the table reports one post-lazy baseline
and the median of three post-change runs. Before calling checked nearest
division, LLL now proves `|λ[k][j]| <= d[j+1] / 2` with the overflow-safe
comparison `magnitude <= denominator - magnitude`. That condition is exactly
the zero-quotient condition, including ties. Values whose magnitude cannot be
represented, such as the signed minimum, retain the full division path.

| Dimension | Before | After | Reduction |
| ---: | ---: | ---: | ---: |
| 8 | 5.536 µs | 4.856 µs | 12.3% |
| 16 | 58.354 µs | 50.970 µs | 12.7% |
| 24 | 208.929 µs | 182.916 µs | 12.5% |

The nine shear geometries improved by 2.7% to 12.5%; geometries with more
nonzero reductions gain less because their divisions remain necessary.
Factorization, nonzero size-reduction, swap, iteration, Gram-copy, and
checked-update counts were unchanged. The randomized exact-certificate suite
continued to pass.

## Reduction observability baseline

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
taskset -c 2 cargo bench --bench fplll_compare
```

Measured 2026-08-24 after the exact zero-quotient filter. The timed
`optimization` path now calls ordinary `lll`; one untimed profiled reduction
per geometry supplies unstable operation counters. Counter collection therefore
does not distort the public-path latency that future changes compare.

| Dimension | Shear bits | LLL | Checks | Zero proofs | Divisions | Reductions | Swaps | Swap terms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 3.690 µs | 63 | 45 | 18 | 18 | 9 | 56 |
| 8 | 4 | 4.577 µs | 109 | 84 | 25 | 25 | 19 | 104 |
| 8 | 6 | 4.789 µs | 116 | 82 | 34 | 34 | 20 | 104 |
| 16 | 2 | 17.897 µs | 541 | 513 | 28 | 28 | 58 | 898 |
| 16 | 4 | 20.953 µs | 587 | 525 | 62 | 62 | 63 | 956 |
| 16 | 6 | 21.785 µs | 664 | 593 | 71 | 71 | 68 | 952 |
| 24 | 2 | 46.103 µs | 846 | 762 | 84 | 84 | 48 | 1,068 |
| 24 | 4 | 50.049 µs | 854 | 795 | 59 | 59 | 48 | 1,052 |
| 24 | 6 | 51.655 µs | 860 | 797 | 63 | 63 | 49 | 1,086 |

For every geometry, `checks == zero proofs + divisions`, and every entered
division produces one nonzero size reduction. The exact filter avoids 70.7% to
77.1% of quotient divisions at dimension 8 and 89.3% to 94.8% at dimensions 16
and 24. Existing factorization, reduction, swap, iteration, Gram-copy, and
checked-update counts remain unchanged.

The median of three standard `fplll_compare` runs was `5.144/49.854/177.207 µs`
at dimensions 8/16/24. This is the ordinary-LLL baseline for the contiguous
transactional state-update pass.

## Contiguous transactional state updates

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
taskset -c 2 cargo bench --bench fplll_compare
```

Measured 2026-08-24 against the committed observability baseline. Reduction now
borrows validated contiguous source/target rows, preflights the Gram and
transform updates into reusable scratch, commits complete target rows, and
mirrors the Gram column through one validated helper. The checked expressions
and diagonal update order are unchanged.

| Dimension | Shear bits | Before | After | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 3.690 µs | 3.288 µs | 10.9% |
| 8 | 4 | 4.577 µs | 4.416 µs | 3.5% |
| 8 | 6 | 4.789 µs | 4.378 µs | 8.6% |
| 16 | 2 | 17.897 µs | 16.744 µs | 6.4% |
| 16 | 4 | 20.953 µs | 19.261 µs | 8.1% |
| 16 | 6 | 21.785 µs | 20.039 µs | 8.0% |
| 24 | 2 | 46.103 µs | 44.596 µs | 3.3% |
| 24 | 4 | 50.049 µs | 47.829 µs | 4.4% |
| 24 | 6 | 51.655 µs | 49.145 µs | 4.9% |

All fingerprints and factorization, quotient, reduction, swap, iteration,
Gram-copy, swap-term, and checked-update counts are unchanged. Width-generic
tests compare the state update against an independently formed congruence at
`i32`, `i64`, and `i128`; overflowing updates leave both persistent matrices
unchanged.

The median of three standard comparison runs changed from
`5.144/49.854/177.207 µs` to `4.888/47.706/169.351 µs`, reductions of
5.0%, 4.3%, and 4.4%. In the post-change profile, `size_reduce_pair` falls from
38.0% to 32.4% of sampled cycles; `Gso::swap_adjacent` is the next isolated
update target. Generated benchmark assembly keeps remaining bounds failures in
cold panic blocks rather than the contiguous arithmetic loops.

## Comparison target selection

fplll remains the useful general-CVP target: its in-process public API exposes
both fast and claimed-proved closest-vector modes, and no conversion or process
startup appears in the timing. Its LLL boundary is less direct because fplll
accepts an ambient basis while `lattica` owns a Gram matrix.

[FLINT](https://flintlib.org/doc/fmpz_lll.html) is the more appropriate LLL
target. Its public `fmpz_lll` API accepts a Gram matrix and returns the
unimodular transform, matching `lattica::reduce::lll` at the representation and
output boundaries. `benches/flint_compare.cpp` duplicates the existing
deterministic corpus, checks `U G U^T == G_reduced`, and requires FLINT's
reducedness certificate before timing. FLINT remains an external benchmark,
not a crate dependency.

The complete local FLINT 3.6.0 measurement is:

```sh
c++ -O3 -march=native -DNDEBUG -std=c++17 \
  benches/flint_compare.cpp -lflint -lmpfr -lgmp \
  -o target/flint-compare
taskset -c 2 target/flint-compare
```

Both libraries include their public result allocation, input copy, and
unimodular transform. FLINT uses arbitrary-precision integers and its adaptive
public LLL driver; `lattica` uses checked `i128`. This is a contract-aligned
comparison, not an identical-arithmetic claim.

| Dimension | `lattica` median | FLINT median | `lattica` speedup |
| ---: | ---: | ---: | ---: |
| 8 | 6.608 µs | 78.726 µs | 11.9x |
| 16 | 64.895 µs | 378.025 µs | 5.83x |
| 24 | 238.518 µs | 1.128 ms | 4.73x |

FLINT is also the right future target for determinant, adjugate/inverse, HNF,
and SNF throughput because those operations have direct `fmpz_mat` APIs. They
should use a separate magnitude-stratified corpus: the present unit
lower-bidiagonal matrix mostly measures structural fast paths.

## fplll comparison

[`fplll`](https://github.com/fplll/fplll) overlaps with this crate at LLL
reduction. The general-CVP overlap moved to `lattice-engine` with the
enumeration decoders and is compared there.

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

| Dimension | `lattica` median | fplll median | Faster library |
| ---: | ---: | ---: | ---: |
| 8 | 4.888 µs | 20.829 µs | `lattica` 4.26x |
| 16 | 47.706 µs | 46.171 µs | fplll 1.03x |
| 24 | 169.351 µs | 164.156 µs | fplll 1.03x |

Incremental exact GSO updates, lazy reduction, the exact zero-quotient filter,
and contiguous transactional state updates removed the earlier
orders-of-magnitude gap and narrowed the remaining comparison. fplll remains
slightly faster at dimensions 16 and 24, while `lattica` is faster at
dimension 8. The comparison is deliberately not called
contract-equivalent: fplll reduces an ambient basis without returning the
transform, while `lattica` reduces a Gram matrix and always returns it.

The general-CVP comparison, its fplll `CVPM_PROVED` miss and Babai-cycle
reproducers, and their data files moved to `lattice-engine` with the CVP
harness. The setup above still builds the fplll library the LLL comparison
uses.

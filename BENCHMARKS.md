# Benchmarks

Performance thresholds are recorded here rather than in API documentation.
Re-run the named harness before changing either policy. Decoder measurements
moved to `lattice-engine`'s `BENCHMARKS.md` with the decoders.

## Real-transform dispatch

Command:

```sh
taskset -c 2 cargo bench --bench kernel --features internals
```

Latest measurement 2026-08-25 on an Intel Core Ultra 7 258V, Linux x86-64,
with `rustc 1.93.0`. Criterion default settings (100 samples after a
three-second warm-up), core-pinned, three full rounds of one binary.

Dispatched shapes on x86 v3 hardware:

1. sixteen outputs, any row count, at sixty-four vectors or more — unchanged
   from the original decision below;
2. the exact twenty-four-by-twenty-four geometry at every batch size.

### Twenty-four outputs

The twenty-four-output experiments recorded below as unselected used the
generic column-at-a-time kernel; re-measuring it reproduces their instability.
Across the three rounds its cells move up to 29% between runs and it loses to
the scalar kernel below roughly thirty-two vectors. The registered kernel is a
different design: output columns advance in twelve-column blocks so every
loaded input chunk feeds twelve output planes, and all twelve accumulators live
in ymm registers across the full row loop — one input load and twelve
broadcast/multiply/add triples per chunk-row instead of the generic path's
per-column accumulator reload. There is still no FMA, and every lane
accumulates rows in ascending order, so the result stays bit-identical to the
portable reference; batches smaller than four vectors take the kernel's scalar
tail, which is also bit-identical.

Medians of the three rounds for the shipped gate (`dispatched` goes through
`transform_batch_soa`; `avx2_generic` is the old candidate called directly):

| Vectors | Scalar (ns) | Dispatched (ns) | Speedup | Generic (ns) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 773 / 783 / 783 | 206 / 203 / 208 | 3.8x | 869 / 869 / 869 |
| 4 | 869 / 901 / 869 | 101 / 104 / 98 | 8.6x | 971 / 1,157 / 972 |
| 8 | 1,141 / 1,120 / 1,124 | 192 / 194 / 193 | 5.9x | 1,286 / 1,500 / 1,311 |
| 16 | 1,503 / 1,529 / 1,496 | 371 / 400 / 367 | 3.9x | 2,083 / 2,156 / 1,673 |
| 32 | 2,554 / 2,587 / 2,543 | 755 / 796 / 749 | 3.3x | 2,420 / 2,907 / 3,025 |
| 64 | 4,443 / 4,371 / 4,313 | 1,561 / 1,563 / 1,481 | 2.8x | 3,918 / 4,350 / 4,342 |
| 128 | 8,246 / 8,254 / 8,038 | 3,096 / 3,099 / 3,055 | 2.6x | 6,883 / 8,166 / 8,226 |
| 257 | 24,128 / 23,036 / 23,285 | 6,883 / 6,683 / 6,698 | 3.5x | 18,411 / 18,031 / 18,473 |

The dispatched win never drops below 2.6x in any round at any count, far
outside this host's dispersion (scalar cells stay within about 4% across
rounds except where harness relayout shifted them together with the portable
reference). That makes the crossover unconditional within the shape: the gate
checks only `rows == cols == 24` and the selected backend tier. Block-size
selection inside the design: at 257 vectors block 12 measures 6.70–6.94 us
against block 8's 7.48–7.68 and block 6's 8.07–8.27, and block 12 ties or
beats both in every round; an earlier exploratory binary ordered blocks 6 and
8 oppositely after unrelated code-layout changes, so the choice rests on
block 12 never measuring worse rather than on a fixed runner-up order. Block 4
was dominated everywhere and was dropped before the formal rounds.

Stage separation answers the layout question for array-of-structures consumers.
Converting an AoS batch to SoA planes costs roughly one-tenth of the transform
it replaces (about 90 ns at eight vectors and 0.46–0.47 us at sixty-four,
against scalar transforms of 0.67–0.69 and 4.3–4.4 us there). End to end,
transpose plus dispatched kernel beats the pure scalar array-of-structures
transform at every count: 0.26–0.28 us against 0.67–0.69 us at eight vectors
and 9.3–10.0 us against 21.0–21.6 us at 257, a 2.1x to 2.6x win.
Consumers holding SoA batches see the full 2.6x-or-better column above;
consumers holding AoS batches still roughly double end-to-end throughput by
converting once and staying in SoA.

Generated release assembly was inspected: the hot loop holds twelve ymm
accumulators with one `vmovupd`, twelve `vbroadcastsd`, twelve `vmulpd`, and
twelve `vaddpd` per row — no FMA, no spills, no division, and slice bounds
checks hoisted outside the loops. Differential tests pin bit-identical output
against the portable kernel across lane boundaries, ragged tails (one to
seventeen, thirty-one, sixty-three to sixty-five, one hundred twenty-seven to
one hundred twenty-nine, and two hundred fifty-seven vectors), non-square
row-count fallbacks at twenty-three/twenty-four/twenty-five rows, and backend
overrides (`SIMD_BACKEND=scalar` and `SIMD_BACKEND=v3`).

This supersedes the twenty-four-output portion of the rejection below; the
sixteen-output decision and its threshold stand.

### Sixteen outputs (original decision)

Measured 2026-08-12 on the same host. The shape is a 16-by-16 column-major
transform over a structure-of-arrays batch.

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

The AVX2 kernels use separate multiply and add instructions, not FMA. SIMD
lanes are independent received vectors and each lane accumulates rows in
scalar order; differential tests require bit-identical output across 1–65
vectors, odd row counts, and scalar tails.

AArch64 NEON and wider x86 tiers remain unmeasured: no capable hardware or
canonical `simdispatch`/`archmage` support is available on this host.

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

## Contiguous exact adjacent swaps

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
taskset -c 2 cargo bench --bench fplll_compare
```

Measured 2026-08-24 against the committed contiguous-state baseline. The exact
adjacent-swap recurrence now hoists its three minors and swapped lambda, reads
the two affected fraction-free rows through validated contiguous slices,
preflights both outputs into split update storage, and commits the trailing rows
with slice copies. No sparse branch, common-factor cancellation, or unchecked
division was selected.

| Dimension | Shear bits | Before | After | Reduction |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 3.288 µs | 3.340 µs | -1.6% |
| 8 | 4 | 4.416 µs | 3.975 µs | 10.0% |
| 8 | 6 | 4.378 µs | 4.104 µs | 6.3% |
| 16 | 2 | 16.744 µs | 15.413 µs | 7.9% |
| 16 | 4 | 19.261 µs | 18.059 µs | 6.2% |
| 16 | 6 | 20.039 µs | 18.930 µs | 5.5% |
| 24 | 2 | 44.596 µs | 42.061 µs | 5.7% |
| 24 | 4 | 47.829 µs | 45.808 µs | 4.2% |
| 24 | 6 | 49.145 µs | 46.941 µs | 4.5% |

The dimension-8/shear-2 movement is within short-run noise; the other eight
geometries improve, including every dimension-16/24 case. All fingerprints and
factorization, quotient, reduction, swap, iteration, Gram-copy, swap-term, and
checked-update counts are unchanged. Differential tests compare every minor and
lambda with a fresh factorization after each forward and reverse swap at
`i32`, `i64`, and `i128`.

The median of three standard comparison runs changed from
`4.888/47.706/169.351 µs` to `4.753/45.528/163.560 µs`, reductions of
2.8%, 4.6%, and 3.4%. This puts exact Gram-plus-transform LLL at measurement
parity with fplll's ambient-basis boundary at dimensions 16 and 24. The
post-change profile is dominated by checked `i128` division and the two exact
update families; further ordinary-LLL recurrence changes need new evidence.

## Deep-insertion reduction baseline

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 on the same machine and toolchain. Each dimension has
three 16-basis insertion-heavy corpora. Their diagonal scales put all
length-two basis vectors before all length-one vectors; deterministic
upper-row shears then perturb one quarter, one half, or three quarters as many
rows as the dimension. This preserves a small exact determinant while forcing
deep movement. The fingerprints below cover every Gram entry, not just the
determinant.

The table reports the median of three core-pinned runs, each itself the median
of 11 in-process samples. Reduction and full certificate validation are timed
separately. Certificate validation independently checks size reduction and
Lovász, determinant preservation, unimodularity, and
`U G U^T == G_reduced`.

| Dimension | Geometry | LLL | Deep LLL | LLL certificate | Deep certificate |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 8 | light | 2.671 µs | 5.249 µs | 2.491 µs | 2.437 µs |
| 8 | medium | 2.982 µs | 5.818 µs | 2.644 µs | 2.690 µs |
| 8 | dense | 3.317 µs | 6.402 µs | 2.836 µs | 2.768 µs |
| 16 | light | 19.255 µs | 37.468 µs | 18.644 µs | 18.569 µs |
| 16 | medium | 21.017 µs | 40.697 µs | 19.390 µs | 19.203 µs |
| 16 | dense | 23.306 µs | 45.006 µs | 20.394 µs | 20.173 µs |
| 24 | light | 64.477 µs | 121.847 µs | 61.719 µs | 61.626 µs |
| 24 | medium | 69.530 µs | 131.469 µs | 65.197 µs | 64.773 µs |
| 24 | dense | 78.097 µs | 147.623 µs | 68.959 µs | 67.478 µs |

Unstable counters are collected in one untimed profiled pass and aggregated
over all 16 bases. Predicate terms count the exact suffix-sum terms formed by
the single reverse pass. Exact divisions count one denominator weight per
coefficient plus one additional division whenever the carried sum is rescaled.

| Dimension | Geometry | Insertions | Adjacent swaps | Predicate terms | Rescalings | Exact divisions | Checked updates | Fingerprint |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | light | 79 | 291 | 2,122 | 6 | 1,666 | 17,878 | 180,033 |
| 8 | medium | 94 | 326 | 2,274 | 39 | 1,809 | 19,670 | 216,530 |
| 8 | dense | 108 | 349 | 2,367 | 119 | 1,956 | 21,212 | 236,162 |
| 16 | light | 198 | 1,190 | 13,116 | 41 | 11,553 | 156,210 | 1,509,307 |
| 16 | medium | 233 | 1,297 | 13,944 | 243 | 12,451 | 166,428 | 1,547,581 |
| 16 | dense | 260 | 1,419 | 14,900 | 799 | 13,814 | 178,034 | 2,413,529 |
| 24 | light | 342 | 2,707 | 40,089 | 146 | 36,847 | 544,076 | 5,069,362 |
| 24 | medium | 391 | 2,894 | 42,598 | 1,134 | 40,103 | 567,878 | 6,320,229 |
| 24 | dense | 459 | 3,254 | 45,977 | 2,390 | 44,318 | 617,792 | 6,825,752 |

Deep LLL costs 1.87x to 1.98x ordinary LLL on these corpora. Certificate
latencies remain comparable because both paths return the same exact contract,
not because their non-canonical reduced Gram matrices are expected to match.
Increasing shear density raises both deep insertions and denominator
rescalings, selecting the suffix predicate arithmetic as the next isolated
measurement target.

## Deep suffix denominator reuse

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
taskset -c 2 cargo bench --bench fplll_compare
```

Measured 2026-08-24 with interleaved release binaries on CPU 2. For each
suffix term, the carried common scale already contains the incoming
denominator in 93.5% to 99.6% of cases. The selected recurrence takes that
exact quotient directly. When the denominator does not divide the scale, it
uses `g = gcd(scale, denominator)` and the identities
`L / scale = denominator / g` and `L / denominator = scale / g`. It forms
`L` as `(scale / g) * denominator`, preserving the former checked LCM
multiplication and overflow boundary.

The largest incoming denominator and common scale in each corpus are:

| Dimension | Geometry | Divisible suffixes | Maximum denominator | Maximum scale |
| ---: | :--- | ---: | ---: | ---: |
| 8 | light | 99.6% | 19 bits | 20 bits |
| 8 | medium | 97.8% | 21 bits | 25 bits |
| 8 | dense | 93.5% | 21 bits | 29 bits |
| 16 | light | 99.6% | 37 bits | 39 bits |
| 16 | medium | 98.0% | 39 bits | 50 bits |
| 16 | dense | 93.9% | 41 bits | 59 bits |
| 24 | light | 99.6% | 53 bits | 57 bits |
| 24 | medium | 97.1% | 58 bits | 81 bits |
| 24 | dense | 94.3% | 62 bits | 95 bits |

Three interleaved process runs, each using 11 internal samples, produced:

| Dimension | Geometry | Before | After | Reduction |
| ---: | :--- | ---: | ---: | ---: |
| 8 | light | 5.234 µs | 4.361 µs | 16.7% |
| 8 | medium | 6.121 µs | 4.904 µs | 19.9% |
| 8 | dense | 6.613 µs | 5.518 µs | 16.6% |
| 16 | light | 38.101 µs | 31.229 µs | 18.0% |
| 16 | medium | 41.929 µs | 34.595 µs | 17.5% |
| 16 | dense | 49.940 µs | 40.808 µs | 18.3% |
| 24 | light | 128.404 µs | 104.342 µs | 18.7% |
| 24 | medium | 140.852 µs | 119.840 µs | 14.9% |
| 24 | dense | 156.919 µs | 132.372 µs | 15.6% |

Corpus fingerprints, insertions, adjacent swaps, predicate terms, rescalings,
exact divisions, and checked updates are unchanged. A width-generic
differential compares every candidate position with the former LCM recurrence
at `i32`, `i64`, and `i128`; randomized full certificates independently check
the returned reductions. The signed-minimum coefficient still bypasses the
absolute-value filter and enters checked nearest division.

Seven interleaved insertion-corpus runs put ordinary LLL movement between
-1.9% and +1.6%, without a broad direction. Three comparison-corpus runs moved
from `4.847/45.004/160.253 µs` to `4.913/44.814/160.288 µs` at dimensions
8/16/24. Both are noise boundaries rather than an ordinary-path regression.

## Initial-factorization stratification

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 on the same machine and toolchain. The `factorization_ns`
cells time `Gso::new` alone over the existing deterministic corpora,
stratified by dimension, shear density, and entry width. Each cell narrows the
same generated bases into `i32`, `i64`, and `i128`, so fingerprints are
identical across the widths a geometry fits.

| Dimension | Geometry | `i32` | `i64` | `i128` |
| ---: | :--- | ---: | ---: | ---: |
| 8 | skew_s2 | 187 ns | 211 ns | 794 ns |
| 8 | skew_s4 | overflow | 209 ns | 712 ns |
| 8 | skew_s6 | overflow | overflow | 706 ns |
| 8 | insertion_light | 168 ns | 207 ns | 735 ns |
| 8 | insertion_medium | 174 ns | 195 ns | 722 ns |
| 8 | insertion_dense | 164 ns | 193 ns | 701 ns |
| 16 | all six geometries | overflow | 1315–1400 ns | 5678–5931 ns |
| 24 | all six geometries | overflow | 4188–4527 ns | 19496–19675 ns |

Three findings size the reusable-workspace question. First, one exact
factorization costs 11% to 16% of the corresponding total reduction at the
canonical width, so amortizing it across repeated calls is bounded by that
share even before allocation overhead. Second, checked `i128` factorization
costs about 4 times its `i64` counterpart on identical geometries; operand
width, not loop structure, dominates setup cost. Third, the `overflow` markers
are the accepted-domain boundary per width: the same corpus that fits `i128`
at every dimension already exceeds `i32` almost everywhere by dimension 16.
Any prepared workspace must preserve these exact overflow boundaries, since
widening them changes the contract rather than optimizing it.

## Prepared reduction workspace

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
taskset -c 2 cargo bench --bench fplll_compare
```

Measured 2026-08-24 against the factorization-stratification baseline.
`ReductionWorkspace` (unstable, behind `internals`) allocates the Gram copy,
transform, row scratch, and factorization buffers once and refactors them per
same-dimension call. The descent is shared with the one-shot path and inlined
at both call sites: an out-of-line shared loop measurably regressed the public
path and was rejected in favor of the inline hint.

A counting-allocator test pins steady-state reuse to exactly two allocations
per call — the returned Gram buffer and transform buffer — down from nine in
the one-shot shape. Outputs are pinned bit-identical to `lll`/`lll_deep` by
width-generic differential tests across ordinary and deep reduction,
including after rejected calls.

Median of three core-pinned rounds, each cell comparing the prepared path
against one-shot `lll` inside the same process:

| Dimension | Geometry | Ordinary | Deep |
| ---: | :--- | ---: | ---: |
| 8 | shear_2 / shear_4 / shear_6 | -3.2 / -12.8 / -5.5% | — |
| 8 | insertion_light / medium / dense | -2.3 / -3.3 / -3.4% | -5.0 / -2.4 / -3.2% |
| 16 | shear_2 / shear_4 / shear_6 | -2.7 / -2.1 / -1.7% | — |
| 16 | insertion_light / medium / dense | -0.4 / -1.8 / -2.2% | -1.6 / -1.5 / -1.3% |
| 24 | shear_2 / shear_4 / shear_6 | -0.2 / -0.3 / -1.0% | — |
| 24 | insertion_light / medium / dense | +0.2 / -0.4 / -0.5% | -0.1 / -0.5 / -0.0% |

The win tracks the fixed setup share: largest at dimension 8 where setup is a
larger fraction of total latency, tapering toward parity at dimension 24
where checked division dominates every path. The one-shot public boundary was
re-measured under alternated-order interleaving; its medians moved within this
host's run-to-run dispersion for identical binaries, which is a noise
boundary rather than a measured regression.

## Enumeration observability baseline

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 on the same machine and toolchain. The enumeration corpus
times the public one-shot path over named lattices at radii whose vector
counts have closed-form oracles, recovered by the enumeration itself before
anything is timed: `Z^n` at radius two holds exactly its `2n²` axis vectors
and pairwise sums, `A_n` at radius two is its `n(n+1)`-element root system,
`D_n` at radius four adds the weight-two and weight-four layers to its roots,
and E8 at radius two recovers its published kissing number of 240.

Unstable counters come from one untimed profiled pass per cell. Direct norms
are the `O(n²)` `c G cᵀ` recomputations at emitted vectors; tail terms are
the exact multiply-adds formed by the per-node suffix sums.

| Dimension | Geometry | Total | Nodes | Tail terms | Direct norms | Time |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 8 | zn_radius_2 | 128 | 417 | 504 | 128 | 15.3 us |
| 8 | a_n_radius_2 | 72 | 249 | 420 | 72 | 12.1 us |
| 8 | d_n_radius_4 | 1,248 | 2,783 | 6,704 | 1,248 | 196.0 us |
| 8 | e8_radius_2 | 240 | 751 | 2,170 | 240 | 53.4 us |
| 16 | zn_radius_2 | 512 | 3,009 | 4,720 | 512 | 125.6 us |
| 16 | a_n_radius_2 | 272 | 1,649 | 6,120 | 272 | 125.7 us |
| 16 | d_n_radius_4 | 29,632 | 107,969 | 683,276 | 29,632 | 15.9 ms |
| 24 | zn_radius_2 | 1,152 | 9,825 | 16,744 | 1,152 | 571.8 us |
| 24 | a_n_radius_2 | 600 | 5,225 | 29,900 | 600 | 537.0 us |
| 24 | d_n_radius_4 | 171,168 | 910,521 | 9,594,188 | 171,168 | 181.1 ms |

Allocations measure 9 to 13 per call across the corpus (factorization
buffers, widened Gram, coordinates); the counting allocator lives in the
harness because internal counters cannot observe allocation. Two costs
dominate and both are recomputation: the per-node tail dot product grows to
9.6 million terms at dimension 24, and the per-vector direct norm
recomputation performs roughly `n²` operations per emitted vector — about
98 million at `d_n_radius_4`, which is most of that cell's wall time. These
are the measurement targets for tail caching and carried norms.

## Amortized enumeration tails

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 against the observability baseline. Each node previously
recomputed its own suffix dot product on entry; now a parent forms, once per
value loop, the row suffix sum every child share, and each child receives its
tail as one rank-one update with its chosen coordinate.

Tail-term counts drop by 23% to 57% in every cell:

| Dimension | Geometry | Tail terms | Time |
| ---: | :--- | ---: | ---: |
| 8 | zn_radius_2 | 504 → 322 | -2.1% |
| 8 | a_n_radius_2 | 420 → 252 | +2.3% |
| 8 | d_n_radius_4 | 6,704 → 2,882 | -4.3% |
| 8 | e8_radius_2 | 2,170 → 1,204 | -0.2% |
| 16 | zn_radius_2 | 4,720 → 3,850 | +2.3% |
| 16 | a_n_radius_2 | 6,120 → 4,760 | -7.1% |
| 16 | d_n_radius_4 | 683,276 → 453,502 | -3.1% |
| 24 | zn_radius_2 | 16,744 → 14,674 | -8.7% |
| 24 | a_n_radius_2 | 29,900 → 25,300 | -4.3% |
| 24 | d_n_radius_4 | 9,594,188 → 7,426,654 | -2.0% |

Eight of ten cells improve; the two small movements are within this host's
identical-binary dispersion and sit in cells where tails are under 5% of
total work. The direct-norm recomputation remains untouched here and still
dominates — that is the next isolated change.

## Carried enumeration norms

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 on top of the amortized tails. At a complete assignment,
the accumulated scaled partial norm `acc` equals `c G cᵀ · scale` exactly —
every level contributed its `S_k² · weights[k]` term on the way down — so the
leaf derives the exact norm with one checked division instead of an `O(n²)`
recomputation over the Gram matrix. The direct evaluation survives as a
differential oracle: a test compares every carried norm against
`Gram::norm_sq` across five lattices at three times their minimal diagonal.
The unstable counter is renamed from `direct_norms` to `leaf_norms`
accordingly.

Combined effect on the corpus, against the pre-amortization baseline:

| Dimension | Geometry | Time | Reduction |
| ---: | :--- | ---: | ---: |
| 8 | zn_radius_2 | 15.3 → 13.3 us | -12.9% |
| 8 | a_n_radius_2 | 12.1 → 9.5 us | -21.9% |
| 8 | d_n_radius_4 | 196.0 → 75.5 us | -61.5% |
| 8 | e8_radius_2 | 53.4 → 23.8 us | -55.3% |
| 16 | zn_radius_2 | 125.6 → 110.3 us | -12.2% |
| 16 | a_n_radius_2 | 125.7 → 76.1 us | -39.5% |
| 16 | d_n_radius_4 | 15.92 → 4.53 ms | -71.5% |
| 24 | zn_radius_2 | 571.8 → 453.9 us | -20.6% |
| 24 | a_n_radius_2 | 537.0 → 309.8 us | -42.3% |
| 24 | d_n_radius_4 | 181.1 → 50.2 ms | -72.3% |

Every cell improves, by up to 3.6x at the dimension-24 weight-four shell.
Node, leaf, vector-count, and allocation figures are unchanged; emitted
vectors and norms are pinned by the closed-form shell oracles and the
per-vector differential.

Two further candidates were measured or profiled and not selected. A
prepared enumerator reusing factorization buffers would save the 9-to-13
per-call allocations and roughly one factorization of setup; after the
carried norm that setup is under 10% of even the smallest cell (`Gso::new`
on the same eight-dimensional Gram measures about 0.8 us against a 13.3 us
census), it approaches zero on the large shells, and no crate consumer
repeats censuses over one lattice today — the prepared type fails its own
admission test. An iterative rewrite of the recursion was also rejected:
the post-change profile shows the walk dominated by `descend`'s own checked
arithmetic (37% of cycles, plus 11% in `i128` software division for the
bound and norm divisions), with no measurable call or stack overhead
signature.

## Relevant-vector stage baseline

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 on the same machine and toolchain. The relevant-vector
corpus proves published facet counts before timing — `2n` for `Z^n`,
`n(n+1)` roots for `A_n`, `2n(n−1)` roots for `D_n`, 240 roots for E8 — then
reports the public call, a counting-allocator total, and unstable stage
timings that separate parity-representative radius evaluation from the
enumeration walk and from final output materialization. Dimensions stop at
12: at 14 the walk takes seconds, and 16 exceeds the default node budget,
which is the exponential state count the dimension cap exists to name.

| Dimension | Geometry | Total | Setup | Walk | Finalize | Allocations |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 8 | zn | 1.263 ms | 36.7 us | 1.335 ms | 2.1 us | 10,883 |
| 8 | a_n | 666.8 us | 30.9 us | 602.7 us | 9.9 us | 5,234 |
| 8 | d_n | 1.604 ms | 29.2 us | 1.515 ms | 12.9 us | 7,362 |
| 8 | e8 | 1.836 ms | 36.4 us | 1.576 ms | 32.7 us | 7,935 |
| 10 | zn | 19.53 ms | 188 us | 20.25 ms | 4.2 us | 105,558 |
| 10 | a_n | 8.42 ms | 168 us | 7.75 ms | 25.3 us | 43,515 |
| 10 | d_n | 25.82 ms | 177 us | 25.07 ms | 57.0 us | 60,721 |
| 12 | zn | 292.8 ms | 902 us | 291.9 ms | 14.2 us | 1,015,307 |
| 12 | a_n | 120.3 ms | 829 us | 111.3 ms | 49.6 us | 367,963 |
| 12 | d_n | 435.2 ms | 844 us | 437.5 ms | 75.9 us | 494,623 |

Radius evaluation is 0.2% to 3% of every cell, and materialization is
negligible; the coset-classification walk dominates completely. Allocations
track emissions — `Z^12` performs over one million allocations to report 24
vectors — because every first-of-coset and tied minimum stores a fresh
coordinate vector.

## Flat relevant-vector storage

Command:

```sh
taskset -c 2 cargo bench --bench optimization --features internals
```

Measured 2026-08-24 against the stage baseline. Each coset now keeps its
best norm, an arrival count capped past two, and at most two coordinate
blocks in one flat buffer. A coset is relevant exactly when its minimum is
attained by precisely two opposite vectors, so ties beyond the second are
proved irrelevant and never stored; the per-vector heap allocation
disappears from the walk entirely.

| Dimension | Geometry | Time | Allocations |
| ---: | :--- | ---: | ---: |
| 8 | zn | 1.263 → 0.972 ms (-23.0%) | 10,883 → 32 |
| 8 | a_n | 666.8 → 505.3 us (-24.2%) | 5,234 → 91 |
| 8 | d_n | 1.604 → 1.410 ms (-12.1%) | 7,362 → 131 |
| 8 | e8 | 1.836 → 1.525 ms (-16.9%) | 7,935 → 261 |
| 10 | zn | 19.53 → 15.00 ms (-23.2%) | 105,558 → 39 |
| 10 | a_n | 8.42 → 7.06 ms (-16.1%) | 43,515 → 131 |
| 10 | d_n | 25.82 → 23.97 ms (-7.2%) | 60,721 → 203 |
| 12 | zn | 292.8 → 239.4 ms (-18.2%) | 1,015,307 → 43 |
| 12 | a_n | 120.3 → 101.6 ms (-15.6%) | 367,963 → 178 |
| 12 | d_n | 435.2 → 409.6 ms (-5.9%) | 494,623 → 288 |

Every cell improves, allocations fall by three to four orders of magnitude,
and the remaining count is output-proportional plus the fixed flat buffers.
Emitted vectors are pinned by the facet-count oracles, the opposite-pairing
and lexicographic-order fixtures, and the unchanged public signature.

The Gray-code radius traversal was rejected on the stage evidence above:
the parity-representative loop it would accelerate is bounded at 3% of any
cell, below this host's run-to-run dispersion, so no stable broad win was
available.

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
| 8 | 4.753 µs | 20.829 µs | `lattica` 4.38x |
| 16 | 45.528 µs | 46.171 µs | parity |
| 24 | 163.560 µs | 164.156 µs | parity |

Incremental exact GSO updates, lazy reduction, the exact zero-quotient filter,
contiguous transactional state updates, and contiguous adjacent swaps removed
the earlier orders-of-magnitude gap. `lattica` is faster at dimension 8 and at
measurement parity with fplll at dimensions 16 and 24. The comparison is
deliberately not called contract-equivalent: fplll reduces an ambient basis
without returning the transform, while `lattica` reduces a Gram matrix and
always returns it.

### Re-measurement (2026-08-25)

Re-ran after rebuilding `target/fplll-compare` from the pinned 5.5.0
static library: the binary on disk predated the lattice-engine CVP
extraction and its dimension-24 LLL figure did not match the recorded
baseline. Three interleaved rounds, fplll and `lattica` alternating on
CPU 2, each side the median of 11 in-process samples; both sides report
identical input fingerprints (`202872`/`1230409`/`3738818`).

| Dimension | `lattica` median | fplll median | Faster library |
| ---: | ---: | ---: | ---: |
| 8 | 5.121 µs | 14.405 µs | `lattica` 2.81x |
| 16 | 47.543 µs | 48.800 µs | `lattica` 2.6% (parity) |
| 24 | 168.883 µs | 167.009 µs | fplll 1.1% (parity) |

The parity conclusion is unchanged: `lattica` is fastest at dimension 8
and at measurement parity at 16 and 24. The two parity cells sit inside
this host's identical-binary dispersion.

The general-CVP comparison, its fplll `CVPM_PROVED` miss and Babai-cycle
reproducers, and their data files moved to `lattice-engine` with the CVP
harness. The setup above still builds the fplll library the LLL comparison
uses.

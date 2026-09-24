# Benchmarks

Public API timings at source revision `87897cc`. Paired cells are
**Core Ultra 7 258V / Core i7-12700K**. Aggregation varies by harness, as
listed below.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Linux 7.2.6, Arch Linux | 1.98.0 | 3 |
| Golden Cove | Intel Core i7-12700K | Linux 7.2.6, CachyOS | 1.98.1 | 8 |

| Setting | Value |
| --- | --- |
| Source | Same `lattica` tree on both hosts, `87897cc` |
| SIMD dependency | `simdispatch` resolved to the local working copy through the umbrella patch table; revision not recorded |
| Kernel builds | `--features internals`; portable comparison adds `--no-default-features` |
| Native compiler | GCC 16.2.1 on both hosts |
| Backend evidence | x86 v3 dispatch policy; resolved backend not recorded |
| Input setup | Deterministic inputs and reusable preparation outside timing |
| Allocation | Public result allocation included |

## Real transforms

Column-major coefficients; strided vectors for array-of-structures (AoS)
batches and coordinate planes for structure-of-arrays (SoA) batches.

### Sixteen-output structure-of-arrays dispatch

16-by-16 `transform_batch_soa`, Criterion middle estimates.

| Vectors | Scalar (µs) | Dispatched (µs) |
| ---: | ---: | ---: |
| 8 | 0.442/0.495 | 0.450/0.506 |
| 64 | 1.941/1.943 | 1.604/1.349 |
| 257 | 8.892/8.058 | 6.747/5.835 |

### Twenty-four-output structure-of-arrays dispatch

24-by-24 `transform_batch_soa`.

| Vectors | Scalar (ns) | Dispatched (ns) |
| ---: | ---: | ---: |
| 1 | 780/602 | 202/204 |
| 4 | 879/684 | 97/93 |
| 8 | 1,122/1,014 | 190/179 |
| 16 | 1,507/1,529 | 370/351 |
| 32 | 2,527/2,423 | 743/692 |
| 64 | 4,335/4,285 | 1,482/1,414 |
| 128 | 8,054/8,079 | 2,962/2,818 |
| 257 | 22,758/19,925 | 6,660/6,435 |

### Twenty-four-output array-of-structures dispatch

24-by-24 `transform_batch`, Criterion means. The portable build disables
default features; the dispatched build enables `simd`.

| Vectors | Portable (µs) | Dispatched (µs) |
| ---: | ---: | ---: |
| 1 | 0.0879/0.0896 | 0.0883/0.0909 |
| 4 | 0.3431/0.3477 | 0.2709/0.2621 |
| 8 | 0.6868/0.6886 | 0.4062/0.3837 |
| 64 | 5.4311/5.4322 | 2.1225/2.0312 |
| 257 | 22.009/21.774 | 8.7612/8.2594 |

### Single-vector transforms

Square `transform`, Criterion middle estimates.

| Dimension | Time (ns) |
| ---: | ---: |
| 8 | 16.28/16.20 |
| 16 | 47.64/48.23 |
| 24 | 86.74/86.58 |

### Dispatch geometry

| Public entry | Geometry | Dispatched when |
| --- | --- | --- |
| `transform_batch_soa` | 16 outputs, any row count | At least 64 vectors |
| `transform_batch_soa` | Exact 24-by-24 geometry | Every batch size |
| `transform_batch` | Exact 24-by-24 array-of-structures geometry | At least 4 vectors |

- Other shapes and single-vector calls use the portable path.
- Separate multiply and add preserve scalar accumulation order and bit identity.
- Forced backend requests do not prove that a particular geometry used SIMD.

## Lattice reduction

Public one-shot calls over 16 deterministic integral bases per cell,
`Delta::STRONG`, exact `i128` Gram and transform arithmetic. Result construction
is included; times are medians per basis.

### Ordinary LLL

Adjacent-swap workloads built with deterministic shears.

| Dimension | Geometry | `lll` (µs) |
| ---: | :--- | ---: |
| 8 | `shear_2` | 3.593/3.923 |
| 8 | `shear_4` | 4.181/4.526 |
| 8 | `shear_6` | 4.473/4.942 |
| 16 | `shear_2` | 17.965/18.183 |
| 16 | `shear_4` | 20.405/21.115 |
| 16 | `shear_6` | 20.597/21.987 |
| 24 | `shear_2` | 46.166/48.312 |
| 24 | `shear_4` | 49.318/52.015 |
| 24 | `shear_6` | 50.458/53.755 |

### Deep-insertion LLL

Short basis vectors precede long ones; upper-row shears force deep insertion.

| Dimension | Geometry | `lll_deep` (µs) |
| ---: | :--- | ---: |
| 8 | `insertion_light` | 4.792/4.836 |
| 8 | `insertion_medium` | 5.352/5.544 |
| 8 | `insertion_dense` | 5.801/6.167 |
| 16 | `insertion_light` | 32.708/33.910 |
| 16 | `insertion_medium` | 35.925/37.881 |
| 16 | `insertion_dense` | 40.766/42.406 |
| 24 | `insertion_light` | 107.722/111.617 |
| 24 | `insertion_medium` | 117.629/123.434 |
| 24 | `insertion_dense` | 133.452/139.418 |

### Exact-algebra operations

| Dimension | Positive-definite check (ns) | Triangular determinant (ns) | Fraction-free adjugate solve (µs) | Classical HNF (µs) | Determinant-modular HNF (µs) | Invariant factors (µs) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 776/845 | 62/58 | 8.422/6.754 | 0.340/0.387 | 0.523/0.563 | 0.469/0.508 |
| 16 | 5,615/6,030 | 140/154 | 71.643/60.013 | 1.111/1.128 | 2.099/1.925 | 1.570/1.703 |
| 24 | 18,580/19,942 | 258/274 | 251.323/201.941 | 2.390/2.417 | 4.088/4.344 | 3.780/4.573 |

## Short-vector enumeration

Public `for_each_short` by lattice and squared-radius bound. Shell counts or
the published E8 kissing number are checked before timing.

| Dimension | Geometry | Vectors | Median (µs) |
| ---: | :--- | ---: | ---: |
| 8 | `Z^8`, squared radius 2 | 128 | 12.6/11.4 |
| 8 | `A_8`, squared radius 2 | 72 | 8.4/7.8 |
| 8 | `D_8`, squared radius 4 | 1,248 | 69.1/64.4 |
| 8 | `E_8`, squared radius 2 | 240 | 22.0/20.6 |
| 16 | `Z^16`, squared radius 2 | 512 | 102.0/109.9 |
| 16 | `A_16`, squared radius 2 | 272 | 68.6/70.8 |
| 16 | `D_16`, squared radius 4 | 29,632 | 4,224.5/4,051.5 |
| 24 | `Z^24`, squared radius 2 | 1,152 | 424.8/449.2 |
| 24 | `A_24`, squared radius 2 | 600 | 281.3/306.6 |
| 24 | `D_24`, squared radius 4 | 171,168 | 46,091.8/46,359.5 |

## Relevant vectors

Public `relevant_vectors`, with published facet counts checked before timing.

| Dimension | Geometry | Facets | Median (µs) |
| ---: | :--- | ---: | ---: |
| 8 | `Z^8` | 16 | 2.6/2.2 |
| 8 | `A_8` | 72 | 460.4/450.0 |
| 8 | `D_8` | 112 | 1,285.4/1,211.7 |
| 8 | `E_8` | 240 | 1,386.7/1,315.8 |
| 10 | `Z^10` | 20 | 3.0/3.1 |
| 10 | `A_10` | 110 | 6,485.2/6,115.0 |
| 10 | `D_10` | 180 | 22,029.0/20,606.6 |
| 12 | `Z^12` | 24 | 3.8/3.8 |
| 12 | `A_12` | 156 | 92,862.6/89,591.6 |
| 12 | `D_12` | 264 | 371,313.4/359,232.1 |

## LLL comparison

The same 16 deterministic integral bases per dimension: a banded basis with
`2n` bounded unimodular row shears, `δ = 0.99`.

| Dimension | `lattica` (µs) | fplll (µs) |
| ---: | ---: | ---: |
| 8 | 5.012/22.867 | 13.712/14.679 |
| 16 | 47.556/54.936 | 49.188/49.417 |
| 24 | 171.235/187.169 | 170.685/173.354 |

| Detail | Value |
| --- | --- |
| Competitor | [fplll](https://github.com/fplll/fplll) 5.5.0, `a8dedce384689047daba154bd50d6215e35bf03b` |
| Native build | Static, GCC 16.2.1, `-O3 -march=native -DNDEBUG`; GMP 6.3.0, MPFR 4.2.2 |
| Input fingerprints, dimensions 8/16/24 | `202872` / `1230409` / `3738818` on both hosts |
| Included | Input copy, result allocation, and `lattica` transform construction |
| Excluded | Input generation and process startup |

- The contracts differ: `lattica` returns a reduced Gram matrix and unimodular
  transform; fplll reduces an ambient basis without returning a transform.
- Closest-vector rows from the native binary have no counterpart here and are
  excluded.
- The comparison harness is repository-only under `external/`, outside the
  published crate's build and dependency graph.

## Reproduction and aggregation

Run from the crate root with the host's pinned CPU:

```sh
export FEC_GOLDEN_CORE=<cpu>
just bench-save kernel
just bench kernel
just bench optimization
just bench-fplll
```

- `kernel` uses Criterion's `before` baseline. The recorded portable build
  used `--no-default-features --features internals`; the shared benchmark
  recipe forces all features and does not reproduce that build configuration.
- `optimization` emits CSV; Criterion baseline flags do not supply its
  before/after comparison. Capture separate runs with matching inputs.
- `bench-fplll` runs only the Rust side. Alternate it with the separately built
  native binary for the comparison rounds.

| Harness / result | Aggregation | Sampling |
| --- | --- | --- |
| Criterion SoA and single-vector | Middle estimate, one run per host and feature set | 100 samples, 3 s warmup |
| Criterion AoS | Mean, one run per host and feature set | 100 samples, 3 s warmup |
| 24-by-24 SoA | Statistic not specified in the source record | - |
| Custom algebra, reduction, enumeration | Median of 11 in-process samples | Deterministic inputs |
| LLL comparison | Paired median of 5 interleaved rounds per side | 11 in-process samples per round; alternate first binary |

Native comparison build; verify the checkout against the revision above:

```sh
git clone --depth 1 --branch 5.5.0 \
  https://github.com/fplll/fplll.git target/fplll-5.5.0
(cd target/fplll-5.5.0 && ./autogen.sh && \
 ./configure --disable-shared CXXFLAGS="-O3 -march=native -DNDEBUG" && make -j)
c++ -O3 -march=native -DNDEBUG -std=c++17 \
  -Itarget/fplll-5.5.0 external/bench-fplll/fplll_compare.cpp \
  target/fplll-5.5.0/fplll/.libs/libfplll.a \
  -lmpfr -lgmp -lpthread -o target/fplll-compare
taskset -c "$FEC_GOLDEN_CORE" target/fplll-compare
```

| Dimension-8 comparison dispersion | Per-round range (µs) |
| --- | --- |
| `lattica` | 4.4–5.2/5.8–33.2 |
| fplll | 13.4–14.0/14.5–19.9 |

- Short cases are sensitive to timer resolution and run-to-run dispersion.
- Paired cells are independent host measurements, not cross-host speedups.
- Rerun matched baselines before changing a threshold; these recorded values
  do not establish a new performance claim.

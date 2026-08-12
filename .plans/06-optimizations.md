# Optimization roadmap

A private engineering map for making `lattica` faster without weakening the
contracts that make its answers useful. This is deliberately broader than a
release plan: it records every credible optimization avenue found in the
current implementation and in mature lattice / exact-linear-algebra systems,
then ranks the work by evidence.

The order below is the default decision. A later measurement may reorder it;
intuition alone may not.

## Non-negotiable contracts

An optimization is invalid if it violates any of these:

1. Integer arithmetic stays checked and exact. No wrapping fast path, hidden
   big-integer fallback, or approximate certificate.
2. Reduction still acts on the integral Gram matrix and confirms every
   reduction predicate with exact `Delta`. Floating-point data may at most
   propose work; it may not decide that a basis is reduced.
3. Decoder decisions keep the current operation order. No FMA, reassociation,
   approximate pruning, or `fast-math`. Equal-distance answers retain the
   lexicographically smallest basis coordinates.
4. A successful `nearest_ml` / named ML decode is globally nearest. A fast
   bounded-distance decoder may seed an exhaustive proof, but may not replace
   that proof unless equivalence to the complete ML contract is established.
5. Node and iteration budgets remain hard. Parallel or reordered traversal may
   not turn a bounded deterministic call into an unbounded one.
6. Warm nearest-point and batch paths remain allocation-free. Setup and result
   materialization may allocate where the API says they do.
7. Runtime backend policy remains `simdispatch`'s. `archmage` only supplies the
   safe capability token for the selected tier.
8. No `unsafe`, and no dependency beyond those allowed by the crate rules.
9. SIMD differential tests require bit-identical results, not a tolerance.
10. Optimizations must be workload-backed. Crossover constants are benchmark
    results and belong with the benchmark record.

These rules deliberately exclude several common shortcuts used by
cryptanalytic lattice software: arbitrary-precision arithmetic, heuristic LLL,
extreme pruning that can miss a point, and relaxed floating-point answers.
Their throughput is informative; their contracts are not interchangeable with
this crate's.

## Evidence available now

Measurements are from the pinned comparison record in `BENCHMARKS.md`, on the
same machine and toolchain unless stated otherwise.

### Exact LLL is the dominant measured deficit

| Dimension | `lattica` | fplll | fplll speedup |
| ---: | ---: | ---: | ---: |
| 8 | 86.62 µs | 14.07 µs | 6.16x |
| 16 | 3.3341 ms | 46.553 µs | 71.6x |
| 24 | 28.167 ms | 162.673 µs | 173x |

The growth identifies the cause more strongly than a profiler sample would:
`reduce_with` rebuilds a complete fraction-free `Gso` after every basis
operation. Each rebuild is cubic; `State::subtract` and `State::swap` also
clone the whole Gram matrix and revalidate symmetry. fplll instead maintains
reduction state incrementally. Reduction is offline, but this gap also blocks
cheap automatic preconditioning for every decoder.

fplll is not an exact behavioral oracle. The benchmark found both a public
`CVPM_PROVED` wrong-answer fixture and a non-terminating Babai preprocessing
fixture in fplll 5.5.0 and current master. Its implementation techniques are
still useful when independently re-derived and tested against `lattica`'s
stronger invariants.

### Generic CVP has competitive per-query machinery, but poor preparation

On the current reduced benchmark corpus, warm `lattica` CVP is 13.9x and 3.49x
faster than fplll `FAST` at dimensions 8 and 16, and within 1.07x at dimension
24. Cold `lattica` is also faster at dimensions 8 and 16. That means a rewrite
must separate two costs:

- **search-tree size**, controlled mostly by basis quality and the initial
  candidate radius;
- **cost per visited node**, controlled by center updates, bounds, conversions,
  and traversal state.

The current node performs an `O(n)` center sum and a square root plus
`ceil`/`floor` to create every child interval. Both are avoidable. The current
`Enumerator` does at least correctly prepare exact GSO once and reuse scratch.

### Named high-dimensional decoding is correct but not production-fast

`BarnesWall16` and `Leech24` currently run generic exhaustive
Schnorr–Euchner search over their published bases. This gives a strong ML
contract, but the acceptance sweep reaches the node budget on difficult words.
The published recursive Barnes–Wall and hexacode / Leech algorithms exist
precisely to avoid this tree. They cannot silently replace exhaustive ML where
their failure region differs, but they can supply much tighter initial
candidates and admissible structure-aware bounds.

### SIMD has one proved crossover, not a general mandate

The AVX2 structure-of-arrays 16-output transform is slower than scalar at 8
vectors and about 1.17--1.18x faster at 64--257 vectors. The measured threshold
is therefore 64 vectors. AoS and single-vector dispatch remain scalar by
choice. No data yet supports SIMD for 24 outputs, closed-form quantizers,
integer algebra, or search.

### Allocation is already good on the primary hot path

`tests/zero_alloc.rs` proves warm closed-form batch quantization and prepared
enumeration allocate nothing. Remaining allocations are mostly setup, exact
algebra, or output-proportional list/relevant-vector materialization. Optimize
them only when they affect a measured repeated workload.

## What mature implementations teach

The useful patterns are architectural, not code to copy.

### fplll / modern LLL implementations

- Compute the exact Gram matrix once, then update it when the basis changes.
- Keep GSO state and row exponents incrementally rather than refactorizing after
  every size reduction or swap.
- Separate fast floating estimates from proved modes, increasing precision or
  falling back when an estimate is inconclusive.
- Implement enumeration as an iterative state machine with per-level arrays:
  centers, partial distances, coordinates, and zig-zag deltas. The inner loop
  updates state instead of rebuilding a child range at every node.
- Preprocess CVP with a reduced basis and use Babai only as a candidate bound.

Relevant sources:

- fplll enumeration source:
  <https://github.com/fplll/fplll/blob/master/fplll/enum/enumerate.cpp>
- Nguyen and Stehle, *An LLL Algorithm with Quadratic Complexity* (exact Gram
  matrix plus incremental updates):
  <https://perso.ens-lyon.fr/damien.stehle/downloads/fpLLL_journal.pdf>
- fplll project and its proved / heuristic / fast modes:
  <https://github.com/fplll/fplll>

The default adaptation here is stricter: estimates may schedule an exact test,
but an exact integer comparison makes the decision.

### FLINT / mature exact linear algebra

FLINT does not use one algorithm for every shape. It selects among direct,
Bareiss, and modular determinant methods; computes an inverse by solving all
identity right-hand sides with one fraction-free LU; and offers classical,
extended-GCD, modular, minors, and Pernet--Stein HNF variants. The important
lessons for a fixed-width crate are:

- share one decomposition across all right-hand sides;
- have small direct kernels and structure fast paths;
- dispatch algorithms on dimension and entry magnitude only after measuring;
- prefer deterministic proved modular reconstruction if modular methods ever
  become worthwhile.

Source: <https://flintlib.org/doc/fmpz_mat.html#determinant>,
<https://flintlib.org/doc/fmpz_mat.html#inverse>, and
<https://flintlib.org/doc/fmpz_mat.html#hermite-normal-form-hnf>.

FLINT's arbitrary precision and probabilistic modes are not candidates for
`lattica`; the decomposition reuse and algorithm-selection ideas are.

### Specialized lattice decoders

- Barnes--Wall has recursive bounded-distance decoding that exploits the
  `RM(1,m)` / squaring structure rather than treating the basis as generic.
- Leech decoding can project through the hexacode or use bounded-distance Golay
  structure; published algorithms use orders of magnitude fewer real
  operations than generic enumeration.
- A fast candidate generator and an exhaustive proof compose safely: any valid
  candidate gives a smaller radius, while the existing enumerator still proves
  ML and applies the crate's tie rule.

References already governing the named decoders:

- G. D. Forney, *Coset codes -- Part II: Binary lattices and related codes*.
- D. Micciancio and A. Nicolosi, *Efficient bounded distance decoders for
  Barnes-Wall lattices*.
- A. Vardy and Y. Be'ery, *Maximum likelihood decoding of the Leech lattice*:
  <https://doi.org/10.1109/18.243466>.
- O. Amrani and Y. Be'ery, *Fast decoding of the Leech lattice*:
  <https://doi.org/10.1109/49.29617>.

## Priority order

| Priority | Work | Why now |
| --- | --- | --- |
| P0 | Benchmark decomposition and profiles | Prevents optimizing the wrong half of CVP or LLL. |
| P1 | Incremental exact LLL state | Largest measured gap: 6x to 173x and worsening with dimension. |
| P2 | Lower-cost generic enumeration nodes | Hot, reusable by generic and both named ML decoders. |
| P3 | Better ML bounds and named-lattice preprocessing | Reduces tree size without weakening correctness. |
| P4 | Specialized Barnes--Wall and Leech candidate engines | Largest likely high-dimensional gain, but high proof cost. |
| P5 | Exact linear algebra algorithm repairs | Clear asymptotic defects, mostly setup/offline today. |
| P6 | Additional measured batch SIMD | Modest proved gain; useful only for actual batch consumers. |
| P7 | Allocation, layout, and compiler cleanup | Take only profiler-proved wins after the algorithms change. |

P0--P3 are the default first implementation sequence. P4 is not allowed to
block the generic engine improvements. P5--P7 remain independent tracks once
benchmarks identify a consumer.

## P0 -- Measure costs separately

Add or extend benchmark cases before changing algorithms:

1. **LLL operation counters.** Record factorization count, size reductions,
   swaps, deep insertions, Gram copies, and exact arithmetic operations for the
   existing dimensions 8, 16, and 24 plus skew/bit-width sweeps. Time setup and
   certificate verification separately.
2. **CVP node economics.** For cold and warm calls, record preparation time,
   Babai time, node count, nanoseconds per node, successful radius shrinks, and
   maximum depth. Use easy, median, and budget-exhausting targets. A faster run
   caused only by a different tree must not be reported as a faster node loop.
3. **Named decoder stages.** Time ambient-to-coefficient transform, initial
   candidate, exhaustive proof, and coordinate-to-ambient transform separately.
4. **Exact algebra matrix.** Benchmark `det`, `adjugate`, positive-definite
   check, HNF, determinant-modular HNF, SNF, Gram construction, and matrix
   multiplication across dimensions, density, and entry bit width.
5. **Quantizer batches.** Benchmark every closed form at realistic batch sizes
   in AoS and SoA layouts before adding a kernel.
6. **Allocation counters.** Keep the existing steady-state contract; add counts
   for list enumeration, relevant vectors, repeated short-vector census, and
   Construction A membership only if those become repeated workloads.
7. **Profiles.** Use pinned release builds and collect cycles, instructions,
   branches, branch misses, cache misses, and allocation profiles. Inspect
   generated code for a proposed inner-loop change; do not infer codegen from
   source shape.

Acceptance: each proposed optimization has a stable corpus, a correctness
fingerprint, and a named metric. Criterion wall time alone is insufficient for
search because node count can hide a regression.

## P1 -- Incremental exact LLL

### P1.1 Remove full-matrix transactional copies

Current `State::subtract` clones `n²` entries, performs a row and column update,
then scans the matrix again in `Gram::new`. `swap` does the same. Introduce a
crate-private symmetric reduction buffer that:

- owns one row-major Gram matrix and the transform;
- preflights checked products/sums into reusable row scratch so failure remains
  atomic;
- writes the symmetric row/column exactly once;
- swaps symmetric indices in place;
- converts to validated `Gram` only at the boundary.

This removes allocation, copying, and redundant symmetry checks without
changing the algorithm. Land it separately so its gain can be measured before
GSO changes.

### P1.2 Reuse GSO storage

`Gso::new` allocates and copies on every call. First make a reusable
factorization workspace and refactor-in-place API. This is not the final
asymptotic fix, but it separates allocator/copy cost from arithmetic cost and
provides the oracle for the updater.

### P1.3 Maintain exact reduction state incrementally

Re-derive exact update formulae for the fraction-free `lambda` and leading
minors under:

- `b_k <- b_k - q b_j` size reduction;
- adjacent swaps;
- the sequence of adjacent swaps used for deep insertion.

The updater must use `try_*` operations and exact division. After every
operation in debug/differential tests, compare every minor and `lambda` against
a fresh `Gso::new`. Production performs no periodic approximate reset: either
the exact recurrence succeeds or the call returns its normal range error.

Start with ordinary LLL. Deep insertion should consume the same updater only
after adjacent swaps are proved. Do not create two GSO conventions.

### P1.4 Avoid repeated deep-insertion denominator work

`deep_insertion_point` rebuilds an LCM and weighted sum for every candidate
position, producing cubic work and unnecessary magnitude growth. Evaluate
suffix products/sums incrementally from `k` down to zero, cancelling gcds before
multiplication where the exact identity permits. Confirm each result against
the current implementation on random positive-definite Gram matrices.

### P1.5 Optional exact-confirmed estimate layer

If exact recurrence cost remains dominant, cache `f64` approximations of `mu`
and projected norms only to identify likely zero quotients and likely Lovasz
outcomes. Every nontrivial action and every decision to advance `k` is confirmed
by the existing exact comparison. Near a boundary, go directly to exact. This
can skip expensive exact work but can never alter the certificate.

Do not add adaptive multiprecision. Do not import fplll's heuristic/proved
wrapper semantics. A more ambitious Nguyen--Stehle L2 implementation is a
replacement candidate only after the incremental exact path is measured.

### P1 acceptance

- Every output passes independently recomputed size-reduction, Lovasz,
  determinant, and unimodularity checks.
- Differential randomized tests compare updater state with a fresh exact GSO
  after each elementary operation.
- Overflow leaves public outputs untouched and reports the same error class.
- Ordinary LLL improves materially at dimensions 16 and 24; the target is to
  remove the current super-cubic-looking growth, not to match fplll at the cost
  of semantics.
- The fplll corpus remains diagnostic only. A different reduced basis is valid;
  a failed certificate is not.

## P2 -- Reduce generic enumeration cost per node

### P2.1 Cache real triangular coefficients

`center` and `weight` repeatedly convert exact `Int` values to `f64` and divide.
At `Enumerator::new`, derive immutable `mu: Vec<f64>` and diagonal weights once
from the exact GSO, in the same loop order and with the same conversions used
now. Keep the exact `Gso` for Babai and diagnostics. Differential tests require
bit-identical centers and output decisions.

### P2.2 Replace recursive recomputation with per-level state

Expand `EnumerationScratch` with the classic state-machine arrays:

- coordinate `x[k]`;
- center and center partial sums;
- partial squared distance;
- zig-zag step and direction;
- per-level remaining bound.

Descend and backtrack iteratively. Updating a chosen coordinate adjusts the
next level's center from cached state rather than summing all higher
coordinates. Preserve the current deepest-to-shallowest, nearest-first zig-zag
order and node accounting so budget behavior stays comparable.

### P2.3 Remove square root from child generation

The current `Children::new` computes `sqrt`, `ceil`, and `floor` at every node.
Start at the rounded center, advance alternating integer offsets, and stop when
the next offset's contribution exceeds the remaining radius. Positive diagonal
weights make the stop monotone. This is the pattern used by mature
Schnorr--Euchner loops and removes a high-latency operation without pruning any
candidate.

Boundary checks must retain the current inclusive radius and i64-range rules.
No FMA and no algebraic reassociation of the accumulated distance.

### P2.4 Improve the initial candidate, separately from traversal

Expose an internal entry point that accepts a validated lattice candidate and
its radius. Candidate sources, in increasing cost:

1. current Babai nearest plane;
2. Babai over a stronger reduced basis;
3. several deterministic basis orderings / nearest-plane variants;
4. a named-lattice candidate engine.

The candidate only shrinks the starting radius. Search still proves ML. Measure
candidate time, starting radius, and nodes saved; a smaller radius that costs
more than the saved tree is a loss.

### P2.5 Add automatic prepared-basis preconditioning

Create a prepared CVP form that owns a reduced Gram matrix, the unimodular
coordinate transform and inverse mapping, and one enumerator. Convert targets
to reduced-basis coordinates, enumerate there, then convert the exact integer
answer back to the caller's original basis coordinates.

Default decision: add this as a distinct prepared type or constructor first;
do not silently change `Enumerator::new` node counts or setup cost. Named
high-dimensional decoders may use it internally once the mapping is verified.
Strong LLL (`delta = 0.99`) is the first benchmark; deep insertion and multiple
bases are later candidates.

### P2.6 Make list output proportional but not wasteful

The full list API must allocate one answer per emitted point. Still possible:

- use `sort_unstable_by`: the comparator's distance-plus-coordinate order is
  total, so stable ordering adds no semantics;
- add a callback iterator/sink form for consumers that do not need ownership;
- add a caller-owned flat coordinate slab form for repeated calls;
- add a separate top-`k` API using a bounded max-heap and radius shrink rather
  than materializing every point in the original radius.

Do not change `list` into top-`k`; they are different contracts.

### P2 acceptance

- Frozen and randomized targets produce the same nearest point and distance.
- Equal-distance fixtures retain lexicographically smallest coordinates.
- Preserving traversal work preserves node counts; any preconditioned API
  documents that its count is for the transformed search.
- Budget exhaustion and outside-radius errors leave outputs untouched.
- Measure nodes and nanoseconds/node independently at dimensions 8, 16, and 24.
- Warm search remains zero-allocation.

## P3 -- Shrink the proof tree safely

### P3.1 Precondition the named bases

The published `BW_16` and Leech generators are representation authorities, not
necessarily good enumeration bases. During decoder construction:

1. reduce their Gram matrix strongly;
2. retain the unimodular transform;
3. transform the ambient dual map into reduced coordinates;
4. enumerate the reduced Gram;
5. map the integer result back before producing published ambient numerators.

All transforms are setup work. Test both the reduced and original coordinate
forms against the same ambient point and exact squared distance. This is the
lowest-risk way to attack named-decoder budget exhaustion after P1 makes setup
cheap.

### P3.2 Stronger admissible lower bounds

Investigate only bounds that cannot remove a valid point:

- cached suffix projected-norm bounds from the triangular system;
- parity/coset constraints known from the named lattice;
- exact or outward-rounded bounds where a floating computation is used;
- branch ordering from a cheap candidate score, with the original proof bounds
  unchanged.

Extreme pruning and probabilistic pruning are explicitly rejected for ML.
They are appropriate for cryptanalytic success-probability searches, not this
API.

### P3.3 Deterministic multi-start candidates

For difficult targets, run a small fixed set of cheap nearest-plane candidates
on deterministically permuted/reduced bases, keep the best by the exact public
tie order, then launch one proof search. No randomness enters the result or
budget. Benchmark the break-even target difficulty before enabling more than
one start.

### P3.4 Subtree scheduling is deferred

Parallel subtree search can help large trees, but it complicates radius sharing,
node budgets, reproducibility, and lexicographic ties. Do not put Rayon inside a
single decode until the sequential state machine and bounds are exhausted.
Consumers can already decode independent received words in parallel. If later
needed, partition a fixed top level deterministically, assign fixed node-budget
slices, and merge by the total tie order.

## P4 -- Specialized named-lattice candidate engines

### P4.1 Barnes--Wall recursion

Implement the published recursive decomposition as a scalar reference candidate
engine:

- reuse caller scratch at every recursion level;
- produce one or more valid `BW_16` candidates in published ambient scaling;
- verify membership and compute distance before using a candidate;
- feed the best candidate into generic exhaustive proof.

Do not label bounded-distance failure as `NotInLattice`, and do not return an
unproved candidate from the ML API. First success criterion is fewer exhaustive
nodes, not direct replacement.

### P4.2 Leech hexacode / Golay candidate engine

Implement the published soft-decision projection as a separate candidate
engine with fixed tables:

- generate coordinate-pair metrics without allocation;
- decode the small hexacode/Golay structure;
- enumerate the required parity/sign families in a fixed order;
- validate the resulting Leech candidate against the generator;
- use its squared distance as the exhaustive radius.

Only after broad differential testing against exhaustive enumeration should a
published maximum-likelihood algorithm be considered as a direct engine. Even
then, the crate's total tie rule and output scaling must be proved, not assumed
from a paper that permits any nearest point.

### P4.3 Tables, layout, and batch

Precompute immutable generator/dual tables that are currently derived by
adjugate during every decoder construction, provided generated constants carry
an independent identity test. Keep scratch as structure-of-arrays where metrics
for many candidate symbols are evaluated together. Batch ambient transforms may
reuse `transform_batch_soa`; divergent recursive decisions remain scalar until
measured otherwise.

### P4 acceptance

- Candidate engines never worsen the initial radius.
- Every candidate is independently shown to be in the named lattice.
- Exhaustive output remains identical on frozen ties, random low-noise words,
  boundary words, and known budget-exhausting words.
- Report candidate cost, node reduction, success rate, and total latency. A
  dramatic candidate speedup with unchanged end-to-end budget exhaustion does
  not count.

## P5 -- Exact integer linear algebra

These are credible defects, but most are setup/offline and need P0 evidence.

### P5.1 Positive definiteness in one factorization

`Gram::is_positive_definite` currently allocates and computes a determinant for
every leading submatrix, roughly quartic work. One fraction-free `Gso::new`
already computes every leading principal minor and applies Sylvester's test in
cubic work. Share that implementation and map its errors without building
prefix matrices.

### P5.2 Adjugate through one multi-right-hand-side solve

`adjugate` computes `n²` determinants of size `n-1`, roughly quintic work. Use
one fraction-free LU/Bareiss decomposition and solve against the identity to
obtain `(adj(A), det(A))`, as mature exact libraries do. Small `n <= 3` direct
formulae can avoid setup if benchmarks support them.

Important: compare intermediate-overflow behavior. A faster recurrence that
overflows on an input whose cofactors fit is a regression. A modular fallback
is considered only if fixed-width reconstruction can prove the result and its
bounds without a new dependency.

### P5.3 Determinant algorithm selection

Keep Bareiss as the general oracle. Measure:

- diagonal, triangular, block-diagonal, and permutation fast paths;
- direct formulae for dimensions 1--4;
- reusable workspace for repeated determinants;
- row-content gcd cancellation where exact identities prove it;
- deterministic multimodular determinant plus CRT for larger fixed-width
  matrices, using a Hadamard bound to know when reconstruction is complete.

Multimodular work is low priority: dimensions are small, implementing prime and
CRT machinery expands the crate's mathematical surface, and fixed-width output
does not get FLINT's big-integer advantage.

### P5.4 HNF

The crate already has classical and determinant-modular full-rank paths.
Potential improvements:

- choose between them by measured dimension, density, determinant, and entry
  width;
- use extended-GCD row combinations to reduce repeated Euclidean descent;
- avoid scanning the full trailing matrix after local changes;
- reduce row entries modulo the determinant in contiguous row slices;
- add a Pernet--Stein/minors path only if tall Construction A/D matrices make it
  worthwhile.

The HNF must remain canonical. Algorithm dispatch may change the route but not
the returned form.

### P5.5 SNF invariant factors

Current diagonalization repeatedly scans the trailing submatrix for the
smallest nonzero and alternates full row/column operations. Candidates:

- extended-GCD 2-by-2 transforms that clear a pair in one operation;
- track candidate nonzeros instead of rescanning unchanged regions;
- determinantal-divisor or modular rank methods for large matrices;
- separate square full-rank and rectangular/rank-deficient paths.

Do not compute transform matrices: the API intentionally returns only invariant
factors, and transforms increase growth and work.

### P5.6 Matrix storage and loops

- Add crate-private mutable row slices and unchecked-by-construction geometry
  helpers that remain safe Rust; validate indices once per outer loop instead
  of through `get`/`set` on every entry.
- Use contiguous row operations and explicit transpose/cached columns when a
  measured algorithm is column-bound.
- Consider packed symmetric storage for read-mostly Gram matrices only after
  measuring. It halves memory but complicates row/column congruence updates and
  can hurt locality.
- Add sparse fast paths only to algorithms whose input is actually sparse.
  LDLC support remains CSR at its seam; do not turn `IntMatrix` into a hybrid
  abstraction.
- Reuse caller/prepared workspaces for repeated determinant, Gram, HNF, and GSO
  operations. Do not add hidden global caches.

### P5.7 Gram and dual construction

Named Gram construction currently evaluates both halves of a symmetric matrix;
compute one triangle and mirror it. General `Basis::gram` can exploit row slices,
zero rows, and symmetry. Decoder dual setup should consume the new shared
fraction-free inverse rather than build every cofactor separately.

## P6 -- SIMD and batch work

SIMD is useful only across independent vectors; a single dimension-16/24 vector
is too short and branch-heavy.

### P6.1 Complete measured transform coverage

- Measure a 24-output AVX2 kernel for Leech batches.
- Measure AArch64 NEON with the same structure-of-arrays contract.
- Consider wider x86 tiers only when `simdispatch` and `archmage` expose them
  canonically.
- Generate kernels specialized for the common fixed row/column counts if that
  removes loop overhead and bounds checks.
- Preserve separate multiply/add instructions and the scalar accumulation
  order. No FMA.

Every shape gets its own crossover. The existing 64-vector threshold is not a
universal constant.

### P6.2 Batch closed-form quantizers

`nearest_batch` currently invokes one scalar decoder per vector. For `Z^n`,
`D_n`, `D_n^+`, `A_n`, `A_n^*`, and the E8 coset candidates, investigate SoA
kernels that process independent received words per lane. Selection and parity
steps may use masks; scalar tails remain the oracle.

First measure coordinate arithmetic separately from layout conversion. Requiring
callers to transpose AoS into SoA for a small batch can erase the kernel gain.
Do not duplicate a quantizer's algorithm in `kernel`; the module owning the
layout owns its kernel and exposes one scalar semantic implementation.

### P6.3 Batch metric kernels

The planned `norm_sq_batch`, `inner_batch`, and Gram accumulation kernels are
still candidates. They are valuable for decoder candidate scoring and
statistics only if a consumer submits enough independent vectors. Add them in
consumer-driven order, not to fill an API table.

### P6.4 Integer SIMD is unlikely

Checked, variable-width exact arithmetic, early overflow, and small matrices
are poor SIMD targets. A specialized `i64` dot product may safely accumulate in
`i128` and narrow once only if that is proved equivalent to the documented
intermediate-overflow contract. Otherwise leave exact algebra scalar.

## P7 -- Remaining allocation and local improvements

### Exact short-vector enumeration

- Introduce a prepared exact enumerator holding widened Gram, factorization,
  weights, and reusable coordinates for repeated radii/censuses.
- Cache per-level tail sums incrementally instead of recomputing an `O(n)` dot
  product at every node.
- Carry the exact norm from the final scaled contribution when possible instead
  of recomputing `c G c^T` in `O(n²)` at every leaf; retain the direct norm as a
  differential oracle.
- Replace recursion with a state machine only if profiles show call overhead or
  stack traffic after tail caching.

### Relevant vectors

- Compute the maximum parity-representative norm in Gray-code order, updating
  the quadratic form in `O(n)` per flipped bit instead of calling `norm_sq` in
  `O(n²)` for all `2^n` masks.
- Store per-coset count and at most two candidate coordinate blocks in flat
  buffers; the characterization only needs to know exactly two opposites.
- Avoid allocating a `Vec` for every transient equal minimum.
- Keep the `2^n` dimension limit: the exponential state is mathematical, not an
  implementation accident.

### Construction A and nested lattices

- Add caller scratch to repeated `ConstructionA::contains` if membership checks
  appear in a hot workload; it currently allocates residues per call.
- For quantization, cache lifted symbols and reduce repeated integer/f64
  conversions. SIMD the cost table only for large `q`/batch cases with evidence.
- Let callers stream coset representatives instead of forcing `all_cosets` to
  clone every vector when the index is large. Materializing `all_cosets`
  remains intentionally output-proportional.
- Cache SNF/adjugate-derived prepared nesting state in `Nested`, which already
  represents a reusable pair; do not cache inside immutable `Gram` globally.

### Closed-form scalar code

- `A_n` selection already uses `select_nth_unstable`; do not restore full sort.
- Specialize fixed dimensions only when generated code removes measured bounds
  checks or branches. Const generics that merely duplicate monomorphizations are
  not a win.
- Mark tiny cross-crate methods `#[inline]` only after codegen shows a missed
  inlining boundary. Blanket `always` attributes increase code size and can
  degrade instruction-cache behavior.
- Use `sort_unstable` where ordering is total and stability carries no meaning.

### Build-level options

LTO, PGO, target-specific CPU flags, allocator choice, and codegen-unit tuning
belong to final binaries and benchmark profiles, not a library's Cargo release
profile. Record their effect for consumers, but do not make benchmark-only
profile settings look like an algorithmic library gain.

## Rejected or deferred shortcuts

| Idea | Decision |
| --- | --- |
| Depend on fplll/FLINT | Reject: contract, dependency, FFI, and reproducibility mismatch. Use as comparison and literature source only. |
| Floating-point Cholesky for reduction decisions | Reject: contradicts the exact certificate. Approximation may only suggest an exact check. |
| fplll `FAST`/heuristic result as oracle | Reject: no closest-point guarantee; the current corpus also found concrete upstream failures. |
| Extreme/probabilistic enumeration pruning | Reject for ML/list completeness. Could only exist under a separately named approximate API requested by consumers. |
| Return Barnes--Wall/Leech bounded-distance output from ML methods | Reject: different failure region. Use as candidate bound. |
| FMA or reassociation | Reject: changes decision bytes and cross-platform results. |
| Per-call thread pools / Rayon inside a decode | Defer: dimensions are small and deterministic budgets become harder. Parallelize independent words outside first. |
| General sparse `IntMatrix` | Reject: obscures dense small-matrix invariants. Sparse LDLC support already has its own representation. |
| Hidden memoization in `Gram` | Defer: increases every value's size and synchronization cost. Use explicit prepared objects. |
| Arbitrary precision | Reject by crate scope. Better algorithms may widen the accepted fixed-width domain but must still report checked bounds. |
| Hand-written assembly or `unsafe` intrinsics | Reject. `archmage` supplies safe tokenized SIMD. |
| Optimize constructors before hot paths | Defer unless profiles show repeated construction. Correctness and clear ownership beat cold microseconds. |

## Work packages and commit boundaries

Keep commits reviewable and independently measurable, following the existing
small-slice history:

1. benchmark counters/corpora only;
2. in-place transactional reduction state;
3. reusable GSO workspace;
4. exact size-reduction updater;
5. exact adjacent-swap updater;
6. deep-insertion updater and suffix arithmetic;
7. prepared real coefficients and enumeration differential oracle;
8. iterative zig-zag state machine;
9. square-root-free child stepping;
10. prepared reduced-basis CVP;
11. named-basis preconditioning;
12. Barnes--Wall candidate engine;
13. Leech candidate engine;
14. one exact-algebra replacement per operation;
15. one SIMD shape/backend plus its crossover measurement.

No commit combines an algorithm change with unrelated formatting, public docs,
or a second optimization whose effect cannot be separated.

## Verification matrix for every optimization

1. **Behavior:** existing unit, differential, property, interop, and end-to-end
   checks pass in default, all-feature, and no-default-feature configurations.
2. **Exact reduction:** independently recompute certificate, determinant, and
   unimodular transform; updater tests compare internal state after every
   elementary operation.
3. **Decoder:** compare point, squared distance, tie outcome, error atomicity,
   and node budget on frozen and randomized targets.
4. **SIMD:** scalar/dispatched bit identity across lane boundaries, ragged tails,
   odd dimensions, and backend overrides.
5. **Allocation:** warm paths remain zero-allocation; any new prepared scratch is
   warmed before counting.
6. **Performance:** interleaved, core-pinned release runs. Report geometry,
   corpus fingerprint, node/operation count, median latency, dispersion,
   compiler, CPU, features, and backend.
7. **Regression threshold:** do not land a default-path slowdown outside normal
   benchmark noise to win a niche case. Use a measured crossover or an explicit
   prepared API when workloads differ.
8. **Release proof:** `cargo run --release --example e8_awgn` still reproduces
   the shaping-gain gate.

## Default next action

Start with P0 counters around `reduce_with`, then P1.1 and P1.2. The evidence is
unambiguous: exact LLL is the only measured operation whose relative deficit
grows from 6x to 173x across the crate's core dimensions. Once GSO work per LLL
call is visible, implement exact incremental size-reduction and swap updates.
In parallel only after those contracts are stable, prototype P2.1--P2.3 against
the existing enumerator as the differential oracle.

Do not start with another SIMD kernel. The only measured SIMD opportunity is a
modest batch gain, while LLL and proof-tree size are orders-of-magnitude
problems and directly enable better named decoders.

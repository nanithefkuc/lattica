//! Acceptance tests: representation, named lattices, and the exact
//! short-vector enumeration.
//!
//! The determinants, minimal norms, kissing numbers and theta-series
//! coefficients asserted here are published properties of the classical
//! lattices. Nothing in `lattica` stores them: the named constructors build
//! Cartan matrices from Dynkin diagrams, and every constant below is recovered
//! by enumeration. A constructor that hardcoded a kissing number would make
//! this file circular.
//!
//! The pruned enumerator is additionally checked against a box search whose
//! completeness is proved rather than measured — see `box_enumerate`.

use lattica::basis::Gram;
use lattica::named::{a_n, a_n_basis, d_n, d_n_basis, e8, zn, zn_basis};
use lattica::shortvec::{DEFAULT_NODE_BUDGET, census, for_each_short};

/// Reference: Conway & Sloane, *Sphere Packings, Lattices and Groups*, 3rd ed.,
/// tables 4.1 and 6.1.
struct Published {
    det: i64,
    min_norm_sq: i64,
    kissing: u64,
}

fn expected_zn(n: usize) -> Published {
    Published {
        det: 1,
        min_norm_sq: 1,
        kissing: 2 * u64::try_from(n).unwrap(),
    }
}

fn expected_an(n: usize) -> Published {
    Published {
        det: i64::try_from(n).unwrap() + 1,
        min_norm_sq: 2,
        kissing: u64::try_from(n * (n + 1)).unwrap(),
    }
}

fn expected_dn(n: usize) -> Published {
    Published {
        det: 4,
        min_norm_sq: 2,
        kissing: u64::try_from(2 * n * (n - 1)).unwrap(),
    }
}

fn check(name: &str, g: &Gram<i64>, want: &Published) {
    assert_eq!(g.det().unwrap(), want.det, "{name}: determinant");
    let c = census(g, DEFAULT_NODE_BUDGET).unwrap();
    assert_eq!(
        c.min_norm_sq,
        Some(want.min_norm_sq),
        "{name}: minimal squared norm"
    );
    assert_eq!(c.kissing_number, want.kissing, "{name}: kissing number");
    assert!(g.is_positive_definite().unwrap(), "{name}: definiteness");
}

#[test]
fn the_constants_table_is_recovered_by_enumeration() {
    for n in 1..=10 {
        check(&alloc_name("Z", n), &zn(n).unwrap(), &expected_zn(n));
        check(&alloc_name("A", n), &a_n(n).unwrap(), &expected_an(n));
    }
    for n in 3..=10 {
        check(&alloc_name("D", n), &d_n(n).unwrap(), &expected_dn(n));
    }
    check(
        "E8",
        &e8().unwrap(),
        &Published {
            det: 1,
            min_norm_sq: 2,
            kissing: 240,
        },
    );
}

fn alloc_name(prefix: &str, n: usize) -> String {
    format!("{prefix}_{n}")
}

/// Number of vectors at each squared norm up to `radius`, indexed by norm.
fn theta_counts(g: &Gram<i64>, radius: i64) -> Vec<u64> {
    let mut counts = vec![0u64; usize::try_from(radius).unwrap() + 1];
    for_each_short(g, i128::from(radius), DEFAULT_NODE_BUDGET, |_, norm_sq| {
        counts[usize::try_from(norm_sq).unwrap()] += 1;
    })
    .unwrap();
    counts
}

#[test]
fn theta_series_coefficients_match_the_published_expansions() {
    // Theta_{E_8} = 1 + 240 q + 2160 q^2 + 6720 q^3 + ..., with q tracking
    // norm/2. Getting the kissing number right is easy; getting the second and
    // third shells right is what says the enumeration is complete.
    let counts = theta_counts(&e8().unwrap(), 6);
    assert_eq!(counts[2], 240);
    assert_eq!(counts[4], 2160);
    assert_eq!(counts[6], 6720);
    assert_eq!(counts[1], 0, "E_8 is an even lattice");
    assert_eq!(counts[3], 0, "E_8 is an even lattice");
    assert_eq!(counts[5], 0, "E_8 is an even lattice");

    // Jacobi's four-square theorem: r_4(m) = 8 * sum of divisors of m that are
    // not divisible by 4.
    let counts = theta_counts(&zn(4).unwrap(), 5);
    assert_eq!(&counts[1..=5], &[8, 24, 32, 24, 48]);

    // Theta_{D_4} = 1 + 24 q^2 + 24 q^4 + 96 q^6 + ...
    let counts = theta_counts(&d_n(4).unwrap(), 6);
    assert_eq!(counts[2], 24);
    assert_eq!(counts[4], 24);
    assert_eq!(counts[6], 96);

    // Theta_{A_2}, the hexagonal lattice: shells of 6 at norms 2, 6, 8.
    let counts = theta_counts(&a_n(2).unwrap(), 8);
    assert_eq!(&counts[1..=8], &[0, 6, 0, 0, 0, 6, 0, 6]);
}

/// A box search whose completeness is a proof, not an estimate.
///
/// If `c G cᵀ ≤ R` then by Cauchy–Schwarz in the `G` inner product,
/// `c_i² ≤ (G⁻¹)_ii · R`, and `G⁻¹ = adj(G)/det(G)`. So
/// `c_i² · det(G) ≤ R · adj(G)_ii` bounds every coordinate exactly, in
/// integers. Enumerating that box and filtering by exact norm therefore cannot
/// miss a vector. It is exponential, which is why it is an oracle and not the
/// implementation.
fn box_enumerate(g: &Gram<i128>, radius: i128) -> Vec<(Vec<i128>, i128)> {
    let n = g.dim();
    let det = g.det().unwrap();
    let adj = g.adjugate().unwrap();
    let bounds: Vec<i128> = (0..n)
        .map(|i| {
            let limit = radius * adj.entry(i, i) / det;
            i128::try_from(u128::try_from(limit).unwrap().isqrt()).unwrap()
        })
        .collect();

    let mut found = Vec::new();
    let mut c: Vec<i128> = bounds.iter().map(|&b| -b).collect();
    loop {
        if c.iter().any(|&v| v != 0) {
            let norm = g.norm_sq(&c).unwrap();
            if norm <= radius {
                found.push((c.clone(), norm));
            }
        }
        // Odometer over the box.
        let mut i = 0;
        while i < n {
            c[i] += 1;
            if c[i] <= bounds[i] {
                break;
            }
            c[i] = -bounds[i];
            i += 1;
        }
        if i == n {
            break;
        }
    }
    found.sort();
    found
}

fn widen(g: &Gram<i64>) -> Gram<i128> {
    let n = g.dim();
    let data: Vec<i128> = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| i128::from(g.entry(i, j)))
        .collect();
    Gram::from_rows(n, &data).unwrap()
}

fn pruned_enumerate(g: &Gram<i128>, radius: i128) -> Vec<(Vec<i128>, i128)> {
    let mut found = Vec::new();
    for_each_short(g, radius, DEFAULT_NODE_BUDGET, |c, norm| {
        found.push((c.to_vec(), norm));
    })
    .unwrap();
    found.sort();
    found
}

#[test]
fn pruned_enumeration_agrees_with_a_provably_complete_box_search() {
    let cases: Vec<Gram<i64>> = vec![
        zn(2).unwrap(),
        zn(3).unwrap(),
        zn(4).unwrap(),
        a_n(2).unwrap(),
        a_n(3).unwrap(),
        a_n(4).unwrap(),
        d_n(3).unwrap(),
        d_n(4).unwrap(),
    ];
    for g in &cases {
        let wide = widen(g);
        for radius in 1..=6 {
            assert_eq!(
                pruned_enumerate(&wide, radius),
                box_enumerate(&wide, radius),
                "dim {} at radius {radius}",
                g.dim()
            );
        }
    }
}

#[test]
fn pruned_enumeration_agrees_on_skewed_random_lattices() {
    // The Dynkin lattices are well conditioned by construction. A random
    // integral basis produces a Gram matrix with none of that structure, which
    // is where an enumerator's bounds are most likely to be wrong.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        i128::from(state % 7) - 3
    };
    for n in 2..=3usize {
        for _ in 0..12 {
            let data: Vec<i128> = (0..n * n).map(|_| next()).collect();
            let Ok(basis) = lattica::Basis::from_rows(n, n, &data) else {
                continue;
            };
            let Ok(g) = basis.gram() else { continue };
            if !g.is_positive_definite().unwrap_or(false) {
                continue;
            }
            for radius in 1..=8 {
                assert_eq!(
                    pruned_enumerate(&g, radius),
                    box_enumerate(&g, radius),
                    "gram {data:?} at radius {radius}"
                );
            }
        }
    }
}

#[test]
fn every_enumerated_vector_lies_in_the_ball_and_none_is_the_origin() {
    let g = e8().unwrap();
    for_each_short(&g, 4, DEFAULT_NODE_BUDGET, |c, norm| {
        assert!(c.iter().any(|&v| v != 0), "the origin was emitted");
        let narrow: Vec<i64> = c.iter().map(|&v| i64::try_from(v).unwrap()).collect();
        assert_eq!(g.norm_sq(&narrow).unwrap(), i64::try_from(norm).unwrap());
        assert!(norm > 0 && norm <= 4);
    })
    .unwrap();
}

#[test]
fn duality_identities_hold_exactly() {
    // The dual lattice has Gram matrix G* = adj(G)/det(G). Two identities pin
    // that down in integers: det(adj G) = det(G)^(n-1), which is the statement
    // det(L*) = 1/det(L); and adj(adj G) = det(G)^(n-2) * G, which is
    // dual-of-dual being the identity.
    let cases: Vec<Gram<i64>> = vec![
        zn(3).unwrap(),
        zn(5).unwrap(),
        a_n(2).unwrap(),
        a_n(4).unwrap(),
        a_n(5).unwrap(),
        d_n(4).unwrap(),
        d_n(5).unwrap(),
        e8().unwrap(),
    ];
    for g in &cases {
        let n = g.dim();
        let det = g.det().unwrap();
        let adj = g.adjugate().unwrap();

        // adj(G) * G == det(G) * I
        let product = adj.as_matrix().mul(g.as_matrix()).unwrap();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(product.get(i, j), if i == j { det } else { 0 });
            }
        }

        assert_eq!(
            adj.det().unwrap(),
            det.pow(u32::try_from(n - 1).unwrap()),
            "det(L*) = 1/det(L) at dim {n}"
        );

        let double = adj.adjugate().unwrap();
        let factor = det.pow(u32::try_from(n - 2).unwrap());
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    double.entry(i, j),
                    factor * g.entry(i, j),
                    "dual of dual at dim {n}"
                );
            }
        }
    }
}

#[test]
fn e8_is_self_dual() {
    // Unimodular, so adj(G) is literally G inverse, and E_8 is isomorphic to
    // its own dual: the inverse Cartan matrix is again even with determinant 1
    // and the same minimal norm and kissing number.
    let g = e8().unwrap();
    let dual = g.adjugate().unwrap();
    assert_eq!(g.det().unwrap(), 1);
    assert_eq!(dual.det().unwrap(), 1);
    let c = census(&dual, DEFAULT_NODE_BUDGET).unwrap();
    assert_eq!(c.min_norm_sq, Some(2));
    assert_eq!(c.kissing_number, 240);
}

#[test]
fn coding_gain_of_e8_is_exactly_two() {
    // gamma = d_min^2 / det^(1/n), so gamma = 2 means (d_min^2)^n == 2^n * det.
    // Exponentiating clears the root and keeps the check in integers.
    let g: Gram<i64> = e8().unwrap();
    let c = census(&g, DEFAULT_NODE_BUDGET).unwrap();
    let d = i128::from(c.min_norm_sq.unwrap());
    let det = i128::from(g.det().unwrap());
    assert_eq!(d.pow(8), 2i128.pow(8) * det);

    // gamma(D_4) = sqrt(2): (d^2)^4 == 4 * det, with det = 4.
    let g: Gram<i64> = d_n(4).unwrap();
    let c = census(&g, DEFAULT_NODE_BUDGET).unwrap();
    let d = i128::from(c.min_norm_sq.unwrap());
    assert_eq!(d.pow(4), 4 * i128::from(g.det().unwrap()));
}

#[test]
fn d3_and_a3_are_the_same_lattice() {
    // A classical coincidence of the Dynkin classification. The two
    // constructors share no code, so agreement across every invariant this
    // crate can compute is a genuine cross-check.
    let d3 = d_n(3).unwrap();
    let a3 = a_n(3).unwrap();
    assert_eq!(d3.det().unwrap(), a3.det().unwrap());
    assert_eq!(
        census(&d3, DEFAULT_NODE_BUDGET).unwrap().min_norm_sq,
        census(&a3, DEFAULT_NODE_BUDGET).unwrap().min_norm_sq
    );
    assert_eq!(
        census(&d3, DEFAULT_NODE_BUDGET).unwrap().kissing_number,
        census(&a3, DEFAULT_NODE_BUDGET).unwrap().kissing_number
    );
    assert_eq!(theta_counts(&d3, 8), theta_counts(&a3, 8));
}

#[test]
fn ambient_bases_and_dynkin_gram_matrices_are_two_routes_to_one_lattice() {
    for n in 1..=8 {
        assert_eq!(zn_basis::<i64>(n).unwrap().gram().unwrap(), zn(n).unwrap());
        assert_eq!(
            a_n_basis::<i64>(n).unwrap().gram().unwrap(),
            a_n(n).unwrap()
        );
        assert_eq!(a_n_basis::<i64>(n).unwrap().rank().unwrap(), n);
    }
    for n in 3..=8 {
        assert_eq!(
            d_n_basis::<i64>(n).unwrap().gram().unwrap(),
            d_n(n).unwrap()
        );
        assert_eq!(d_n_basis::<i64>(n).unwrap().rank().unwrap(), n);
        // D_n sits inside Z^n with index 2, so its ambient basis spans a
        // sublattice of determinant 4 = 2^2.
        assert_eq!(d_n::<i64>(n).unwrap().det().unwrap(), 4);
    }
}

#[test]
fn enumeration_cost_stays_modest_on_the_named_lattices() {
    // Not a benchmark -- a guard that the pruning works at all. A box search
    // over D_16 at radius 2 would visit on the order of 7^16 points.
    for n in [8usize, 12, 16] {
        let c = census(&d_n::<i64>(n).unwrap(), DEFAULT_NODE_BUDGET).unwrap();
        assert_eq!(c.kissing_number, u64::try_from(2 * n * (n - 1)).unwrap());
        assert!(c.nodes < 200_000, "D_{n} visited {} nodes", c.nodes);
    }
    let c = census(&e8::<i64>().unwrap(), DEFAULT_NODE_BUDGET).unwrap();
    assert!(c.nodes < 20_000, "E_8 visited {} nodes", c.nodes);
}

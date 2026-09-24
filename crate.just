# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to lattica belong in this file and are documented in `AGENTS.md`.

CRATE := 'lattica'
MSRV := '1.89'

# Backend tiers this crate dispatches over, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process. Empty means the crate owns no dispatch surface.
TIERS := 'v3 scalar'

# This crate owns no library unsafe; no Miri target is listed.
MIRI := ''

# Coverage exclusions for code that cannot execute on a CI host. Empty means
# every line of the crate counts.
COV_IGNORE := ''

# The fplll comparison's Rust side. It prints the matched CSV shape that the
# manually built C++ binary in `external/bench-fplll/` prints, so the two
# outputs diff directly; see `BENCHMARKS.md` for the pin and the C++ build.
[group('bench')]
bench-fplll:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "${FEC_GOLDEN_CORE:-}" ]]; then
        echo "pinned to core ${FEC_GOLDEN_CORE}"
        exec taskset -c "${FEC_GOLDEN_CORE}" cargo run --release --manifest-path external/bench-fplll/Cargo.toml
    else
        echo "warning: FEC_GOLDEN_CORE unset; the run is not core-pinned" >&2
        exec cargo run --release --manifest-path external/bench-fplll/Cargo.toml
    fi

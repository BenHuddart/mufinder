# NEB Direct Endpoint Seeding Plan

This plan tracks replacing the LST/QST transition-state-search step in the Energy
Barriers workflow with a Nudged Elastic Band (NEB) calculation seeded directly
from two relaxed muon-site endpoints. The goal is a simpler workflow and UI: the
user picks a reactant/product muon site pair and runs NEB, with no separate
transition-state search to configure, run, and parse first.

## Motivation

The original `Cu_diffusion` TS-search jobs were marked *failed* even though CASTEP
completed, because they ran `TSSEARCH_LSTQST_PROTOCOL : LSTMAXIMUM`, which writes
only `LST`/`PRO` blocks to the `.ts` file. MuFinder's `.ts` parser is NEB-native
(`REA`/`TST`/`PRO`), so it could not parse LST output and reported
"No complete transition-state path could be parsed". LSTMAXIMUM also produced
unphysical barriers (~7000 eV) because the straight-line path drives the muon
through a host Cu atom. Climbing-image NEB is the robust route for interstitial
muon hops, gives the barrier directly, and matches the parser we already have.

## Goals

- Remove the LST/QST transition-state-search step from the Energy Barriers workflow.
- Seed NEB directly from two relaxed endpoint muon sites (reactant + product).
- Reuse the existing path "sources" (endpoint pairs), re-optimised for NEB.
- Use CASTEP's built-in band interpolation; defer client-side IDPP pre-interpolation.
- Keep barrier extraction correct (eV) and well tested.

## Key references (not committed to the repo)

- `/Users/bhuddart/Source/castep_keywords.json` — NEB keyword names/defaults.
- `/Users/bhuddart/Source/CASTEP-26.11/Source/Functional/tssearch_neb.f90` — `.ts`
  output format (`TSTYPE` header, `REA`/`PRO` once, `TST` per iteration; Hartree/Bohr).

CASTEP facts that drive the design:

- `TSSEARCH_METHOD` ∈ `{LSTQST, conNEB, NEB}` (default `LSTQST`). Use `NEB`.
- `TSSEARCH_NEB_CLIMBING : TRUE` is needed to land on the saddle (the barrier).
- The NEB **image count is `TSSEARCH_MAX_PATH_POINTS`**, and it must be odd when
  climbing (CASTEP increments by one otherwise).
- NEB shape: `TSSEARCH_NEB_METHOD` (ODE12R), `TSSEARCH_NEB_SPRING_CONSTANT`
  (0.1 eV/Ang^2), `TSSEARCH_NEB_TANGENT_MODE` (SPLINE), `TSSEARCH_NEB_MAX_ITER` (20).
- NEB convergence: `TSSEARCH_ENERGY_TOL`, `TSSEARCH_FORCE_TOL`, `TSSEARCH_DISP_TOL`.

## Phase 0: Parser correctness

Status: **done.**

Implemented:

- `castep_io/ts_parser.py` converts `.ts` energies (Hartree) to eV at parse time, so
  `TransitionStateImage.energy`, `TransitionStatePath.barrier_height_eV`, and the
  `image_energies_eV` payload in `main_window` are all consistently in eV. This fixes
  a latent ~27.2x unit error in the reported barrier.
- `tests/fixtures/generate_neb_ts.py` generates a format-faithful multi-iteration NEB
  `.ts` (matching the CASTEP Fortran write formats); `tests/fixtures/neb_path.ts` is the
  committed fixture.
- `tests/test_castep_ts_parser.py` gained: a multi-iteration NEB band test (asserts the
  converged final-iteration band and a ~0.12 eV barrier) and a test that LST/QST output
  is rejected.

## Phase 1: Workflow / param layer (`workflow/energy_barriers.py`)

Status: **done** (retiring the LST/QST expansion path is deferred to Phase 2/3 cleanup).

Implemented:

- `_nudged_elastic_band_param_template` emits `TSSEARCH_METHOD : NEB` and
  `TSSEARCH_NEB_CLIMBING : TRUE` by default; image count / tolerances come from config.
- Confirmed the cell layer already omits the intermediate block when none is supplied
  (CASTEP interpolates the band).
- Test: `test_expand_nudged_elastic_band_jobs_without_intermediate_seeds_from_endpoints`.

Next:

- Add an odd-image-count default for `TSSEARCH_MAX_PATH_POINTS`; retire the LST/QST
  expansion path (`expand_transition_state_search_jobs`, `_cell_template`, TS param template)
  once nothing references it.

## Phase 2: App orchestration (`app/main_window.py`)

Status: **core done** (candidate decoupling + materialization); keyword-set/source
optimisation and TS-handler removal still pending.

Implemented:

- `_EnergyBarrierNebCandidate` TS fields are optional; `_energy_barrier_neb_source_candidates`
  builds one candidate per endpoint path source and attaches a TS result only if one exists.
- `_materialize_energy_barrier_neb_jobs` no longer requires a `.ts` file or intermediate; it
  seeds NEB from endpoints and uses a TS intermediate only when present.
- `_serialize_energy_barrier_neb_sources` and the NEB parameter dialog intro no longer assume
  a TS job/run/result.
- Tests: existing TS-seeded materialization still passes; new
  `test_main_window_materializes_neb_jobs_directly_from_endpoints` covers endpoint-only seeding
  (product block, no intermediate block).

Next:

- Make the Energy Barriers definition an NEB set (`method="NEB"`; fix the
  `setdefault("method", "LSTQST")` defaults).
- Optimise reused sources for NEB: strip LST/QST-only config
  (`tssearch_lstqst_protocol`, `tssearch_qst_max_iter`, `tssearch_cg_max_iter`); move
  `tssearch_max_path_points` (as image count, enforced odd when climbing) and the NEB
  tolerances into `_ENERGY_BARRIER_NEB_KEYWORD_NAMES`.
- Relax the path-enumeration `max_path_length_ang` default: the straight-line-through-an-atom
  failure mode is gone, so longer/curved hops are valid; optionally scale image count with
  path length.
- Remove the TS submit handlers, TS candidate logic, and `_ENERGY_BARRIER_TSS_KEYWORD_NAMES`.

## Phase 3: UI (`app/energy_barriers_widget.py`)

Status: not started.

Plan:

- Remove the TS-search section. Make "Generate paths -> set up NEB -> submit (local/cluster)"
  the single flow on endpoint pairs (reuse the existing `submit_neb_*_requested(keys)` signals).
- Surface NEB-relevant params (image count, climbing, spring constant, tangent mode,
  tolerances); drop the LST/QST controls.

## Phase 4: Tests

Status: parser tests done (Phase 0); workflow/app tests pending.

Plan:

- `tests/test_workflow_energy_barriers.py`: NEB expansion with no intermediate emits a valid
  `NEB` `.cell`/`.param` (product block, no intermediate block, odd `MAX_PATH_POINTS`).
- App test: an endpoint candidate with no TS result yields a submittable NEB config.

## Out of scope (future work)

- Client-side IDPP / custom intermediate pre-interpolation.
- `conNEB` (constrained NEB) support.

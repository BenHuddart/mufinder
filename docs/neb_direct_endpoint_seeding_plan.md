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
- Synthesise the CASTEP-required INTERMEDIATE structure ourselves via **IDPP** (with a linear
  minimum-image midpoint as the fallback).

> **Correction (CASTEP NEB requires three structures).** The earlier assumption that CASTEP
> builds the band from the two endpoints alone is wrong. `tssearch.f90:307-314` makes the NEB
> path *unconditionally* read REACTANT, PRODUCT **and** INTERMEDIATE; a cell without the
> intermediate aborts with `cell_read_coordinates - one of POSITION_FRAC_INTERMEDIATE or
> POSITION_ABS_INTERMEDIATE must be present`. CASTEP places the intermediate at the central
> image and linearly interpolates reactant->intermediate->product (`tssearch_neb.f90:1497-1513`).
> We therefore supply a single central image as the intermediate.
> The "no separate LST/QST search" goal still holds: we compute the intermediate, we don't search for it.

> **IDPP seeding (implemented).** Because CASTEP only consumes the *central* image (re-interpolating
> each half-band linearly), a plain linear midpoint can drop the muon onto a host atom — on the Cu
> diffusion project the Oct→Oct path2 midpoint landed 0.038 Å from a Cu, giving a 191 eV/Å initial
> force and a NEB that couldn't converge in the walltime; two Oct→Tet paths plateaued just above the
> force tol because their midpoints started close to a Cu. We now seed the intermediate with IDPP
> (Smidstrup et al., JCP 140, 214106 (2014)) via `science/energy_barriers.idpp_intermediate_structure`:
> build a small odd-length band, `NEB.interpolate(mic=True)` then `idpp_interpolate(..., mic=True)`,
> and take the rc=0.5 image. It falls back to the linear midpoint on any failure or if the optimiser
> would reduce the muon's clearance from the host. Measured clearances on the real failing paths:
> Oct→Oct path2 0.038 → 1.43 Å, Oct→Tet path2 0.81 → 1.46 Å, Oct→Tet path3 1.51 → 1.85 Å; already-clean
> paths are left unchanged. IDPP fixes the catastrophic seed; the marginal long-hop plateaus may still
> want `TSSEARCH_NEB_MAX_ITER` raised from 20 (separate follow-up).
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

Status: **data layer done**; UI-side TS-search removal (submit handlers, source
configure dialog, `_ENERGY_BARRIER_TSS_KEYWORD_NAMES`) deferred to Phase 3.

Implemented:

- `_EnergyBarrierNebCandidate` now carries only `key` + `source`; `_energy_barrier_neb_source_candidates`
  builds one candidate per endpoint path source (no TS dependency).
- `_materialize_energy_barrier_neb_jobs` seeds NEB directly from endpoints with no
  intermediate (CASTEP interpolates). The TS-seeded intermediate path and its plumbing
  (`_serialize_energy_barrier_neb_sources` TS ids, the dead `_transition_state_intermediate_image`
  and `_position_block_from_transition_state_image` helpers) were removed — this also eliminated
  a latent Bohr-vs-Angstrom unit bug in the intermediate seed.
- `tssearch_max_path_points` (the NEB image count) moved into `_ENERGY_BARRIER_NEB_KEYWORD_NAMES`;
  the NEB param template emits a default odd image count plus climbing-image.
- Climbing image is **locked on** (a barrier needs the top image on the saddle): `tssearch_neb_climbing`
  was removed from the editable surface and added to the NEB dialog's locked_values. The image
  count defaults to **7** (`_ENERGY_BARRIER_NEB_DEFAULT_IMAGES`) and any even user value is rounded
  up to the next odd in `_energy_barrier_neb_row_configuration_with_source_defaults` (CASTEP requires
  an odd count under climbing).
- Tests: removed the obsolete TS-seeded materialization test; added
  `test_main_window_materializes_neb_jobs_directly_from_endpoints` (product block, no
  intermediate). All energy-barrier/NEB GUI tests pass.

Next (mostly Phase 3 / UI-coupled):

- Relax the path-enumeration `max_path_length_ang` default now that straight-line overlap
  is no longer a failure mode; optionally scale image count with path length.
- Remove the TS submit handlers (`_on_submit_*_from_energy_barriers`), the source-configure
  TS dialog, and `_ENERGY_BARRIER_TSS_KEYWORD_NAMES` as part of the widget teardown.
- (Deeper) promote the energy-barriers definition to a first-class NEB set rather than reusing
  `castep_transition_state_search_set`.

## Phase 3: UI (`app/energy_barriers_widget.py`)

Status: **in progress** — TS-search submission removed and terminology rebranded to NEB;
table/column consolidation still pending.

Implemented:

- Removed the TS-search job submission from the Paths tab ("Run Local/Cluster Calculations"
  buttons, their emit methods, and the `_on_submit_*_from_energy_barriers` handlers + signal
  connections in `main_window`). Users can no longer launch a transition-state search.
- Rebranded the UI: the Paths tab's "Diffusion Paths" list (generate/preview/delete), the
  middle tab renamed "Transition States" -> "NEB Setup" with NEB-focused group titles, hints,
  and intro; the results column header "Job" -> "Path".
- Verified: 28 energy-barrier GUI tests pass; the widget constructs with tabs
  `['Paths', 'NEB Setup', 'NEB Analysis']` and the NEB configure/submit controls intact.

Implemented (continued):

- Dropped the vestigial `Barrier`/`Images`/`TS File` columns from the NEB Setup table
  (now `Path | NEB Status | Parameters`), along with the matching `EnergyBarrierResultRow`
  fields; updated the widget tests' cellWidget indices.
- Confirmed the path-enumeration `max_path_length_ang` is already uncapped when the field is
  blank, so no default needs relaxing.

Implemented (continued):

- Consolidated the two path tables into a single "Diffusion Paths" table on the Paths tab
  (Reactant / Product / Charge / Path Length / NEB Status / Parameters / View) with the NEB
  configure + build-local/cluster + delete actions; removed the separate "NEB Setup" tab and
  the `EnergyBarrierResultRow` / `set_results` / `_fill_result_table` machinery.
- Repointed `_energy_barrier_neb_source_candidates` at the generated-path union
  (`_energy_barrier_generated_sources`) so NEB no longer depends on the TS-search-set
  definition being populated — a generated path is immediately NEB-ready. The merged table's
  per-row + batch configure now drives NEB parameters (`configure_result_requested`).

Implemented (continued):

- UI polish: dropped the "transition-state" qualifiers from the subtitle/intro/hint, and made
  the workflow tab bar/widget background transparent so the tab strip matches the window.
- Dead-code sweep: removed the now-unused TS pipeline — `_on_configure_energy_barrier_source`,
  `_materialize_energy_barriers_jobs`, `_jobs_for_energy_barrier_source_keys`,
  `_build_energy_barriers_states`, `_ensure_energy_barriers_definition`,
  `_serialize_energy_barrier_sources`, `_energy_barrier_source_parameter_specs`,
  `_energy_barrier_row_configuration_with_source_defaults`, `_active_energy_barriers_definition`,
  `_job_for_energy_barrier_source_key`, `_ENERGY_BARRIER_TSS_KEYWORD_NAMES` (main_window); and
  the `expand_transition_state_search_jobs` / `_cell_template` / `_param_template` /
  `TransitionStateSearchSource` / `TransitionStateSearchJobConfig` workflow API. Kept
  `_transition_state_param_template` (the NEB template delegates to it) and the
  `castep_transition_state_search*` calc-type constants (still used for backward-compatible
  reads of old projects). TS-centric tests were deleted or repointed to the NEB pipeline.

Parameter inheritance: NEB jobs inherit the reactant endpoint's relaxation settings (kpoint
grid + cut-off) on top of the project convergence defaults, via
`_energy_barrier_neb_row_configuration_with_source_defaults` (precedence: project defaults <
reactant relaxation settings < user config). This applies to both the materialized job and the
NEB configure dialog. Covered by
`test_main_window_neb_inherits_kpoints_from_reactant_relaxation_cell_file`.

Next:

- Full visual verification by launching the app on a real project.

## Phase 4: Tests

Status: parser tests done (Phase 0); workflow/app tests pending.

Plan:

- `tests/test_workflow_energy_barriers.py`: NEB expansion with no intermediate emits a valid
  `NEB` `.cell`/`.param` (product block, no intermediate block, odd `MAX_PATH_POINTS`).
- App test: an endpoint candidate with no TS result yields a submittable NEB config.

## Out of scope (future work)

- Client-side IDPP / custom intermediate pre-interpolation.
- `conNEB` (constrained NEB) support.

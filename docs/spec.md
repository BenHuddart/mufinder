# MuFinder Product and Architecture Specification

- Spec ID: MUFINDER-SPEC
- Spec Version: 1.15.0
- Last Updated: 2026-05-09
- Status: Active
- Scope: PySide6 desktop rebuild

## Version History

| Version | Date | Summary |
|---|---|---|
| 1.0.0 | 2026-04-05 | Initial full rebuild specification authored. |
| 1.1.0 | 2026-04-05 | Added mandatory quality gates: pytest coverage and Sphinx warning-free docs build. |
| 1.2.0 | 2026-04-05 | Added CIF/CELL/MCIF import requirements and MCIF magnetic-structure extraction behavior. |
| 1.3.0 | 2026-04-05 | Added repository-specific spec-sync pass policy and authority/override rules for user directives. |
| 1.4.0 | 2026-04-06 | Refined convergence UX requirements: pre-submission ready status, cutoff-value locking in ready/progressed states, and higher-precision convergence metric display. |
| 1.5.0 | 2026-04-07 | Added host-structure magnetic editing and visualization requirements (spin column, propagation vector, supercell preview) and spin-polarized POSITIONS_FRAC SPIN serialization behavior. |
| 1.6.0 | 2026-04-07 | Added remote SSH/SLURM execution requirements: persisted cluster setup, template-based submission scripts, queue polling, connection indicators, remote job-id tracking, STOPPED lifecycle state, and auto retrieval of completion outputs. |
| 1.7.0 | 2026-04-07 | Added Redwood/Glamdring addqueue submission support: Redwood preset, addqueue runtime controls (partition/memory/comment), optional walltime for addqueue clusters, configurable CASTEP executable, and remote command-path probe with fallback overrides for scheduler commands. |
| 1.8.0 | 2026-04-07 | Added HPC connectivity security and UX hardening requirements: TOFU SSH host-key trust, host-key mismatch blocking, remote job-label safety validation, SFTP-based recursive deletion, and actionable cluster failure messaging including queue-polling failures. |
| 1.9.0 | 2026-04-07 | Added workflow-scoped simulation-cell architecture: saved supercell assets as `Structure` entities, per-workflow selection persistence in `state.json`, convergence snapshot freezing of resolved structure ids, cross-workflow clone-on-load behavior, and legacy stale-spin repair requirements for saved supercells. |
| 1.10.0 | 2026-04-08 | Added a dedicated parent Convergence Testing overview page with notebook-style pinned cutoff/k-point plots, adoption-driven snapshot persistence, child-workflow action cards, and stale-snapshot fallback behavior on host/default changes. |
| 1.11.0 | 2026-04-08 | Added startup responsiveness requirements: splash screen must be shown before heavyweight startup imports/initialization, with deferred feature imports after splash exposure. |
| 1.12.0 | 2026-04-10 | Added pristine-relaxation requirements for per-row supercell/fixed-cell controls, project-default cutoff+k-point propagation, equivalent-density k-point scaling, `-out.cell` parsing, and geometry-convergence completion gating. |
| 1.13.0 | 2026-04-29 | Expanded MCIF magnetic import requirements to preserve atom labels, multiple propagation vectors, label-to-atom mappings, and phase-corrected magnetic-model promotion. |
| 1.14.0 | 2026-05-09 | Refined Energy Barriers UX to require explicit initial/final site selection before path generation and added a plotted NEB reaction-coordinate profile in the inline inspector. |
| 1.15.0 | 2026-05-09 | Split the Energy Barriers workflow across dedicated tabs for path generation, transition-state jobs, and NEB analysis to improve usability on smaller displays. |

## 1. Executive Summary

MuFinder is a scientific desktop application for muon stopping-site workflows centered on CASTEP and scientific Python tools. The rebuild is a fresh product architecture in PySide6, not a framework port.

The new system is designed around:

- project-centric organization
- full save/load persistence
- structured provenance for reproducibility
- centralized job management for local and HPC runs
- explicit staged scientific workflows
- clean architecture boundaries
- extension-ready domain and execution models

## 2. Product Goals

1. Use a project as the unit of scientific work for a material.
2. Support resumable work across app restarts with full relevant state persistence.
3. Track reproducibility through structured provenance, not console text only.
4. Manage all jobs at the project level with lifecycle-aware filtering.
5. Make scientific stage dependencies explicit in product navigation.
6. Keep domain logic, persistence, orchestration, and UI clearly separated.
7. Maintain growth paths for new workflows and execution backends.

## 3. Non-Goals (Current Major Release)

- Multi-user collaborative editing of one project.
- Cloud-native orchestration backends beyond local and SSH HPC.
- Full support for DFT engines beyond CASTEP.
- Web UI replacement for desktop app.
- Public plugin API.

## 4. User Workflow Model

Primary staged workflow:

1. Project setup and host structure import
2. Convergence testing
3. Optional pristine host relaxation
4. Muon-site generation and relaxation
5. Site analysis and clustering to confirmed sites
6. Downstream workflows consuming confirmed sites:
   - magnetic-field calculations
   - quantum corrections
   - energy barriers

Dependency policy:

- Downstream workflows consume confirmed muon sites as durable project entities.
- Convergence outputs can prefill default calculation parameters.
- Pristine relaxation output may become active host basis for later stages.

## 5. Information Architecture and Navigation

Main shell:

- Left: Project navigator tree by stage
- Center: Stage/workflow panel (stacked views)
- Bottom dock: Global job dashboard

Navigation principles:

- Project-wide views are always available regardless of selected stage.
- Stage panels must surface prerequisites and dependency gaps clearly.

## 6. Project / Workspace Model

A project is a directory ending in `.mufinder` that stores all scientific and application state for one material.

Project contains:

- structures, calculations, runs, jobs, results
- muon sites and clusters
- provenance records
- UI state snapshot
- imported raw source files and generated run artifacts

## 7. Domain and Data Model

Core entities:

- Project
- Structure
- WorkflowStage
- CalculationDefinition
- CalculationRun
- Job
- Result
- MuonSite
- MuonSiteCluster
- ProvenanceRecord
- ExecutionTarget (application-level profile, referenced by project jobs)

Project currently also stores:

- host_structure_id
- magnetic_structure (project-level magnetic metadata, including MCIF-derived content)

Key linking model:

- UUID-based references across all entities.
- Confirmed MuonSite entities are first-class and consumed by downstream workflows.

## 8. Persistence Model

Project storage format:

- `project.json` manifest
- `entities/<type>/<uuid>.json` for entity objects
- `state.json` for UI state
- `provenance.jsonl` append-only provenance stream
- `imports/` for copied source inputs
- `files/` for run input/output artifacts

Persistence requirements:

- Save/load projects explicitly.
- Persist workflow-relevant UI state where appropriate.
- Preserve provenance and source files needed for reproducibility.
- Validate cross-entity references before save.
- Persist per-workflow simulation-cell selection state in `state.json` under `workflow_supercells`.
- Persist saved supercell assets as `Structure` entities with `properties.supercell_asset` metadata rather than introducing a separate entity type.

## 9. Job Model and Execution Architecture

Job lifecycle states:

- defined
- submitting
- queued
- running
- retrieving
- stopped
- completed
- failed
- cancelled

Execution backends:

- local backend for workstation execution
- HPC backend for SSH + scheduler orchestration

Remote HPC requirements:

- Cluster profiles are persisted globally and selectable at submission time.
- SSH key auth is primary; password fallback is supported for compatibility.
- Unknown SSH host keys must require explicit user trust confirmation before first use (TOFU flow).
- Trusted SSH host keys must be persisted in user-scoped settings keyed by cluster endpoint; host-key mismatch must hard-fail with explicit recovery guidance.
- Scheduler abstraction must support SLURM script submission and addqueue command submission.
- Submission scripts are generated from user templates with runtime variable substitution for script-based schedulers.
- Redwood/Glamdring support must include a built-in preset profile.
- addqueue clusters must expose partition, memory-per-process, and comment runtime controls.
- addqueue clusters must allow blank walltime; script-based SLURM clusters require walltime.
- Cluster profiles must support configurable CASTEP executable paths.
- Scheduler command resolution must probe the non-interactive SSH shell first and then use configured absolute command-path overrides when needed.
- Dashboard tracks cluster name and remote scheduler job id alongside status.
- Queue monitoring supports periodic polling and manual refresh.
- Connection state is surfaced in UI via status indicator and last-checked timestamp.
- Queue polling transport failures must surface an explicit user-visible error with troubleshooting guidance, in addition to connection-state updates.
- Completed remote jobs trigger automatic output retrieval for workflow parsing.
- Walltime-stopped jobs map to stopped and are resettable for direct resubmission.
- Remote execution labels used in scheduler commands and remote paths must be validated against a strict safe character set before submission.
- Recursive remote directory deletion for cluster job workspaces must use SFTP traversal rather than shelling out to `rm -rf`.
- Cluster-facing connection/submission/cancellation/refresh failures must present concise summaries with expandable details and actionable recovery suggestions.

Job system requirements:

- centralized registry and dashboard
- typed tags for stage and calculation type
- lifecycle transitions with strict validation
- restart/reload synchronization behavior

## 10. Provenance, Logging, and Reproducibility

Provenance is structured and queryable.

Each significant action records:

- action type
- stage context
- parameters snapshot
- input and output entity links
- associated job where relevant
- timestamp and actor

Provenance must support reconstruction of what was run, with which inputs and options.

Journal presentation requirements:

- the `provenance_log` workflow route must surface as a user-facing `Journal` page
- the Journal must be derived from persisted `provenance.jsonl` records rather than inferred job history
- Journal entries must be grouped by local calendar day with human-readable date headers and no precise times
- the default presentation must be curated and narrative-focused rather than exposing a raw append-only ledger
- the curated Journal must include major workflow actions, calculation submission milestones, comparison-table creation, and terminal calculation outcomes
- intermediate runtime states such as queued, running, and retrieving must not be journalized by default
- the Journal UI must follow MuFinder branding as a notebook-style expedition record, using existing palette and typography tokens
- completion and warning/failure milestones may use stamp-style affordances to reinforce workflow progress
- Journal export must support plain text by default and Markdown as a secondary format
- export output must mirror the curated Journal rather than dumping raw JSON or JSONL provenance rows

## 11. UI Architecture

Recommended architecture:

- MVVM-like separation with service layer
- views are thin and delegate to services
- services own orchestration and validation
- domain layer stays GUI-independent

Concurrency requirements:

- long-running operations must not block UI
- background execution and thread-safe signaling for updates

Startup responsiveness requirements:

- application startup must display the splash screen as early as practical in the Qt boot path
- heavyweight feature-module imports and non-critical initialization must be deferred until after the splash has been shown/exposed
- new startup functionality must preserve this ordering to avoid pre-splash stalls and regressions in first-visible feedback

## 12. Workflow-Specific Design Requirements

### 12.1 Convergence Testing

- the parent `Convergence Testing` workflow node must open a dedicated overview page rather than falling back to the home screen
- the parent workflow completion state remains satisfied only when both project default cutoff and project default k-point grid are set
- the overview page must explain that both plane-wave cutoff energy and reciprocal-space sampling must be converged before later workflows are trusted
- the overview page must show two horizontally arranged notebook-style pinned regions:

  - cutoff convergence on the left
  - k-point convergence on the right

- each pinned region must include:

  - a short explanation of the corresponding subtask
  - a summary of the currently stored project defaults that mirrors the child workflow formatting
  - a direct action-card/button that opens the corresponding child workflow page

- pinned plots must be updated only when the user adopts a project default from the corresponding convergence analysis dialog
- overview snapshots must persist in `Document.ui_state["convergence_overview_snapshots"]` separately from the existing examiner-dialog restore state
- each persisted overview snapshot must record:

  - `host_structure_id`
  - `simulation_cell_structure_id`
  - adopted `default_value`
  - convergence `method`
  - `energy_threshold`
  - `force_threshold`
  - plot payload derived from examiner `plot_data`
  - k-point point-label metadata when needed to redraw grid labels

- overview snapshots must render only when they match the active host structure and the current recommended default for that workflow
- if no valid adopted snapshot exists, the overview page must fall back to a dotted placeholder plus the current project-default summary without blocking the child workflows
- setup, run, analysis stages
- recommended cutoff and k-point outputs
- persisted convergence datasets and plots metadata
- workflow-aware CASTEP parameter entry with curated basic fields and advanced disclosure
- typed controls and parameter help dialogs backed by keyword metadata
- cutoff sweep builder supporting both manual lists and start/stop/step range generation
- optional subset overrides that apply alternate parameter values to selected cutoff rows
- configured cutoff rows must be represented as ``ready`` before submission-time materialization
- cutoff-energy values must become read-only once a row is ``ready`` or in any later runtime state
- per-row total energy and max-force displays must preserve sufficient precision for small-difference comparisons

### 12.2 Pristine Relaxation

- optional host-only geometry relaxation
- relaxed structure output persisted as first-class structure
- workflow UI must support multiple user-labeled rows in one configuration snapshot
- each row must support its own simulation cell choice (unit cell or explicit 3x3 supercell matrix)
- each row must expose a fixed-cell control that serializes to `FIX_ALL_CELL : TRUE/FALSE`
- workflow defaults must include project recommended cutoff energy (when set)
- workflow defaults must include project recommended k-point grid (when set)
- when a row uses a supercell, its recommended k-point grid must be recomputed to preserve equivalent reciprocal-space density from the recommendation reference cell
- project k-point recommendation persistence must store the lattice used to derive that recommendation so later equivalent-density scaling is deterministic
- successful completion handling must retrieve and parse both `.castep` and `-out.cell` outputs
- pristine-relaxation runs must not be treated as successful when geometry optimization convergence is not achieved
- parsed relaxed lattices must expose `a`, `b`, `c` values in the workflow table and result payload

### 12.3 Muon Site Generation and Relaxation

Generation methods:

- random constrained filling, either by direct muon-muon separation or by a
  target number of initial positions with automatic spacing estimation
- near-element placement
- manual site entry (fractional or Cartesian)
- UEP cluster source rows from saved UEP analysis state

Current implemented generation constraints:

- periodic minimum-image muon-muon distance
- periodic minimum-image muon-atom distance
- attempt-cap controlled sampling
- symmetry-equivalent random-position reservation
- internal deterministic seed support for regression tests

Relaxation staging:

- selected generated positions must prompt for positive muon, muonium, or both
  charge states before rows are staged
- the relaxation table must present charge states as separate positive-muon and
  muonium tabs rather than a per-row charge column
- CASTEP total charge remains workflow-controlled: `+1` for positive muon and
  `0` for muonium

### 12.4 Site Analysis and Clustering

- extract relaxed sites and energies
- filtering by displacement and quality constraints
- clustering methods and representative-site selection
- promotion to confirmed muon sites

### 12.5 Magnetic-Field Workflows

- consume confirmed muon sites
- support vector, magnitude, and tensor modes
- include Van Vleck paths

### 12.8 Host Structure Magnetic Editing and Preview

- host-structure view must support per-atom scalar spin entry (units: hbar/2)
- spin-arrow display should default to disabled for newly loaded non-magnetic hosts and should auto-enable when any per-atom spin is set non-zero
- host-structure view must provide propagation-vector entry in reciprocal-lattice units
- host-structure visualization must render clearly visible 3D arrows for spins, with signed styling for opposite directions
- host-structure panel should default to an expanded viewer area and allow user resizing between metadata table and viewer
- current-structure metadata should be arranged compactly (two-row summary) to preserve horizontal viewer area
- left-clicking empty space in the 3D viewer should clear current atom selection
- host-structure supercell preview must provide:
   - a simple diagonal matrix mode (Nx, Ny, Nz)
   - an advanced full 3x3 integer matrix mode
   - read-only visual verification of magnetic pattern propagation across generated supercells
- host-structure supercell preview has been promoted to a Supercell Editor that can save, name, edit, delete, and select reusable project supercells
- matrix edits to saved supercells already frozen into configured calculations must create a future-use copy rather than rewriting the frozen calculation structure

### 12.9 Workflow-Scoped Simulation Cell Selection and Supercell Assets

- each workflow stage must resolve its own effective simulation cell independently
- workflow simulation-cell mode must support:
  - `unit_cell` (active host structure)
  - `saved_supercell` (saved derived structure snapshot)
- workflow selection state must persist under `state.json.workflow_supercells`
- saved supercell assets must be represented as regular `Structure` entities with:
  - `parent_structure_id` equal to the active host structure id
  - `properties.supercell_asset.matrix`
  - `properties.supercell_asset.owner_workflow_stage`
  - `properties.supercell_asset.source_asset_structure_id` (nullable clone provenance)
- saved supercell assets must be reusable across workflows for the same host structure
- saved supercell assets must support user-facing names that can be changed in the Supercell Editor
- selecting a saved supercell owned by a different workflow must perform clone-on-load into the current workflow scope, with deduplication when an identical current-workflow host+matrix asset already exists
- convergence configuration persistence must freeze the resolved simulation cell by storing:
  - the resolved structure id in `CalculationDefinition.input_structure_ids[0]`
  - lightweight provenance in calculation parameters (for example `simulation_cell_mode` and `supercell_source_workflow`)
- generated workflow `.cell` files must materialize from the resolved/frozen structure snapshot, not from global host-preview state
- `Project.supercell_matrix` must remain compatibility/read-only for host-preview defaults and must not control workflow calculation behavior
- invalid/missing/mismatched-host saved selections must fall back safely to unit-cell mode
- legacy stale all-zero supercell spin arrays must be repairable from host magnetic metadata so spin-polarized workflows continue to serialize expected `SPIN=` tokens

### 12.6 Quantum Corrections

- quantum phonon calculations must start only from completed structural muon relaxation results
- UEP candidates, generated sites, pristine relaxations, failed jobs, and manual sites without structural-relaxation provenance are not valid sources
- CASTEP phonon jobs are Gamma-point only and must write `PHONON_KPOINT_MP_GRID 1 1 1`
- generated inputs must include `TASK : PHONON`, a selected `PHONON_METHOD`, and explicit `H:mu` mass/gamma cell blocks
- finite displacement is the default method; DFPT/linear-response workflows must reject known ultrasoft pseudopotentials and warn for unknown pseudopotential family metadata
- `.phonon` files are expected outputs for quantum phonon jobs and must be retrieved from cluster runs
- parsed results must persist selected muon-dominated modes, per-mode and total ZPE, witness metrics, output files, and source relaxation provenance
- single-particle quantum approximations should be implemented as a later analysis layer using directional effective potentials; PIMD is out of scope

### 12.7 Energy Barriers

- path generation is explicit and pair-scoped: users select initial and final confirmed relaxed sites, then generate distinct paths for that pair on demand
- TSS and NEB support
- the workflow is split across dedicated tabs for path generation, transition-state results, and NEB analysis so the interface remains usable on smaller displays
- completed NEB jobs expose an inline path inspector with image-by-image geometry preview in the shared structure viewer
- the inline NEB inspector includes a plotted reaction-coordinate profile synchronized with the image table, slider, and structure viewer
- persisted profiles and export paths

## 13. Directory and Storage Strategy

Canonical structure (abridged):

- `project.json`
- `state.json`
- `provenance.jsonl`
- `entities/structures|sites|calculations|runs|jobs|results/...`
- `imports/`
- `files/`
- `exports/`

## 14. Migration Strategy

Migration goals from legacy conceptual model:

- preserve scientific scope
- improve project cohesion and traceability
- move from ad hoc file operations to explicit project entities

Compatibility principles:

- preserve key scientific parameterization and output semantics
- prefer behavior continuity where it affects scientific validity

## 15. Risks, Assumptions, Open Questions

High-value risks:

- parser variation across CASTEP/MCIF ecosystem inputs
- HPC connection instability and recovery complexity
- scale limits of entity-per-file storage as project size grows

Assumptions:

- desktop single-user operation
- CASTEP primary backend in current release window

Open questions include:

- advanced magnetic metadata normalization from diverse MCIF dialects
- future data-store evolution path if entity-file scale becomes bottleneck

## 16. Implementation Roadmap

Phase 1 (foundation):

- project shell, persistence, core entities, generation flow, centralized jobs

Phase 2:

- convergence + pristine workflows and robust HPC submission lifecycle

Phase 3:

- downstream workflows: magnetic fields, quantum corrections, barriers

Phase 4:

- polish, performance, advanced UX and reliability hardening

## 17. Quality and Documentation Governance (Mandatory)

The following are required for every new feature and bug fix:

1. Automated tests covering normal and failure paths.
2. Sphinx documentation updates describing behavior and constraints.
3. CI pass for test and docs gates before merge.

Current enforced gates:

- `pytest` with coverage threshold (`--cov-fail-under=85`)
- `sphinx -W -b html docs docs/_build/html`

Definition of done:

- implementation complete
- tests added and passing
- coverage threshold maintained
- documentation updated and warning-free

## 18. Structure Import and MCIF Magnetic Extraction Requirements

Supported input types:

- CIF (`.cif`)
- MCIF (`.mcif`)
- CASTEP CELL (`.cell`)

Import behavior:

- parse structure with ASE
- copy source file into `imports/` to preserve provenance
- create structure entity and optionally set as active host
- append structured provenance record for import action

MCIF magnetic extraction behavior:

- parse propagation vectors from known scalar tags and `_magnetic_propagation_vector_kxkykz` triplet entries where present
- preserve multiple imported propagation vectors in reciprocal-lattice units, up to the three-vector basis used by the magnetic model
- mirror the first imported propagation vector to the legacy `propagation_vector` property for existing supercell workflows
- parse atom labels from `_atom_site_label`
- parse atom-site moment vectors from dotted and underscored `_atom_site_moment` loops where present, including CIF uncertainty and fraction numeric notation
- map atom-site moment labels onto host atom indices by exact label or a unique `label_...` atom-label expansion
- attach extracted magnetic metadata to structure properties
- when imported structure is active host, populate project-level magnetic structure
- derive and store per-atom scalar spins from extracted moments for host-structure editing/visualization
- promote MCIF-derived moments to propagation-vector components with phase-corrected Fourier coefficients so the unit-cell evaluated moments reproduce the imported labelled MCIF moments
- persist extracted propagation vectors in structure-level properties for supercell spin propagation workflows

Host visualization and CASTEP output behavior:

- host structure UI must expose editable per-atom spins and propagation vector controls
- generated CASTEP `.cell` files must include inline `SPIN=<value>` tokens on `POSITIONS_FRAC` rows when spin polarization is enabled
- generated CASTEP `.cell` files must omit inline spin tokens when spin polarization is not enabled

Failure behavior:

- unsupported extension, invalid content, and missing file conditions must return explicit user-visible errors

## 19. Spec-Sync and Authority Policy (Repository-Specific)

This policy applies to this repository only.

Spec-sync trigger:

- For code changes that affect user-visible behavior, architecture, validation logic, or persistence model, perform a spec-sync pass against this file before considering the change complete.

Spec-sync minimum actions:

1. Check whether the implementation changes any normative requirement in this specification.
2. Update this file if scope, behavior, or constraints changed.
3. Bump `Spec Version` and append a `Version History` row when updates are made.
4. Report completion of the spec-sync pass in change communication.

Authority and precedence:

- Direct user instructions in the current session take precedence over this specification.
- If a user request conflicts with this specification, follow the user request and explicitly report the disagreement.
- If no conflict exists, follow this specification as the default repository contract.

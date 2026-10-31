Project Specification
=====================

This document is the normative product and architecture specification for
MuFinder, a PySide6 desktop application for muon stopping-site workflows built
around CASTEP and scientific Python tools. It states what the product must do
and the constraints every feature must satisfy. The development process itself
(branching, gates, where code lives) is defined by ``AGENTS.md`` and
:doc:`/architecture`; the per-workflow user guides describe the implemented
behaviour in detail.

When a change alters user-visible behaviour, architecture, validation logic or
the persistence model, update the affected requirement here in the same change.
Git history is the version history of this document.

Product goals
-------------

1. Use a project as the unit of scientific work for a material.
2. Support resumable work across app restarts with full relevant state
   persistence.
3. Track reproducibility through structured provenance, not console text only.
4. Manage all jobs at the project level with lifecycle-aware filtering.
5. Make scientific stage dependencies explicit in product navigation.
6. Keep domain logic, persistence, orchestration, and UI clearly separated.
7. Maintain growth paths for new workflows and execution backends.

Non-goals for the current major release:

- multi-user collaborative editing of one project
- cloud-native orchestration backends beyond local and SSH HPC execution
- DFT engines beyond CASTEP
- a web UI replacing the desktop application
- a public plugin API

Workflow model
--------------

The staged workflow is:

1. project setup and host structure import
2. convergence testing
3. optional pristine host relaxation
4. muon-site generation and structural relaxation
5. relaxation analysis and clustering to distinct muon sites
6. downstream workflows consuming completed relaxed sites: hyperfine coupling,
   magnetic fields, quantum corrections and energy barriers

Dependency policy:

- downstream workflows consume completed structural muon relaxations as durable
  project entities, never unrelaxed candidates, with one stated exception:
  magnetic fields may use UEP sites as unrelaxed estimates, for a quick field
  estimate before any relaxation has run. Fields, dipolar tensors and nuclear
  relaxation computed at a UEP site are labelled as estimates wherever they are
  shown, and the workflow offers relaxation-analysis clusters before UEP sites
  once they exist for the host
- hyperfine coupling, quantum corrections, energy barriers and magnetic fields
  list the distinct
  sites pinned to the Muon Sites summary by default, one per pinned site of the
  matching charge state; a per-workflow ``Show every relaxed site`` option
  lists every eligible relaxed site instead
- convergence outputs prefill the project calculation defaults
- a pristine relaxation output may become the active host basis for later
  stages

Information architecture
------------------------

The main shell has a project navigator tree by stage on the left, a stacked
stage/workflow panel in the centre, and the global job dashboard in a bottom
dock. Project-wide views are always available regardless of the selected stage,
and stage panels surface their prerequisites and dependency gaps.

The project home shows where a project stands: a chart of the workflow stages
with their dependencies and progress, opened at the stage the user last worked
on; notes on the selected stage with a link to it; and a survey of the host
cell with that stage's muon positions (see :doc:`/user_guide/project_home`).

Project model and persistence
-----------------------------

A project is a directory ending in ``.mufinder`` that stores all scientific and
application state for one material: structures, calculations, runs, jobs,
results, muon sites and clusters, provenance records, a UI state snapshot,
imported raw source files and generated run artifacts.

Storage layout:

- ``project.json`` manifest
- ``state.json`` for UI state, including per-workflow simulation-cell selection
  under ``workflow_supercells``
- ``provenance.jsonl`` append-only provenance stream
- ``entities/<type>/<uuid>.json`` for structures, sites, calculations, runs,
  jobs and results
- ``imports/`` for copied source inputs
- ``files/`` for run input and output artifacts

Persistence requirements:

- projects are saved and loaded explicitly, and closing a visible session with
  an open project prompts to save first
- workflow-relevant UI state is persisted where appropriate
- provenance and the source files needed for reproducibility are preserved
- cross-entity references are validated before save
- saved supercell assets are persisted as ``Structure`` entities with
  ``properties.supercell_asset`` metadata rather than as a separate entity type
- persisted names and compatibility shims follow :doc:`/compatibility_policy`

Domain model
------------

Core entities: ``Project``, ``Structure``, ``WorkflowStage``,
``CalculationDefinition``, ``CalculationRun``, ``Job``, ``Result``,
``MuonSite``, ``MuonSiteCluster``, ``ProvenanceRecord``, and the
application-level ``ExecutionTarget`` profile referenced by project jobs. The
project also stores its ``host_structure_id`` and a project-level
``magnetic_structure`` (including MCIF-derived content). All references between
entities are UUID-based.

Job model and execution
-----------------------

Job lifecycle
~~~~~~~~~~~~~

Job records move through ``defined``, ``submitting``, ``queued``, ``running``
and ``retrieving`` to a terminal ``completed``, ``failed``, ``stopped`` or
``cancelled`` state (``JobStatus`` in ``mufinder.domain.enums``); workflow rows
show a ``defined`` job whose inputs are configured as ``ready``.
:doc:`/job_lifecycle` is the authoritative transition contract. The job system
provides a centralized registry and dashboard, typed tags for stage and
calculation type, strictly validated lifecycle transitions, and restart and
reload synchronization.

Calculation job architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Workflow expansion produces resolved ``JobSpec`` objects.
- CASTEP input rendering consumes only resolved ``JobSpec`` objects.
- Workflow logic, template expansion and variation handling stay separate from
  CASTEP syntax rendering.
- Validation and serialization are separate concerns.
- Default and override behaviour uses one explicit precedence order in one
  implementation location (``mufinder.workflow.merge``).
- HPC execution metadata accompanies ``JobSpec`` objects and never replaces
  their scientific content.

Local execution
~~~~~~~~~~~~~~~

- Submission entry points exist on workflow rows in ``ready`` state and on
  shared job-dashboard rows once jobs have entered the submission lifecycle.
- Each workflow panel exposes a run action that persists the current
  configuration and submits the selected rows, or every row when none is
  selected; the action's label states which of the two it will do.
- Configuring a sweep persists the configuration snapshot without generating
  CASTEP input files; inputs are materialized when the user submits ready rows.
- Configured rows show ``ready`` before submission even though no input files
  exist yet, and a row's sweep value becomes read-only once it is ``ready`` or
  in any later runtime state.
- Local execution runs in background process management and never blocks the
  UI. Multi-job submission supports sequential and parallel modes with
  user-controlled concurrency.
- Local execution configuration persists an MPI launcher path (``mpirun``) with
  user path selection and discovery fallback.
- Output parsing extracts at least the final energy (preferring
  ``NB est. 0K energy`` where available), the maximum force magnitude, and
  success or failure markers. Convergence tables show final energy and maximum
  force per row with enough numeric precision to compare small differences.
- Cancellation is available for active local jobs. Completed jobs leave the
  shared dashboard after a successful finish.
- Failed, stopped and cancelled jobs can be resubmitted from the dashboard,
  reusing existing remote files so continuation runs start from the same
  working directory; their ``*.err`` files can be inspected with contextual
  excerpts around failure lines; they can be reset to ``ready`` for rerun with
  optional cleanup of outputs that preserves generated inputs.
- Status displays use intuitive colour cues for running, completed, failed and
  cancelled states.

Remote (HPC) execution
~~~~~~~~~~~~~~~~~~~~~~

- Cluster profiles are persisted globally and selectable at submission time,
  and support configurable CASTEP executable paths. A profile lists the
  cluster's queues (partitions), each able to override the walltime, cores per
  node, memory, executable and template, and a submission picks one.
- SSH key authentication is primary; password fallback is supported.
- Unknown SSH host keys require explicit user trust confirmation before first
  use (trust on first use). Trusted keys are persisted in user-scoped settings
  keyed by cluster endpoint, and a host-key mismatch hard-fails with explicit
  recovery guidance.
- The scheduler abstraction supports SLURM script submission and addqueue
  command submission. Submission scripts for script-based schedulers are
  generated from user templates with runtime variable substitution. A built-in
  Redwood/Glamdring preset exists. Both schedulers expose a queue (partition)
  choice; addqueue clusters also expose memory-per-process and comment
  runtime controls and allow a blank walltime, while script-based SLURM
  clusters require one.
- Scheduler command resolution uses a configured absolute command path when
  one is set and otherwise probes the non-interactive SSH shell.
- The dashboard tracks cluster name and remote scheduler job id alongside
  status; queue monitoring supports periodic polling and manual refresh;
  connection state is surfaced with a status indicator and last-checked
  timestamp; queue-polling transport failures surface an explicit error with
  troubleshooting guidance.
- Completed remote jobs trigger automatic output retrieval for workflow
  parsing. Walltime-stopped jobs map to ``stopped`` and are resettable for
  direct resubmission; a successful resubmission updates the row with the
  replacement scheduler id.
- Before a job is submitted into a new remote directory, the application
  checks whether that directory exists and asks before overwriting one that
  does. A job whose directory cannot be checked is not submitted and is
  reported as a submission failure.
- Labels used in scheduler commands and remote paths are validated against a
  strict safe character set before submission, and recursive remote deletion
  uses SFTP traversal rather than a shell ``rm -rf``.
- Cluster-facing connection, submission, cancellation, refresh and polling
  failures present concise summaries with expandable details and actionable
  recovery suggestions.

Job tables and the dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Shared dashboard rows appear in the order their jobs were created, and that
  order survives closing and reopening the project; changing a job's state,
  including resubmitting or resetting it, never moves its row.
- Dashboard controls are per-row action buttons; a context menu may offer the
  same actions for the selection in addition.
- Text inside table rows does not change with the selection. Selection counts,
  hints and scope are shown outside the rows so tables stay usable on a small
  screen.
- Every workflow job table offers the same right-click row menu, in the same
  order, omitting only the actions that table does not have.
- Workflow row tables support removing the current row or the selected rows
  with Delete and Backspace, not only with pointer-driven controls.
- Editing the parameters of several rows in one dialog never changes a
  parameter the user did not touch. Parameters whose values differ between the
  rows are marked as such, keep each row's own value unless edited, and apply
  to every row once edited.
- An action on a selection that mixes applicable and inapplicable rows acts on
  the applicable rows and reports how many it left out: before the action where
  the control can show it (menu entries, prompts) and afterwards otherwise (the
  status bar after a submission). It is unavailable only when no selected row
  qualifies.
- A control embedded in a table row acts on the whole selection when its row is
  part of a multi-row selection, after a prompt offering "all selected rows",
  "this row only" and "cancel". A control on a row outside the selection first
  moves the selection to that row and then acts on it alone. Inherently
  single-row controls never prompt.
- Every action that removes, cancels, resets, dismisses or deletes is confirmed
  before it takes effect, from every entry point, with the safe choice as the
  default. An action covering several operations at once (Delete on a mixed
  dashboard selection) presents them in a single summary before any runs.
- Removing rows asks one confirmation for the whole batch, stating how many
  rows have a submitted job and how many running jobs will be cancelled.
  Removing a row that already has associated files warns and offers both
  permanent deletion and moving the files into an archival subfolder outside
  the active workflow set; for several such rows the prompt offers
  ``Delete all`` / ``Move all``.

Provenance and the journal
--------------------------

Provenance is structured and queryable. Each significant action records its
action type, stage context, parameter snapshot, input and output entity links,
associated job where relevant, timestamp and actor, so that what was run, with
which inputs and options, can be reconstructed.

The ``provenance_log`` route surfaces as a user-facing ``Journal`` page derived
from the persisted ``provenance.jsonl`` records rather than inferred job
history. Entries are grouped by local calendar day with human-readable date
headers, one compact card each with the local time. The default presentation is
curated and narrative-focused: it includes major workflow actions, submission
milestones, comparison-table creation, terminal calculation outcomes and the
findings that conclude a study (UEP minima, grouped sites, magnetic fields,
polarisation simulations, zero-point energies and barriers), and omits intermediate runtime states such as queued, running and retrieving. A card
expands to the details logged for it and the exact time, and links to its
workflow page. The Journal follows MuFinder branding as a notebook-style
expedition record using the existing palette and typography tokens; a stamp
marks milestones, completions, entries needing attention and failures, and red
marks failures only. Export supports plain text by default and Markdown as a
secondary format, giving the curated summary of each entry rather than its
details or raw provenance rows.

UI architecture
---------------

- MVVM-like separation with a service layer: views are thin and delegate to
  services; services own orchestration and validation; the domain layer stays
  GUI-independent. The package layers and their dependency directions are
  defined in :doc:`/architecture`.
- Long-running operations never block the UI; background execution signals
  updates thread-safely.
- Startup shows the splash screen as early as practical after ``QApplication``
  setup. Heavyweight feature imports and non-critical initialization run only
  after the splash is exposed, and new startup functionality must preserve
  this ordering.

Workflow requirements
---------------------

Convergence testing
~~~~~~~~~~~~~~~~~~~

- The parent ``Convergence Testing`` node opens a dedicated overview page. The
  parent stage is complete only when both the project default cutoff and the
  project default k-point grid are set.
- The overview page explains that both the plane-wave cutoff energy and the
  reciprocal-space sampling must be converged before later workflows are
  trusted, and shows two horizontally arranged notebook-style pinned regions
  (cutoff on the left, k-points on the right). Each region has a short
  explanation, a summary of the stored project default that mirrors the child
  workflow's formatting, and an action card that opens the child workflow.
- Pinned plots update only when the user adopts a project default from the
  corresponding convergence analysis dialog. Overview snapshots persist in
  ``Document.ui_state["convergence_overview_snapshots"]``, separately from the
  examiner-dialog restore state, and record ``host_structure_id``,
  ``simulation_cell_structure_id``, the adopted ``default_value``,
  ``energy_threshold``, ``force_threshold``, the plot
  payload derived from the examiner's ``plot_data``, and k-point point-label
  metadata when needed. A snapshot renders only when it matches the active host
  structure and the current recommended default; otherwise the page shows a
  dotted placeholder plus the current project-default summary without blocking
  the child workflows.
- Each child workflow has setup, run and analysis stages with persisted
  datasets and plot metadata, workflow-aware CASTEP parameter entry with
  curated basic fields and advanced disclosure, typed controls and parameter
  help backed by keyword metadata, a sweep builder supporting manual lists and
  start/stop/step range generation, and optional subset overrides that apply
  alternate parameter values to selected rows.
- The examiner judges energy per atom, maximum force and optional maximum
  stress with one criterion, relative to the finest point of the sweep, and
  checks a cutoff sweep's length with a power-law fit that never changes the
  recommendation. It persists its inputs and outputs so plots and results are
  restorable when it is reopened.
- Adopting a converged default persists the value, the thresholds, the cell
  and the functional, and records it as converged rather than set manually.
  Project-level defaults, including the cutoff, are visible from the GUI.
- The k-point spacing and grid mapping follows
  :doc:`/castep/kpoint_spacing_convention`.

Project defaults
~~~~~~~~~~~~~~~~

- The project stores three calculation defaults: XC functional, cutoff energy
  and k-point grid.
- Every workflow tab that consumes them shows the same compact project-defaults
  strip; criteria and converged-with functional belong in its tooltips
  and details dialog, not in the tab header.
- Adopting a cutoff or k-point default records the XC functional the sweep was
  run with and makes it the project default functional; a sweep whose points
  used different functionals records none and does not change the project
  default. Adopting a result converged with a functional different from the
  current project default must be confirmed by the user.
- All three defaults are directly editable and clearable without running a
  convergence test. A manually entered cutoff or grid is recorded as such (no
  criteria or converged-with functional) and does not change the
  project default functional.
- Changing the functional never discards a converged cutoff or k-point default;
  a default converged with a different functional is visibly flagged instead.
- The project default functional seeds the default ``XC_FUNCTIONAL`` of new
  rows in every workflow and never overrides a functional set explicitly on a
  row. Projects without a stored functional behave as they did before the
  default existed.

Pristine relaxation
~~~~~~~~~~~~~~~~~~~

- Optional host-only geometry relaxation whose relaxed structure is persisted
  as a first-class ``Structure``.
- The workflow supports multiple user-labelled rows in one configuration
  snapshot. Each row chooses its own simulation cell (unit cell or a saved
  supercell) and exposes a fixed-cell control that serializes to
  ``FIX_ALL_CELL : TRUE/FALSE``.
- Workflow defaults include the project cutoff energy and k-point grid when
  set. A row that uses a supercell has its k-point grid recomputed to preserve
  equivalent reciprocal-space density relative to the recommendation reference
  lattice, scaling each component by the reciprocal-axis change and rounding
  up; the recommendation persists the lattice it was derived from so the
  scaling is deterministic.
- Completion handling retrieves and parses both ``.castep`` and ``-out.cell``.
  A run whose geometry optimization did not converge is not treated as
  successful. Parsed relaxed lattices expose ``a``, ``b`` and ``c`` in the
  workflow table, each with its change against the host, and in the result
  payload.
- A ``Generate Comparison Table`` feature compares one or more completed DFT
  relaxations against the active experimental host structure as a persisted
  subtab with editable title, a one-line tooltip on each metric, a link to the
  metric descriptions in the user guide and CSV export
  (see :doc:`/user_guide/pristine_relaxation` and
  :doc:`/user_guide/comparison_table`).

Muon site generation and relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Generation methods: random constrained filling (by direct muon-muon
  separation or by a target number of positions with automatic spacing
  estimation), near-element placement, manual entry (fractional or Cartesian),
  and UEP cluster source rows from saved UEP analysis state (see
  :doc:`/user_guide/uep_workflow`).
- Generation constraints: periodic minimum-image muon-muon and muon-atom
  distances, attempt-capped sampling, symmetry-equivalent random-position
  reservation, and an internal deterministic seed for regression tests (not
  exposed in the UI).
- Staging selected positions prompts for positive muon, muonium or both charge
  states before rows are created. The relaxation table presents charge states
  as separate positive-muon and muonium tabs rather than a per-row charge
  column, and CASTEP total charge stays workflow-controlled: ``+1`` for the
  positive muon and ``0`` for muonium.
- Relaxations performed outside MuFinder can be imported as read-only rows
  (see :doc:`/user_guide/legacy_import`).

Relaxation analysis and clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Relaxed sites and energies are extracted from completed relaxations,
  filtered by displacement and quality constraints, clustered, and summarised
  as distinct sites with representative structures (see
  :doc:`/user_guide/structural_relaxation_analysis` and :doc:`/user_guide/site_generation`).
- Distinct sites can be pinned on the Muon Sites summary page, which is the
  source downstream workflows read, and exported as CIF. Each pinned site
  records its cluster's representative relaxation job, which resolves it to the
  relaxed muon site quantum corrections and energy barriers start from, under
  the same eligibility rules as the full list. A charge state with no pin, or a
  pinned site that no longer resolves, is reported on the downstream page with
  a link to the Muon Sites page.

Host structure magnetic editing and the supercell editor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The host-structure view supports per-atom scalar spin entry in units of
  ``hbar/2`` and propagation-vector entry in reciprocal-lattice units.
- Spin-arrow display defaults to disabled for a non-magnetic host and
  auto-enables when any per-atom spin is set non-zero; spins render as clearly
  visible 3D arrows with signed styling for opposite directions.
- The panel opens the atom table at the width of its contents and gives the rest
  to the viewer, with a user-resizable split between them; current-structure
  metadata is a one-line summary; left-clicking empty space in the viewer clears
  the atom selection.
- The Supercell Editor provides a compact diagonal scaling row (Nx × Ny × Nz)
  and an advanced full 3×3 integer matrix mode, read-only visual verification of the
  magnetic pattern across the generated supercell, and lets the user save,
  name, edit, delete and select reusable project supercells. Editing the
  matrix of a saved supercell already frozen into a configured calculation
  creates a future-use copy with its own name, reported in the status line, rather
  than rewriting the frozen structure (see
  :doc:`/user_guide/supercells`).

Workflow-scoped simulation cells and supercell assets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Each workflow stage resolves its own effective simulation cell independently
  and persists its selection under ``state.json`` ``workflow_supercells``. The
  modes are ``unit_cell`` (the active host) and ``saved_supercell`` (a saved
  derived structure snapshot).
- Saved supercell assets are regular ``Structure`` entities carrying
  ``parent_structure_id`` equal to the active host id,
  ``properties.supercell_asset.matrix``,
  ``properties.supercell_asset.owner_workflow_stage`` (nullable: the workflow it
  was saved from, null when saved from the host page),
  ``properties.supercell_asset.source_asset_structure_id`` (nullable: the frozen
  cell a future-use copy came from, or the source of a copy in an older project) and
  ``properties.supercell_asset.saved_at`` (when it was last saved). They are
  reusable across workflows for the same host and have user-facing names
  editable in the Supercell Editor.
- Workflows that select the same asset share it; selecting never copies, so the
  library holds one entry per cell. Saving a changed matrix on an asset that other
  workflows select asks first and names them.
- Convergence definition snapshots freeze the resolved simulation cell: the
  resolved structure id in ``CalculationDefinition.input_structure_ids[0]`` and
  lightweight provenance (``simulation_cell_mode``,
  ``supercell_source_workflow``) in the calculation parameters. Materialized
  ``.cell`` files render from that frozen snapshot, never from host-preview
  state.
- ``Project.supercell_matrix`` is read-only compatibility data: it seeds the
  Supercell Editor only in a project with no saved supercell for the active host,
  nothing writes it, and it does not govern workflow calculations.
- Invalid, missing or host-mismatched saved selections fall back safely to
  unit-cell mode, and selection state can be normalized.
- Stale all-zero supercell spin arrays are repairable from host magnetic
  metadata, and host spin extraction falls back to magnetic moment vectors, so
  spin-polarized workflows keep serializing the expected ``SPIN=`` tokens.

Magnetic fields
~~~~~~~~~~~~~~~

- Magnetic structures are built with the propagation-vector formalism from
  Fourier components over the host atoms; MCIF-derived structures seed them.
- Dipolar local fields and field distributions, dipolar tensors and nuclear
  dipolar linewidths are evaluated at muon sites, with symmetry-equivalent
  sites and muon-induced distortion corrections available; optional
  ``muesr`` / ``muLFC`` backends are used where installed, with clear errors
  otherwise (see :doc:`/user_guide/magnetic_fields`).
- The contact field follows the backend's contact shell, averaged once over
  every component of the order, and a shell that splits a tie of ions with
  different moments is warned of, in the settings and on each result.

Hyperfine coupling
~~~~~~~~~~~~~~~~~~

- Hyperfine jobs start from the same completed structural muon relaxations as
  quantum corrections, under the same eligibility rules, and are row-driven:
  each eligible relaxed site runs one CASTEP job whose status is the row's.
- Each job writes ``TASK : MAGRES``, ``MAGRES_TASK : HYPERFINE``,
  ``SPIN_POLARIZED : TRUE``, ``WRITE_FORMATTED_DENSITY : TRUE`` and the
  ``H:mu`` ``SPECIES_MASS`` / ``SPECIES_GAMMA`` blocks; muonium sites run
  neutral with ``SPIN=1`` seeded on the muon. Row settings cannot override the
  workflow-controlled keywords.
- A completed job reports the muon's hyperfine tensor in the physical sign
  convention (CASTEP's is the opposite), its isotropic and dipolar parts in
  MHz, the contact field in tesla, and the contact field from the pseudo spin
  density of the job's ``.den_fmt`` as a cross-check. A missing or unreadable
  ``.magres`` fails the job; a missing ``.den_fmt`` only leaves the cross-check
  empty.
- A result's coupling, in Å⁻³ per μB, can be written into every Magnetic
  Fields form whose selected muon site is the result's site, divided by the
  ``1/r^3``-weighted mean Mulliken moment of that form's contact shell from the
  source relaxation. A form's default site counts before the form is first
  edited. Overwriting a different nonzero coupling is confirmed first; nothing
  changes when no form shows the site. A shell that splits a tie of different
  moments is warned of.

Quantum corrections
~~~~~~~~~~~~~~~~~~~

- Quantum phonon calculations start only from completed structural muon
  relaxation results. UEP candidates, generated sites, pristine relaxations,
  failed jobs and manual sites without relaxation provenance are not valid
  sources.
- The workflow is row-driven: each eligible relaxed site runs one CASTEP
  phonon job. The source table exposes per-row ``Displacements``
  (``3 × atom count``) and ``Farms``, and the row's status is its job's status,
  ``ready`` before it has one. A row's parameters are editable while it has no
  job or its job is ``defined``, and read-only otherwise with a one-line
  reason; a failed, stopped or cancelled job recovers through the shared
  failed-job reset, with optional directory clean-up.
- Each job writes ``TASK : PHONON``, ``CONTINUATION : DEFAULT`` (a new run
  without a checkpoint, a resumed one with it), ``PHONON_KPOINT_MP_GRID 1 1 1``,
  ``PHONON_METHOD : FINITEDISPLACEMENT``, ``NONSCF_EIGENVALUE_TOL`` and explicit
  ``H:mu`` ``SPECIES_MASS`` / ``SPECIES_GAMMA`` blocks, which a fresh run reads
  from its own ``.cell``; ``num_farms`` greater than one adds ``NUM_FARMS``.
- Finite displacement is the only phonon method; the configuration offers no
  method or pseudopotential choice. A row an older version configured for DFPT
  or linear response is submitted as finite displacement with a one-line
  notice naming it.
- Only completed phonon jobs produce persisted result rows. ``.phonon`` files
  are expected outputs and are retrieved from cluster runs; with task farming
  active, ``*_farm001.castep`` is accepted as the primary CASTEP output. The
  muon mass CASTEP reports in the ``.phonon`` header is checked on every run.
- A phonon job can run on any execution target. Projects that hold the
  ``castep_quantum_singlepoint`` jobs of the earlier two-job design keep them as
  records: a row with only such a job is ready and writes its phonon job to a
  directory of its own, and a row whose singlepoint is still active is refused
  submission until it finishes or is cancelled.
- Parsed results persist the selected muon-dominated modes, per-mode and total
  zero-point energy, witness metrics, output files and source relaxation
  provenance. A ``Launch Analysis`` action opens an interactive window with a
  ``Phonon Modes`` tab (all Gamma-point modes, the muon-dominated subset
  highlighted, one selected mode synchronized to a muonated-structure viewer
  with amplitude-controlled playback) and a ``Muon Coupling`` tab
  (covariance/correlation tensors plus pairwise ``w2`` witnesses). Path-integral
  molecular dynamics is out of scope.

Energy barriers
~~~~~~~~~~~~~~~

- Path generation is explicit and pair-scoped: the user selects initial and
  final completed relaxed sites and generates the symmetry-distinct paths for
  that ordered pair on demand. Selecting the same site for both endpoints is
  allowed so symmetry-equivalent same-site hops can be generated, with
  zero-length self-maps excluded. An optional maximum path length filters the
  candidates.
- Two hops are the same path only when one symmetry operation maps both
  endpoints of one onto those of the other. The operations respect a collinear
  magnetic order on the host: those that keep the spin pattern or flip it as a
  whole. The page states whether paths use the crystal symmetry or the lower
  symmetry of the magnetic order.
- Generated path rows remain available after the endpoint selectors change,
  and are removable, including with Delete and Backspace when the table has
  focus.
- Barriers are computed with climbing-image nudged-elastic-band jobs seeded
  directly from the two relaxed endpoints, with the intermediate image
  synthesised by IDPP interpolation; no separate transition-state search step
  is required.
- The page is split across tabs for path generation and NEB analysis so it
  stays usable on smaller displays. Completed NEB jobs expose an inline
  inspector with image-by-image geometry preview in the shared structure
  viewer and a plotted reaction-coordinate profile synchronized with the image
  table, slider and viewer. Profiles are persisted and exportable (see
  :doc:`/user_guide/energy_barriers`).

Structure import and MCIF magnetic extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Supported inputs are CIF (``.cif``), MCIF (``.mcif``) and CASTEP CELL
  (``.cell``). Import parses CIF and MCIF with ASE and CELL files with the
  geometry-only mode of ``mufinder.castep_io.cell_parser``, copies the source file
  into ``imports/``, creates the structure entity (optionally as the active
  host) and appends a provenance record. Unsupported extensions, invalid
  content and missing files produce explicit user-visible errors.
- MCIF extraction parses propagation vectors from the known scalar tags and
  ``_magnetic_propagation_vector_kxkykz`` triplets, preserving several vectors
  in reciprocal-lattice units up to the three-vector basis of the magnetic
  model and mirroring the first to the legacy ``propagation_vector`` property;
  parses atom labels from ``_atom_site_label`` and moment vectors from dotted
  and underscored ``_atom_site_moment`` loops, including uncertainty and
  fraction notation; maps moment labels onto host atom indices by exact label
  or a unique ``label_...`` expansion; attaches the metadata to structure
  properties; populates the project-level magnetic structure when the import
  becomes the active host; derives per-atom scalar spins for editing and
  visualization; and promotes the moments to propagation-vector components with
  phase-corrected Fourier coefficients so the evaluated unit-cell moments
  reproduce the imported labelled moments.
- Generated ``.cell`` files include inline ``SPIN=<value>`` tokens on
  ``POSITIONS_FRAC`` rows when spin polarization is enabled and omit them
  otherwise.

Assumptions and known risks
---------------------------

Assumptions: single-user desktop operation, with CASTEP as the primary backend
in the current release window.

Known risks: parser variation across CASTEP and MCIF inputs; HPC connection
instability and recovery complexity; scale limits of entity-per-file storage
as projects grow; normalization of magnetic metadata from diverse MCIF
dialects.

Engineering requirements and definition of done
-----------------------------------------------

Every feature and bug fix must satisfy these requirements, which CI enforces
(:doc:`/quality_and_automation`):

1. Tests in ``tests/`` cover the normal execution paths, input validation and
   failure modes, and persistence behaviour where the feature writes or reads
   project data.
2. Documentation in ``docs/`` describes the user-facing behaviour, the inputs,
   outputs and constraints, and any workflow dependencies the feature
   introduces.
3. The change is not complete until both the test and documentation gates pass
   in automation.

A feature is done only when the implementation is merged, its tests are
present and passing, the coverage threshold is maintained, the Sphinx
documentation builds cleanly, and the documentation reflects the final
implemented behaviour.

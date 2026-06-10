Implemented Functionality
=========================

Application shell
-----------------

The current application provides:

- Project create/open/save actions from the main window.
- A project navigator with staged workflow sections.
- A project-wide job dashboard dock.
- Persisted per-project UI state.

Domain and persistence
----------------------

The data model includes:

- Project, Structure, MuonSite.
- CalculationDefinition, CalculationRun, Result.
- Job and structured ProvenanceRecord.

Persistence includes:

- JSON entity-per-file storage under ``entities/``.
- Project manifest, state snapshot, and provenance JSONL stream.
- Validation of cross-entity references before save.

User-facing save behavior is intentionally split between autosave and manual
save:

- Durable project state such as imported structures, saved workflow assets,
  generated/removed muon-site entities, calculation definitions, run tracking,
  and results is autosaved so active work is not lost if MuFinder needs to
  rebuild the project view.
- Analysis-session state stays in session memory until the user explicitly
  chooses ``Save Project`` (or accepts the save prompt when closing a modified
  project).
- This manual-save bucket includes exploratory UI state such as UEP wizard tab
  choices, clustering-window state, viewer camera/orientation, bond and spin
  visibility, and slice/isosurface display settings.
- In practice, users should treat ``Save Project`` as the action that commits
  their current analysis layout and visualization choices to disk for the next
  session, while autosave mainly protects durable project and job state.

Development harness
-------------------

The repository now includes an agent-legible development harness:

- ``AGENTS.md`` is a short route map to durable repository docs.
- :doc:`agent_harness` documents the expected agent workflow and feedback loop.
- :doc:`architecture` defines package layers and dependency directions.
- ``tests/test_agent_harness.py`` enforces the first structural guardrails for
  docs indexing, layer imports, and tracked bytecode artifacts.

Journal view
------------

The project navigator now includes a ``Journal`` page on the existing
``provenance_log`` route.

Current Journal behavior includes:

- Day-grouped rendering of persisted provenance with headers such as
  ``Tuesday 21 April 2026``.
- A curated notebook-style narrative focused on major workflow milestones
  rather than a raw event ledger.
- Milestone stamps for notable states such as defaults adopted, calculations
  completed, warnings, and failures.
- Export controls for:

  - plain text ``.txt`` output
  - Markdown ``.md`` output

- Export output that mirrors the curated Journal content.

Current Journal coverage includes:

- structure import
- convergence / pristine-relaxation / UEP configuration snapshots
- adopted project defaults
- candidate-site generation and UEP promotion
- calculation submission milestones
- terminal calculation outcomes (completed, failed, stopped, cancelled, reset)
- comparison-table creation

Current Journal limitations:

- It is strictly provenance-driven and does not backfill missing older history
  from saved job metadata.
- Intermediate runtime states such as ``queued``, ``running``, and
  ``retrieving`` are intentionally omitted to avoid noisy logs.
- Some analysis-only artifacts are still not represented as first-class
  Journal milestones.

Structure import
----------------

Host-structure loading currently supports:

- CIF input files (``.cif``).
- Magnetic CIF input files (``.mcif``) via CIF parser compatibility.
- CASTEP cell input files (``.cell``).

Import behavior currently includes:

- Parsing via ASE-backed readers.
- Copying source files into project ``imports/`` for provenance.
- Setting imported structure as active host structure by default.
- Capturing import provenance records.
- Extracting MCIF magnetic metadata when present, including atom labels,
  single-vector legacy tags, ``_magnetic_propagation_vector_kxkykz`` entries,
  and atom-site moment vectors.
- Preserving all imported propagation vectors as reciprocal-lattice-unit
  ``propagation_vectors_rlu`` metadata while keeping the first vector mirrored
  to the legacy ``propagation_vector`` property.
- Mapping imported moment labels onto atom indices by exact MCIF label match or
  the unique ``label_...`` expansion used by mag2spectra MCIF fixtures.
- Promoting extracted MCIF magnetic metadata to project-level magnetic structure when active host is imported.
- Deriving per-atom scalar spins from imported magnetic metadata and storing them in host structure properties.
- Converting MCIF-derived moments into propagation-vector components with the
  phase correction needed for the unit-cell moment at each labelled site to
  reconstruct the imported MCIF moment.
- Parsing inline ``SPIN=...`` tokens from ``POSITIONS_FRAC`` rows when loading ``.cell`` files.

Host-structure visualization and magnetic editing
-------------------------------------------------

The host-structure panel now includes:

- A resizable side-by-side splitter layout with atom table on the left and 3D structure viewer on the right.
- A compact two-row current-structure summary block (label/source/atom count/species) to preserve horizontal room for the viewer.
- Orthographic-only viewing (perspective toggle removed).
- Per-atom editable spin values in units of ``hbar/2`` via an inline table column.
- Propagation-vector entry controls in reciprocal-lattice units.
- 3D spin-arrow rendering in the structure viewer using fully 3D shafts/cones, centered on atom spheres with signed color mapping for up/down moments.
- Spin rendering disabled by default, with automatic enablement when any edited/imported spin value is non-zero.
- Atom transparency automatically applied when spin rendering is enabled.
- Viewer fitting accounts for full atom sphere extents (not just atom centers), preventing unnecessary clipping at left/right boundaries.
- Crystallographic axis triad rendered below the main viewer to maximize primary structure viewport area, with viewer and triad colours refreshed from the active light/dark palette.
- Left-clicking empty space in the viewer clears the current atom selection.
- A project-level supercell editor with simple diagonal controls, optional
  advanced full 3x3 matrix entry, naming, preview, save, edit, delete, and
  workflow selection support.
- A PyVista/VTK structure-viewer backend with camera-preserving overlay updates for bonds, spins, slice planes, point markers, and UEP isosurfaces.

Magnetic-field and nuclear-relaxation analysis
----------------------------------------------

The Magnetic Fields workflow currently includes:

- propagation-vector magnetic-structure editing and preview;
- commensurate dipolar-field calculations and incommensurate field-distribution
  sampling, including independent multi-q phase sampling with Auto, Grid, and
  Monte Carlo controls, active phase dimensions derived from the used
  component wavevectors, and separate lookup-resolution and distribution-sample
  limits;
- demagnetizing-field corrections for none, sphere, infinite plate, advanced
  ellipsoid, and advanced rectangular-prism sample shapes, with PyVista shape
  preview for advanced shapes, analytic rectangular-prism tensor averaging, and
  per-phase correction for incommensurate distributions;
- dipolar tensor calculations for selected magnetic ions;
- selected muon-site reuse from UEP and structural-relaxation clustering;
- native nuclear second-moment relaxation calculations for ZF/TF and
  single-crystal/polycrystal modes;
- bundled natural-abundance isotope data for reproducible nuclear relaxation;
- optional muon-induced distortion corrections when a selected
  structural-relaxation cluster site has a derived distorted structure.

Quantum Corrections
-------------------

The Quantum Corrections workflow now has a project page and implemented core
phonon analysis support.

Current behavior includes:

- listing structurally relaxed MuFinder muon sites that are eligible phonon
  sources;
- showing per-row displacement counts, current stage, and stage-2 farm-count
  metadata directly in the source table;
- rejecting generated, UEP-only, pristine-relaxation, failed, and
  provenance-free sources;
- running Quantum Corrections as a two-stage workflow: a required
  singlepoint job first, then a manual phonon continuation step once the
  singlepoint has completed;
- locking shared Quantum Corrections parameters after the completed
  singlepoint so only the stage-2 farm count remains editable before phonon
  submission, with an explicit workflow action to reset a row back to stage 1;
- expanding stage-1 CASTEP ``JobSpec`` objects with ``TASK : SINGLEPOINT``;
- expanding Gamma-only stage-2 CASTEP phonon ``JobSpec`` objects with
  ``TASK : PHONON``, ``CONTINUATION : DEFAULT``, ``PHONON_METHOD``,
  ``PHONON_KPOINT_MP_GRID 1 1 1``, optional ``NUM_FARMS``, and explicit
  ``H:mu`` mass/gamma cell blocks;
- submitting selected eligible sites through the shared local and cluster
  CASTEP job runners;
- reusing same-cluster stage-1 remote directories for stage-2 continuation
  submissions without prompting to delete the directory first;
- finite displacement as the default phonon method, with DFPT/linear-response
  compatibility checks for known pseudopotential restrictions;
- parsing CASTEP ``.phonon`` files into typed q-point, atom, frequency, and
  eigenvector data;
- selecting the three Gamma modes with largest muon participation;
- calculating per-mode and total muon ZPE, covariance/correlation tensors, and
  legacy-style ``w1``/``w2`` witnesses;
- solving one-dimensional single-particle effective potentials for
  Gomilsek-style weakly-bound and imported strongly-bound adiabatic analyses;
- launching an interactive analysis window from each completed phonon result,
  with single-particle mode plots, square-root energy scaling, covariance and
  correlation tensor inspection, and pairwise ``w2`` evaluation;
- retrieving ``.phonon`` files from cluster jobs and listing them with other
  job outputs, while accepting task-farmed ``*_farm001.castep`` outputs as the
  primary CASTEP text output when present;
- persisting completed quantum phonon analysis summaries into ``Result``
  entities.
- exposing a curated parameter dialog with searchable advanced CASTEP controls
  for each row, including stage-2 ``num_farms`` configuration.

See :doc:`quantum_corrections` for method assumptions and the planned
Gomilsek-style single-particle approximation layer.

Energy Barriers
---------------

The Energy Barriers workflow now has a dedicated project page for staged path,
transition-state, and NEB work.

Current behavior includes:

- explicit selection of initial and final relaxed endpoints before path
  generation;
- support for generating same-site Energy Barriers hops across
  symmetry-equivalent images, while filtering out zero-length paths;
- an optional maximum path-length filter for Energy Barriers path generation;
- symmetry-guided enumeration of distinct muon paths between the selected
  endpoints;
- generated transition-state path rows staying available after the endpoint
  selectors change;
- deletion of generated transition-state path rows from the Energy Barriers
  workflow;
- a path preview dialog that shows the undistorted matching pristine supercell
  with initial/final muon positions and an optional path arrow;
- curated per-row parameter dialogs for both transition-state and NEB jobs,
  with searchable advanced CASTEP controls available on demand;
- transition-state search job expansion with aligned reactant/product endpoint
  structures;
- supercell-aware Energy Barriers k-point defaults that preserve equivalent
  reciprocal-space sampling density relative to the recommended project
  lattice;
- NEB job expansion from completed transition-state results, including product
  and intermediate images when present in the parsed path file;
- inline NEB inspection with synchronized plot, image table, slider, and
  structure viewer.

See :doc:`energy_barriers` for the user-facing workflow, assumptions, and
current scope.

Supercell editor
----------------

The project navigator and workflow tabs expose a reusable ``Supercell Editor``.

Current behavior includes:

- listing saved supercells for the active source structure,
- naming saved supercells,
- generating previews before save,
- saving new supercells and saving copies,
- renaming and regenerating unused saved supercells,
- preventing deletion of supercells already frozen into configured calculations,
- creating a new future-use copy when matrix edits are made to a frozen saved
  supercell,
- returning a selected saved supercell to workflow tabs opened in
  ``Manage Supercells...`` mode.

Existing configured calculations remain reproducible because their
``CalculationDefinition.input_structure_ids`` continue to point at the original
frozen structure snapshot.

Workflow-scoped simulation-cell selection
-----------------------------------------

Simulation-cell selection is now workflow-scoped rather than project-global for calculation input generation.

Current UI support (convergence workflow) includes:

- A ``Simulation Cell`` section in the cutoff-convergence panel.
- A summary status label describing the active simulation-cell source.
- A dropdown with:

	- ``Single unit cell``
	- saved supercell assets derived from the currently active host structure

- ``Manage Supercells...`` action to open the project supercell editor and
  select or create a saved structure snapshot.
- ``Use Unit Cell`` action that returns the current workflow selection to host unit-cell mode without deleting saved assets.

Cross-workflow reuse behavior includes:

- Saved supercell assets are project-scoped and visible across workflows for the same active host.
- Selecting an asset owned by another workflow performs clone-on-load into the current workflow scope.
- Clone-on-load reuses an existing current-workflow asset when an identical host+matrix asset already exists.

Persistence model:

- Saved supercells are persisted as standard ``Structure`` entities (no separate entity type).
- Saved supercell structures set ``parent_structure_id`` to the active host structure id.
- Saved supercell structures store asset metadata under ``properties["supercell_asset"]``:

	- ``matrix`` (3x3 integer matrix)
	- ``owner_workflow_stage``
	- ``source_asset_structure_id`` (nullable, used for clone provenance)

- Asset labels are auto-generated from workflow stage and matrix (for example ``convergence_cutoff 2x2x1``).
- Per-workflow active selection is persisted in ``state.json`` under ``workflow_supercells``:

.. code-block:: json

   {
     "workflow_supercells": {
       "convergence_cutoff": { "mode": "unit_cell" },
       "muon_relaxation": { "mode": "saved_supercell", "structure_id": "<uuid>" }
     }
   }

Effective-structure resolution behavior:

- ``unit_cell`` mode resolves to active host structure.
- ``saved_supercell`` mode resolves to selected saved supercell structure.
- Missing/invalid/mismatched-host saved selections safely fall back to host unit cell, with persisted state normalization when requested.

Convergence definition/materialization behavior:

- Convergence configuration snapshots now freeze the resolved simulation cell at configuration time:

	- ``CalculationDefinition.input_structure_ids[0]`` stores selected supercell structure id, or host structure id for unit-cell mode.
	- ``CalculationDefinition.parameters`` stores lightweight provenance including:

		- ``simulation_cell_mode``
		- ``supercell_source_workflow``

- Materialized convergence ``.cell`` files are rendered from this frozen resolved structure, so expanded lattice/positions/spins are preserved even if later UI selection changes.

Compatibility behavior:

- ``Project.supercell_matrix`` is retained for host preview defaults and compatibility reads.
- ``Project.supercell_matrix`` is no longer authoritative for workflow calculation input structures.

Legacy spin-data repair behavior:

- Host spin extraction now falls back to ``magnetic_structure.moments[*].vector[2]`` when scalar spin lists are missing/invalid.
- For host structures only, all-zero scalar spins with non-zero magnetic moments are treated as stale and replaced by magnetic-moment-derived fallback spins.
- Saved supercell assets remain authoritative for their own spin arrays (fallback does not overwrite explicit asset spins during bridging).
- Legacy saved supercell assets with stale zero spins are automatically repaired from host magnetic data when selected/resolved/materialized, preventing missing ``SPIN=...`` tokens in generated spin-polarized convergence ``.cell`` files.

Muon candidate-site generation
------------------------------

Generation methods currently implemented:

- Random constrained filling in periodic cells. Random generation now evaluates
  the full symmetry-equivalent orbit of each proposed position before
  accepting it, following the MuFinder CPC initial-position algorithm. Users
  can generate by specifying a muon-muon separation, or by specifying a target
  number of initial positions and allowing MuFinder to estimate the spacing
  from the free cell volume, symmetry multiplicity, and the historical
  ``0.350`` random-filling factor.
- Element-attached constrained placement around host atoms selected from a
  dropdown populated by the active host structure. For every atom of the
  selected element, the requested number of muons per atom is chosen
  deterministically from local neighbour geometry and ranked by host-atom
  clearance.
- Manual entry in fractional or Cartesian coordinates, with fractional
  coordinates as the canonical stored representation and a live viewer preview.
- Clustered UEP batch-search results selected from saved UEP analysis state.
- Selected generated rows can be clustered around a fractional anchor before
  staging. MuFinder replaces each selected initial position by the
  symmetry-equivalent representative closest to that anchor, helping users
  avoid launching relaxations from different realizations of the same
  symmetry-equivalent site.
- Selected generated rows can be deleted from the generation table before
  relaxation jobs are configured.

Constraint handling currently includes:

- Minimum muon-muon distance under periodic minimum-image convention for
  random filling. This is a candidate-start diversity filter: every relaxation
  input still contains only one muon, and the spacing prevents redundant starts
  that are likely to relax to the same final site.
- Minimum muon-host distance under periodic minimum-image convention for
  random and element-attached generation.
- Attempt caps for random placement. The generation UI no longer exposes a
  random seed, but internal seed parameters remain available for regression
  tests.

Error handling currently includes:

- Explicit exceptions for unsatisfiable constraints.
- Explicit exceptions for unknown target elements.
- Duplicate protection for UEP cluster rows that have already been added as generated sites.
- UI surfacing of generation failures.
- UI surfacing of structure import failures.

UEP cluster candidates are added from the muon-generation workflow rather than
directly from the UEP wizard. The UEP stage remains responsible for probing,
batch optimization, zero-point estimates, clustering, and visualization; the
generation stage is where selected cluster representatives become initial
positions for full structural relaxation.
Cluster representatives use the optimized position of the lowest-energy member
chosen by the clustering summary, not an averaged cluster position or centroid.

When generated positions are turned into relaxation jobs, MuFinder asks whether
to create positive-muon jobs, muonium jobs, or both. The structural-relaxation
view separates those rows into ``Positive muon`` and ``Muonium`` tabs, while the
CASTEP total charge remains locked by MuFinder for each tab.

The symmetry-aware random method follows the initial-position generation
algorithm described by Huddart *et al.*, ``MuFinder: A program to determine and
analyse muon stopping sites``, *Computer Physics Communications* **280**,
108488 (2022), DOI: ``10.1016/j.cpc.2022.108488``. The element-attached method
is a deterministic seeding heuristic intended to provide chemically sensible
starts near selected species; final site assignment still requires subsequent
structural relaxation and clustering. For broader context on interstitial-site
generation and defect-style candidate enumeration, see the
``pymatgen-analysis-defects`` interstitial-generator documentation and the
``pymuon-suite`` AIRSS/UEP workflow material referenced in the UEP workflow
documentation.

UEP clustering analysis
-----------------------

The UEP analysis dialog now separates relaxed-site generation from clustering.

Current clustering behavior includes:

- Random-start batch searches that stop after generating and relaxing candidate
  muon sites.
- A separate ``Calculate ZPE`` action that evaluates and stores harmonic
  zero-point energies for the successfully optimized sites in eV.
- A separate non-modal clustering window launched from the batch-search log.
- Hierarchical clustering exploration through a dendrogram with a live
  tolerance control.
- K-means clustering exploration through a PCA projection of the normalized
  Soprano-backed feature vectors.
- A shared structure view that colors every relaxed muon position by its active
  cluster assignment.
- A shared cluster-summary bubble plot showing energy spread against average
  cluster energy relative to the global minimum.
- Batch-tab cluster tables that reflect the most recent separate clustering run
  while preserving cluster-representative promotion.

Calculation job framework
-------------------------

Workflow-driven CASTEP job generation now includes:

- A resolved ``JobSpec`` abstraction as the canonical output of workflow expansion.
- Separation between structure bridging (domain ``Structure`` to ``StructureSpec``) and CASTEP rendering.
- Layered ``InputTemplate`` merging with explicit precedence:

	- global defaults
	- workflow defaults
	- structure-derived values
	- sweep-generated overrides
	- explicit user overrides

- Explicit variation modeling with ``JobPatch`` for structure, cell, and param overrides.
- Concrete workflow expansion for:

	- convergence sweeps over ``CUT_OFF_ENERGY`` and ``KPOINT_MP_GRID``
	- muon-site sweeps over initial fractional muon positions

- Deterministic, filesystem-safe job naming and one output directory per resolved job.
- A CASTEP rendering adapter that renders only resolved ``JobSpec`` objects.
- Local output writing for ``.cell`` and ``.param`` files plus ``manifest.json`` for reproducibility/debugging.
- Conditional spin serialization in ``.cell`` files by appending inline ``SPIN=...`` tokens to ``POSITIONS_FRAC`` rows when and only when spin polarization is enabled.

Workflow-aware CASTEP parameter GUI
-----------------------------------

The application now includes a metadata-driven CASTEP parameter entry framework with:

- Job-type-aware parameter visibility driven from CASTEP keyword metadata.
- Basic and advanced progressive disclosure for parameter entry.
- Typed widgets by parameter kind (dropdowns, checkboxes, numeric controls, triplets, unit-aware float inputs).
- Parameter-level help dialogs sourced from keyword metadata.
- Structured form output mapped to ``InputTemplate``/``JobPatch`` compatible data rather than raw text.

Convergence overview workflow page
----------------------------------

The parent ``Convergence Testing`` workflow now opens a dedicated overview page with:

- A short explanation that both plane-wave cutoff energy and reciprocal-space sampling must be converged before later workflows are trusted.
- Two horizontally arranged notebook-style cards, one for cutoff convergence and one for k-point convergence.
- Dotted pinned-plot regions that visually distinguish adopted analysis snapshots from live editing views.
- Per-card action buttons that jump directly into the corresponding child workflow page.
- Summary rows that mirror the project-default formatting used inside the child workflow panels.
- Read-only pinned plots that update only when a user adopts a project default from the corresponding analysis dialog.
- Safe placeholder fallback when no adopted snapshot exists yet, or when a persisted snapshot no longer matches the current host/default context.
- Persisted overview snapshot state in ``ui_state["convergence_overview_snapshots"]`` separate from the existing examiner-dialog restore state.

Cutoff convergence GUI
----------------------

The convergence stage now includes a cutoff convergence panel with:

- Manual cutoff list entry and editable table-based review.
- Range generation from start/stop/step controls.
- An ``Examine Cutoff Convergence`` action placed directly below the cutoff table at the bottom-right of the sweep section.
- Shared CASTEP parameter form applied by default to all cutoffs.
- Optional subset-specific parameter overrides by selecting cutoff rows.
- Structured validation and persistence of cutoff configurations before runtime materialization.
- Reload-time restoration that rebuilds cutoff rows from persisted form state, with fallback to the active convergence definition when UI state is incomplete.
- Regenerated cutoff lists are treated as fresh ``ready`` rows and do not inherit stale runtime/status result columns from prior active definitions.
- ``ready`` row status for configured cutoffs as soon as convergence form parameters are saved.
- Cutoff-energy cells locked from direct editing once a row has been queued for execution or has progressed through runtime states.
- A dedicated convergence-examiner dialog with configurable thresholds and energy-method selection.
- Supported energy methods include Delta-threshold (successive per-atom energy differences), derivative-based analysis (``dE/dE_cutoff``), and asymptotic extrapolation toward ``E_inf`` with fit-quality safeguards.
- Force convergence analysis using delta-threshold checks on successive ``max_force`` values.
- Optional max-stress convergence analysis (delta-threshold over ``max_stress`` values, default
  threshold ``0.1 GPa`` to mirror ``GEOM_STRESS_TOL``). Exposed in the examiner via an
  ``Include stress`` tickbox that is hard-disabled until every completed sweep row reports a
  stress (i.e. every job was run with ``CALCULATE_STRESS=true``). When enabled, max stress is
  plotted on a secondary y-axis sharing the force subplot, and the stress converged value
  participates in the combined ``recommended_value`` alongside energy and force.
- ``CALCULATE_STRESS`` exposed as a per-workflow ``flag`` parameter in both the cutoff and
  k-point convergence parameter dialogs (basic visibility), defaulting to ``false``.
- Method help text available directly in the examiner dialog via an info action.
- Persisted examiner state (selected method, thresholds, text output, and plot inputs) so results and plots can be reopened without recomputation.
- Project-default cutoff display inside the convergence panel, including both the stored default cutoff value and the method used to determine that default.
- Adopting an examiner recommendation persists both project default cutoff energy and convergence-method metadata used to determine that cutoff.
- Adopting an examiner recommendation also freezes a parent-overview pinned snapshot keyed to the active host structure, selected simulation cell, adopted default value, and plot payload.

K-point convergence GUI
-----------------------

The convergence stage now also includes a dedicated k-point convergence panel with a matching UI style:

- Manual Monkhorst-Pack grid entry and editable table-based review.
- Grid generation from a user-defined minimum reciprocal-space spacing (1/Ang).
- CASTEP-compatible reciprocal-space conversion rule for maximum density:

	- ``n_i = floor(|b_i| / (2*pi*spacing)) + 1``
	- ``|b_i|`` are reciprocal-lattice-vector magnitudes from the active simulation cell.

- Generated sweep enumerates the distinct CASTEP grids from ``(1,1,1)`` up to
  the spacing-derived maximum grid by walking CASTEP's spacing trajectory; every
  emitted grid is reproducible from some ``KPOINT_MP_SPACING``. For anisotropic
  cells this typically produces more grids than an isotropic level ramp (each
  axis transitions independently at its own spacing thresholds).
- Cell-sensitive generation tracking: the panel records the simulation cell used for grid generation and warns if the simulation cell changes afterwards.
- Shared CASTEP parameter form with per-grid override support, equivalent to cutoff convergence behavior.
- Automatic propagation of project default cutoff energy into k-point convergence defaults when available.
- K-point convergence examiner plotting total energy and max force vs number of k-points,
  with optional max-stress secondary axis on the force subplot when ``CALCULATE_STRESS`` was
  enabled across every grid in the sweep.
- Hover tooltips on plotted points that show the originating Monkhorst-Pack grid.
- Delta-threshold-only convergence decision for k-point studies (no derivative/asymptotic method selection).
- Displayed k-point spacing for a grid is the inclusive upper bound of the
  CASTEP-valid range, ``min_i(|b_i|/(2*pi*(N_i-1)))`` over axes with
  ``N_i >= 2``. Feeding this value to CASTEP as ``KPOINT_MP_SPACING``
  reproduces the displayed grid exactly. The (1, 1, 1) Gamma-only grid has
  no single representative spacing and is shown without an annotation.
- Persisted project defaults for adopted k-point recommendations, including method, thresholds, and the cell identifier used for the recommendation.
- Adopting a k-point recommendation also freezes a parent-overview pinned snapshot, including persisted grid-label metadata for compact redraw on the parent convergence page.

Pristine-relaxation workflow GUI
--------------------------------

The optional ``Pristine Relaxation`` stage now includes:

- A table-driven multi-job configuration with explicit per-row labels.
- Per-row supercell selection via the supercell preview dialog (unit cell or custom 3x3 matrix).
- Per-row ``FIX_ALL_CELL`` control with deterministic CASTEP ``FIX_ALL_CELL`` serialization.
- Per-row parameter configuration through the shared workflow-aware CASTEP parameter dialog.
- Project-default propagation of cutoff energy and k-point grid into generated pristine-relaxation jobs.
- Supercell-aware k-point remapping that preserves equivalent reciprocal-space sampling density relative to the recommended reference lattice by scaling each k-point component with the corresponding reciprocal-axis change and always rounding up to avoid undersampling. For example, a ``4x4x4`` default maps to ``2x2x2`` for diagonal ``2x2x2`` and ``2x2x3`` supercells.
- Runtime status columns for each row including ``a``, ``b``, ``c`` lattice parameters extracted from relaxed outputs.

Pristine-relaxation result handling now includes:

- Retrieval/listing of both ``<seed>.castep`` and ``<seed>-out.cell`` outputs.
- Geometry-convergence gating: pristine jobs are marked failed when CASTEP reports incomplete geometry convergence.
- Parsing of ``-out.cell`` relaxed geometry into structured lattice/species/fractional coordinates.
- Persistence of relaxed ``a``, ``b``, ``c`` in ``Result.data``.
- Persistence of one derived relaxed ``Structure`` entity per successful pristine job.

Pristine-relaxation comparison tables
-------------------------------------

The ``Pristine Relaxation`` stage now includes a ``Generate Comparison Table`` feature with:

- a dialog listing completed pristine-relaxation jobs with include tickboxes,
- support for selecting one or more DFT jobs against the active experimental host structure,
- generation of a persisted comparison-table subtab under ``Pristine Relaxation`` in the project tree,
- inline editing of comparison-table titles after creation, with the project-tree subtab label kept in sync,
- deletion from either a top-right ``Delete`` button in the comparison view or a project-tree context menu,
- metric rows with per-metric ``Info`` dialogs,
- a collapsible ``DFT Parameter Differences`` section that shows only parameter values that differ across selected DFT jobs,
- CSV export of rendered comparison data.

Comparison science currently includes:

- supercell detection and experimental-structure tiling for supercell-aware comparisons,
- periodic-boundary-aware species-partitioned atom matching,
- lattice, displacement, bonding, topology, and symmetry metrics,
- a composite similarity score assembled from normalized sub-metrics,
- explicit warnings when exact matching cannot be achieved.

Future extension boundary:

- HPC scheduler and transport metadata is intentionally deferred.
- Future execution metadata will accompany ``JobSpec`` rather than replacing it.

Local CASTEP execution
----------------------

The application now supports local CASTEP submission from UI tables for cutoff
configurations that have been saved into the ``ready`` state.

Setup integration includes:

- CASTEP executable path persistence.
- ``castep_keywords.json`` metadata path persistence.
- ``mpirun`` launcher path persistence.
- ``mpirun`` auto-discovery through PATH/common install prefixes.
- In-dialog workflow guidance for keyword generation:
  set the ``Keywords JSON`` destination path, generate metadata, then save
  setup with ``OK`` so the selected metadata source is applied.

Submission entry points include:

- Dedicated ``Run Local Calculation(s)`` action in the cutoff convergence panel,
  operating on the current selection (or current row when a single cutoff is in focus).
- Right-click context menu in cutoff convergence rows.
- Delete/Backspace keyboard removal for the current cutoff row or a selected set of rows.
- Per-row action buttons in the shared job dashboard table after a job has entered the queue/submission lifecycle.
- Failed/cancelled-job resubmit plus failed-job inspection and reset actions in the shared job dashboard.

Submission controls include:

- Cores per job.
- Sequential execution for multi-job submission.
- Parallel execution for multi-job submission with user-selected
	maximum concurrency.
- Core-count guidance from local logical CPU count.
- Oversubscription warnings when requested concurrency exceeds
	logical CPU availability.

Runtime behavior includes:

- Background execution using Qt ``QProcess`` to avoid UI blocking.
- Input-file generation at submission time for selected ready cutoff jobs.
- Pressing the cutoff-panel run action persists the current convergence configuration
	snapshot and then submits the selected cutoff rows.
- Per-job queueing/running/completion state updates reflected in UI.
- Configured cutoff rows staying in ``ready`` until queued for execution.
- Completed jobs leaving the shared dashboard after successful finish.
- Cancellation for queued and running jobs.
- Shared execution service reusable by convergence and dashboard flows.
- Removing a cutoff entry now warns before touching associated files and either
	deletes them or moves them into a ``removed_from_workflow`` archive subfolder.
- Multi-cutoff removals support shared ``Yes to All`` / ``No to All`` choices for
	repeated removal and file-handling confirmations.

- Resetting failed jobs back to ``ready`` for rerun, with optional cleanup
	of output files while preserving generated input files.
- Resubmitting failed or cancelled dashboard jobs directly, reusing existing
	remote files for continuation runs and replacing the stored remote scheduler
	id with the new id.
- Save-on-close prompting for open visible projects.
- Color-coded status text for dashboard and convergence status displays.

The shared job dashboard is intended to show jobs that still need attention.
Ready jobs are submitted from their workflow tables. Active jobs appear in the
dashboard while they are submitting, queued, running, or retrieving outputs.
Completed jobs leave the dashboard after successful result handling. Failed,
stopped, and cancelled jobs remain visible so they can be inspected,
resubmitted, or reset. See :doc:`job_lifecycle` for the full lifecycle contract
and the current stuck-row prevention recommendations.

Failure diagnostics include:

- Failed-job inspection in the shared dashboard.
- ``.err`` file discovery in job output directories.
- Context-style excerpts around failure lines in ``.err`` files.

Status transitions for local execution include:

- ``ready`` -> ``submitting`` -> ``queued`` -> ``running`` -> ``completed``
- ``ready``/``submitting``/``queued``/``running`` -> ``cancelled``
- ``submitting``/``queued``/``running`` -> ``failed``

Status transitions across local and cluster execution now also include:

- ``running``/``retrieving`` -> ``stopped`` (for walltime/time-limit stops)
- ``failed``/``stopped``/``cancelled`` -> ``defined`` via reset for resubmission
- ``failed``/``cancelled`` -> ``submitting`` via dashboard resubmission

Output handling includes:

- In-place execution inside job directories created at submission time.
- Post-run parsing of ``.castep`` output files.
- Structured extraction of:

	- completion/failure markers
	- final total energy with preference for ``NB est. 0K energy`` when available
	- enthalpy (where available)
	- force and stress blocks (latest block)
	- maximum force magnitude and maximum stress component magnitude
	- SCF cycle count
	- wall time
	- warnings and error markers
	- geometry-convergence status markers

- Persisting parsed data into ``Result`` entities and linking to
	``CalculationRun`` records.
- Total energy and maximum force propagation into the cutoff convergence table,
	formatted with higher precision to support small-difference comparisons.

Remote HPC execution
--------------------

The application now supports remote cluster submission for convergence jobs.

Cluster setup and authentication includes:

- Global persisted cluster profiles via Setup -> HPC Clusters, including built-in
	SCARF and Redwood presets.
- Configurable hostname, username, queue system, default working directory,
	cores-per-node, default walltime, submission-script template path, CASTEP
	executable path, default queue/partition, and default memory-per-process.
- Optional SSH jump-host settings for clusters behind gateways, including
	separate jump hostname, port, username, authentication method, and key path.
- The built-in Redwood preset targets ``glamdring.physics.ox.ac.uk`` through
	the Physics gateway ``bastion.physics.ox.ac.uk``. MuFinder prompts separately
	for Physics/bastion credentials and Glamdring credentials because those
	accounts can differ.
- Optional explicit scheduler command paths for submit/queue/cancel commands,
	with non-interactive-shell command probing fallback.
- SSH key authentication with password fallback support.
- Explicit host-key trust prompts are preserved for both the jump host and the
	target cluster host; bastion trust records are stored separately from
	Glamdring trust records.

Remote submission and monitoring includes:

- A dedicated ``Run on Cluster`` action in the convergence panel.
- Cluster selection and runtime override dialog at submission time.
- Upload of generated ``.cell`` / ``.param`` inputs and scheduler-aware submission.
- SLURM submission through ``sbatch`` with populated templates and capture of
	scheduler job ids.
- Redwood submission through ``addqueue`` with exposed runtime controls for
	queue/partition, memory per process, and comment.
- addqueue submission uses sbatch-compatible response mode to keep scheduler
	job-id parsing stable.
- Dashboard columns for execution target (cluster/local) and remote job id.
- Dashboard column for per-job process count used at submission time.
- Connection state indicator and manual queue refresh action.
- Manual queue refresh retries disconnected sessions by attempting reconnect for
	clusters that currently have active dashboard jobs.
- When a matching cluster connection is already active, submission reuses that
	connection context and does not request password/passphrase again.
- Periodic queue polling with last-checked timestamp updates.
- SLURM queue polling tolerates stale/expired job ids (``Invalid job id specified``)
	by falling back to per-job status checks instead of failing the full refresh.
- Poll failures now force a full runner disconnect state (not just UI signal changes),
	so subsequent ``Refresh Queue`` actions can cleanly reconnect rather than remaining stuck.

Remote lifecycle and retrieval includes:

- Scheduler state mapping to MuFinder job states (queued/running/completed/failed/cancelled/stopped).
- Automatic retrieval of completion outputs for parser ingestion.
- ``stopped`` state support for walltime-limited runs with reset flow.
- Shared job execution lifecycle orchestration records submissions, creates
  calculation runs, applies local/remote status transitions, and records
  terminal provenance events outside the Qt window layer.

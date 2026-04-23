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
  calculation definitions, run tracking, and results is autosaved so active
  work is not lost if MuFinder needs to rebuild the project view.
- Analysis-session state stays in session memory until the user explicitly
  chooses ``Save Project`` (or accepts the save prompt when closing a modified
  project).
- This manual-save bucket includes exploratory UI state such as UEP wizard tab
  choices, clustering-window state, viewer camera/orientation, bond and spin
  visibility, and slice/isosurface display settings.
- In practice, users should treat ``Save Project`` as the action that commits
  their current analysis layout and visualization choices to disk for the next
  session, while autosave mainly protects durable project and job state.

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
- Extracting MCIF magnetic metadata when present (propagation vector and site moments).
- Promoting extracted MCIF magnetic metadata to project-level magnetic structure when active host is imported.
- Deriving per-atom scalar spins from imported magnetic metadata and storing them in host structure properties.
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
- Crystallographic axis triad rendered below the main viewer to maximize primary structure viewport area.
- Left-clicking empty space in the viewer clears the current atom selection.
- A supercell-preview dialog with simple diagonal controls and optional advanced full 3x3 matrix entry.
- A PyVista/VTK structure-viewer backend with camera-preserving overlay updates for bonds, spins, slice planes, point markers, and UEP isosurfaces.

Workflow-scoped simulation-cell selection
-----------------------------------------

Simulation-cell selection is now workflow-scoped rather than project-global for calculation input generation.

Current UI support (convergence workflow) includes:

- A ``Simulation Cell`` section in the cutoff-convergence panel.
- A summary status label describing the active simulation-cell source.
- A dropdown with:

	- ``Single unit cell``
	- saved supercell assets derived from the currently active host structure

- ``Create From Preview...`` action to open supercell preview and save a structure snapshot.
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

- Random constrained filling in periodic cells.
- Near-element constrained placement around host atoms.
- Manual entry in fractional or Cartesian coordinates.

Constraint handling currently includes:

- Minimum muon-muon distance under periodic minimum-image convention.
- Minimum muon-host distance under periodic minimum-image convention.
- Attempt caps and optional deterministic random seed.

Error handling currently includes:

- Explicit exceptions for unsatisfiable constraints.
- Explicit exceptions for unknown target elements.
- UI surfacing of generation failures.
- UI surfacing of structure import failures.

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

- Generated sweep series from ``(1,1,1)`` up to the spacing-derived maximum grid.
- Cell-sensitive generation tracking: the panel records the simulation cell used for grid generation and warns if the simulation cell changes afterwards.
- Shared CASTEP parameter form with per-grid override support, equivalent to cutoff convergence behavior.
- Automatic propagation of project default cutoff energy into k-point convergence defaults when available.
- K-point convergence examiner plotting total energy and max force vs number of k-points.
- Hover tooltips on plotted points that show the originating Monkhorst-Pack grid.
- Delta-threshold-only convergence decision for k-point studies (no derivative/asymptotic method selection).
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
- Supercell-aware k-point remapping that preserves equivalent reciprocal-space sampling density relative to the recommended reference lattice.
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
- Failed-job inspection and reset actions in the shared job dashboard.

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
- Save-on-close prompting for open visible projects.
- Color-coded status text for dashboard and convergence status displays.

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
- Optional explicit scheduler command paths for submit/queue/cancel commands,
	with non-interactive-shell command probing fallback.
- SSH key authentication with password fallback support.

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
- ``stopped`` state support for walltime-limited runs with reset/resubmit flow.

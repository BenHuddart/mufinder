Architecture
============

MuFinder is organized around a small set of package layers. The layers are
designed to keep scientific calculations, project state, execution services,
and PySide widgets understandable to both humans and coding agents.

Package layers
--------------

``mufinder.domain``
   Dataclasses, enums, timestamps, and durable domain records. This layer must
   not import outer MuFinder packages.

``mufinder.castep_io``
   CASTEP parsing, writing, schema metadata, presets, and validation. It owns
   CASTEP syntax details and should not depend on UI, persistence, services,
   workflow orchestration, or scientific algorithms.

``mufinder.persistence``
   JSON storage and codec helpers. Persistence may serialize domain objects but
   must not reach into UI, CASTEP I/O, services, workflow expansion, or science
   modules.

``mufinder.science``
   Numerical and crystallographic algorithms, including site generation,
   structural comparison, UEP analysis, magnetic structures, magnetic fields,
   and demagnetizing tensors. Science code may use domain records and CASTEP
   parsing helpers where those are part of scientific inputs, but it must remain
   UI-free and service-free, and it must not import ``persistence`` or
   ``workflow``.

``mufinder.workflow``
   Pure workflow expansion and rendering contracts. This layer turns resolved
   workflow state into ``JobSpec`` objects and rendered input files without
   owning project mutation, runner state, or widgets. It may use ``domain``,
   ``castep_io``, and ``persistence`` codec helpers, but not ``science``,
   ``services``, or ``app``. Each workflow is exposed
   as a keyword-only ``expand_*`` function returning ``list[JobSpec]`` that
   layers ``InputTemplate`` parameters in ``MERGE_PRECEDENCE_ORDER``; see
   ``mufinder.workflow.base`` for the contract. Every ``expand_*`` function
   accepts the full five-layer template stack — ``global_defaults``,
   ``workflow_defaults``, ``structure_defaults``, the per-job patch (sweep)
   layer, and ``user_overrides`` — and threads each into
   :func:`~mufinder.workflow.merge.resolve_job`. Function-based workflows
   compose their per-job patch layer with
   :func:`~mufinder.workflow.merge.compose_patch`, whose rule is
   ``patch = compose_patch(workflow_base_template, config_overrides)``: the
   workflow-computed base template (e.g. ``task``/``continuation``) is passed
   first and the per-config ``cell_overrides``/``param_overrides`` last, so the
   base template sits in the patch layer *above* ``workflow_defaults`` and
   ``structure_defaults`` while per-config overrides still win over the base.

``mufinder.services``
   Project-level state management, local and remote execution, journals,
   scheduling, and service orchestration. Services own lifecycle transitions and
   persistence-facing workflows, but they must not import PySide widgets. The
   local and cluster job runners share a structural contract, ``JobRunner``
   (``mufinder.services.runner_protocol``), which documents the runner signal
   surface, the deliberately-different submit error contracts, and the rule that
   runners never mutate ``Job`` objects; see :doc:`job_lifecycle` for the full
   state-authority contract.

``mufinder.app``
   PySide widgets, dialogs, theme helpers, and main-window wiring. UI code may
   depend on lower layers but should delegate reusable planning, execution, and
   calculation logic back to services, workflow, or science modules.

Dependency directions
---------------------

The intended direction is inward from UI to services/workflow/science and then
to domain/data helpers. Lower layers must not import higher layers.

Allowed examples:

- ``mufinder.app`` calling ``mufinder.services`` or ``mufinder.science``
- ``mufinder.services`` mutating project state and invoking science helpers
- ``mufinder.workflow`` consuming ``mufinder.domain`` and ``mufinder.castep_io``
- ``mufinder.science`` consuming ``mufinder.domain`` records

Disallowed examples:

- ``mufinder.science`` importing ``mufinder.app`` widgets
- ``mufinder.workflow`` importing ``mufinder.services`` runners
- ``mufinder.services`` importing ``mufinder.app`` dialogs
- ``mufinder.domain`` importing any outer MuFinder package

These boundaries are enforced by ``tests/test_agent_harness.py``
(``FORBIDDEN_LAYER_IMPORTS``), which also forbids ``PySide6.QtWidgets`` and
``PySide6.QtGui`` imports in every layer package except ``mufinder.app``
(``PySide6.QtCore`` is allowed for signals, threads and settings). If a future
feature genuinely requires a new dependency direction, update this document and
the structural test in the same change.

A few top-level modules sit outside the seven layers and are not checked by
the structural test: ``mufinder.main`` (application entry point),
``mufinder.branding`` (Qt icons), ``mufinder.assets`` / ``mufinder.data``
(packaged resources, including the starter ``keyword_catalog.yaml``),
``mufinder._version``, ``mufinder.castep_setup`` (saved CASTEP setup and
keyword-catalog management, used by the app), and ``mufinder.cluster_setup``
(the built-in cluster presets and the ``QSettings`` persistence of cluster
configurations, used by the app). ``castep_setup`` and ``cluster_setup``
depend on ``PySide6.QtCore`` but not on widgets, so no layer package imports
them; the ``ClusterConfig`` record that services consume lives in
``mufinder.domain.cluster``. Keep new code out of this top level unless it is
genuinely cross-cutting configuration.

Refactoring policy
------------------

When duplicated UI code starts to encode workflow, execution, or styling
policy, move the reusable part into the appropriate shared layer:

- workflow-row planning belongs in ``mufinder.app.workflow_submission`` or a
  lower non-UI service once it does not depend on widgets
- any other cross-workflow UI concern belongs in the module listed under
  Shared workflow infrastructure below
- job lifecycle transitions belong in ``mufinder.services``
- scientific calculation behavior belongs in ``mufinder.science``; within it,
  lattice maths (fractional/Cartesian conversion, minimum-image distances,
  periodic deduplication) belongs in ``mufinder.science.lattice``, symmetry
  plumbing (spglib cells, cached symmetry operations, orbits) in
  ``mufinder.science.spglib_compat``, conversion to ASE ``Atoms`` in
  ``mufinder.science.ase_bridge``, the UEP field in ``mufinder.science.uep``
  and clustering of its optimized positions in
  ``mufinder.science.uep_clustering``
- reading CASTEP text belongs in ``mufinder.castep_io``, including the
  formatted density, pseudopotential reports and cell charge the UEP analysis
  needs (``mufinder.castep_io.charge_density``); the keyword schema keeps
  catalog discovery, loading and the lookup API in
  ``mufinder.castep_io.schema`` and the bundled starter-catalog translation in
  ``mufinder.castep_io.starter_catalog``
- reading the magnetic metadata persisted on a structure (per-atom spins,
  moment-derived spins, the propagation vector) belongs in
  ``mufinder.domain.magnetic_properties``, so the workflow bridge, services and
  science modules interpret those properties the same way
- reusable PySide styling belongs in ``mufinder.app.theme``
- shared adapter logic that turns GUI value dicts into workflow templates
  belongs in ``mufinder.app.adapter_common``
- persisted compatibility behavior belongs in ``mufinder.persistence`` or the
  owning service, with a ledger entry in :doc:`compatibility_policy`

The preferred outcome is code that future agents can navigate by package name,
not by remembering where a previous implementation happened to grow.

Workflow adapters
-----------------

The per-workflow adapters in ``mufinder.app`` (``convergence_adapter``,
``relaxation_adapter``, ``uep_adapter``) translate GUI configuration objects
into workflow-layer ``expand_*`` calls. Their shared machinery lives in
``mufinder.app.adapter_common``:

- :func:`~mufinder.app.adapter_common.build_input_template` parses a GUI value
  dict into an :class:`~mufinder.workflow.types.InputTemplate`. Each adapter
  passes its own ``context_label`` (for the unsupported-keyword error), its own
  ``forbidden`` map (canonical keyword → exact validation message, covering
  sweep-list, structure-controlled, and row-controlled keywords), and its own
  ``kpoints_list_reset`` sentinel (``()`` for convergence to clear an inherited
  list layer, ``None`` for relaxation/UEP to inherit).
  :func:`~mufinder.app.adapter_common.build_workflow_settings_template` is the
  same parser with the settings the quantum-corrections and energy-barriers
  rows use: the structure-controlled cell keywords are refused and a scalar
  k-point grid or spacing clears an inherited list.
- :func:`~mufinder.app.adapter_common.rescale_kpoint_grid` rescales a project
  k-point grid to an equivalent reciprocal-space density for supercell rows and
  returns an optional adjustment message. Convergence sweeps a fixed structure
  and never rescales.
- :class:`~mufinder.app.adapter_common.AdapterExpansion` is the uniform return
  shape (``jobs`` plus UI-facing ``messages``) for every adapter's public
  ``expand_*`` function. ``main_window`` surfaces non-empty ``messages`` from
  the configure path.
- The supercell adapters (pristine relaxation, muon relaxation, UEP) share one
  expansion prologue. :func:`~mufinder.app.adapter_common.seed_workflow_defaults`
  layers the project exchange-correlation functional and cut-off energy over
  the workflow defaults, and
  :func:`~mufinder.app.adapter_common.kpoint_reference_lattice` picks the
  lattice the project grid was converged on (the recorded reference, else the
  host cell). Per row, :func:`~mufinder.app.adapter_common.require_row_label`
  and :func:`~mufinder.app.adapter_common.claim_unique_name` reject empty and
  duplicate rows, and :func:`~mufinder.app.adapter_common.prepare_row` converts
  the row structure, drops or forces the workflow-controlled values, builds the
  row template and rescales the project grid unless the row pins its own
  k-points.
- :func:`~mufinder.app.adapter_common.coerce_supercell_matrix`,
  :func:`~mufinder.app.adapter_common.is_identity_supercell` and
  :func:`~mufinder.app.adapter_common.supercell_label` are the one parser,
  identity test and label for a 3×3 supercell matrix, used by the adapters and
  the workflow widgets alike.
- :func:`~mufinder.app.adapter_common.project_convergence_parameter_values`
  turns the project defaults (functional, cut-off energy in eV, k-point grid
  rescaled to a target cell) into GUI parameter values, the lowest layer of an
  energy-barrier path's configuration.

``adapter_common`` must remain free of PySide imports; it depends only on
``workflow``, ``castep_io``, ``science``, and ``domain``.

Shared workflow infrastructure
------------------------------

The workflow pages are built from one set of shared helpers. Each
cross-workflow concern has exactly one home; a workflow extends the helper there
rather than carrying a private copy (``AGENTS.md``, Reuse before you write).
The table below is the complete inventory of those homes, grouped by layer from
``app`` inwards; read it before adding a helper, and add a row in the same
change that creates a home. ``tests/test_agent_harness.py`` checks that every
``mufinder.*`` path in this section names a module or package under
``src/mufinder``.
``tests/test_agent_harness.py`` pins, for every private method that two or more
workflow widget modules define, how many define it; the pin table is empty, so
no two workflow widgets may define the same private method.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Module
     - Owns
   * - ``mufinder.app.workflow_table_utils``
     - Workflow row tables (``create_workflow_table``,
       ``set_workflow_table_item``), the shared right-click menu
       (``standard_row_actions``, ``bind_reset_failed_context_menu``),
       the supercell column and its k-point suggestions (``SupercellColumn``),
       destructive confirmations (``confirm_destructive``), selection feedback
       (``SelectionFeedback``, ``selected_table_rows``), delete/backspace binding
       (``bind_delete_shortcuts``) and the locked-row rule
       (``workflow_row_is_locked``: rows with no job yet and ``cancelled``
       rows are editable, every other status locks the row), and energies in
       a results table (``format_energy_ev``).
   * - ``mufinder.app.selection_snapshot``
     - Targeting for in-row controls (``SelectionSnapshotPushButton``,
       ``SelectionSnapshotToolButton``, ``resolve_target_keys``): which rows an
       in-row button acts on when its row is, or is not, part of a multi-row
       selection, and the three-way prompt.
   * - ``mufinder.app.workflow_row_widget``
     - The label-keyed row-table widget base that pristine relaxation, muon
       relaxation and UEP subclass (``RowTableWidget``, and
       ``RelaxationRowTableWidget`` for the two geometry relaxations; see Row
       table widgets), and the source-keyed table base that quantum
       corrections and energy barriers subclass (``SourceTableWidget``; see
       Source table widgets).
   * - ``mufinder.app.sweep_convergence_widget``
     - The sweep-keyed widget base that the cutoff and k-point convergence
       widgets subclass (``SweepConvergenceWidget``; see Sweep convergence
       widgets), parametrised by a ``SweepKeyCodec`` from
       ``mufinder.app.convergence_types``.
   * - ``mufinder.app.workflow_parameters``
     - Parameter specs per workflow stage
       (``build_workflow_parameter_surface``,
       ``geometry_relaxation_parameter_specs``, ``clone_spec_for_workflow``),
       the shared-row ordering, schema and project defaults
       (``workflow_default_params``, ``drop_default_conflicts``,
       ``drop_explicit_conflicts``, ``seed_xc_functional_default``),
       sanitising and restoring persisted parameter values
       (``sanitize_parameter_mapping``, ``restore_structured_value``), the
       values rows edited together share or differ on
       (``shared_parameter_values``, ``mixed_parameter_names``) and the
       species-keyword options of a structure (``structure_species_options``,
       ``merge_species_options``).
   * - ``mufinder.app.workflow_config_codecs``
     - The persistence codecs for each workflow's stored
       ``CalculationDefinition.parameters["configuration"]``
       (``serialize_*_config`` and ``deserialize_*_config`` for UEP, pristine
       relaxation, muon relaxation and convergence),
       ``normalized_supercell_matrix``, and the per-row
       ``configuration["row_settings"]`` entries of the source-keyed workflows
       (``workflow_row_configuration``, ``workflow_row_configuration_exists``,
       ``num_farms_from_extras``, ``row_settings_payload``,
       ``batch_row_payload``, ``store_workflow_row_configuration``). It sits
       in ``app`` because the configuration dataclasses and
       ``restore_structured_value`` do.
   * - ``mufinder.app.workflow_dialogs``
     - The row parameter dialog (``run_row_parameter_dialog``, driven by a
       ``RowParameterRequest`` and returning a ``RowParameterResult``), the
       source tables' Configure dialog runner (``run_source_configure_dialog``,
       driven by a ``SourceConfigureRequest`` and returning a
       ``SourceConfigureResult``) and the workflow configure dialogs.
   * - ``mufinder.app.param_dialog``
     - The CASTEP parameter dialog the configure dialogs build on
       (``ParameterDialog``).
   * - ``mufinder.app.adapter_common``
     - GUI value dict to ``InputTemplate``, k-point rescaling for supercell
       rows, the adapters' shared expansion prologue, supercell-matrix parsing
       and labels, and the ``AdapterExpansion`` return shape (see Workflow
       adapters).
   * - ``mufinder.app.workflow_submission``
     - Row submission planning, ``effective_row_status`` and the dismissed-row
       rules.
   * - ``mufinder.app.workflow_controllers.base``
     - The submit, materialize, result and refresh cycle (see Workflow
       controllers), ``dedupe_jobs`` for the jobs a ``materialize`` or
       ``jobs_for_keys`` returns, the active-host guard on a submission's
       structures and sites (``confirm_active_host``), derived-structure
       upsert and row-parameter mirroring.
   * - ``mufinder.app.workflow_controllers.sweep_convergence``
     - The cycle the two convergence sweeps share
       (``SweepConvergenceController``).
   * - ``mufinder.app.project_defaults_widget``
     - The project-defaults strip in every workflow header (see Project
       defaults surface).
   * - ``mufinder.app.theme``
     - Reusable styling, palettes and status brushes.
   * - ``mufinder.app.plot_canvas``
     - Embedded matplotlib plots (``create_plot_canvas``, ``PlotFrame``; see Plot
       canvas policy).
   * - ``mufinder.services.job_execution_controller``
     - Job state transitions: submission, start, completion, failure, cancellation
       and retrieval (``JobExecutionController``; see :doc:`job_lifecycle`).
   * - ``mufinder.services.job_paths``
     - Where a job's files live: ``definition_output_directory``,
       ``job_output_directory`` (the definition mappings that claim a job, then
       the ``files/`` fallbacks), ``job_seed_name`` and ``job_output_files``.
   * - ``mufinder.services.job_results``
     - Scalar readings from the result of a job's latest run
       (``total_energy_for_job``, ``max_force_for_job``,
       ``max_stress_for_job``) and the payload extractors behind them.
   * - ``mufinder.services.job_artifacts``
     - Job output paths and parsed secondary outputs.
   * - ``mufinder.services.definition_mappings``
     - Keeping a workflow's definitions consistent: copying job bindings from
       equivalent definitions (``sync_equivalent_mappings``, with the two
       binding priority orders), dropping rows and their bindings
       (``remove_labels_from_definition``, ``remove_definition_if_no_rows``),
       the sweep and source-row equivalents, and dropping generated paths
       from the Energy Barriers form state (``remove_generated_sources``; see
       Definition mappings and row removal).
   * - ``mufinder.services.cluster_submission_planner``
     - The decisions of a cluster submission: stranded quantum phonon jobs,
       connection reuse, each job's remote target (start again, continuation,
       resubmission, quantum stage-1 directory or a fresh directory), the
       overwrite prompts to ask as data and their *Yes/No to All* resolution,
       the protected-directory rule, and the ``ClusterJobSubmission`` items
       with the failed, skipped and missing jobs
       (``plan_cluster_submission``). The window keeps the dialog, the
       prompts and the message boxes.
   * - ``mufinder.services.source_candidates``
     - Read-only source queries over the project document: the relaxed muon
       sites eligible for quantum corrections, their relative energies,
       labels, seed names, stored ``sources`` entries and two-stage state, and
       the settings a follow-on row inherits from its source relaxation's
       input files (``source_relaxation_parameter_values``); the generated
       diffusion paths, their labels, stored form and preview supercell, the
       Energy Barriers form state they live in and the NEB results for energy
       barriers; and the completed pristine and muon relaxations offered to
       comparison and relaxation analysis.
   * - ``mufinder.workflow.naming``
     - Job names (``job_name``) and the remote naming of a cluster
       submission: the workflow directory fragment, the default scheduler
       job-name prefix and remote path sanitising.
   * - ``mufinder.workflow.energy_barriers``
     - Besides NEB expansion, the merge and per-pair replacement of serialized
       diffusion-path sources and the removal of stored placeholder
       transition-state parameters.
   * - ``mufinder.science.lattice``
     - Lattice algebra: scalar and broadcast fractional/Cartesian conversion,
       lattice inversion, minimum-image displacements and distances
       (``minimum_image_distance``, ``minimum_image_distance_matrix``,
       ``nearest_image_distance``) and periodic deduplication
       (``deduplicate_periodic_positions``).
   * - ``mufinder.science.spglib_compat``
     - spglib access: the ``(lattice, positions, numbers)`` cell
       (``spglib_cell``), cached symmetry operations
       (``symmetry_operations``) and symmetry orbits (``symmetry_orbit``),
       with tolerances passed by each caller.
   * - ``mufinder.science.ase_bridge``
     - Structure to ASE ``Atoms`` (``structure_to_ase_atoms`` with
       ``skip_muon``, ``sanitize_symbols`` and ``record_species_labels``) and
       the muon-label and element-symbol rules.
   * - ``mufinder.castep_io``
     - CASTEP syntax: cell and param parsing and writing, keyword metadata
       (``schema``) and validation (``validators``); see
       :doc:`castep_keyword_catalog`.
   * - ``mufinder.domain.magnetic_properties``
     - Spins and propagation vectors read from ``Structure.properties``
       (``atomic_spins_from_structure``,
       ``atomic_spins_from_magnetic_moments``,
       ``propagation_vector_from_structure``, ``has_nonzero_spins``).

Widget contract
~~~~~~~~~~~~~~~

A workflow widget that stages jobs follows one shape, so ``MainWindow`` and the
controllers treat every workflow alike:

- **Signals.** ``form_state_changed``, ``configure_requested``,
  ``submit_local_requested``, ``submit_cluster_requested``, ``cancel_requested``,
  ``inspect_failed_requested``, ``reset_failed_requested`` and
  ``remove_requested`` (the convergence sweeps name it
  ``remove_cutoff_requested`` / ``remove_grid_requested``). The submit
  signals carry the selected keys, or every key when nothing is selected, and
  the run buttons' labels say which.
- **Methods.** ``collect_state`` / ``set_form_state`` round-trip the persisted
  form state, and ``set_form_state`` suppresses ``form_state_changed`` while it
  restores; ``set_runtime_states``, ``set_parameter_species_options``,
  ``set_supercell_options``, and ``set_project_defaults`` or
  ``set_xc_functional_default`` push project state into the widget.
- **Persisted form state.** ``ui_state["<workflow>_form"]`` holds a job-name
  prefix and a row list. Each row stores its label, its supercell matrix as a
  3×3 list, its parameter values as the diff from the implicit defaults (a UEP
  row stores the dialog's values with its pinned task and density flag), and
  its sorted ``added_advanced`` names. The two convergence sweeps keep their
  own key-list shape instead (see Sweep convergence widgets).
- **Rows.** The row id lives in the label item's ``UserRole``; per-row controls
  are ``SelectionSnapshot`` buttons that resolve their targets through the shared
  targeting rule; a delete-shortcut handler returns early while the table is
  editing a cell; a row whose job is in a runtime state is locked by
  ``workflow_row_is_locked`` and its parameter dialog opens read-only. A
  ``cancelled`` row is editable but is not submitted from the table: the row
  menu's Run entries follow ``READY_WORKFLOW_ROW_STATUSES``, the submit filter.
- **Supercells.** A widget with a supercell column (pristine relaxation, muon
  relaxation, UEP) delegates it to one ``SupercellColumn``. The column holds the
  project supercell options, each row's matrix and the project k-point
  reference; it builds every row's combo (``Unit cell``, the project options,
  the row's own matrix when it is not an option, ``Manage Supercells...``),
  applies a choice to the targeted rows through the shared targeting rule, and
  derives the row's suggested k-point grid, the grid's equivalent spacing for
  the parameter dialog, and the grid-adjustment message. The widget supplies
  only its hooks: normalise row parameters, refresh rows, emit state, the
  manager request and which rows may change (locked and read-only rows may
  not).
- **Row parameter dialog.** Configure and inspect go through
  ``run_row_parameter_dialog``. The widget supplies the targeted rows' effective
  parameters and advanced names, the dialog seed, its locked values and
  sanitiser; the launcher titles the dialog ``Configure …`` or ``Inspect …``,
  applies the k-point grid/spacing preference, marks parameters that differ
  between the rows as mixed, sanitises what the dialog returns and resolves it
  per row, keeping each row's own value for a mixed parameter left untouched.
  It returns ``None`` on cancel and for a read-only inspection; the widget
  only stores the per-row result.
- **Parameters.** Specs exclude ``task``; defaults are seeded from the schema
  and the project defaults; effective parameters carry exactly one of
  ``kpoint_mp_grid`` / ``kpoint_mp_spacing``; every restore passes through the
  shared sanitiser (``sanitize_parameter_mapping``); the shared rows come first (see Workflow parameter surface
  policy).
- **Layout.** A header row with the title, subtitle and project-defaults strip;
  a controls row (``Add...``, a count label, ``Remove Selected``,
  ``Configure Selected Parameters...``); and a run row with the job-name prefix
  and the two run buttons.

The quantum-corrections and energy-barriers widgets list project sources
rather than staged rows, so they follow the run, Configure and row-menu parts
of this contract through ``SourceTableWidget`` (see Source table widgets) and
have no form state, supercell column or Add button.

Row table widgets
~~~~~~~~~~~~~~~~~

Pristine relaxation, muon relaxation and UEP subclass one base,
``mufinder.app.workflow_row_widget.RowTableWidget``, which implements the widget
contract above once for a table of labelled job rows. A row is keyed by its
row id; it is submitted, tracked and reset under its *submission key*, the
label (muon relaxation qualifies it with the charge state, and keeps one table
per charge state).

The base owns the table and the shared cells (label, supercell, status,
Configure, remove), row lookup, the selection and the row menu, the delete
shortcuts, per-row Configure and remove targeting, removal with its
confirmation, the parameter dialog, runtime refresh and the locked-row rule,
config validation (empty and duplicate labels, then each row's parameters)
and submission, the form-state plumbing (``_restoring_form`` clears the rows,
re-normalises them afterwards and emits the restored state once) and the page
layout. ``RelaxationRowTableWidget`` adds the geometry-relaxation parameter
surface, project-default seeding, row values stored as the diff from the
row's implicit defaults, and the "Continue relaxation…" row action
(``set_continuable_predicate``, ``continue_relaxation_dialog_seed``).

A subclass declares its columns, workflow stage, pinned parameters, dialog
wording and empty-row status as class attributes and implements three public
hooks, as a controller implements its hooks: ``create_row_cells`` and
``refresh_row_cells`` for its own columns, and ``config_for_rows`` for its run
configuration from the validated rows. It keeps its own ``collect_state`` and
``set_form_state``, built from the base's row helpers, and calls
``_build_layout`` (with its title, extra header widgets, table view and run-row
extras) and ``_connect_signals`` at the end of ``__init__``. A behaviour only
one workflow has is an override of a private base method with a default:
muon relaxation's read-only imported rows (``_row_is_read_only``,
``_configurable_row_ids``, ``_removable_row_ids``), its muonium spin default
(``_row_default_overrides``) and its charge-state tables and keys; UEP's
parameter surface and k-point rule. Never copy a base method into a
subclass to change it; add the hook.

Source table widgets
~~~~~~~~~~~~~~~~~~~~

Quantum corrections and energy barriers subclass one base,
``mufinder.app.workflow_row_widget.SourceTableWidget``. Their rows are not
staged by the user: each row is a source read from the project through
``mufinder.services.source_candidates`` (a relaxed muon site, a diffusion
path), shown in the order the widget's ``set_sources`` receives, and
submitted, configured and tracked under the row's source key. The source
rows are frozen dataclasses carrying at least ``key``, ``status`` and
``configured``.

The base owns the key lookups (``selected_source_keys``, the key at a row,
the keys at a set of rows), the run targeting (the selection, else every
row), Configure Selected, the source table's row menu (Run, Cancel, Inspect,
Configure, an optional Remove and failed-job reset, then the widget's extra
entries), the table fill with the per-status colour of the Status cell and
the per-row Configure button with the shared targeting rule, and the
action-state refresh that enables Configure Selected from the selection.

A subclass declares its status and Configure columns, the workflow named on
the Configure button and the names of its configure and run signals as class
attributes, so each widget keeps its own signal names. It implements three
public hooks: ``source_row_values`` (the text of the leading columns),
``create_source_row_cells`` (its own cells) and ``refresh_source_actions``
(its own actions for the selected rows). Quantum corrections adds the stage
and farm columns, the editable site label and "Reset to Stage 1"; energy
barriers adds the endpoint pair that generates paths, the View column, path
removal and the NEB inspector, and its view is created lazily (see
Construction).

Sweep convergence widgets
~~~~~~~~~~~~~~~~~~~~~~~~~

The cutoff and k-point convergence widgets subclass one base,
``mufinder.app.sweep_convergence_widget.SweepConvergenceWidget``. A row is keyed
by its sweep value, typed into the key column (a cutoff energy in eV, or a
Monkhorst-Pack grid), and is submitted, tracked and removed under that key.
The sweep's ``SweepKeyCodec`` (``CUTOFF_KEY_CODEC``, ``KPOINT_GRID_CODEC`` in
``mufinder.app.convergence_types``) parses and formats the key column, names a
key in dialogs and warnings, normalises keys from signals, runtime states and
persisted state, and gives the key's JSON form.

The base owns the simulation-cell picker, the table (key, status, total
energy, max force, Configure, remove), the selection and row menu, the delete
shortcuts, per-row Configure and remove targeting, removal with its
confirmation (a row whose key does not parse is dropped in place; the others
are emitted on the widget's removal signal and removed through
``remove_keys``), the parameter dialog, runtime refresh with the locked-row
rule and the Examine button (enabled by two completed rows with energy and
force), config validation (unparseable, empty and duplicate keys, then each
row's parameters) and submission, and the form-state round trip. The
persisted form state keeps each sweep's own field names:
``ui_state["convergence_form"]`` holds ``cutoff_energies`` and
``cutoff_parameter_rows`` (each row's ``cutoff``), and
``ui_state["kpoint_convergence_form"]`` holds ``kpoint_grids`` and
``kpoint_parameter_rows`` (each row's ``grid``); both also hold the job-name
prefix, the generator dialog's ``range`` values, and per configured row its
parameters and sorted ``added_advanced`` names. Restoring lists every key the
key list or a configured row names, once, in that order.

A subclass declares its codec, workflow stage, sweep keyword, persisted field
names, removal signal name and wording as class attributes and implements
three public hooks: ``next_key`` (the key a new row starts from),
``prompt_new_key`` (what Add asks for: the cutoff sweep adds the next cutoff,
the k-point sweep opens a grid dialog) and ``generate_keys`` (the range or
spacing generator dialog). A behaviour only one sweep has is an override of a
private base method with a default: the cutoff sweep titles a one-row
dialog by its cutoff (``_dialog_subject``), reads a task-only row as the
defaults (``_is_default_parameter_set``) and restores legacy
``shared_params`` / ``overrides`` state (``_restore_legacy_parameters``); the
k-point sweep persists the cell its grids were generated for and warns when
the current cell differs (``_extra_state`` / ``_restore_extra_state``), and
drops the shared defaults that conflict with the project cutoff
(``set_cutoff_default``).

Controller naming
~~~~~~~~~~~~~~~~~

A workflow's definition stores its jobs in ``job_ids_by_<key>`` (and
``job_seed_names_by_<key>`` when seed names differ from labels), its calc type
is ``castep_<workflow>``, and ``build_states`` projects each job through
``effective_row_status`` so a ``defined`` job with configured inputs reads
``ready``.

Workflow controllers
--------------------

``mufinder.app.workflow_controllers`` is a thin controller layer inside the app
layer, between the ``MainWindow`` widgets and the services, workflow and science
layers. Each
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController` owns one
workflow's submit, materialize, result and refresh cycle. ``MainWindow`` keeps
only the two cross-workflow dashboard entry points, which fan out to every
controller; ``tests/test_agent_harness.py`` pins that set and fails if a
per-workflow ``_on_submit_*`` / ``_materialize_*`` handler appears on
``MainWindow``.

Construction
~~~~~~~~~~~~

A controller is a plain Python object (not a ``QObject``; it owns no signals)
built from three inputs:

1. the shared
   :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`:
   the narrow, late-bound callables every controller needs (document access,
   persistence, job dispatch, status reporting,
   ``prepare_jobs_for_resubmission``). Nothing workflow-specific goes in it;
2. an optional per-controller deps dataclass of required, precisely typed
   callables, defined in the controller's own module (for example
   ``ConvergenceCutoffDeps``);
3. the controller's own view. A controller whose view is created lazily takes
   a ``view_getter: Callable[[], Widget | None]`` instead and makes
   ``refresh_view`` a no-op while the getter returns ``None``; it is still
   constructed and registered at startup so a completed job dispatches through
   it whether or not the view was ever opened.

A controller holds no reference to ``MainWindow`` and never imports the
``main_window`` module, so it is unit-testable without a window.

Three class attributes declare which calculation definitions a controller owns:

- ``stage``: the :class:`~mufinder.domain.enums.WorkflowStage` of its
  definitions;
- ``definition_calc_type``: the calc type of its definitions, or ``None`` to
  accept any at ``stage``. It is distinct from ``calc_types``, which lists the
  calc types of the controller's *jobs* for result dispatch;
- ``active_key``: the ``ui_state`` key that stores the id of the definition the
  user last configured, or ``None`` when the newest definition is always the
  active one (quantum corrections, energy barriers).

From these the base class answers every definition and job lookup through the
read-only queries in :mod:`mufinder.services.project_service`, so no workflow
repeats them:

- ``definitions()`` returns the matching definitions, newest first
  (``definitions_for_stage``);
- ``active_definition()`` returns the definition named by ``active_key`` when
  it still matches, and otherwise the newest match (``active_definition``);
- ``job_for_key(definition, key, *, mapping_name="job_ids_by_label",
  calc_type=None)`` returns the job a definition's id mapping binds to a row
  key, or ``None`` when the mapping, the id or the job is missing or the job's
  calc type differs (``job_for_mapping_key``).

Controllers whose rows bind jobs differently wrap ``job_for_key``: the two
convergence sweeps read ``job_ids_by_cutoff`` / ``job_ids_by_grid`` and fall
back, for a definition written before the mapping, to legacy ``job_labels``
(cutoff) or the grid's job label under the configured prefix (k-point), muon
relaxation tries the charge-qualified key before the bare label, and energy
barriers accepts only NEB jobs.

Hooks
~~~~~

``submit`` is a template method; a subclass supplies workflow behaviour by
overriding these hooks.

.. list-table::
   :header-rows: 1
   :widths: 26 50 24

   * - Hook
     - Responsibility
     - Overridden by
   * - ``normalize(value)``
     - Turn a widget-emitted key into the controller's canonical key.
     - all
   * - ``build_states()``
     - Map every row key to its run state from the active definition's job
       mapping and the live form state.
     - all
   * - ``is_ready(states, key)``
     - Decide whether a key can be submitted.
     - muon relaxation
   * - ``jobs_for_keys(keys)``
     - Resolve keys to existing ``Job`` records. Each workflow table's row-menu
       Cancel, Inspect Failure and Reset Failed Jobs signals connect to one
       generic window handler per action, bound to the table's controller,
       which passes the selected keys here.
     - all
   * - ``materialize(keys)``
     - Register or update the definition, expand the jobs, and write input
       directories for ``defined`` jobs only.
     - all
   * - ``pre_submit_check(keys)``
     - Refuse the submission before any plan is built (host-mismatch guards).
     - muon relaxation, quantum corrections, energy barriers
   * - ``sync_mappings(definition)``
     - Copy job bindings into a definition from its equivalent definitions
       (see Definition mappings and row removal). The default does nothing.
     - convergence sweeps, pristine and muon relaxation, UEP
   * - ``remove_labels_from_definition(definition, labels)``
     - Drop the named rows and their job bindings from one definition and
       report whether it was deleted for having no row left. Called by the
       ``remove_rows(labels)`` template method for each of ``definitions()``.
     - k-point sweep, pristine and muon relaxation, UEP, energy barriers
   * - ``row_key_matcher(labels)``
     - The predicate picking the job-mapping keys of removed rows, which the
       window's job-file cleanup uses. The default matches a key whose string
       form is one of the labels.
     - muon relaxation
   * - ``submission_groups(jobs)``
     - Split the planned jobs into separately dispatched groups.
     - muon relaxation, quantum corrections
   * - ``validate_artifacts(job, parsed, run, artifacts)``
     - Return the message that fails a completed job before any result is
       built, or ``None`` to accept it (Result dispatch, below).
     - pristine relaxation, muon relaxation, UEP, quantum corrections, energy
       barriers
   * - ``handles_job(job)`` / ``build_result(...)``
     - Own a completed job's calc types and contribute its
       :class:`~mufinder.app.workflow_controllers.base.ControllerResult`.
     - every controller except the two convergence sweeps
   * - ``refresh_view()``
     - Push the latest states into the widget.
     - all
   * - ``update_row_parameters(...)``
     - Edit one submitted row's CASTEP parameters in place (below).
     - pristine relaxation, muon relaxation

Submission sequence
~~~~~~~~~~~~~~~~~~~

``submit`` runs ``pre_submit_check``, then ``build_states``, then builds the
plan (ready keys and skipped rows), then ``materialize``, then the
dismissed-job reset, then ``materialize`` again for the plan's ready keys, then
one backend call per ``submission_groups`` group, and finally ``refresh_view``
with a status message naming the skipped count.

The dismissed-job reset is the only step that mutates job state before
dispatch. ``submit`` partitions the planned jobs with
:func:`~mufinder.app.workflow_submission.job_is_resubmittable_after_dismissal`
and, when any row's bound job was dismissed from the dashboard, calls the
context's ``prepare_jobs_for_resubmission`` with exactly those jobs. The hook
runs the stale-output cleanup prompt and resets them to ``defined``; returning
``False`` (the user cancelled) aborts the submission silently. The second
``materialize`` pass exists because controllers only regenerate input
directories for ``defined`` jobs and the first pass ran before the reset. Both
steps act on the plan's own job objects, so a row never gains a duplicate job,
and non-dismissed failed jobs are never passed to the hook (see
:doc:`job_lifecycle`, Dismissed rows stay resubmittable).

Task farming for the quantum-phonon and NEB stages is applied by the shared
local submit backend, which resolves each job's farm count by calc type; the
runners use the same count to decide whether a finished job's outputs carry
CASTEP's ``_farmNNN`` seedname suffix.

Result dispatch
~~~~~~~~~~~~~~~

Dispatch is registry-based: the job-finished handler calls
:func:`~mufinder.app.workflow_controllers.base.validate_controller_artifacts`
and then
:func:`~mufinder.app.workflow_controllers.base.merge_controller_result`. Both
ask the registered controllers ``handles_job`` and use the first that claims the
job. The merge additively combines the owning controller's
:class:`~mufinder.app.workflow_controllers.base.ControllerResult`: the result
payload, derived structures, and sanctioned document side effects such as
rewriting a muon site's status, run id and total energy.

**Parse once, off the GUI thread.** A completed job's secondary outputs (the
relaxed ``-out.cell``, the ``.phonon`` file with its derived quantum
quantities, and the ``.ts`` transition-state file) are parsed once, on the
runner worker thread. Each runner calls
:func:`~mufinder.services.job_artifacts.collect_job_artifacts` after the primary
``.castep`` parse succeeds and delivers a
:class:`~mufinder.services.job_artifacts.ParsedJobArtifacts` carrier with the
``job_finished`` signal. Collection follows the
:class:`~mufinder.services.job_artifacts.RetrievalFlags` the runners track
(per-calc-type output flags plus the farm count; see :doc:`job_lifecycle`),
and every failure travels as data (missing-file flags and
error strings), never as an exception across the signal boundary. Callers with
no carrier (reconciliation, retry, direct test invocation) fall back to one
synchronous ``collect_job_artifacts`` call. ``build_result`` consumes the
carrier and never re-reads disk.

**Failing a completion.** ``ControllerResult`` cannot fail a job; the owning
controller's ``validate_artifacts(job, parsed, run, artifacts) -> str | None``
does. The job-finished handler resolves the carrier and the job's run, then
dispatches the hook once. A returned message fails the job through the shared
job-failure path with that text, and no result is built; ``None`` lets
``build_result`` run. The hook reads the carrier's missing-file flags and error
strings and never re-reads a file the carrier holds. The overrides are:

- pristine and muon relaxation: the geometry optimisation did not converge, or
  the ``-out.cell`` is missing, unparseable, or older than the current run (a
  continuation reuses its directory). Both share
  :func:`~mufinder.app.workflow_controllers.base.relaxation_artifact_failure`;
  muon relaxation also fails a relaxed cell with no ``H:mu`` position;
- UEP: ``{seed}.den_fmt`` is missing (a plain existence check, because the
  density is not parsed on completion);
- quantum corrections: a phonon job's ``.phonon`` is missing or unparseable,
  including a ``PhononParseError`` from the ZPE, gamma-point or witness
  computation;
- energy barriers: an NEB or transition-state job's ``.ts`` is missing or
  unparseable.

``build_result`` therefore contributes only the success-path payload. A new
failure check goes in the owning controller's ``validate_artifacts`` and reads
the carrier rather than disk.

Controller summary
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Controller
     - Specifics
   * - ``ConvergenceCutoffController`` /
       ``ConvergenceKpointController``
     - Sweep workflows keyed by cutoff energy and by k-point grid, with jobs
       mapped in ``job_ids_by_cutoff`` / ``job_ids_by_grid``. Both subclass
       ``SweepConvergenceController``
       (``mufinder.app.workflow_controllers.sweep_convergence``), which owns
       ``build_states``, ``jobs_for_keys``, ``materialize`` and the form
       snapshot (the live widget state first, then the persisted form); the
       concrete controllers declare the key codec, sweep parameter, job calc
       type, mapping name and form-state key, and supply ``mapping_key``,
       ``parse_mapping_key`` (its inverse), ``sweep_point_signatures`` (a
       point's job-name prefix and effective parameters, which
       ``sync_mappings`` compares) and ``legacy_job_for_key``. The cutoff
       sweep removes one cutoff at a time through
       ``remove_cutoff_from_definition``. The definition a key
       belongs to is the active definition when its configuration sweeps that
       key. No ``build_result``: the convergence metrics come from the primary
       ``.castep`` parse.
   * - ``PristineRelaxationController``
     - Keyed by row label; rows sync by row signature under the relaxation
       priority order. ``build_result`` reads the carrier's relaxed cell,
       contributes ``lattice_a`` / ``lattice_b`` / ``lattice_c``, and creates
       or refreshes the derived relaxed structure.
   * - ``MuonRelaxationController``
     - Keyed by ``"{charge prefix}:{label}"``. ``sync_mappings`` reads a
       bare-label key as its row's qualified key, carries only jobs that fit
       the key's charge state and have left ``defined``, and copies each
       winner's ``job_seed_names_by_label`` entry; row removal also clears
       seed names. ``pre_submit_check`` requires
       every staged site to match the active host; ``submission_groups``
       dispatches positive-muon jobs and then muonium jobs; ``build_result``
       adds the muon's final fractional position (supercell and folded host
       coordinates) and rewrites the source site's status, run id and total
       energy.
   * - ``UepController``
     - Keyed by row label; rows sync by row signature under the convergence
       priority order. ``build_result`` contributes the
       ``density_available`` flag only.
   * - ``QuantumCorrectionsController``
     - Two stages per source: singlepoint, then phonon. ``materialize``
       dispatches each source to its current stage (``stage_state``) and
       writes ``singlepoint_job_ids_by_label`` / ``phonon_job_ids_by_label``
       plus ``job_seed_names_by_label``; ``pre_submit_check`` checks the active
       definition against the active host (``confirm_active_host``) and a
       stage-2 row's stage-1 muon mass; ``build_result`` derives the muon ZPE,
       gamma-point eigenvalues, mode participations and the ``w1`` witness from
       the carrier's parsed phonon calculation, and a singlepoint job
       contributes an empty result. Both calc types are ``handles_job``-owned.
       Its source rows come from :mod:`mufinder.services.source_candidates`.
       It also answers, for the cluster submission, where a phonon job's
       stage 1 ran and which remote directory it continues in
       (``stage_one_target``, ``continuation_remote_target``), and lists the
       rows "Reset to Stage 1" may rewind (``stage_one_resets``).
   * - ``EnergyBarriersController``
     - Keyed by diffusion-path source key, stamped on each spec as
       ``provenance={"source_key": key}``. ``materialize`` aligns the endpoints
       into the pristine host and seeds the middle image by IDPP before
       expanding the NEB band; ``pre_submit_check`` validates the endpoints
       against the active host; ``build_result`` reads the carrier's parsed
       transition-state path and contributes ``transition_state_file``,
       ``barrier_height_eV``, ``num_images``, ``image_labels``,
       ``reaction_coordinates`` and ``image_energies_eV``. Both the NEB and the
       transition-state-search calc types are ``handles_job``-owned. Takes a
       ``view_getter`` because its view is created on first visit. Its
       generated paths, NEB sources and completed NEB results come from
       :mod:`mufinder.services.source_candidates`. It generates paths between
       the selected endpoints into the form state (``generate_paths``), drops
       them from the form state and from both its NEB definitions and the
       transition-state-search sets older configurations stored them in
       (``remove_rows``), and drives the path viewer and NEB preview
       (``path_preview``, ``show_neb_preview``).

Source table controllers
~~~~~~~~~~~~~~~~~~~~~~~~

The quantum-corrections and energy-barriers controllers back the two
``SourceTableWidget`` subclasses, so they share a small contract beyond the
hooks above:

- ``candidates()`` returns the table's sources, ``ensure_definition(keys)``
  returns the active definition with the given sources recorded (creating it
  on first use) and ``definition_and_key_for_job(job)`` finds the row a job
  belongs to, which the submit backends use to read the row's farm count.
- ``configure_request(keys)`` returns the Configure dialog's seed, a
  ``SourceConfigureRequest`` (each row's effective parameters, the dialog's
  title and intro text, whether it opens read-only, the agreed farm count and
  the species options); the window builds the workflow's dialog, runs it
  through ``run_source_configure_dialog`` and passes the result to
  ``apply_configuration(keys, result, extras)``, which stores each row's
  ``row_settings`` entry (a mixed parameter left untouched keeps each row's
  own value), refreshes the view and persists.
- A row's effective parameters layer its stored ``row_settings`` over the
  settings of its source relaxation
  (``source_relaxation_parameter_values``); an energy-barrier path also
  starts from the project defaults and rounds its image count up to an odd
  number.

Their deps hold only what needs the window: the host-mismatch dialogs
(``handle_structure_host_mismatch``, ``handle_site_host_mismatch``), and a
yes/no question (quantum corrections) or a warning (energy barriers). The
window keeps the dialogs themselves, the job-file cleanup before a path is
removed or a row is reset to stage 1, and the phonon analysis windows.

Definition mappings and row removal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reconfiguring a workflow registers a new definition, so one workflow context
can hold several definitions, each binding rows to jobs in its own
``job_ids_by_<key>`` mapping. :mod:`mufinder.services.definition_mappings`
keeps them consistent; the controllers pass it their configuration codec and
row helpers, and ``MainWindow`` never edits a mapping or a definition's rows
itself.

``sync_equivalent_mappings`` runs before a controller reads its active
definition's bindings. A definition is a source when it shares the target's
context (stage, calc type, input structures and output directory for the row
workflows; input structure, output directory, simulation cell mode and
supercell source for the sweeps) and its configuration deserialises. A
source's binding is a candidate when its row has the same signature in both
configurations; for the row workflows a key neither configuration lists also
matches, for the sweeps both must list it. The highest-priority candidate per
key wins, and it replaces the target's own binding only when that is missing
or ranks strictly lower. Two priority orders exist: in flight, then completed,
then failed, stopped or cancelled, then defined for the sweeps and UEP
(``convergence_mapping_job_priority``); the relaxations rank a defined job
above a failed one (``relaxation_mapping_job_priority``). A target with no
candidate is left untouched.

Removing rows is ``remove_rows(labels)`` on the controller, after the window
has confirmed the removal and asked what to do with the rows' job files.
``remove_labels_from_definition`` drops the matching configuration rows
(rewriting the configuration through the codec only when one is dropped) and
the matching keys of the named mappings, then ``remove_definition_if_no_rows``
deletes a definition whose ``rows`` list is empty and clears the workflow's
``active_key`` when it names it. ``remove_rows`` returns the deleted
definitions' ids. The k-point sweep drops grids from ``sweep_values`` and
keeps the definition; the cutoff sweep drops one cutoff at a time from the
equivalent definitions and deletes a definition with none left; the
energy-barrier definitions drop sources and their row settings through
``remove_source_keys_from_definition`` and are deleted with no source left,
and the removed paths also leave the Energy Barriers form state
(``remove_generated_sources``).

In-place row parameter edits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Editing one already-submitted relaxation row's CASTEP parameters goes through
``update_row_parameters`` on the pristine or muon controller (keyed by row label
and by the charge-qualified key respectively), never through
``register_*_relaxation_definition``. Registering produces a new
``CalculationDefinition`` with an empty ``job_ids_by_label``, which severs the
row-to-job binding and lets ``materialize`` mint a duplicate job onto the row's
existing directory. The helper rewrites that single row inside the active
definition's ``parameters["configuration"]`` through the serialize/deserialize
pair (so the stored shape stays canonical), leaves every mapping and every
other row untouched, mirrors the edit into the widget form state with
:func:`~mufinder.app.workflow_controllers.base.apply_row_parameters_to_form_state`,
and stores back the values the widget reports, because the tables strip
implicit project defaults and ``build_states`` matches each row's form
signature against its definition signature. ``ui_state`` is then refreshed
from the widget and persisted through the context.

Workflow parameter surface policy
---------------------------------

Workflow configuration dialogs should present shared CASTEP parameters first,
in a stable cross-workflow order, before any workflow-specific controls.

The shared prefix lives in ``mufinder.app.workflow_parameters`` and currently
orders the common rows as:

- ``basis_precision``
- ``cut_off_energy``
- ``elec_energy_tol``
- ``kpoint_mp_grid``

Workflow-specific rows should follow after that common prefix. Advanced or
rarely adjusted controls such as ``fine_cut_off_energy`` should remain advanced
unless a workflow has a strong reason to expose them by default.

When adding a new workflow parameter surface, use the shared helper in
``mufinder.app.workflow_parameters`` instead of encoding another local ordering
rule in an individual widget.

Plot canvas policy
------------------

Embedded matplotlib plots are created through
``mufinder.app.plot_canvas.create_plot_canvas``, which returns a figure with
constrained layout and a canvas with a minimum size. Margins set as figure
fractions (``Figure.subplots_adjust``) reserve a fixed *share* of the canvas,
while tick and axis labels need a fixed number of *pixels*. On a small screen (a
13" laptop, or Windows at 125-150 % scaling) the dialogs open near their
minimum size, the share falls below what the labels need, and Qt clips them.
That includes the minus signs of negative tick values, so a clipped axis can
show the wrong sign. Constrained layout recomputes the margins from the
rendered labels on every resize, and the minimum canvas size stops a splitter
or box layout from squeezing a plot into an unreadable strip.

Rules for plot widgets:

- build new figures with ``create_plot_canvas``; pass a larger
  ``minimum_size`` when a plot needs more room (the magnetic field distribution
  uses 400x280), never a smaller one;
- do not call ``subplots_adjust``, and do not switch a figure's layout engine
  off on small windows; that is exactly when the layout engine is needed;
- keep axis labels short enough to fit the plot at its minimum size, since no
  layout engine can fit a label longer than its axis;
- place a legend outside the axes only through the axes' own ``legend`` call,
  so constrained layout reserves room for it.
- put a plot that shares a dialog or tab with other content in a
  ``PlotFrame`` (also in ``mufinder.app.plot_canvas``), with a short title. Its
  corner button opens the plot in a separate resizable window with zoom, pan
  and save; the window borrows the live canvas, so the plot keeps updating and
  stays interactive, and it returns when the window or dialog closes. While
  it is out, any navigation toolbar the dialog had on that canvas is switched
  out of pan/zoom and disabled, so only one set of tools drives the canvas, and
  drag handlers of their own (like the UEP dendrogram cut line) should stand
  down whenever the canvas's ``widgetlock`` is held. The
  window is a dialog-type window (``POP_OUT_WINDOW_FLAGS``): many plot dialogs
  run with ``exec()``, and while a modal dialog runs macOS keeps an ordinary
  ``Qt.Window`` child behind it and will not bring it forward. A dialog
  whose only content is its plot needs no frame. Extra controls that belong on
  the plot go in the frame's overlay (``add_overlay_widget``) so they do not
  cover the pop-out button;
- size markers and labels that must fit inside the plot (such as the
  relaxation clustering graph's nodes) in data units, as patches, rather than
  in points: a point-sized marker keeps its size as the canvas shrinks and
  spills past the axes.

``tests/test_agent_harness.py`` rejects ``subplots_adjust`` anywhere in
``mufinder.app`` and pins the modules that still build a ``FigureCanvasQTAgg``
directly (the set may only shrink). ``tests/app/test_plot_canvas_layout.py``
opens each plot dialog at its minimum size and at small-laptop window sizes and
fails if any tick label, axis label, title or legend extends past its canvas,
or if a canvas is squeezed below its minimum size. Add a new plot dialog to its
``PLOT_DIALOGS`` table; ``tests/app/test_plot_frame.py`` then also checks that
each of its plots can be popped out.

Project defaults surface
------------------------

The project-wide calculation defaults (XC functional, cutoff energy, k-point
grid) are shown by one shared widget,
``mufinder.app.project_defaults_widget.ProjectDefaultsStrip``, placed in the
header of every workflow tab that consumes them. Workflow widgets expose it as
``project_defaults_strip`` and must not build their own "Project Defaults"
labels: ``MainWindow`` pushes one ``ProjectDefaultsSummary`` (built from the
``Project`` record) to every strip, so the tabs cannot drift apart.

The strip only *displays* defaults and requests edits through its
``xc_functional_change_requested``, ``cutoff_energy_change_requested`` and
``kpoint_grid_change_requested`` signals (``None`` clears); ``ProjectService``
owns the mutation and its provenance record. A hand-entered cutoff or grid is
stored with ``MANUAL_DEFAULT_METHOD`` (``mufinder.domain.models``) as its
method. Seeding a workflow widget's row defaults
is a separate step: the widgets' ``set_project_defaults`` /
``set_xc_functional_default`` methods update the shared row defaults, using
``seed_xc_functional_default`` from ``mufinder.app.workflow_parameters`` for
the functional. When adding a workflow that consumes project defaults, give
its widget a strip, add it to ``MainWindow._project_defaults_strips``, and pass
``project_xc_functional`` to its adapter alongside the cutoff and k-point
defaults.

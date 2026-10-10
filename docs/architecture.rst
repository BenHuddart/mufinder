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
   ``ConvergenceWorkflow`` and ``MuonSiteWorkflow`` remain as deprecated class
   wrappers over their ``expand_*`` functions (see :doc:`compatibility_policy`).

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
(the ``ClusterConfig`` record and its ``QSettings`` persistence, used by
services and the app). ``castep_setup`` and ``cluster_setup`` depend on
``PySide6.QtCore`` but not on widgets; keep new code out of this top level
unless it is genuinely cross-cutting configuration.

Refactoring policy
------------------

When duplicated UI code starts to encode workflow, execution, or styling
policy, move the reusable part into the appropriate shared layer:

- workflow-row planning belongs in ``mufinder.app.workflow_submission`` or a
  lower non-UI service once it does not depend on widgets
- any other cross-workflow UI concern belongs in the module listed under
  Shared workflow infrastructure below
- job lifecycle transitions belong in ``mufinder.services``
- scientific calculation behavior belongs in ``mufinder.science``
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
- :func:`~mufinder.app.adapter_common.rescale_kpoint_grid` rescales a project
  k-point grid to an equivalent reciprocal-space density for supercell rows and
  returns an optional adjustment message. Convergence sweeps a fixed structure
  and never rescales.
- :class:`~mufinder.app.adapter_common.AdapterExpansion` is the uniform return
  shape (``jobs`` plus UI-facing ``messages``) for every adapter's public
  ``expand_*`` function. ``main_window`` surfaces non-empty ``messages`` from
  the configure path.

``adapter_common`` must remain free of PySide imports; it depends only on
``workflow``, ``castep_io``, ``science``, and ``domain``.

Shared workflow infrastructure
------------------------------

The workflow pages are built from one set of shared helpers. Each
cross-workflow concern has exactly one home; a workflow extends the helper there
rather than carrying a private copy (``AGENTS.md``, Reuse before you write).
``tests/test_agent_harness.py`` pins, for every private method that two or more
workflow widget modules define, how many define it; the counts may only fall.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Module
     - Owns
   * - ``mufinder.app.workflow_table_utils``
     - Workflow row tables (``create_workflow_table``,
       ``set_workflow_table_item``), the shared right-click menu
       (``standard_row_actions``, ``bind_reset_failed_context_menu``),
       destructive confirmations (``confirm_destructive``), selection feedback
       (``SelectionFeedback``), delete/backspace binding
       (``bind_delete_shortcuts``) and the locked-row rule
       (``workflow_row_is_locked``: rows with no job yet and ``cancelled``
       rows are editable, every other status locks the row).
   * - ``mufinder.app.selection_snapshot``
     - Targeting for in-row controls (``SelectionSnapshotPushButton``,
       ``SelectionSnapshotToolButton``, ``resolve_target_keys``): which rows an
       in-row button acts on when its row is, or is not, part of a multi-row
       selection, and the three-way prompt.
   * - ``mufinder.app.workflow_parameters``
     - Parameter specs per workflow stage
       (``build_workflow_parameter_surface``, ``clone_spec_for_workflow``), the
       shared-row ordering, schema and project defaults
       (``workflow_default_params``, ``drop_default_conflicts``,
       ``seed_xc_functional_default``), and sanitising and restoring persisted
       parameter values (``sanitize_parameter_mapping``,
       ``restore_structured_value``).
   * - ``mufinder.app.workflow_dialogs`` / ``mufinder.app.param_dialog``
     - The row parameter dialog (``run_row_parameter_dialog``, driven by a
       ``RowParameterRequest`` and returning a ``RowParameterResult``) and the
       configure dialogs.
   * - ``mufinder.app.adapter_common``
     - GUI value dict to ``InputTemplate``, k-point rescaling for supercell
       rows, and the ``AdapterExpansion`` return shape (see Workflow adapters).
   * - ``mufinder.app.workflow_submission``
     - Row submission planning, ``effective_row_status`` and the dismissed-row
       rules.
   * - ``mufinder.app.workflow_controllers.base``
     - The submit, materialize, result and refresh cycle (see Workflow
       controllers), derived-structure upsert and row-parameter mirroring.
   * - ``mufinder.app.project_defaults_widget``
     - The project-defaults strip in every workflow header (see Project
       defaults surface).
   * - ``mufinder.app.theme`` / ``mufinder.app.plot_canvas``
     - Styling and status brushes; embedded plots (see Plot canvas policy).
   * - ``mufinder.services.job_artifacts``
     - Job output paths and parsed secondary outputs.

Widget contract
~~~~~~~~~~~~~~~

A workflow widget that stages jobs follows one shape, so ``MainWindow`` and the
controllers treat every workflow alike:

- **Signals.** ``form_state_changed``, ``configure_requested``,
  ``submit_local_requested``, ``submit_cluster_requested``, ``cancel_requested``,
  ``inspect_failed_requested``, ``reset_failed_requested`` and
  ``remove_requested``. The submit signals carry the selected keys, or every
  key when nothing is selected, and the run buttons' labels say which.
- **Methods.** ``collect_state`` / ``set_form_state`` round-trip the persisted
  form state, and ``set_form_state`` suppresses ``form_state_changed`` while it
  restores; ``set_runtime_states``, ``set_parameter_species_options``,
  ``set_supercell_options``, and ``set_project_defaults`` or
  ``set_xc_functional_default`` push project state into the widget.
- **Persisted form state.** ``ui_state["<workflow>_form"]`` holds a job-name
  prefix and a row list. Each row stores its label, its supercell matrix as a
  3×3 list, its parameter values as the diff from the implicit defaults, and
  its sorted ``added_advanced`` names.
- **Rows.** The row id lives in the label item's ``UserRole``; per-row controls
  are ``SelectionSnapshot`` buttons that resolve their targets through the shared
  targeting rule; a delete-shortcut handler returns early while the table is
  editing a cell; a row whose job is in a runtime state is locked by
  ``workflow_row_is_locked`` and its parameter dialog opens read-only. A
  ``cancelled`` row is editable but is not submitted from the table: the row
  menu's Run entries follow ``READY_WORKFLOW_ROW_STATUSES``, the submit filter.
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
     - Resolve keys to existing ``Job`` records for cancel, inspect and reset.
     - all
   * - ``materialize(keys)``
     - Register or update the definition, expand the jobs, and write input
       directories for ``defined`` jobs only.
     - all
   * - ``pre_submit_check(keys)``
     - Refuse the submission before any plan is built (host-mismatch guards).
     - muon relaxation, quantum corrections, energy barriers
   * - ``submission_groups(jobs)``
     - Split the planned jobs into separately dispatched groups.
     - muon relaxation, quantum corrections
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
:func:`~mufinder.app.workflow_controllers.base.merge_controller_result`, which
asks the registered controllers ``handles_job`` and additively merges the
owning controller's
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
``job_finished`` signal. Collection follows the per-calc-type retrieval flags
the runners track, and every failure travels as data (missing-file flags and
error strings), never as an exception across the signal boundary. Callers with
no carrier (reconciliation, retry, direct test invocation) fall back to one
synchronous ``collect_job_artifacts`` call. ``build_result`` consumes the
carrier and never re-reads disk.

**No failure channel.** ``ControllerResult`` cannot fail a job. Guards that
must abort a job before any result is built therefore stay in the job-finished
handler, read the carrier, and run before dispatch:

- a relaxation job whose geometry optimisation did not converge;
- a relaxation job whose ``-out.cell`` is missing, unparseable, or older than
  the current run (a continuation reuses its directory), or whose muon position
  cannot be located;
- a UEP job whose ``{seed}.den_fmt`` is missing (a plain existence check);
- a quantum-phonon job whose ``.phonon`` is missing or unparseable, including a
  ``PhononParseError`` from the ZPE, gamma-point or witness computation;
- an NEB or transition-state job whose ``.ts`` is missing or unparseable.

A controller whose job passes these guards contributes only the success-path
payload. Giving ``ControllerResult`` a failure channel, so the guards can move
onto the owning controller, is the intended next step; until then a new guard
goes beside the existing ones and reads the carrier rather than disk.

Controller summary
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Controller
     - Specifics
   * - ``ConvergenceCutoffController`` /
       ``ConvergenceKPointController``
     - Sweep workflows keyed by cutoff energy and by k-point grid, with jobs
       mapped in ``job_ids_by_cutoff`` / ``job_ids_by_grid``. No
       ``build_result``: the convergence metrics come from the primary
       ``.castep`` parse.
   * - ``PristineRelaxationController``
     - Keyed by row label. ``build_result`` reads the carrier's relaxed cell,
       contributes ``lattice_a`` / ``lattice_b`` / ``lattice_c``, and creates
       or refreshes the derived relaxed structure.
   * - ``MuonRelaxationController``
     - Keyed by ``"{charge prefix}:{label}"``. ``pre_submit_check`` requires
       every staged site to match the active host; ``submission_groups``
       dispatches positive-muon jobs and then muonium jobs; ``build_result``
       adds the muon's final fractional position (supercell and folded host
       coordinates) and rewrites the source site's status, run id and total
       energy.
   * - ``UepController``
     - Keyed by row label. ``build_result`` contributes the
       ``density_available`` flag only.
   * - ``QuantumCorrectionsController``
     - Two stages per source: singlepoint, then phonon. ``materialize``
       dispatches each source to its current stage and writes
       ``singlepoint_job_ids_by_label`` / ``phonon_job_ids_by_label`` plus
       ``job_seed_names_by_label``; ``pre_submit_check`` runs the host guard
       shared with energy barriers; ``build_result`` derives the muon ZPE,
       gamma-point eigenvalues, mode participations and the ``w1`` witness from
       the carrier's parsed phonon calculation, and a singlepoint job
       contributes an empty result. Both calc types are ``handles_job``-owned.
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
       ``view_getter`` because its view is created on first visit.

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

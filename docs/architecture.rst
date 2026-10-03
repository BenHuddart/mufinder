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
   CASTEP syntax details and should not depend on UI, services, workflow
   orchestration, or scientific algorithms.

``mufinder.persistence``
   JSON storage and codec helpers. Persistence may serialize domain objects but
   must not reach into UI, services, workflow expansion, or science modules.

``mufinder.science``
   Numerical and crystallographic algorithms, including site generation,
   structural comparison, UEP analysis, magnetic structures, magnetic fields,
   and demagnetizing tensors. Science code may use domain records and CASTEP
   parsing helpers where those are part of scientific inputs, but it must remain
   UI-free and service-free.

``mufinder.workflow``
   Pure workflow expansion and rendering contracts. This layer turns resolved
   workflow state into ``JobSpec`` objects and rendered input files without
   owning project mutation, runner state, or widgets. Each workflow is exposed
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

These boundaries are enforced by ``tests/test_agent_harness.py``. If a future
feature genuinely requires a new dependency direction, update this document and
the structural test in the same change.

Refactoring policy
------------------

When duplicated UI code starts to encode workflow, execution, or styling
policy, move the reusable part into the appropriate shared layer:

- workflow-row planning belongs in ``mufinder.app.workflow_submission`` or a
  lower non-UI service if it no longer depends on widgets
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

Workflow controllers
--------------------

``mufinder.app.workflow_controllers`` is a thin controller layer *within* the
app layer, sitting between the ``MainWindow`` widgets and the
services/workflow/science layers. Each
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController` owns one
workflow's ``submit → materialize → result → refresh`` cycle, replacing the
per-workflow tangle of ``_on_submit_*`` / ``_materialize_*`` handlers that
previously accreted on ``MainWindow``.

- Controllers are plain Python objects (not ``QObject``): they own no signals.
- A controller's constructor inputs follow a three-part pattern: the shared
  :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`
  (exactly the narrow, late-bound callables every workflow controller needs —
  workflow-specific dependencies never go in it), an optional per-controller
  deps dataclass of required, precisely-typed callables defined in the
  controller's own module (e.g. ``ConvergenceCutoffDeps``), and the
  controller's own widget/view reference. A controller holds **no** reference
  to ``MainWindow`` and never imports the ``main_window`` module, which keeps
  controllers unit-testable without constructing a window. The one sanctioned
  deviation from the third part is ``EnergyBarriersController``, whose view is
  created *lazily* (only when the workflow route is first opened): it takes a
  ``view_getter: Callable[[], EnergyBarriersWidget | None]`` instead of a view
  instance, and its ``refresh_view`` no-ops when the getter returns ``None``.
  The controller is still constructed and registered in ``MainWindow.__init__``
  regardless, so a completed NEB job dispatches through its ``build_result`` even
  when the view was never opened.
- :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.submit`
  is a template method mirroring the former
  ``MainWindow._submit_workflow_selection`` funnel; workflow-specific behavior
  is supplied by overriding ``normalize`` / ``build_states`` / ``jobs_for_keys``
  / ``materialize`` / ``is_ready`` and, where needed, ``pre_submit_check`` and
  ``submission_groups``. ``MuonRelaxationController`` is the first migration
  exercising the latter two hooks: its ``pre_submit_check`` validates that
  every staged site matches the active host (silently refusing the submission
  before any plan is built when the mismatch is not resolved), and its
  ``submission_groups`` splits a mixed job list into per-charge-state groups
  (positive muon first, then muonium), each dispatched by the base template
  loop as a separate backend call with no extra kwargs.
- **Pre-dispatch reset of dismissed jobs.** Between building the plan and
  dispatching the submission groups, ``submit`` partitions the planned jobs with
  :func:`~mufinder.app.workflow_submission.job_is_resubmittable_after_dismissal`
  and, when any row's bound job was dismissed from the jobs dashboard, calls the
  shared context's ``prepare_jobs_for_resubmission`` with exactly those jobs. The
  hook resets them back to ``defined`` (after the stale-output cleanup prompt) so
  they clear the ``{DEFINED}`` gate in the submit backends; returning ``False``
  (the user cancelled the prompt) aborts the submission silently. ``materialize``
  is then re-run for the plan's ready keys, because controllers only regenerate a
  job's input directory when the job is ``defined`` and the first pass ran before
  the reset. This is the only step in the cycle that mutates job state before
  dispatch, and it acts on the plan's own job objects, so a row never gains a
  duplicate job. Non-dismissed failed jobs are never passed to the hook and keep
  their existing behavior. See ``docs/job_lifecycle.rst`` (Dismissed rows stay
  resubmittable).
- Result dispatch is registry-based:
  :func:`~mufinder.app.workflow_controllers.base.merge_controller_result`
  consults the registered controllers in ``MainWindow._on_local_job_finished``
  and additively merges the owning controller's
  :class:`~mufinder.app.workflow_controllers.base.ControllerResult`. Unmigrated
  workflows fall through to the existing inline handling unchanged.
- **Parse-once off the GUI thread.** A completed job's *secondary* CASTEP
  outputs (the relaxed ``-out.cell``, the ``.phonon`` file plus its derived
  quantum quantities, and the ``.ts`` transition-state file) are parsed **once**
  on the runner worker thread, not on the GUI thread. Each runner
  (``LocalJobRunner`` on its ``QProcess.finished`` callback, ``ClusterJobRunner``
  on its worker ``QThread`` in ``_retrieve_results``) calls
  :func:`~mufinder.services.job_artifacts.collect_job_artifacts` after the
  primary ``.castep`` parse succeeds and delivers the resulting
  :class:`~mufinder.services.job_artifacts.ParsedJobArtifacts` as the third
  payload of the ``job_finished`` signal, alongside the ``CastepResult``.
  Collection is driven by the same per-calc-type retrieval flags the runners
  already track (``retrieve_out_cell`` / ``retrieve_phonon`` / ``retrieve_ts``);
  every failure is captured on the carrier **as data** (missing-file flags and
  error strings) rather than raised across the Qt signal boundary.
  ``_on_local_job_finished`` decides its failure guards purely from the carrier
  and threads it into ``merge_controller_result`` so the owning controller's
  ``build_result`` consumes the pre-parsed data. Non-runner callers
  (reconciliation/retry paths, direct test invocations) pass no carrier; the
  handler falls back to a single synchronous ``collect_job_artifacts`` call, so
  parse-once still holds on those paths.
- ``PristineRelaxationController`` is the first genuine
  :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.build_result`
  implementation: it consumes the carrier's parsed ``relaxed_cell`` /
  ``lattice_abc``, contributes the relaxed ``lattice_a/b/c`` payload, and creates
  the derived relaxed structure on the document. ``MuonRelaxationController``
  ``.build_result`` carries the richest result logic so far: in addition to the
  lattice payload and the derived structure, it contributes the muon's final
  fractional position in supercell and folded host coordinates, and mutates the
  source muon site (``status``/``relaxation_run_id``/``relative_energy_eV``) — a
  sanctioned document side effect per the ``ControllerResult`` contract. Because
  ``ControllerResult`` has no failure channel, failure-detection prologues that
  must abort a job before any result is built (the ``geom_converged`` guard, the
  shared ``-out.cell`` missing/unparseable guard, and the muon ``H:mu`` position
  guard) stay inline in ``_on_local_job_finished``; they read the carrier
  (``cell_missing`` / ``cell_error`` and the already-parsed ``relaxed_cell``)
  rather than re-reading disk, and the controllers consume the same carried cell
  on the success path. ``UepController`` is a simpler migration whose
  ``build_result`` contributes only the ``density_available`` boolean flag (no
  derived structures, no document mutation): the *guard* that fails a UEP job
  whose ``{seed}.den_fmt`` output is missing is a missing-output guard and stays
  inline for the same no-failure-channel reason (den_fmt is a plain
  existence check outside the artifact carrier), while only the flag assignment
  moves into ``build_result``. ``QuantumCorrectionsController`` is the two-stage
  (singlepoint → phonon) migration and the richest of the six: its
  ``materialize`` dispatches each selected relaxed source to its current stage,
  building singlepoint *and* phonon configs in one pass and writing the two
  per-stage job-id mappings (``singlepoint_job_ids_by_label`` /
  ``phonon_job_ids_by_label``) plus the shared ``job_seed_names_by_label``; its
  ``pre_submit_check`` runs the host guard shared with the energy-barriers
  workflow; and its ``build_result`` derives the muon ZPE, gamma-point
  eigenvalues, mode participations, and the ``w1`` entanglement witness from the
  carrier's parsed ``phonon`` calculation for a quantum-*phonon* job while a
  quantum-*singlepoint* job contributes an empty result (both calc types are
  ``handles_job``-owned so the two-stage pair is coherent). Following the
  no-failure-channel convention, the ``.phonon`` guards (missing/unparseable
  file, and any downstream ``PhononParseError`` from the ZPE / gamma / witness
  computation, which the collector exercises on the worker thread) stay inline in
  ``_on_local_job_finished`` and read the carrier (``phonon_missing`` /
  ``phonon_error``). Task farming for the phonon stage is applied by the shared
  ``_submit_jobs_locally`` backend (not the controller), so
  ``QuantumCorrectionsController`` keeps the base single-group ``submission_groups``.
  That backend resolves each job's farm count through the calc-type-dispatching
  ``MainWindow._task_farming_for_job``, which serves both the quantum-phonon and
  NEB workflows; the runners use it to decide whether a finished job's outputs
  carry CASTEP's ``_farmNNN`` seedname suffix.
  ``EnergyBarriersController`` is the seventh and **last** of the per-workflow
  migrations. Its ``materialize`` keys jobs by their diffusion-path *source key*
  (writing ``job_ids_by_label`` keyed by source key and stamping each expanded
  spec with ``provenance={"source_key": key}``), aligns the reactant/product
  endpoints into the pristine host via ``prepare_transition_state_structures`` and
  seeds the band's middle image with ``idpp_intermediate_structure`` before
  expanding through ``expand_nudged_elastic_band_jobs``; its ``pre_submit_check``
  runs the former ``_check_energy_barrier_neb_host`` guard (validating the selected
  paths' endpoint structures/sites against the active host); and its
  ``build_result`` reads the carrier's parsed ``transition_state`` path and
  contributes the ``transition_state_file`` / ``barrier_height_eV`` /
  ``num_images`` / ``image_labels`` / ``reaction_coordinates`` /
  ``image_energies_eV`` payload. Both the NEB and transition-state-search calc
  types are ``handles_job``-owned so the ``.ts``-producing pair is coherent.
  Following the no-failure-channel convention, the ``.ts`` guard
  (missing/unparseable file) stays inline in ``_on_local_job_finished`` and reads
  the carrier (``ts_missing`` / ``ts_error``).

- **In-place row parameter mutation.** Editing one already-submitted relaxation
  row's CASTEP parameters goes through
  ``PristineRelaxationController.update_row_parameters`` /
  ``MuonRelaxationController.update_row_parameters`` (keyed by row label and by
  the charge-qualified ``"{prefix}:{label}"`` key respectively), never through
  ``register_*_relaxation_definition``. Registering a definition produces a *new*
  ``CalculationDefinition`` with an empty ``job_ids_by_label``, which severs the
  row→job binding — ``materialize`` would then mint a duplicate ``Job`` onto the
  row's existing directory. The helpers rewrite that single row inside the
  **active** definition's ``parameters["configuration"]`` (through the
  serialize/deserialize pair, so the stored shape stays canonical), leaving
  ``job_ids_by_label``, ``job_seed_names_by_label`` and every other row untouched.
  They then mirror the same edit into the widget form state with
  :func:`~mufinder.app.workflow_controllers.base.apply_row_parameters_to_form_state`
  and store back the values the *widget* reports, because the tables strip
  implicit project defaults and ``build_states`` matches each row's form
  signature against its definition signature — pristine relaxation has no
  active-job escape hatch, so a desynced row would fall back to a jobless
  "ready". ``ui_state`` is refreshed from the widget and persisted through the
  shared context.

The set of ``_on_submit_*`` / ``_materialize_*`` handlers still on
``MainWindow`` is pinned by ``tests/test_agent_harness.py`` as a ratchet: it may
only shrink as workflows migrate onto controllers, and no new handler may appear
on ``MainWindow``. With energy barriers migrated the per-workflow migration
series is complete: the only pinned entries that remain are the two cross-workflow
dashboard handlers (``_on_submit_from_dashboard`` / ``_on_submit_local_from_dashboard``),
which fan out to every workflow's controller and legitimately stay on ``MainWindow``.

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
``PLOT_DIALOGS`` table.

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

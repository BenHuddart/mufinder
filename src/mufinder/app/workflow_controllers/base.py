"""Per-workflow job controller base class and shared context.

Each :class:`WorkflowJobController` owns one workflow's
``submit → materialize → result → refresh`` cycle. Controllers are plain
Python objects (not :class:`~PySide6.QtCore.QObject`): they own no signals and,
critically, hold **no reference to** :class:`~mufinder.app.main_window.MainWindow`
and never import that module.

A controller's constructor inputs follow a three-part pattern:

- **shared context** — :class:`WorkflowControllerContext`, the narrow,
  late-bound callables that *every* workflow controller needs plus the
  cross-workflow job/definition readers used by three or more controllers. This
  dataclass never grows workflow-specific fields.
- **per-controller deps** — each concrete controller that needs additional
  handles (e.g. shared MainWindow helpers its moved logic still calls) defines
  its own frozen deps dataclass of required, precisely-typed callables in its
  own module, and takes it as a constructor argument. The base class knows
  nothing about deps.
- **view** — the controller's own widget reference, also a subclass-constructor
  concern.

This split keeps controllers testable in isolation and prevents the shared
context from re-accreting into a god object as workflows migrate.

See ``docs/architecture.rst`` (Workflow controllers) for the layer contract.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from mufinder.castep_io.cell_parser import ParsedCell
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Structure
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.job_execution_controller import JobExecutionController
from mufinder.services.job_file_cleanup import JobFileCleanup
from mufinder.services.project_service import (
    ProjectDocument,
    ProjectService,
    active_definition,
    definitions_for_stage,
    job_for_mapping_key,
)

from mufinder.app.workflow_submission import (
    WorkflowSubmissionCoordinator,
    WorkflowSubmissionPlan,
    job_is_resubmittable_after_dismissal,
    workflow_row_state_is_ready,
)


@dataclass(frozen=True)
class WorkflowControllerContext:
    """Narrow, late-bound handles a controller needs from :class:`MainWindow`.

    Every callable is invoked lazily so that a controller sees the *current*
    document and any test-time monkeypatching of the window's methods. The
    controller never stores the window itself — only these callables — which is
    what keeps ``app.workflow_controllers`` decoupled from ``app.main_window``.
    """

    document: Callable[[], ProjectDocument | None]
    project_service: ProjectService
    job_execution: JobExecutionController
    submission_coordinator: WorkflowSubmissionCoordinator
    submit_jobs_locally: Callable[..., None]
    submit_jobs_on_cluster: Callable[..., None]
    #: Reset dismissed terminal jobs back to ``defined`` so they clear the
    #: ``{DEFINED}`` gate in the submit backends. Called by
    #: :meth:`WorkflowJobController.submit` immediately before dispatch, with
    #: the plan's dismissed jobs only. Returns ``False`` when the user cancels
    #: the stale-output cleanup prompt, which aborts the whole submission
    #: silently.
    prepare_jobs_for_resubmission: Callable[[Sequence[Job]], bool]
    show_info: Callable[[str, str], None]
    request_dashboard_flush: Callable[..., None]
    persist_ui_state: Callable[..., None]
    status_message: Callable[[str], None]

    # Cross-workflow job/definition readers. These belong here (not in any Deps
    # class) because they read durable job/definition state the same way for
    # every workflow, and three or more controllers each need them. See the RULE
    # below for the promotion boundary.
    definition_output_directory: Callable[[CalculationDefinition], Path | None]
    job_output_directory: Callable[[Job], Path | None]
    job_seed_name: Callable[[Job], str]
    #: Plans and runs the clean-up of jobs' files and records (ownership
    #: checks, delete or archive, record deletion). Prompts stay with the caller.
    job_file_cleanup: JobFileCleanup
    run_for_job: Callable[[UUID], CalculationRun | None]
    total_energy_for_job: Callable[[UUID], float | None]
    max_force_for_job: Callable[[UUID], float | None]

    # RULE: workflow-specific dependencies NEVER enter this shared context, but
    # genuinely cross-workflow job/definition readers do. A handle belongs here
    # only when it reads durable job/definition state uniformly and three or more
    # workflow controllers need it (for example the readers above). Anything a
    # single workflow — or only one or two — needs (a workflow-family-specific
    # helper, or shared MainWindow helpers its moved logic still calls) belongs
    # in a per-controller deps dataclass of REQUIRED, precisely-typed fields,
    # defined in that controller's own module and passed to its constructor — see
    # ``ConvergenceCutoffDeps`` in ``convergence_cutoff.py`` for the pattern.


@dataclass
class ControllerResult:
    """Additions a controller contributes to a completed job's :class:`Result`.

    The base result-dispatch merge is strictly additive: ``payload`` updates the
    standard payload, and the sequence fields are appended to the standard ones.
    An empty ``ControllerResult`` (the default) is therefore a no-op, so a
    workflow that does not customize its result keeps byte-identical behavior.
    """

    payload: dict[str, object] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)
    derived_structure_ids: list[UUID] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _success_message(target: str, skipped: int) -> Callable[..., str]:
    """Build the backend status message that also reports ``skipped`` rows."""

    rows = "row" if skipped == 1 else "rows"
    note = f"; skipped {skipped} selected {rows} that {'was' if skipped == 1 else 'were'} not ready"
    if target == "local":
        return lambda count: f"Submitted {count} local job(s){note}"
    return lambda count, cluster: f"Submitted {count} cluster job(s) on {cluster}{note}"


class WorkflowJobController:
    """Owns one workflow's submit → materialize → result → refresh cycle."""

    #: calc_type strings this controller's jobs use, for result dispatch.
    calc_types: frozenset[str] = frozenset()
    #: Message shown when a submit request produced no ready jobs.
    no_ready_message: str = "No ready jobs to submit."
    #: Workflow stage of this controller's calculation definitions. ``None``
    #: (the default) means the controller reads no definitions.
    stage: WorkflowStage | None = None
    #: calc_type of this controller's *definitions* (not its jobs, which
    #: :attr:`calc_types` lists). ``None`` accepts any calc type at :attr:`stage`.
    definition_calc_type: str | None = None
    #: ``ui_state`` key holding the id of the definition the user last
    #: configured. ``None`` means the newest matching definition is active.
    active_key: str | None = None

    def __init__(self, context: WorkflowControllerContext) -> None:
        self._context = context

    # --- definition and job lookups ---

    def definitions(self) -> list[CalculationDefinition]:
        """Return this workflow's definitions in the current document, newest first."""

        document = self._context.document()
        if document is None or self.stage is None:
            return []
        return definitions_for_stage(document, self.stage, calc_type=self.definition_calc_type)

    def active_definition(self) -> CalculationDefinition | None:
        """Return the definition this workflow currently works on, if any.

        See :func:`~mufinder.services.project_document.active_definition` for the
        ``active_key`` lookup and the newest-definition fallback.
        """

        document = self._context.document()
        if document is None or self.stage is None:
            return None
        return active_definition(
            document,
            self.stage,
            calc_type=self.definition_calc_type,
            active_key=self.active_key,
        )

    def job_for_key(
        self,
        definition: CalculationDefinition | None,
        key: str,
        *,
        mapping_name: str = "job_ids_by_label",
        calc_type: str | None = None,
    ) -> Job | None:
        """Return the job ``definition``'s ``mapping_name`` mapping binds to ``key``.

        See :func:`~mufinder.services.project_document.job_for_mapping_key`.
        """

        document = self._context.document()
        if document is None:
            return None
        return job_for_mapping_key(
            document, definition, key, mapping_name=mapping_name, calc_type=calc_type
        )

    def directory_companions(self, job: Job) -> list[Job]:
        """Return the jobs that share ``job``'s output directory by design.

        The shared failed-job reset may clean ``job``'s directory although
        these jobs also use it, instead of refusing to. Default: none.
        """

        return []

    # --- definition maintenance ---

    def sync_mappings(self, definition: CalculationDefinition) -> None:
        """Copy job bindings from definitions equivalent to ``definition`` into it.

        Workflows that keep a job mapping override this with a call to
        :func:`~mufinder.services.definition_mappings.sync_equivalent_mappings`.
        Default: nothing to sync.
        """

    def row_key_matcher(self, labels: Collection[str]) -> Callable[[object], bool]:
        """Return a predicate picking the job-mapping keys of the rows ``labels`` name.

        Default: a key matches when its string form is one of ``labels``.
        """

        return lambda key: str(key) in labels

    def jobs_for_removed_rows(self, labels: Collection[str]) -> list[Job]:
        """Return the jobs the rows ``labels`` name are bound to, in every definition.

        Walks each of :meth:`definitions`' ``job_ids_by_label`` mappings for the
        keys :meth:`row_key_matcher` picks, in definition and mapping order,
        once per job. These are the jobs whose files and records a row removal
        cleans up.
        """

        document = self._context.document()
        if document is None:
            return []
        key_matches = self.row_key_matcher(labels)
        jobs: list[Job] = []
        seen: set[UUID] = set()
        for definition in self.definitions():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            for key, raw_job_id in mapping.items():
                if not key_matches(key):
                    continue
                try:
                    job_id = UUID(str(raw_job_id))
                except (TypeError, ValueError):
                    continue
                job = document.jobs.get(job_id)
                if job is None or job.id in seen:
                    continue
                jobs.append(job)
                seen.add(job.id)
        return jobs

    def remove_labels_from_definition(
        self, definition: CalculationDefinition, labels: Collection[str]
    ) -> bool:
        """Drop the rows ``labels`` name from ``definition``; return whether it was deleted.

        Workflows that support row removal override this with a call to
        :func:`~mufinder.services.definition_mappings.remove_labels_from_definition`.
        """

        raise NotImplementedError

    def remove_rows(self, labels: Iterable[str]) -> tuple[UUID, ...]:
        """Template method: drop the rows ``labels`` name from every definition.

        Calls :meth:`remove_labels_from_definition` on each of
        :meth:`definitions` and returns the ids of the definitions it deleted
        because no row was left. Confirmation and job-file cleanup happen
        before this, in the window; refreshing and persisting happen after.
        """

        if self._context.document() is None:
            return ()
        removed = set(labels)
        return tuple(
            definition.id
            for definition in self.definitions()
            if self.remove_labels_from_definition(definition, removed)
        )

    # --- submission contract (mirrors _submit_workflow_selection today) ---

    def normalize(self, value: object) -> object:
        """Return a canonical key for a raw selection value."""

        raise NotImplementedError

    def build_states(self) -> Mapping[object, object]:
        """Return the current per-key runtime states for this workflow."""

        raise NotImplementedError

    def is_ready(self, states: Mapping[object, object], key: object) -> bool:
        """Return whether ``key`` can be materialized and submitted."""

        return workflow_row_state_is_ready(states, key)

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        """Return already-existing jobs for the selected keys."""

        raise NotImplementedError

    def materialize(self, keys: list[object]) -> list[Job]:
        """Create (and return) jobs for the ready keys that lack one."""

        raise NotImplementedError

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Return whether submission may proceed. Default: always."""

        return True

    def submission_groups(
        self, jobs: list[Job]
    ) -> list[tuple[list[Job], dict[str, object]]]:
        """Split jobs into ``(group, extra-submit-kwargs)`` tuples.

        Default: one group, no extras. Workflows that submit heterogeneous
        job sets (e.g. muon relaxation charge-state groups, or quantum jobs
        that pass ``task_farming_for_job``) override this later.
        """

        return [(jobs, {})]

    def submit(
        self, values: list[object], *, target: Literal["local", "cluster"]
    ) -> None:
        """Template method: plan the submission and dispatch each group.

        Mirrors ``MainWindow._submit_workflow_selection`` today:
        ``pre_submit_check`` → coordinator plan → if no jobs,
        ``show_info(no_ready_message)`` → else submit each
        :meth:`submission_groups` group via the local/cluster backend.

        One step is *not* a behavior-preserving move: a row whose bound job was
        dismissed from the jobs dashboard projects a ready status (see
        :func:`~mufinder.app.workflow_submission.effective_row_status`) but its
        job is still terminal, so it would be dropped by the ``{DEFINED}`` gate
        in the submit backends. Those jobs — and *only* those; a non-dismissed
        failed job keeps its current, unsubmittable behavior — are reset back to
        ``defined`` here via ``prepare_jobs_for_resubmission`` before dispatch,
        after which :meth:`materialize` is re-run so the now-``defined`` jobs get
        their input directories regenerated. Because both steps act on the very
        job objects the plan already holds, no duplicate job is created for the
        row.
        """

        keys = list(values)
        if not self.pre_submit_check(keys):
            return

        plan = self._context.submission_coordinator.submission_plan(
            keys,
            normalize=self.normalize,
            build_states=self.build_states,
            jobs_for_values=self.jobs_for_keys,
            materialize_values=self.materialize,
            is_ready=self.is_ready,
        )
        if not plan.has_jobs:
            self._context.show_info("No ready jobs", self.no_ready_message)
            return

        dismissed_jobs = [
            job for job in plan.jobs if job_is_resubmittable_after_dismissal(job)
        ]
        if dismissed_jobs:
            if not self._context.prepare_jobs_for_resubmission(dismissed_jobs):
                return
            # Materialize ran while building the plan, when the reset jobs were
            # still terminal — and a controller only rewrites a job directory
            # for a DEFINED job. Re-running it now that they are DEFINED again
            # regenerates their inputs from the current configuration, matching
            # what an explicit reset-then-submit produces. Every controller
            # reuses the row's existing job, so this mints no duplicates; the
            # returned jobs are already in the plan.
            self.materialize(list(plan.ready_values))

        backend = (
            self._context.submit_jobs_locally
            if target == "local"
            else self._context.submit_jobs_on_cluster
        )
        groups = self.submission_groups(list(plan.jobs))
        skipped = self.skipped_row_count(plan)
        for index, (group, extras) in enumerate(groups):
            # The backends report what they submitted in the status bar. When
            # the selection also held rows that were not ready, the last
            # group's message says how many were left out, so a "Run All" or a
            # mixed selection never looks as if it covered every row.
            if skipped and index == len(groups) - 1 and "success_message" not in extras:
                extras = {**extras, "success_message": _success_message(target, skipped)}
            backend(group, **extras)

    @staticmethod
    def skipped_row_count(plan: WorkflowSubmissionPlan) -> int:
        """Return how many requested rows the submission will leave out.

        Each requested row contributes at most one job to the plan, and the
        backends submit only ``DEFINED`` jobs, so the rows left out are those
        with no job in the plan (not ready, nothing to materialize) or whose
        job is in another state (running, completed, failed and not dismissed).
        """

        submittable = sum(1 for job in plan.jobs if job.status == JobStatus.DEFINED)
        return max(0, len(plan.ordered_values) - submittable)

    # --- result contract ---

    def handles_job(self, job: Job) -> bool:
        """Return whether this controller owns ``job`` for result dispatch."""

        return job.calc_type in self.calc_types

    def validate_artifacts(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> str | None:
        """Return why a completed ``job`` must fail before any result is built.

        Called once per completion, through :func:`validate_controller_artifacts`,
        after the job-finished handler has resolved the run and before
        :meth:`build_result`. A non-``None`` return is the failure message the
        handler passes to the job-failure path; ``None`` lets the result be
        built. Overrides read the pre-parsed ``artifacts`` carrier (missing-file
        flags and error strings) and never re-read a file the carrier holds.

        Default: ``None`` (every completion is accepted).
        """

        return None

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Return payload/file/structure additions for a completed job.

        ``artifacts`` carries the job's secondary outputs (relaxed cell, phonon,
        transition state) parsed **once** on the runner worker thread. Overriding
        controllers consume them instead of re-reading disk. This controller's
        :meth:`validate_artifacts` has already accepted the completion, so
        whatever it checks is present when dispatch reaches here.

        Default: empty :class:`ControllerResult` (base payload only). Cutoff
        convergence uses the default; workflows that derive structures or mutate
        the document (relaxation, muon sites) override this.
        """

        return ControllerResult()

    # --- view contract ---

    def refresh_view(self) -> None:
        """Re-render this workflow's runtime view from current state."""

        raise NotImplementedError


def dedupe_jobs(jobs: Sequence[Job]) -> list[Job]:
    """Return ``jobs`` without repeats (by job id), in first-seen order.

    ``materialize`` collects a selection's existing jobs and the jobs it
    creates, and ``jobs_for_keys`` resolves several keys that may share a job;
    both return each job once.
    """

    seen: set[UUID] = set()
    unique: list[Job] = []
    for job in jobs:
        if job.id in seen:
            continue
        seen.add(job.id)
        unique.append(job)
    return unique


def apply_row_parameters_to_form_state(
    form_state: object,
    *,
    row_matches: Callable[[Mapping[str, object]], bool],
    parameter_values: Mapping[str, object],
    added_advanced: Sequence[str] | None,
    parameter_key: str,
) -> dict[str, object] | None:
    """Return a copy of ``form_state`` with one row's parameters replaced.

    ``parameter_key`` is the widget's own spelling for the row's parameter
    mapping (``"params"`` for the pristine-relaxation table, ``"parameter_values"``
    for the muon-relaxation tables), because callers pass the state the widget
    itself produced. ``added_advanced`` of ``None`` leaves the row's advanced-name
    list alone.

    Returns ``None`` when no row matches, so the caller can tell "mirrored" from
    "the form does not hold this row" instead of silently writing nothing. The
    input is never mutated.
    """

    if not isinstance(form_state, dict):
        return None
    rows = form_state.get("rows")
    if not isinstance(rows, list):
        return None

    updated_rows: list[object] = []
    matched = False
    for row in rows:
        if matched or not isinstance(row, dict) or not row_matches(row):
            updated_rows.append(row)
            continue
        matched = True
        updated_row = dict(row)
        updated_row[parameter_key] = dict(parameter_values)
        # Keep the serialized alias consistent when the source state carries
        # both spellings (``_restore_*_form_from_definition`` writes
        # ``parameter_values`` alongside the widget's ``params``): the
        # deserializer prefers ``parameter_values``, so a stale alias would win.
        if parameter_key != "parameter_values" and "parameter_values" in row:
            updated_row["parameter_values"] = dict(parameter_values)
        if added_advanced is not None:
            updated_row["added_advanced"] = [str(name) for name in added_advanced]
        updated_rows.append(updated_row)

    if not matched:
        return None
    updated_state = dict(form_state)
    updated_state["rows"] = updated_rows
    return updated_state


def find_job_derived_structure(
    document: ProjectDocument, job: Job, *, origin: str
) -> Structure | None:
    """Return the derived structure an earlier completion of ``job`` created.

    Two job-scoped keys are tried, in order:

    1. ``properties["source_job_id"]``. Both relaxation controllers have written
       it on every derived structure since the first release, so in practice
       every candidate carries it.
    2. The ``derived_structure_ids`` of the :class:`~mufinder.domain.models.Result`
       reachable from each of this job's own runs. This is the fallback for a
       structure written before that property existed; runs are walked in
       ``run_index`` order so the choice is deterministic.

    A structure's *label* is deliberately **not** a key. A muon-relaxation job
    is labelled with its site label alone, so one site's two charge states are
    two jobs with the same ``job.label`` and two derived structures both called
    ``"{label} relaxed"``; matching on that could hand the positive-muon job the
    muonium structure.

    When several structures match — the duplicate pair a continued job minted by
    completing twice before the update-in-place behaviour existed — the first in
    ``document.structures`` insertion order wins. That is the older one, which is
    what any comparison table or downstream consumer built in the meantime
    already references.
    """

    job_id = str(job.id)
    for structure in document.structures.values():
        if structure.origin != origin:
            continue
        if str(structure.properties.get("source_job_id", "")) == job_id:
            return structure

    runs = sorted(
        (run for run in document.runs.values() if run.job_id == job.id),
        key=lambda run: run.run_index,
    )
    for run in runs:
        result = document.results.get(run.result_id) if run.result_id else None
        if result is None:
            continue
        for structure_id in result.derived_structure_ids:
            structure = document.structures.get(structure_id)
            if structure is not None and structure.origin == origin:
                return structure
    return None


def upsert_job_derived_structure(
    project_service: ProjectService,
    document: ProjectDocument,
    job: Job,
    *,
    origin: str,
    workflow_stage: WorkflowStage,
    relaxed: ParsedCell,
    parent_structure_id: UUID | None,
) -> Structure:
    """Create — or refresh in place — ``job``'s derived relaxed structure.

    A job can complete more than once: "Continue relaxation…" returns a finished
    job to ``defined`` and re-runs it from its CASTEP checkpoint. Minting a
    second ``"{label} relaxed"`` structure on that second completion would
    duplicate the entry, orphan the first :class:`~mufinder.domain.models.Result`,
    and strand every reference to the old structure id — comparison tables hold
    ``dft_structure_ids``, and downstream workflows (quantum corrections, energy
    barriers) consume derived structures by id.

    So when :func:`find_job_derived_structure` finds one, its geometry is
    refreshed from the new ``-out.cell`` and the **same id** is returned: both
    Results then name the same structure, and every existing reference keeps
    resolving to the current geometry. Only ``id`` and ``label`` survive the
    refresh — the label because a user may have renamed the structure, and
    ``job.label`` (its only input) cannot change across a continuation anyway.
    ``parent_structure_id`` is left alone when the caller cannot resolve one, so
    a missing active definition never wipes a good parent.

    On first completion this is exactly the former behaviour: a new
    :class:`~mufinder.domain.models.Structure` registered through
    ``ProjectService.add_structure``.
    """

    properties = {
        "source_job_id": str(job.id),
        "source_workflow_stage": workflow_stage.value,
    }
    lattice_vectors = [list(vector) for vector in relaxed.lattice_vectors]
    species = list(relaxed.species)
    fractional_positions = [list(position) for position in relaxed.fractional_positions]

    existing = find_job_derived_structure(document, job, origin=origin)
    if existing is None:
        structure = Structure(
            label=f"{job.label} relaxed",
            origin=origin,
            lattice_vectors=lattice_vectors,
            species=species,
            fractional_positions=fractional_positions,
            parent_structure_id=parent_structure_id,
            properties=properties,
        )
        project_service.add_structure(document, structure)
        return structure

    existing.lattice_vectors = lattice_vectors
    existing.species = species
    existing.fractional_positions = fractional_positions
    if parent_structure_id is not None:
        existing.parent_structure_id = parent_structure_id
    existing.properties.update(properties)
    return existing


#: Slack allowed between a run's start and an output file's mtime before the
#: file counts as predating the run. It absorbs filesystem mtime granularity
#: (1 s on ext3/HFS+, 2 s on FAT) and the small clock offset a network-mounted
#: job directory can introduce. A genuinely stale output is a whole run old, so
#: a few seconds of slack costs nothing.
STALE_OUTPUT_SLACK_SECONDS = 5.0


def output_predates_run(path: Path | None, started_at: datetime | None) -> bool:
    """Return whether ``path`` was last written before the current run started.

    Guards a continued relaxation against the *previous* run's ``-out.cell``. A
    continuation re-uses its job directory, so a run that dies before writing a
    new relaxed structure leaves the old file exactly where the completion path
    looks for it — ``cell_missing`` is ``False`` and the last run's geometry is
    silently accepted as this run's result.

    This compares *local* mtimes, which is decisive for a locally-run job (the
    file carries its write time) and for a cluster job only because retrieval
    refuses to download a remote ``-out.cell`` that predates its submission:
    ``sftp.get`` restamps the local copy with the download time, so an
    unconditional download would make every retrieved file look fresh. Skipping
    that download leaves the local copy carrying the *previous* retrieval's
    mtime, which is what this comparison then rejects. The remote-side half of
    the guard lives in ``cluster_retrieval.retrieve_job_outputs``, which also refuses
    to parse a ``.castep`` that predates the submission — the case this function
    cannot see, because a run that never started has no ``started_at``.

    ``started_at`` of ``None`` (a job the poller never observed running) returns
    ``False``: the guard fails open rather than failing a job on no evidence.
    """

    if path is None or started_at is None:
        return False
    try:
        modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return False
    return (started_at - modified).total_seconds() > STALE_OUTPUT_SLACK_SECONDS


def relaxation_artifact_failure(
    job: Job,
    parsed: CastepResult,
    run: CalculationRun,
    artifacts: ParsedJobArtifacts,
    *,
    seed_name: str,
) -> str | None:
    """Return why a completed relaxation job must fail, shared by both relaxations.

    The pristine and muon relaxation controllers' ``validate_artifacts`` both
    apply these checks, in order:

    1. the geometry optimisation did not converge (``geom_converged is False``);
    2. the ``-out.cell`` is missing (``cell_missing``) or unparseable
       (``cell_error``);
    3. the ``-out.cell`` predates the current run (:func:`output_predates_run`
       against the run's ``started_at``, falling back to the job's).

    Returns ``None`` when the relaxed cell on the carrier belongs to this run.
    """

    if parsed.geom_converged is False:
        return (
            parsed.error_message
            or "Geometry optimization did not satisfy CASTEP convergence criteria."
        )
    if artifacts.cell_missing:
        return f"Expected relaxed structure output was not found: {seed_name}-out.cell"
    if artifacts.cell_error is not None:
        return f"Failed to parse relaxed CELL output: {artifacts.cell_error}"
    if output_predates_run(artifacts.out_cell_path, run.started_at or job.started_at):
        return (
            f"{seed_name}-out.cell predates this run: no relaxed structure "
            "was written, so the previous run's geometry is still in place."
        )
    return None


def confirm_active_host(
    project_service: ProjectService,
    document: ProjectDocument,
    *,
    structure_ids: Sequence[UUID],
    site_ids: Sequence[UUID],
    handle_structure_mismatch: Callable[[object], bool],
    handle_site_mismatch: Callable[[object], bool],
) -> bool:
    """Return whether jobs built on these structures and sites may be submitted.

    The structures are checked against the active host first, through their
    ``parent_structure_id`` chain, then the sites. The first mismatch goes to
    its handler, which asks the user whether to switch hosts and returns
    whether the submission may proceed. With no mismatch, the answer is yes.
    """

    if structure_ids:
        mismatch = project_service.validate_structures_match_active_host(document, structure_ids)
        if mismatch is not None:
            return handle_structure_mismatch(mismatch)
    if site_ids:
        mismatch = project_service.validate_sites_match_active_host(document, site_ids)
        if mismatch is not None:
            return handle_site_mismatch(mismatch)
    return True


def owning_controller(
    controllers: Sequence[WorkflowJobController], job: Job
) -> WorkflowJobController | None:
    """Return the first registered controller that handles ``job``, if any."""

    return next((c for c in controllers if c.handles_job(job)), None)


def validate_controller_artifacts(
    controllers: Sequence[WorkflowJobController],
    job: Job,
    parsed: CastepResult,
    run: CalculationRun,
    artifacts: ParsedJobArtifacts,
) -> str | None:
    """Return the owning controller's failure message for a completed ``job``.

    Uses the same lookup as :func:`merge_controller_result`. ``None`` when no
    controller owns the job or the owner accepts the completion.
    """

    controller = owning_controller(controllers, job)
    if controller is None:
        return None
    return controller.validate_artifacts(job, parsed, run, artifacts)


def merge_controller_result(
    controllers: Sequence[WorkflowJobController],
    job: Job,
    parsed: CastepResult,
    run: CalculationRun,
    artifacts: ParsedJobArtifacts,
    *,
    payload: dict[str, object],
    output_files: list[str],
    derived_structure_ids: list[UUID],
    warnings: list[str],
) -> tuple[dict[str, object], list[str], list[UUID], list[str]]:
    """Merge the owning controller's :class:`ControllerResult` into the result.

    Finds the first registered controller that
    :meth:`~WorkflowJobController.handles_job`, asks it for a
    :class:`ControllerResult`, and merges it *additively* onto the standard
    result inputs. When no controller matches (every unmigrated workflow) or the
    controller returns an empty result (cutoff convergence), this is a strict
    no-op — the caller's result is unchanged.

    Returns the merged ``(payload, output_files, derived_structure_ids,
    warnings)``. New lists are returned; the caller's inputs are not mutated.
    """

    merged_payload = dict(payload)
    merged_files = list(output_files)
    merged_structures = list(derived_structure_ids)
    merged_warnings = list(warnings)

    controller = owning_controller(controllers, job)
    if controller is None:
        return merged_payload, merged_files, merged_structures, merged_warnings

    contribution = controller.build_result(job, parsed, run, artifacts)
    merged_payload.update(contribution.payload)
    merged_files.extend(contribution.output_files)
    merged_structures.extend(contribution.derived_structure_ids)
    merged_warnings.extend(contribution.warnings)
    return merged_payload, merged_files, merged_structures, merged_warnings


__all__ = [
    "STALE_OUTPUT_SLACK_SECONDS",
    "ControllerResult",
    "WorkflowControllerContext",
    "WorkflowJobController",
    "apply_row_parameters_to_form_state",
    "confirm_active_host",
    "dedupe_jobs",
    "find_job_derived_structure",
    "merge_controller_result",
    "output_predates_run",
    "owning_controller",
    "relaxation_artifact_failure",
    "upsert_job_derived_structure",
    "validate_controller_artifacts",
]

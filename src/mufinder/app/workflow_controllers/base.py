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
from dataclasses import dataclass, field, replace
from functools import partial
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from mufinder.castep_io.cell_parser import ParsedCell
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
    ProvenanceRecord,
    Structure,
)
from mufinder.domain.text import plural
from mufinder.services.definition_mappings import (
    definition_stores_configuration,
    ensure_job_mapping,
)
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.job_execution_controller import JobExecutionController
from mufinder.services.job_file_cleanup import JobFileCleanup
from mufinder.services.job_file_export import JobFilesExport, job_files_export
from mufinder.services.job_paths import free_job_seed, job_output_directory
from mufinder.services.job_results import find_job_derived_structure
from mufinder.services.relaxed_structure_spins import apply_relaxed_cell_spins
from mufinder.services.structure_export import (
    RelaxedStructureExport,
    relaxed_structure_export_for_job,
)
from mufinder.services.project_service import (
    ProjectDocument,
    ProjectService,
    UnsavedChanges,
    active_definition,
    definitions_for_stage,
    job_for_mapping_key,
)
from mufinder.workflow import (
    StructureBridgeError,
    restore_job_inputs,
    undo_job_writes,
    write_job_directories,
)
from mufinder.workflow.types import JobSpec

from mufinder.app.workflow_submission import (
    WorkflowSubmissionCoordinator,
    CONFIGURABLE_WORKFLOW_ROW_STATUSES,
    AcceptedJobs,
    ClearOutputs,
    RUN_AGAIN_STATES,
    RunAgainPlan,
    WorkflowSubmissionPlan,
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
    #: Decide how the plan's ended jobs run again (continue from their previous
    #: run's files, or start again), asking when there is something to lose,
    #: and change nothing yet. Called by :meth:`WorkflowJobController.submit`
    #: before dispatch, with the plan's ended jobs only; the backends apply the
    #: returned :class:`~mufinder.app.workflow_submission.RunAgainPlan` once the
    #: submission dialog is accepted. ``None`` when the user cancels, which
    #: aborts the whole submission silently.
    prepare_jobs_for_resubmission: Callable[[Sequence[Job]], RunAgainPlan | None]
    show_info: Callable[[str, str], None]
    request_dashboard_flush: Callable[..., None]
    persist_ui_state: Callable[..., None]
    status_message: Callable[[str], None]
    #: Names the rows the submission in progress covers, ``()`` once it is over,
    #: so the backends' status message can report what applies to those rows.
    set_submitted_rows: Callable[[Sequence[object]], None]
    #: Returns the jobs given less those in a folder another job uses, telling the user why.
    jobs_free_to_start: Callable[[list[Job]], list[Job]]

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


@dataclass(frozen=True)
class PlannedJob:
    """A new row's job a submission has planned but not created yet.

    ``job`` is a ``defined`` :class:`~mufinder.domain.models.Job` that is in no
    project and has no files; ``folder`` is the directory its inputs will be
    written to (its name is the job's seed), ``spec`` the inputs, ``key`` its
    row and ``definition`` the definition whose mapping will bind it (``None``
    when the submission will register one). Once the submission is accepted the
    controller writes ``spec`` into ``folder`` and adopts ``job`` under its id.
    """

    job: Job
    key: str
    folder: Path
    spec: JobSpec
    definition: CalculationDefinition | None


@dataclass
class RowWrites:
    """What :meth:`WorkflowJobController.collect_row_writes` found to write for the rows going ahead.

    ``writes`` are the inputs to write, ``jobs`` the rows' jobs in row order,
    ``adopted`` the planned jobs to adopt once the inputs are written,
    ``rewritten`` each existing job rewritten as ``(key, spec, job, folder
    name)`` and ``keys`` the keys of every row going ahead.
    """

    writes: list[tuple[JobSpec, Path]] = field(default_factory=list)
    jobs: list[Job] = field(default_factory=list)
    adopted: list[PlannedJob] = field(default_factory=list)
    rewritten: list[tuple[str, JobSpec, Job, str]] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)


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
        return lambda count: f"Submitted {plural(count, 'local job')}{note}"
    return lambda count, cluster: f"Submitted {plural(count, 'cluster job')} on {cluster}{note}"


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
    #: The definition mapping that binds this workflow's rows to their jobs.
    mapping_name: str = "job_ids_by_label"
    #: The calc type of the jobs the shared :meth:`plan_jobs` plans.
    planned_calc_type: str = ""
    #: The folder under the project's ``files`` its definitions write to.
    output_folder: str = ""

    def __init__(self, context: WorkflowControllerContext) -> None:
        self._context = context
        #: The planned jobs of the submission in progress, by job id.
        self._planned: dict[UUID, PlannedJob] = {}
        #: "Configured …" records of configurations no accepted submission has run yet.
        self._held_records: list[ProvenanceRecord] = []
        #: What the last :meth:`write_inputs` replaced, so one job's inputs can be put back.
        self._previous_inputs: dict[Path, bytes | None] = {}
        #: The folders the last :meth:`write_inputs` made.
        self._made_folders: list[Path] = []
        #: Row bookkeeping for jobs ``write_jobs`` rewrote, applied once each goes ahead.
        self._row_updates: dict[UUID, list[Callable[[], object]]] = {}
        #: The folder ``write_jobs`` rewrote each existing job's inputs in.
        self._rewritten_folders: dict[UUID, Path] = {}
        #: Existing jobs the submission in progress will rewrite in another folder.
        self._planned_moves: dict[UUID, Path] = {}

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

    def store_configuration(
        self,
        configuration: Mapping[str, object],
        *,
        input_structure_id: UUID,
        output_directory: Path,
        register: Callable[..., CalculationDefinition],
    ) -> CalculationDefinition:
        """Make the definition holding ``configuration`` active, registering it only if it changed.

        A page stores its configuration when Run is pressed. When the active
        definition already holds ``configuration``, in its persisted form and
        for the same input structure, input sites (:meth:`configured_site_ids`)
        and output directory
        (:func:`~mufinder.services.definition_mappings.definition_stores_configuration`),
        it stays active and nothing is registered, so the Run adds no
        definition and no Journal record. Otherwise ``register``, a project
        service ``register_*_definition`` method, registers a new definition
        (and its "Configured …" record) from the same arguments, and it
        becomes the active one. That record is held and written once a
        submission is accepted (:meth:`write_held_records`). Either way the definition is named under
        ``active_key`` (already so after a Run that registered nothing, unless
        the key was missing) and its mappings are synced (:meth:`sync_mappings`,
        which every read of the active definition runs too); it is returned.
        """

        document = self._context.document()
        if document is None:
            raise RuntimeError("store_configuration needs an open project")
        definition = self.active_definition()
        if definition is None or not definition_stores_configuration(
            document,
            definition,
            configuration=configuration,
            input_structure_ids=[input_structure_id],
            output_directory=output_directory,
            input_site_ids=self.configured_site_ids(configuration),
        ):
            definition = register(
                document,
                input_structure_id=input_structure_id,
                parameters=configuration,
                output_directory=output_directory,
                hold_record=self._held_records.append,
            )
        # Pinned either way; writing the key it already holds changes nothing.
        document.ui_state[self.active_key] = str(definition.id)
        self.sync_mappings(definition)
        return definition

    # --- planned jobs (workflows that defer job creation) ---

    def planned_job(self, job_id: UUID) -> PlannedJob | None:
        """Return the planned job ``job_id`` names in the submission in progress, if any."""

        return self._planned.get(job_id)

    def planned_jobs(self) -> tuple[Job, ...]:
        """Return the jobs the submission in progress has planned and not created."""

        return tuple(planned.job for planned in self._planned.values())

    def planned_for_key(self, key: object) -> PlannedJob | None:
        """Return the planned job of row ``key``, if the submission in progress planned one."""

        return next((p for p in self._planned.values() if p.key == str(key)), None)

    def clear_planned(self) -> None:
        """Forget the planned jobs and moves: the submission is over, accepted or not."""

        self._planned.clear()
        self._planned_moves.clear()

    def plan_move(self, job: Job, folder: Path) -> None:
        """Note that the submission in progress rewrites existing ``job`` in ``folder``, not its own.

        The window's folder read answers with ``folder`` until the submission
        ends, so the dialogs and the cluster plan use the folder the job will
        run in once accepted (a job that follows its source's label).
        """

        self._planned_moves[job.id] = folder

    def planned_folder(self, job_id: UUID) -> Path | None:
        """Return the folder an existing job moves to in the submission in progress, if any."""

        return self._planned_moves.get(job_id)

    def plan_existing(self, job: Job, spec: JobSpec | None, output_root: Path) -> None:
        """Plan what the submission does with row job ``job``, which exists already.

        Called by the shared :meth:`plan_jobs` for each such row. Default:
        nothing (it is rewritten in its own folder, if at all).
        """

    def reserved_names(self, output_root: Path) -> set[str]:
        """Return the folder names under ``output_root`` the planned jobs and moves have taken."""

        return {
            *(p.folder.name for p in self._planned.values() if p.folder.parent == output_root),
            *(
                folder.name
                for folder in self._planned_moves.values()
                if folder.parent == output_root
            ),
        }

    def plan_new_job(
        self,
        spec: JobSpec,
        output_root: Path,
        key: object,
        *,
        calc_type: str,
        definition: CalculationDefinition | None,
        label: str | None = None,
        seed: str | None = None,
    ) -> Job:
        """Plan row ``key``'s new job from ``spec`` and return it, changing nothing.

        The job is named for the folder it will be written to: the spec's own
        name, or a free one when another job, or another planned job, uses that
        name. A folder no job's records use is taken as it stands, as it is for a
        row removed and added again, and the row ``key`` names keeps a folder its
        earlier jobs used. ``label`` overrides the job
        label for a workflow that does not name its jobs after their folder;
        ``seed`` is the folder name a workflow chose itself (see
        :meth:`reserved_names`).
        """

        reserved = self.reserved_names(output_root)
        name = spec.name if seed is None else seed
        if seed is None and (
            name in reserved
            or self._folder_taken(output_root / name, self._row_job_ids({str(key)}))
        ):
            name = free_job_seed(spec.name, output_root, reserved)
        job = self._context.project_service.new_job(
            label=name if label is None else label,
            workflow_stage=spec.workflow_stage,
            calc_type=calc_type,
        )
        self._planned[job.id] = PlannedJob(
            job=job,
            key=str(key),
            folder=output_root / name,
            spec=replace(spec, name=name),
            definition=definition,
        )
        return job

    def adopt_planned(self, planned: PlannedJob) -> Job:
        """Add ``planned``'s job to the project, under its id, and forget it as planned."""

        document = self._context.document()
        if document is None:
            raise RuntimeError("adopt_planned needs an open project")
        self._planned.pop(planned.job.id, None)
        return self._context.project_service.adopt_job(document, planned.job)

    def describe_rows(self, keys: list[object], *, also: Collection[UUID] = ()) -> object:
        """Return what :meth:`plan_jobs` and :meth:`write_jobs` read for rows ``keys``, changing nothing.

        Each workflow gives its own shape: the
        definition, the output folder and each row's spec and job. ``also``
        names jobs whose rows count although they are not ready (rows running
        again). The relaxation workflows build it on :meth:`active_expansion`.
        """

        raise NotImplementedError

    def active_expansion(
        self,
        deserialize: Callable[[object], object | None],
        expand: Callable[[CalculationDefinition, object], object | None],
    ) -> tuple[CalculationDefinition, Path, object, object] | None:
        """Return the active definition, its output folder, its configuration and their expansion.

        The definition's mappings are synced first, as every read does.
        ``None`` when there is no project, no active definition, no readable
        configuration, no input structure or no output folder. ``expand``
        returning ``None`` (its structure is no longer in the project) raises
        ``ValueError``, as does any ``ValueError`` or ``StructureBridgeError``
        it raises.
        """

        document = self._context.document()
        definition = self.active_definition() if document is not None else None
        if definition is None:
            return None
        self.sync_mappings(definition)
        config = deserialize(definition.parameters.get("configuration", {}))
        if config is None or not definition.input_structure_ids:
            return None
        output_root = self._context.definition_output_directory(definition)
        if output_root is None:
            return None
        expansion = expand(definition, config)
        if expansion is None:
            raise ValueError("the structure the rows start from is no longer in the project")
        return definition, output_root, config, expansion

    def planned_output_root(self, definition: CalculationDefinition | None) -> Path | None:
        """Return the folder the rows' jobs are written to, with or without a definition yet.

        Before its first submission a source workflow has no definition: its
        jobs are planned under ``files/`` :attr:`output_folder`, where the one
        the submission registers writes.
        """

        document = self._context.document()
        if document is None:
            return None
        if definition is None:
            return document.project_root / "files" / self.output_folder
        return self._context.definition_output_directory(definition)

    def plan_jobs(self, keys: list[object]) -> list[Job]:
        """Return each ready row's existing job, or a planned one (:meth:`plan_new_job`).

        Changes nothing in the project or on disk. The shared version reads
        :meth:`describe_rows` as ``(definition, output folder, [(key, spec,
        job)])`` and plans a :attr:`planned_calc_type` job named for its folder
        for each row with a spec and no job; a configuration that does not
        expand plans nothing. Workflows whose jobs are named otherwise override it.
        """

        try:
            described = self.describe_rows(keys)
        except (ValueError, StructureBridgeError):
            return []
        if described is None:
            return []
        definition, output_root, rows = described
        jobs: list[Job] = []
        for key, spec, existing in rows:
            if existing is not None:
                jobs.append(existing)
                self.plan_existing(existing, spec, output_root)
            elif spec is not None:
                jobs.append(
                    self.plan_new_job(
                        spec,
                        output_root,
                        key,
                        calc_type=self.planned_calc_type,
                        definition=definition,
                    )
                )
        return dedupe_jobs(jobs)

    def collect_row_writes(
        self,
        rows: Iterable[tuple[str, JobSpec | None, Job | None]],
        output_root: Path,
        *,
        going: Collection[UUID],
        run_again_ids: Collection[UUID] = (),
        rewrite_names: Callable[[Job, JobSpec], Sequence[str]] = lambda job, _spec: (job.label,),
    ) -> RowWrites:
        """Return what to write for the rows ``(key, spec, job)`` whose jobs ``going`` names.

        A planned job is written as planned. An existing job that is
        ``defined``, or that ``run_again_ids`` names, is rewritten from its
        spec in the first of ``rewrite_names(job, spec)`` (default: its label)
        that no other job uses; none free, it stays as it is and the backends
        refuse to start it. Nothing is written yet.
        """

        collected = RowWrites()
        for key, spec, existing in rows:
            if existing is not None:
                if existing.id not in going:
                    continue
                collected.jobs.append(existing)
                collected.keys.append(key)
                rewrite = existing.status == JobStatus.DEFINED or existing.id in run_again_ids
                if not rewrite or spec is None:
                    continue
                for name in rewrite_names(existing, spec):
                    if self.job_may_write_folder(existing, output_root / name):
                        collected.writes.append((replace(spec, name=name), output_root))
                        collected.rewritten.append((key, spec, existing, name))
                        self._rewritten_folders[existing.id] = output_root / name
                        break
                continue
            planned = self.planned_for_key(key)
            if planned is None or planned.job.id not in going:
                continue
            collected.writes.append((planned.spec, planned.folder.parent))
            collected.adopted.append(planned)
            collected.jobs.append(planned.job)
            collected.keys.append(key)
        return collected

    def bind_planned(self, mapping: dict[str, object], adopted: Iterable[PlannedJob]) -> None:
        """Adopt each planned job and bind it to its row in ``mapping``."""

        for planned in adopted:
            mapping[planned.key] = str(self.adopt_planned(planned).id)

    def rows_written(self, definition: CalculationDefinition, written: RowWrites) -> None:
        """Make the workflow's other changes once the shared :meth:`write_jobs` wrote its rows.

        Changes for the jobs going ahead go through :meth:`when_going`. Default:
        none.
        """

    def write_inputs(self, writes: Sequence[tuple[JobSpec, Path]]) -> None:
        """Write ``writes``' inputs all or none, remembering what they replaced.

        :meth:`write_jobs` writes through this, so :meth:`accept_submission`
        can put back the inputs of a job that then does not go ahead.
        """

        write_job_directories(
            writes, previous_inputs=self._previous_inputs, made_folders=self._made_folders
        )

    def when_going(self, job: Job, update: Callable[[], object]) -> None:
        """Apply ``update`` to ``job``'s row once ``job`` goes ahead, not before.

        :meth:`write_jobs` records here what it changes for a job it rewrote
        (its label, its seed or source-run map), since a start-again job whose
        previous output cannot be cleared then does not go ahead and must keep
        its row as it was. :meth:`accept_submission` applies the updates of the
        jobs that go ahead; :meth:`materialize` applies them all.
        """

        self._row_updates.setdefault(job.id, []).append(update)

    def _apply_row_updates(self, job_ids: Collection[UUID]) -> None:
        updates, self._row_updates = self._row_updates, {}
        for job_id in job_ids:
            for update in updates.get(job_id, ()):
                update()

    def _forget_writes(self) -> None:
        self._previous_inputs = {}
        self._made_folders = []
        self._row_updates = {}
        self._rewritten_folders = {}

    def write_jobs(
        self,
        keys: list[object],
        *,
        going: Collection[UUID],
        run_again_ids: Collection[UUID] = (),
    ) -> list[Job]:
        """Write the jobs of rows ``keys`` that ``going`` names, once their submission is accepted.

        Creates each planned job (writing its inputs, then adopting it) and
        rewrites the inputs of each existing ``defined`` job, and of each job
        ``run_again_ids`` names, from its row. All the inputs are written or
        none (:func:`~mufinder.workflow.write_job_directories`), and nothing is
        adopted or bound until they are: an ``OSError``, ``ValueError`` or
        ``StructureBridgeError`` leaves the project and its folder as they were.

        The shared version reads :meth:`describe_rows` as for :meth:`plan_jobs`,
        writes through :meth:`collect_row_writes`, binds each new job in the
        definition's :attr:`mapping_name` mapping, then calls
        :meth:`rows_written`. Workflows with other bookkeeping override it.
        """

        described = self.describe_rows(keys)
        if described is None:
            if any(job_id in going for job_id in self._planned):
                raise ValueError("the rows' configuration can no longer be read")
            return []
        definition, output_root, rows = described
        collected = self.collect_row_writes(
            rows, output_root, going=going, run_again_ids=run_again_ids
        )
        self.write_inputs(collected.writes)
        self.bind_planned(ensure_job_mapping(definition, self.mapping_name), collected.adopted)
        self.rows_written(definition, collected)
        return dedupe_jobs(collected.jobs)

    def ensure_definition(
        self, source_keys: Sequence[str], *, hold_record: bool = False
    ) -> CalculationDefinition | None:
        """Return the active definition with the sources ``source_keys`` added, creating one if needed.

        Only a source workflow (Quantum Corrections, Hyperfine Coupling, Energy
        Barriers) implements it.
        """

        raise NotImplementedError

    def write_source_rows(
        self,
        keys: list[object],
        *,
        going: Collection[UUID],
        run_again_ids: Collection[UUID] = (),
        rewrite_names: Callable[[Job, JobSpec], Sequence[str]] = lambda job, _spec: (job.label,),
    ) -> tuple[CalculationDefinition, RowWrites] | None:
        """Write the rows ``going`` names for a source workflow; return its definition and the writes.

        :meth:`describe_rows` and :meth:`collect_row_writes` as for the shared
        :meth:`write_jobs`; once every input is written the definition gains
        the sources of the rows going ahead (:meth:`ensure_definition`, which
        registers one when there is none, its "Configured …" record held until
        the submission is accepted) and each new job is bound in its
        :attr:`mapping_name` mapping. ``None`` when no row goes ahead.
        """

        described = self.describe_rows(keys, also=run_again_ids)
        if described is None:
            if any(job_id in going for job_id in self._planned):
                raise ValueError("the rows' sources can no longer be read")
            return None
        _active, output_root, rows = described
        collected = self.collect_row_writes(
            rows, output_root, going=going, run_again_ids=run_again_ids, rewrite_names=rewrite_names
        )
        if not collected.jobs:
            return None
        self.write_inputs(collected.writes)
        definition = self.ensure_definition(collected.keys, hold_record=True)
        if definition is None:
            raise ValueError("the rows' sources are no longer eligible")
        self.bind_planned(ensure_job_mapping(definition, self.mapping_name), collected.adopted)
        return definition, collected

    def write_held_records(self) -> None:
        """Write the held "Configured …" record of the active definition, and drop the others.

        Called once a submission is accepted: the active definition is the one
        it runs. A configuration stored by a Run that was cancelled, and
        changed again since, never ran, so its record is dropped unwritten. The
        record written is rebuilt from the definition as it is now
        (:meth:`configuration_record`, when the workflow has one), and an
        active definition with no record at all, held or in the Journal, gets
        one too.
        """

        document = self._context.document()
        definition = self.active_definition()
        held, self._held_records = self._held_records, []
        if document is None or definition is None:
            return
        records = [record for record in held if definition.id in record.output_entity_ids]
        unrecorded = not records and not any(
            definition.id in record.output_entity_ids and record.action.startswith("configure")
            for record in document.provenance
        )
        if records or unrecorded:
            # Rebuilt from the definition as it is now: rows removed since the Run that
            # held the record edit the definition in place. A record held in a session
            # that has ended (the project closed unaccepted) is rebuilt the same way.
            rebuilt = self.configuration_record(definition)
            records = [rebuilt] if rebuilt is not None else records[-1:]
        for record in records:
            self._context.project_service.append_provenance(
                document, replace(record, timestamp=datetime.now(UTC))
            )

    def configuration_record(self, definition: CalculationDefinition) -> ProvenanceRecord | None:
        """Return ``definition``'s "Configured …" record, rebuilt, or ``None`` when there is none.

        A submission writes this rebuilt record
        once a submission is accepted, so it describes the definition as it is
        then, and a definition run after a reopen is still recorded. Default:
        the relaxation definitions' record
        (:meth:`~mufinder.services.project_service.ProjectService.row_configure_record`).
        """

        return self._context.project_service.row_configure_record(definition)

    def accept_submission(
        self,
        keys: Sequence[object],
        run_again: RunAgainPlan | None,
        jobs: Sequence[Job],
        start_again_ids: Collection[UUID],
        clear_outputs: ClearOutputs,
    ) -> AcceptedJobs | None:
        """Create what an accepted submission of ``jobs`` needs, and return what goes ahead.

        The backends call it once the submission dialog is accepted and every
        refusal is past, with the jobs going ahead (``on_accepted``). In order:

        1. :meth:`write_jobs` for those jobs. A failure is reported and leaves
           everything as it was, previous output included; the submission then
           starts nothing (``None``).
        2. ``clear_outputs`` empties the previous output of the start-again
           jobs (``start_again_ids``), keeping their ``.cell`` and ``.param``.
           A job whose output cannot be cleared does not go ahead: its inputs
           are put back as they were and an error line says why.
        3. The held "Configured …" record, the run-again plan
           (``run_again.prepare``, whose inputs :meth:`write_jobs` has already
           rebuilt) and a save when anything is unsaved, once any job goes ahead.

        A submission of several groups calls it once per group, each with its
        own jobs, so a group that is cancelled or refused creates nothing.
        """

        going = {job.id for job in jobs}
        self._forget_writes()
        try:
            self.write_jobs(
                list(keys),
                going=going,
                run_again_ids=(run_again.job_ids & going) if run_again is not None else (),
            )
        except (OSError, ValueError, StructureBridgeError) as exc:
            # A step after the inputs were written failed: they go too.
            undo_job_writes(self._previous_inputs, self._made_folders)
            self._forget_writes()
            self._context.show_info(
                "Jobs not started",
                f"The jobs' input files could not be written, so nothing was submitted: {exc}",
            )
            return None
        restarting = [job for job in jobs if job.id in start_again_ids]
        cleaned, errors = clear_outputs(restarting) if restarting else ([], [])
        not_cleaned = {job.id for job in restarting} - {job.id for job in cleaned}
        for job in jobs:
            if job.id not in not_cleaned:
                continue
            # The folder its inputs were rewritten in, which may not be its own yet.
            folder = self._rewritten_folders.get(job.id) or self._context.job_output_directory(job)
            if folder is None:
                continue
            errors.extend(
                f"- {job.label}: {problem}"
                for problem in restore_job_inputs(self._previous_inputs, folder)
            )
            if folder in self._made_folders:
                try:
                    folder.rmdir()
                except OSError:
                    pass
        kept = [job for job in jobs if job.id not in not_cleaned]
        self._apply_row_updates({job.id for job in kept})
        self._forget_writes()
        if not kept:
            return AcceptedJobs(jobs=[], errors=errors)
        self.write_held_records()
        if run_again is not None:
            run_again.prepare(kept, cleaned=[job.id for job in cleaned])
        self.save_materialized(kept, failure_message="Jobs were generated, but autosave failed.")
        return AcceptedJobs(jobs=kept, errors=errors)

    def materialize(self, keys: list[object]) -> list[Job]:
        """Create (and write) the rows' jobs now, and rewrite their defined jobs' inputs.

        A submission does this only once it is accepted (:meth:`plan_jobs`,
        then :meth:`write_jobs`); outside one, continuing a relaxation does it
        at once, here. A configuration that does not expand creates nothing.
        """

        held = len(self._held_records)
        try:
            jobs = self.plan_jobs(keys)
            written = self.write_jobs(keys, going={job.id for job in jobs}) if jobs else []
            self._apply_row_updates({job.id for job in written})
            # A definition this registered is recorded at once, as the jobs are.
            document = self._context.document()
            registered, self._held_records = self._held_records[held:], self._held_records[:held]
            for record in registered if document is not None else ():
                self._context.project_service.append_provenance(document, record)
            return written
        except (ValueError, StructureBridgeError):
            # A step after the inputs were written failed: they go too.
            undo_job_writes(self._previous_inputs, self._made_folders)
            self._held_records = self._held_records[:held]
            return []
        finally:
            self.clear_planned()
            self._forget_writes()

    def configured_site_ids(self, configuration: Mapping[str, object]) -> list[UUID]:
        """Return the input sites a definition registered from ``configuration`` records.

        Default: none, as for the pristine relaxation and UEP definitions.
        """

        return []

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

    def folder_sharers(self, job: Job) -> set[UUID]:
        """Return the jobs that may use ``job``'s folder without it being another job's.

        ``job`` itself, the jobs earlier definitions bind to its row (a reconfigured row runs
        in the same folder), and its :meth:`directory_companions`.
        """

        keys = {
            str(key)
            for definition in self.definitions()
            if isinstance(mapping := definition.parameters.get(self.mapping_name, {}), dict)
            for key, value in mapping.items()
            if str(value) == str(job.id)
        }
        if (planned := self._planned.get(job.id)) is not None:
            # A planned job is in no mapping yet: its row is known from the plan.
            keys.add(planned.key)
        return {
            job.id,
            *self._row_job_ids(keys),
            *(companion.id for companion in self.directory_companions(job)),
        }

    def _row_job_ids(self, keys: Collection[str]) -> set[UUID]:
        """Return the native jobs any definition binds to the rows ``keys`` name.

        A job still running counts as another job, since a second run in its folder would
        overwrite it, and so does an import.
        """

        return {
            job.id
            for job in self.jobs_for_removed_rows(keys)
            if IMPORTED_JOB_TAG not in (job.tags or ()) and job.status != JobStatus.ACTIVE
        }

    def _folder_taken(self, directory: Path, together: Collection[UUID]) -> bool:
        """Return whether writing into ``directory`` would touch another job's files.

        A folder not on disk holds nothing to overwrite, and most rows have none yet.
        """

        return directory.exists() and self._context.job_file_cleanup.folder_used_by_another_job(
            directory, together=together
        )

    def job_may_write_folder(self, job: Job, directory: Path) -> bool:
        """Return whether ``job``'s inputs may be rewritten in ``directory``."""

        return not self._folder_taken(directory, self.folder_sharers(job))

    def directory_companions(self, job: Job) -> list[Job]:
        """Return the jobs that share ``job``'s output directory by design.

        Starting ``job`` again may clean its directory although these jobs
        also use it, instead of refusing to. Default: none.
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

        Walks each of :meth:`definitions`' :attr:`mapping_name` mappings for the
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
            mapping = definition.parameters.get(self.mapping_name, {})
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

    # --- submission contract ---

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

    def key_for_job(self, job_id: UUID) -> object | None:
        """Return the key of the row bound to ``job_id``, or ``None`` when no row is."""

        states = self.build_states()
        for key, state in states.items():
            if getattr(state, "job_id", None) == job_id:
                return key
        # A row whose form has diverged from its run shows no job, but stays bound.
        for key in states:
            if any(job.id == job_id for job in self.jobs_for_keys([key])):
                return key
        return None

    def relaxed_structure_export(self, key: object) -> RelaxedStructureExport:
        """Return ``key``'s relaxed structure for export, or why it has none.

        Resolves the row's job through :meth:`jobs_for_keys`, so only a
        relaxation workflow's completed jobs ever have one (see
        :func:`~mufinder.services.structure_export.relaxed_structure_export_for_job`).
        Called live each time a row menu opens.
        """

        jobs = self.jobs_for_keys([key])
        return relaxed_structure_export_for_job(self._context.document(), jobs[0] if jobs else None)

    def job_files_exports(self, keys: Sequence[object]) -> list[JobFilesExport]:
        """Return the CASTEP files of the jobs behind ``keys``, one entry per job.

        Rows without a job yet contribute nothing. Called live each time a row
        menu opens, for Open Containing Folder and Export Inputs/Outputs….
        """

        return [
            job_files_export(
                job, self._context.job_output_directory(job), self._context.job_seed_name(job)
            )
            for job in self.jobs_for_keys(list(keys))
        ]

    def save_materialized(self, jobs: Sequence[Job], *, failure_message: str) -> None:
        """Save the project after an accepted submission or a ``materialize`` wrote ``jobs``.

        Only when anything is unsaved, whether the write made it (a new job or
        mapping, a job returned to ``defined``) or it was there before; with no
        ``jobs`` nothing is saved. Rewritten input files are on disk already
        and need no save.
        """

        document = self._context.document()
        if not jobs or document is None:
            return
        if self._context.project_service.unsaved_changes(document) is UnsavedChanges.NONE:
            return
        self._context.persist_ui_state(failure_message=failure_message)

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Return whether submission may proceed. Default: always."""

        return True

    def completed_jobs_to_run_again(self, plan: WorkflowSubmissionPlan) -> list[Job]:
        """Return the plan's completed jobs that start again, replacing their results.

        A completed job never runs again unless its controller names it here:
        Quantum Corrections and Hyperfine Coupling name the jobs of their
        ``edited`` rows. They go to
        :meth:`submit`'s run-again plan as start-again jobs, so nothing changes
        until the submission dialog is accepted. Default: none.
        """

        return []

    def submission_groups(self, jobs: list[Job]) -> list[tuple[list[Job], dict[str, object]]]:
        """Split jobs into ``(group, extra-submit-kwargs)`` tuples.

        Default: one group, no extras. Workflows that submit heterogeneous
        job sets (e.g. muon relaxation charge-state groups, or quantum jobs
        that pass ``task_farming_for_job``) override this later.
        """

        return [(jobs, {})]

    def submit(
        self,
        values: list[object],
        *,
        target: Literal["local", "cluster"],
        default_cluster_name: str | None = None,
    ) -> None:
        """Template method: plan the submission and dispatch each group.

        ``pre_submit_check`` → coordinator plan → if no jobs,
        ``show_info(no_ready_message)`` → else submit each
        :meth:`submission_groups` group via the local/cluster backend.

        A row whose job ended (failed, stopped or cancelled) is submittable, and
        so is a completed job :meth:`completed_jobs_to_run_again` names.
        ``prepare_jobs_for_resubmission`` decides how those jobs run again
        (continued from their previous run's files or started again) without
        changing anything, and the plan goes to the backend as ``run_again``.

        No job is created before the submission is accepted: the plan holds
        each row's existing job, or a planned one (:meth:`plan_jobs`), and
        nothing is written. The backends call :meth:`accept_submission`
        (``on_accepted``) once the dialog is accepted and every refusal is
        past; that writes the jobs going ahead, adopts the planned ones, clears
        the start-again jobs' previous output, writes the held "Configured …"
        records, applies the run-again plan and saves, all on the very job
        objects the plan holds, so no duplicate job is created for a row. A
        cancelled or refused submission changes nothing. The planned jobs are
        forgotten when the submission ends, accepted or not.

        ``default_cluster_name`` preselects a cluster in the cluster submission
        dialog, as the Job Dashboard's ``Resubmit…`` does with a job's own
        cluster; ``None`` leaves the last-used one.
        """

        keys = list(values)
        if not self.pre_submit_check(keys):
            return

        try:
            self._submit(keys, target=target, default_cluster_name=default_cluster_name)
        finally:
            self.clear_planned()

    def _submit(
        self,
        keys: list[object],
        *,
        target: Literal["local", "cluster"],
        default_cluster_name: str | None,
    ) -> None:
        plan = self._context.submission_coordinator.submission_plan(
            keys,
            normalize=self.normalize,
            build_states=self.build_states,
            jobs_for_values=self.jobs_for_keys,
            plan_values=self.plan_jobs,
            is_ready=self.is_ready,
        )
        if not plan.has_jobs:
            self._context.show_info("No ready jobs", self.no_ready_message)
            return

        replacing = {job.id for job in self.completed_jobs_to_run_again(plan)}
        # Before anything runs again: a job left in another job's folder keeps its record as it is.
        startable = [
            job
            for job in plan.jobs
            if job.status in {JobStatus.DEFINED, JobStatus.ENDED} or job.id in replacing
        ]
        free = {job.id for job in self._context.jobs_free_to_start(startable)}
        refused = {job.id for job in startable} - free
        if refused:
            plan = replace(plan, jobs=tuple(job for job in plan.jobs if job.id not in refused))
            if not plan.has_jobs:
                return

        ended_jobs = [
            job
            for job in plan.jobs
            if job.status == JobStatus.ENDED
            or (job.status == JobStatus.COMPLETED and job.id in replacing)
        ]
        run_again: RunAgainPlan | None = None
        if ended_jobs:
            run_again = self._context.prepare_jobs_for_resubmission(ended_jobs)
            if run_again is None:
                return

        backend = (
            self._context.submit_jobs_locally
            if target == "local"
            else self._context.submit_jobs_on_cluster
        )
        groups = self.submission_groups(list(plan.jobs))
        if target == "cluster" and default_cluster_name:
            groups = [
                (group, {"default_cluster_name": default_cluster_name, **extras})
                for group, extras in groups
            ]
        if run_again is not None:
            groups = [
                (group, {**extras, "run_again": run_again})
                if any(job.id in run_again.job_ids for job in group)
                else (group, extras)
                for group, extras in groups
            ]
        on_accepted = partial(self.accept_submission, list(plan.ready_values), run_again)
        groups = [(group, {**extras, "on_accepted": on_accepted}) for group, extras in groups]
        skipped = self.skipped_row_count(
            plan, run_again.job_ids if run_again is not None else frozenset()
        )
        self._context.set_submitted_rows(plan.ready_values)
        try:
            for index, (group, extras) in enumerate(groups):
                # The backends report what they submitted in the status bar. When
                # the selection also held rows that were not ready, the last
                # group's message says how many were left out, so a "Run All" or a
                # mixed selection never looks as if it covered every row.
                if skipped and index == len(groups) - 1 and "success_message" not in extras:
                    extras = {**extras, "success_message": _success_message(target, skipped)}
                backend(group, **extras)
        finally:
            self._context.set_submitted_rows(())

    @staticmethod
    def skipped_row_count(
        plan: WorkflowSubmissionPlan, running_again: Collection[UUID] = frozenset()
    ) -> int:
        """Return how many requested rows the submission will leave out.

        Each requested row contributes at most one job to the plan, and the
        backends submit ``DEFINED`` jobs and the ended jobs ``running_again``
        names, so the rows left out are those with no job in the plan (not
        ready, nothing to materialize) or whose job is in another state (active
        or completed).
        """

        submittable = sum(
            1
            for job in plan.jobs
            if job.status == JobStatus.DEFINED
            or (job.status in RUN_AGAIN_STATES and job.id in running_again)
        )
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


def locked_configuration_reason(
    jobs: Sequence[Job | None], statuses: Sequence[str], *, job_noun: str
) -> str:
    """Return why a source Configure dialog opens read-only for rows in ``statuses``.

    ``jobs`` are the rows' jobs and ``statuses`` their projected row statuses,
    in the same order; ``job_noun`` names the job ("phonon job"). The reason is
    one line naming what keeps the first locked row locked; it is empty when
    every row is configurable (:data:`CONFIGURABLE_WORKFLOW_ROW_STATUSES`: no
    job yet, or one that ended, since running it again rebuilds its inputs).
    A caller passes ``None`` for a row it can configure although its status is
    not among those: a completed row whose job's inputs are there to compare it
    with (see :class:`~mufinder.app.workflow_controllers.completed_row_edits.CompletedRowEditsController`).
    So a completed row that reaches this reason is one whose inputs are missing.
    """

    for job, status in zip(jobs, statuses, strict=True):
        if job is None or status in CONFIGURABLE_WORKFLOW_ROW_STATUSES:
            continue
        if job.status == JobStatus.ACTIVE:
            return (
                f"Parameters are read-only while a selected row's {job_noun} is {status}; "
                "cancel it to edit them."
            )
        return (
            f"The input files of a selected row's completed {job_noun} are missing, so "
            "its parameters cannot be compared with its result and stay as they are."
        )
    return ""


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

    So when :func:`~mufinder.services.job_results.find_job_derived_structure` finds one, its geometry is
    refreshed from the new ``-out.cell`` and the **same id** is returned: both
    Results then name the same structure, and every existing reference keeps
    resolving to the current geometry. Only ``id`` and ``label`` survive the
    refresh — the label because a user may have renamed the structure, and
    ``job.label`` (its only input) cannot change across a continuation anyway.
    ``parent_structure_id`` is left alone when the caller cannot resolve one, so
    a missing active definition never wipes a good parent.

    Both paths store the ``-out.cell``'s ``SPIN=`` values as
    ``properties["atomic_spins"]`` and drop spins the new cell does not carry
    (:func:`~mufinder.services.relaxed_structure_spins.apply_relaxed_cell_spins`),
    so jobs started from the relaxed structure inherit the run's spins.

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
        apply_relaxed_cell_spins(structure, relaxed)
        project_service.add_structure(document, structure)
        return structure

    existing.lattice_vectors = lattice_vectors
    existing.species = species
    existing.fractional_positions = fractional_positions
    if parent_structure_id is not None:
        existing.parent_structure_id = parent_structure_id
    existing.properties.update(properties)
    apply_relaxed_cell_spins(existing, relaxed)
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


def planned_job_entry(controllers: Iterable[WorkflowJobController], job: Job) -> PlannedJob | None:
    """Return the planned job ``job`` is in a submission in progress, if any controller planned it.

    The window's folder, seed, farm-count and shared-folder reads ask this first,
    since a planned job is in no definition's mapping yet.
    """

    return next(
        (planned for c in controllers if (planned := c.planned_job(job.id)) is not None), None
    )


def job_folder(
    controllers: Iterable[WorkflowJobController],
    document: ProjectDocument | None,
    job: Job,
    *,
    quantum_seed_name: Callable[[CalculationDefinition, str], str],
) -> Path | None:
    """Return the folder of ``job``: a planned job's from its plan, any other's from its mapping.

    A job a submission has planned is in no definition's mapping yet, so
    :func:`~mufinder.services.job_paths.job_output_directory` cannot find it;
    an existing job the submission moves (:meth:`WorkflowJobController.plan_move`)
    answers with the folder it moves to.
    """

    planned = planned_job_entry(controllers, job)
    if planned is not None:
        return planned.folder
    moved = next(
        (folder for c in controllers if (folder := c.planned_folder(job.id)) is not None), None
    )
    if moved is not None:
        return moved
    return job_output_directory(document, job, quantum_seed_name=quantum_seed_name)


def all_planned_jobs(controllers: Iterable[WorkflowJobController]) -> list[Job]:
    """Return every job a submission in progress has planned and not created."""

    return [job for controller in controllers for job in controller.planned_jobs()]


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
    "merge_controller_result",
    "output_predates_run",
    "owning_controller",
    "relaxation_artifact_failure",
    "upsert_job_derived_structure",
    "validate_controller_artifacts",
]

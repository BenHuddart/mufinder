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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.job_execution_controller import JobExecutionController
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_submission import (
    WorkflowSubmissionCoordinator,
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


class WorkflowJobController:
    """Owns one workflow's submit → materialize → result → refresh cycle."""

    #: calc_type strings this controller's jobs use, for result dispatch.
    calc_types: frozenset[str] = frozenset()
    #: Message shown when a submit request produced no ready jobs.
    no_ready_message: str = "No ready jobs to submit."

    def __init__(self, context: WorkflowControllerContext) -> None:
        self._context = context

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
        for group, extras in self.submission_groups(list(plan.jobs)):
            backend(group, **extras)

    # --- result contract ---

    def handles_job(self, job: Job) -> bool:
        """Return whether this controller owns ``job`` for result dispatch."""

        return job.calc_type in self.calc_types

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
        controllers consume them instead of re-reading disk. The inline guards in
        ``MainWindow._on_local_job_finished`` have already validated whatever
        this controller relies on before dispatch reaches here.

        Default: empty :class:`ControllerResult` (base payload only). Cutoff
        convergence uses the default; workflows that derive structures or mutate
        the document (relaxation, muon sites) override this.
        """

        return ControllerResult()

    # --- view contract ---

    def refresh_view(self) -> None:
        """Re-render this workflow's runtime view from current state."""

        raise NotImplementedError


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

    controller = next((c for c in controllers if c.handles_job(job)), None)
    if controller is None:
        return merged_payload, merged_files, merged_structures, merged_warnings

    contribution = controller.build_result(job, parsed, run, artifacts)
    merged_payload.update(contribution.payload)
    merged_files.extend(contribution.output_files)
    merged_structures.extend(contribution.derived_structure_ids)
    merged_warnings.extend(contribution.warnings)
    return merged_payload, merged_files, merged_structures, merged_warnings


__all__ = [
    "ControllerResult",
    "WorkflowControllerContext",
    "WorkflowJobController",
    "apply_row_parameters_to_form_state",
    "merge_controller_result",
]

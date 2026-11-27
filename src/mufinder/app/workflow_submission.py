"""Shared workflow submission planning helpers."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from uuid import UUID
from mufinder.services.job_monitoring import job_status_word

_WorkflowKey = TypeVar("_WorkflowKey")

#: Runtime/display statuses a workflow row may be in and still be submittable.
#: This is the single source of truth shared by the submit predicate
#: (:func:`workflow_row_state_is_ready`) and by widget-level submit-enablement
#: gates, so run-button enablement and the submit filter cannot drift apart.
#: A row whose job ended (``failed``, ``stopped``, ``cancelled``) is
#: submittable: the shared submit cycle returns the job to ``defined`` first
#: and rebuilds its inputs from the row. So is an ``edited`` row
#: (:data:`EDITED_ROW_STATUS`).
UNSUBMITTED_WORKFLOW_ROW_STATUSES = frozenset({"ready", "defined", "not generated"})

#: The status of a row whose job completed and whose parameters have changed
#: since: it keeps showing its result until it runs again, and running it again
#: replaces that result. Only Quantum Corrections and Hyperfine Coupling rows reach
#: it.
EDITED_ROW_STATUS = "edited"

READY_WORKFLOW_ROW_STATUSES = UNSUBMITTED_WORKFLOW_ROW_STATUSES | {
    "failed",
    "stopped",
    "cancelled",
    EDITED_ROW_STATUS,
}
_READY_WORKFLOW_ROW_STATUSES = READY_WORKFLOW_ROW_STATUSES

#: Row statuses whose configuration (parameters, supercell, sweep value, farm
#: count) can change: exactly the submittable ones. Every run of an ended row
#: rebuilds its inputs from the row, so an edit takes effect on the next run
#: from the row or from the Job Dashboard. Active and completed rows are locked,
#: except a completed Quantum Corrections or Hyperfine Coupling row, whose
#: controller allows it.
CONFIGURABLE_WORKFLOW_ROW_STATUSES = READY_WORKFLOW_ROW_STATUSES

#: Every ended job is resubmittable, a stopped one too (walltime, lost track): the
#: submit script is regenerated, so a longer walltime takes effect.
RESUBMITTABLE_JOB_STATES = frozenset({JobStatus.ENDED})

#: The job states a :class:`RunAgainPlan` returns to ``defined``: an ended job,
#: and a completed one whose row was edited.
RUN_AGAIN_STATES = frozenset({JobStatus.ENDED, JobStatus.COMPLETED})


def _apply_nothing(
    _continued: Sequence[Job], _restarted: Sequence[Job], _cleaned: Collection[UUID]
) -> None:
    return None


def _keep_nothing(_continued: Sequence[Job]) -> None:
    return None


@dataclass(frozen=True)
class AcceptedJobs:
    """What an accepted submission's ``on_accepted`` lets go ahead.

    ``jobs`` are the jobs to start; ``errors`` hold one ``"- label: reason"``
    line for each start-again job left out because its previous output could
    not be cleared.
    """

    jobs: list[Job]
    errors: list[str] = field(default_factory=list)


#: Clears the previous output of start-again jobs: returns those it cleared and
#: an error line for each it could not (``JobsController._clear_outputs_for_fresh_start``).
ClearOutputs = Callable[[Sequence[Job]], tuple[list[Job], list[str]]]

#: What a backend calls once a submission is accepted and every refusal is past,
#: with the jobs going ahead, the ids of the start-again jobs among them and the
#: backend's clearing of their previous output; ``None`` when the submission
#: must start nothing. See ``WorkflowJobController.accept_submission``.
OnAccepted = Callable[[Sequence[Job], Collection[UUID], ClearOutputs], "AcceptedJobs | None"]


@dataclass(frozen=True)
class RunAgainPlan:
    """How the ended jobs of one submission run again, decided but not yet applied.

    The answer to the run-again question (see :doc:`/job_lifecycle`): the jobs
    that continue from their previous run's files and those that start again.
    A completed job whose row was edited (:data:`EDITED_ROW_STATUS`) always
    starts again, and its result goes when the plan is applied.
    Nothing changes until the submission is accepted: the accept step
    (``WorkflowJobController.accept_submission``) rebuilds the jobs' inputs from
    their rows, clears the start-again jobs' previous output and calls
    :meth:`prepare` on the jobs going ahead, which returns them to ``defined``
    (``apply``) and then puts back the inputs a continued relaxation keeps from
    its run (``keep_continued_inputs``). A cancelled or refused submission
    leaves every job, file and journal entry as it was.
    """

    continue_job_ids: frozenset[UUID] = frozenset()
    start_again_job_ids: frozenset[UUID] = frozenset()
    #: Return ended jobs to ``defined``: ``(continued, restarted, cleaned)``,
    #: where ``cleaned`` names the restarted jobs whose previous output was
    #: cleared.
    apply: Callable[[Sequence[Job], Sequence[Job], Collection[UUID]], None] = field(
        default=_apply_nothing
    )
    #: Write back, over the rebuilt inputs, the ones the continued jobs keep
    #: from their previous run.
    keep_continued_inputs: Callable[[Sequence[Job]], object] = field(default=_keep_nothing)

    @property
    def job_ids(self) -> frozenset[UUID]:
        """Return every job this plan runs again."""

        return self.continue_job_ids | self.start_again_job_ids

    def prepare(self, jobs: Sequence[Job], *, cleaned: Collection[UUID] = ()) -> None:
        """Return this plan's jobs among ``jobs`` to ``defined``, keeping continued jobs' inputs."""

        going = [job for job in jobs if job.id in self.job_ids and job.status in RUN_AGAIN_STATES]
        if going:
            continued = [job for job in going if job.id in self.continue_job_ids]
            self.apply(
                continued,
                [job for job in going if job.id not in self.continue_job_ids],
                cleaned,
            )
            if continued:
                self.keep_continued_inputs(continued)


def effective_row_status(job: Job | None, *, ready_status: str = "ready") -> str:
    """Project ``job`` onto the status string its workflow table row shows.

    A row with no job, or whose job is ``defined`` (staged or reset, never
    run), reads ``ready_status``: either way it is ready to run. Every other
    row reads its job's status word
    (:func:`~mufinder.services.job_monitoring.job_status_word`), from the job's
    state alone. Dismissing a job from the Job Dashboard does not change it.
    """

    if job is None or job.status == JobStatus.DEFINED:
        return ready_status
    return job_status_word(job)


@dataclass(frozen=True, slots=True)
class WorkflowSubmissionPlan(Generic[_WorkflowKey]):
    """Prepared jobs for a workflow-row submission action."""

    ordered_values: tuple[_WorkflowKey, ...]
    ready_values: tuple[_WorkflowKey, ...]
    jobs: tuple[Job, ...]

    @property
    def has_jobs(self) -> bool:
        """Return whether the plan contains any jobs to submit."""

        return bool(self.jobs)


def workflow_row_state_is_ready(
    states: Mapping[_WorkflowKey, object],
    key: _WorkflowKey,
) -> bool:
    """Return whether a workflow row can be materialized and submitted."""

    return str(getattr(states.get(key), "status", "")) in _READY_WORKFLOW_ROW_STATUSES


def workflow_row_state_is_configurable(
    states: Mapping[_WorkflowKey, object],
    key: _WorkflowKey,
) -> bool:
    """Return whether a workflow row's configuration can change (no job running or completed)."""

    return str(getattr(states.get(key), "status", "")) in CONFIGURABLE_WORKFLOW_ROW_STATUSES


class WorkflowSubmissionCoordinator:
    """Build submission job lists from workflow row selections."""

    def __init__(
        self,
        *,
        dedupe_jobs: Callable[[Sequence[Job]], Sequence[Job]] | None = None,
    ) -> None:
        self._dedupe_jobs = dedupe_jobs or self._dedupe_by_id

    def submission_plan(
        self,
        values: Sequence[_WorkflowKey],
        *,
        normalize: Callable[[_WorkflowKey], _WorkflowKey],
        build_states: Callable[[], Mapping[_WorkflowKey, object]],
        jobs_for_values: Callable[[list[_WorkflowKey]], Sequence[Job]],
        plan_values: Callable[[list[_WorkflowKey]], Sequence[Job]],
        is_ready: Callable[[Mapping[_WorkflowKey, object], _WorkflowKey], bool],
    ) -> WorkflowSubmissionPlan[_WorkflowKey]:
        """Build a deduplicated list of the rows' existing jobs and the jobs planned for the rest.

        ``plan_values`` plans the ready rows' jobs and changes nothing: no job
        is created before the submission is accepted.
        """

        ordered = self.ordered_unique_values(values, normalize)
        states = build_states()
        ready_values = [value for value in ordered if is_ready(states, value)]
        jobs = list(jobs_for_values(ordered))
        jobs.extend(plan_values(ready_values))
        return WorkflowSubmissionPlan(
            ordered_values=tuple(ordered),
            ready_values=tuple(ready_values),
            jobs=tuple(self._dedupe_jobs(jobs)),
        )

    @staticmethod
    def ordered_unique_values(
        values: Sequence[_WorkflowKey],
        normalize: Callable[[_WorkflowKey], _WorkflowKey],
    ) -> list[_WorkflowKey]:
        """Return normalized values in first-seen order."""

        ordered: list[_WorkflowKey] = []
        seen: set[_WorkflowKey] = set()
        for value in values:
            normalized = normalize(value)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
        return ordered

    @staticmethod
    def _dedupe_by_id(jobs: Sequence[Job]) -> list[Job]:
        deduped: list[Job] = []
        seen: set[object] = set()
        for job in jobs:
            if job.id in seen:
                continue
            seen.add(job.id)
            deduped.append(job)
        return deduped

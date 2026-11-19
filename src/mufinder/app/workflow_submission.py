"""Shared workflow submission planning helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from mufinder.services.job_monitoring import job_status_word

_WorkflowKey = TypeVar("_WorkflowKey")

#: Runtime/display statuses a workflow row may be in and still be submittable.
#: This is the single source of truth shared by the submit predicate
#: (:func:`workflow_row_state_is_ready`) and by widget-level submit-enablement
#: gates, so run-button enablement and the submit filter cannot drift apart.
#: A row whose job ended (``failed``, ``stopped``, ``cancelled``) is
#: submittable: the shared submit cycle resets the job to ``defined`` first.
#: Only :data:`UNSUBMITTED_WORKFLOW_ROW_STATUSES` rows are configurable, so an
#: ended row's inputs stay those its job ran with until it is reset.
UNSUBMITTED_WORKFLOW_ROW_STATUSES = frozenset({"ready", "defined", "not generated"})
READY_WORKFLOW_ROW_STATUSES = UNSUBMITTED_WORKFLOW_ROW_STATUSES | {
    "failed",
    "stopped",
    "cancelled",
}
_READY_WORKFLOW_ROW_STATUSES = READY_WORKFLOW_ROW_STATUSES


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


def workflow_row_state_is_unsubmitted(
    states: Mapping[_WorkflowKey, object],
    key: _WorkflowKey,
) -> bool:
    """Return whether a workflow row has no submitted job, so its configuration can change."""

    return str(getattr(states.get(key), "status", "")) in UNSUBMITTED_WORKFLOW_ROW_STATUSES


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
        materialize_values: Callable[[list[_WorkflowKey]], Sequence[Job]],
        is_ready: Callable[[Mapping[_WorkflowKey, object], _WorkflowKey], bool],
    ) -> WorkflowSubmissionPlan[_WorkflowKey]:
        """Build a deduplicated list of existing and newly materialized jobs."""

        ordered = self.ordered_unique_values(values, normalize)
        states = build_states()
        ready_values = [value for value in ordered if is_ready(states, value)]
        jobs = list(jobs_for_values(ordered))
        jobs.extend(materialize_values(ready_values))
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

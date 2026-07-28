"""Shared workflow submission planning helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job

_WorkflowKey = TypeVar("_WorkflowKey")

#: Runtime/display statuses a workflow row may be in and still be submittable.
#: This is the single source of truth shared by the submit predicate
#: (:func:`workflow_row_state_is_ready`) and by widget-level submit-enablement
#: gates, so run-button enablement and the submit filter cannot drift apart.
READY_WORKFLOW_ROW_STATUSES = frozenset({"ready", "defined", "not generated"})
_READY_WORKFLOW_ROW_STATUSES = READY_WORKFLOW_ROW_STATUSES

#: Terminal job statuses whose *dismissed* jobs project a submittable workflow
#: row. Dismissing a terminal job from the jobs dashboard removes its row — and
#: with it the only generic Reset/Resubmit controls — so dismissal is read as
#: "failure acknowledged, this row is actionable again". The set is deliberately
#: written out here rather than imported from ``services`` (the app layer may
#: import inward, but duplicating the literal is what lets a test prove the two
#: definitions agree); it must stay identical to
#: ``ProjectService.TERMINAL_RECOVERABLE_JOB_STATUSES``, which gates both
#: ``dismiss_job`` and ``reset_failed_job``.
RESUBMITTABLE_AFTER_DISMISSAL_STATUSES = frozenset(
    {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
)


def job_is_resubmittable_after_dismissal(job: Job | None) -> bool:
    """Return whether ``job`` is a dismissed terminal job awaiting a reset.

    Such a job keeps its terminal status (dismissal is not a status transition)
    but has been acknowledged by the user, so its workflow row becomes
    submittable again and the shared controller submit cycle resets the job back
    to ``defined`` before dispatching it.
    """

    if job is None:
        return False
    return job.status in RESUBMITTABLE_AFTER_DISMISSAL_STATUSES and job.dismissed_at is not None


def effective_row_status(job: Job | None, *, ready_status: str = "ready") -> str:
    """Project ``job`` onto the status string its workflow table row shows.

    This is the **only** place the dismissed-terminal rule lives: a dismissed
    failed/stopped/cancelled job reads as ``ready_status`` so its row can be
    submitted again, while every other job (including a *non*-dismissed failed
    one, whose recovery path is still the dashboard Resubmit/Reset controls)
    reads as its own status value. A ``None`` job also reads as ``ready_status``
    — a row with no job has nothing to run yet.
    """

    if job is None or job_is_resubmittable_after_dismissal(job):
        return ready_status
    return job.status.value


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


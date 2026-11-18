"""Active-job watchdog: detect local jobs stuck active with no process behind them.

A local job can stay in an active state (``submitting``/``queued``/``running``)
with nothing executing it: a ``QProcess`` that died without a finish event, or a
queue entry that never reached the runner. Orphaned-local reconciliation (see
``ProjectService.reconcile_orphaned_local_jobs``) covers the app-restart case;
this module covers the running session.

Cluster jobs are never flagged. A cluster job that no runner tracks may still be
queued, running or finished on the cluster, so losing track of it is not
evidence that it ended. The dashboard marks such a job as not monitored
(``services.job_monitoring.job_staleness``) and it stays active.

The core is a pure, Qt-free function, :func:`detect_stuck_jobs`, plus a small
stateful wrapper, :class:`StuckJobWatchdog`, that requires a finding to persist
across two consecutive evaluations before acting. The two-tick confirmation
avoids racing an in-flight runner signal that has not yet reached the document.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import UUID

from mufinder.domain.enums import ACTIVE_JOB_STATUSES, JobStatus
from mufinder.domain.models import Job
from mufinder.services.job_monitoring import is_cluster_job

DEFAULT_GRACE_SECONDS = 120
"""Age a ``submitting`` local job must exceed before it can be flagged."""

_LOCAL_STUCK_MESSAGE = (
    "MuFinder's records say this job is {status}, but no local process is "
    "tracking it. The job watchdog stopped it; reset or resubmit it."
)


def _age_seconds(now: datetime, then: datetime) -> float:
    # Stamps written by older builds may be naive; treat them as UTC, matching
    # mufinder.domain.models.utc_now.
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (now - then).total_seconds()


def detect_stuck_jobs(
    jobs: Iterable[Job],
    *,
    local_active_ids: frozenset[UUID],
    local_pending_ids: frozenset[UUID],
    now: datetime,
    grace_seconds: int = DEFAULT_GRACE_SECONDS,
) -> list[tuple[Job, str]]:
    """Return ``(job, message)`` findings for active local jobs no process is running.

    Only jobs in :data:`ACTIVE_JOB_STATUSES` are considered, and cluster jobs
    (:func:`services.job_monitoring.is_cluster_job`) are skipped.
    """

    findings: list[tuple[Job, str]] = []
    local_known = local_active_ids | local_pending_ids

    for job in jobs:
        if job.status not in ACTIVE_JOB_STATUSES:
            continue

        if is_cluster_job(job):
            continue

        if job.status == JobStatus.QUEUED:
            if job.id not in local_known:
                findings.append((job, _LOCAL_STUCK_MESSAGE.format(status=job.status.value)))
        elif job.status == JobStatus.RUNNING:
            if job.id not in local_active_ids:
                findings.append((job, _LOCAL_STUCK_MESSAGE.format(status=job.status.value)))
        elif job.status == JobStatus.SUBMITTING:
            if job.submitted_at is None:
                continue
            if _age_seconds(now, job.submitted_at) <= grace_seconds:
                continue
            if job.id not in local_known:
                findings.append((job, _LOCAL_STUCK_MESSAGE.format(status=job.status.value)))
        # RETRIEVING local jobs are not flagged: local runs never enter the
        # retrieving state (only cluster retrieval does), so there is no local
        # runner condition to check against.

    return findings


class StuckJobWatchdog:
    """Confirm findings across two consecutive evaluations before acting.

    An in-flight runner signal (e.g. a just-started process whose ``running``
    notification has not yet reached the document) can make a healthy job look
    momentarily stuck. Requiring a job to be flagged in two consecutive
    :meth:`evaluate` calls collapses that race: a genuine stuck job stays flagged;
    a transient one clears on the next tick.
    """

    def __init__(self, *, grace_seconds: int = DEFAULT_GRACE_SECONDS) -> None:
        self._grace_seconds = grace_seconds
        # Jobs flagged on the previous evaluation, with their message.
        self._pending: dict[UUID, str] = {}

    def evaluate(
        self,
        jobs: Iterable[Job],
        *,
        local_active_ids: frozenset[UUID],
        local_pending_ids: frozenset[UUID],
        now: datetime,
    ) -> list[tuple[Job, str]]:
        """Return confirmed findings (flagged on this tick and the previous one).

        A job that disappears from the findings resets its strike, so a recovered
        job must be flagged twice again before it is acted on.
        """

        jobs = list(jobs)
        current = detect_stuck_jobs(
            jobs,
            local_active_ids=local_active_ids,
            local_pending_ids=local_pending_ids,
            now=now,
            grace_seconds=self._grace_seconds,
        )
        current_by_id = {job.id: (job, message) for job, message in current}

        confirmed: list[tuple[Job, str]] = []
        for job_id, (job, message) in current_by_id.items():
            if job_id in self._pending:
                confirmed.append((job, message))

        # Carry only jobs still flagged this tick into the next strike window.
        self._pending = {job_id: message for job_id, (_job, message) in current_by_id.items()}
        return confirmed

    def forget(self, job_id: UUID) -> None:
        """Drop a job's pending strike (e.g. after it was stopped)."""

        self._pending.pop(job_id, None)

    def reset(self) -> None:
        """Clear all pending strikes (e.g. on project switch)."""

        self._pending.clear()

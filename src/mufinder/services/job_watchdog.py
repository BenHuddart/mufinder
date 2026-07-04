"""Active-job watchdog: detect jobs stuck active with no runner tracking them.

Several failure modes can leave a job in an active state
(``submitting``/``queued``/``running``/``retrieving``) with nothing actually
executing it: a local ``QProcess`` that died without a finish event, a cluster
job the runner silently stopped tracking, or a queue entry that never reached a
runner. Orphaned-local reconciliation (see
``ProjectService.reconcile_orphaned_local_jobs``) only covers the app-restart
case; this module covers the *running-session* case periodically.

The core is a pure, Qt-free function, :func:`detect_stuck_jobs`, plus a small
stateful wrapper, :class:`StuckJobWatchdog`, that requires a finding to persist
across two consecutive evaluations before acting. The two-tick confirmation
avoids racing an in-flight runner signal that has not yet reached the document.

Design notes / deviations from the recommendation sketch:

* The recommendation's signature named a bare ``cluster_connected`` bool. To
  match the semantics already pinned by ``services.job_monitoring.job_staleness``
  and the runner's ``_tracked_job_matches_connected_cluster``, this function also
  accepts ``connected_cluster_name`` and only flags a cluster job when the
  connected cluster is the one *that job* targets (blank job target = wildcard,
  case-insensitive compare). A watchdog false positive stops a healthy job, so
  every ambiguous case is biased toward *not* flagging.
* When ``cluster_connected`` is ``False`` no cluster job is ever flagged: a
  disconnected cluster is the staleness feature's job (a dashboard marker), not
  the watchdog's.
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
_CLUSTER_STUCK_MESSAGE = (
    "MuFinder's records say this job is {status}, but the connected cluster "
    "session is not tracking it. The job watchdog stopped it; reset or "
    "resubmit it."
)


def _age_seconds(now: datetime, then: datetime) -> float:
    # Stamps written by older builds may be naive; treat them as UTC, matching
    # mufinder.domain.models.utc_now.
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (now - then).total_seconds()


def _cluster_matches_connected(job: Job, connected_cluster_name: str | None) -> bool:
    """Return True when the connected cluster is the one this job targets.

    Mirrors ``ClusterJobRunner._tracked_job_matches_connected_cluster``: a blank
    job target or a blank connected name is treated as a wildcard match.
    """

    job_target = (job.execution_target or "").strip()
    connected = (connected_cluster_name or "").strip()
    if not job_target or not connected:
        return True
    return job_target.casefold() == connected.casefold()


def detect_stuck_jobs(
    jobs: Iterable[Job],
    *,
    local_active_ids: frozenset[UUID],
    local_pending_ids: frozenset[UUID],
    cluster_tracked_ids: frozenset[UUID],
    cluster_connected: bool,
    now: datetime,
    connected_cluster_name: str | None = None,
    grace_seconds: int = DEFAULT_GRACE_SECONDS,
) -> list[tuple[Job, str]]:
    """Return ``(job, message)`` findings for active jobs no runner is tracking.

    Only jobs in :data:`ACTIVE_JOB_STATUSES` are considered. Local and cluster
    jobs are distinguished with :func:`services.job_monitoring.is_cluster_job`.
    See the module docstring for the cluster connection/target semantics.
    """

    findings: list[tuple[Job, str]] = []
    local_known = local_active_ids | local_pending_ids

    for job in jobs:
        if job.status not in ACTIVE_JOB_STATUSES:
            continue

        if is_cluster_job(job):
            if not cluster_connected:
                continue
            if not _cluster_matches_connected(job, connected_cluster_name):
                continue
            if job.id in cluster_tracked_ids:
                continue
            findings.append((job, _CLUSTER_STUCK_MESSAGE.format(status=job.status.value)))
            continue

        # Local job.
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
        cluster_tracked_ids: frozenset[UUID],
        cluster_connected: bool,
        now: datetime,
        connected_cluster_name: str | None = None,
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
            cluster_tracked_ids=cluster_tracked_ids,
            cluster_connected=cluster_connected,
            now=now,
            connected_cluster_name=connected_cluster_name,
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

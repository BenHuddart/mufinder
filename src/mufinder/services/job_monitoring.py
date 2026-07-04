"""Staleness assessment for monitored jobs shown on the dashboard.

An active cluster job's persisted status is only as fresh as the last queue
poll that confirmed it. This module derives a human-readable staleness marker
from the job's monitor metadata (``last_checked_at``) and the current cluster
connection state, so the dashboard can distinguish "still queued" from "not
checked recently" without inventing a new ``JobStatus``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mufinder.domain.enums import ACTIVE_JOB_STATUSES
from mufinder.domain.models import Job

STALENESS_THRESHOLD_SECONDS = 180
"""Age beyond which a last check counts as stale (three 60 s poll cycles)."""


def is_cluster_job(job: Job) -> bool:
    """Return True when the job targets a cluster rather than local execution.

    Mirrors the local-job selection used by
    ``ProjectService.reconcile_orphaned_local_jobs``: a blank/empty
    ``execution_target`` or the literal ``"local"`` (case-insensitively) means
    local execution; anything else names a cluster.
    """

    target = (job.execution_target or "").strip()
    return bool(target) and target.casefold() != "local"


def _age_seconds(now: datetime, then: datetime) -> float:
    # Stamps written by older builds may be naive; treat them as UTC, the
    # convention used by mufinder.domain.models.utc_now.
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (now - then).total_seconds()


def job_staleness(
    job: Job,
    *,
    now: datetime,
    cluster_connected: bool,
    threshold_seconds: int = STALENESS_THRESHOLD_SECONDS,
) -> str | None:
    """Return a short reason the job's displayed status may be out of date.

    Only active cluster jobs can be stale; everything else returns ``None``.
    ``cluster_connected`` must describe the cluster *this job* targets: pass
    ``connected_cluster_name == job.execution_target``, not merely "some
    cluster is connected".
    """

    if job.status not in ACTIVE_JOB_STATUSES or not is_cluster_job(job):
        return None
    if not cluster_connected:
        return "cluster disconnected"
    if job.last_checked_at is not None:
        age = _age_seconds(now, job.last_checked_at)
        if age > threshold_seconds:
            return f"not checked for {int(age // 60)} min"
        return None
    if job.submitted_at is not None and _age_seconds(now, job.submitted_at) > threshold_seconds:
        return "not checked since submission"
    return None

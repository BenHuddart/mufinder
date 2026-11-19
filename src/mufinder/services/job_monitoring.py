"""Staleness assessment for monitored jobs shown on the dashboard.

An active cluster job's persisted status is only as fresh as the last queue
poll that confirmed it. This module derives a human-readable staleness marker
from the job's monitor metadata (``last_checked_at``) and the current cluster
connection state, so the dashboard can distinguish "still queued" from "not
checked recently" without inventing a new ``JobStatus``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from mufinder.domain.enums import ACTIVE_JOB_STATUSES, FailureReason, JobStatus
from mufinder.domain.models import Job

STALENESS_THRESHOLD_SECONDS = 180
"""Age beyond which a last check counts as stale (three 60 s poll cycles)."""

#: Marker of an active cluster job whose cluster is not connected.
DISCONNECTED_MARKER = "cluster disconnected"
#: Marker of an active cluster job that has no scheduler job ID.
NO_JOB_ID_MARKER = "no scheduler job ID was recorded"


def is_cluster_job(job: Job) -> bool:
    """Return True when the job targets a cluster rather than local execution.

    A blank/empty ``execution_target`` or the literal ``"local"``
    (case-insensitively) means local execution; anything else names a cluster.
    """

    target = (job.execution_target or "").strip()
    return bool(target) and target.casefold() != "local"


def job_where_text(job: Job) -> str:
    """Return where the job runs: ``local``, ``SCARF`` or ``SCARF #4127731``."""

    if not is_cluster_job(job):
        return "local"
    remote_id = (job.remote_job_id or "").strip()
    target = job.execution_target.strip()
    return f"{target} #{remote_id}" if remote_id else target


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
    marker: str | None = None,
    submitting: bool = False,
    threshold_seconds: int = STALENESS_THRESHOLD_SECONDS,
) -> str | None:
    """Return a short reason the job's displayed status may be out of date.

    Only active cluster jobs can be stale; everything else returns ``None``.
    ``cluster_connected`` must describe the cluster *this job* targets: pass
    ``connected_cluster_name == job.execution_target``, not merely "some
    cluster is connected". ``marker`` is the note the job's last check left
    when it could not confirm the job (see
    :class:`~mufinder.services.job_reconcile.Decision`); it outranks the age
    of the last check, because that check did happen. ``submitting`` says a
    submit request for the job has not answered yet: such a job has no
    scheduler job ID by design and is not stale.
    """

    if job.status not in ACTIVE_JOB_STATUSES or not is_cluster_job(job):
        return None
    if not cluster_connected:
        return DISCONNECTED_MARKER
    if submitting:
        return None
    if marker:
        return marker
    if not (job.remote_job_id or "").strip():
        return NO_JOB_ID_MARKER
    if job.last_checked_at is not None:
        age = _age_seconds(now, job.last_checked_at)
        if age > threshold_seconds:
            return f"not checked for {int(age // 60)} min"
        return None
    if job.submitted_at is not None and _age_seconds(now, job.submitted_at) > threshold_seconds:
        return "not checked since submission"
    return None


#: Failure reasons under which a cluster job's output was never brought back.
_UNRETRIEVED_OUTPUT_REASONS = frozenset(
    {FailureReason.OUTPUT_UNAVAILABLE, FailureReason.INTERRUPTED}
)


def has_unretrieved_cluster_output(job: Job) -> bool:
    """Return whether ``job`` ended without its output being retrieved from the cluster.

    True for a failed or stopped cluster job that kept its remote folder and
    whose recorded cause is a failed retrieval or MuFinder losing track of it.
    The calculation itself may have finished, so retrieval is worth trying
    before the job is resubmitted or dismissed.
    """

    return (
        job.status in {JobStatus.FAILED, JobStatus.STOPPED}
        and is_cluster_job(job)
        and job.failure_reason in _UNRETRIEVED_OUTPUT_REASONS
        and bool((job.remote_directory or "").strip())
    )


def unretrieved_output_note(jobs: Sequence[Job], *, prompt_job_count: int | None = None) -> str:
    """Return the dismissal warning for the jobs in ``jobs`` with unretrieved output.

    Empty when none has any. Names the clusters involved, first seen first.
    ``prompt_job_count`` is how many jobs the confirmation is about when it
    covers more than ``jobs`` (the jobs it also cancels); the note says "Its
    output" only when the prompt is about a single job.
    """

    unretrieved = [job for job in jobs if has_unretrieved_cluster_output(job)]
    if not unretrieved:
        return ""
    clusters: dict[str, str] = {}
    for job in unretrieved:
        target = job.execution_target.strip()
        clusters.setdefault(target.casefold(), target)
    where = " or ".join(clusters.values())
    if (len(jobs) if prompt_job_count is None else prompt_job_count) == 1:
        return (
            f"Its output may still be on {where} and has not been retrieved. Dismissing "
            "does not retrieve it; use Check now first to keep the result."
        )
    count = len(unretrieved)
    return (
        f"The output of {count} of these jobs may still be on {where} and has not been "
        "retrieved. Dismissing does not retrieve it; use Check now first to keep "
        f"the {'result' if count == 1 else 'results'}."
    )

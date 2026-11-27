"""Gather the evidence :func:`~mufinder.services.job_reconcile.reconcile` decides from.

The functions here look at a job from outside: which process holds it and what
its output folder contains. They change nothing and decide nothing.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult, parse_castep_output
from mufinder.domain.models import Job
from mufinder.services.job_progress import castep_output_names
from mufinder.services.job_reconcile import JobEvidence, OutputState, ProcessState

LOCAL_STALE_OUTPUT_SLACK_SECONDS = 5.0
"""Slack allowed when ordering a local output's mtime against the job's submission.

Covers filesystem mtime granularity and the clock offset of a network mount.
"""


def local_job_evidence(
    job: Job,
    *,
    pending_ids: Collection[UUID],
    active_ids: Collection[UUID],
    output_dir: Path | None,
    seed_name: str,
    now: datetime,
    on_request: bool = False,
) -> JobEvidence:
    """Return what the local runner and the output folder show for local ``job``.

    ``pending_ids`` and ``active_ids`` are the local runner's queued and running
    jobs. The output folder is read only when the runner holds no process for
    the job, since a held job is still writing it.
    """

    if job.id in active_ids:
        return JobEvidence(checked_at=now, process=ProcessState.RUNNING, on_request=on_request)
    if job.id in pending_ids:
        return JobEvidence(checked_at=now, process=ProcessState.PENDING, on_request=on_request)
    output, detail, parsed = local_output_evidence(
        output_dir, seed_name, not_before=job.submitted_at
    )
    return JobEvidence(
        checked_at=now,
        process=ProcessState.NONE,
        output=output,
        output_detail=detail,
        parsed=parsed,
        on_request=on_request,
    )


def local_output_evidence(
    output_dir: Path | None,
    seed_name: str,
    *,
    not_before: datetime | None,
) -> tuple[OutputState, str, CastepResult | None]:
    """Return the state of a local job's CASTEP output, a detail and the parsed result.

    The output is ``<seed>_farm001.castep`` when a task-farmed run wrote one,
    otherwise ``<seed>.castep``. A file last written before ``not_before`` (the
    job's submission) belongs to an earlier run in the same folder and counts
    as no output. The detail is CASTEP's error message for incomplete output
    and the parse error for unreadable output.
    """

    if output_dir is None:
        return OutputState.ABSENT, "", None
    earliest = None if not_before is None else _timestamp(not_before)
    for name in castep_output_names(seed_name, 2):
        path = output_dir / name
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if earliest is not None and earliest - modified > LOCAL_STALE_OUTPUT_SLACK_SECONDS:
            continue
        try:
            parsed = parse_castep_output(path)
        except Exception as exc:  # noqa: BLE001 - reported as unreadable output
            return OutputState.UNREADABLE, str(exc), None
        if parsed.completed:
            return OutputState.COMPLETE, "", parsed
        return OutputState.INCOMPLETE, parsed.error_message or "", parsed
    return OutputState.ABSENT, "", None


def _timestamp(moment: datetime) -> float:
    # Stamps written by older builds may be naive; treat them as UTC, the
    # convention used by mufinder.domain.models.utc_now.
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()

"""Tests for the active-job watchdog detection core and stop path (REC #7)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_execution_controller import JobExecutionController
from mufinder.services.job_watchdog import (
    DEFAULT_GRACE_SECONDS,
    StuckJobWatchdog,
    detect_stuck_jobs,
)
from mufinder.services.project_service import ProjectDocument, ProjectService

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

_LOCAL_MESSAGE_TEMPLATE = (
    "MuFinder's records say this job is {status}, but no local process is "
    "tracking it. The job watchdog stopped it; reset or resubmit it."
)
_CLUSTER_MESSAGE_TEMPLATE = (
    "MuFinder's records say this job is {status}, but the connected cluster "
    "session is not tracking it. The job watchdog stopped it; reset or "
    "resubmit it."
)


def _local_job(status: JobStatus, *, submitted_at: datetime | None = None) -> Job:
    return Job(
        label="local",
        status=status,
        execution_target="local",
        submitted_at=submitted_at,
    )


def _cluster_job(status: JobStatus, *, cluster: str = "scarf") -> Job:
    return Job(label="remote", status=status, execution_target=cluster)


def _detect(jobs, **overrides):
    kwargs = {
        "local_active_ids": frozenset(),
        "local_pending_ids": frozenset(),
        "cluster_tracked_ids": frozenset(),
        "cluster_connected": True,
        "now": _NOW,
        "connected_cluster_name": "scarf",
    }
    kwargs.update(overrides)
    return detect_stuck_jobs(jobs, **kwargs)


# ----------------------------------------------------------------------------
# Local detection rules
# ----------------------------------------------------------------------------
def test_local_queued_without_runner_entry_is_stuck() -> None:
    job = _local_job(JobStatus.QUEUED)
    findings = _detect([job])
    assert findings == [(job, _LOCAL_MESSAGE_TEMPLATE.format(status="queued"))]


def test_local_queued_tracked_as_pending_is_not_stuck() -> None:
    job = _local_job(JobStatus.QUEUED)
    assert _detect([job], local_pending_ids=frozenset({job.id})) == []


def test_local_queued_tracked_as_active_is_not_stuck() -> None:
    job = _local_job(JobStatus.QUEUED)
    assert _detect([job], local_active_ids=frozenset({job.id})) == []


def test_local_running_without_active_process_is_stuck() -> None:
    job = _local_job(JobStatus.RUNNING)
    findings = _detect([job])
    assert findings == [(job, _LOCAL_MESSAGE_TEMPLATE.format(status="running"))]


def test_local_running_tracked_as_active_is_not_stuck() -> None:
    job = _local_job(JobStatus.RUNNING)
    assert _detect([job], local_active_ids=frozenset({job.id})) == []


def test_local_running_only_pending_is_still_stuck() -> None:
    # RUNNING requires an *active* process; a pending entry is not enough.
    job = _local_job(JobStatus.RUNNING)
    findings = _detect([job], local_pending_ids=frozenset({job.id}))
    assert findings == [(job, _LOCAL_MESSAGE_TEMPLATE.format(status="running"))]


def test_local_submitting_within_grace_is_not_stuck() -> None:
    job = _local_job(
        JobStatus.SUBMITTING,
        submitted_at=_NOW - timedelta(seconds=DEFAULT_GRACE_SECONDS - 1),
    )
    assert _detect([job]) == []


def test_local_submitting_past_grace_without_runner_entry_is_stuck() -> None:
    job = _local_job(
        JobStatus.SUBMITTING,
        submitted_at=_NOW - timedelta(seconds=DEFAULT_GRACE_SECONDS + 1),
    )
    findings = _detect([job])
    assert findings == [(job, _LOCAL_MESSAGE_TEMPLATE.format(status="submitting"))]


def test_local_submitting_past_grace_but_tracked_is_not_stuck() -> None:
    job = _local_job(
        JobStatus.SUBMITTING,
        submitted_at=_NOW - timedelta(seconds=DEFAULT_GRACE_SECONDS + 1),
    )
    assert _detect([job], local_pending_ids=frozenset({job.id})) == []


def test_local_submitting_without_submitted_at_is_skipped() -> None:
    job = _local_job(JobStatus.SUBMITTING, submitted_at=None)
    assert _detect([job]) == []


def test_local_retrieving_is_never_flagged() -> None:
    # Local runs never enter retrieving; there is no local runner to check.
    job = _local_job(JobStatus.RETRIEVING)
    assert _detect([job]) == []


def test_naive_submitted_at_is_treated_as_utc() -> None:
    job = _local_job(
        JobStatus.SUBMITTING,
        submitted_at=(_NOW - timedelta(seconds=DEFAULT_GRACE_SECONDS + 5)).replace(tzinfo=None),
    )
    findings = _detect([job])
    assert findings == [(job, _LOCAL_MESSAGE_TEMPLATE.format(status="submitting"))]


# ----------------------------------------------------------------------------
# Cluster detection rules
# ----------------------------------------------------------------------------
def test_cluster_untracked_while_connected_is_stuck() -> None:
    job = _cluster_job(JobStatus.RUNNING)
    findings = _detect([job])
    assert findings == [(job, _CLUSTER_MESSAGE_TEMPLATE.format(status="running"))]


def test_cluster_tracked_is_not_stuck() -> None:
    job = _cluster_job(JobStatus.RUNNING)
    assert _detect([job], cluster_tracked_ids=frozenset({job.id})) == []


def test_cluster_disconnected_is_never_flagged() -> None:
    job = _cluster_job(JobStatus.RUNNING)
    assert _detect([job], cluster_connected=False, connected_cluster_name=None) == []


def test_cluster_job_targeting_other_cluster_is_not_flagged() -> None:
    # Connected to "scarf" but the job targets "young": not our concern.
    job = _cluster_job(JobStatus.RUNNING, cluster="young")
    assert _detect([job], connected_cluster_name="scarf") == []


def test_cluster_job_matches_connected_cluster_case_insensitively() -> None:
    job = _cluster_job(JobStatus.QUEUED, cluster="SCARF")
    findings = _detect([job], connected_cluster_name="scarf")
    assert findings == [(job, _CLUSTER_MESSAGE_TEMPLATE.format(status="queued"))]


def test_cluster_blank_connected_name_is_wildcard() -> None:
    # Some connect paths may not carry the cluster name; a blank name matches.
    job = _cluster_job(JobStatus.RUNNING, cluster="scarf")
    findings = _detect([job], connected_cluster_name="")
    assert findings == [(job, _CLUSTER_MESSAGE_TEMPLATE.format(status="running"))]


# ----------------------------------------------------------------------------
# General filtering
# ----------------------------------------------------------------------------
def test_terminal_jobs_are_ignored() -> None:
    jobs = [
        _local_job(JobStatus.COMPLETED),
        _local_job(JobStatus.FAILED),
        _local_job(JobStatus.STOPPED),
        _local_job(JobStatus.CANCELLED),
        Job(label="defined", status=JobStatus.DEFINED, execution_target="local"),
    ]
    assert _detect(jobs) == []


# ----------------------------------------------------------------------------
# Two-tick confirmation
# ----------------------------------------------------------------------------
def _evaluate(watchdog: StuckJobWatchdog, jobs, **overrides):
    kwargs = {
        "local_active_ids": frozenset(),
        "local_pending_ids": frozenset(),
        "cluster_tracked_ids": frozenset(),
        "cluster_connected": True,
        "now": _NOW,
        "connected_cluster_name": "scarf",
    }
    kwargs.update(overrides)
    return watchdog.evaluate(jobs, **kwargs)


def test_two_tick_confirmation_requires_two_consecutive_hits() -> None:
    watchdog = StuckJobWatchdog()
    job = _local_job(JobStatus.RUNNING)

    # First tick: flagged but not yet confirmed.
    assert _evaluate(watchdog, [job]) == []
    # Second consecutive tick: confirmed.
    confirmed = _evaluate(watchdog, [job])
    assert confirmed == [(job, _LOCAL_MESSAGE_TEMPLATE.format(status="running"))]


def test_two_tick_recovery_resets_the_strike() -> None:
    watchdog = StuckJobWatchdog()
    job = _local_job(JobStatus.RUNNING)

    # First tick flags it.
    assert _evaluate(watchdog, [job]) == []
    # Second tick: the runner is now tracking it (recovered) -> not flagged.
    assert _evaluate(watchdog, [job], local_active_ids=frozenset({job.id})) == []
    # Third tick: stuck again, but the strike was reset -> still not confirmed.
    assert _evaluate(watchdog, [job]) == []
    # Fourth tick: now two consecutive -> confirmed.
    assert _evaluate(watchdog, [job]) == [
        (job, _LOCAL_MESSAGE_TEMPLATE.format(status="running"))
    ]


def test_forget_drops_pending_strike() -> None:
    watchdog = StuckJobWatchdog()
    job = _local_job(JobStatus.RUNNING)
    assert _evaluate(watchdog, [job]) == []
    watchdog.forget(job.id)
    # Strike forgotten: needs two fresh consecutive hits again.
    assert _evaluate(watchdog, [job]) == []
    assert _evaluate(watchdog, [job]) == [
        (job, _LOCAL_MESSAGE_TEMPLATE.format(status="running"))
    ]


def test_reset_clears_all_pending_strikes() -> None:
    watchdog = StuckJobWatchdog()
    job = _local_job(JobStatus.RUNNING)
    assert _evaluate(watchdog, [job]) == []
    watchdog.reset()
    assert _evaluate(watchdog, [job]) == []


# ----------------------------------------------------------------------------
# Controller stop path
# ----------------------------------------------------------------------------
class _FakeRunner:
    def submit(self, **kwargs: object) -> object:  # pragma: no cover - unused here
        return None


def _controller_with_active_job(
    tmp_path: Path, status: JobStatus
) -> tuple[JobExecutionController, ProjectService, ProjectDocument, UUID, UUID]:
    service = ProjectService()
    document = service.create_project(tmp_path / "watchdog.mufinder", "Watchdog")
    job = service.add_job(
        document,
        label="job",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
    )
    # Drive the job into the requested active status through the primitives.
    service.submit_job(job)
    if status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING}:
        service.mark_job_queued(job, remote_job_id=None)
    if status in {JobStatus.RUNNING, JobStatus.RETRIEVING}:
        service.start_job(job)
    if status == JobStatus.RETRIEVING:
        service.mark_job_retrieving(job)
    assert job.status == status

    run = CalculationRun(job_id=job.id, run_index=1, status=status)
    document.runs[run.id] = run
    active_run_by_job = {job.id: run.id}

    controller = JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=_FakeRunner(),
        cluster_runner=_FakeRunner(),
        job_output_directory=lambda _job: tmp_path,
        job_seed_name=lambda job: job.label,
        active_run_by_job=active_run_by_job,
    )
    return controller, service, document, job.id, run.id


def test_stop_stuck_job_stops_and_stamps_bookkeeping(tmp_path: Path) -> None:
    controller, _service, document, job_id, run_id = _controller_with_active_job(
        tmp_path, JobStatus.RUNNING
    )
    message = _LOCAL_MESSAGE_TEMPLATE.format(status="running")

    update = controller.stop_stuck_job(job_id, message)

    job = document.jobs[job_id]
    run = document.runs[run_id]
    assert update.changed is True
    assert update.journal_changed is True
    assert job.status == JobStatus.STOPPED
    assert job.error_message == message
    assert job.last_status_source == "watchdog"
    assert job.last_status_message == message
    assert job.last_checked_at is not None
    assert run.status == JobStatus.STOPPED
    assert run.completed_at is not None
    assert job_id not in controller.active_run_by_job


def test_stop_stuck_job_records_journal_event(tmp_path: Path) -> None:
    controller, _service, document, job_id, _run_id = _controller_with_active_job(
        tmp_path, JobStatus.QUEUED
    )
    before = len(document.provenance)

    controller.stop_stuck_job(job_id, "msg")

    assert len(document.provenance) == before + 1


def test_stop_stuck_job_ignores_unknown_job(tmp_path: Path) -> None:
    controller, _service, _document, _job_id, _run_id = _controller_with_active_job(
        tmp_path, JobStatus.RUNNING
    )
    update = controller.stop_stuck_job(uuid4(), "msg")
    assert update.job is None
    assert update.changed is False


def test_stop_stuck_job_ignores_terminal_job(tmp_path: Path) -> None:
    controller, service, document, job_id, _run_id = _controller_with_active_job(
        tmp_path, JobStatus.RUNNING
    )
    service.fail_job(document.jobs[job_id], "already failed")

    update = controller.stop_stuck_job(job_id, "msg")

    assert update.changed is False
    assert document.jobs[job_id].status == JobStatus.FAILED


def test_stop_stuck_job_stops_submitting_job(tmp_path: Path) -> None:
    controller, _service, document, job_id, _run_id = _controller_with_active_job(
        tmp_path, JobStatus.SUBMITTING
    )
    update = controller.stop_stuck_job(job_id, "msg")
    assert update.changed is True
    assert document.jobs[job_id].status == JobStatus.STOPPED

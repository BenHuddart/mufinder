"""Tests for the continuation primitive (relaxation warm starts).

``ProjectService.prepare_job_continuation`` returns a finished job to DEFINED so
it can be resubmitted into the *same* directory, preserving ``remote_directory``
as the durable marker a continuation run is wanted. ``record_job_continuation``
appends the matching provenance milestone. See docs/job_lifecycle.rst.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.project_service import (
    CONTINUABLE_JOB_STATUSES,
    TERMINAL_RECOVERABLE_JOB_STATUSES,
    ProjectService,
)
from tests.support.jobs import make_job

_CONTINUABLE = [
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.STOPPED,
    JobStatus.CANCELLED,
]


_job = partial(
    make_job,
    label="relax_row_1",
    workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
    calc_type="castep_pristine_relaxation",
)


def _job_with_run_metadata(status: JobStatus) -> Job:
    job = _job(status=status)
    stamp = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
    job.submitted_at = stamp
    job.started_at = stamp
    job.completed_at = stamp
    job.remote_job_id = "123456"
    job.remote_directory = "/scratch/user/relax_row_1"
    job.num_processes = 32
    job.error_message = "walltime exceeded"
    job.last_checked_at = stamp
    job.last_status_source = "scheduler_poll"
    job.last_status_message = "scheduler reports TIMEOUT"
    return job


def test_continuable_statuses_are_wider_than_the_reset_dismiss_set() -> None:
    # The third status set is a strict superset: COMPLETED relaxations are
    # continuable, and the reset/dismiss pair stays untouched.
    assert CONTINUABLE_JOB_STATUSES == frozenset(_CONTINUABLE)
    assert TERMINAL_RECOVERABLE_JOB_STATUSES < CONTINUABLE_JOB_STATUSES
    assert CONTINUABLE_JOB_STATUSES - TERMINAL_RECOVERABLE_JOB_STATUSES == {JobStatus.COMPLETED}


@pytest.mark.parametrize("source", _CONTINUABLE)
def test_prepare_job_continuation_accepts_every_continuable_status(
    source: JobStatus,
) -> None:
    service = ProjectService()
    job = _job(status=source)

    service.prepare_job_continuation(job)

    assert job.status == JobStatus.DEFINED


@pytest.mark.parametrize(
    "source",
    [
        JobStatus.DEFINED,
        JobStatus.SUBMITTING,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.RETRIEVING,
    ],
)
def test_prepare_job_continuation_rejects_non_continuable_states(
    source: JobStatus,
) -> None:
    service = ProjectService()
    job = _job(status=source)
    job.remote_directory = "/scratch/user/relax_row_1"

    with pytest.raises(
        ValueError,
        match="Only completed, failed, stopped, or cancelled jobs can be continued.",
    ):
        service.prepare_job_continuation(job)

    assert job.status == source
    assert job.remote_directory == "/scratch/user/relax_row_1"


@pytest.mark.parametrize("source", _CONTINUABLE)
def test_prepare_job_continuation_preserves_remote_directory(source: JobStatus) -> None:
    service = ProjectService()
    job = _job_with_run_metadata(source)

    service.prepare_job_continuation(job)

    # DEFINED + remote_directory is the combination the cluster submission path
    # reads to route the job back into its existing remote directory.
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == "/scratch/user/relax_row_1"


@pytest.mark.parametrize("source", _CONTINUABLE)
def test_prepare_job_continuation_clears_stale_run_metadata(source: JobStatus) -> None:
    service = ProjectService()
    job = _job_with_run_metadata(source)

    service.prepare_job_continuation(job)

    assert job.submitted_at is None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.remote_job_id is None
    assert job.num_processes is None
    assert job.error_message is None
    assert job.last_checked_at is None
    assert job.last_status_source is None
    assert job.last_status_message is None
    assert job.dismissed_at is None


def test_prepare_job_continuation_makes_a_dismissed_job_visible_again() -> None:
    # A dismissed failed job is filtered out of the dashboard by its stamp.
    # Continuation clears it, exactly as reset_failed_job does, so the row comes
    # back on screen for the continuation run.
    service = ProjectService()
    job = _job_with_run_metadata(JobStatus.FAILED)
    service.dismiss_job(job)
    assert job.dismissed_at is not None

    service.prepare_job_continuation(job)

    assert job.dismissed_at is None
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == "/scratch/user/relax_row_1"


def test_prepare_job_continuation_keeps_a_local_job_without_a_directory() -> None:
    service = ProjectService()
    job = _job(status=JobStatus.COMPLETED)
    job.execution_target = "local"

    service.prepare_job_continuation(job)

    assert job.status == JobStatus.DEFINED
    assert job.remote_directory is None


def test_record_job_continuation_appends_provenance(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "continue.mufinder", "Continue")
    job = _job_with_run_metadata(JobStatus.COMPLETED)
    document.jobs[job.id] = job

    service.prepare_job_continuation(job)
    service.record_job_continuation(document, job=job, remote_directory=job.remote_directory)

    record = document.provenance[-1]
    assert record.action == "job_continuation"
    assert record.actor == "user"
    assert record.job_id == job.id
    assert job.id in record.output_entity_ids
    assert record.parameters["job_label"] == job.label
    assert record.parameters["preserved_remote_directory"] is True
    assert record.parameters["remote_directory"] == "/scratch/user/relax_row_1"


def test_record_job_continuation_marks_a_missing_remote_directory(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "continue_local.mufinder", "ContinueLocal")
    job = _job(status=JobStatus.STOPPED)
    document.jobs[job.id] = job

    service.prepare_job_continuation(job)
    service.record_job_continuation(document, job=job, remote_directory=None)

    record = document.provenance[-1]
    assert record.parameters["preserved_remote_directory"] is False
    assert record.parameters["remote_directory"] is None

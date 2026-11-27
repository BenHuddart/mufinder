"""Tests for the dismiss action (job-lifecycle hardening, rec #8).

``ProjectService.dismiss_job`` hides a terminal job from the active dashboard by
stamping ``dismissed_at``; it is deliberately NOT a status transition (the job
record, runs, results, and provenance all remain). ``record_job_dismissed``
appends the matching provenance milestone.
"""

from __future__ import annotations

from mufinder.services.job_monitoring import job_status_word
from pathlib import Path

import pytest

from mufinder.domain.enums import JobActivity, JobStatus
from mufinder.services.project_service import ProjectService
from tests.support.jobs import make_job


@pytest.mark.parametrize(
    "source",
    ["failed", "stopped", "cancelled"],
)
def test_dismiss_job_stamps_dismissed_at_without_status_change(source: JobStatus) -> None:
    service = ProjectService()
    job = make_job(status=source)
    job.error_message = "prior diagnostics"

    service.dismiss_job(job)

    assert job.dismissed_at is not None
    # Status is unchanged: dismissal is not a transition.
    assert job_status_word(job) == source
    # Diagnostics are preserved for later data-level recovery.
    assert job.error_message == "prior diagnostics"


@pytest.mark.parametrize(
    "source",
    [
        JobStatus.DEFINED,
        "submitting",
        "queued",
        "running",
        "retrieving",
        JobStatus.COMPLETED,
    ],
)
def test_dismiss_job_rejects_non_terminal_states(source: JobStatus) -> None:
    service = ProjectService()
    job = make_job(status=source)
    with pytest.raises(
        ValueError, match="Only failed, stopped, or cancelled jobs can be dismissed."
    ):
        service.dismiss_job(job)
    assert job.dismissed_at is None


def test_dismiss_job_rejects_already_dismissed() -> None:
    service = ProjectService()
    job = make_job(status="failed")
    service.dismiss_job(job)
    first_stamp = job.dismissed_at

    with pytest.raises(
        ValueError, match="Only failed, stopped, or cancelled jobs can be dismissed."
    ):
        service.dismiss_job(job)
    # Idempotence rejection leaves the original stamp untouched.
    assert job.dismissed_at == first_stamp


def test_record_job_dismissed_appends_provenance(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "dismiss.mufinder", "Dismiss")
    job = make_job(status="stopped")
    document.jobs[job.id] = job

    service.dismiss_job(job)
    service.record_job_dismissed(document, job=job)

    record = document.provenance[-1]
    assert record.action == "job_dismissed"
    assert record.actor == "user"
    assert record.job_id == job.id
    assert job.id in record.output_entity_ids
    assert record.parameters["job_label"] == job.label


@pytest.mark.parametrize("leave", ["reset", "continue", "reactivate"])
def test_every_change_of_state_clears_the_dismissal(leave: str) -> None:
    """A dismissed job that is reset, continued or made active again shows on the dashboard."""

    service = ProjectService()
    job = make_job(status="failed", remote_directory="/work/job")
    service.dismiss_job(job)

    if leave == "reset":
        service.reset_failed_job(job)
    elif leave == "continue":
        service.prepare_job_continuation(job)
    else:
        service.mark_job_active(job, JobActivity.RETRIEVING)

    assert job.dismissed_at is None

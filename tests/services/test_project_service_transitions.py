"""Characterization tests for ProjectService job state-transition guards.

These pin the 4x4 transition table enforced by
``ProjectService._assert_transition`` (both allowed edges and the exact
``ValueError`` message on disallowed edges), and the primitives that move a job
along it.
"""

from __future__ import annotations

import pytest

from mufinder.domain.enums import EndReason, JobActivity, JobStatus
from mufinder.domain.models import utc_now
from mufinder.services.project_service import ProjectService
from tests.support.jobs import make_job

# The state machine as implemented in ProjectService._assert_transition. Pinned
# here so a refactor that changes an edge must consciously update this table too.
ALLOWED: dict[JobStatus, set[JobStatus]] = {
    # A defined job ends only when it is cancelled before it was submitted.
    JobStatus.DEFINED: {JobStatus.ACTIVE, JobStatus.ENDED},
    JobStatus.ACTIVE: {JobStatus.COMPLETED, JobStatus.ENDED},
    # A completed job returns to defined only for a continuation.
    JobStatus.COMPLETED: {JobStatus.DEFINED},
    # An ended job is reset, resubmitted, or found alive or with output.
    JobStatus.ENDED: {JobStatus.DEFINED, JobStatus.ACTIVE},
}

_ALL_STATUSES = tuple(JobStatus)
_ALL_EDGES = [(current, target) for current in _ALL_STATUSES for target in _ALL_STATUSES]


def _service() -> ProjectService:
    return ProjectService()


@pytest.mark.parametrize("current,target", _ALL_EDGES)
def test_assert_transition_full_matrix(current: JobStatus, target: JobStatus) -> None:
    service = _service()
    if target in ALLOWED[current]:
        service._assert_transition(current, target)
    else:
        with pytest.raises(ValueError) as excinfo:
            service._assert_transition(current, target)
        assert str(excinfo.value) == (
            f"Invalid job transition from {current.value} to {target.value}"
        )


def test_allowed_map_matches_implementation_exactly() -> None:
    service = _service()
    reconstructed: dict[JobStatus, set[JobStatus]] = {}
    for current in _ALL_STATUSES:
        allowed_targets: set[JobStatus] = set()
        for target in _ALL_STATUSES:
            try:
                service._assert_transition(current, target)
            except ValueError:
                continue
            allowed_targets.add(target)
        reconstructed[current] = allowed_targets
    assert reconstructed == ALLOWED


@pytest.mark.parametrize("source", ["defined", "failed", "stopped", "cancelled"])
def test_submit_job_makes_a_defined_or_ended_job_active(source: str) -> None:
    job = make_job(status=source, error_message="prior")

    _service().submit_job(job)

    assert (job.status, job.activity, job.end_reason) == (
        JobStatus.ACTIVE,
        JobActivity.SUBMITTING,
        None,
    )
    assert job.error_message is None
    assert job.submitted_at is not None


@pytest.mark.parametrize("source", ["submitting", "running", "completed"])
def test_submit_job_refuses_an_active_or_completed_job(source: str) -> None:
    job = make_job(status=source)

    with pytest.raises(ValueError, match="Invalid job transition"):
        _service().submit_job(job)


@pytest.mark.parametrize("source", list(JobActivity))
@pytest.mark.parametrize("target", list(JobActivity))
def test_an_active_job_may_move_to_any_activity(source: JobActivity, target: JobActivity) -> None:
    job = make_job(status=source.value)

    _service().mark_job_active(job, target, remote_job_id="4242")

    assert (job.status, job.activity, job.remote_job_id) == (JobStatus.ACTIVE, target, "4242")


def test_mark_job_active_stamps_the_start_when_the_job_starts_running() -> None:
    service = _service()
    job = make_job(status="queued")

    service.mark_job_active(job, JobActivity.RUNNING)
    started = job.started_at
    service.mark_job_active(job, JobActivity.RUNNING)

    assert started is not None
    assert job.started_at == started


@pytest.mark.parametrize("source", ["failed", "stopped", "cancelled"])
@pytest.mark.parametrize(
    "target", [JobActivity.QUEUED, JobActivity.RUNNING, JobActivity.RETRIEVING]
)
def test_an_ended_job_made_active_clears_its_end_and_keeps_remote_metadata(
    source: str, target: JobActivity
) -> None:
    service = _service()
    job = make_job(status=source, remote_job_id="4242", remote_directory="/remote/job")
    job.error_message = "missing from the scheduler"
    job.completed_at = utc_now()

    service.mark_job_active(job, target)

    assert (job.status, job.activity) == (JobStatus.ACTIVE, target)
    assert job.error_message is None
    assert job.end_reason is None
    assert job.completed_at is None
    assert (job.started_at is not None) is (target == JobActivity.RUNNING)
    assert (job.remote_job_id, job.remote_directory) == ("4242", "/remote/job")


@pytest.mark.parametrize(
    ("source", "message"),
    [("defined", "only when it is submitted"), ("completed", "Invalid job transition")],
)
def test_mark_job_active_refuses_a_defined_or_completed_job(source: str, message: str) -> None:
    job = make_job(status=source)

    with pytest.raises(ValueError, match=message):
        _service().mark_job_active(job, JobActivity.RUNNING)
    assert job.activity is None


@pytest.mark.parametrize("source", list(JobActivity))
def test_complete_job_completes_an_active_job(source: JobActivity) -> None:
    job = make_job(status=source.value, error_message="old")

    _service().complete_job(job)

    assert (job.status, job.activity, job.end_reason) == (JobStatus.COMPLETED, None, None)
    assert job.completed_at is not None
    assert job.error_message is None


@pytest.mark.parametrize("source", ["defined", "completed", "failed"])
def test_complete_job_refuses_a_job_that_is_not_active(source: str) -> None:
    with pytest.raises(ValueError, match="Invalid job transition"):
        _service().complete_job(make_job(status=source))


@pytest.mark.parametrize("reason", list(EndReason))
def test_end_job_records_the_reason_and_message(reason: EndReason) -> None:
    job = make_job(status="running")

    _service().end_job(job, reason, "what happened")

    assert (job.status, job.activity, job.end_reason) == (JobStatus.ENDED, None, reason)
    assert job.error_message == "what happened"
    assert job.completed_at is not None


def test_a_cancellation_keeps_the_jobs_message() -> None:
    job = make_job(status="running", error_message="kept")

    _service().end_job(job, EndReason.CANCELLED)

    assert job.error_message == "kept"


def test_only_a_cancellation_ends_a_defined_job() -> None:
    service = _service()
    cancelled = make_job()
    service.end_job(cancelled, EndReason.CANCELLED)
    failed = make_job()

    with pytest.raises(ValueError, match="Only a cancellation can end a defined job."):
        service.end_job(failed, EndReason.UNKNOWN, "boom")
    assert cancelled.status == JobStatus.ENDED
    assert failed.status == JobStatus.DEFINED


@pytest.mark.parametrize("source", ["completed", "failed"])
def test_end_job_refuses_a_completed_or_ended_job(source: str) -> None:
    with pytest.raises(ValueError, match="Invalid job transition"):
        _service().end_job(make_job(status=source), EndReason.UNKNOWN, "boom")


@pytest.mark.parametrize("source", ["failed", "stopped", "cancelled"])
def test_reset_failed_job_returns_an_ended_job_to_defined(source: str) -> None:
    job = make_job(status=source, error_message="prior", remote_job_id="999")

    _service().reset_failed_job(job)

    assert (job.status, job.activity, job.end_reason) == (JobStatus.DEFINED, None, None)
    assert job.error_message is None
    assert job.remote_job_id is None


@pytest.mark.parametrize("source", ["completed", "failed", "stopped", "cancelled"])
def test_prepare_job_continuation_keeps_the_remote_directory(source: str) -> None:
    job = make_job(status=source)
    job.remote_directory = "/scratch/run"

    _service().prepare_job_continuation(job)

    assert (job.status, job.end_reason) == (JobStatus.DEFINED, None)
    # The remote directory is the continuation marker and must survive.
    assert job.remote_directory == "/scratch/run"


@pytest.mark.parametrize(
    "source", ["defined", "submitting", "queued", "running", "retrieving", "completed"]
)
def test_reset_failed_job_rejects_non_resettable_states(source: str) -> None:
    job = make_job(status=source)
    with pytest.raises(ValueError, match="Only failed, stopped, or cancelled jobs can be reset."):
        _service().reset_failed_job(job)
    assert make_job(status=source).status == job.status

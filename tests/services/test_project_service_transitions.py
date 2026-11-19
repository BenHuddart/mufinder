"""Characterization tests for ProjectService job state-transition guards.

These pin the full 9x9 transition matrix enforced by
``ProjectService._assert_transition`` (both allowed edges and the exact
``ValueError`` message on disallowed edges), plus the separate
``reset_failed_job`` and ``prepare_job_continuation`` guards, which bypass
``_assert_transition`` entirely.
"""

from __future__ import annotations

import pytest

from mufinder.domain.enums import FailureReason, JobStatus
from mufinder.domain.models import utc_now
from mufinder.services.project_service import ProjectService
from tests.support.jobs import make_job

# The state machine as implemented in ProjectService._assert_transition
# (src/mufinder/services/project_service.py, ~line 2615). Pinned here so a
# refactor that changes an edge must consciously update this table too.
ALLOWED: dict[JobStatus, set[JobStatus]] = {
    JobStatus.DEFINED: {JobStatus.SUBMITTING, JobStatus.CANCELLED},
    # Milestone 6.3: SUBMITTING/QUEUED -> STOPPED added so orphaned local jobs
    # left mid-submission when MuFinder closed can be reconciled to STOPPED on
    # the next project open (JobExecutionController.check_local_jobs).
    JobStatus.SUBMITTING: {
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.STOPPED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.QUEUED: {
        JobStatus.RUNNING,
        JobStatus.STOPPED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.RUNNING: {
        JobStatus.RETRIEVING,
        JobStatus.STOPPED,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    # characterization: RETRIEVING -> CANCELLED IS allowed by the code
    # (line ~2631 lists CANCELLED). Earlier review reports disagreed; the code
    # wins and is pinned here.
    JobStatus.RETRIEVING: {
        JobStatus.STOPPED,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    # An ended job returns to QUEUED, RUNNING or RETRIEVING when evidence shows
    # it is alive or has output (ProjectService.reactivate_job).
    JobStatus.STOPPED: {
        JobStatus.SUBMITTING,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.RETRIEVING,
        JobStatus.CANCELLED,
    },
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: {
        JobStatus.SUBMITTING,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.RETRIEVING,
        JobStatus.CANCELLED,
    },
    JobStatus.CANCELLED: {
        JobStatus.SUBMITTING,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.RETRIEVING,
    },
}

_ALL_STATUSES = tuple(JobStatus)
_ALL_EDGES = [(current, target) for current in _ALL_STATUSES for target in _ALL_STATUSES]


def _service() -> ProjectService:
    return ProjectService()


@pytest.mark.parametrize("current,target", _ALL_EDGES)
def test_assert_transition_full_matrix(current: JobStatus, target: JobStatus) -> None:
    service = _service()
    if target in ALLOWED[current]:
        # Allowed edges must not raise.
        service._assert_transition(current, target)
    else:
        with pytest.raises(ValueError) as excinfo:
            service._assert_transition(current, target)
        # Pin the exact message so a refactor cannot silently reword it.
        assert str(excinfo.value) == (
            f"Invalid job transition from {current.value} to {target.value}"
        )


def test_allowed_map_matches_implementation_exactly() -> None:
    # Guard against the table above drifting from the code: reconstruct the
    # allowed set purely by probing _assert_transition.
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


# The public primitives each drive exactly one target transition. We prove a
# representative allowed edge succeeds AND a representative disallowed edge
# raises through the primitive, confirming the guard is actually invoked.
_PRIMITIVES = {
    "submit_job": (JobStatus.SUBMITTING, lambda s, j: s.submit_job(j)),
    "mark_job_queued": (JobStatus.QUEUED, lambda s, j: s.mark_job_queued(j)),
    "start_job": (JobStatus.RUNNING, lambda s, j: s.start_job(j)),
    "mark_job_retrieving": (JobStatus.RETRIEVING, lambda s, j: s.mark_job_retrieving(j)),
    "complete_job": (JobStatus.COMPLETED, lambda s, j: s.complete_job(j)),
    "stop_job": (JobStatus.STOPPED, lambda s, j: s.stop_job(j)),
    "fail_job": (JobStatus.FAILED, lambda s, j: s.fail_job(j, "boom")),
    "cancel_job": (JobStatus.CANCELLED, lambda s, j: s.cancel_job(j)),
}


@pytest.mark.parametrize("name", sorted(_PRIMITIVES))
def test_primitive_invokes_transition_guard(name: str) -> None:
    target, call = _PRIMITIVES[name]
    service = _service()

    # Find a source status that IS allowed to reach the target, and one that is not.
    allowed_source = next(src for src in _ALL_STATUSES if target in ALLOWED[src])
    disallowed_source = next(src for src in _ALL_STATUSES if target not in ALLOWED[src])

    job = make_job(status=allowed_source)
    call(service, job)
    assert job.status == target

    job = make_job(status=disallowed_source)
    with pytest.raises(ValueError, match="Invalid job transition"):
        call(service, job)
    # The rejected transition must leave the source status untouched.
    assert job.status == disallowed_source


_ENDED = [JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED]
_REACTIVATED = [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING]


@pytest.mark.parametrize("source", _ENDED)
@pytest.mark.parametrize("target", _REACTIVATED)
def test_reactivate_job_clears_the_failure_and_keeps_remote_metadata(
    source: JobStatus, target: JobStatus
) -> None:
    service = _service()
    job = make_job(status=source, remote_job_id="4242", remote_directory="/remote/job")
    job.error_message = "missing from the scheduler"
    job.failure_reason = FailureReason.OUTPUT_UNAVAILABLE
    job.completed_at = utc_now()

    service.reactivate_job(job, target)

    assert job.status == target
    assert job.error_message is None
    assert job.failure_reason is None
    assert job.completed_at is None
    assert (job.started_at is not None) is (target == JobStatus.RUNNING)
    assert (job.remote_job_id, job.remote_directory) == ("4242", "/remote/job")


@pytest.mark.parametrize("target", [JobStatus.COMPLETED, JobStatus.SUBMITTING, JobStatus.FAILED])
def test_reactivate_job_rejects_other_targets(target: JobStatus) -> None:
    job = make_job(status=JobStatus.FAILED)

    with pytest.raises(ValueError, match="cannot be reactivated"):
        _service().reactivate_job(job, target)
    assert job.status == JobStatus.FAILED


@pytest.mark.parametrize(
    "source",
    [status for status in JobStatus if status not in _ENDED],
)
def test_reactivate_job_rejects_a_job_that_has_not_ended(source: JobStatus) -> None:
    job = make_job(status=source)

    with pytest.raises(ValueError, match="Only failed, stopped, or cancelled"):
        _service().reactivate_job(job, JobStatus.RETRIEVING)
    assert job.status == source


def test_reset_failed_job_bypasses_assert_transition_and_targets_defined() -> None:
    # characterization: reset_failed_job does NOT go through _assert_transition.
    # It has its own guard set {FAILED, STOPPED, CANCELLED} -> DEFINED, and
    # DEFINED is a transition target that appears nowhere in the allowed map.
    service = _service()
    for source in (JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED):
        job = make_job(status=source)
        job.error_message = "prior"
        job.remote_job_id = "999"
        service.reset_failed_job(job)
        assert job.status == JobStatus.DEFINED
        assert job.error_message is None
        assert job.remote_job_id is None


def test_prepare_job_continuation_bypasses_assert_transition_and_targets_defined() -> None:
    # characterization: prepare_job_continuation does NOT go through
    # _assert_transition either. Its guard set {COMPLETED, FAILED, STOPPED,
    # CANCELLED} -> DEFINED includes COMPLETED, which the validated matrix gives
    # no exit at all, and DEFINED is a target that appears nowhere in it.
    service = _service()
    for source in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.STOPPED,
        JobStatus.CANCELLED,
    ):
        assert JobStatus.DEFINED not in ALLOWED[source]
        job = make_job(status=source)
        job.remote_directory = "/scratch/run"
        service.prepare_job_continuation(job)
        assert job.status == JobStatus.DEFINED
        # The remote directory is the continuation marker and must survive.
        assert job.remote_directory == "/scratch/run"


@pytest.mark.parametrize(
    "source",
    [
        JobStatus.DEFINED,
        JobStatus.SUBMITTING,
        JobStatus.QUEUED,
        JobStatus.RUNNING,
        JobStatus.RETRIEVING,
        JobStatus.COMPLETED,
    ],
)
def test_reset_failed_job_rejects_non_resettable_states(source: JobStatus) -> None:
    service = _service()
    job = make_job(status=source)
    with pytest.raises(ValueError, match="Only failed, stopped, or cancelled jobs can be reset."):
        service.reset_failed_job(job)
    assert job.status == source

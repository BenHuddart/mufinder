"""Characterization tests for ProjectService job state-transition guards.

These pin the full 9x9 transition matrix enforced by
``ProjectService._assert_transition`` (both allowed edges and the exact
``ValueError`` message on disallowed edges), plus the separate ``reset_failed_job``
guard which bypasses ``_assert_transition`` entirely.
"""

from __future__ import annotations

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.project_service import ProjectService

# The state machine as implemented in ProjectService._assert_transition
# (src/mufinder/services/project_service.py, ~line 2615). Pinned here so a
# refactor that changes an edge must consciously update this table too.
ALLOWED: dict[JobStatus, set[JobStatus]] = {
    JobStatus.DEFINED: {JobStatus.SUBMITTING, JobStatus.CANCELLED},
    # Milestone 6.3: SUBMITTING/QUEUED -> STOPPED added so orphaned local jobs
    # left mid-submission when MuFinder closed can be reconciled to STOPPED on
    # the next project open (reconcile_orphaned_local_jobs).
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
    # Job-lifecycle hardening: FAILED/STOPPED -> RETRIEVING added so the
    # dashboard "Retry retrieval" action can re-attempt output retrieval for a
    # cluster job that finished on the scheduler but whose retrieval or parsing
    # failed, without resubmitting a duplicate scheduler job.
    JobStatus.STOPPED: {
        JobStatus.SUBMITTING,
        JobStatus.RETRIEVING,
        JobStatus.CANCELLED,
    },
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: {
        JobStatus.SUBMITTING,
        JobStatus.RETRIEVING,
        JobStatus.CANCELLED,
    },
    JobStatus.CANCELLED: {JobStatus.SUBMITTING},
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
            f"Invalid job transition: {current.value} -> {target.value}"
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


def _job(status: JobStatus) -> Job:
    return Job(
        label="job",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
        status=status,
    )


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
    # retry_job_retrieval drives the same RETRIEVING target as mark_job_retrieving
    # but exists for the dashboard "Retry retrieval" action (FAILED/STOPPED ->
    # RETRIEVING). It only calls _assert_transition; the failed/stopped-only
    # precondition is enforced by JobExecutionController, not this primitive, so
    # the generic guard test exercises the shared RUNNING -> RETRIEVING edge.
    "retry_job_retrieval": (JobStatus.RETRIEVING, lambda s, j: s.retry_job_retrieval(j)),
}


@pytest.mark.parametrize("name", sorted(_PRIMITIVES))
def test_primitive_invokes_transition_guard(name: str) -> None:
    target, call = _PRIMITIVES[name]
    service = _service()

    # Find a source status that IS allowed to reach the target, and one that is not.
    allowed_source = next(src for src in _ALL_STATUSES if target in ALLOWED[src])
    disallowed_source = next(src for src in _ALL_STATUSES if target not in ALLOWED[src])

    job = _job(allowed_source)
    call(service, job)
    assert job.status == target

    job = _job(disallowed_source)
    with pytest.raises(ValueError, match="Invalid job transition"):
        call(service, job)
    # The rejected transition must leave the source status untouched.
    assert job.status == disallowed_source


def test_reset_failed_job_bypasses_assert_transition_and_targets_defined() -> None:
    # characterization: reset_failed_job does NOT go through _assert_transition.
    # It has its own guard set {FAILED, STOPPED, CANCELLED} -> DEFINED, and
    # DEFINED is a transition target that appears nowhere in the allowed map.
    service = _service()
    for source in (JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED):
        job = _job(source)
        job.error_message = "prior"
        job.remote_job_id = "999"
        service.reset_failed_job(job)
        assert job.status == JobStatus.DEFINED
        assert job.error_message is None
        assert job.remote_job_id is None


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
    job = _job(source)
    with pytest.raises(ValueError, match="Only failed, stopped, or cancelled jobs can be reset."):
        service.reset_failed_job(job)
    assert job.status == source

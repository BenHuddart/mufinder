from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from mufinder.app.workflow_submission import (
    RESUBMITTABLE_AFTER_DISMISSAL_STATUSES,
    WorkflowSubmissionCoordinator,
    effective_row_status,
    job_is_resubmittable_after_dismissal,
    workflow_row_state_is_ready,
)
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.project_service import TERMINAL_RECOVERABLE_JOB_STATUSES


@dataclass(frozen=True, slots=True)
class _State:
    status: str


def _job(label: str) -> Job:
    return Job(label=label, workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_cutoff")


def _job_with(status: JobStatus, *, dismissed: bool) -> Job:
    job = _job(f"{status.value}-{'dismissed' if dismissed else 'active'}")
    job.status = status
    if dismissed:
        job.dismissed_at = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    return job


def test_submission_plan_normalizes_dedupes_and_materializes_ready_values() -> None:
    existing = _job("existing-300")
    materialized = _job("materialized-360")
    calls: dict[str, object] = {}
    coordinator = WorkflowSubmissionCoordinator()

    def _existing_jobs(values: list[int]) -> list[Job]:
        calls["existing_values"] = list(values)
        return [existing]

    def _materialized_jobs(values: list[int]) -> list[Job]:
        calls["ready_values"] = list(values)
        return [materialized, existing]

    plan = coordinator.submission_plan(
        [300, "300", 360, 420],
        normalize=lambda value: int(value),
        build_states=lambda: {300: _State("queued"), 360: _State("ready"), 420: _State("failed")},
        jobs_for_values=_existing_jobs,
        materialize_values=_materialized_jobs,
        is_ready=workflow_row_state_is_ready,
    )

    assert plan.ordered_values == (300, 360, 420)
    assert plan.ready_values == (360,)
    assert plan.jobs == (existing, materialized)
    assert calls["existing_values"] == [300, 360, 420]
    assert calls["ready_values"] == [360]


def test_workflow_row_state_ready_accepts_defined_and_not_generated() -> None:
    states = {
        "a": _State("defined"),
        "b": _State("not generated"),
        "c": _State("ready"),
        "d": _State("running"),
    }

    assert workflow_row_state_is_ready(states, "a") is True
    assert workflow_row_state_is_ready(states, "b") is True
    assert workflow_row_state_is_ready(states, "c") is True
    assert workflow_row_state_is_ready(states, "d") is False
    assert workflow_row_state_is_ready(states, "missing") is False


def test_submission_plan_reports_empty_jobs_without_ui_dependencies() -> None:
    coordinator = WorkflowSubmissionCoordinator()

    plan = coordinator.submission_plan(
        ["alpha"],
        normalize=str,
        build_states=lambda: {"alpha": _State("ready")},
        jobs_for_values=lambda _values: [],
        materialize_values=lambda _values: [],
        is_ready=workflow_row_state_is_ready,
    )

    assert plan.has_jobs is False
    assert plan.ready_values == ("alpha",)
    assert plan.jobs == ()


# --------------------------------------------------------------------------- #
# Dismissed-terminal row projection
# --------------------------------------------------------------------------- #


def test_resubmittable_after_dismissal_matches_service_terminal_set() -> None:
    """The app-side set must track the statuses ``dismiss_job`` accepts.

    Both literals are written out independently (``services`` may not import
    ``app``), so this comparison is what keeps them from drifting: a status that
    can be dismissed must also be one this projection can recover.
    """

    assert RESUBMITTABLE_AFTER_DISMISSAL_STATUSES == TERMINAL_RECOVERABLE_JOB_STATUSES


@pytest.mark.parametrize(
    "status",
    [JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED],
)
def test_dismissed_terminal_job_projects_ready_row(status: JobStatus) -> None:
    job = _job_with(status, dismissed=True)

    assert job_is_resubmittable_after_dismissal(job) is True
    assert effective_row_status(job) == "ready"
    assert workflow_row_state_is_ready({"row": _State(effective_row_status(job))}, "row") is True
    # Dismissal is still not a status transition.
    assert job.status is status


@pytest.mark.parametrize(
    "status",
    [JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED],
)
def test_non_dismissed_terminal_job_keeps_its_own_row_status(status: JobStatus) -> None:
    job = _job_with(status, dismissed=False)

    assert job_is_resubmittable_after_dismissal(job) is False
    assert effective_row_status(job) == status.value
    assert workflow_row_state_is_ready({"row": _State(effective_row_status(job))}, "row") is False


@pytest.mark.parametrize("dismissed", [False, True])
def test_non_terminal_job_status_is_never_rewritten(dismissed: bool) -> None:
    """Only the terminal-recoverable statuses participate in the rule.

    ``dismissed_at`` cannot legitimately be set on a completed or running job,
    but the projection must not depend on that invariant holding.
    """

    completed = _job_with(JobStatus.COMPLETED, dismissed=dismissed)
    running = _job_with(JobStatus.RUNNING, dismissed=dismissed)

    assert job_is_resubmittable_after_dismissal(completed) is False
    assert effective_row_status(completed) == "completed"
    assert effective_row_status(running) == "running"


def test_defined_job_projects_defined_and_missing_job_projects_ready() -> None:
    assert effective_row_status(_job_with(JobStatus.DEFINED, dismissed=False)) == "defined"
    assert job_is_resubmittable_after_dismissal(None) is False
    assert effective_row_status(None) == "ready"
    assert effective_row_status(None, ready_status="not generated") == "not generated"

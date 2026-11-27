from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from mufinder.services.job_monitoring import job_status_word
from mufinder.app.workflow_submission import (
    RunAgainPlan,
    WorkflowSubmissionCoordinator,
    effective_row_status,
    workflow_row_state_is_ready,
    workflow_row_state_is_configurable,
)
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import Job
from tests.support.jobs import set_job_state
from tests.support.jobs import make_job


@dataclass(frozen=True, slots=True)
class _State:
    status: str


def _job_with(status: str, *, dismissed: bool) -> Job:
    job = make_job(f"{status}-{'dismissed' if dismissed else 'active'}")
    set_job_state(job, status)
    if dismissed:
        job.dismissed_at = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
    return job


def test_submission_plan_normalizes_dedupes_and_materializes_ready_values() -> None:
    existing = make_job("existing-300")
    materialized = make_job("materialized-360")
    calls: dict[str, object] = {}
    coordinator = WorkflowSubmissionCoordinator()

    def _existing_jobs(values: list[int]) -> list[Job]:
        calls["existing_values"] = list(values)
        return [existing]

    def _planned_jobs(values: list[int]) -> list[Job]:
        calls["ready_values"] = list(values)
        return [materialized, existing]

    plan = coordinator.submission_plan(
        [300, "300", 360, 420],
        normalize=lambda value: int(value),
        build_states=lambda: {
            300: _State("queued"),
            360: _State("ready"),
            420: _State("completed"),
        },
        jobs_for_values=_existing_jobs,
        plan_values=_planned_jobs,
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
        plan_values=lambda _values: [],
        is_ready=workflow_row_state_is_ready,
    )

    assert plan.has_jobs is False
    assert plan.ready_values == ("alpha",)
    assert plan.jobs == ()


# --------------------------------------------------------------------------- #
# Row projection from job state alone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dismissed", [False, True])
@pytest.mark.parametrize("status", ["failed", "stopped", "cancelled"])
def test_ended_job_row_reads_its_status_word_and_is_ready(status: str, dismissed: bool) -> None:
    """An ended row says what happened and can be submitted, dismissed or not."""

    job = _job_with(status, dismissed=dismissed)

    assert effective_row_status(job) == status == job_status_word(job)
    assert workflow_row_state_is_ready({"row": _State(effective_row_status(job))}, "row") is True


@pytest.mark.parametrize("dismissed", [False, True])
def test_active_and_completed_rows_are_not_ready(dismissed: bool) -> None:
    """The projection must not depend on ``dismissed_at`` being set only on ended jobs."""

    completed = _job_with(JobStatus.COMPLETED, dismissed=dismissed)
    running = _job_with("running", dismissed=dismissed)

    assert effective_row_status(completed) == "completed"
    assert effective_row_status(running) == "running"
    for job in (completed, running):
        assert not workflow_row_state_is_ready({"row": _State(effective_row_status(job))}, "row")


def test_defined_and_missing_jobs_both_project_ready() -> None:
    """A staged or reset job has not run, so its row reads like a row with no job."""

    assert effective_row_status(_job_with(JobStatus.DEFINED, dismissed=False)) == "ready"
    assert (
        effective_row_status(_job_with(JobStatus.DEFINED, dismissed=False), ready_status="x") == "x"
    )
    assert effective_row_status(None) == "ready"
    assert effective_row_status(None, ready_status="not generated") == "not generated"


@pytest.mark.parametrize(
    ("status", "configurable"),
    [
        ("ready", True),
        ("defined", True),
        ("not generated", True),
        ("failed", True),
        ("stopped", True),
        ("cancelled", True),
        ("running", False),
        ("completed", False),
    ],
)
def test_ended_rows_are_configurable_as_well_as_unsubmitted_ones(
    status: str, configurable: bool
) -> None:
    """An ended row can change: running it again, from its row or the dashboard, rebuilds it."""

    states = {"row": _State(status)}

    assert workflow_row_state_is_configurable(states, "row") is configurable


def test_run_again_plan_puts_back_the_kept_inputs_after_returning_the_jobs() -> None:
    continued = make_job(label="continued")
    restarted = make_job(label="restarted")
    for job in (continued, restarted):
        set_job_state(job, "failed")
    calls: list[tuple[str, list[str]]] = []
    plan = RunAgainPlan(
        continue_job_ids=frozenset({continued.id}),
        start_again_job_ids=frozenset({restarted.id}),
        apply=lambda kept, again, _cleaned: calls.append(("apply", [job.label for job in kept])),
        keep_continued_inputs=lambda jobs: calls.append(("keep", [job.label for job in jobs])),
    )

    plan.prepare([continued, restarted])

    assert calls == [("apply", ["continued"]), ("keep", ["continued"])]


def test_run_again_plan_keeps_no_inputs_when_nothing_continues() -> None:
    restarted = make_job(label="restarted")
    set_job_state(restarted, "failed")
    kept: list[object] = []
    plan = RunAgainPlan(
        start_again_job_ids=frozenset({restarted.id}), keep_continued_inputs=kept.append
    )

    plan.prepare([restarted])

    assert kept == []

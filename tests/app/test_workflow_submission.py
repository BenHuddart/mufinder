from __future__ import annotations

from dataclasses import dataclass

from mufinder.app.workflow_submission import (
    WorkflowSubmissionCoordinator,
    workflow_row_state_is_ready,
)
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Job


@dataclass(frozen=True, slots=True)
class _State:
    status: str


def _job(label: str) -> Job:
    return Job(label=label, workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF, calc_type="castep_cutoff")


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

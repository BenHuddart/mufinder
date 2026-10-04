"""The dashboard and Inspect dialog name why a job failed or stopped."""

from __future__ import annotations

from pathlib import Path

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService

_SLURM_TIMEOUT = (
    "slurmstepd: error: *** JOB 4242 ON node1 CANCELLED AT 2026-10-04T10:00:00 "
    "DUE TO TIME LIMIT ***"
)


def _window_with_job(tmp_path: Path, *, label: str, status: JobStatus):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / f"{label}.mufinder", label)
    job = service.add_job(
        doc,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = status
    window._document = doc
    return window, doc, job


def _status_item(window: MainWindow):
    window._refresh_jobs_table()
    item = window._jobs_table.item(0, window._JOB_STATUS_COLUMN)
    assert item is not None
    return item


def test_dashboard_status_names_the_failure_reason(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="timed_out", status=JobStatus.STOPPED)
    job.failure_reason = FailureReason.WALLTIME
    job.error_message = "The scheduler stopped the job when it reached its walltime limit."

    item = _status_item(window)

    assert item.text() == "stopped · walltime"
    # The summary is not repeated when it is also the recorded message; the
    # tooltip ends with what to do next.
    assert item.toolTip() == (
        "The scheduler stopped the job when it reached its walltime limit.\n\n"
        + MainWindow._WALLTIME_NEXT_STEP
    )
    window.close()


def test_stopped_job_can_be_resubmitted_from_the_dashboard(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="timed_out", status=JobStatus.STOPPED)
    job.failure_reason = FailureReason.WALLTIME
    job.execution_target = "alpha"
    job.remote_directory = "/remote/work/timed_out"

    specs = {kind: enabled for _column, _text, kind, enabled in window._job_button_specs(job)}
    assert specs["resubmit"] is True
    assert window._dashboard_action_applies("resubmit", job) is True

    calls: list[dict] = []
    monkeypatch.setattr(
        window,
        "_submit_jobs_on_cluster",
        lambda jobs, **kwargs: calls.append({"jobs": jobs, **kwargs}),
    )

    window._on_resubmit_from_dashboard([job.id])

    assert len(calls) == 1
    assert calls[0]["jobs"] == [job]
    assert JobStatus.STOPPED in calls[0]["eligible_statuses"]
    assert calls[0]["reuse_existing_remote_files"] is True
    window.close()


def test_dashboard_tooltip_adds_the_recorded_message(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="oom", status=JobStatus.FAILED)
    job.failure_reason = FailureReason.OUT_OF_MEMORY
    job.error_message = "Scheduler reported the job as OUT_OF_MEMORY."

    item = _status_item(window)

    assert item.text() == "failed · memory"
    assert item.toolTip() == (
        "The job ran out of memory and the scheduler killed it.\n\n"
        "Scheduler reported the job as OUT_OF_MEMORY."
    )
    window.close()


def test_dashboard_undiagnosed_failure_shows_its_message(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="odd", status=JobStatus.FAILED)
    job.error_message = "process exited with code 1"

    item = _status_item(window)

    assert item.text() == "failed"
    assert item.toolTip() == "process exited with code 1\n\nUse Inspect to see the job's logs."
    window.close()


def test_dashboard_active_job_has_no_failure_tooltip(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="busy", status=JobStatus.RUNNING)
    # A stale reason from an earlier run must not leak onto an active row.
    job.failure_reason = FailureReason.WALLTIME

    item = _status_item(window)

    assert item.text() == "running"
    assert item.toolTip() == ""
    window.close()


def test_dashboard_status_updates_in_place_when_the_reason_arrives(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="later", status=JobStatus.RUNNING)
    first = _status_item(window)

    job.status = JobStatus.STOPPED
    job.failure_reason = FailureReason.WALLTIME
    second = _status_item(window)

    assert second is first
    assert second.text() == "stopped · walltime"
    window.close()


def test_inspect_leads_with_the_cause_and_its_evidence(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, doc, job = _window_with_job(tmp_path, label="timed_out", status=JobStatus.STOPPED)
    job.remote_job_id = "4242"
    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm-4242.out").write_text(f"noise\n{_SLURM_TIMEOUT}\n", encoding="utf-8")

    captured: dict[str, str] = {}

    class _FakeMessageBox:
        class Icon:
            Information = 1

        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def setText(self, text: str) -> None:
            captured["text"] = text

        def setDetailedText(self, details: str) -> None:
            captured["details"] = details

        def exec(self) -> int:
            return 0

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    monkeypatch.setattr(main_window_module, "QMessageBox", _FakeMessageBox)

    window._inspect_failed_jobs([job])

    assert captured["text"] == (
        "timed_out: The scheduler stopped the job when it reached its walltime limit."
    )
    assert "Cause: The scheduler stopped the job when it reached its walltime limit." in captured["details"]
    assert f"Evidence (slurm-4242.out): {_SLURM_TIMEOUT}" in captured["details"]
    window.close()


def test_inspect_keeps_a_recorded_cause_the_logs_cannot_show(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="orphan", status=JobStatus.STOPPED)
    job.failure_reason = FailureReason.INTERRUPTED

    diagnostics = window._failure_diagnostics_for_job(job)

    assert "Cause: MuFinder lost track of the job before it finished." in diagnostics
    assert "Evidence" not in diagnostics
    window.close()


def test_inspect_headline_is_capped(qapp, tmp_path: Path) -> None:
    window, doc, first = _window_with_job(tmp_path, label="job0", status=JobStatus.FAILED)
    service = ProjectService()
    jobs = [first]
    for index in range(1, 11):
        job = service.add_job(
            doc,
            label=f"job{index}",
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = JobStatus.FAILED
        jobs.append(job)

    diagnoses = [window._job_failure_diagnosis(job) for job in jobs]
    headline = window._failure_headline(jobs, diagnoses)

    lines = headline.splitlines()
    assert len(lines) == MainWindow._FAILURE_HEADLINE_LIMIT + 1
    assert lines[-1] == "…and 3 more job(s)."
    window.close()

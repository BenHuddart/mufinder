"""The dashboard and Inspect dialog name why a job failed or stopped."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService
from tests.support.main_window import capture_failed_job_dialogs

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
    dialogs = capture_failed_job_dialogs(monkeypatch)

    window._inspect_failed_jobs([job])

    (report,) = dialogs[0]._reports
    assert report.headline == "The scheduler stopped the job when it reached its walltime limit."
    assert report.notes == (f"From slurm-4242.out: {_SLURM_TIMEOUT}",)
    assert "[slurm-4242.out]" in report.log_excerpt
    assert report.output_directory == output_dir
    window.close()


def test_inspect_headlines_the_recorded_message_when_no_cause_is_found(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="scf", status=JobStatus.FAILED)
    job.error_message = "CASTEP exited with status 1: SCF cycles did not converge"
    dialogs = capture_failed_job_dialogs(monkeypatch)

    window._inspect_failed_jobs([job])

    (report,) = dialogs[0]._reports
    assert report.headline == "CASTEP exited with status 1: SCF cycles did not converge"
    assert report.output_directory is None
    assert report.log_excerpt == ""
    assert dialogs[0].log_toggle is None
    window.close()


def test_inspect_keeps_a_recorded_cause_the_logs_cannot_show(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="orphan", status=JobStatus.STOPPED)
    job.failure_reason = FailureReason.INTERRUPTED
    dialogs = capture_failed_job_dialogs(monkeypatch)

    window._inspect_failed_jobs([job])

    (report,) = dialogs[0]._reports
    assert report.headline == "MuFinder lost track of the job before it finished."
    assert not any(note.startswith("From ") for note in report.notes)
    window.close()

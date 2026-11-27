"""The dashboard and Inspect dialog name why a job failed or stopped."""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.app.job_dashboard import JobDashboard
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import EndReason, JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService
from tests.support.jobs import set_job_state
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
    set_job_state(job, status)
    window._document = doc
    return window, doc, job


def _status_item(window: MainWindow):
    window._refresh_jobs_table()
    item = window._job_dashboard.table.item(0, window._job_dashboard.STATUS_COLUMN)
    assert item is not None
    return item


def test_dashboard_status_names_the_failure_reason(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="timed_out", status="stopped")
    job.end_reason = EndReason.WALLTIME
    job.error_message = "The scheduler stopped the job when it reached its walltime limit."

    item = _status_item(window)

    assert item.text() == "stopped · walltime"
    # The summary is not repeated when it is also the recorded message; the
    # tooltip ends with what to do next.
    assert item.toolTip() == (
        "The scheduler stopped the job when it reached its walltime limit.\n\n"
        + JobDashboard._WALLTIME_NEXT_STEP
    )
    window.close()


@pytest.mark.parametrize(
    ("stage", "next_step"),
    [
        (WorkflowStage.MUON_RELAXATION, JobDashboard._RESUMABLE_WALLTIME_NEXT_STEP),
        (WorkflowStage.PRISTINE_RELAXATION, JobDashboard._RESUMABLE_WALLTIME_NEXT_STEP),
        (WorkflowStage.CONVERGENCE_CUTOFF, JobDashboard._WALLTIME_NEXT_STEP),
    ],
)
def test_walltime_tooltip_says_resubmit_runs_the_row_again(
    qapp, tmp_path: Path, stage: WorkflowStage, next_step: str
) -> None:
    """Resubmit runs the job's row again; only a relaxation can continue from its checkpoint."""

    window, _doc, job = _window_with_job(tmp_path, label="timed_out", status="stopped")
    job.workflow_stage = stage
    job.end_reason = EndReason.WALLTIME

    assert JobDashboard.failure_tooltip(job).endswith(next_step)
    assert next_step.startswith("Resubmit to run its row again")
    window.close()


def test_dashboard_tooltip_adds_the_recorded_message(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="oom", status="failed")
    job.end_reason = EndReason.OUT_OF_MEMORY
    job.error_message = "Scheduler reported the job as OUT_OF_MEMORY."

    item = _status_item(window)

    assert item.text() == "failed · memory"
    assert item.toolTip() == (
        "The job ran out of memory and the scheduler killed it.\n\n"
        "Scheduler reported the job as OUT_OF_MEMORY."
    )
    window.close()


def test_dashboard_undiagnosed_failure_shows_its_message(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="odd", status="failed")
    job.error_message = "process exited with code 1"

    item = _status_item(window)

    assert item.text() == "failed · process exited with code 1"
    assert item.toolTip() == "process exited with code 1\n\nUse Inspect to see the job's logs."
    window.close()


def test_dashboard_active_job_has_no_failure_tooltip(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="busy", status="running")
    # A stale reason from an earlier run must not leak onto an active row.
    job.end_reason = EndReason.WALLTIME

    item = _status_item(window)

    assert item.text() == "running"
    assert item.toolTip() == ""
    window.close()


def test_dashboard_status_updates_in_place_when_the_reason_arrives(qapp, tmp_path: Path) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="later", status="running")
    first = _status_item(window)

    job.status = JobStatus.ENDED
    job.end_reason = EndReason.WALLTIME
    second = _status_item(window)

    assert second is first
    assert second.text() == "stopped · walltime"
    window.close()


def test_inspect_leads_with_the_cause_and_its_evidence(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, doc, job = _window_with_job(tmp_path, label="timed_out", status="stopped")
    job.remote_job_id = "4242"
    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm-4242.out").write_text(f"noise\n{_SLURM_TIMEOUT}\n", encoding="utf-8")
    dialogs = capture_failed_job_dialogs(monkeypatch)

    window._jobs.inspect_failed_jobs([job])

    (report,) = dialogs[0]._reports
    assert report.headline == "The scheduler stopped the job when it reached its walltime limit."
    assert report.notes == (f"From slurm-4242.out: {_SLURM_TIMEOUT}",)
    assert "[slurm-4242.out]" in report.log_excerpt
    assert report.output_directory == output_dir
    window.close()


def test_inspect_headlines_the_recorded_message_when_no_cause_is_found(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="scf", status="failed")
    job.error_message = "CASTEP exited with status 1: SCF cycles did not converge"
    dialogs = capture_failed_job_dialogs(monkeypatch)

    window._jobs.inspect_failed_jobs([job])

    (report,) = dialogs[0]._reports
    assert report.headline == "CASTEP exited with status 1: SCF cycles did not converge"
    assert report.output_directory is None
    assert report.log_excerpt == ""
    assert dialogs[0].log_toggle is None
    window.close()


def test_inspect_keeps_a_recorded_cause_the_logs_cannot_show(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, job = _window_with_job(tmp_path, label="orphan", status="stopped")
    job.end_reason = EndReason.INTERRUPTED
    dialogs = capture_failed_job_dialogs(monkeypatch)

    window._jobs.inspect_failed_jobs([job])

    (report,) = dialogs[0]._reports
    assert report.headline == "MuFinder lost track of the job before it finished."
    assert not any(note.startswith("From ") for note in report.notes)
    window.close()

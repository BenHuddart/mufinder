"""The main window's job check: on project open and once a minute."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest

from mufinder.services.job_monitoring import job_status_word
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun
from mufinder.services.project_service import ProjectService


def _running_local_job(window: MainWindow, service: ProjectService, document) -> object:
    job = service.add_job(
        document,
        label="stuck-local",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
    )
    # Drive it to RUNNING through the primitives, exactly as a real submission
    # would, then register an active run so close-out has something to touch.
    service.submit_job(job)
    service.mark_job_active(job, JobActivity.RUNNING)
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.ACTIVE)
    document.runs[run.id] = run
    window._active_run_by_job[job.id] = run.id
    return job


def test_job_check_stops_a_local_job_with_no_process(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "watchdog_wiring.mufinder", "Watchdog Wiring")
    window._document = document
    window._refresh_ui_after_project_change()

    job = _running_local_job(window, service, document)
    # The local runner holds no process for it and its folder has no output.

    window._jobs.check_jobs()

    stopped = document.jobs[job.id]
    assert job_status_word(stopped) == "stopped"
    assert stopped.end_reason == EndReason.INTERRUPTED
    assert stopped.last_status_source == "reconciliation"
    assert stopped.last_status_message
    # The active run was closed out and unmapped, mirroring the walltime-stop path.
    assert job.id not in window._active_run_by_job
    stopped_runs = [run for run in document.runs.values() if run.job_id == job.id]
    assert stopped_runs and all(run.status == JobStatus.ENDED for run in stopped_runs)
    assert "1 local job was stopped" in window.statusBar().currentMessage()


def test_job_check_leaves_a_local_job_the_runner_holds(
    main_window, tmp_path: Path, monkeypatch
) -> None:
    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "watchdog_held.mufinder", "Watchdog Held")
    window._document = document
    window._refresh_ui_after_project_change()
    job = _running_local_job(window, service, document)
    monkeypatch.setattr(window._local_runner, "active_job_ids", lambda: frozenset({job.id}))

    window._jobs.check_jobs()

    assert job_status_word(document.jobs[job.id]) == "running"


def test_job_check_noop_without_active_jobs(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "watchdog_empty.mufinder", "Watchdog Empty")
    window._document = document
    window._refresh_ui_after_project_change()

    # No active jobs: the handler returns early and does not raise.
    window._jobs.check_jobs()
    window._jobs.check_jobs()


def test_job_check_leaves_an_untracked_cluster_job_active(main_window, tmp_path: Path) -> None:
    # The cluster is connected and nothing tracks the job. It may be queued,
    # running or finished there, so MuFinder does not end it.
    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "watchdog_cluster.mufinder", "Watchdog Cluster")
    window._document = document
    window._refresh_ui_after_project_change()
    job = service.add_job(
        document,
        label="untracked-cluster",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
    )
    job.execution_target = "SCARF"
    service.submit_job(job)
    service.mark_job_active(job, JobActivity.RUNNING)
    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "SCARF"

    window._jobs.check_jobs()
    window._jobs.check_jobs()

    assert job_status_word(document.jobs[job.id]) == "running"
    assert document.jobs[job.id].end_reason is None


_COMPLETE_OUTPUT = Path(__file__).parents[1] / "fixtures" / "uep" / "Si2.castep"


def _interrupted_local_job(window: MainWindow, service: ProjectService, document, folder: Path):
    job = _running_local_job(window, service, document)
    window._jobs.check_jobs()  # MuFinder has no process for it: it is stopped.
    assert job.end_reason == EndReason.INTERRUPTED
    folder.mkdir(exist_ok=True)
    return job


@pytest.fixture
def local_window(main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "outlived.mufinder", "Outlived")
    window._document = document
    window._refresh_ui_after_project_change()
    folder = tmp_path / "stuck-local"
    monkeypatch.setattr(window._job_execution, "_job_output_directory", lambda _job: folder)
    monkeypatch.setattr(window._job_execution, "_job_seed_name", lambda job: job.label)
    return window, service, document, folder


def test_opening_a_project_completes_a_local_job_whose_process_outlived_mufinder(
    local_window,
) -> None:
    window, service, document, folder = local_window
    job = _interrupted_local_job(window, service, document, folder)
    # CASTEP kept running after MuFinder closed and has now finished.
    (folder / "stuck-local.castep").write_bytes(_COMPLETE_OUTPUT.read_bytes())

    window._jobs.check_jobs()
    assert job_status_word(job) == "stopped"

    window._refresh_ui_after_project_change()

    assert job.status == JobStatus.COMPLETED
    run = next(run for run in document.runs.values() if run.job_id == job.id and run.result_id)
    assert run.result_id in document.results


def test_opening_a_project_leaves_an_interrupted_local_job_with_an_earlier_runs_output(
    local_window,
) -> None:
    window, service, document, folder = local_window
    job = _interrupted_local_job(window, service, document, folder)
    output = folder / "stuck-local.castep"
    output.write_bytes(_COMPLETE_OUTPUT.read_bytes())
    earlier = (job.submitted_at - timedelta(hours=1)).timestamp()
    os.utime(output, (earlier, earlier))

    window._refresh_ui_after_project_change()

    assert job_status_word(job) == "stopped"
    assert job.end_reason == EndReason.INTERRUPTED

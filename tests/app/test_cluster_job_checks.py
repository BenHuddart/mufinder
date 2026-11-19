"""The main window asks the cluster about its jobs and applies what comes back.

Which jobs a check covers is decided in ``JobExecutionController`` and tested
there. These tests pin when the main window asks (project open, connect, the
once-a-minute tick, the queue button) and that evidence reaching it changes the
job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.app.main_window import MainWindow
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.domain.models import Job, utc_now
from mufinder.services.cluster_runner import QueueCheckSnapshot
from mufinder.services.job_reconcile import JobEvidence, OutputState, SchedulerState
from mufinder.services.project_service import ProjectService

_REMOTE_DIR = "/remote/work/job"


def _add_cluster_job(
    document,
    service: ProjectService,
    *,
    label: str,
    status: JobStatus,
    execution_target: str = "SCARF",
    remote_job_id: str | None = "12345",
) -> Job:
    job = service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
        execution_target=execution_target,
    )
    job.status = status
    job.remote_job_id = remote_job_id
    job.remote_directory = _REMOTE_DIR
    return job


@pytest.fixture
def checking(main_window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Return ``(service, document, window, checks)`` connected to SCARF.

    ``checks`` records each request the window sends to the cluster worker as
    ``(labels, skip_if_busy)``; nothing reaches a real cluster.
    """

    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "checks.mufinder", "Checks")
    window._document = document
    checks: list[tuple[list[str], bool]] = []

    def record(targets, *, skip_if_busy: bool = False) -> bool:
        checks.append(([target.seed_name for target in targets], skip_if_busy))
        return True

    monkeypatch.setattr(window._cluster_runner, "check_jobs", record)
    execution = window._job_execution
    monkeypatch.setattr(execution, "_job_output_directory", lambda job: tmp_path / job.label)
    monkeypatch.setattr(execution, "_job_seed_name", lambda job: job.label)
    window._cluster_runner._connected = True
    window._cluster_runner._connected_cluster_name = "SCARF"
    return service, document, window, checks


def _evidence(job: Job, scheduler: SchedulerState, output=OutputState.NOT_LOOKED, **fields):
    return JobEvidence(
        checked_at=utc_now(),
        scheduler=scheduler,
        output=output,
        remote_job_id=job.remote_job_id or "",
        remote_directory=job.remote_directory or "",
        **fields,
    )


def test_opening_a_project_asks_about_its_active_cluster_jobs(checking) -> None:
    service, document, window, checks = checking
    _add_cluster_job(document, service, label="queued", status=JobStatus.QUEUED)
    _add_cluster_job(
        document, service, label="no_id", status=JobStatus.SUBMITTING, remote_job_id=None
    )
    _add_cluster_job(document, service, label="failed", status=JobStatus.FAILED)
    _add_cluster_job(
        document, service, label="elsewhere", status=JobStatus.RUNNING, execution_target="PSI"
    )

    window._refresh_ui_after_project_change()

    # The check that opens a project gives way to one already on the worker. A
    # job with no scheduler job ID is looked at only when the user asks.
    assert checks == [(["queued"], True)]

    no_id = next(job for job in document.jobs.values() if job.label == "no_id")
    window._on_check_now_from_dashboard([no_id.id])
    assert checks[-1] == (["no_id"], False)


def test_connecting_asks_about_the_clusters_jobs(checking) -> None:
    service, document, window, checks = checking
    _add_cluster_job(document, service, label="running", status=JobStatus.RUNNING)

    window._on_cluster_connection_status_changed(True)

    assert checks == [(["running"], True)]


def test_disconnecting_asks_nothing(checking) -> None:
    service, document, window, checks = checking
    _add_cluster_job(document, service, label="running", status=JobStatus.RUNNING)
    window._cluster_runner._connected = False

    window._on_cluster_connection_status_changed(False)
    window._on_job_check_tick()

    assert checks == []


def test_the_minute_tick_skips_a_check_while_one_is_running(checking) -> None:
    service, document, window, checks = checking
    _add_cluster_job(document, service, label="running", status=JobStatus.RUNNING)

    window._on_job_check_tick()

    assert checks == [(["running"], True)]


def test_the_queue_button_asks_and_reports_the_pass(checking) -> None:
    service, document, window, checks = checking
    job = _add_cluster_job(document, service, label="running", status=JobStatus.RUNNING)
    window._refresh_jobs_table()
    window._sync_cluster_status_widgets()

    window._on_refresh_cluster_queue()

    # A pass already on the worker answers the request, so it is not queued twice.
    assert checks == [(["running"], True)]
    window._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(job.id,))
    )
    assert window.statusBar().currentMessage().startswith("Refreshed queue: checked 1 job at ")


def test_a_pass_over_another_projects_jobs_is_ignored(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="running", status=JobStatus.RUNNING)
    other = Job(label="other")
    before = window._cluster_queue_checked_label.text()

    window._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(other.id,))
    )
    assert window._cluster_queue_checked_label.text() == before

    window._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(job.id,))
    )
    assert window._cluster_queue_checked_label.text() != before


def test_evidence_moves_the_job_and_says_so(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="queued", status=JobStatus.QUEUED)

    window._on_cluster_job_checked(job.id, _evidence(job, SchedulerState.RUNNING))

    assert job.status == JobStatus.RUNNING
    assert window._dashboard_flush_pending
    # The status bar speaks once, when the check ends.
    assert window.statusBar().currentMessage() != "Cluster job started running: queued"
    window._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(job.id,))
    )
    assert window.statusBar().currentMessage() == "Cluster job started running: queued"


def test_evidence_of_complete_output_records_the_result(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="done", status=JobStatus.RUNNING)
    evidence = _evidence(
        job,
        SchedulerState.COMPLETED,
        OutputState.COMPLETE,
        parsed=CastepResult(completed=True, total_energy=-12.5),
    )

    window._on_cluster_job_checked(job.id, evidence)

    assert job.status == JobStatus.COMPLETED
    run = next(run for run in document.runs.values() if run.job_id == job.id)
    assert document.results[run.result_id].data["total_energy"] == -12.5
    assert window.statusBar().currentMessage() == "Completed cluster job done"


def test_a_job_the_scheduler_lost_stays_active_with_a_lost_track_marker(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="lost", status=JobStatus.RUNNING)

    window._on_cluster_job_checked(
        job.id, _evidence(job, SchedulerState.NOT_LISTED, OutputState.ABSENT)
    )

    assert job.status == JobStatus.RUNNING
    assert job.failure_reason is None
    marker = window._job_execution.staleness(job, now=utc_now())
    assert marker is not None and marker.startswith("lost track")


def test_a_scheduler_failure_fails_the_job(checking) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="oom", status=JobStatus.RUNNING)

    window._on_cluster_job_checked(
        job.id,
        _evidence(
            job, SchedulerState.FAILED, OutputState.INCOMPLETE, scheduler_detail="OUT_OF_MEMORY"
        ),
    )

    assert job.status == JobStatus.FAILED
    assert job.failure_reason == FailureReason.OUT_OF_MEMORY
    window._on_cluster_queue_checked(
        QueueCheckSnapshot(checked_at=utc_now(), elapsed_by_job_id={}, job_ids=(job.id,))
    )
    assert window.statusBar().currentMessage() == "Cluster job failed (memory): oom"


def test_evidence_with_no_open_project_is_ignored(main_window: MainWindow) -> None:
    job = Job(label="gone", execution_target="SCARF", status=JobStatus.RUNNING)

    main_window._on_cluster_job_checked(job.id, _evidence(job, SchedulerState.RUNNING))


def test_a_check_that_changes_several_jobs_ends_with_one_summary(checking) -> None:
    service, document, window, _checks = checking
    done = _add_cluster_job(document, service, label="done", status=JobStatus.RUNNING)
    timed_out = _add_cluster_job(document, service, label="late", status=JobStatus.QUEUED)
    same = _add_cluster_job(document, service, label="same", status=JobStatus.RUNNING)
    window._refresh_jobs_table()
    window._sync_cluster_status_widgets()
    window._on_refresh_cluster_queue()

    window._on_cluster_job_checked(
        done.id,
        _evidence(
            done,
            SchedulerState.COMPLETED,
            OutputState.COMPLETE,
            parsed=CastepResult(completed=True, total_energy=-1.0),
        ),
    )
    window._on_cluster_job_checked(
        timed_out.id, _evidence(timed_out, SchedulerState.TIMED_OUT, OutputState.ABSENT)
    )
    window._on_cluster_job_checked(same.id, _evidence(same, SchedulerState.RUNNING))
    window._on_cluster_queue_checked(
        QueueCheckSnapshot(
            checked_at=utc_now(),
            elapsed_by_job_id={},
            job_ids=(done.id, timed_out.id, same.id),
        )
    )

    assert window.statusBar().currentMessage() == (
        "Checked 3 cluster jobs: 1 completed, 1 stopped at its walltime limit"
    )


def test_a_failed_check_discards_what_it_had_changed(checking, monkeypatch) -> None:
    service, document, window, _checks = checking
    job = _add_cluster_job(document, service, label="queued", status=JobStatus.QUEUED)
    monkeypatch.setattr(
        "mufinder.app.main_window.show_detailed_warning", lambda *args, **kwargs: None
    )
    window._on_cluster_job_checked(job.id, _evidence(job, SchedulerState.RUNNING))

    window._on_cluster_poll_failed("Queue polling failed for SCARF: boom")

    assert window._job_execution.take_check_summary(1) == ""

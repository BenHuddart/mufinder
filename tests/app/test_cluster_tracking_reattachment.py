"""Characterization tests for cluster-tracking reattachment on project load.

``MainWindow._sync_cluster_tracking_from_document`` re-registers persisted
cluster jobs with the cluster runner when a project is loaded. These tests pin:

- which job statuses are re-registered,
- that local jobs are NOT reattached (current behavior),
- and the remote-id / remote-dir gating.

The full main window is heavy; following the pattern in
``tests/app/main_window/`` we build a ``MainWindow`` instance,
attach a document directly, record the cluster runner's ``track_existing_job``
calls, and stub the per-job output-directory / seed-name helpers so the test
isolates the status-and-target gating that this method owns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job, utc_now
from mufinder.services.failure_diagnosis import MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE
from mufinder.services.project_service import ProjectService


def _add_cluster_job(
    document,
    service: ProjectService,
    *,
    label: str,
    status: JobStatus,
    execution_target: str = "SCARF",
    remote_job_id: str | None = "12345",
    remote_directory: str | None = "/remote/work/job",
    calc_type: str = "castep_cutoff",
) -> Job:
    job = service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type=calc_type,
        execution_target=execution_target,
    )
    job.status = status
    job.remote_job_id = remote_job_id
    job.remote_directory = remote_directory
    return job


def _prepare_window(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service = ProjectService()
    document = service.create_project(tmp_path / "reattach.mufinder", "Reattach")
    window = MainWindow()
    window._document = document
    # Keep the real cluster runner (its Qt signal wiring is needed for a clean
    # window.close()); record only its track_existing_job calls.
    tracked: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner,
        "track_existing_job",
        lambda **kwargs: tracked.append(kwargs),
    )
    # Isolate the status/target gating: give every job a resolvable output dir
    # and seed so the method reaches (or skips) track_existing_job based only on
    # status, execution target, and remote ids.
    monkeypatch.setattr(window, "_job_output_directory", lambda job: tmp_path / "out" / job.label)
    monkeypatch.setattr(window, "_job_seed_name", lambda job: job.label)
    return service, document, window, tracked


def test_active_cluster_statuses_are_reattached(qapp, tmp_path, monkeypatch) -> None:
    service, document, window, tracked = _prepare_window(qapp, tmp_path, monkeypatch)

    # characterization: _ACTIVE_LOCAL_JOB_STATES drives reattachment and it
    # includes SUBMITTING in addition to queued/running/retrieving. (The task
    # spec listed only queued/running/retrieving; SUBMITTING is also re-registered.)
    for status in (JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING):
        _add_cluster_job(document, service, label=f"active_{status.value}", status=status)

    window._sync_cluster_tracking_from_document()

    tracked_labels = {str(entry["seed_name"]) for entry in tracked}
    assert tracked_labels == {
        "active_submitting",
        "active_queued",
        "active_running",
        "active_retrieving",
    }
    window.close()


@pytest.mark.parametrize(
    "status",
    [
        JobStatus.DEFINED,
        JobStatus.STOPPED,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    ],
)
def test_inactive_cluster_statuses_are_ignored(qapp, tmp_path, monkeypatch, status) -> None:
    service, document, window, tracked = _prepare_window(qapp, tmp_path, monkeypatch)
    _add_cluster_job(document, service, label="inactive", status=status)

    window._sync_cluster_tracking_from_document()

    assert tracked == []
    window.close()


def test_local_jobs_are_not_reattached(qapp, tmp_path, monkeypatch) -> None:
    # characterization: local jobs are intentionally NOT reattached to the
    # cluster runner even when active.
    service, document, window, tracked = _prepare_window(qapp, tmp_path, monkeypatch)
    _add_cluster_job(
        document,
        service,
        label="local_running",
        status=JobStatus.RUNNING,
        execution_target="local",
    )
    # An empty execution target is treated the same as local.
    _add_cluster_job(
        document,
        service,
        label="blank_target_running",
        status=JobStatus.RUNNING,
        execution_target="",
    )

    window._sync_cluster_tracking_from_document()

    assert tracked == []
    window.close()


def test_active_cluster_job_without_remote_id_is_skipped(qapp, tmp_path, monkeypatch) -> None:
    service, document, window, tracked = _prepare_window(qapp, tmp_path, monkeypatch)
    _add_cluster_job(
        document,
        service,
        label="no_remote_id",
        status=JobStatus.RUNNING,
        remote_job_id=None,
    )

    window._sync_cluster_tracking_from_document()

    assert tracked == []
    window.close()


def test_reattached_entry_carries_status_and_cluster_name(qapp, tmp_path, monkeypatch) -> None:
    service, document, window, tracked = _prepare_window(qapp, tmp_path, monkeypatch)
    job = _add_cluster_job(
        document,
        service,
        label="running_job",
        status=JobStatus.RUNNING,
        execution_target="SCARF",
        remote_job_id="98765",
        remote_directory="/remote/work/running_job",
    )

    window._sync_cluster_tracking_from_document()

    assert len(tracked) == 1
    entry = tracked[0]
    assert entry["job_id"] == job.id
    assert entry["remote_job_id"] == "98765"
    assert entry["remote_dir"] == "/remote/work/running_job"
    assert entry["cluster_name"] == "SCARF"
    assert entry["status"] == JobStatus.RUNNING
    window.close()


def test_jobs_failed_for_a_missing_scheduler_entry_are_watched(qapp, tmp_path, monkeypatch) -> None:
    # A job failed because neither the queue nor accounting listed it may still
    # be waiting in a hidden queue, so it is registered to be watched.
    service, document, window, tracked = _prepare_window(qapp, tmp_path, monkeypatch)
    missing = _add_cluster_job(document, service, label="missing", status=JobStatus.FAILED)
    missing.error_message = MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE
    dismissed = _add_cluster_job(document, service, label="dismissed", status=JobStatus.FAILED)
    dismissed.error_message = MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE
    dismissed.dismissed_at = utc_now()
    other = _add_cluster_job(document, service, label="other", status=JobStatus.FAILED)
    other.error_message = "Scheduler reported the job as FAILED."
    _add_cluster_job(document, service, label="running", status=JobStatus.RUNNING)

    window._sync_cluster_tracking_from_document()

    by_label = {str(entry["seed_name"]): entry for entry in tracked}
    assert set(by_label) == {"missing", "running"}
    assert by_label["missing"]["status"] == JobStatus.FAILED
    assert by_label["missing"]["watch_for_recovery"] is True
    assert by_label["running"]["watch_for_recovery"] is False
    window.close()


def test_refresh_reconnects_clusters_of_jobs_awaiting_reappearance(qapp, tmp_path, monkeypatch) -> None:
    # Refresh Queue while disconnected reconnects the clusters it must poll,
    # including one whose only job is a failed job awaiting reappearance.
    service, document, window, _tracked = _prepare_window(qapp, tmp_path, monkeypatch)
    watched = _add_cluster_job(
        document, service, label="watched", status=JobStatus.FAILED, execution_target="PSI"
    )
    watched.error_message = MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE
    other = _add_cluster_job(
        document, service, label="other", status=JobStatus.FAILED, execution_target="ARCHER"
    )
    other.error_message = "Scheduler reported the job as FAILED."
    _add_cluster_job(document, service, label="running", status=JobStatus.RUNNING)

    assert window._active_dashboard_cluster_targets() == ["PSI", "SCARF"]
    window.close()

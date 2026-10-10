"""Load-path wiring tests for orphaned-local-job reconciliation (Milestone 6.3).

These prove the end-to-end behavior through the real project-open flow
(``MainWindow._open_project_from_path``):

- local jobs left active when MuFinder closed are stopped on open,
- the reconciliation runs BEFORE the dashboard/jobs table is rendered so the
  user never sees phantom running rows,
- the status-bar message is surfaced (and not clobbered by the generic
  "Opened project ..." message that follows it in the real flow),
- cluster jobs are not stopped by this path.
"""

from __future__ import annotations

from pathlib import Path


from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun
from mufinder.services.project_service import (
    ORPHANED_LOCAL_JOB_STOP_MESSAGE,
    ProjectService,
)


def _seed_project_with_jobs(tmp_path: Path):
    """Create a persisted project containing an orphaned local + a cluster job."""

    service = ProjectService()
    root = tmp_path / "project"
    document = service.create_project(root, "Orphan")

    local_job = service.add_job(
        document,
        label="local_running",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
        execution_target="local",
    )
    local_job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=local_job.id, run_index=1, status=JobStatus.RUNNING)
    document.runs[run.id] = run

    cluster_job = service.add_job(
        document,
        label="cluster_running",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
        execution_target="SCARF",
    )
    cluster_job.status = JobStatus.RUNNING
    cluster_job.remote_job_id = "12345"
    cluster_job.remote_directory = "/remote/work/cluster_running"

    service.save_project(document)
    return root, local_job.id, cluster_job.id


def test_open_project_stops_orphaned_local_job_and_shows_message(qapp, tmp_path, monkeypatch) -> None:
    root, local_id, cluster_id = _seed_project_with_jobs(tmp_path)

    window = MainWindow()
    # Isolate cluster reattachment side-effects; the real cluster runner would
    # try to talk to SCARF otherwise.
    monkeypatch.setattr(window._cluster_runner, "track_existing_job", lambda **kwargs: None)
    monkeypatch.setattr(window, "_job_output_directory", lambda job: tmp_path / "out" / job.label)
    monkeypatch.setattr(window, "_job_seed_name", lambda job: job.label)

    assert window._open_project_from_path(root) is True

    document = window._document
    local_job = document.jobs[local_id]
    cluster_job = document.jobs[cluster_id]

    # Local job reconciled to STOPPED with the exact message; its run closed.
    assert local_job.status == JobStatus.STOPPED
    assert local_job.error_message == ORPHANED_LOCAL_JOB_STOP_MESSAGE
    latest_run = next(r for r in document.runs.values() if r.job_id == local_id)
    assert latest_run.status == JobStatus.STOPPED

    # Cluster job untouched by reconciliation.
    assert cluster_job.status == JobStatus.RUNNING

    # Status-bar message surfaced (past the generic "Opened project" message).
    message = window.statusBar().currentMessage()
    assert "local job(s) were stopped" in message
    assert "1 " in message

    window.close()


def test_reopen_does_not_re_reconcile_already_stopped_job(qapp, tmp_path, monkeypatch) -> None:
    """Reconciliation must be idempotent across reopens.

    STOPPED is not an active status, so the first open reconciles the orphaned
    local job and the flush persists it; the second open finds it already
    STOPPED and leaves it alone, appending no duplicate ``job_stopped`` record.
    """

    root, local_id, _ = _seed_project_with_jobs(tmp_path)

    window = MainWindow()
    monkeypatch.setattr(window._cluster_runner, "track_existing_job", lambda **kwargs: None)
    monkeypatch.setattr(window, "_job_output_directory", lambda job: tmp_path / "out" / job.label)
    monkeypatch.setattr(window, "_job_seed_name", lambda job: job.label)

    assert window._open_project_from_path(root) is True
    assert window._document.jobs[local_id].status == JobStatus.STOPPED
    # Force the coalesced flush to persist the reconciled status to disk.
    window._flush_dashboard()

    # Reopen the same project; the stopped job must not be reconciled again.
    assert window._open_project_from_path(root) is True
    assert window._reconciled_orphaned_local_jobs == []
    assert window._document.jobs[local_id].status == JobStatus.STOPPED
    stopped_records = [
        record
        for record in window._document.provenance
        if record.action == "job_stopped" and record.job_id == local_id
    ]
    assert len(stopped_records) == 1

    window.close()


def test_reconciliation_runs_before_jobs_table_render(qapp, tmp_path, monkeypatch) -> None:
    root, local_id, _ = _seed_project_with_jobs(tmp_path)

    window = MainWindow()
    monkeypatch.setattr(window._cluster_runner, "track_existing_job", lambda **kwargs: None)
    monkeypatch.setattr(window, "_job_output_directory", lambda job: tmp_path / "out" / job.label)
    monkeypatch.setattr(window, "_job_seed_name", lambda job: job.label)

    # Capture the local job's status at the moment the jobs table is rendered.
    status_at_render: list[JobStatus] = []
    original_refresh = window._refresh_jobs_table

    def _spy_refresh():
        job = window._document.jobs.get(local_id)
        if job is not None:
            status_at_render.append(job.status)
        return original_refresh()

    monkeypatch.setattr(window, "_refresh_jobs_table", _spy_refresh)

    assert window._open_project_from_path(root) is True

    # The jobs table was rendered at least once and, every time, the orphaned
    # local job was already STOPPED — never a phantom RUNNING row.
    assert status_at_render, "jobs table was never rendered on open"
    assert all(status == JobStatus.STOPPED for status in status_at_render)

    window.close()


def test_stopped_local_muon_job_stays_linked_and_resettable(qapp, tmp_path, monkeypatch) -> None:
    """A reconciled local muon-relaxation job must remain visible on its row.

    Real submission persists the job into the definition's ``job_ids_by_label``
    mapping. Reconciliation only changes the job's status to STOPPED, so the
    row->job resolution (which does not filter by status) still finds it, and a
    stopped job is a resettable/visible dashboard state. This is the check that
    a stopped local job does NOT silently vanish from its muon row on reopen.
    """

    from mufinder.app import main_window as main_window_module
    from mufinder.domain.models import MuonSite, Structure

    service = ProjectService()
    root = tmp_path / "muon"
    document = service.create_project(root, "Muon")

    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    site = MuonSite(label="Site 1", fractional_position=[0.25, 0.25, 0.25])
    document.sites[site.id] = site

    calc_type = main_window_module.muon_relaxation_calc_type("positive_muon")
    job = service.add_job(
        document,
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=calc_type,
        execution_target="local",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    document.runs[run.id] = run

    # Mirror real submission: persist the row->job mapping on the definition.
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters={},
        output_directory=root / "files" / "muon_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}
    document.ui_state[MainWindow._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    service.save_project(document)

    reloaded = service.load_project(root)
    reconciled = service.reconcile_orphaned_local_jobs(reloaded)
    assert [j.id for j in reconciled] == [job.id]

    reloaded_definition = reloaded.calculations[definition.id]
    resolved = MainWindow._muon_relaxation_job_for_table_row(
        reloaded, reloaded_definition, "Site 1", "positive_muon"
    )
    assert resolved is not None
    assert resolved.status == JobStatus.STOPPED
    # STOPPED is a resettable state, so the job can be resubmitted from the row.
    service.reset_failed_job(resolved)
    assert resolved.status == JobStatus.DEFINED


def test_open_project_without_orphans_shows_opened_message(qapp, tmp_path, monkeypatch) -> None:
    service = ProjectService()
    root = tmp_path / "clean"
    document = service.create_project(root, "Clean")
    job = service.add_job(
        document,
        label="done",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
        execution_target="local",
    )
    job.status = JobStatus.COMPLETED
    service.save_project(document)

    window = MainWindow()
    monkeypatch.setattr(window._cluster_runner, "track_existing_job", lambda **kwargs: None)
    monkeypatch.setattr(window, "_job_output_directory", lambda job: tmp_path / "out" / job.label)
    monkeypatch.setattr(window, "_job_seed_name", lambda job: job.label)

    assert window._open_project_from_path(root) is True

    message = window.statusBar().currentMessage()
    assert message == "Opened project Clean"
    assert window._reconciled_orphaned_local_jobs == []

    window.close()

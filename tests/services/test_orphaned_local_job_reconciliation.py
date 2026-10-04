"""Tests for ProjectService.reconcile_orphaned_local_jobs (Milestone 6.3).

A local job left in an active state (submitting/queued/running/retrieving) when
MuFinder closes has no live process behind it on the next launch. On project
open these jobs are reconciled to STOPPED with a fixed message and their latest
in-flight run is closed out. Cluster jobs are re-attached elsewhere and must be
left untouched by this reconciliation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun
from mufinder.services.project_service import (
    ORPHANED_LOCAL_JOB_STOP_MESSAGE,
    ProjectService,
)

_ACTIVE_STATUSES = (
    JobStatus.SUBMITTING,
    JobStatus.QUEUED,
    JobStatus.RUNNING,
    JobStatus.RETRIEVING,
)

_TERMINAL_OR_DEFINED = (
    JobStatus.DEFINED,
    JobStatus.STOPPED,
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
)


def _service_and_document(tmp_path: Path):
    service = ProjectService()
    document = service.create_project(tmp_path / "recon.mufinder", "Recon")
    return service, document


def _add_job(
    service: ProjectService,
    document,
    *,
    label: str,
    status: JobStatus,
    execution_target: str = "local",
):
    job = service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_cutoff",
        execution_target=execution_target,
    )
    job.status = status
    return job


def _add_run(document, job, *, status: JobStatus, run_index: int = 1) -> CalculationRun:
    run = CalculationRun(job_id=job.id, run_index=run_index, status=status)
    document.runs[run.id] = run
    return run


@pytest.mark.parametrize("status", _ACTIVE_STATUSES)
def test_active_local_job_is_stopped_with_exact_message(tmp_path, status) -> None:
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="local", status=status)
    run = _add_run(document, job, status=status)

    reconciled = service.reconcile_orphaned_local_jobs(document)

    assert reconciled == [job]
    assert job.status == JobStatus.STOPPED
    assert job.error_message == ORPHANED_LOCAL_JOB_STOP_MESSAGE
    assert job.failure_reason == FailureReason.INTERRUPTED
    assert job.completed_at is not None
    # The in-flight run is closed out to keep run/job status consistent.
    assert run.status == JobStatus.STOPPED
    assert run.completed_at is not None


def test_exact_message_text() -> None:
    assert ORPHANED_LOCAL_JOB_STOP_MESSAGE == (
        "MuFinder was closed while this job was in progress locally; "
        "reset or resubmit it."
    )


def test_blank_execution_target_is_treated_as_local(tmp_path) -> None:
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="blank", status=JobStatus.RUNNING, execution_target="")

    reconciled = service.reconcile_orphaned_local_jobs(document)

    assert reconciled == [job]
    assert job.status == JobStatus.STOPPED


def test_local_target_matches_case_insensitively(tmp_path) -> None:
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="LOCAL", status=JobStatus.RUNNING, execution_target="Local")

    reconciled = service.reconcile_orphaned_local_jobs(document)

    assert reconciled == [job]
    assert job.status == JobStatus.STOPPED


@pytest.mark.parametrize("status", _ACTIVE_STATUSES)
def test_active_cluster_job_is_left_untouched(tmp_path, status) -> None:
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="cluster", status=status, execution_target="SCARF")

    reconciled = service.reconcile_orphaned_local_jobs(document)

    assert reconciled == []
    assert job.status == status


@pytest.mark.parametrize("status", _TERMINAL_OR_DEFINED)
def test_inactive_local_job_is_left_untouched(tmp_path, status) -> None:
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="inactive", status=status)

    reconciled = service.reconcile_orphaned_local_jobs(document)

    assert reconciled == []
    assert job.status == status


def test_submitting_local_job_without_a_run_does_not_crash(tmp_path) -> None:
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="no_run", status=JobStatus.SUBMITTING)

    reconciled = service.reconcile_orphaned_local_jobs(document)

    assert reconciled == [job]
    assert job.status == JobStatus.STOPPED


def test_only_latest_active_run_is_closed(tmp_path) -> None:
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="multi_run", status=JobStatus.RUNNING)
    # A terminal earlier run and a newer active run.
    old_run = _add_run(document, job, status=JobStatus.FAILED, run_index=1)
    active_run = _add_run(document, job, status=JobStatus.RUNNING, run_index=2)

    service.reconcile_orphaned_local_jobs(document)

    assert active_run.status == JobStatus.STOPPED
    assert active_run.completed_at is not None
    # The already-terminal earlier run is untouched.
    assert old_run.status == JobStatus.FAILED
    assert old_run.completed_at is None


def test_reconciliation_records_job_stopped_provenance(tmp_path) -> None:
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="prov", status=JobStatus.RUNNING)

    before = len(document.provenance)
    service.reconcile_orphaned_local_jobs(document)

    new_records = document.provenance[before:]
    stopped = [r for r in new_records if r.action == "job_stopped"]
    assert len(stopped) == 1
    record = stopped[0]
    assert record.job_id == job.id
    assert record.actor == "system"
    assert record.parameters.get("error_message") == ORPHANED_LOCAL_JOB_STOP_MESSAGE


def test_mixed_document_only_reconciles_active_local_jobs(tmp_path) -> None:
    service, document = _service_and_document(tmp_path)
    local_active = _add_job(service, document, label="local_active", status=JobStatus.RUNNING)
    _add_job(service, document, label="cluster_active", status=JobStatus.RUNNING, execution_target="SCARF")
    _add_job(service, document, label="local_done", status=JobStatus.COMPLETED)
    blank_active = _add_job(
        service, document, label="blank_active", status=JobStatus.QUEUED, execution_target=""
    )

    reconciled = service.reconcile_orphaned_local_jobs(document)

    assert {job.id for job in reconciled} == {local_active.id, blank_active.id}


def test_reconciled_job_records_monitor_metadata(tmp_path) -> None:
    # Job-lifecycle hardening rec #2: reconciliation stamps the monitor
    # metadata so the dashboard can show who last observed the job and why.
    service, document = _service_and_document(tmp_path)
    job = _add_job(service, document, label="local", status=JobStatus.RUNNING)

    service.reconcile_orphaned_local_jobs(document)

    assert job.last_checked_at is not None
    assert job.last_status_source == "reconciliation"
    assert job.last_status_message == ORPHANED_LOCAL_JOB_STOP_MESSAGE

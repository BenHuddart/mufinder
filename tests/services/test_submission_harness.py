"""Demonstration tests for the shared submission-flow harness.

These exercise the harness end-to-end for both a local and a cluster job,
driving the lifecycle by *emitting fake-runner signals* (the wiring
``MainWindow`` performs) rather than by calling controller methods directly.
"""

from __future__ import annotations

from pathlib import Path

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus

from tests.support.submission_harness import build_submission_harness


def test_harness_local_job_full_lifecycle(qapp, tmp_path: Path) -> None:
    harness = build_submission_harness(tmp_path)
    job = harness.add_job("local-job")

    outcome = harness.submit_local(job, num_cores=4, max_concurrent=2)
    assert outcome.submitted_count == 1
    assert harness.local_runner.submitted[0]["job_id"] == job.id
    assert harness.local_runner.max_concurrent == 2
    assert job.status == JobStatus.SUBMITTING

    # Drive the lifecycle purely by emitting runner signals.
    harness.emit_local_status(job.id, JobStatus.QUEUED)
    harness.emit_local_started(job.id)
    assert job.status == JobStatus.RUNNING

    harness.emit_local_finished(
        job.id,
        CastepResult(completed=True, total_energy=-42.0, warnings=["minor warning"]),
    )

    run = harness.controller.run_for_job(job.id)
    assert job.status == JobStatus.COMPLETED
    assert run is not None
    assert run.status == JobStatus.COMPLETED
    assert run.result_id == harness.results[job.id].id
    assert run.warnings == ["minor warning"]
    assert harness.controller.active_run_by_job == {}
    assert harness.document.provenance[-1].action == "job_completed"


def test_harness_local_job_failure_via_signal(qapp, tmp_path: Path) -> None:
    harness = build_submission_harness(tmp_path)
    job = harness.add_job("local-fail")

    harness.submit_local(job)
    harness.emit_local_status(job.id, JobStatus.QUEUED)
    harness.emit_local_started(job.id)

    harness.emit_local_failed(job.id, "castep crashed")

    assert job.status == JobStatus.FAILED
    assert job.error_message == "castep crashed"
    assert harness.controller.active_run_by_job == {}
    assert harness.document.provenance[-1].action == "job_failed"


def test_harness_cluster_job_full_lifecycle(qapp, tmp_path: Path) -> None:
    harness = build_submission_harness(tmp_path)
    job = harness.add_job("cluster-job", execution_target="test-cluster")

    outcome = harness.submit_cluster(job)
    assert outcome.submitted_job is job
    assert harness.cluster_runner.submitted[0]["job_id"] == job.id
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id == "12345"

    # Cluster runner reports progress via status polling, then a finished signal.
    harness.emit_cluster_status(job.id, JobStatus.RUNNING)
    assert job.status == JobStatus.RUNNING
    harness.emit_cluster_status(job.id, JobStatus.RETRIEVING)
    assert job.status == JobStatus.RETRIEVING

    harness.emit_cluster_finished(
        job.id,
        CastepResult(completed=True, total_energy=-10.0),
    )

    run = harness.controller.run_for_job(job.id)
    assert job.status == JobStatus.COMPLETED
    assert run is not None
    assert run.status == JobStatus.COMPLETED
    assert harness.controller.active_run_by_job == {}
    assert harness.document.provenance[-1].action == "job_completed"


def test_harness_cluster_job_started_signal_marks_running(qapp, tmp_path: Path) -> None:
    # The cluster runner normally advances jobs via status polling, but MainWindow
    # also wires job_started -> mark_started; the harness mirrors that wiring.
    harness = build_submission_harness(tmp_path)
    job = harness.add_job("cluster-started", execution_target="test-cluster")

    harness.submit_cluster(job)
    assert job.status == JobStatus.QUEUED

    harness.emit_cluster_started(job.id)
    assert job.status == JobStatus.RUNNING


def test_harness_cluster_job_stopped_is_terminal(qapp, tmp_path: Path) -> None:
    harness = build_submission_harness(tmp_path)
    job = harness.add_job("cluster-stopped", execution_target="test-cluster")

    harness.submit_cluster(job)
    harness.emit_cluster_status(job.id, JobStatus.RUNNING)
    harness.emit_cluster_status(job.id, JobStatus.STOPPED)

    run = harness.controller.run_for_job(job.id)
    assert job.status == JobStatus.STOPPED
    assert run is not None
    assert run.status == JobStatus.STOPPED
    assert harness.controller.active_run_by_job == {}
    assert harness.document.provenance[-1].action == "job_stopped"


def test_harness_local_queued_status_routes_through_project_service(qapp, tmp_path: Path) -> None:
    # Decision 2 pin: a local runner status_changed(queued) on a SUBMITTING job
    # advances it to QUEUED via ProjectService.mark_job_queued (not by the runner
    # or controller mutating Job.status directly). The transition is validated,
    # so remote_job_id stays None for a local job.
    harness = build_submission_harness(tmp_path)
    job = harness.add_job("local-queued")

    harness.submit_local(job)
    assert job.status == JobStatus.SUBMITTING

    harness.emit_local_status(job.id, JobStatus.QUEUED)
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id is None
    run = harness.controller.run_for_job(job.id)
    assert run is not None
    assert run.status == JobStatus.QUEUED


def test_harness_local_queued_status_is_noop_for_already_queued_job(qapp, tmp_path: Path) -> None:
    # Decision 2 pin: mark_local_status_changed is QUEUED-only and gated on the
    # job being SUBMITTING. A second queued notification for an already-QUEUED
    # job (or any job that is not SUBMITTING) is an inert no-op -- it must not
    # corrupt the job, re-run the transition, or disturb the active run.
    harness = build_submission_harness(tmp_path)
    job = harness.add_job("local-requeued")

    harness.submit_local(job)
    harness.emit_local_status(job.id, JobStatus.QUEUED)
    assert job.status == JobStatus.QUEUED
    run_id_before = harness.controller.active_run_by_job.get(job.id)

    # A redundant queued signal for the (now) QUEUED job is ignored.
    harness.emit_local_status(job.id, JobStatus.QUEUED)
    assert job.status == JobStatus.QUEUED
    assert harness.controller.active_run_by_job.get(job.id) == run_id_before

    # Once the job is RUNNING, a stray queued signal is likewise ignored (RUNNING
    # is not SUBMITTING, so the guard short-circuits) and does not regress state.
    harness.emit_local_started(job.id)
    assert job.status == JobStatus.RUNNING
    harness.emit_local_status(job.id, JobStatus.QUEUED)
    assert job.status == JobStatus.RUNNING

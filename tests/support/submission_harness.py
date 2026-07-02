"""Reusable submission-flow harness for characterization tests.

This wires together the real :class:`JobExecutionController` with fake local
and cluster runners that expose the *same* Qt signals as the production
runners (:class:`mufinder.services.local_runner.LocalJobRunner` and
:class:`mufinder.services.cluster_runner.ClusterJobRunner`). The harness
connects those signals to the controller in the same shape that
``MainWindow`` does (see ``_on_local_job_*`` / ``_on_cluster_job_*``), so a
test can drive a full lifecycle by *emitting runner signals* rather than by
poking controller methods directly.

The point of this harness (versus the inline fakes already in
``tests/services/test_job_execution_controller.py``) is to characterize the
signal → handler → controller translation, i.e. the wiring the app performs.

Coverage note: this module lives under ``tests/`` and coverage is scoped to
``source = ["mufinder"]``, so nothing here is measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from PySide6.QtCore import QObject, Signal

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job, Result
from mufinder.services.job_execution_controller import (
    ClusterJobSubmission,
    JobExecutionController,
)
from mufinder.services.project_service import ProjectDocument, ProjectService


class FakeLocalRunner(QObject):
    """Fake local runner exposing LocalJobRunner's public signal surface."""

    job_started = Signal(object)
    job_finished = Signal(object, object)
    job_failed = Signal(object, str)
    job_cancelled = Signal(object)
    status_changed = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self.max_concurrent = 1
        self.submitted: list[dict[str, object]] = []

    def submit(self, **kwargs: object) -> None:
        self.submitted.append(kwargs)


class FakeClusterRunner(QObject):
    """Fake cluster runner exposing the runner signals MainWindow connects."""

    job_started = Signal(object)
    job_finished = Signal(object, object)
    job_failed = Signal(object, str)
    job_cancelled = Signal(object)
    status_changed = Signal(object, object)

    def __init__(self, *, fail_message: str | None = None) -> None:
        super().__init__()
        self.fail_message = fail_message
        self.submitted: list[dict[str, object]] = []

    def submit(self, **kwargs: object) -> str:
        self.submitted.append(kwargs)
        if self.fail_message is not None:
            raise ConnectionError(self.fail_message)
        return "12345"


@dataclass
class SubmissionHarness:
    """End-to-end driver over a real controller and fake runners.

    Tests call :meth:`add_job`, then :meth:`submit_local` /
    :meth:`submit_cluster`, then emit runner signals via the ``local_runner`` /
    ``cluster_runner`` attributes (or use the convenience ``emit_*`` methods).
    """

    service: ProjectService
    document: ProjectDocument
    controller: JobExecutionController
    local_runner: FakeLocalRunner
    cluster_runner: FakeClusterRunner
    output_root: Path
    results: dict[UUID, Result] = field(default_factory=dict)

    # -- job creation ----------------------------------------------------
    def add_job(
        self,
        label: str,
        *,
        workflow_stage: WorkflowStage = WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type: str = "castep_cutoff",
        execution_target: str = "local",
    ) -> Job:
        return self.service.add_job(
            self.document,
            label=label,
            workflow_stage=workflow_stage,
            calc_type=calc_type,
            execution_target=execution_target,
        )

    def job_output_dir(self, job: Job) -> Path:
        return self.output_root / job.label

    # -- submission ------------------------------------------------------
    def submit_local(self, job: Job, *, num_cores: int = 1, max_concurrent: int = 1):
        return self.controller.submit_local_jobs(
            [job],
            num_cores=num_cores,
            max_concurrent=max_concurrent,
            castep_executable="/bin/castep",
            mpirun_path="",
        )

    def submit_cluster(self, job: Job, *, execution_target: str = "test-cluster"):
        return self.controller.submit_cluster_job(
            ClusterJobSubmission(
                job=job,
                output_dir=self.job_output_dir(job),
                seed_name=job.label,
                remote_target=f"/remote/{job.label}",
                execution_target=execution_target,
                num_processes=8,
                walltime="01:00:00",
                working_directory="/remote",
                template_script_path="",
                cores_per_node=4,
            )
        )

    # -- runner-signal emission (drives the wired handlers) --------------
    def emit_local_status(self, job_id: UUID, status: JobStatus) -> None:
        self.local_runner.status_changed.emit(job_id, status.value)

    def emit_local_started(self, job_id: UUID) -> None:
        self.local_runner.job_started.emit(job_id)

    def emit_local_failed(self, job_id: UUID, message: str) -> None:
        self.local_runner.job_failed.emit(job_id, message)

    def emit_local_cancelled(self, job_id: UUID) -> None:
        self.local_runner.job_cancelled.emit(job_id)

    def emit_local_finished(self, job_id: UUID, parsed: CastepResult) -> None:
        self.local_runner.job_finished.emit(job_id, parsed)

    def emit_cluster_status(self, job_id: UUID, status: JobStatus) -> None:
        self.cluster_runner.status_changed.emit(job_id, status.value)

    def emit_cluster_started(self, job_id: UUID) -> None:
        self.cluster_runner.job_started.emit(job_id)

    def emit_cluster_failed(self, job_id: UUID, message: str) -> None:
        self.cluster_runner.job_failed.emit(job_id, message)

    def emit_cluster_cancelled(self, job_id: UUID) -> None:
        self.cluster_runner.job_cancelled.emit(job_id)

    def emit_cluster_finished(self, job_id: UUID, parsed: CastepResult) -> None:
        self.cluster_runner.job_finished.emit(job_id, parsed)

    # -- result materialization convenience ------------------------------
    def record_result(self, job: Job, *, result_type: str, data: dict[str, object]) -> Result:
        result = Result(result_type=result_type, data=dict(data))
        self.document.results[result.id] = result
        self.results[job.id] = result
        return result


def _make_finished_handler(harness: SubmissionHarness):
    """Return a job_finished handler mirroring MainWindow's core completion path.

    This intentionally implements only the workflow-agnostic completion that
    ``MainWindow._on_local_job_finished`` / ``_on_cluster_job_finished`` share:
    materialize a Result from the parsed output and complete the run. The
    calc-type-specific file parsing (phonon, .ts, -out.cell, density) is out of
    scope for a submission-flow harness and is exercised elsewhere.
    """

    def _handle(job_id: UUID, parsed: CastepResult) -> None:
        job = harness.document.jobs.get(job_id)
        if job is None:
            return
        run = harness.controller.ensure_run_for_job(job, status=JobStatus.RUNNING, started=True)
        if run is None:
            return
        result = harness.record_result(
            job,
            result_type=job.calc_type,
            data={
                "completed": parsed.completed,
                "total_energy": parsed.total_energy,
                "warnings": list(parsed.warnings),
            },
        )
        harness.controller.complete_job_with_result(
            job,
            result=result,
            warnings=list(parsed.warnings),
            convergence_achieved=parsed.completed,
        )

    return _handle


def build_submission_harness(tmp_path: Path) -> SubmissionHarness:
    """Construct a wired harness rooted at ``tmp_path``.

    The runner signals are connected to the controller in the same topology
    ``MainWindow`` uses, so emitting a runner signal drives the real lifecycle.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / "harness.mufinder", "Harness")
    output_root = tmp_path / "jobs"
    output_root.mkdir(parents=True, exist_ok=True)

    local_runner = FakeLocalRunner()
    cluster_runner = FakeClusterRunner()

    controller = JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=local_runner,
        cluster_runner=cluster_runner,
        job_output_directory=lambda job: output_root / job.label,
        job_seed_name=lambda job: job.label,
    )

    harness = SubmissionHarness(
        service=service,
        document=document,
        controller=controller,
        local_runner=local_runner,
        cluster_runner=cluster_runner,
        output_root=output_root,
    )

    finished_handler = _make_finished_handler(harness)

    # Wire fake-runner signals to controller methods, mirroring MainWindow.
    local_runner.status_changed.connect(
        lambda job_id, status_value: controller.mark_local_status_changed(job_id, status_value)
    )
    local_runner.job_started.connect(lambda job_id: controller.mark_started(job_id))
    local_runner.job_failed.connect(lambda job_id, message: controller.fail_job(job_id, message))
    local_runner.job_cancelled.connect(lambda job_id: controller.cancel_job(job_id))
    local_runner.job_finished.connect(finished_handler)

    cluster_runner.status_changed.connect(
        lambda job_id, status_value: controller.mark_cluster_status_changed(job_id, status_value)
    )
    # MainWindow connects cluster job_started to mark_started (see
    # _on_cluster_job_started); in practice the cluster runner advances jobs via
    # status polling, but we wire it for fidelity so tests can emit it.
    cluster_runner.job_started.connect(lambda job_id: controller.mark_started(job_id))
    cluster_runner.job_failed.connect(lambda job_id, message: controller.fail_job(job_id, message))
    cluster_runner.job_cancelled.connect(lambda job_id: controller.cancel_job(job_id))
    cluster_runner.job_finished.connect(finished_handler)

    return harness

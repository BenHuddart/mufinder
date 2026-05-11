"""Shared job execution lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationRun, Job, Result, utc_now
from mufinder.services.project_service import ProjectDocument, ProjectService


@dataclass(frozen=True, slots=True)
class SubmissionOutcome:
    """Summary of a batch submission request."""

    submitted_jobs: tuple[Job, ...] = ()
    missing_paths: tuple[str, ...] = ()
    failed_messages: tuple[str, ...] = ()
    journal_changed: bool = False

    @property
    def submitted_count(self) -> int:
        """Return the number of jobs accepted by a runner."""

        return len(self.submitted_jobs)


@dataclass(frozen=True, slots=True)
class ClusterJobSubmission:
    """Controller-owned inputs for submitting one remote job."""

    job: Job
    output_dir: Path
    seed_name: str
    remote_target: str | None
    execution_target: str
    num_processes: int
    walltime: str
    working_directory: str
    template_script_path: str
    cores_per_node: int
    memory_gb: float | None = None
    queue_name: str = ""
    comment: str = ""
    castep_executable: str = ""
    retrieve_den_fmt: bool = False
    retrieve_out_cell: bool = False
    retrieve_phonon: bool = False
    retrieve_ts: bool = False
    reuse_existing_files: bool = False


@dataclass(frozen=True, slots=True)
class ClusterJobSubmissionOutcome:
    """Result of submitting one remote job."""

    submitted_job: Job | None = None
    failed_message: str | None = None
    journal_changed: bool = False


@dataclass(frozen=True, slots=True)
class LifecycleUpdate:
    """Result of a runner lifecycle notification."""

    job: Job | None = None
    changed: bool = False
    journal_changed: bool = False


class JobExecutionController:
    """Coordinate job/run state transitions around execution runners."""

    def __init__(
        self,
        *,
        document_getter: Callable[[], ProjectDocument | None],
        project_service: ProjectService,
        local_runner: Any,
        cluster_runner: Any,
        job_output_directory: Callable[[Job], Path | None],
        job_seed_name: Callable[[Job], str],
        active_run_by_job: dict[UUID, UUID] | None = None,
    ) -> None:
        self._document_getter = document_getter
        self._project_service = project_service
        self._local_runner = local_runner
        self._cluster_runner = cluster_runner
        self._job_output_directory = job_output_directory
        self._job_seed_name = job_seed_name
        self._active_run_by_job = active_run_by_job if active_run_by_job is not None else {}

    @property
    def active_run_by_job(self) -> dict[UUID, UUID]:
        """Mapping from active job IDs to the run currently receiving events."""

        return self._active_run_by_job

    def next_run_index(self, job_id: UUID) -> int:
        """Return the next persisted run index for a job."""

        document = self._document_getter()
        if document is None:
            return 1
        indices = [run.run_index for run in document.runs.values() if run.job_id == job_id]
        return max(indices, default=0) + 1

    def run_for_job(self, job_id: UUID) -> CalculationRun | None:
        """Return the active run for a job, falling back to the latest persisted run."""

        document = self._document_getter()
        if document is None:
            return None

        run_id = self._active_run_by_job.get(job_id)
        if run_id is not None:
            return document.runs.get(run_id)

        runs = [run for run in document.runs.values() if run.job_id == job_id]
        if not runs:
            return None
        return max(runs, key=lambda run: run.run_index)

    def ensure_run_for_job(
        self,
        job: Job,
        *,
        status: JobStatus = JobStatus.RUNNING,
        started: bool = False,
    ) -> CalculationRun | None:
        """Return the active run for a job, creating a fallback run when needed."""

        document = self._document_getter()
        if document is None:
            return None

        run = self.run_for_job(job.id)
        if run is not None:
            return run

        run = CalculationRun(
            job_id=job.id,
            run_index=self.next_run_index(job.id),
            status=status,
            started_at=utc_now() if started else None,
        )
        document.runs[run.id] = run
        self._active_run_by_job[job.id] = run.id
        return run

    def submit_local_jobs(
        self,
        jobs: Sequence[Job],
        *,
        num_cores: int,
        max_concurrent: int,
        castep_executable: str,
        mpirun_path: str,
        eligible_statuses: set[JobStatus] | None = None,
    ) -> SubmissionOutcome:
        """Submit eligible jobs to the local runner and create tracked runs."""

        document = self._document_getter()
        if document is None:
            return SubmissionOutcome()

        allowed_statuses = eligible_statuses or {JobStatus.DEFINED}
        self._local_runner.max_concurrent = max(1, int(max_concurrent))
        submitted_jobs: list[Job] = []
        missing_paths: list[str] = []
        failed_messages: list[str] = []
        journal_changed = False

        for job in jobs:
            if job.status not in allowed_statuses:
                continue
            output_dir = self._job_output_directory(job)
            if output_dir is None:
                missing_paths.append(job.label)
                continue
            seed_name = self._job_seed_name(job)

            try:
                job.execution_target = "local"
                self._project_service.submit_job(job)
            except ValueError:
                continue
            job.remote_job_id = None
            job.remote_directory = None
            job.num_processes = max(1, int(num_cores))
            self._create_active_run(document, job, JobStatus.SUBMITTING)

            try:
                self._local_runner.submit(
                    job_id=job.id,
                    job_dir=output_dir,
                    seed_name=seed_name,
                    castep_executable=castep_executable,
                    mpirun_path=mpirun_path,
                    num_cores=num_cores,
                )
            except Exception as exc:  # noqa: BLE001
                failed_messages.append(f"- {job.label}: {exc}")
                self.fail_job(job.id, str(exc))
                journal_changed = True
                continue
            if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED, JobStatus.RUNNING}:
                submitted_jobs.append(job)

        if submitted_jobs:
            self._project_service.record_job_submission(
                document,
                jobs=submitted_jobs,
                execution_target="local",
            )
            journal_changed = True

        return SubmissionOutcome(
            submitted_jobs=tuple(submitted_jobs),
            missing_paths=tuple(missing_paths),
            failed_messages=tuple(failed_messages),
            journal_changed=journal_changed,
        )

    def submit_cluster_job(self, request: ClusterJobSubmission) -> ClusterJobSubmissionOutcome:
        """Submit or resubmit one cluster job and update lifecycle state."""

        document = self._document_getter()
        if document is None or request.job.status not in {JobStatus.DEFINED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return ClusterJobSubmissionOutcome()

        job = request.job
        try:
            self._project_service.submit_job(job)
        except ValueError:
            return ClusterJobSubmissionOutcome()
        job.num_processes = max(1, int(request.num_processes))
        job.execution_target = request.execution_target
        job.remote_directory = request.remote_target
        run = self._create_active_run(document, job, JobStatus.SUBMITTING)

        try:
            remote_job_id = self._cluster_runner.submit(
                job_id=job.id,
                local_job_dir=request.output_dir,
                seed_name=request.seed_name,
                num_processes=request.num_processes,
                walltime=request.walltime,
                working_directory=request.working_directory,
                template_script_path=request.template_script_path,
                cores_per_node=request.cores_per_node,
                memory_gb=request.memory_gb,
                queue_name=request.queue_name,
                comment=request.comment,
                castep_executable=request.castep_executable,
                retrieve_den_fmt=request.retrieve_den_fmt,
                retrieve_out_cell=request.retrieve_out_cell,
                retrieve_phonon=request.retrieve_phonon,
                retrieve_ts=request.retrieve_ts,
                remote_directory=request.remote_target,
                reuse_existing_files=request.reuse_existing_files,
            )
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            try:
                self._project_service.fail_job(job, message)
            except ValueError:
                pass
            run.status = JobStatus.FAILED
            run.completed_at = utc_now()
            self._active_run_by_job.pop(job.id, None)
            self._record_terminal_event("job_failed", job, error_message=message)
            return ClusterJobSubmissionOutcome(
                failed_message=f"- {job.label}: {message}",
                journal_changed=True,
            )

        job.remote_job_id = remote_job_id
        job.remote_directory = request.remote_target
        if job.status == JobStatus.SUBMITTING:
            self._project_service.mark_job_queued(job, remote_job_id=remote_job_id)
        run.status = job.status
        return ClusterJobSubmissionOutcome(submitted_job=job)

    def record_submission(self, jobs: Sequence[Job], execution_target: str) -> bool:
        """Record one provenance event for a batch of submitted jobs."""

        document = self._document_getter()
        if document is None or not jobs:
            return False
        self._project_service.record_job_submission(
            document,
            jobs=jobs,
            execution_target=execution_target,
        )
        return True

    def mark_local_status_changed(self, job_id: UUID, status_value: str) -> LifecycleUpdate:
        """Apply local-runner queue notifications."""

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()
        if status_value != JobStatus.QUEUED.value or job.status != JobStatus.SUBMITTING:
            return LifecycleUpdate(job=job)

        try:
            self._project_service.mark_job_queued(job, remote_job_id=None)
        except ValueError:
            return LifecycleUpdate(job=job)
        run = self.run_for_job(job_id)
        if run is not None:
            run.status = JobStatus.QUEUED
        return LifecycleUpdate(job=job, changed=True)

    def mark_started(self, job_id: UUID) -> LifecycleUpdate:
        """Mark a job and its active run as running."""

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()

        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}:
            try:
                self._project_service.start_job(job)
            except ValueError:
                pass

        run = self.run_for_job(job_id)
        if run is not None:
            run.status = JobStatus.RUNNING
            if run.started_at is None:
                run.started_at = utc_now()
        return LifecycleUpdate(job=job, changed=True)

    def complete_job_with_result(
        self,
        job: Job,
        *,
        result: Result,
        warnings: Sequence[str],
        convergence_achieved: bool | None,
    ) -> bool:
        """Complete a job after workflow-specific result materialization."""

        run = self.run_for_job(job.id)
        if run is None:
            run = self.ensure_run_for_job(job, status=JobStatus.RUNNING, started=True)
        if run is None:
            return False

        run.result_id = result.id
        run.status = JobStatus.COMPLETED
        if run.started_at is None:
            run.started_at = utc_now()
        run.completed_at = utc_now()
        run.warnings = list(warnings)
        run.convergence_achieved = convergence_achieved

        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}:
            try:
                self._project_service.start_job(job)
            except ValueError:
                pass
        if job.status in {JobStatus.RUNNING, JobStatus.RETRIEVING}:
            try:
                self._project_service.complete_job(job)
            except ValueError:
                pass

        self._active_run_by_job.pop(job.id, None)
        self._record_terminal_event("job_completed", job, result=result)
        return True

    def fail_job(self, job_id: UUID, message: str) -> LifecycleUpdate:
        """Mark a job as failed and record a terminal provenance event."""

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()

        if job.status in {
            JobStatus.SUBMITTING,
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.RETRIEVING,
        }:
            try:
                self._project_service.fail_job(job, message)
            except ValueError:
                pass

        run = self.run_for_job(job_id)
        if run is not None:
            if run.started_at is None:
                run.started_at = utc_now()
            run.completed_at = utc_now()
            run.status = JobStatus.FAILED
            run.warnings = [message]

        self._active_run_by_job.pop(job_id, None)
        self._record_terminal_event("job_failed", job, error_message=message)
        return LifecycleUpdate(job=job, changed=True, journal_changed=True)

    def cancel_job(self, job_id: UUID) -> LifecycleUpdate:
        """Mark a job as cancelled and record a terminal provenance event."""

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()

        if job.status in {
            JobStatus.DEFINED,
            JobStatus.SUBMITTING,
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.RETRIEVING,
        }:
            try:
                self._project_service.cancel_job(job)
            except ValueError:
                pass

        run = self.run_for_job(job_id)
        if run is not None:
            run.completed_at = utc_now()
            run.status = JobStatus.CANCELLED

        self._active_run_by_job.pop(job_id, None)
        self._record_terminal_event("job_cancelled", job)
        return LifecycleUpdate(job=job, changed=True, journal_changed=True)

    def mark_cluster_status_changed(self, job_id: UUID, status_value: str) -> LifecycleUpdate:
        """Apply cluster-runner status polling notifications."""

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()

        status_map = {
            JobStatus.QUEUED.value: JobStatus.QUEUED,
            JobStatus.RUNNING.value: JobStatus.RUNNING,
            JobStatus.RETRIEVING.value: JobStatus.RETRIEVING,
            JobStatus.STOPPED.value: JobStatus.STOPPED,
        }
        mapped = status_map.get(status_value)
        if mapped is None:
            return LifecycleUpdate(job=job)

        try:
            if mapped == JobStatus.QUEUED and job.status == JobStatus.SUBMITTING:
                self._project_service.mark_job_queued(job, remote_job_id=job.remote_job_id)
            elif mapped == JobStatus.RUNNING and job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}:
                self._project_service.start_job(job)
            elif mapped == JobStatus.RETRIEVING and job.status == JobStatus.RUNNING:
                self._project_service.mark_job_retrieving(job)
            elif mapped == JobStatus.STOPPED and job.status in {JobStatus.RUNNING, JobStatus.RETRIEVING}:
                self._project_service.stop_job(job, "Job stopped before successful completion.")
            else:
                return LifecycleUpdate(job=job)
        except ValueError:
            return LifecycleUpdate(job=job)

        run = self.run_for_job(job_id)
        if run is not None:
            run.status = mapped
            if mapped == JobStatus.RUNNING and run.started_at is None:
                run.started_at = utc_now()
            if mapped == JobStatus.STOPPED:
                run.completed_at = utc_now()

        journal_changed = False
        if mapped == JobStatus.STOPPED:
            self._active_run_by_job.pop(job_id, None)
            self._record_terminal_event(
                "job_stopped",
                job,
                error_message=job.error_message,
            )
            journal_changed = True
        return LifecycleUpdate(job=job, changed=True, journal_changed=journal_changed)

    def _create_active_run(
        self,
        document: ProjectDocument,
        job: Job,
        status: JobStatus,
    ) -> CalculationRun:
        run = CalculationRun(
            job_id=job.id,
            run_index=self.next_run_index(job.id),
            status=status,
        )
        document.runs[run.id] = run
        self._active_run_by_job[job.id] = run.id
        return run

    def _job_for_id(self, job_id: UUID) -> Job | None:
        document = self._document_getter()
        if document is None:
            return None
        return document.jobs.get(job_id)

    def _record_terminal_event(
        self,
        action: str,
        job: Job,
        *,
        result: Result | None = None,
        error_message: str | None = None,
    ) -> None:
        document = self._document_getter()
        if document is None:
            return
        self._project_service.record_job_terminal_event(
            document,
            action=action,
            job=job,
            result=result,
            error_message=error_message,
        )

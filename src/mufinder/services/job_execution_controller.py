"""Shared job execution lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from mufinder.domain.enums import FailureReason, JobStatus
from mufinder.domain.models import CalculationRun, Job, Result, utc_now
from mufinder.services.failure_diagnosis import (
    FailureDiagnosis,
    diagnose_failure,
    failure_reason_summary,
    find_failure_log_files,
)
from mufinder.services.job_monitoring import is_cluster_job
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
    mem_per_cpu_gb: float | None = None
    queue_name: str = ""
    comment: str = ""
    castep_executable: str = ""
    retrieve_den_fmt: bool = False
    retrieve_out_cell: bool = False
    retrieve_phonon: bool = False
    retrieve_ts: bool = False
    reuse_existing_files: bool = False
    allow_existing_remote_directory: bool = False
    task_farming_enabled: bool = False
    num_farms: int | None = None


@dataclass(frozen=True, slots=True)
class ClusterJobSubmissionOutcome:
    """Result of submitting one remote job."""

    submitted_job: Job | None = None
    failed_message: str | None = None
    journal_changed: bool = False


@dataclass(frozen=True, slots=True)
class PreparedClusterSubmission:
    """Document-side state created before the remote SSH submit runs.

    Splitting submission into *prepare* (mutates the project document on the GUI
    thread), the SSH *submit* (runs on the cluster worker thread), and *finalize*
    (mutates the document again on the GUI thread) lets the blocking network work
    move off the GUI thread without touching project state from two threads.
    """

    job: Job
    run: CalculationRun
    submit_kwargs: dict[str, Any]


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
        task_farming_for_job: Callable[[Job], tuple[bool, int | None]] | None = None,
        retrieve_flags_for_job: Callable[[Job], dict[str, bool]] | None = None,
    ) -> SubmissionOutcome:
        """Submit eligible jobs to the local runner and create tracked runs.

        ``retrieve_flags_for_job`` maps a job to its per-calc-type secondary-output
        flags (``retrieve_out_cell`` / ``retrieve_phonon`` / ``retrieve_ts``),
        mirroring the cluster path. The local runner uses them to parse those
        outputs off the GUI thread on completion, exactly once.
        """

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
            task_farming_enabled, num_farms = (
                task_farming_for_job(job) if task_farming_for_job is not None else (False, None)
            )
            retrieve_flags = (
                retrieve_flags_for_job(job) if retrieve_flags_for_job is not None else {}
            )

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
                    task_farming_enabled=task_farming_enabled,
                    num_farms=num_farms,
                    retrieve_out_cell=bool(retrieve_flags.get("retrieve_out_cell", False)),
                    retrieve_phonon=bool(retrieve_flags.get("retrieve_phonon", False)),
                    retrieve_ts=bool(retrieve_flags.get("retrieve_ts", False)),
                )
            except Exception as exc:  # noqa: BLE001
                failed_messages.append(f"- {job.label}: {exc}")
                self.fail_job(job.id, str(exc), reason=FailureReason.SUBMISSION)
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

    def prepare_cluster_submission(
        self, request: ClusterJobSubmission
    ) -> PreparedClusterSubmission | None:
        """Mutate document state for one job and return its SSH submit arguments.

        Runs on the GUI thread. The returned ``submit_kwargs`` are passed to the
        cluster runner's ``submit`` on the worker thread; the document is not
        touched again until :meth:`finalize_cluster_submission` or
        :meth:`fail_cluster_submission`.

        The accepted statuses are the ones that can legally transition to
        SUBMITTING from outside an active run: DEFINED plus the terminal statuses
        the resubmit paths accept.
        """

        document = self._document_getter()
        accepted_statuses = {
            JobStatus.DEFINED,
            JobStatus.FAILED,
            JobStatus.STOPPED,
            JobStatus.CANCELLED,
        }
        if document is None or request.job.status not in accepted_statuses:
            return None

        job = request.job
        try:
            self._project_service.submit_job(job)
        except ValueError:
            return None
        job.num_processes = max(1, int(request.num_processes))
        job.execution_target = request.execution_target
        job.remote_directory = request.remote_target
        run = self._create_active_run(document, job, JobStatus.SUBMITTING)

        submit_kwargs: dict[str, Any] = {
            "job_id": job.id,
            "local_job_dir": request.output_dir,
            "seed_name": request.seed_name,
            "num_processes": request.num_processes,
            "walltime": request.walltime,
            "working_directory": request.working_directory,
            "template_script_path": request.template_script_path,
            "cores_per_node": request.cores_per_node,
            "memory_gb": request.memory_gb,
            "mem_per_cpu_gb": request.mem_per_cpu_gb,
            "queue_name": request.queue_name,
            "comment": request.comment,
            "castep_executable": request.castep_executable,
            "retrieve_den_fmt": request.retrieve_den_fmt,
            "retrieve_out_cell": request.retrieve_out_cell,
            "retrieve_phonon": request.retrieve_phonon,
            "retrieve_ts": request.retrieve_ts,
            "remote_directory": request.remote_target,
            "reuse_existing_files": request.reuse_existing_files,
            "allow_existing_remote_directory": request.allow_existing_remote_directory,
            "task_farming_enabled": request.task_farming_enabled,
            "num_farms": request.num_farms,
        }
        return PreparedClusterSubmission(job=job, run=run, submit_kwargs=submit_kwargs)

    def finalize_cluster_submission(
        self, prepared: PreparedClusterSubmission, remote_job_id: str
    ) -> ClusterJobSubmissionOutcome:
        """Record a successful remote submit. Runs on the GUI thread."""

        job = prepared.job
        run = prepared.run
        job.remote_job_id = remote_job_id
        if job.status == JobStatus.SUBMITTING:
            self._project_service.mark_job_queued(job, remote_job_id=remote_job_id)
        run.status = job.status
        return ClusterJobSubmissionOutcome(submitted_job=job)

    def fail_cluster_submission(
        self, prepared: PreparedClusterSubmission, message: str
    ) -> ClusterJobSubmissionOutcome:
        """Record a failed remote submit. Runs on the GUI thread."""

        job = prepared.job
        run = prepared.run
        try:
            self._project_service.fail_job(job, message, reason=FailureReason.SUBMISSION)
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

    def submit_cluster_job(self, request: ClusterJobSubmission) -> ClusterJobSubmissionOutcome:
        """Submit or resubmit one cluster job synchronously and update state.

        Convenience wrapper retained for non-GUI callers and tests; the GUI path
        drives :meth:`prepare_cluster_submission`/:meth:`finalize_cluster_submission`
        directly so the SSH submit can run on the cluster worker thread.
        """

        prepared = self.prepare_cluster_submission(request)
        if prepared is None:
            return ClusterJobSubmissionOutcome()

        try:
            remote_job_id = self._cluster_runner.submit(**prepared.submit_kwargs)
        except Exception as exc:  # noqa: BLE001
            return self.fail_cluster_submission(prepared, str(exc))
        return self.finalize_cluster_submission(prepared, remote_job_id)

    def prepare_retry_retrieval(self, job_id: UUID) -> Job | None:
        """Move a failed/stopped cluster job back into RETRIEVING for re-download.

        Validates the retry preconditions (failed or stopped, cluster target,
        remote id and directory present), calls
        :meth:`ProjectService.retry_job_retrieval`, stamps the user-initiated
        monitor metadata, and registers a *fresh* ``CalculationRun`` (status
        RETRIEVING) as the job's active run.

        A fresh run is required: the failure/stop path already popped the active
        run and marked the previous run terminal, so without a new active run
        ``run_for_job`` would fall back to that terminal run and the eventual
        completion would overwrite a FAILED run to COMPLETED. Returning the job
        lets the GUI dispatch the retrieval; ``None`` means the job was
        ineligible (the button is normally disabled in that case).
        """

        document = self._document_getter()
        if document is None:
            return None
        job = document.jobs.get(job_id)
        if job is None:
            return None
        if job.status not in {JobStatus.FAILED, JobStatus.STOPPED}:
            return None
        target = (job.execution_target or "").strip()
        if not target or target.casefold() == "local":
            return None
        if not (job.remote_job_id or "").strip() or not (job.remote_directory or "").strip():
            return None

        try:
            self._project_service.retry_job_retrieval(job)
        except ValueError:
            return None

        job.last_checked_at = utc_now()
        job.last_status_source = "user"
        job.last_status_message = "Retry retrieval requested"
        self._create_active_run(document, job, JobStatus.RETRIEVING)
        return job

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
        self._stamp_monitor_metadata(job, source="local_runner", message="queued locally")
        run = self.run_for_job(job_id)
        if run is not None:
            run.status = JobStatus.QUEUED
        return LifecycleUpdate(job=job, changed=True)

    def mark_started(
        self,
        job_id: UUID,
        *,
        source: str | None = None,
        status_message: str | None = None,
    ) -> LifecycleUpdate:
        """Mark a job and its active run as running.

        ``source``/``status_message`` override the monitor-metadata stamp.
        When both are ``None`` (the runner signal path), local jobs are stamped
        ``local_runner`` / ``"process started"``; cluster jobs are left
        unchanged because their monitor metadata is stamped from queue polls
        via :meth:`mark_queue_checked`.
        """

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()

        if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}:
            try:
                self._project_service.start_job(job)
            except ValueError:
                pass

        if source is not None:
            self._stamp_monitor_metadata(job, source=source, message=status_message or "process started")
        elif not is_cluster_job(job):
            self._stamp_monitor_metadata(
                job, source="local_runner", message=status_message or "process started"
            )

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
        source: str | None = None,
        status_message: str | None = None,
    ) -> bool:
        """Complete a job after workflow-specific result materialization.

        ``source``/``status_message`` override the monitor-metadata stamp.
        When ``None`` (the shared local/cluster ``job_finished`` path), the
        stamp is inferred from the job's execution target: cluster jobs reach
        this method through output retrieval (``"retrieval"`` /
        ``"output retrieved and parsed"``), local jobs through the local
        runner (``"local_runner"`` / ``"process completed"``).
        """

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

        if is_cluster_job(job):
            resolved_source = source or "retrieval"
            resolved_message = status_message or "output retrieved and parsed"
        else:
            resolved_source = source or "local_runner"
            resolved_message = status_message or "process completed"
        self._stamp_monitor_metadata(job, source=resolved_source, message=resolved_message)

        self._active_run_by_job.pop(job.id, None)
        self._record_terminal_event("job_completed", job, result=result)
        return True

    def fail_job(
        self,
        job_id: UUID,
        message: str,
        *,
        source: str | None = None,
        status_message: str | None = None,
        reason: FailureReason | None = None,
    ) -> LifecycleUpdate:
        """Mark a job as failed and record a terminal provenance event.

        ``reason`` records a cause the caller already knows (a rejected
        submission). Otherwise the cause is diagnosed from ``message`` and the
        job's scheduler/CASTEP logs (see :meth:`diagnose_failure`).

        ``source``/``status_message`` override the monitor-metadata stamp.
        When ``None`` (the shared local/cluster ``job_failed`` path), the
        stamp is inferred from the job's execution target: cluster failures
        reach this method through output retrieval (``"retrieval"`` /
        ``"output retrieval failed"``), local failures through the local
        runner (``"local_runner"`` / ``"process failed"``). The short
        observation deliberately does not duplicate the full ``message``,
        which is persisted on ``job.error_message``.
        """

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()

        if job.status in {
            JobStatus.SUBMITTING,
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.RETRIEVING,
        }:
            if reason is None:
                reason = self.diagnose_failure(job, message).reason
            try:
                self._project_service.fail_job(job, message, reason=reason)
            except ValueError:
                pass

        if is_cluster_job(job):
            resolved_source = source or "retrieval"
            resolved_message = status_message or "output retrieval failed"
        else:
            resolved_source = source or "local_runner"
            resolved_message = status_message or "process failed"
        self._stamp_monitor_metadata(job, source=resolved_source, message=resolved_message)

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

    def diagnose_failure(self, job: Job, message: str | None) -> FailureDiagnosis:
        """Classify why ``job`` failed from ``message`` and its output-directory logs.

        Logs older than the job's submission are ignored, so a continuation in
        a reused directory is not blamed on the previous run's errors.
        """

        output_dir = self._job_output_directory(job)
        log_files = (
            []
            if output_dir is None
            else find_failure_log_files(
                output_dir, [job.label, self._job_seed_name(job)], job.remote_job_id
            )
        )
        return diagnose_failure(message, log_files, not_before=job.submitted_at)

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
                # The cluster runner reports ``stopped`` only for a walltime
                # stop: the scheduler's TIMEOUT state, or a time-limit kill in
                # the scheduler log of a run with incomplete output.
                self._project_service.stop_job(
                    job,
                    failure_reason_summary(FailureReason.WALLTIME),
                    reason=FailureReason.WALLTIME,
                )
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

    def mark_queue_checked(self, snapshot: Any) -> bool:
        """Stamp per-job monitor metadata from one queue polling pass.

        ``snapshot`` is a ``QueueCheckSnapshot`` (duck-typed: any object with a
        ``checked_at`` datetime and an ``observations_by_job_id`` mapping).
        Every observed job that exists in the document gets
        ``last_checked_at``/``last_status_source``/``last_status_message``
        stamped with the poll observation. No status transitions and no run
        mutations happen here — this is bookkeeping only.

        :returns: ``True`` when at least one job was stamped.
        """

        document = self._document_getter()
        if document is None:
            return False
        checked_at = getattr(snapshot, "checked_at", None)
        observations = getattr(snapshot, "observations_by_job_id", None)
        if not isinstance(checked_at, datetime) or not isinstance(observations, Mapping):
            return False

        changed = False
        for job_id, observation in observations.items():
            job = document.jobs.get(job_id)
            if job is None:
                continue
            job.last_checked_at = checked_at
            job.last_status_source = "scheduler_poll"
            job.last_status_message = str(observation)
            changed = True
        return changed

    @staticmethod
    def _stamp_monitor_metadata(job: Job, *, source: str, message: str) -> None:
        """Record when, from where, and with what observation a job was last seen.

        These are bookkeeping fields (not status transitions), so the
        controller mutates them directly per the state-authority contract.
        """

        job.last_checked_at = utc_now()
        job.last_status_source = source
        job.last_status_message = message

    def stop_stuck_job(self, job_id: UUID, message: str) -> LifecycleUpdate:
        """Stop an active job the watchdog found untracked by any runner.

        Routes the transition through ``ProjectService.stop_job`` (the single
        authority for the ``-> stopped`` edge), closes out the job's active run
        the same way the walltime-stop and failure paths do, records a
        ``job_stopped`` journal event, and stamps the watchdog bookkeeping fields
        (``last_checked_at``/``last_status_source``/``last_status_message``).
        """

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()
        if job.status not in {
            JobStatus.SUBMITTING,
            JobStatus.QUEUED,
            JobStatus.RUNNING,
            JobStatus.RETRIEVING,
        }:
            return LifecycleUpdate(job=job)

        try:
            self._project_service.stop_job(job, message, reason=FailureReason.INTERRUPTED)
        except ValueError:
            return LifecycleUpdate(job=job)

        # Watchdog bookkeeping (mutated directly by the controller, per contract).
        stamped_at = utc_now()
        job.last_checked_at = stamped_at
        job.last_status_source = "watchdog"
        job.last_status_message = message

        run = self.run_for_job(job_id)
        if run is not None:
            if run.started_at is None:
                run.started_at = stamped_at
            run.completed_at = stamped_at
            run.status = JobStatus.STOPPED

        self._active_run_by_job.pop(job_id, None)
        self._record_terminal_event("job_stopped", job, error_message=message)
        return LifecycleUpdate(job=job, changed=True, journal_changed=True)

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

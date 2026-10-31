"""Shared job execution lifecycle orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from mufinder.domain.enums import ACTIVE_JOB_STATUSES, FailureReason, JobStatus
from mufinder.domain.models import CalculationRun, Job, Result, utc_now
from mufinder.services.failure_diagnosis import (
    MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE,
    FailureDiagnosis,
    diagnose_failure,
    failure_reason_summary,
    find_failure_log_files,
)
from mufinder.services.job_artifacts import RetrievalFlags
from mufinder.services.job_findings import record_job_finding
from mufinder.services.job_monitoring import is_cluster_job
from mufinder.services.project_service import ProjectDocument, ProjectService

LOGGER = logging.getLogger(__name__)


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


#: Project ``ui_state`` key holding the cores per job of the last local run.
LOCAL_SUBMISSION_CORES_KEY = "local_submission_cores"


def remembered_local_cores(ui_state: Mapping[str, Any]) -> int | None:
    """Return the cores per job last used for a local run in this project.

    ``None`` when the project has not run anything locally yet, or the saved
    value is not a positive whole number.
    """

    value = ui_state.get(LOCAL_SUBMISSION_CORES_KEY)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def remember_local_cores(ui_state: dict[str, Any], num_cores: int) -> None:
    """Record ``num_cores`` as the cores per job the next local run offers."""

    ui_state[LOCAL_SUBMISSION_CORES_KEY] = max(1, int(num_cores))


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
    reuse_existing_files: bool = False
    allow_existing_remote_directory: bool = False
    retrieval: RetrievalFlags = RetrievalFlags()


#: ``ClusterJobSubmission`` fields that are document-side only and never reach
#: ``ClusterJobRunner.submit``.
_SUBMISSION_ONLY_FIELDS = frozenset({"job", "execution_target", "retrieval"})
#: ``ClusterJobSubmission`` fields whose ``ClusterJobRunner.submit`` keyword differs.
_SUBMIT_KEYWORD_BY_FIELD = {"output_dir": "local_job_dir", "remote_target": "remote_directory"}


def _cluster_submit_kwargs(request: ClusterJobSubmission) -> dict[str, Any]:
    """Return the ``ClusterJobRunner.submit`` keywords for ``request``.

    Every runner-facing field is copied under its submit keyword, and the
    retrieval flags are expanded into the keywords they travel as.
    """

    kwargs: dict[str, Any] = {"job_id": request.job.id}
    for field in fields(request):
        if field.name in _SUBMISSION_ONLY_FIELDS:
            continue
        kwargs[_SUBMIT_KEYWORD_BY_FIELD.get(field.name, field.name)] = getattr(request, field.name)
    kwargs.update(request.retrieval.as_kwargs())
    return kwargs


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
        retrieval_for_job: Callable[[Job], RetrievalFlags] | None = None,
        task_farming_for_job: Callable[[Job], tuple[bool, int | None]] | None = None,
        retrieve_flags_for_job: Callable[[Job], Mapping[str, bool]] | None = None,
    ) -> SubmissionOutcome:
        """Submit eligible jobs to the local runner and create tracked runs.

        ``retrieval_for_job`` maps a job to its
        :class:`~mufinder.services.job_artifacts.RetrievalFlags`, mirroring the
        cluster path. The local runner uses them to find farmed outputs and to
        parse the secondary outputs once, on completion. Without it every job
        is submitted with default (all-off) flags.

        ``task_farming_for_job`` (``(enabled, num_farms)``) and
        ``retrieve_flags_for_job`` (a mapping of ``retrieve_*`` flags) are the
        compatibility form of ``retrieval_for_job``, used only when it is
        ``None``; see :doc:`/compatibility_policy`.
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
            if retrieval_for_job is not None:
                retrieval = retrieval_for_job(job)
            else:
                retrieval = _legacy_retrieval_flags(
                    job, task_farming_for_job, retrieve_flags_for_job
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
                    **retrieval.as_kwargs(),
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
        return PreparedClusterSubmission(
            job=job, run=run, submit_kwargs=_cluster_submit_kwargs(request)
        )

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
        try:
            self._project_service.fail_job(job, message, reason=FailureReason.SUBMISSION)
        except ValueError:
            pass
        self._close_active_run(
            job, prepared.run, JobStatus.FAILED, "job_failed", error_message=message
        )
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
            self._stamp_monitor_metadata(
                job, source=source, message=status_message or "process started"
            )
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
        document = self._document_getter()
        if document is not None:
            try:
                record_job_finding(document, job, result)
            except Exception:  # noqa: BLE001 - the job has completed; the Journal must not undo that
                LOGGER.exception("Could not record the finding of job %s", job.id)
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
            run.warnings = [message]
        self._close_active_run(
            job, run, JobStatus.FAILED, "job_failed", error_message=message, mark_started=True
        )
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

        self._close_active_run(job, self.run_for_job(job_id), JobStatus.CANCELLED, "job_cancelled")
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
            elif mapped == JobStatus.RUNNING and job.status in {
                JobStatus.SUBMITTING,
                JobStatus.QUEUED,
            }:
                self._project_service.start_job(job)
            elif mapped == JobStatus.RETRIEVING and job.status == JobStatus.RUNNING:
                self._project_service.mark_job_retrieving(job)
            elif mapped == JobStatus.STOPPED and job.status in {
                JobStatus.RUNNING,
                JobStatus.RETRIEVING,
            }:
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
        if mapped == JobStatus.STOPPED:
            self._close_active_run(
                job, run, JobStatus.STOPPED, "job_stopped", error_message=job.error_message
            )
            return LifecycleUpdate(job=job, changed=True, journal_changed=True)
        if run is not None:
            run.status = mapped
            if mapped == JobStatus.RUNNING and run.started_at is None:
                run.started_at = utc_now()
        return LifecycleUpdate(job=job, changed=True)

    def resume_reappeared_cluster_job(self, job_id: UUID, status_value: str) -> LifecycleUpdate:
        """Resume a job failed for a missing scheduler entry that the scheduler lists again.

        Applies the cluster runner's ``job_reappeared`` notification. Only a job
        for which :func:`awaits_scheduler_reappearance` holds is resumed: a
        ``queued`` or ``running`` report returns it to that status, and a
        ``completed`` report to ``retrieving`` so its output is fetched. The
        transition goes through ``ProjectService.resume_reappeared_job``, and a
        *fresh* ``CalculationRun`` becomes the active run, as for
        :meth:`prepare_retry_retrieval`, so the outcome cannot overwrite the
        failed run. The caller re-registers the job with the cluster runner.
        """

        document = self._document_getter()
        job = self._job_for_id(job_id)
        if document is None or job is None:
            return LifecycleUpdate()
        if not awaits_scheduler_reappearance(job):
            return LifecycleUpdate(job=job)
        status = {
            JobStatus.QUEUED.value: JobStatus.QUEUED,
            JobStatus.RUNNING.value: JobStatus.RUNNING,
            JobStatus.COMPLETED.value: JobStatus.RETRIEVING,
        }.get(status_value)
        if status is None:
            return LifecycleUpdate(job=job)

        try:
            self._project_service.resume_reappeared_job(job, status)
        except ValueError:
            return LifecycleUpdate(job=job)
        self._stamp_monitor_metadata(
            job,
            source="scheduler_poll",
            message=f"scheduler reports {status_value.upper()} again; tracking resumed",
        )
        run = self._create_active_run(document, job, status)
        if status == JobStatus.RUNNING:
            run.started_at = utc_now()
        return LifecycleUpdate(job=job, changed=True)

    def mark_queue_checked(self, snapshot: Any) -> bool:
        """Stamp per-job monitor metadata from one queue polling pass.

        ``snapshot`` is a ``QueueCheckSnapshot`` (duck-typed: any object with a
        ``checked_at`` datetime and an ``observations_by_job_id`` mapping).
        Every observed job that exists in the document gets
        ``last_checked_at``/``last_status_source``/``last_status_message``
        stamped with the poll observation, except a reappearance watch whose
        job no longer awaits it (see :meth:`stale_reappearance_watches`). No
        status transitions and no run mutations happen here — this is
        bookkeeping only.

        :returns: ``True`` when at least one job was stamped.
        """

        document = self._document_getter()
        if document is None:
            return False
        checked_at = getattr(snapshot, "checked_at", None)
        observations = getattr(snapshot, "observations_by_job_id", None)
        if not isinstance(checked_at, datetime) or not isinstance(observations, Mapping):
            return False

        stale = set(self.stale_reappearance_watches(snapshot))
        changed = False
        for job_id, observation in observations.items():
            job = document.jobs.get(job_id)
            if job is None or job_id in stale:
                continue
            job.last_checked_at = checked_at
            job.last_status_source = "scheduler_poll"
            job.last_status_message = str(observation)
            changed = True
        return changed

    def stale_reappearance_watches(self, snapshot: Any) -> list[UUID]:
        """Return the watched jobs in ``snapshot`` the runner should stop watching.

        A poll's ``watched_job_ids`` are jobs the cluster runner checks only for
        the scheduler listing them again. One whose job has since been reset,
        dismissed, deleted, or otherwise no longer satisfies
        :func:`awaits_scheduler_reappearance` (and has not resumed) is stale.
        """

        document = self._document_getter()
        watched = getattr(snapshot, "watched_job_ids", None)
        if document is None or not watched:
            return []
        stale: list[UUID] = []
        for job_id in watched:
            job = document.jobs.get(job_id)
            if job is None or not (
                job.status in ACTIVE_JOB_STATUSES or awaits_scheduler_reappearance(job)
            ):
                stale.append(job_id)
        return stale

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

        self._stamp_monitor_metadata(job, source="watchdog", message=message)
        self._close_active_run(
            job,
            self.run_for_job(job_id),
            JobStatus.STOPPED,
            "job_stopped",
            error_message=message,
            mark_started=True,
        )
        return LifecycleUpdate(job=job, changed=True, journal_changed=True)

    def _close_active_run(
        self,
        job: Job,
        run: CalculationRun | None,
        status: JobStatus,
        event: str,
        *,
        error_message: str | None = None,
        mark_started: bool = False,
    ) -> None:
        """Finish ``run`` as ``status``, release the job's active run, and journal ``event``.

        ``mark_started`` also stamps a run that never started with the same
        completion time, so a run that failed or was stopped before its process
        began still records a start.
        """

        if run is not None:
            completed_at = utc_now()
            if mark_started and run.started_at is None:
                run.started_at = completed_at
            run.completed_at = completed_at
            run.status = status
        self._active_run_by_job.pop(job.id, None)
        self._record_terminal_event(event, job, error_message=error_message)

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


def awaits_scheduler_reappearance(job: Job) -> bool:
    """Return whether ``job`` failed only because the scheduler stopped listing it.

    Such a cluster job may still be waiting in a queue the poll cannot see, so
    the cluster runner keeps watching it and the job resumes if the scheduler
    lists it again. A dismissed job has been acknowledged and is not watched.
    """

    return (
        job.status == JobStatus.FAILED
        and job.dismissed_at is None
        and is_cluster_job(job)
        and bool((job.remote_job_id or "").strip())
        and job.error_message == MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE
    )


def _legacy_retrieval_flags(
    job: Job,
    task_farming_for_job: Callable[[Job], tuple[bool, int | None]] | None,
    retrieve_flags_for_job: Callable[[Job], Mapping[str, bool]] | None,
) -> RetrievalFlags:
    """Build ``RetrievalFlags`` from the compatibility form of ``submit_local_jobs``."""

    task_farming_enabled, num_farms = (
        task_farming_for_job(job) if task_farming_for_job is not None else (False, None)
    )
    flags = retrieve_flags_for_job(job) if retrieve_flags_for_job is not None else {}
    return RetrievalFlags(
        retrieve_den_fmt=bool(flags.get("retrieve_den_fmt", False)),
        retrieve_out_cell=bool(flags.get("retrieve_out_cell", False)),
        retrieve_phonon=bool(flags.get("retrieve_phonon", False)),
        retrieve_ts=bool(flags.get("retrieve_ts", False)),
        retrieve_magres=bool(flags.get("retrieve_magres", False)),
        task_farming_enabled=task_farming_enabled,
        num_farms=num_farms,
    )

"""Shared job execution lifecycle orchestration."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any
from uuid import UUID

from mufinder.domain.enums import ACTIVE_JOB_STATUSES, FailureReason, JobStatus
from mufinder.domain.models import CalculationRun, Job, Result, utc_now
from mufinder.services.cluster_runner import CheckTarget
from mufinder.services.failure_diagnosis import (
    RESULT_NOT_RECORDED_PREFIX,
    FailureDiagnosis,
    diagnose_failure,
    failure_reason_label,
    find_failure_log_files,
)
from mufinder.castep_io.output_parser import CastepResult
from mufinder.services.job_artifacts import ParsedJobArtifacts, RetrievalFlags
from mufinder.services.job_evidence import local_job_evidence
from mufinder.services.job_findings import record_job_finding
from mufinder.services.job_monitoring import is_cluster_job, job_staleness
from mufinder.services.job_reconcile import (
    ENDED_JOB_STATUSES,
    Decision,
    JobEvidence,
    OutputState,
    SchedulerState,
    reconcile,
)
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
    #: Equal processes per node for addqueue; ``None`` leaves placement to the
    #: scheduler.
    processes_per_node: int | None = None
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
class FinishedOutput:
    """Complete CASTEP output found for a job whose result is still to be recorded.

    ``artifacts`` is ``None`` when the secondary outputs have not been parsed yet.
    """

    parsed: CastepResult
    artifacts: ParsedJobArtifacts | None = None


@dataclass(frozen=True, slots=True)
class LifecycleUpdate:
    """Result of a runner lifecycle notification or of applying evidence.

    ``finished`` is set when the evidence shows complete output: the job has
    not changed yet, and the caller records the result and completes it.
    """

    job: Job | None = None
    changed: bool = False
    journal_changed: bool = False
    finished: FinishedOutput | None = None
    #: Only the job's last-checked note changed.
    stamped: bool = False


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
        retrieval_for_job: Callable[[Job], RetrievalFlags] | None = None,
        remote_directory_for_job: Callable[[Job], str] | None = None,
    ) -> None:
        self._document_getter = document_getter
        self._project_service = project_service
        self._local_runner = local_runner
        self._cluster_runner = cluster_runner
        self._job_output_directory = job_output_directory
        self._job_seed_name = job_seed_name
        self._active_run_by_job = active_run_by_job if active_run_by_job is not None else {}
        self._retrieval_for_job = retrieval_for_job
        self._remote_directory_for_job = remote_directory_for_job
        # The marker each active job's last check left, when it could not
        # confirm the job. Derived, never saved: the next check rebuilds it.
        self._markers: dict[UUID, str] = {}
        # Cluster jobs that evidence changed since the last check summary, in
        # the order they changed.
        self._changed_by_checks: dict[UUID, None] = {}

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
        unchanged because their monitor metadata is stamped from each check
        by :meth:`apply_evidence`.
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

    def fail_unrecorded_finish(
        self, job_id: UUID, error: Exception, *, results_before: frozenset[UUID]
    ) -> LifecycleUpdate:
        """Fail a finished job whose result could not be recorded.

        A runner reports a finished job once and then lets go of it, so a job
        still active after ``error`` would have nothing left to end it. Such a
        job fails with the reason, the results added since ``results_before``
        are removed, and no run keeps pointing at one of them. A job that had
        already completed keeps its result and is left as it is.
        """

        document = self._document_getter()
        job = self._job_for_id(job_id)
        if document is None or job is None or job.status not in ACTIVE_JOB_STATUSES:
            return LifecycleUpdate(job=job)
        removed = set(document.results) - results_before
        for result_id in removed:
            del document.results[result_id]
        for run in document.runs.values():
            if run.result_id in removed:
                run.result_id = None
        return self.fail_job(job_id, f"{RESULT_NOT_RECORDED_PREFIX}: {error}")

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

    @staticmethod
    def _stamp_monitor_metadata(job: Job, *, source: str, message: str) -> None:
        """Record when, from where, and with what observation a job was last seen.

        These are bookkeeping fields (not status transitions), so the
        controller mutates them directly per the state-authority contract.
        """

        job.last_checked_at = utc_now()
        job.last_status_source = source
        job.last_status_message = message

    def check_local_jobs(self) -> list[LifecycleUpdate]:
        """Reconcile every active local job with the local runner and its output folder.

        A local job the runner holds no process for has nothing left to report
        on it: complete output finishes it, and anything else stops it. Returns
        the updates that changed a job or found finished output.
        """

        document = self._document_getter()
        if document is None:
            return []
        active = [
            job
            for job in document.jobs.values()
            if job.status in ACTIVE_JOB_STATUSES and not is_cluster_job(job)
        ]
        if not active:
            return []
        pending_ids = self._local_runner.pending_job_ids()
        active_ids = self._local_runner.active_job_ids()
        now = utc_now()
        updates: list[LifecycleUpdate] = []
        for job in active:
            evidence = local_job_evidence(
                job,
                pending_ids=pending_ids,
                active_ids=active_ids,
                output_dir=self._job_output_directory(job),
                seed_name=self._job_seed_name(job),
                now=now,
            )
            update = self.apply_evidence(job.id, evidence)
            if update.changed or update.finished is not None:
                updates.append(update)
        return updates

    def apply_evidence(self, job_id: UUID, evidence: JobEvidence) -> LifecycleUpdate:
        """Apply the state :func:`~mufinder.services.job_reconcile.reconcile` gives ``evidence``.

        Every transition goes through the ``ProjectService`` primitives, with
        the job's run, journal and monitor stamps kept in step. Complete output
        is returned as ``finished`` instead of completing the job, because the
        caller builds the result first. Evidence about a cluster job is told
        here whether the job is still being submitted, which only this side
        knows.
        """

        job = self._job_for_id(job_id)
        if job is None:
            return LifecycleUpdate()
        cluster = is_cluster_job(job)
        if cluster:
            evidence = replace(evidence, submitting=self._is_submitting(job_id))
        decision = reconcile(job, evidence)
        if decision is None:
            return LifecycleUpdate(job=job)
        stamped = cluster and self._note_check(job, evidence, decision)
        if decision.status == job.status:
            return LifecycleUpdate(job=job, stamped=stamped)
        update = self._apply_decision(job, evidence, decision)
        if cluster and (update.changed or update.finished is not None):
            self._changed_by_checks[job.id] = None
        if update.changed or update.finished is not None:
            return update
        return replace(update, stamped=stamped)

    def take_check_summary(self, checked: int) -> str:
        """Return one status-bar line for what cluster checks changed, and start afresh.

        ``checked`` is how many jobs the pass that just ended looked at. The
        line covers every cluster job that evidence has changed since the last
        call, described by the state it is in now: one job is named, several
        are counted by outcome. Empty when nothing changed.
        """

        document = self._document_getter()
        changed, self._changed_by_checks = list(self._changed_by_checks), {}
        if document is None:
            return ""
        outcomes = [
            (job, outcome)
            for job in (document.jobs.get(job_id) for job_id in changed)
            if job is not None and (outcome := _check_outcome(job))
        ]
        if not outcomes:
            return ""
        if len(outcomes) == 1:
            job, outcome = outcomes[0]
            if job.status == JobStatus.COMPLETED:
                return f"Completed cluster job {job.label}"
            return f"Cluster job {outcome}: {job.label}"
        counts: dict[str, int] = {}
        for _job, outcome in outcomes:
            counts[outcome] = counts.get(outcome, 0) + 1
        looked_at = max(checked, len(outcomes))
        parts = ", ".join(f"{count} {outcome}" for outcome, count in counts.items())
        return f"Checked {looked_at} cluster jobs: {parts}"

    def _apply_decision(
        self, job: Job, evidence: JobEvidence, decision: Decision
    ) -> LifecycleUpdate:
        """Move ``job`` to ``decision.status``, which differs from its own."""

        target = decision.status
        ended = job.status in ENDED_JOB_STATUSES
        cluster = is_cluster_job(job)
        source = _evidence_source(evidence) if cluster else "reconciliation"
        if target == JobStatus.COMPLETED:
            if evidence.parsed is None:
                return LifecycleUpdate(job=job)
            if ended and not self._reactivate(job, JobStatus.RETRIEVING):
                return LifecycleUpdate(job=job)
            return LifecycleUpdate(
                job=job,
                changed=ended,
                finished=FinishedOutput(evidence.parsed, evidence.artifacts),
            )
        if target in {JobStatus.QUEUED, JobStatus.RUNNING} and ended:
            return LifecycleUpdate(job=job, changed=self._reactivate(job, target))
        if target == JobStatus.QUEUED:
            if cluster:
                return self._mark_cluster_queued(job)
            return self.mark_local_status_changed(job.id, JobStatus.QUEUED.value)
        if target == JobStatus.RUNNING:
            return self.mark_started(job.id)
        if target == JobStatus.RETRIEVING:
            return self._mark_retrieving(job)
        if target == JobStatus.FAILED:
            return self.fail_job(
                job.id,
                decision.message or "",
                source=source,
                status_message=decision.observation or None,
                reason=decision.failure_reason,
            )
        if target == JobStatus.STOPPED:
            return self._stop_job(job, decision, source=source, mark_started=not cluster)
        if target == JobStatus.CANCELLED:
            return self.cancel_job(job.id)
        return LifecycleUpdate(job=job)

    def _note_check(self, job: Job, evidence: JobEvidence, decision: Decision) -> bool:
        """Record what a check of cluster ``job`` saw; return whether its note changed."""

        if decision.marker:
            self._markers[job.id] = decision.marker
        else:
            self._markers.pop(job.id, None)
        if evidence.scheduler in {None, SchedulerState.UNAVAILABLE}:
            return False
        message = decision.observation or decision.marker
        if not message:
            return False
        job.last_checked_at = evidence.checked_at
        job.last_status_source = _evidence_source(evidence)
        job.last_status_message = message
        return True

    def _is_submitting(self, job_id: UUID) -> bool:
        return bool(self._cluster_runner.is_submitting(job_id))

    def _mark_cluster_queued(self, job: Job) -> LifecycleUpdate:
        try:
            self._project_service.mark_job_queued(job, remote_job_id=job.remote_job_id)
        except ValueError:
            return LifecycleUpdate(job=job)
        run = self.run_for_job(job.id)
        if run is not None:
            run.status = JobStatus.QUEUED
        return LifecycleUpdate(job=job, changed=True)

    def _mark_retrieving(self, job: Job) -> LifecycleUpdate:
        # Only a running job has a transition to retrieving; a job that
        # finished between two checks keeps its status while its output is read.
        if job.status != JobStatus.RUNNING:
            return LifecycleUpdate(job=job)
        try:
            self._project_service.mark_job_retrieving(job)
        except ValueError:
            return LifecycleUpdate(job=job)
        run = self.run_for_job(job.id)
        if run is not None:
            run.status = JobStatus.RETRIEVING
        return LifecycleUpdate(job=job, changed=True)

    def _reactivate(self, job: Job, status: JobStatus) -> bool:
        """Return an ended job to ``status`` under a fresh run; say whether it happened.

        The ended run keeps its outcome: whatever follows is recorded on the
        new run, never over the old one.
        """

        document = self._document_getter()
        if document is None:
            return False
        try:
            self._project_service.reactivate_job(job, status)
        except ValueError:
            return False
        run = self._create_active_run(document, job, status)
        if status == JobStatus.RUNNING:
            run.started_at = utc_now()
        return True

    def _stop_job(
        self, job: Job, decision: Decision, *, source: str, mark_started: bool
    ) -> LifecycleUpdate:
        """Stop ``job`` as ``decision`` says, closing its run and recording the stop."""

        message = decision.message or ""
        try:
            self._project_service.stop_job(job, message, reason=decision.failure_reason)
        except ValueError:
            return LifecycleUpdate(job=job)

        self._stamp_monitor_metadata(job, source=source, message=decision.observation or message)
        run = self.run_for_job(job.id)
        if run is not None and run.status not in ACTIVE_JOB_STATUSES:
            # The job's latest run already ended; it is not this run to close.
            run = None
        self._close_active_run(
            job,
            run,
            JobStatus.STOPPED,
            "job_stopped",
            error_message=message,
            mark_started=mark_started,
        )
        return LifecycleUpdate(job=job, changed=True, journal_changed=True)

    # ------------------------------------------------------------------
    # Cluster checks
    # ------------------------------------------------------------------
    def check_target(self, job: Job, *, on_request: bool = False) -> CheckTarget | None:
        """Return where to look for cluster ``job``, or ``None`` when there is nowhere.

        A job needs a local output folder and either a scheduler job ID or a
        remote folder. A job with an ID but no recorded folder gets the folder
        its cluster profile would have given it.
        """

        if not is_cluster_job(job):
            return None
        remote_job_id = (job.remote_job_id or "").strip()
        remote_dir = (job.remote_directory or "").strip()
        if remote_job_id and not remote_dir and self._remote_directory_for_job is not None:
            remote_dir = self._remote_directory_for_job(job).strip()
            if remote_dir:
                job.remote_directory = remote_dir
        if not remote_dir:
            return None
        output_dir = self._job_output_directory(job)
        if output_dir is None:
            return None
        return CheckTarget(
            job_id=job.id,
            remote_job_id=remote_job_id,
            remote_dir=remote_dir,
            seed_name=self._job_seed_name(job),
            local_output_dir=output_dir,
            cluster_name=job.execution_target.strip(),
            retrieval=(
                self._retrieval_for_job(job)
                if self._retrieval_for_job is not None
                else RetrievalFlags()
            ),
            on_request=on_request,
        )

    def check_targets_for(self, job_ids: Sequence[UUID]) -> list[CheckTarget]:
        """Return the check targets of the jobs in ``job_ids`` that have one."""

        document = self._document_getter()
        if document is None:
            return []
        jobs = (document.jobs.get(job_id) for job_id in job_ids)
        targets = (self.check_target(job) for job in jobs if job is not None)
        return [target for target in targets if target is not None]

    def read_cluster_progress(self, job_ids: Sequence[UUID]) -> list[UUID]:
        """Ask the cluster worker for jobs' geometry progress; return the jobs it cannot ask about.

        A job with nowhere to look (no remote folder, no local output folder,
        or no longer in the project) gets no answer from the worker, so the
        caller must not wait for one.
        """

        targets = self.check_targets_for(job_ids)
        if targets:
            self._cluster_runner.read_geometry_progress(targets)
        asked = {target.job_id for target in targets}
        return [job_id for job_id in job_ids if job_id not in asked]

    def check_cluster_jobs(
        self,
        job_ids: Sequence[UUID] | None = None,
        *,
        on_request: bool = False,
        skip_if_busy: bool = False,
    ) -> int:
        """Ask the connected cluster about this project's jobs; return how many.

        Without ``job_ids`` every active job on the connected cluster is
        looked at. With them, only those jobs, and ``on_request`` also allows
        ended ones. A job still being submitted is left out: the scheduler has
        not answered for it yet. A job with no scheduler job ID is left out
        unless the user asked: there is no one to ask about it, and its folder
        is looked at only on request. Nothing is asked while no cluster is
        connected. ``skip_if_busy`` drops the request while an earlier check is
        still running, which is what the once-a-minute check wants. The answers
        arrive as evidence for :meth:`apply_evidence`.
        """

        document = self._document_getter()
        if document is None or not self._cluster_runner.is_connected:
            return 0
        wanted = None if job_ids is None else set(job_ids)
        statuses = (
            ACTIVE_JOB_STATUSES | ENDED_JOB_STATUSES
            if on_request and wanted is not None
            else ACTIVE_JOB_STATUSES
        )
        targets: list[CheckTarget] = []
        for job in document.jobs.values():
            if wanted is not None and job.id not in wanted:
                continue
            if job.status not in statuses or not self._connected_to_cluster_of(job):
                continue
            if self._is_submitting(job.id):
                continue
            target = self.check_target(job, on_request=on_request)
            if target is not None and (on_request or target.remote_job_id):
                targets.append(target)
        if targets or wanted is None:
            self._cluster_runner.check_jobs(targets, skip_if_busy=skip_if_busy)
        return len(targets)

    def can_cancel_on_cluster(self, job: Job) -> bool:
        """Return whether cancelling ``job`` can be put to its cluster.

        The scheduler is asked only about a job with a scheduler job ID on the
        connected cluster. A job still being submitted counts: its request is
        already with the worker.
        """

        if self._is_submitting(job.id):
            return True
        return bool((job.remote_job_id or "").strip()) and self._connected_to_cluster_of(job)

    def can_check_now(self, job: Job) -> bool:
        """Return whether ``Check now`` has somewhere to look for ``job``.

        A cluster job that is active or ended needs a remote folder and a
        connection to its cluster. A local job qualifies once it has ended:
        while it is active the local runner reports on it.
        """

        if is_cluster_job(job):
            return (
                job.status in ACTIVE_JOB_STATUSES | ENDED_JOB_STATUSES
                and bool((job.remote_directory or "").strip())
                and self._connected_to_cluster_of(job)
                and not self._is_submitting(job.id)
            )
        return job.status in ENDED_JOB_STATUSES and self._job_output_directory(job) is not None

    def check_now(self, job_ids: Sequence[UUID]) -> tuple[list[LifecycleUpdate], int]:
        """Check the jobs in ``job_ids`` because the user asked.

        Local jobs are decided at once from their output folder and returned as
        updates. Cluster jobs are asked about on the worker; the count of those
        is returned and their evidence follows.
        """

        document = self._document_getter()
        if document is None:
            return [], 0
        updates: list[LifecycleUpdate] = []
        cluster_ids: list[UUID] = []
        now = utc_now()
        for job_id in dict.fromkeys(job_ids):
            job = document.jobs.get(job_id)
            if job is None or not self.can_check_now(job):
                continue
            if is_cluster_job(job):
                cluster_ids.append(job_id)
                continue
            evidence = local_job_evidence(
                job,
                pending_ids=(),
                active_ids=(),
                output_dir=self._job_output_directory(job),
                seed_name=self._job_seed_name(job),
                now=now,
                on_request=True,
            )
            updates.append(self.apply_evidence(job_id, evidence))
        requested = self.check_cluster_jobs(cluster_ids, on_request=True) if cluster_ids else 0
        return updates, requested

    def staleness(self, job: Job, *, now: Any) -> str | None:
        """Return why ``job``'s displayed status may be out of date, or ``None``."""

        return job_staleness(
            job,
            now=now,
            cluster_connected=self._connected_to_cluster_of(job),
            marker=self._markers.get(job.id),
            submitting=self._is_submitting(job.id),
        )

    def _connected_to_cluster_of(self, job: Job) -> bool:
        if not is_cluster_job(job) or not self._cluster_runner.is_connected:
            return False
        connected = (self._cluster_runner.connected_cluster_name or "").strip().casefold()
        return connected == job.execution_target.strip().casefold()

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


def _evidence_source(evidence: JobEvidence) -> str:
    """Return the monitor source of cluster ``evidence``: the queue, or the output folder."""

    return "scheduler_poll" if evidence.output == OutputState.NOT_LOOKED else "retrieval"


def _check_outcome(job: Job) -> str:
    """Return how a check summary describes ``job``'s state, or ``""`` to leave it out."""

    if job.status == JobStatus.COMPLETED:
        return "completed"
    if job.status == JobStatus.RUNNING:
        return "started running"
    if job.status == JobStatus.QUEUED:
        return "queued"
    if job.status == JobStatus.CANCELLED:
        return "cancelled"
    if job.status == JobStatus.STOPPED:
        if job.failure_reason == FailureReason.WALLTIME:
            return "stopped at its walltime limit"
        return "stopped"
    if job.status == JobStatus.FAILED:
        label = failure_reason_label(job.failure_reason)
        return f"failed ({label})" if label else "failed"
    # Retrieving is a step on the way; the pass ends with what came of it.
    return ""


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

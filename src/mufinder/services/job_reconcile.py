"""Decide a job's state from what was observed about it.

:func:`reconcile` is a pure function: the same job and the same evidence always
give the same decision, whatever led to the check. Evidence is what can be seen
of a job at one moment: whether a local process holds it, what the scheduler
says about it, and what its output folder contains. Nothing here reads a file,
talks to a cluster or changes a job; the caller gathers the evidence and applies
the decision.

Two rules shape the table. MuFinder never ends a cluster job because it lost
track of it: with no scheduler entry and no output the job stays as it is and
carries a marker instead. Valid CASTEP output completes a job, however the
tracking went.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import EndReason, JobActivity, JobStatus
from mufinder.domain.models import Job
from mufinder.services.failure_diagnosis import RETRIEVAL_FAILURE_PREFIX, failure_reason_summary
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.job_monitoring import NO_JOB_ID_MARKER, is_cluster_job

#: Message of a local job that no process is running and whose output is not complete.
NO_LOCAL_PROCESS_MESSAGE = (
    "MuFinder has no local process for this job: it was closed while the job ran, "
    "or the process ended without reporting. Reset or resubmit it."
)
#: Message used when CASTEP output ends without a completion summary and names no error.
INCOMPLETE_OUTPUT_MESSAGE = "CASTEP did not report successful completion."

__all__ = [
    "INCOMPLETE_OUTPUT_MESSAGE",
    "NO_JOB_ID_MARKER",
    "NO_LOCAL_PROCESS_MESSAGE",
    "Decision",
    "JobEvidence",
    "OutputState",
    "ProcessState",
    "SchedulerState",
    "reconcile",
    "wants_output",
]


class ProcessState(StrEnum):
    """Whether the local runner holds a process for a local job.

    A process the runner sees end is not evidence for :func:`reconcile`: the
    runner reads the exit code and the output itself and reports the outcome.
    """

    PENDING = "pending"
    RUNNING = "running"
    #: The runner holds no process for the job and reported no outcome for it.
    NONE = "none"


class SchedulerState(StrEnum):
    """What the cluster's queue and accounting say about a job."""

    #: The queue command failed, so the scheduler was not heard from.
    UNAVAILABLE = "unavailable"
    #: The job has no scheduler job ID to ask about.
    NO_JOB_ID = "no_job_id"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    #: Neither the queue nor accounting lists the job.
    NOT_LISTED = "not_listed"


class OutputState(StrEnum):
    """What the job's output folder holds."""

    NOT_LOOKED = "not_looked"
    #: CASTEP output that parses and reports completion.
    COMPLETE = "complete"
    #: CASTEP output from this run that did not reach completion.
    INCOMPLETE = "incomplete"
    #: Output that was fetched but could not be parsed.
    UNREADABLE = "unreadable"
    #: The folder holds no CASTEP output written by this run.
    ABSENT = "absent"
    #: The folder could not be read.
    UNREACHABLE = "unreachable"


_LISTED_UNFINISHED = frozenset({SchedulerState.PENDING, SchedulerState.RUNNING})
_REPORTED_FINISHED = frozenset(
    {SchedulerState.COMPLETED, SchedulerState.TIMED_OUT, SchedulerState.FAILED}
)


@dataclass(frozen=True, slots=True)
class JobEvidence:
    """What was observed about one job at ``checked_at``.

    A local job carries ``process``; a cluster job carries ``scheduler``.
    ``remote_job_id`` and ``remote_directory`` name the submission the cluster
    evidence is about, so evidence gathered for an earlier submission of the
    same job can be told apart and dropped.
    """

    checked_at: datetime
    #: Local jobs: what the local runner holds for the job.
    process: ProcessState | None = None
    #: Cluster jobs: a submit request for this job has not answered yet.
    submitting: bool = False
    scheduler: SchedulerState | None = None
    #: The scheduler's own name for the state, or why it was not heard from.
    scheduler_detail: str = ""
    #: The state came from accounting because the queue did not list the job.
    from_accounting: bool = False
    output: OutputState = OutputState.NOT_LOOKED
    #: CASTEP's error message, or why the output could not be read or parsed.
    output_detail: str = ""
    #: The scheduler log of an incomplete run shows a walltime kill.
    walltime_seen: bool = False
    parsed: CastepResult | None = None
    artifacts: ParsedJobArtifacts | None = None
    remote_job_id: str = ""
    remote_directory: str = ""
    #: The user asked for this check. Ended jobs change only on such evidence.
    on_request: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    """The state the evidence supports for a job.

    ``status`` and ``activity`` equal the job's own when nothing changes.
    ``end_reason`` is ``None`` for an ended job whose cause the caller
    diagnoses from the job's logs. ``marker`` is the note shown beside an
    active job that could not be confirmed; it is never stored. Whether a
    job's output has to be looked at is :func:`wants_output`'s answer, from the
    scheduler state alone.
    """

    status: JobStatus
    #: What an active job is doing; ``None`` for any other state.
    activity: JobActivity | None = None
    end_reason: EndReason | None = None
    #: The job's error message when the decision ends it.
    message: str | None = None
    #: One line on what was seen, for the job's "last checked" note.
    observation: str = ""
    marker: str | None = None


def wants_output(scheduler: SchedulerState | None, *, on_request: bool = False) -> bool:
    """Return whether a job's output must be looked at, given what the scheduler says.

    Output is never looked at while the scheduler lists the job as pending or
    running, or when the scheduler was not heard from. After a cancellation,
    and for a job with no scheduler job ID, it is looked at only when the user
    asked for the check: nothing else says that output in such a job's folder
    belongs to this run.
    """

    if scheduler is None or scheduler in _LISTED_UNFINISHED:
        return False
    if scheduler == SchedulerState.UNAVAILABLE:
        return False
    if scheduler in {SchedulerState.CANCELLED, SchedulerState.NO_JOB_ID}:
        return on_request
    return True


def reconcile(job: Job, evidence: JobEvidence) -> Decision | None:
    """Return the state ``evidence`` supports for ``job``.

    ``None`` means the evidence is about another submission of the job and
    says nothing about this one. A ``defined`` or ``completed`` job is never
    changed, and an ended job only by evidence the user asked for.
    """

    cluster = is_cluster_job(job)
    if cluster and not _about_this_submission(job, evidence):
        return None
    if job.status == JobStatus.ACTIVE:
        return _active_cluster(job, evidence) if cluster else _active_local(job, evidence)
    if job.status == JobStatus.ENDED and evidence.on_request:
        return _ended(job, evidence, cluster=cluster)
    return _unchanged(job)


def _about_this_submission(job: Job, evidence: JobEvidence) -> bool:
    return evidence.remote_job_id == (job.remote_job_id or "").strip() and (
        evidence.remote_directory == (job.remote_directory or "").strip()
    )


def _unchanged(job: Job, *, observation: str = "", marker: str | None = None) -> Decision:
    return Decision(job.status, job.activity, observation=observation, marker=marker)


def _active(activity: JobActivity, observation: str = "") -> Decision:
    return Decision(JobStatus.ACTIVE, activity, observation=observation)


def _active_local(job: Job, evidence: JobEvidence) -> Decision:
    process = evidence.process
    if process == ProcessState.PENDING:
        return _active(JobActivity.QUEUED, "queued locally")
    if process == ProcessState.RUNNING:
        return _active(JobActivity.RUNNING, "process started")
    if process is None:
        return _unchanged(job)
    if evidence.output == OutputState.COMPLETE:
        return Decision(JobStatus.COMPLETED, observation="output found complete")
    if evidence.output == OutputState.NOT_LOOKED:
        return _unchanged(job)
    return Decision(
        JobStatus.ENDED,
        end_reason=EndReason.INTERRUPTED,
        message=NO_LOCAL_PROCESS_MESSAGE,
        observation="no local process is running this job",
    )


def _active_cluster(job: Job, evidence: JobEvidence) -> Decision:
    scheduler = evidence.scheduler
    if evidence.submitting or scheduler is None:
        return _unchanged(job)
    if scheduler == SchedulerState.NO_JOB_ID:
        # With no scheduler answer, nothing ties the folder's output to this
        # run. Only a check the user asked for adopts complete output.
        if evidence.on_request and evidence.output == OutputState.COMPLETE:
            return Decision(JobStatus.COMPLETED, observation="output retrieved and parsed")
        return _unchanged(job, marker=NO_JOB_ID_MARKER)
    if scheduler == SchedulerState.UNAVAILABLE:
        return _unchanged(job, marker=f"could not be checked: {evidence.scheduler_detail}")
    if scheduler == SchedulerState.PENDING:
        return _active(JobActivity.QUEUED, _scheduler_observation(evidence, "QUEUED"))
    if scheduler == SchedulerState.RUNNING:
        return _active(JobActivity.RUNNING, _scheduler_observation(evidence, "RUNNING"))
    if scheduler == SchedulerState.CANCELLED and evidence.output != OutputState.COMPLETE:
        # Output is looked at after a cancellation only when the user asks;
        # complete output then completes the job like any other.
        return Decision(
            JobStatus.ENDED,
            end_reason=EndReason.CANCELLED,
            observation=_scheduler_observation(evidence, "CANCELLED"),
        )

    # The scheduler no longer runs the job, or cannot say: the output decides.
    output = evidence.output
    if output == OutputState.NOT_LOOKED:
        if scheduler in _REPORTED_FINISHED:
            return _active(JobActivity.RETRIEVING)
        return _unchanged(job)
    if output == OutputState.UNREACHABLE:
        return _unchanged(
            job,
            observation=_queue_note(evidence, "output retrieval failed"),
            marker=f"output could not be read: {evidence.output_detail}",
        )
    if output == OutputState.COMPLETE:
        return Decision(
            JobStatus.COMPLETED, observation=_queue_note(evidence, "output retrieved and parsed")
        )
    if scheduler == SchedulerState.TIMED_OUT:
        return _walltime_stop(_scheduler_observation(evidence, "TIMEOUT"))
    if scheduler == SchedulerState.FAILED:
        return Decision(
            JobStatus.ENDED,
            message=f"Scheduler reported the job as {evidence.scheduler_detail}.",
            observation=_scheduler_observation(evidence, evidence.scheduler_detail),
        )
    if output == OutputState.INCOMPLETE:
        if evidence.walltime_seen:
            return _walltime_stop(_queue_note(evidence, "output incomplete; stopped (walltime)"))
        return Decision(
            JobStatus.ENDED,
            message=evidence.output_detail or INCOMPLETE_OUTPUT_MESSAGE,
            observation=_queue_note(evidence, "output incomplete; job failed"),
        )
    if output == OutputState.UNREADABLE:
        return Decision(
            JobStatus.ENDED,
            end_reason=EndReason.OUTPUT_UNAVAILABLE,
            message=f"{RETRIEVAL_FAILURE_PREFIX}: {evidence.output_detail}",
            observation=_queue_note(evidence, "output parsing failed"),
        )
    # No output from this run.
    if scheduler == SchedulerState.COMPLETED:
        return Decision(
            JobStatus.ENDED,
            end_reason=EndReason.OUTPUT_UNAVAILABLE,
            message=(
                "The scheduler reports the job finished, but "
                f"{evidence.remote_directory} holds no CASTEP output from this run."
            ),
            observation=_queue_note(evidence, "no output from this run"),
        )
    return _unchanged(
        job,
        observation=_queue_note(evidence, "not in scheduler accounting; no output from this run"),
        marker=(
            f"lost track: the scheduler does not list job {evidence.remote_job_id} "
            "and its folder holds no output from this run"
        ),
    )


def _ended(job: Job, evidence: JobEvidence, *, cluster: bool) -> Decision:
    """Decide an ended job the user asked to have checked."""

    if cluster and evidence.scheduler == SchedulerState.PENDING:
        return _active(JobActivity.QUEUED, _scheduler_observation(evidence, "QUEUED") + " again")
    if cluster and evidence.scheduler == SchedulerState.RUNNING:
        return _active(JobActivity.RUNNING, _scheduler_observation(evidence, "RUNNING") + " again")
    if evidence.output == OutputState.COMPLETE:
        return Decision(JobStatus.COMPLETED, observation="output retrieved and parsed")
    if evidence.output == OutputState.NOT_LOOKED:
        return _unchanged(job)
    return _unchanged(job, observation="checked; no complete output found")


def _walltime_stop(observation: str) -> Decision:
    return Decision(
        JobStatus.ENDED,
        end_reason=EndReason.WALLTIME,
        message=failure_reason_summary(EndReason.WALLTIME),
        observation=observation,
    )


def _queue_note(evidence: JobEvidence, detail: str) -> str:
    """Return ``detail``, saying first when the queue did not list the job."""

    missing = evidence.from_accounting or evidence.scheduler == SchedulerState.NOT_LISTED
    return f"missing from scheduler queue; {detail}" if missing else detail


def _scheduler_observation(evidence: JobEvidence, state: str) -> str:
    source = "scheduler accounting" if evidence.from_accounting else "scheduler"
    return _queue_note(evidence, f"{source} reports {state}")

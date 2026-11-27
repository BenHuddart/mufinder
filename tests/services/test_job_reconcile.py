"""The decision table of ``services.job_reconcile.reconcile``, one case per row."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from mufinder.domain.enums import EndReason, JobActivity, JobStatus
from mufinder.services.failure_diagnosis import RETRIEVAL_FAILURE_PREFIX, failure_reason_summary
from mufinder.services.job_reconcile import (
    INCOMPLETE_OUTPUT_MESSAGE,
    NO_JOB_ID_MARKER,
    NO_LOCAL_PROCESS_MESSAGE,
    Decision,
    JobEvidence,
    OutputState,
    ProcessState,
    SchedulerState,
    reconcile,
    wants_output,
)
from mufinder.services.job_monitoring import job_status_word
from tests.support.jobs import make_job

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
REMOTE_ID = "4127731"
REMOTE_DIR = "/work/demo/cof2/cutoff_500"

A = JobStatus
S = SchedulerState
O = OutputState  # noqa: E741 - reads as a column heading in the tables below
P = ProcessState

#: The status words of an active job and of an ended one.
ACTIVE_WORDS = [activity.value for activity in JobActivity]
ENDED_WORDS = ["cancelled", "failed", "stopped"]


def local_job(status: JobStatus):
    return make_job(status=status, execution_target="local")


def cluster_job(status: JobStatus, *, remote_job_id: str | None = REMOTE_ID):
    return make_job(
        status=status,
        execution_target="SCARF",
        remote_job_id=remote_job_id,
        remote_directory=REMOTE_DIR,
    )


def local_evidence(process: ProcessState | None, output: OutputState = O.NOT_LOOKED, **fields):
    return JobEvidence(checked_at=NOW, process=process, output=output, **fields)


def cluster_evidence(
    scheduler: SchedulerState | None, output: OutputState = O.NOT_LOOKED, **fields
):
    fields.setdefault("remote_job_id", REMOTE_ID)
    fields.setdefault("remote_directory", REMOTE_DIR)
    return JobEvidence(checked_at=NOW, scheduler=scheduler, output=output, **fields)


# --- Local job, active -------------------------------------------------------

LOCAL_ACTIVE_ROWS = [
    # id, job status, process, output, expected status, reason, message
    ("L1 pending from submitting", "submitting", P.PENDING, O.NOT_LOOKED, "queued", None, None),
    ("L1 pending stays queued", "queued", P.PENDING, O.NOT_LOOKED, "queued", None, None),
    ("L2 running from submitting", "submitting", P.RUNNING, O.NOT_LOOKED, "running", None, None),
    ("L2 running from queued", "queued", P.RUNNING, O.NOT_LOOKED, "running", None, None),
    ("L3 no process, complete", "running", P.NONE, O.COMPLETE, A.COMPLETED, None, None),
    ("L3 no process, complete, queued", "queued", P.NONE, O.COMPLETE, A.COMPLETED, None, None),
    (
        "L4 no process, incomplete",
        "running",
        P.NONE,
        O.INCOMPLETE,
        "stopped",
        EndReason.INTERRUPTED,
        NO_LOCAL_PROCESS_MESSAGE,
    ),
    (
        "L4 no process, absent",
        "queued",
        P.NONE,
        O.ABSENT,
        "stopped",
        EndReason.INTERRUPTED,
        NO_LOCAL_PROCESS_MESSAGE,
    ),
    (
        "L4 no process, unreadable",
        "submitting",
        P.NONE,
        O.UNREADABLE,
        "stopped",
        EndReason.INTERRUPTED,
        NO_LOCAL_PROCESS_MESSAGE,
    ),
]


@pytest.mark.parametrize(
    ("job_status", "process", "output", "status", "reason", "message"),
    [row[1:] for row in LOCAL_ACTIVE_ROWS],
    ids=[row[0] for row in LOCAL_ACTIVE_ROWS],
)
def test_active_local_job(job_status, process, output, status, reason, message) -> None:
    evidence = local_evidence(process, output)

    decision = reconcile(local_job(job_status), evidence)

    assert decision is not None
    assert (job_status_word(decision), decision.end_reason, decision.message) == (
        status,
        reason,
        message,
    )
    assert decision.marker is None


def test_local_job_with_no_process_asks_for_its_output_before_it_is_stopped() -> None:
    decision = reconcile(local_job("running"), local_evidence(P.NONE))

    assert decision == Decision(JobStatus.ACTIVE, JobActivity.RUNNING)


# --- Cluster job, active -----------------------------------------------------

LOST_TRACK_MARKER = (
    f"lost track: the scheduler does not list job {REMOTE_ID} "
    "and its folder holds no output from this run"
)
NO_OUTPUT_MESSAGE = (
    f"The scheduler reports the job finished, but {REMOTE_DIR} holds no CASTEP output "
    "from this run."
)
WALLTIME_MESSAGE = failure_reason_summary(EndReason.WALLTIME)
CASTEP_MESSAGE = "SCF did not converge"
SFTP_ERROR = "connection reset"

CLUSTER_ACTIVE_ROWS = [
    # id, job status, scheduler, output, extra evidence,
    #   expected status, reason, message, marker
    ("C3 queue command failed", "queued", S.UNAVAILABLE, O.NOT_LOOKED,
     {"scheduler_detail": "squeue: timed out"},
     "queued", None, None, "could not be checked: squeue: timed out"),
    ("C4 pending from submitting", "submitting", S.PENDING, O.NOT_LOOKED, {},
     "queued", None, None, None),
    ("C4 pending stays queued", "queued", S.PENDING, O.NOT_LOOKED, {},
     "queued", None, None, None),
    ("C4 pending from running", "running", S.PENDING, O.NOT_LOOKED, {},
     "queued", None, None, None),
    ("C5 running from submitting", "submitting", S.RUNNING, O.NOT_LOOKED, {},
     "running", None, None, None),
    ("C5 running from queued", "queued", S.RUNNING, O.NOT_LOOKED, {},
     "running", None, None, None),
    ("C5 running from retrieving", "retrieving", S.RUNNING, O.NOT_LOOKED, {},
     "running", None, None, None),
    ("C6 cancelled", "running", S.CANCELLED, O.NOT_LOOKED, {},
     "cancelled", EndReason.CANCELLED, None, None),
    ("C7 completed, not looked", "running", S.COMPLETED, O.NOT_LOOKED, {},
     "retrieving", None, None, None),
    ("C7 timed out, not looked", "running", S.TIMED_OUT, O.NOT_LOOKED, {},
     "retrieving", None, None, None),
    ("C7 failed, not looked", "queued", S.FAILED, O.NOT_LOOKED, {},
     "retrieving", None, None, None),
    ("C7 not listed, not looked", "queued", S.NOT_LISTED, O.NOT_LOOKED, {},
     "queued", None, None, None),
    ("C8 completed, complete", "retrieving", S.COMPLETED, O.COMPLETE, {},
     A.COMPLETED, None, None, None),
    ("C8 timed out, complete", "running", S.TIMED_OUT, O.COMPLETE, {},
     A.COMPLETED, None, None, None),
    ("C8 failed, complete", "running", S.FAILED, O.COMPLETE, {},
     A.COMPLETED, None, None, None),
    ("C8 not listed, complete", "queued", S.NOT_LISTED, O.COMPLETE, {},
     A.COMPLETED, None, None, None),
    ("C9 timed out, incomplete", "running", S.TIMED_OUT, O.INCOMPLETE, {},
     "stopped", EndReason.WALLTIME, WALLTIME_MESSAGE, None),
    ("C9 timed out, absent", "submitting", S.TIMED_OUT, O.ABSENT, {},
     "stopped", EndReason.WALLTIME, WALLTIME_MESSAGE, None),
    ("C10 failed, incomplete", "running", S.FAILED, O.INCOMPLETE,
     {"scheduler_detail": "OUT_OF_MEMORY"},
     "failed", None, "Scheduler reported the job as OUT_OF_MEMORY.", None),
    ("C10 failed, absent", "queued", S.FAILED, O.ABSENT, {"scheduler_detail": "NODE_FAIL"},
     "failed", None, "Scheduler reported the job as NODE_FAIL.", None),
    ("C11 completed, incomplete, walltime", "retrieving", S.COMPLETED, O.INCOMPLETE,
     {"walltime_seen": True},
     "stopped", EndReason.WALLTIME, WALLTIME_MESSAGE, None),
    ("C11 not listed, incomplete, walltime", "running", S.NOT_LISTED, O.INCOMPLETE,
     {"walltime_seen": True},
     "stopped", EndReason.WALLTIME, WALLTIME_MESSAGE, None),
    ("C11 completed, incomplete", "retrieving", S.COMPLETED, O.INCOMPLETE,
     {"output_detail": CASTEP_MESSAGE},
     "failed", None, CASTEP_MESSAGE, None),
    ("C11 not listed, incomplete, no message", "running", S.NOT_LISTED, O.INCOMPLETE, {},
     "failed", None, INCOMPLETE_OUTPUT_MESSAGE, None),
    ("C12 completed, unreadable", "retrieving", S.COMPLETED, O.UNREADABLE,
     {"output_detail": "bad header"},
     "failed", EndReason.OUTPUT_UNAVAILABLE,
     f"{RETRIEVAL_FAILURE_PREFIX}: bad header", None),
    ("C12 not listed, unreadable", "running", S.NOT_LISTED, O.UNREADABLE,
     {"output_detail": "bad header"},
     "failed", EndReason.OUTPUT_UNAVAILABLE,
     f"{RETRIEVAL_FAILURE_PREFIX}: bad header", None),
    ("C13 completed, absent", "retrieving", S.COMPLETED, O.ABSENT, {},
     "failed", EndReason.OUTPUT_UNAVAILABLE, NO_OUTPUT_MESSAGE, None),
    ("C14 not listed, absent", "running", S.NOT_LISTED, O.ABSENT, {},
     "running", None, None, LOST_TRACK_MARKER),
    ("C14 not listed, absent, queued", "queued", S.NOT_LISTED, O.ABSENT, {},
     "queued", None, None, LOST_TRACK_MARKER),
    ("C15 completed, unreachable", "retrieving", S.COMPLETED, O.UNREACHABLE,
     {"output_detail": SFTP_ERROR},
     "retrieving", None, None, f"output could not be read: {SFTP_ERROR}"),
    ("C15 not listed, unreachable", "running", S.NOT_LISTED, O.UNREACHABLE,
     {"output_detail": SFTP_ERROR},
     "running", None, None, f"output could not be read: {SFTP_ERROR}"),
    ("C15 timed out, unreachable", "running", S.TIMED_OUT, O.UNREACHABLE,
     {"output_detail": SFTP_ERROR},
     "running", None, None, f"output could not be read: {SFTP_ERROR}"),
]  # fmt: skip


@pytest.mark.parametrize(
    (
        "job_status",
        "scheduler",
        "output",
        "extra",
        "status",
        "reason",
        "message",
        "marker",
    ),
    [row[1:] for row in CLUSTER_ACTIVE_ROWS],
    ids=[row[0] for row in CLUSTER_ACTIVE_ROWS],
)
def test_active_cluster_job(
    job_status, scheduler, output, extra, status, reason, message, marker
) -> None:
    decision = reconcile(cluster_job(job_status), cluster_evidence(scheduler, output, **extra))

    assert decision is not None
    assert (
        job_status_word(decision),
        decision.end_reason,
        decision.message,
        decision.marker,
    ) == (status, reason, message, marker)


@pytest.mark.parametrize("scheduler", list(SchedulerState))
@pytest.mark.parametrize("output", list(OutputState))
def test_c1_a_job_still_being_submitted_is_left_alone(scheduler, output) -> None:
    job = cluster_job("submitting")

    decision = reconcile(job, cluster_evidence(scheduler, output, submitting=True))

    assert decision == Decision(JobStatus.ACTIVE, JobActivity.SUBMITTING)


NO_ID_ROWS = [
    # id, output, on request, expected status, marker
    ("C16 complete output is adopted on request", O.COMPLETE, True, A.COMPLETED, None),
    ("C16 complete output is not adopted unasked", O.COMPLETE, False, "submitting", NO_JOB_ID_MARKER),
    ("C16 incomplete output decides nothing", O.INCOMPLETE, True, "submitting", NO_JOB_ID_MARKER),
    ("C16 unreadable output decides nothing", O.UNREADABLE, True, "submitting", NO_JOB_ID_MARKER),
    ("C16 no output decides nothing", O.ABSENT, True, "submitting", NO_JOB_ID_MARKER),
    ("C16 unreachable folder decides nothing", O.UNREACHABLE, True, "submitting", NO_JOB_ID_MARKER),
    ("C16 not looked decides nothing", O.NOT_LOOKED, False, "submitting", NO_JOB_ID_MARKER),
]  # fmt: skip


@pytest.mark.parametrize(
    ("output", "on_request", "status", "marker"),
    [row[1:] for row in NO_ID_ROWS],
    ids=[row[0] for row in NO_ID_ROWS],
)
def test_active_cluster_job_with_no_scheduler_id(output, on_request, status, marker) -> None:
    job = cluster_job("submitting", remote_job_id=None)
    evidence = cluster_evidence(S.NO_JOB_ID, output, remote_job_id="", on_request=on_request)

    decision = reconcile(job, evidence)

    assert decision is not None
    assert (job_status_word(decision), decision.marker) == (status, marker)
    assert decision.end_reason is None and decision.message is None


def test_a_cancelled_job_whose_output_was_looked_at_and_is_complete_completes() -> None:
    evidence = cluster_evidence(S.CANCELLED, O.COMPLETE, on_request=True)

    decision = reconcile(cluster_job("running"), evidence)

    assert decision is not None and decision.status == A.COMPLETED


def test_c17_no_scheduler_id_and_no_folder_is_never_looked_at() -> None:
    job = make_job(status="submitting", execution_target="SCARF")
    evidence = cluster_evidence(S.NO_JOB_ID, remote_job_id="", remote_directory="")

    assert reconcile(job, evidence) == Decision(
        JobStatus.ACTIVE, JobActivity.SUBMITTING, marker=NO_JOB_ID_MARKER
    )


@pytest.mark.parametrize("job_status", ACTIVE_WORDS)
@pytest.mark.parametrize(
    ("scheduler", "output"),
    [
        (S.UNAVAILABLE, O.NOT_LOOKED),
        (S.NOT_LISTED, O.ABSENT),
        (S.NOT_LISTED, O.UNREACHABLE),
        (S.NO_JOB_ID, O.ABSENT),
    ],
)
def test_a_cluster_job_is_never_ended_for_want_of_evidence(job_status, scheduler, output) -> None:
    decision = reconcile(cluster_job(job_status), cluster_evidence(scheduler, output))

    assert decision is not None
    assert job_status_word(decision) == job_status
    assert decision.marker


@pytest.mark.parametrize("job_status", ACTIVE_WORDS)
@pytest.mark.parametrize("scheduler", [S.COMPLETED, S.TIMED_OUT, S.FAILED, S.NOT_LISTED])
def test_complete_output_completes_an_active_cluster_job(job_status, scheduler) -> None:
    decision = reconcile(cluster_job(job_status), cluster_evidence(scheduler, O.COMPLETE))

    assert decision is not None and decision.status == A.COMPLETED


def test_accounting_evidence_says_the_queue_did_not_list_the_job() -> None:
    decision = reconcile(cluster_job("queued"), cluster_evidence(S.PENDING, from_accounting=True))

    assert decision is not None
    assert decision.observation == (
        "missing from scheduler queue; scheduler accounting reports QUEUED"
    )


# --- Ended job ---------------------------------------------------------------

ENDED_CLUSTER_ROWS = [
    ("E1 pending", S.PENDING, O.NOT_LOOKED, "queued"),
    ("E1 running", S.RUNNING, O.NOT_LOOKED, "running"),
    ("E2 completed, complete", S.COMPLETED, O.COMPLETE, A.COMPLETED),
    ("E2 not listed, complete", S.NOT_LISTED, O.COMPLETE, A.COMPLETED),
    ("E2 cancelled, complete", S.CANCELLED, O.COMPLETE, A.COMPLETED),
    ("E2 no ID, complete", S.NO_JOB_ID, O.COMPLETE, A.COMPLETED),
    ("E3 completed, incomplete", S.COMPLETED, O.INCOMPLETE, None),
    ("E3 not listed, absent", S.NOT_LISTED, O.ABSENT, None),
    ("E3 failed, unreadable", S.FAILED, O.UNREADABLE, None),
    ("E3 completed, unreachable", S.COMPLETED, O.UNREACHABLE, None),
    ("E3 queue command failed", S.UNAVAILABLE, O.NOT_LOOKED, None),
    ("E3 completed, not looked", S.COMPLETED, O.NOT_LOOKED, None),
    ("E3 cancelled, not looked", S.CANCELLED, O.NOT_LOOKED, None),
    ("E3 not listed, not looked", S.NOT_LISTED, O.NOT_LOOKED, None),
]


@pytest.mark.parametrize("job_status", ENDED_WORDS)
@pytest.mark.parametrize(
    ("scheduler", "output", "status"),
    [row[1:] for row in ENDED_CLUSTER_ROWS],
    ids=[row[0] for row in ENDED_CLUSTER_ROWS],
)
def test_ended_cluster_job_checked_on_request(job_status, scheduler, output, status) -> None:
    job = cluster_job(job_status)
    job.end_reason = EndReason.OUTPUT_UNAVAILABLE
    job.error_message = "earlier failure"
    remote_job_id = "" if scheduler == S.NO_JOB_ID else REMOTE_ID
    job.remote_job_id = remote_job_id or None
    evidence = cluster_evidence(scheduler, output, on_request=True, remote_job_id=remote_job_id)

    decision = reconcile(job, evidence)

    assert decision is not None
    if status is None:
        assert (decision.status, decision.activity) == (JobStatus.ENDED, None)
    else:
        assert job_status_word(decision) == status
    assert decision.end_reason is None and decision.message is None
    assert decision.marker is None


@pytest.mark.parametrize("job_status", ENDED_WORDS)
@pytest.mark.parametrize(
    ("output", "status"),
    [
        (O.COMPLETE, A.COMPLETED),
        (O.INCOMPLETE, None),
        (O.ABSENT, None),
        (O.NOT_LOOKED, None),
    ],
)
def test_ended_local_job_checked_on_request(job_status, output, status) -> None:
    decision = reconcile(local_job(job_status), local_evidence(P.NONE, output, on_request=True))

    assert decision is not None
    assert decision.status == (JobStatus.ENDED if status is None else status)


@pytest.mark.parametrize("job_status", ENDED_WORDS)
@pytest.mark.parametrize(
    "evidence",
    [
        cluster_evidence(S.RUNNING),
        cluster_evidence(S.COMPLETED, O.COMPLETE),
        cluster_evidence(S.NOT_LISTED, O.NOT_LOOKED),
    ],
)
def test_an_ended_job_changes_only_when_the_user_asked(job_status, evidence) -> None:
    assert reconcile(cluster_job(job_status), evidence) == Decision(JobStatus.ENDED)


@pytest.mark.parametrize("job_status", [A.DEFINED, A.COMPLETED])
@pytest.mark.parametrize("on_request", [False, True])
def test_defined_and_completed_jobs_are_never_changed(job_status, on_request) -> None:
    cluster = cluster_evidence(S.RUNNING, O.COMPLETE, on_request=on_request)
    local = local_evidence(P.NONE, O.ABSENT, on_request=on_request)

    assert reconcile(cluster_job(job_status), cluster) == Decision(job_status)
    assert reconcile(local_job(job_status), local) == Decision(job_status)


# --- Evidence about another submission ---------------------------------------


@pytest.mark.parametrize(
    "stale",
    [{"remote_job_id": "4000000"}, {"remote_directory": "/work/demo/cof2/older"}],
    ids=["other scheduler ID", "other folder"],
)
@pytest.mark.parametrize("job_status", ["queued", "failed"])
def test_evidence_about_another_submission_is_dropped(stale, job_status) -> None:
    evidence = replace(cluster_evidence(S.COMPLETED, O.COMPLETE, on_request=True), **stale)

    assert reconcile(cluster_job(job_status), evidence) is None


def test_the_same_evidence_always_gives_the_same_decision() -> None:
    evidence = cluster_evidence(S.NOT_LISTED, O.INCOMPLETE, output_detail=CASTEP_MESSAGE)

    decisions = {reconcile(cluster_job(status), evidence) for status in ACTIVE_WORDS}

    assert len(decisions) == 1


# --- wants_output ------------------------------------------------------------


@pytest.mark.parametrize(
    ("scheduler", "automatic", "on_request"),
    [
        (None, False, False),
        (S.UNAVAILABLE, False, False),
        (S.PENDING, False, False),
        (S.RUNNING, False, False),
        (S.CANCELLED, False, True),
        (S.COMPLETED, True, True),
        (S.TIMED_OUT, True, True),
        (S.FAILED, True, True),
        (S.NOT_LISTED, True, True),
        (S.NO_JOB_ID, False, True),
    ],
)
def test_wants_output(scheduler, automatic, on_request) -> None:
    assert wants_output(scheduler) is automatic
    assert wants_output(scheduler, on_request=True) is on_request

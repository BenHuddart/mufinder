"""How a job reads on the dashboard: its status word, its staleness and its runtime.

:func:`job_status_word` is the one word every job table shows for a job.
:func:`job_status_counts` counts a list of jobs by that word for the Job
Dashboard's summary, and :func:`jobs_in_states` keeps the jobs in the states the
summary is filtered by. An active cluster job's persisted status is only as fresh as the last queue
poll that confirmed it. This module derives a human-readable staleness marker
from the job's monitor metadata (``last_checked_at``) and the current cluster
connection state, so the dashboard can distinguish "still queued" from "not
checked recently" without inventing a new ``JobStatus``.
:func:`parse_scheduler_elapsed_seconds` reads the scheduler's elapsed time a
running cluster job's runtime cell counts on from. :func:`job_is_imported` and
:func:`job_resumes_from_checkpoint` say how a job may be run again.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import EndReason, JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE

STALENESS_THRESHOLD_SECONDS = 180
"""Age beyond which a last check counts as stale (three 60 s poll cycles)."""

#: Marker of an active cluster job whose cluster is not connected.
DISCONNECTED_MARKER = "cluster disconnected"
#: Marker of an active cluster job that has no scheduler job ID.
NO_JOB_ID_MARKER = "no scheduler job ID was recorded"
#: Start of the note on a job whose submission closing MuFinder cut off.
SUBMISSION_CUT_OFF_NOTE = "Its submission was cut off while MuFinder was closing"


def submission_cut_off_message(job: Job) -> str:
    """Return the note for a job whose submission closing MuFinder cut off.

    Whether the scheduler took the job is unknown, so the note says where it
    may be running.
    """

    cluster = (job.execution_target or "").strip() or "the cluster"
    return f"{SUBMISSION_CUT_OFF_NOTE}; it may be running on {cluster}."


def submission_was_cut_off(job: Job) -> bool:
    """Return whether an ended ``job`` carries the note that closing cut its submission off.

    The scheduler may have taken such a job, so its output may be on its cluster.
    """

    return (job.error_message or "").startswith(SUBMISSION_CUT_OFF_NOTE)


#: End reasons under which an ended job reads ``stopped`` rather than ``failed``:
#: something outside the calculation stopped it.
STOPPED_END_REASONS = frozenset({EndReason.WALLTIME, EndReason.INTERRUPTED})


def job_status_word(job: Job) -> str:
    """Return the status word the dashboard and workflow tables show for ``job``.

    An active job reads as its activity (``submitting``, ``queued``,
    ``running``, ``retrieving``). An ended job reads ``cancelled``, ``stopped``
    when :data:`STOPPED_END_REASONS` holds its reason, and ``failed``
    otherwise. Every other job reads as its state.
    """

    if job.status == JobStatus.ACTIVE:
        return str(job.activity or job.status)
    if job.status == JobStatus.ENDED:
        if job.end_reason == EndReason.CANCELLED:
            return "cancelled"
        return "stopped" if job.end_reason in STOPPED_END_REASONS else "failed"
    return str(job.status)


#: The order the Job Dashboard's summary gives its counts in: active states, then ended ones.
JOB_SUMMARY_STATE_ORDER = (
    "running",
    "queued",
    "submitting",
    "retrieving",
    "failed",
    "stopped",
    "cancelled",
)


def job_status_counts(
    jobs: Iterable[Job], *, always: Iterable[str] = ()
) -> tuple[tuple[str, int], ...]:
    """Return how many of ``jobs`` read as each status word, as ``(word, count)`` pairs.

    The words are those of :func:`job_status_word`, in
    :data:`JOB_SUMMARY_STATE_ORDER`; a word outside that order follows it,
    alphabetically. A word no job reads as is left out, so no jobs give ``()``,
    unless it is one of ``always`` (the states the dashboard is filtered by),
    which get a pair even at count 0, in the same order.
    """

    counts = Counter(job_status_word(job) for job in jobs)
    shown = {word for word, count in counts.items() if count} | {str(word) for word in always}
    ordered = [word for word in JOB_SUMMARY_STATE_ORDER if word in shown]
    ordered += sorted(word for word in shown if word not in JOB_SUMMARY_STATE_ORDER)
    return tuple((word, counts[word]) for word in ordered)


def jobs_in_states(jobs: Iterable[Job], words: Iterable[str]) -> list[Job]:
    """Return the jobs of ``jobs`` whose :func:`job_status_word` is one of ``words``, in order.

    No ``words`` keep every job: an unfiltered dashboard lists them all.
    """

    wanted = {str(word) for word in words}
    if not wanted:
        return list(jobs)
    return [job for job in jobs if job_status_word(job) in wanted]


def is_cluster_job(job: Job) -> bool:
    """Return True when the job targets a cluster rather than local execution.

    A blank/empty ``execution_target`` or the literal ``"local"``
    (case-insensitively) means local execution; anything else names a cluster.
    """

    target = (job.execution_target or "").strip()
    return bool(target) and target.casefold() != "local"


def job_where_text(job: Job) -> str:
    """Return where the job runs: ``local``, ``SCARF`` or ``SCARF #4127731``."""

    if not is_cluster_job(job):
        return "local"
    remote_id = (job.remote_job_id or "").strip()
    target = job.execution_target.strip()
    return f"{target} #{remote_id}" if remote_id else target


def _age_seconds(now: datetime, then: datetime) -> float:
    # Stamps written by older builds may be naive; treat them as UTC, the
    # convention used by mufinder.domain.models.utc_now.
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    return (now - then).total_seconds()


def job_staleness(
    job: Job,
    *,
    now: datetime,
    cluster_connected: bool,
    marker: str | None = None,
    submitting: bool = False,
    threshold_seconds: int = STALENESS_THRESHOLD_SECONDS,
) -> str | None:
    """Return a short reason the job's displayed status may be out of date.

    Only active cluster jobs can be stale; everything else returns ``None``.
    ``cluster_connected`` must describe the cluster *this job* targets: pass
    ``connected_cluster_name == job.execution_target``, not merely "some
    cluster is connected". ``marker`` is the note the job's last check left
    when it could not confirm the job (see
    :class:`~mufinder.services.job_reconcile.Decision`); it outranks the age
    of the last check, because that check did happen. ``submitting`` says a
    submit request for the job has not answered yet: such a job has no
    scheduler job ID by design and is not stale.
    """

    if job.status != JobStatus.ACTIVE or not is_cluster_job(job):
        return None
    if not cluster_connected:
        return DISCONNECTED_MARKER
    if submitting:
        return None
    if marker:
        return marker
    if not (job.remote_job_id or "").strip():
        return NO_JOB_ID_MARKER
    if job.last_checked_at is not None:
        age = _age_seconds(now, job.last_checked_at)
        if age > threshold_seconds:
            return f"not checked for {int(age // 60)} min"
        return None
    if job.submitted_at is not None and _age_seconds(now, job.submitted_at) > threshold_seconds:
        return "not checked since submission"
    return None


#: Failure reasons under which a cluster job's output was never brought back.
_UNRETRIEVED_OUTPUT_REASONS = frozenset({EndReason.OUTPUT_UNAVAILABLE, EndReason.INTERRUPTED})


def has_unretrieved_cluster_output(job: Job) -> bool:
    """Return whether ended cluster ``job`` may still have state to fetch from its cluster.

    True for an ended cluster job that kept its remote folder and whose
    recorded cause is a failed retrieval or MuFinder losing track of it, whose
    submission closing MuFinder cut off and no check has answered for since
    (:func:`cut_off_submission_unanswered`), whose result could not be
    recorded (:func:`result_not_recorded`), or that was cancelled without its
    scheduler being told (:func:`cancel_not_sent_to_cluster`). The calculation
    may have finished, or may still be running, out of MuFinder's sight, so a
    check is worth trying before the job is resubmitted or dismissed.
    """

    return (
        job.status == JobStatus.ENDED
        and is_cluster_job(job)
        and (
            job.end_reason in _UNRETRIEVED_OUTPUT_REASONS
            or cut_off_submission_unanswered(job)
            or result_not_recorded(job)
            or cancel_not_sent_to_cluster(job)
        )
        and bool((job.remote_directory or "").strip())
    )


#: Start of the failure message for a finished job whose result could not be stored.
RESULT_NOT_RECORDED_PREFIX = (
    "The job's output was retrieved, but MuFinder could not record the result"
)

#: Monitor source of a job whose submission closing MuFinder cut off.
SUBMISSION_CUT_OFF_SOURCE = "submission"
#: Monitor source of a cluster job cancelled in the project without the scheduler being told.
CANCEL_NOT_SENT_SOURCE = "cancelled_in_project"
#: Monitor message stamped beside :data:`CANCEL_NOT_SENT_SOURCE`.
CANCEL_NOT_SENT_MESSAGE = "cancelled in the project only; the scheduler was not told"


def result_not_recorded(job: Job) -> bool:
    """Return whether ended ``job`` finished but MuFinder could not record its result.

    Its output was retrieved and building or storing the result raised, so
    checking it again can record the result without running it again.
    """

    return job.status == JobStatus.ENDED and (job.error_message or "").startswith(
        RESULT_NOT_RECORDED_PREFIX
    )


def cancel_not_sent_to_cluster(job: Job) -> bool:
    """Return whether cluster ``job`` was cancelled in the project without the scheduler being told.

    Such a job may still be queued or running on its cluster. The stamp is its
    monitor source (:data:`CANCEL_NOT_SENT_SOURCE`); a check that gets an
    answer about the job replaces it, which settles the question.
    """

    return (
        job.status == JobStatus.ENDED
        and job.end_reason == EndReason.CANCELLED
        and is_cluster_job(job)
        and job.last_status_source == CANCEL_NOT_SENT_SOURCE
    )


def cut_off_submission_unanswered(job: Job) -> bool:
    """Return whether ended cluster ``job``'s cut-off submission still awaits a check's answer.

    The scheduler may have taken the job (:func:`submission_was_cut_off`). The
    monitor source :data:`SUBMISSION_CUT_OFF_SOURCE` stays until a check gets
    an answer about the job, which replaces it and settles the question; the
    note stays in the job's message.
    """

    return (
        job.status == JobStatus.ENDED
        and is_cluster_job(job)
        and submission_was_cut_off(job)
        and job.last_status_source == SUBMISSION_CUT_OFF_SOURCE
    )


def awaits_cluster_answer(job: Job) -> bool:
    """Return whether ended cluster ``job`` may still be on its cluster until a check answers.

    A job cancelled without its scheduler being told, or whose submission
    closing MuFinder cut off. Its monitor stamp says so, and a check keeps it
    until the check has an answer about the job.
    """

    return cancel_not_sent_to_cluster(job) or cut_off_submission_unanswered(job)


def check_now_reason(job: Job) -> str:
    """Return why ``job``'s row offers ``Check now``, in the words the status tooltip uses.

    Empty unless :func:`has_unretrieved_cluster_output` holds through one of
    the cases a failure reason does not explain: a cut-off submission, a
    cancel the scheduler was not told about, or a result that could not be
    recorded.
    """

    if not has_unretrieved_cluster_output(job):
        return ""
    cluster = job.execution_target.strip()
    if cancel_not_sent_to_cluster(job):
        return (
            f"Cancelled in the project only: {cluster} was not told, so the job may still be "
            "running there. Use 'Check now' to find out."
        )
    if cut_off_submission_unanswered(job):
        return f"{submission_cut_off_message(job)} Use 'Check now' to find out."
    if result_not_recorded(job):
        return (
            "The job finished, but MuFinder could not record its result. Use 'Check now' to "
            f"record it from the output on {cluster}."
        )
    return ""


#: End reasons a cluster job is given only from what its output folder held.
_JUDGED_OUTPUT_REASONS = frozenset(
    {
        EndReason.WALLTIME,
        EndReason.OUT_OF_MEMORY,
        EndReason.NODE_FAILURE,
        EndReason.CASTEP_ERROR,
        EndReason.UNKNOWN,
    }
)


def cluster_output_judged(job: Job) -> bool:
    """Return whether ended cluster ``job``'s output was already read to a verdict.

    True when the check that ended the job read its folder
    (``last_status_source`` is ``"retrieval"``) and judged the output: the
    job stopped at its walltime or failed for a cause its output or logs gave.
    Nothing new can appear in such a folder unless the job runs again, which
    the scheduler would show, so a queue refresh does not read it again. A job
    whose output could not be read, whose result could not be recorded, that
    MuFinder lost track of, that was cancelled or whose folder was never read
    is not judged.
    """

    return (
        job.status == JobStatus.ENDED
        and is_cluster_job(job)
        and job.end_reason in _JUDGED_OUTPUT_REASONS
        and job.last_status_source == "retrieval"
        and not result_not_recorded(job)
    )


def unretrieved_output_note(
    jobs: Sequence[Job], *, prompt_job_count: int | None = None, action: str = "Dismissing"
) -> str:
    """Return the warning for the jobs in ``jobs`` with unretrieved output.

    ``action`` names what the prompt is about to do (``Dismissing`` or
    ``Starting again``); neither retrieves the output.

    Empty when none has any. Names the clusters involved, first seen first.
    ``prompt_job_count`` is how many jobs the confirmation is about when it
    covers more than ``jobs`` (the jobs it also cancels); the note says "Its
    output" only when the prompt is about a single job.
    """

    unretrieved = [job for job in jobs if has_unretrieved_cluster_output(job)]
    if not unretrieved:
        return ""
    clusters: dict[str, str] = {}
    for job in unretrieved:
        target = job.execution_target.strip()
        clusters.setdefault(target.casefold(), target)
    where = " or ".join(clusters.values())
    if (len(jobs) if prompt_job_count is None else prompt_job_count) == 1:
        return (
            f"Its output may still be on {where} and has not been retrieved. {action} "
            "does not retrieve it; use Check now or Refresh Queue first to keep the result."
        )
    count = len(unretrieved)
    return (
        f"The output of {count} of these jobs may still be on {where} and has not been "
        f"retrieved. {action} does not retrieve it; use Check now or Refresh Queue first "
        "to keep "
        f"the {'result' if count == 1 else 'results'}."
    )


def job_is_imported(job: Job) -> bool:
    """Return whether ``job`` records a calculation performed outside MuFinder.

    Imported jobs must never be reset, resubmitted, or continued: MuFinder did
    not run them, and their directories hold the only copy of the imported
    results together with the archived originals.
    """

    return IMPORTED_JOB_TAG in (job.tags or ())


def job_resumes_from_checkpoint(job: Job) -> bool:
    """Return whether running ``job`` again in its own folder resumes from its checkpoint.

    Relaxation inputs, and those of a Quantum Corrections phonon job, always
    carry ``CONTINUATION : DEFAULT``, so CASTEP picks up a ``{seed}.check``
    left in the run directory. Imported jobs are excluded: their directories
    hold the only copy of the imported results.
    """

    resumes = (
        job.workflow_stage in {WorkflowStage.PRISTINE_RELAXATION, WorkflowStage.MUON_RELAXATION}
        or job.calc_type == QUANTUM_PHONON_CALC_TYPE
    )
    return resumes and not job_is_imported(job)


def parse_scheduler_elapsed_seconds(value: str) -> int | None:
    """Return the seconds in a scheduler's ``[D-]H:M:S`` (or ``M:S``) elapsed time.

    ``None`` when ``value`` is empty or not in that form.
    """

    text = str(value).strip()
    if not text:
        return None
    day_count = 0
    if "-" in text:
        day_text, remainder = text.split("-", 1)
        try:
            day_count = int(day_text)
        except ValueError:
            return None
        text = remainder
    parts = text.split(":")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        return None
    if len(numbers) == 2:
        hours = 0
        minutes, seconds = numbers
    elif len(numbers) == 3:
        hours, minutes, seconds = numbers
    else:
        return None
    return day_count * 86400 + hours * 3600 + minutes * 60 + seconds

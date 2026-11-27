"""Tests for the dashboard staleness helper in services.job_monitoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import partial

import pytest

from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import EndReason, JobStatus, WorkflowStage
from mufinder.services.job_monitoring import (
    CANCEL_NOT_SENT_SOURCE,
    RESULT_NOT_RECORDED_PREFIX,
    STALENESS_THRESHOLD_SECONDS,
    SUBMISSION_CUT_OFF_SOURCE,
    awaits_cluster_answer,
    check_now_reason,
    cancel_not_sent_to_cluster,
    cluster_output_judged,
    has_unretrieved_cluster_output,
    is_cluster_job,
    job_is_imported,
    job_resumes_from_checkpoint,
    job_staleness,
    job_status_counts,
    job_where_text,
    jobs_in_states,
    parse_scheduler_elapsed_seconds,
    submission_cut_off_message,
    unretrieved_output_note,
)
from tests.support.jobs import make_job

_NOW = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)


_job = partial(
    make_job,
    label="Mon",
    status="queued",
    calc_type="castep_convergence_cutoff",
    execution_target="scarf",
)


_tracked = partial(_job, remote_job_id="4127731")


@pytest.mark.parametrize(
    "target,expected",
    [
        ("scarf", True),
        ("local", False),
        ("LOCAL", False),
        ("", False),
        ("  ", False),
    ],
)
def test_is_cluster_job_matches_reconciliation_semantics(target: str, expected: bool) -> None:
    assert is_cluster_job(_job(execution_target=target)) is expected


def test_terminal_and_local_jobs_are_never_stale() -> None:
    old = _NOW - timedelta(hours=5)
    assert (
        job_staleness(
            _job(status="failed", last_checked_at=old),
            now=_NOW,
            cluster_connected=False,
        )
        is None
    )
    assert (
        job_staleness(
            _job(execution_target="local", last_checked_at=old), now=_NOW, cluster_connected=False
        )
        is None
    )


def test_disconnected_cluster_marks_active_job_stale() -> None:
    marker = job_staleness(_job(last_checked_at=_NOW), now=_NOW, cluster_connected=False)
    assert marker == "cluster disconnected"


def test_recent_check_is_fresh_and_old_check_is_stale() -> None:
    fresh = _NOW - timedelta(seconds=STALENESS_THRESHOLD_SECONDS - 1)
    assert job_staleness(_tracked(last_checked_at=fresh), now=_NOW, cluster_connected=True) is None

    old = _NOW - timedelta(minutes=6)
    marker = job_staleness(_tracked(last_checked_at=old), now=_NOW, cluster_connected=True)
    assert marker == "not checked for 6 min"


def test_never_checked_job_falls_back_to_submission_age() -> None:
    recent = _NOW - timedelta(seconds=30)
    assert job_staleness(_tracked(submitted_at=recent), now=_NOW, cluster_connected=True) is None

    old = _NOW - timedelta(minutes=10)
    marker = job_staleness(_tracked(submitted_at=old), now=_NOW, cluster_connected=True)
    assert marker == "not checked since submission"


def test_naive_last_checked_stamp_is_treated_as_utc() -> None:
    naive_old = datetime(2026, 7, 4, 11, 0)  # 60 min before _NOW, no tzinfo
    marker = job_staleness(_tracked(last_checked_at=naive_old), now=_NOW, cluster_connected=True)
    assert marker == "not checked for 60 min"


@pytest.mark.parametrize(
    ("target", "remote_id", "expected"),
    [
        ("local", None, "local"),
        ("", None, "local"),
        ("SCARF", None, "SCARF"),
        ("SCARF", "4127731", "SCARF #4127731"),
    ],
)
def test_job_where_text_names_the_cluster_and_its_job_id(target, remote_id, expected) -> None:
    job = make_job(execution_target=target, remote_job_id=remote_id)

    assert job_where_text(job) == expected


def test_the_marker_of_the_last_check_outranks_its_age() -> None:
    job = _tracked(last_checked_at=_NOW)
    marker = job_staleness(job, now=_NOW, cluster_connected=True, marker="lost track: gone")
    assert marker == "lost track: gone"


def test_job_without_a_scheduler_id_says_so() -> None:
    marker = job_staleness(_job(last_checked_at=_NOW), now=_NOW, cluster_connected=True)
    assert marker == "no scheduler job ID was recorded"


def test_job_on_a_disconnected_cluster_reads_as_disconnected_whatever_else_holds() -> None:
    marker = job_staleness(_job(), now=_NOW, cluster_connected=False, marker="lost track: gone")
    assert marker == "cluster disconnected"


def test_a_job_still_being_submitted_is_not_stale() -> None:
    # The scheduler has not returned its ID yet.
    job = _job(
        status="submitting",
        submitted_at=_NOW - timedelta(minutes=10),
    )

    assert job_staleness(job, now=_NOW, cluster_connected=True, submitting=True) is None
    assert job_staleness(job, now=_NOW, cluster_connected=True) == (
        "no scheduler job ID was recorded"
    )


def _orphaned(label: str) -> object:
    return _job(
        label=label,
        status=JobStatus.ENDED,
        end_reason=EndReason.INTERRUPTED,
        remote_directory=f"/remote/{label}",
    )


def test_unretrieved_note_says_its_only_when_the_prompt_is_about_one_job() -> None:
    job = _orphaned("a")
    assert unretrieved_output_note([job]).startswith("Its output may still be on scarf")
    # The same job dismissed in a prompt that also cancels others.
    note = unretrieved_output_note([job], prompt_job_count=3)
    assert note.startswith("The output of 1 of these jobs may still be on scarf")
    assert note.endswith("to keep the result.")
    assert unretrieved_output_note([_job(status="failed")]) == ""


@pytest.mark.parametrize(
    ("reason", "source", "expected"),
    [
        # The folder was read and its output judged: nothing new can be there.
        (EndReason.WALLTIME, "retrieval", True),
        (EndReason.OUT_OF_MEMORY, "retrieval", True),
        (EndReason.NODE_FAILURE, "retrieval", True),
        (EndReason.CASTEP_ERROR, "retrieval", True),
        (EndReason.UNKNOWN, "retrieval", True),
        # The output could not be read, or was never looked at.
        (EndReason.OUTPUT_UNAVAILABLE, "retrieval", False),
        (EndReason.INTERRUPTED, "retrieval", False),
        (EndReason.CANCELLED, "retrieval", False),
        (EndReason.UNKNOWN, "scheduler_poll", False),
        (EndReason.UNKNOWN, None, False),
        (EndReason.CASTEP_ERROR, "local_runner", False),
    ],
)
def test_cluster_output_judged_only_after_its_folder_was_read_to_a_verdict(
    reason: EndReason, source: str | None, expected: bool
) -> None:
    job = _job(
        status=JobStatus.ENDED,
        end_reason=reason,
        remote_directory="/remote/a",
        last_status_source=source,
    )

    assert cluster_output_judged(job) is expected


def test_cluster_output_judged_needs_an_ended_cluster_job() -> None:
    fields = {"end_reason": EndReason.WALLTIME, "last_status_source": "retrieval"}

    assert cluster_output_judged(_job(status=JobStatus.ENDED, **fields)) is True
    assert cluster_output_judged(_job(status="running", last_status_source="retrieval")) is False
    assert (
        cluster_output_judged(_job(status=JobStatus.ENDED, execution_target="local", **fields))
        is False
    )


def test_a_job_whose_result_was_not_recorded_was_never_judged() -> None:
    # Its folder was read, but recording the result raised: checking again can
    # record it, so a refresh reads the folder again.
    job = _job(
        status=JobStatus.ENDED,
        end_reason=EndReason.UNKNOWN,
        remote_directory="/remote/a",
        last_status_source="retrieval",
        error_message=f"{RESULT_NOT_RECORDED_PREFIX}: unexpected output layout",
    )

    assert cluster_output_judged(job) is False
    assert has_unretrieved_cluster_output(job) is True


def _cancelled(**fields) -> object:
    values = {
        "status": JobStatus.ENDED,
        "end_reason": EndReason.CANCELLED,
        "remote_directory": "/remote/a",
        "remote_job_id": "4127731",
        "last_status_source": CANCEL_NOT_SENT_SOURCE,
    }
    return _job(**(values | fields))


def test_a_cancel_the_scheduler_never_received_leaves_state_to_fetch() -> None:
    job = _cancelled()

    assert cancel_not_sent_to_cluster(job) is True
    assert has_unretrieved_cluster_output(job) is True
    assert awaits_cluster_answer(job) is True
    assert check_now_reason(job) == (
        "Cancelled in the project only: scarf was not told, so the job may still be "
        "running there. Use 'Check now' to find out."
    )
    assert unretrieved_output_note([job]).startswith("Its output may still be on scarf")


@pytest.mark.parametrize(
    "fields",
    [
        # The scheduler confirmed the cancel, or a later check answered.
        {"last_status_source": None},
        {"last_status_source": "scheduler_poll"},
        {"last_status_source": "retrieval"},
        # Only a cancelled cluster job that has ended can carry the stamp.
        {"end_reason": EndReason.UNKNOWN},
        {"status": "running", "end_reason": None},
        {"execution_target": "local"},
    ],
    ids=["confirmed", "settled-by-queue", "settled-by-folder", "failed", "active", "local"],
)
def test_any_other_cancel_or_job_has_nothing_to_fetch(fields) -> None:
    job = _cancelled(**fields)

    assert cancel_not_sent_to_cluster(job) is False
    assert check_now_reason(job) == ""
    assert has_unretrieved_cluster_output(job) is False


def test_an_unsent_cancel_needs_a_folder_to_check() -> None:
    job = _cancelled(remote_directory=None)

    assert cancel_not_sent_to_cluster(job) is True
    assert has_unretrieved_cluster_output(job) is False
    # The row offers no Check now, so the tooltip gives no reason for one.
    assert check_now_reason(job) == ""


def test_unretrieved_note_points_to_check_now_and_refresh_queue() -> None:
    note = unretrieved_output_note([_orphaned("a")])

    assert note.endswith("use Check now or Refresh Queue first to keep the result.")


def test_unretrieved_note_names_the_action_it_warns_about() -> None:
    jobs = [_orphaned("a"), _orphaned("b")]

    assert "Dismissing does not retrieve it" in unretrieved_output_note(jobs)
    assert "Resetting does not retrieve it" in unretrieved_output_note(jobs, action="Resetting")


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("05:07", 307),
        ("1:02:03", 3723),
        ("2-01:00:00", 2 * 86400 + 3600),
        ("  00:00:09 ", 9),
    ],
)
def test_scheduler_elapsed_time_is_read_in_seconds(text: str, seconds: int) -> None:
    assert parse_scheduler_elapsed_seconds(text) == seconds


@pytest.mark.parametrize("text", ["", "   ", "soon", "x-01:00:00", "1:2:3:4", "7", "1:aa"])
def test_an_elapsed_time_not_in_scheduler_form_reads_as_none(text: str) -> None:
    assert parse_scheduler_elapsed_seconds(text) is None


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        (WorkflowStage.PRISTINE_RELAXATION, True),
        (WorkflowStage.MUON_RELAXATION, True),
        (WorkflowStage.CONVERGENCE_CUTOFF, False),
    ],
)
def test_a_relaxation_resumes_from_its_checkpoint(stage: WorkflowStage, expected: bool) -> None:
    assert job_resumes_from_checkpoint(make_job(workflow_stage=stage)) is expected


def test_a_quantum_phonon_job_resumes_from_its_checkpoint() -> None:
    """Its inputs carry ``CONTINUATION : DEFAULT``; the legacy singlepoint's do not."""

    from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE

    stage = WorkflowStage.QUANTUM_CORRECTIONS
    assert job_resumes_from_checkpoint(
        make_job(workflow_stage=stage, calc_type=QUANTUM_PHONON_CALC_TYPE)
    )
    assert not job_resumes_from_checkpoint(
        make_job(workflow_stage=stage, calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE)
    )


def test_an_imported_relaxation_never_resumes_from_a_checkpoint() -> None:
    job = make_job(workflow_stage=WorkflowStage.MUON_RELAXATION, tags=[IMPORTED_JOB_TAG])

    assert job_is_imported(job)
    assert not job_resumes_from_checkpoint(job)
    assert not job_is_imported(make_job(workflow_stage=WorkflowStage.MUON_RELAXATION))


def test_a_job_whose_submission_closing_cut_off_may_have_output_on_its_cluster() -> None:
    job = _job(status="cancelled", remote_directory="/remote/a")
    assert not has_unretrieved_cluster_output(job)

    job.error_message = submission_cut_off_message(job)
    job.last_status_source = SUBMISSION_CUT_OFF_SOURCE

    assert job.error_message == (
        "Its submission was cut off while MuFinder was closing; it may be running on scarf."
    )
    assert has_unretrieved_cluster_output(job)
    assert awaits_cluster_answer(job)
    assert check_now_reason(job) == (
        "Its submission was cut off while MuFinder was closing; it may be running on scarf. "
        "Use 'Check now' to find out."
    )
    job.remote_directory = None
    assert not has_unretrieved_cluster_output(job)
    assert check_now_reason(job) == ""


def test_a_cut_off_submission_a_check_answered_is_settled() -> None:
    job = _job(status="cancelled", remote_directory="/remote/a", remote_job_id="4127731")
    job.error_message = submission_cut_off_message(job)
    # A check that got the scheduler's answer replaced the stamp.
    job.last_status_source = "scheduler_poll"

    assert not has_unretrieved_cluster_output(job)
    assert not awaits_cluster_answer(job)
    assert check_now_reason(job) == ""


def test_a_result_not_recorded_names_check_now_as_the_way_to_record_it() -> None:
    job = _job(
        status=JobStatus.ENDED,
        end_reason=EndReason.UNKNOWN,
        remote_directory="/remote/a",
        error_message=f"{RESULT_NOT_RECORDED_PREFIX}: disk full",
    )

    assert not awaits_cluster_answer(job)
    assert check_now_reason(job) == (
        "The job finished, but MuFinder could not record its result. Use 'Check now' to "
        "record it from the output on scarf."
    )
    assert check_now_reason(_orphaned("lost")) == ""


# --- job_status_counts ----------------------------------------------------------


def test_status_counts_follow_the_summary_order_and_leave_out_empty_states() -> None:
    words = ["failed", "queued", "running", "cancelled", "queued", "stopped", "retrieving"]
    jobs = [make_job(f"job {index}", status=word) for index, word in enumerate(words)]

    assert job_status_counts(jobs) == (
        ("running", 1),
        ("queued", 2),
        ("retrieving", 1),
        ("failed", 1),
        ("stopped", 1),
        ("cancelled", 1),
    )


def test_status_counts_put_submitting_between_queued_and_retrieving() -> None:
    jobs = [make_job(status=word) for word in ("retrieving", "submitting", "queued")]

    assert [word for word, _count in job_status_counts(jobs)] == [
        "queued",
        "submitting",
        "retrieving",
    ]


def test_status_counts_of_no_jobs_are_empty() -> None:
    assert job_status_counts([]) == ()


def test_status_counts_put_a_word_outside_the_order_last() -> None:
    jobs = [make_job(status="completed"), make_job(status="running"), make_job(status="defined")]

    assert job_status_counts(jobs) == (("running", 1), ("completed", 1), ("defined", 1))


def test_status_counts_keep_an_always_state_at_zero_in_its_place() -> None:
    jobs = [make_job(status="cancelled"), make_job(status="running")]

    assert job_status_counts(jobs, always=("failed",)) == (
        ("running", 1),
        ("failed", 0),
        ("cancelled", 1),
    )
    assert job_status_counts([], always=("stopped", "queued")) == (("queued", 0), ("stopped", 0))


def test_status_counts_count_an_always_state_that_has_jobs_once() -> None:
    jobs = [make_job(status="failed"), make_job(status="failed")]

    assert job_status_counts(jobs, always=("failed",)) == (("failed", 2),)


def test_status_counts_put_an_unknown_always_word_last() -> None:
    jobs = [make_job(status="running")]

    assert job_status_counts(jobs, always=("mystery", "failed")) == (
        ("running", 1),
        ("failed", 0),
        ("mystery", 0),
    )


# --- jobs_in_states ---------------------------------------------------------------


def _jobs_by_word() -> list[object]:
    words = ["failed", "running", "stopped", "failed", "queued"]
    return [make_job(f"job {index}", status=word) for index, word in enumerate(words)]


def test_jobs_in_no_states_are_all_the_jobs() -> None:
    jobs = _jobs_by_word()

    assert jobs_in_states(jobs, ()) == jobs


def test_jobs_in_one_state_keep_their_order() -> None:
    jobs = _jobs_by_word()

    assert [job.label for job in jobs_in_states(jobs, ["failed"])] == ["job 0", "job 3"]


def test_jobs_in_several_states_keep_their_order() -> None:
    jobs = _jobs_by_word()

    assert [job.label for job in jobs_in_states(jobs, ("queued", "failed"))] == [
        "job 0",
        "job 3",
        "job 4",
    ]


def test_jobs_in_a_state_no_job_is_in_are_none() -> None:
    assert jobs_in_states(_jobs_by_word(), ("cancelled",)) == []

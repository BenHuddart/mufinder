"""Tests for the dashboard staleness helper in services.job_monitoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import partial

import pytest

from mufinder.domain.enums import FailureReason, JobStatus
from mufinder.services.job_monitoring import (
    STALENESS_THRESHOLD_SECONDS,
    is_cluster_job,
    job_staleness,
    job_where_text,
    unretrieved_output_note,
)
from tests.support.jobs import make_job

_NOW = datetime(2026, 7, 4, 12, 0, tzinfo=UTC)


_job = partial(
    make_job,
    label="Mon",
    status=JobStatus.QUEUED,
    calc_type="castep_convergence_cutoff",
    execution_target="scarf",
)


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
            _job(status=JobStatus.FAILED, last_checked_at=old), now=_NOW, cluster_connected=False
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
    assert job_staleness(_job(last_checked_at=fresh), now=_NOW, cluster_connected=True) is None

    old = _NOW - timedelta(minutes=6)
    marker = job_staleness(_job(last_checked_at=old), now=_NOW, cluster_connected=True)
    assert marker == "not checked for 6 min"


def test_never_checked_job_falls_back_to_submission_age() -> None:
    recent = _NOW - timedelta(seconds=30)
    assert job_staleness(_job(submitted_at=recent), now=_NOW, cluster_connected=True) is None

    old = _NOW - timedelta(minutes=10)
    marker = job_staleness(_job(submitted_at=old), now=_NOW, cluster_connected=True)
    assert marker == "not checked since submission"


def test_naive_last_checked_stamp_is_treated_as_utc() -> None:
    naive_old = datetime(2026, 7, 4, 11, 0)  # 60 min before _NOW, no tzinfo
    marker = job_staleness(_job(last_checked_at=naive_old), now=_NOW, cluster_connected=True)
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


def test_untracked_job_on_a_connected_cluster_says_it_is_not_monitored() -> None:
    job = _job(last_checked_at=_NOW, remote_job_id="4127731")
    marker = job_staleness(job, now=_NOW, cluster_connected=True, tracked=False)
    assert marker == "MuFinder is not monitoring this job"


def test_untracked_job_without_a_scheduler_id_says_why() -> None:
    job = _job(last_checked_at=_NOW)
    marker = job_staleness(job, now=_NOW, cluster_connected=True, tracked=False)
    assert marker == "MuFinder is not monitoring this job: no scheduler job ID was recorded"


def test_untracked_job_on_a_disconnected_cluster_reads_as_disconnected() -> None:
    marker = job_staleness(_job(), now=_NOW, cluster_connected=False, tracked=False)
    assert marker == "cluster disconnected"


def test_a_job_still_being_submitted_is_not_reported_as_unmonitored() -> None:
    # The runner holds a job only once the scheduler has returned its ID.
    job = _job(status=JobStatus.SUBMITTING, submitted_at=_NOW - timedelta(seconds=20))
    assert job_staleness(job, now=_NOW, cluster_connected=True, tracked=False) is None

    stuck = _job(status=JobStatus.SUBMITTING, submitted_at=_NOW - timedelta(minutes=10))
    marker = job_staleness(stuck, now=_NOW, cluster_connected=True, tracked=False)
    assert marker == "MuFinder is not monitoring this job: no scheduler job ID was recorded"


def _orphaned(label: str) -> object:
    return _job(
        label=label,
        status=JobStatus.STOPPED,
        failure_reason=FailureReason.INTERRUPTED,
        remote_directory=f"/remote/{label}",
    )


def test_unretrieved_note_says_its_only_when_the_prompt_is_about_one_job() -> None:
    job = _orphaned("a")
    assert unretrieved_output_note([job]).startswith("Its output may still be on scarf")
    # The same job dismissed in a prompt that also cancels others.
    note = unretrieved_output_note([job], prompt_job_count=3)
    assert note.startswith("The output of 1 of these jobs may still be on scarf")
    assert note.endswith("to keep the result.")
    assert unretrieved_output_note([_job(status=JobStatus.FAILED)]) == ""

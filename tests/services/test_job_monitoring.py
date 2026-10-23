"""Tests for the dashboard staleness helper in services.job_monitoring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import partial

import pytest

from mufinder.domain.enums import JobStatus
from mufinder.services.job_monitoring import (
    STALENESS_THRESHOLD_SECONDS,
    is_cluster_job,
    job_staleness,
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

"""Tests for queue-name and walltime checks."""

from __future__ import annotations

import pytest

from mufinder.domain.cluster import ClusterQueue
from mufinder.services.cluster_queues import (
    is_valid_queue_name,
    parse_walltime_seconds,
    queue_issues,
    walltime_issue,
)


@pytest.mark.parametrize(
    ("walltime", "seconds"),
    [
        ("30", 30 * 60),
        ("30:15", 30 * 60 + 15),
        ("01:00:00", 3600),
        ("120:00:00", 120 * 3600),
        ("2-00", 2 * 86400),
        ("1-12:30", 86400 + 12 * 3600 + 30 * 60),
        ("1-00:00:01", 86401),
        (" 00:10:00 ", 600),
    ],
)
def test_parse_walltime_accepts_slurm_formats(walltime: str, seconds: int) -> None:
    assert parse_walltime_seconds(walltime) == seconds


@pytest.mark.parametrize("walltime", ["", "1h", "1:2:3:4", "-1", "1-", "a:00:00"])
def test_parse_walltime_rejects_other_text(walltime: str) -> None:
    assert parse_walltime_seconds(walltime) is None


def test_walltime_issue_checks_format_and_queue_limit() -> None:
    assert walltime_issue("") == ""
    assert walltime_issue("12:00:00", max_walltime="1-00:00:00") == ""
    assert "not a walltime" in walltime_issue("soon")
    issue = walltime_issue("2-00:00:00", max_walltime="1-00:00:00", queue_name="daily")
    assert "longer than the 1-00:00:00 maximum for queue 'daily'" in issue
    # An unreadable limit is not enforced; the setup dialog reports it instead.
    assert walltime_issue("2-00:00:00", max_walltime="forever") == ""


@pytest.mark.parametrize("name", ["daily", "gpu-a100", "short.q", "hourly,daily", "my_queue"])
def test_valid_queue_names(name: str) -> None:
    assert is_valid_queue_name(name)


@pytest.mark.parametrize("name", ["", "two words", "daily;rm", "a,,b", ",daily", "$(x)"])
def test_invalid_queue_names(name: str) -> None:
    assert not is_valid_queue_name(name)


def test_queue_issues_accepts_valid_queues() -> None:
    queues = [
        ClusterQueue(name="hourly", default_walltime="00:59:00", max_walltime="01:00:00"),
        ClusterQueue(name="daily", max_walltime="1-00:00:00", cores_per_node=88, default_mem_per_cpu_gb=8.0),
    ]

    assert queue_issues(queues, profile_walltime="12:00:00") == []
    assert queue_issues([]) == []


@pytest.mark.parametrize(
    ("queue", "expected"),
    [
        (ClusterQueue(name=""), "The queue in row 1 has no name."),
        (ClusterQueue(name="two words"), "'two words' is not a queue name."),
        (ClusterQueue(name="q", max_walltime="forever"), "Queue 'q': 'forever' is not a walltime."),
        (
            ClusterQueue(name="q", default_walltime="2-00:00:00", max_walltime="1-00:00:00"),
            "Queue 'q': The walltime 2-00:00:00 is longer than the 1-00:00:00 maximum for queue 'q'.",
        ),
        (ClusterQueue(name="q", max_walltime="06:00:00"), "The queue uses the cluster's default walltime"),
        (ClusterQueue(name="q", cores_per_node=0), "Queue 'q': cores per node must be a positive whole number."),
        (ClusterQueue(name="q", default_memory_gb=-1.0), "Queue 'q': memory must be a positive number of GB."),
        (ClusterQueue(name="q", default_mem_per_cpu_gb=0.0), "Queue 'q': memory must be a positive number of GB."),
    ],
)
def test_queue_issues_reports_each_problem(queue: ClusterQueue, expected: str) -> None:
    issues = queue_issues([queue], profile_walltime="12:00:00")

    assert len(issues) == 1 and expected in issues[0]


def test_queue_issues_reports_duplicate_names() -> None:
    assert queue_issues([ClusterQueue(name="q"), ClusterQueue(name="q")]) == ["The queue 'q' is listed twice."]

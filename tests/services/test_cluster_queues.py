"""Tests for queue-name and walltime checks."""

from __future__ import annotations

import pytest

from mufinder.domain.cluster import ClusterNodeGroup, ClusterQueue
from mufinder.services.cluster_queues import (
    ProcessPlacement,
    equal_placements,
    is_valid_queue_name,
    parse_walltime_seconds,
    placement_issue,
    queue_issues,
    walltime_issue,
)

#: 19 nodes that run 32 processes and 28 that run 64.
_REDWOOD_NODES = (
    ClusterNodeGroup(count=19, processes=32),
    ClusterNodeGroup(count=28, processes=64),
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
        ClusterQueue(
            name="daily", max_walltime="1-00:00:00", cores_per_node=88, default_mem_per_cpu_gb=8.0
        ),
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
        (
            ClusterQueue(name="q", max_walltime="06:00:00"),
            "The queue uses the cluster's default walltime",
        ),
        (
            ClusterQueue(name="q", cores_per_node=0),
            "Queue 'q': cores per node must be a positive whole number.",
        ),
        (
            ClusterQueue(name="q", default_memory_gb=-1.0),
            "Queue 'q': memory must be a positive number of GB.",
        ),
        (
            ClusterQueue(name="q", default_mem_per_cpu_gb=0.0),
            "Queue 'q': memory must be a positive number of GB.",
        ),
    ],
)
def test_queue_issues_reports_each_problem(queue: ClusterQueue, expected: str) -> None:
    issues = queue_issues([queue], profile_walltime="12:00:00")

    assert len(issues) == 1 and expected in issues[0]


def test_queue_issues_reports_duplicate_names() -> None:
    assert queue_issues([ClusterQueue(name="q"), ClusterQueue(name="q")]) == [
        "The queue 'q' is listed twice."
    ]


def test_process_placement_formats_the_addqueue_argument() -> None:
    placement = ProcessPlacement(nodes=2, per_node=64)

    assert placement.addqueue_argument() == "2x64"
    assert placement.total == 128


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (128, ["2x64", "4x32", "8x16", "16x8", "32x4"]),
        (96, ["2x48", "3x32", "4x24", "6x16", "8x12", "12x8", "16x6", "24x4", "32x3"]),
        (37, ["1x37", "37x1"]),
        (2000, []),
    ],
)
def test_equal_placements_lists_the_splits_the_nodes_hold(total: int, expected: list[str]) -> None:
    placements = equal_placements(total, _REDWOOD_NODES)

    assert [placement.addqueue_argument() for placement in placements] == expected


def test_equal_placements_without_node_groups_lists_every_divisor() -> None:
    placements = equal_placements(12)

    assert [placement.per_node for placement in placements] == [12, 6, 4, 3, 2, 1]


def test_equal_placements_of_no_processes_is_empty() -> None:
    assert equal_placements(0) == []


@pytest.mark.parametrize(
    ("total", "per_node", "expected"),
    [
        (128, 64, ""),
        (48, 48, ""),
        (128, 0, "Processes per node must be a positive whole number."),
        (32, 64, "64 processes per node is more than the 32 cores the job uses."),
        (128, 48, "48 processes per node does not divide 128 cores into whole nodes."),
        (192, 96, "No node on this cluster runs more than 64 processes."),
        (
            1920,
            40,
            "48 nodes × 40 per node needs more nodes than the 28 on this cluster that "
            "run 40 processes.",
        ),
        (
            128,
            2,
            "64 nodes × 2 per node needs more nodes than the 47 on this cluster that "
            "run 2 processes.",
        ),
    ],
)
def test_placement_issue_explains_each_refusal(total: int, per_node: int, expected: str) -> None:
    assert placement_issue(total, per_node, _REDWOOD_NODES) == expected


def test_placement_issue_without_node_groups_checks_only_the_arithmetic() -> None:
    assert placement_issue(4096, 512) == ""
    assert placement_issue(100, 30) != ""

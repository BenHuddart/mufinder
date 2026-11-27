"""Tests for the cluster profile record and its queue resolution."""

from __future__ import annotations

from mufinder.domain.cluster import ClusterConfig, ClusterQueue

_PROFILE = ClusterConfig(
    name="PSI",
    hostname="login.psi.example",
    cores_per_node=44,
    default_walltime="01:00:00",
    template_script_path="/templates/psi.sh",
    castep_executable="castep.mpi",
    default_mem_per_cpu_gb=4.0,
    default_queue_name="daily",
    queues=(
        ClusterQueue(name="hourly", default_walltime="00:59:00", max_walltime="01:00:00"),
        ClusterQueue(
            name="daily",
            default_walltime="12:00:00",
            max_walltime="1-00:00:00",
            cores_per_node=88,
            default_mem_per_cpu_gb=8.0,
            castep_executable="/opt/castep-avx512/castep.mpi",
            template_script_path="/templates/daily.sh",
        ),
        ClusterQueue(name="general"),
    ),
)


def test_queue_names_put_the_default_queue_first() -> None:
    assert _PROFILE.queue_names() == ["daily", "hourly", "general"]


def test_single_queue_profile_offers_its_default_queue() -> None:
    profile = ClusterConfig(name="Redwood", hostname="glamdring", default_queue_name="redwood")

    assert profile.queue_names() == ["redwood"]
    assert ClusterConfig(name="x", hostname="y").queue_names() == []


def test_for_queue_applies_every_override() -> None:
    daily = _PROFILE.for_queue("daily")

    assert daily.default_queue_name == "daily"
    assert daily.default_walltime == "12:00:00"
    assert daily.cores_per_node == 88
    assert daily.default_mem_per_cpu_gb == 8.0
    assert daily.castep_executable == "/opt/castep-avx512/castep.mpi"
    assert daily.template_script_path == "/templates/daily.sh"
    assert daily.hostname == _PROFILE.hostname


def test_for_queue_keeps_profile_values_the_queue_does_not_override() -> None:
    general = _PROFILE.for_queue("general")

    assert general.default_queue_name == "general"
    assert (general.default_walltime, general.cores_per_node, general.default_mem_per_cpu_gb) == (
        "01:00:00",
        44,
        4.0,
    )
    assert general.castep_executable == "castep.mpi"
    assert general.template_script_path == "/templates/psi.sh"


def test_for_queue_with_an_unlisted_or_empty_name() -> None:
    assert _PROFILE.for_queue("") is _PROFILE
    typed = _PROFILE.for_queue("debug")
    assert typed.default_queue_name == "debug"
    assert typed.cores_per_node == 44


def test_max_walltime_is_per_queue() -> None:
    assert _PROFILE.max_walltime_for_queue("hourly") == "01:00:00"
    assert _PROFILE.max_walltime_for_queue("general") == ""
    assert _PROFILE.max_walltime_for_queue("unknown") == ""

"""Storing a relaxation analysis workspace's grouping, and keeping it when relaxations join."""

from __future__ import annotations

from uuid import uuid4

import pytest

from mufinder.science.relaxation_analysis import (
    RelaxedMuonSite,
    cluster_connected_components,
    normalize_site_energies,
)
from mufinder.services.analysis_grouping import (
    DEFAULT_GROUPING_SETTINGS,
    SiteGrouping,
    extend_grouping_with_new_sites,
    grouped_clustering_state,
    grouping_settings,
    ungrouped_site_count,
)
from tests.support.structures import host_structure


def _site(label: str, frac: tuple[float, float, float], energy: float) -> RelaxedMuonSite:
    return RelaxedMuonSite(
        job_id=uuid4(),
        job_label=label,
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=frac,
        cartesian_position_ang=tuple(5.0 * value for value in frac),
        total_energy_ev=energy,
    )


A = _site("A", (0.5, 0.25, 0.0), -10.0)
B = _site("B", (0.0, 0.5, 0.25), -9.9)
C = _site("C", (0.25, 0.25, 0.25), -9.5)


def _applied(sites: list[RelaxedMuonSite]) -> dict[str, object]:
    result = cluster_connected_components(sites, host_structure(), tolerance_ang=0.25)
    grouping = SiteGrouping(
        method="connected",
        cluster_labels=result.cluster_labels,
        clusters=result.clusters,
        wyckoff_assignments=(),
    )
    return grouped_clustering_state(DEFAULT_GROUPING_SETTINGS, grouping, host_structure())


def test_an_applied_grouping_stores_its_result_and_record() -> None:
    clustering = _applied([A, B, C])

    assert clustering["method"] == "connected"
    assert clustering["result"]["cluster_labels"] == [1, 1, 2]
    assert clustering["grouping"]["method"] == "connected"
    assert clustering["grouping"]["space_group_symbol"] == "Im-3m"


def test_k_means_keeps_the_k_asked_for_and_records_the_k_used() -> None:
    grouping = SiteGrouping(
        method="kmeans",
        cluster_labels=(1, 1, 2),
        clusters=(),
        wyckoff_assignments=(),
        kmeans_k_used=2,
    )

    clustering = grouped_clustering_state(
        {**DEFAULT_GROUPING_SETTINGS, "method": "kmeans", "kmeans_k": 3}, grouping, host_structure()
    )

    assert clustering["kmeans_k"] == 3
    assert clustering["grouping"]["kmeans_k"] == 2


def test_relaxations_that_join_a_grouped_workspace_stay_ungrouped() -> None:
    clustering = _applied([A, C])
    # D is lower in energy than every earlier site, so the earlier sites move down.
    d = _site("D", (0.0, 0.5, 0.25), -11.0)

    extended = extend_grouping_with_new_sites(clustering, [A, C], [A, C, d, B])

    order = [site.job_label for site in normalize_site_energies([A, C, d, B])]
    assert order == ["D", "A", "B", "C"]
    labels = extended["result"]["cluster_labels"]
    assert labels == [0, 1, 0, 2]
    assert [cluster["member_indexes"] for cluster in extended["result"]["clusters"]] == [[1], [3]]
    assert [cluster["representative_index"] for cluster in extended["result"]["clusters"]] == [1, 3]
    assert extended["grouping"] == clustering["grouping"]
    assert extended["tolerance_scan"] == []
    clusters = extended["result"]["clusters"]
    assert [cluster["size"] for cluster in clusters] == [1, 1]
    # D moved the zero down by 1 eV, so A's group is now 1 eV above it.
    assert clusters[0]["minimum_relative_energy_ev"] == pytest.approx(1.0)
    assert ungrouped_site_count(extended, labels) == 2


def test_an_ungrouped_workspace_is_left_as_it_is() -> None:
    ungrouped = {**DEFAULT_GROUPING_SETTINGS, "result": None}

    assert extend_grouping_with_new_sites(ungrouped, [A], [A, B]) == ungrouped
    assert extend_grouping_with_new_sites(None, [A], [A, B]) is None
    assert ungrouped_site_count(ungrouped, [0, 0]) == 0


def test_stale_labels_are_left_alone() -> None:
    clustering = _applied([A, B, C])

    assert extend_grouping_with_new_sites(clustering, [A, B], [A, B, C]) == clustering


def test_a_grouping_that_covers_every_site_is_up_to_date() -> None:
    clustering = _applied([A, B, C])

    assert ungrouped_site_count(clustering, clustering["result"]["cluster_labels"]) == 0


def test_grouping_settings_fill_gaps_and_ignore_unreadable_values() -> None:
    assert grouping_settings(None) == DEFAULT_GROUPING_SETTINGS
    settings = grouping_settings(
        {"method": "bogus", "kmeans_k": "x", "hierarchical_t": "0.5", "connected_tolerance_ang": 1}
    )

    assert settings["method"] == "connected"
    assert settings["kmeans_k"] == 4
    assert settings["hierarchical_t"] == 0.5
    assert settings["connected_tolerance_ang"] == 1.0
    negative = grouping_settings(
        {"connected_tolerance_ang": -1, "kmeans_k": 0, "hierarchical_t": 0}
    )
    assert negative == DEFAULT_GROUPING_SETTINGS
    assert grouping_settings({"wyckoff_match_tolerance_ang": 0})["wyckoff_match_tolerance_ang"] == 0

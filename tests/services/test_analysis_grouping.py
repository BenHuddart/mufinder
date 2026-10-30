"""Grouping a relaxation analysis workspace's relaxed sites."""

from __future__ import annotations

import importlib.util
from uuid import uuid4

import pytest

from mufinder.science.relaxation_analysis import RelaxedMuonSite
from mufinder.science.uep import UepError
from mufinder.services import analysis_grouping
from mufinder.services.analysis_grouping import (
    DEFAULT_GROUPING_SETTINGS,
    grouping_settings,
    regroup_workspace,
)
from tests.support.structures import host_structure

_soprano_required = pytest.mark.skipif(
    importlib.util.find_spec("soprano") is None,
    reason="soprano optional dependency not installed",
)


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


SITES = [
    _site("A", (0.5, 0.25, 0.0), -10.0),
    _site("B", (0.0, 0.5, 0.25), -10.0),
    _site("C", (0.25, 0.25, 0.25), -9.5),
]


def test_a_new_workspace_groups_with_connected_components_within_a_quarter_angstrom() -> None:
    clustering = regroup_workspace({"clustering": None}, SITES, host_structure())

    assert clustering["method"] == "connected"
    assert clustering["connected_tolerance_ang"] == 0.25
    assert clustering["result"]["cluster_labels"] == [1, 1, 2]
    assert len(clustering["result"]["clusters"]) == 2
    assert len(clustering["result"]["wyckoff_assignments"]) == 2
    assert clustering["tolerance_scan"] == []
    grouping = clustering["grouping"]
    assert grouping["method"] == "connected"
    assert grouping["connected_tolerance_ang"] == 0.25
    assert grouping["space_group_symbol"] == "Im-3m"


def test_a_workspace_regroups_with_its_own_stored_settings() -> None:
    workspace = {
        "clustering": {
            **DEFAULT_GROUPING_SETTINGS,
            "connected_tolerance_ang": 5.0,
            "tolerance_scan": [[0.0, 3]],
            "grouping": {"method": "connected", "connected_tolerance_ang": 0.25},
        }
    }

    clustering = regroup_workspace(workspace, SITES, host_structure())

    assert clustering["result"]["cluster_labels"] == [1, 1, 1]
    # The scan and the grouping record describe the new sites, not the old ones.
    assert clustering["tolerance_scan"] == []
    assert clustering["grouping"]["connected_tolerance_ang"] == 5.0


@_soprano_required
def test_k_means_stores_the_k_it_used() -> None:
    workspace = {"clustering": {**DEFAULT_GROUPING_SETTINGS, "method": "kmeans", "kmeans_k": 3}}

    clustering = regroup_workspace(workspace, SITES, host_structure())

    assert clustering["method"] == "kmeans"
    assert clustering["kmeans_k"] == len(set(clustering["result"]["cluster_labels"]))
    assert clustering["grouping"]["kmeans_k"] == clustering["kmeans_k"]


def test_a_method_that_cannot_run_falls_back_to_connected_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise UepError("Soprano is not installed.")

    monkeypatch.setattr(analysis_grouping, "cluster_soprano_style", unavailable)
    workspace = {"clustering": {**DEFAULT_GROUPING_SETTINGS, "method": "hier"}}

    clustering = regroup_workspace(workspace, SITES, host_structure())

    assert clustering["method"] == "connected"
    assert clustering["result"]["cluster_labels"] == [1, 1, 2]


def test_a_workspace_without_sites_holds_no_result() -> None:
    clustering = regroup_workspace({}, [], host_structure())

    assert clustering["result"] is None
    assert clustering["method"] == "connected"


def test_grouping_settings_fill_gaps_and_ignore_unreadable_values() -> None:
    assert grouping_settings(None) == DEFAULT_GROUPING_SETTINGS
    settings = grouping_settings(
        {"method": "bogus", "kmeans_k": "x", "hierarchical_t": "0.5", "connected_tolerance_ang": 1}
    )

    assert settings["method"] == "connected"
    assert settings["kmeans_k"] == 4
    assert settings["hierarchical_t"] == 0.5
    assert settings["connected_tolerance_ang"] == 1.0

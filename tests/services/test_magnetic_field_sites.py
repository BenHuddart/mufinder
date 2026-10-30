"""Tests for the muon sites the Magnetic Fields page offers."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from mufinder.domain.models import MuonSite
from mufinder.services.magnetic_field_sites import (
    CUSTOM_SITE_SOURCE,
    RELAXATION_SITE_SOURCE,
    MagneticMuonSiteOption,
    default_muon_site,
    distortion_structure,
    muon_site_options,
    relaxation_cluster_muon_site_options,
    same_muon_site,
    site_cartesian_position,
    site_source_label,
    uep_cluster_muon_site_options,
)
from mufinder.services.magnetic_field_results import UEP_SITE_SOURCE
from mufinder.services.muon_sites_summary import (
    MAGNETIC_FIELDS_SCOPE,
    set_show_every_relaxed_site,
)
from mufinder.services.project_summary import RELAXATION_ANALYSIS_WORKSPACES_KEY
from mufinder.services.project_survey import UEP_ANALYSIS_STATES_KEY
from tests.support.magnetic_field_sites import (
    RELAXATION_WORKSPACE_NAME,
    RELAXED_SITE,
    displaced_iron_supercell,
    feo_host_structure,
    relaxation_cluster_document,
    uep_cluster_document,
)


def test_uep_batch_cluster_is_offered_at_its_optimised_position(tmp_path) -> None:
    document = uep_cluster_document(tmp_path)

    (option,) = uep_cluster_muon_site_options(document, feo_host_structure())

    assert option.source == UEP_SITE_SOURCE
    assert option.label == "UEP cluster 2"
    assert option.source_label == "UEP batch"
    assert option.cluster_label == "Cluster 2"
    assert option.fractional_position == (0.1, 0.2, 0.3)
    assert option.cartesian_position == (0.3, 0.6, 0.9)
    assert option.distortion_structure_id is None
    assert option.properties is not None
    assert option.properties["cluster"]["cluster_index"] == 2


def test_uep_cluster_without_an_optimised_position_uses_the_representative(tmp_path) -> None:
    document = uep_cluster_document(tmp_path)
    (state,) = document.ui_state[UEP_ANALYSIS_STATES_KEY].values()
    state["batch_results"] = [{}]
    state["batch_summary_data"]["clusters"][0]["representative_fractional_position"] = [
        0.5,
        0.0,
        0.25,
    ]

    (option,) = uep_cluster_muon_site_options(document, feo_host_structure())

    assert option.fractional_position == (0.5, 0.0, 0.25)
    # Without a stored Cartesian position it is computed in the 3 Å cube.
    assert option.cartesian_position == (1.5, 0.0, 0.75)


@pytest.mark.parametrize(
    "summary",
    [
        {"clusters": [{"cluster_index": 1, "representative_index": 5}]},
        {"clusters": [{"cluster_index": 1, "representative_index": "first"}]},
        {"clusters": "not a list"},
        {"clusters": [["not", "a", "cluster"]]},
    ],
)
def test_malformed_uep_clusters_are_skipped(tmp_path, summary) -> None:
    document = uep_cluster_document(tmp_path)
    (state,) = document.ui_state[UEP_ANALYSIS_STATES_KEY].values()
    state["batch_summary_data"] = summary

    assert uep_cluster_muon_site_options(document, feo_host_structure()) == []


@pytest.mark.parametrize("method", ["uep_cluster", "uep_cluster_member"])
def test_generated_uep_sites_are_offered_for_every_uep_cluster_method(tmp_path, method) -> None:
    document = uep_cluster_document(tmp_path)
    document.ui_state.pop(UEP_ANALYSIS_STATES_KEY)
    site = MuonSite(
        label="Mu 1",
        generation_method=method,
        initial_fractional_position=[0.25, 0.25, 0.25],
        initial_cartesian_position=[0.75, 0.75, 0.75],
        properties={"cluster_index": 3},
    )
    document.sites[site.id] = site
    other = MuonSite(
        label="Random", generation_method="random", initial_fractional_position=[0.0] * 3
    )
    document.sites[other.id] = other

    (option,) = uep_cluster_muon_site_options(document, feo_host_structure())

    assert option.label == "Mu 1"
    assert option.source_label == "Generated UEP site"
    assert option.cluster_label == "Cluster 3"
    assert option.fractional_position == (0.25, 0.25, 0.25)


def test_stored_uep_batch_cluster_site_is_not_offered(tmp_path) -> None:
    """A ``uep_batch_cluster`` site that an older project stored is left off the site list."""

    document = uep_cluster_document(tmp_path)
    document.ui_state.pop(UEP_ANALYSIS_STATES_KEY)
    site = MuonSite(
        label="Legacy batch site",
        generation_method="uep_batch_cluster",
        initial_fractional_position=[0.25, 0.25, 0.25],
        initial_cartesian_position=[0.75, 0.75, 0.75],
    )
    document.sites[site.id] = site
    current = MuonSite(
        label="UEP site",
        generation_method="uep_cluster",
        initial_fractional_position=[0.5, 0.5, 0.5],
        initial_cartesian_position=[1.5, 1.5, 1.5],
    )
    document.sites[current.id] = current

    options = muon_site_options(document, feo_host_structure(), UEP_SITE_SOURCE)

    assert [option.label for option in options] == ["UEP site"]


def test_relaxation_cluster_carries_its_relaxed_structure(tmp_path) -> None:
    relaxed = displaced_iron_supercell()
    document = relaxation_cluster_document(tmp_path, relaxed)

    (option,) = relaxation_cluster_muon_site_options(document, feo_host_structure())

    assert option.source == RELAXATION_SITE_SOURCE
    assert option.label == f"{RELAXATION_WORKSPACE_NAME} cluster 1"
    assert option.source_label == RELAXATION_WORKSPACE_NAME
    assert option.cluster_label == "Cluster 1"
    assert option.fractional_position == RELAXED_SITE
    assert option.cartesian_position == (1.5, 0.0, 0.0)
    assert option.distortion_structure_id == relaxed.id
    assert distortion_structure(document, option) is relaxed


def test_relaxation_cluster_without_a_relaxed_structure_has_no_distortion(tmp_path) -> None:
    relaxed = displaced_iron_supercell()
    document = relaxation_cluster_document(tmp_path, relaxed)
    del document.structures[relaxed.id]

    (option,) = relaxation_cluster_muon_site_options(document, feo_host_structure())

    assert option.distortion_structure_id is None
    assert distortion_structure(document, option) is None


def test_distortion_structure_is_none_once_the_structure_is_removed(tmp_path) -> None:
    relaxed = displaced_iron_supercell()
    document = relaxation_cluster_document(tmp_path, relaxed)
    (option,) = relaxation_cluster_muon_site_options(document, feo_host_structure())

    del document.structures[relaxed.id]

    assert distortion_structure(document, option) is None
    assert distortion_structure(None, option) is None


def test_pinned_only_lists_no_relaxation_cluster_without_a_pin(tmp_path) -> None:
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())

    assert (
        relaxation_cluster_muon_site_options(document, feo_host_structure(), pinned_only=True) == []
    )


@pytest.mark.parametrize("workspaces", [None, "not a list", ["not a workspace"], [{"name": "x"}]])
def test_relaxation_clusters_tolerate_malformed_workspaces(tmp_path, workspaces) -> None:
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    document.ui_state[RELAXATION_ANALYSIS_WORKSPACES_KEY] = workspaces

    assert relaxation_cluster_muon_site_options(document, feo_host_structure()) == []


def test_muon_site_options_by_source(tmp_path) -> None:
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    host = feo_host_structure()

    assert len(muon_site_options(document, host, RELAXATION_SITE_SOURCE)) == 1
    assert muon_site_options(document, host, RELAXATION_SITE_SOURCE, pinned_only=True) == []
    assert muon_site_options(document, host, UEP_SITE_SOURCE) == []
    assert muon_site_options(document, host, CUSTOM_SITE_SOURCE) == []
    assert muon_site_options(None, host, RELAXATION_SITE_SOURCE) == []
    assert muon_site_options(document, None, RELAXATION_SITE_SOURCE) == []


def test_default_site_prefers_relaxation_clusters_to_uep_sites(tmp_path) -> None:
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    set_show_every_relaxed_site(document, MAGNETIC_FIELDS_SCOPE, True)
    uep = uep_cluster_document(tmp_path)
    document.ui_state[UEP_ANALYSIS_STATES_KEY] = uep.ui_state[UEP_ANALYSIS_STATES_KEY]
    document.jobs.update(uep.jobs)
    host = feo_host_structure()

    default = default_muon_site(document, host)

    assert default is not None
    assert default.source == RELAXATION_SITE_SOURCE
    document.ui_state.pop(RELAXATION_ANALYSIS_WORKSPACES_KEY)
    assert default_muon_site(document, host).source == UEP_SITE_SOURCE
    document.ui_state.pop(UEP_ANALYSIS_STATES_KEY)
    assert default_muon_site(document, host) is None
    assert default_muon_site(None, host) is None


def test_same_muon_site_compares_source_label_and_position() -> None:
    site = MagneticMuonSiteOption(source="custom", label="A", fractional_position=(0.1, 0.2, 0.3))

    assert same_muon_site(site, replace(site, fractional_position=(0.1, 0.2, 0.3 + 1e-10)))
    assert same_muon_site(site, replace(site, cluster_label="another cluster"))
    assert not same_muon_site(site, replace(site, fractional_position=(0.1, 0.2, 0.31)))
    assert not same_muon_site(site, replace(site, label="B"))
    assert not same_muon_site(site, replace(site, source=UEP_SITE_SOURCE))


def test_site_source_labels() -> None:
    assert site_source_label(UEP_SITE_SOURCE) == "UEP clustering (unrelaxed estimate)"
    assert site_source_label(RELAXATION_SITE_SOURCE) == "Structural relaxation clustering"
    assert site_source_label(CUSTOM_SITE_SOURCE) == "Custom"
    assert site_source_label("pristine_relaxation") == "Pristine Relaxation"


def test_site_cartesian_position_in_a_skewed_cell() -> None:
    host = feo_host_structure(lattice=[[3.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.5, 4.0]])

    cartesian = site_cartesian_position((0.5, 0.5, 0.25), host)

    np.testing.assert_allclose(cartesian, (2.0, 1.125, 1.0))
    assert isinstance(cartesian, tuple) and all(isinstance(value, float) for value in cartesian)


def test_relaxation_cluster_uses_its_workspace_name_or_a_default(tmp_path) -> None:
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    (workspace,) = document.ui_state[RELAXATION_ANALYSIS_WORKSPACES_KEY]
    workspace["name"] = "   "
    workspace["clustering"]["result"]["clusters"][0].pop("cluster_index")
    workspace["clustering"]["result"]["clusters"][0]["representative_job_id"] = str(uuid4())

    (option,) = relaxation_cluster_muon_site_options(document, feo_host_structure())

    assert option.label == "Relaxation analysis cluster 1"
    assert option.cluster_label == "-"
    assert option.distortion_structure_id is None

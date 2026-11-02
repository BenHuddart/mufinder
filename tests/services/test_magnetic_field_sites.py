"""Tests for the muon sites the Magnetic Fields page offers."""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest

from mufinder.domain.models import MuonSite
from mufinder.science.magnetic_fields import MagneticFieldOptions, calculate_commensurate_fields
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
from mufinder.services.site_generation_service import generate_candidate_sites
from tests.support.magnetic_field_sites import (
    RELAXATION_WORKSPACE_NAME,
    RELAXED_SITE,
    cof2_antiferromagnet,
    cof2_ferromagnet,
    cof2_host,
    displaced_iron_supercell,
    feo_host_structure,
    relaxation_cluster_document,
    uep_cluster_document,
    uep_search_document,
)


def test_uep_batch_cluster_is_offered_at_its_optimised_position(tmp_path) -> None:
    document = uep_cluster_document(tmp_path)

    (option,) = uep_cluster_muon_site_options(document, feo_host_structure())

    assert option.source == UEP_SITE_SOURCE
    assert option.label == "UEP site 2"
    assert option.source_label == "UEP batch"
    assert option.cluster_label == "Site 2"
    assert option.fractional_position == pytest.approx((0.1, 0.2, 0.3))
    assert option.cartesian_position == pytest.approx((0.3, 0.6, 0.9))
    assert option.distortion_structure_id is None
    assert option.properties is not None
    assert option.properties["cluster"]["cluster_index"] == 2


_COF2_MEMBERS = {
    # member position: (antiferromagnet field in T, |B| of a ferromagnet along a in T),
    # both as calculated by main at f60fbf3 at the position main's option gives.
    (0.081633, 0.611304, 0.299606): ((-0.0704, -0.1302, 0.1186), 0.5134),
    (0.111304, 0.418367, 0.200394): ((-0.1302, 0.0704, -0.1186), 0.5178),
}


@pytest.mark.parametrize("member", _COF2_MEMBERS)
def test_a_uep_group_gives_the_field_of_its_lowest_member_as_found(
    real_muesr, tmp_path, member
) -> None:
    """The group's image in the asymmetric unit, which Generation uses, changes no field."""

    host = cof2_host()
    a, c = 4.695, 3.180
    document = uep_cluster_document(tmp_path)
    (state,) = document.ui_state[UEP_ANALYSIS_STATES_KEY].values()
    state["batch_results"] = [
        {
            "optimized_fractional_position": list(member),
            "optimized_position_ang": [member[0] * a, member[1] * a, member[2] * c],
        }
    ]
    state["batch_summary_data"]["clusters"][0].update(
        representative_fractional_position=[0.081633, 0.611304, 0.299606],
        representative_position_ang=[0.081633 * a, 0.611304 * a, 0.299606 * c],
    )

    (option,) = uep_cluster_muon_site_options(document, host)

    assert option.fractional_position == pytest.approx(member)
    assert option.cartesian_position == pytest.approx((member[0] * a, member[1] * a, member[2] * c))
    options = MagneticFieldOptions()
    (afm,) = calculate_commensurate_fields(
        host, cof2_antiferromagnet(), [option.fractional_position], options=options
    )
    (fm,) = calculate_commensurate_fields(
        host, cof2_ferromagnet(), [option.fractional_position], options=options
    )
    field, magnitude = _COF2_MEMBERS[member]
    assert [float(value) for value in afm.total] == pytest.approx(field, abs=5e-5)
    assert float(fm.magnitude) == pytest.approx(magnitude, abs=5e-5)


def _field(host, model, site) -> tuple[list[float], float]:
    (result,) = calculate_commensurate_fields(
        host, model, [site.fractional_position], options=MagneticFieldOptions()
    )
    return [float(value) for value in result.total], float(result.magnitude)


@pytest.mark.parametrize(
    ("job_position", "host_position"),
    [
        # The site that read 1.057 T on a 2x1x1 job, from the job-cell fractional.
        ((0.75, 1.0, 0.75), (0.5, 0.0, 0.75)),
        # A general position, one host cell along the doubled axis.
        ((0.5408165, 0.611304, 0.299606), (0.081633, 0.611304, 0.299606)),
    ],
)
@pytest.mark.parametrize("model", [cof2_antiferromagnet, cof2_ferromagnet])
def test_supercell_and_unit_cell_jobs_give_one_field_for_one_site(
    real_muesr, tmp_path, job_position, host_position, model
) -> None:
    host = cof2_host()
    (supercell,) = uep_cluster_muon_site_options(
        uep_search_document(tmp_path / "super", host, job_position, supercell=(2, 1, 1)), host
    )
    (unit_cell,) = uep_cluster_muon_site_options(
        uep_search_document(tmp_path / "unit", host, host_position), host
    )

    assert supercell.fractional_position == pytest.approx(host_position)
    assert unit_cell.fractional_position == pytest.approx(host_position)
    assert supercell.cartesian_position == pytest.approx(
        site_cartesian_position(host_position, host)
    )
    supercell_field, supercell_magnitude = _field(host, model(), supercell)
    unit_cell_field, unit_cell_magnitude = _field(host, model(), unit_cell)
    assert supercell_field == pytest.approx(unit_cell_field, abs=1e-6)
    assert supercell_magnitude == pytest.approx(unit_cell_magnitude, abs=1e-6)


def test_a_position_outside_the_cell_folds_into_it_and_keeps_its_field(
    real_muesr, tmp_path
) -> None:
    host = cof2_host()
    (outside,) = uep_cluster_muon_site_options(
        uep_search_document(tmp_path / "outside", host, (1.0, 0.5, -0.25)), host
    )

    assert outside.fractional_position == pytest.approx((0.0, 0.5, 0.75))
    _field_vector, magnitude = _field(host, cof2_antiferromagnet(), outside)
    assert magnitude == pytest.approx(0.0, abs=1e-6)


def test_a_site_from_generation_and_the_uep_option_agree(real_muesr, tmp_path) -> None:
    """The site the 2x1x1 job found: 1.057 T from the job-cell fractional, 0 T through Generation."""

    host = cof2_host()
    document = uep_search_document(tmp_path, host, (0.75, 1.0, 0.75), supercell=(2, 1, 1))
    (option,) = uep_cluster_muon_site_options(document, host)
    ((raw_job_id, state),) = document.ui_state[UEP_ANALYSIS_STATES_KEY].items()
    (cluster,) = state["batch_summary_data"]["clusters"]
    generate_candidate_sites(
        document,
        method="uep_clusters",
        parameters={
            "selections": [
                {
                    "source_key": f"{raw_job_id}:cluster:1",
                    "fractional_position": cluster["representative_fractional_position"],
                    "cartesian_position": cluster["representative_position_ang"],
                    "generation_method": "uep_cluster",
                    "label": "UEP Site 1",
                    "properties": {"cluster_index": 1},
                }
            ]
        },
    )

    by_label = {site.label: site for site in uep_cluster_muon_site_options(document, host)}

    assert set(by_label) == {"UEP Site 1", "UEP site 1"}
    for model in (cof2_antiferromagnet(), cof2_ferromagnet()):
        _generated_field, generated = _field(host, model, by_label["UEP Site 1"])
        _option_field, offered = _field(host, model, option)
        # They are two images of one site, whose fields differ by the sum's scatter, 6e-5 T.
        assert offered == pytest.approx(generated, abs=1e-4)
    assert _field(host, cof2_antiferromagnet(), option)[1] == pytest.approx(0.0, abs=1e-6)


def test_a_position_is_read_in_the_source_structures_cell_not_the_hosts(tmp_path) -> None:
    """A search on a relaxed copy of the host keeps its fractional, as Generation does."""

    host = cof2_host()
    relaxed = replace(
        host, id=uuid4(), lattice_vectors=[[4.8, 0.0, 0.0], [0.0, 4.8, 0.0], [0.0, 0.0, 3.2]]
    )
    (option,) = uep_cluster_muon_site_options(
        uep_search_document(tmp_path, host, (0.75, 1.0, 0.75), supercell=(2, 1, 1), source=relaxed),
        host,
    )

    assert option.fractional_position == pytest.approx((0.5, 0.0, 0.75))
    assert option.cartesian_position == pytest.approx(
        site_cartesian_position((0.5, 0.0, 0.75), host)
    )


def test_a_search_on_a_relaxed_supercell_is_read_in_the_host_cell(tmp_path) -> None:
    host = cof2_host()
    relaxed = replace(
        host, id=uuid4(), lattice_vectors=[[9.39, 0.0, 0.0], [0.0, 4.695, 0.0], [0.0, 0.0, 3.18]]
    )
    (option,) = uep_cluster_muon_site_options(
        uep_search_document(tmp_path, host, (0.75, 1.0, 0.75), source=relaxed), host
    )

    assert option.fractional_position == pytest.approx((0.5, 0.0, 0.75))


def test_a_position_just_below_a_cell_face_reads_as_on_the_face(tmp_path) -> None:
    host = cof2_host()
    (option,) = uep_cluster_muon_site_options(
        uep_search_document(tmp_path, host, (1.0 - 1e-9, 0.5, -1e-9)), host
    )

    assert option.fractional_position == pytest.approx((0.0, 0.5, 0.0))


def test_a_lattice_that_cannot_be_inverted_does_not_stop_the_site_list(tmp_path) -> None:
    host = cof2_host()
    document = uep_search_document(tmp_path, host, (0.5, 0.0, 0.75))
    host.lattice_vectors = [[0.0, 0.0, 0.0]] * 3
    (state,) = document.ui_state[UEP_ANALYSIS_STATES_KEY].values()

    (option,) = uep_cluster_muon_site_options(document, host)

    assert option.fractional_position == pytest.approx(
        state["batch_summary_data"]["clusters"][0]["representative_fractional_position"]
    )


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


def test_site_without_generation_method_is_not_offered_as_a_uep_site(tmp_path) -> None:
    document = uep_cluster_document(tmp_path)
    document.ui_state.pop(UEP_ANALYSIS_STATES_KEY)
    site = MuonSite(label="No method", initial_fractional_position=[0.25, 0.25, 0.25])
    document.sites[site.id] = site

    assert uep_cluster_muon_site_options(document, feo_host_structure()) == []


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
    assert site_source_label(UEP_SITE_SOURCE) == "UEP sites (unrelaxed estimate)"
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


def test_a_generated_uep_site_reads_as_what_it_was_made_from(tmp_path) -> None:
    document = uep_cluster_document(tmp_path)
    for label, method, index in (
        ("UEP Site 1", "symmetry_distance", 1),
        ("UEP Cluster 3", "hier", 3),
    ):
        site = MuonSite(
            label=label,
            generation_method="uep_cluster",
            initial_fractional_position=[0.25, 0.25, 0.25],
            initial_cartesian_position=[0.75, 0.75, 0.75],
            properties={"cluster_index": index, "clustering_method": method},
        )
        document.sites[site.id] = site

    options = {
        option.label: option.cluster_label
        for option in uep_cluster_muon_site_options(document, feo_host_structure())
    }

    assert options["UEP Site 1"] == "Site 1"
    assert options["UEP Cluster 3"] == "Cluster 3"


def test_a_saved_uep_cluster_selection_is_the_same_site_as_its_uep_site_option(tmp_path) -> None:
    document = uep_cluster_document(tmp_path)
    (option,) = uep_cluster_muon_site_options(document, feo_host_structure())

    saved = replace(option, label="UEP cluster 2", cluster_label="Cluster 2")

    assert same_muon_site(saved, option)
    assert not same_muon_site(replace(saved, label="UEP cluster 3"), option)

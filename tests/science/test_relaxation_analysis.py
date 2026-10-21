"""Tests for structural-relaxation analysis helpers."""

from __future__ import annotations

from functools import partial
from uuid import uuid4

import pytest

from mufinder.domain.models import Structure
from mufinder.science import relaxation_analysis
from mufinder.science.lattice import minimum_image_distance
from mufinder.science.relaxation_analysis import (
    RelaxedMuonSite,
    RelaxationClusterSummary,
    analyze_displacements,
    anchored_sites,
    cluster_connected_components,
    folded_muon_position_from_structure,
    grouped_relative_energies,
    muon_fractional_position_from_cell,
    normalize_site_energies,
    scan_connected_component_tolerances,
)
from mufinder.science.wyckoff_analysis import assign_cluster_wyckoff_sites
from tests.support.structures import host_structure


_host_structure = partial(
    host_structure,
    lattice=[[10.0, 0.0, 0.0], [0.1, 9.0, 0.0], [0.2, 0.3, 8.0]],
    species=("Si", "O", "C"),
    positions=((0.0, 0.0, 0.0), (0.31, 0.23, 0.19), (0.67, 0.41, 0.83)),
)


def _site(label: str, frac: tuple[float, float, float], energy: float) -> RelaxedMuonSite:
    host = _host_structure()
    cart = (
        frac[0] * host.lattice_vectors[0][0] + frac[1] * host.lattice_vectors[1][0] + frac[2] * host.lattice_vectors[2][0],
        frac[0] * host.lattice_vectors[0][1] + frac[1] * host.lattice_vectors[1][1] + frac[2] * host.lattice_vectors[2][1],
        frac[0] * host.lattice_vectors[0][2] + frac[1] * host.lattice_vectors[1][2] + frac[2] * host.lattice_vectors[2][2],
    )
    return RelaxedMuonSite(
        job_id=uuid4(),
        job_label=label,
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=frac,
        cartesian_position_ang=cart,
        total_energy_ev=energy,
    )


def _bcc_host_structure() -> Structure:
    return Structure(
        label="BCC",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _bcc_site(label: str, frac: tuple[float, float, float], energy: float) -> RelaxedMuonSite:
    host = _bcc_host_structure()
    cart = (
        frac[0] * host.lattice_vectors[0][0],
        frac[1] * host.lattice_vectors[1][1],
        frac[2] * host.lattice_vectors[2][2],
    )
    return RelaxedMuonSite(
        job_id=uuid4(),
        job_label=label,
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=frac,
        cartesian_position_ang=cart,
        total_energy_ev=energy,
    )


def test_normalize_site_energies_sorts_and_sets_relative_values() -> None:
    sites = [
        _site("high", (0.8, 0.8, 0.8), -8.0),
        _site("low", (0.1, 0.1, 0.1), -10.0),
    ]

    ordered = normalize_site_energies(sites)

    assert [site.job_label for site in ordered] == ["low", "high"]
    assert ordered[0].relative_energy_ev == pytest.approx(0.0)
    assert ordered[1].relative_energy_ev == pytest.approx(2.0)


def test_connected_components_clusters_sites_within_distance_tolerance() -> None:
    host = _host_structure()
    sites = [
        _site("a", (0.100, 0.100, 0.100), -10.0),
        _site("b", (0.112, 0.104, 0.100), -9.9),
        _site("c", (0.800, 0.800, 0.800), -9.0),
    ]

    result = cluster_connected_components(sites, host, tolerance_ang=0.2)

    assert result.cluster_labels[0] == result.cluster_labels[1]
    assert result.cluster_labels[2] != result.cluster_labels[0]
    assert [cluster.size for cluster in result.clusters] == [2, 1]
    assert result.clusters[0].representative_index == 0


def test_tolerance_scan_reports_stable_cluster_count_regions() -> None:
    host = _host_structure()
    sites = [
        _site("a", (0.100, 0.100, 0.100), -10.0),
        _site("b", (0.112, 0.104, 0.100), -9.9),
        _site("c", (0.800, 0.800, 0.800), -9.0),
    ]

    scan = scan_connected_component_tolerances(sites, host, start_ang=0.0, stop_ang=0.4, points=5)

    assert scan[0] == (0.0, 3)
    assert scan[-1] == (0.4, 2)


def test_anchoring_keeps_equivalent_sites_close_to_cluster_representative() -> None:
    host = _host_structure()
    sites = [
        _site("a", (0.02, 0.02, 0.02), -10.0),
        _site("b", (1.02, 0.02, 0.02), -9.9),
    ]

    anchored = anchored_sites(sites, host, [1, 1])

    assert anchored[1].shifted is False
    assert anchored[1].shifted_fractional_position == pytest.approx((0.02, 0.02, 0.02))


def test_folded_muon_position_reads_last_h_mu_species() -> None:
    structure = Structure(
        label="Relaxed",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [1.2, -0.1, 0.5]],
    )

    assert folded_muon_position_from_structure(structure) == pytest.approx((0.2, 0.9, 0.5))


def test_muon_fractional_position_from_cell_reads_last_h_mu_unfolded() -> None:
    cell = Structure(
        label="Relaxed",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["H:mu", "Si", " h:MU "],
        fractional_positions=[[0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [1.2, -0.1, 0.5]],
    )

    assert muon_fractional_position_from_cell(cell) == pytest.approx((1.2, -0.1, 0.5))


def test_muon_fractional_position_from_cell_rejects_missing_or_malformed_muon() -> None:
    class _Cell:
        def __init__(self, species, positions):
            self.species = species
            self.fractional_positions = positions

    assert muon_fractional_position_from_cell(_Cell(["Si"], [[0.0, 0.0, 0.0]])) is None
    assert muon_fractional_position_from_cell(_Cell(["H:mu", "H:mu"], [[0.1, 0.2, 0.3], [0.5, 0.5]])) is None
    assert muon_fractional_position_from_cell(object()) is None


def test_displacement_analysis_matches_host_atoms_and_reports_radial_motion() -> None:
    comparison = Structure(
        label="Host",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["Si"],
        fractional_positions=[[0.2, 0.5, 0.5]],
    )
    relaxed = Structure(
        label="Relaxed",
        lattice_vectors=comparison.lattice_vectors,
        species=["Si", "H:mu"],
        fractional_positions=[[0.25, 0.5, 0.5], [0.0, 0.5, 0.5]],
    )

    result = analyze_displacements(
        relaxed_structure=relaxed,
        comparison_structure=comparison,
        muon_fractional_position=(0.0, 0.5, 0.5),
    )

    assert len(result.points) == 1
    assert result.points[0].distance_from_muon_ang == pytest.approx(2.0)
    assert result.points[0].displacement_ang == pytest.approx(0.5)
    assert result.points[0].radial_displacement_ang == pytest.approx(0.5)


def test_displacement_analysis_converts_host_frame_muon_position_for_supercells() -> None:
    comparison = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["K"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    comparison_supercell = Structure(
        label="Host supercell",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["K"] * 8,
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
        ],
    )
    relaxed = Structure(
        label="Relaxed supercell",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["K"] * 8 + ["H:mu"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 0.5],
            [0.1, 0.1, 0.1],
        ],
    )

    host_frame_muon = (0.2, 0.2, 0.2)
    supercell_frame_muon = (0.1, 0.1, 0.1)

    primitive_host_result = analyze_displacements(
        relaxed_structure=relaxed,
        comparison_structure=comparison,
        muon_fractional_position=host_frame_muon,
    )
    supercell_host_result = analyze_displacements(
        relaxed_structure=relaxed,
        comparison_structure=comparison_supercell,
        muon_fractional_position=supercell_frame_muon,
    )

    assert len(primitive_host_result.points) == 8
    assert primitive_host_result.points[0].distance_from_muon_ang == pytest.approx(supercell_host_result.points[0].distance_from_muon_ang)
    assert primitive_host_result.points[0].distance_from_muon_ang == pytest.approx(10.0 * (3.0 ** 0.5) * 0.1)
    assert primitive_host_result.points[0].comparison_index == supercell_host_result.points[0].comparison_index


def test_displacement_analysis_measures_from_the_supercell_image_the_muon_relaxed_to() -> None:
    comparison = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["K"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    pristine = [[x, y, z] for x in (0.0, 0.5) for y in (0.0, 0.5) for z in (0.0, 0.5)]
    relaxed_positions = [list(position) for position in pristine]
    # The muon relaxed in the second host cell along a, at host (1.2, 0.2, 0.2),
    # and pushed the K at supercell (0.5, 0, 0) away from it.
    relaxed_positions[pristine.index([0.5, 0.0, 0.0])] = [0.48, -0.02, -0.02]
    relaxed = Structure(
        label="Relaxed supercell",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["K"] * 8 + ["H:mu"],
        fractional_positions=[*relaxed_positions, [0.6, 0.1, 0.1]],
    )

    result = analyze_displacements(
        relaxed_structure=relaxed,
        comparison_structure=comparison,
        muon_fractional_position=(0.2, 0.2, 0.2),
    )

    nearest = result.points[0]
    assert nearest.distance_from_muon_ang == pytest.approx(3.0**0.5)
    assert nearest.radial_displacement_ang == pytest.approx(0.2 * 3.0**0.5)


def test_wyckoff_assignment_finds_high_symmetry_site_for_cluster() -> None:
    host = _bcc_host_structure()
    sites = [
        _bcc_site("a", (0.49, 0.01, 0.00), -10.0),
        _bcc_site("b", (0.51, 0.00, 0.02), -9.9),
    ]
    result = cluster_connected_components(sites, host, tolerance_ang=0.3)

    assignments = assign_cluster_wyckoff_sites(sites, result.clusters, host, match_tolerance_ang=0.25)

    assert len(assignments) == 1
    assert assignments[0].match.matched is True
    assert assignments[0].match.label == "6b"
    assert assignments[0].match.fractional_position == pytest.approx((0.5, 0.0, 0.0))
    assert assignments[0].match.site_symmetry_symbol == "4/mm.m"


def test_wyckoff_assignment_handles_variable_coordinate_special_site() -> None:
    host = _bcc_host_structure()
    sites = [
        _bcc_site("a", (0.24, 0.01, 0.00), -10.0),
        _bcc_site("b", (0.26, 0.00, 0.01), -9.9),
    ]
    result = cluster_connected_components(sites, host, tolerance_ang=0.3)

    assignments = assign_cluster_wyckoff_sites(sites, result.clusters, host, match_tolerance_ang=0.2)

    assert assignments[0].match.matched is True
    assert assignments[0].match.label == "12e"
    assert assignments[0].match.fractional_position == pytest.approx((0.25, 0.0, 0.0))


def test_wyckoff_assignment_requires_all_cluster_members_to_match() -> None:
    host = _bcc_host_structure()
    sites = [
        _bcc_site("a", (0.49, 0.01, 0.00), -10.0),
        _bcc_site("b", (0.51, 0.00, 0.02), -9.9),
        _bcc_site("outlier", (0.18, 0.18, 0.18), -9.8),
    ]
    cluster = RelaxationClusterSummary(
        cluster_index=1,
        size=3,
        representative_index=0,
        representative_job_id=sites[0].job_id,
        representative_fractional_position=sites[0].folded_fractional_position,
        representative_cartesian_position_ang=sites[0].cartesian_position_ang,
        minimum_relative_energy_ev=0.0,
        average_relative_energy_ev=0.1,
        standard_deviation_ev=0.1,
        member_indexes=(0, 1, 2),
    )

    assignments = assign_cluster_wyckoff_sites(sites, [cluster], host, match_tolerance_ang=0.2)

    assert assignments[0].match.label != "6b"


def test_wyckoff_assignment_filters_occupied_host_sites() -> None:
    host = _bcc_host_structure()
    site = _bcc_site("occupied", (0.01, 0.00, 0.00), -10.0)
    cluster = RelaxationClusterSummary(
        cluster_index=1,
        size=1,
        representative_index=0,
        representative_job_id=site.job_id,
        representative_fractional_position=site.folded_fractional_position,
        representative_cartesian_position_ang=site.cartesian_position_ang,
        minimum_relative_energy_ev=0.0,
        average_relative_energy_ev=0.0,
        standard_deviation_ev=0.0,
        member_indexes=(0,),
    )

    assignments = assign_cluster_wyckoff_sites(
        [site],
        [cluster],
        host,
        match_tolerance_ang=0.2,
        occupancy_tolerance_ang=0.08,
    )

    assert assignments[0].match.matched is False
    assert "No unoccupied Wyckoff position" in assignments[0].match.warnings[0]


def test_wyckoff_assignment_reports_unavailable_symmetry_for_unknown_species() -> None:
    host = _bcc_host_structure()
    host.species = ["Xx", "Si"]
    site = _bcc_site("a", (0.49, 0.01, 0.00), -10.0)
    cluster = RelaxationClusterSummary(
        cluster_index=1,
        size=1,
        representative_index=0,
        representative_job_id=site.job_id,
        representative_fractional_position=site.folded_fractional_position,
        representative_cartesian_position_ang=site.cartesian_position_ang,
        minimum_relative_energy_ev=0.0,
        average_relative_energy_ev=0.0,
        standard_deviation_ev=0.0,
        member_indexes=(0,),
    )

    assignments = assign_cluster_wyckoff_sites([site], [cluster], host)

    assert assignments[0].match.matched is False
    assert "species" in assignments[0].match.warnings[0]


def test_tolerance_scan_measures_symmetry_distances_once(monkeypatch: pytest.MonkeyPatch) -> None:
    host = _host_structure()
    sites = [
        _site("a", (0.100, 0.100, 0.100), -10.0),
        _site("b", (0.112, 0.104, 0.100), -9.9),
        _site("c", (0.800, 0.800, 0.800), -9.0),
    ]
    calls: list[int] = []
    real = relaxation_analysis.symmetry_distance_matrix

    def counting(ordered_sites, host_structure, *, symprec):
        calls.append(len(ordered_sites))
        return real(ordered_sites, host_structure, symprec=symprec)

    monkeypatch.setattr(relaxation_analysis, "symmetry_distance_matrix", counting)

    scan = scan_connected_component_tolerances(sites, host, start_ang=0.0, stop_ang=0.4, points=5)

    assert calls == [3]
    # Each row agrees with clustering at that tolerance on its own.
    for tolerance, count in scan:
        assert count == len(cluster_connected_components(sites, host, tolerance_ang=tolerance).clusters)


def test_symmetry_distance_falls_back_to_direct_distance_without_equivalents(monkeypatch: pytest.MonkeyPatch) -> None:
    host = _host_structure()
    sites = [_site("a", (0.1, 0.1, 0.1), -10.0), _site("b", (0.9, 0.1, 0.1), -9.0)]
    monkeypatch.setattr(relaxation_analysis, "symmetry_equivalent_positions", lambda *args, **kwargs: [])

    matrix = relaxation_analysis.symmetry_distance_matrix(sites, host)

    expected = minimum_image_distance((0.1, 0.1, 0.1), (0.9, 0.1, 0.1), host.lattice_vectors)
    assert matrix == ((0.0, expected), (expected, 0.0))
def test_grouped_relative_energies_offsets_each_group_by_its_own_minimum() -> None:
    relative = grouped_relative_energies(
        {
            "a": ("positive_muon", -10.0),
            "b": ("positive_muon", -9.75),
            "c": ("muonium", -12.0),
            "d": ("muonium", -11.5),
        }
    )

    assert relative == {
        "a": pytest.approx(0.0),
        "b": pytest.approx(0.25),
        "c": pytest.approx(0.0),
        "d": pytest.approx(0.5),
    }


def test_grouped_relative_energies_maps_missing_and_non_finite_energies_to_none() -> None:
    relative = grouped_relative_energies(
        {
            "a": ("g", -1.0),
            "missing": ("g", None),
            "nan": ("g", float("nan")),
            "lonely": ("empty", None),
        }
    )

    assert relative == {"a": pytest.approx(0.0), "missing": None, "nan": None, "lonely": None}

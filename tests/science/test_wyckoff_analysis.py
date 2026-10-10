"""Golden-value tests pinning Wyckoff assignments for representative hosts.

The expected values were recorded from the original (unoptimised) implementation
of :func:`assign_cluster_wyckoff_sites`, so any performance work on the module has
to reproduce them exactly: labels, positions, multiplicities, site symmetries,
special flags, and match distances.
"""

from __future__ import annotations

from itertools import combinations
from uuid import UUID

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.relaxation_analysis import (
    RelaxedMuonSite,
    RelaxationClusterSummary,
    _pbc_distance,
    _wrap_fractional,
    cluster_connected_components,
)
from mufinder.science.site_generation import lattice_frac_to_cart
from mufinder.science import wyckoff_analysis
from mufinder.science.spglib_compat import get_crystal_symmetry
from mufinder.science.wyckoff_analysis import (
    _deduplicate_positions,
    _host_cell,
    _operation_images,
    _orbit_from_images,
    _project_nearby_special_candidates,
    _project_to_constraints,
    assign_cluster_wyckoff_sites,
)
from tests.support.symmetry_hosts import (
    bcc_host as _bcc_host,
    fcc_primitive_host as _fcc_primitive_host,
    hcp_host as _hcp_host,
    rutile_host as _rutile_host,
    triclinic_host as _triclinic_host,
)


_HOSTS = {
    "bcc": (
        _bcc_host,
        [
            (0.10, 0.10, 0.10),
            (0.12, 0.10, 0.10),
            (0.30, 0.30, 0.30),
            (0.32, 0.30, 0.31),
            (0.60, 0.20, 0.40),
            (0.61, 0.21, 0.40),
            (0.80, 0.80, 0.10),
            (0.25, 0.75, 0.50),
            (0.49, 0.01, 0.00),
            (0.51, 0.00, 0.02),
            (0.24, 0.01, 0.00),
        ],
        0.3,
    ),
    "triclinic": (_triclinic_host, [(0.10, 0.10, 0.10), (0.112, 0.104, 0.100), (0.5, 0.5, 0.5), (0.8, 0.8, 0.8)], 0.2),
    "hcp": (
        _hcp_host,
        [(0.01, 0.00, 0.01), (0.0, 0.0, 0.5), (1.0 / 3.0, 2.0 / 3.0, 0.62), (0.34, 0.66, 0.63), (0.5, 0.0, 0.0), (0.2, 0.1, 0.3)],
        0.3,
    ),
    "rutile": (
        _rutile_host,
        [(0.5, 0.0, 0.0), (0.5, 0.01, 0.49), (0.0, 0.5, 0.25), (0.3, 0.0, 0.5), (0.12, 0.27, 0.41)],
        0.3,
    ),
    "fcc": (_fcc_primitive_host, [(0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.26, 0.25, 0.24), (0.1, 0.3, 0.2)], 0.3),
}


def _site(host: Structure, index: int, fractional: tuple[float, float, float]) -> RelaxedMuonSite:
    return RelaxedMuonSite(
        job_id=UUID(int=index + 1),
        job_label=f"site-{index}",
        charge_state="positive_muon",
        structure_id=UUID(int=1000 + index),
        site_id=None,
        folded_fractional_position=fractional,
        cartesian_position_ang=tuple(lattice_frac_to_cart(fractional, host.lattice_vectors)),
        total_energy_ev=-10.0 + 0.05 * index,
    )


def _singleton(sites: list[RelaxedMuonSite], index: int) -> RelaxationClusterSummary:
    site = sites[index]
    return RelaxationClusterSummary(
        cluster_index=index + 1,
        size=1,
        representative_index=index,
        representative_job_id=site.job_id,
        representative_fractional_position=site.folded_fractional_position,
        representative_cartesian_position_ang=site.cartesian_position_ang,
        minimum_relative_energy_ev=0.0,
        average_relative_energy_ev=0.0,
        standard_deviation_ev=0.0,
        member_indexes=(index,),
    )


def _case(host_name: str, grouping: str):
    factory, fractions, tolerance = _HOSTS[host_name]
    host = factory()
    sites = [_site(host, index, fraction) for index, fraction in enumerate(fractions)]
    if grouping == "connected":
        clusters = cluster_connected_components(sites, host, tolerance_ang=tolerance).clusters
    else:
        clusters = [_singleton(sites, index) for index in range(len(sites))]
    return host, sites, clusters


# (cluster index, label, position, site symmetry, special, max distance Å, mean distance Å)
_EXPECTED = {
    ("bcc", "connected"): [
        (1, "16f", (0.10333333333333333, 0.10333333333333333, 0.10333333333333333), ".3m", True, 0.08660254037844384, 0.05773502691896254),
        (2, "16f", (0.305, 0.305, 0.305), ".3m", True, 0.08291561975888508, 0.06310844497405352),
        (3, "48k", (0.6025, 0.205, 0.3975), "..m", True, 0.04677071733467394, 0.038694669559731845),
        (4, "48k", (0.8, 0.8, 0.1), "..m", True, 0.0, 0.0),
        (5, "24h", (0.25, 0.75, 0.5), "m.m2", True, 0.0, 0.0),
        (6, "6b", (0.5, 0.0, 0.0), "4/mm.m", True, 0.11180339887498951, 0.09125703849682215),
        (7, "12e", (0.24, 0.0, 0.0), "4m.m", True, 0.05, 0.05),
    ],
    ("bcc", "singles"): [
        (1, "16f", (0.1, 0.1, 0.1), ".3m", True, 0.0, 0.0),
        (2, "16f", (0.10666666666666666, 0.10666666666666667, 0.10666666666666667), ".3m", True, 0.08164965809277262, 0.08164965809277262),
        (3, "16f", (0.3, 0.3, 0.3), ".3m", True, 0.0, 0.0),
        (4, "16f", (0.31, 0.31, 0.31), ".3m", True, 0.07071067811865482, 0.07071067811865482),
        (5, "48k", (0.6, 0.2, 0.4), "..m", True, 0.0, 0.0),
        (6, "48k", (0.605, 0.21, 0.395), "..m", True, 0.03535533905932741, 0.03535533905932741),
        (7, "48k", (0.8, 0.8, 0.1), "..m", True, 0.0, 0.0),
        (8, "24h", (0.25, 0.75, 0.5), "m.m2", True, 0.0, 0.0),
        (9, "6b", (0.5, 0.0, 0.0), "4/mm.m", True, 0.07071067811865478, 0.07071067811865478),
        (10, "6b", (0.5, 0.0, 0.0), "4/mm.m", True, 0.11180339887498951, 0.11180339887498951),
        (11, "12e", (0.24, 0.0, 0.0), "4m.m", True, 0.05, 0.05),
    ],
    ("triclinic", "connected"): [
        (1, "1a", (0.106, 0.102, 0.1), "1", False, 0.0628334305923209, 0.0628334305923208),
        (2, "1a", (0.5, 0.5, 0.5), "1", False, 0.0, 0.0),
        (3, "1a", (0.8, 0.8, 0.8), "1", False, 0.0, 0.0),
    ],
    ("triclinic", "singles"): [
        (1, "1a", (0.1, 0.1, 0.1), "1", False, 0.0, 0.0),
        (2, "1a", (0.112, 0.104, 0.1), "1", False, 0.0, 0.0),
        (3, "1a", (0.5, 0.5, 0.5), "1", False, 0.0, 0.0),
        (4, "1a", (0.8, 0.8, 0.8), "1", False, 0.0, 0.0),
    ],
    ("hcp", "connected"): [
        (1, "2a", (0.0, 0.0, 0.0), "-3m.", True, 0.06119493443088243, 0.030597467215441307),
        (2, "4f", (1.0 / 3.0, 2.0 / 3.0, 0.625), "3m.", True, 0.04530433202244567, 0.03567716601122285),
        (3, "6g", (0.5, 0.0, 0.0), ".2/m.", True, 0.0, 0.0),
        (4, "12k", (0.2, 0.1, 0.3), ".m.", True, 0.0, 0.0),
    ],
    ("hcp", "singles"): [
        (1, "2a", (0.0, 0.0, 0.0), "-3m.", True, 0.06119493443088243, 0.06119493443088243),
        (2, "2a", (0.0, 0.0, 0.5), "-3m.", True, 0.0, 0.0),
        (3, "4f", (1.0 / 3.0, 2.0 / 3.0, 0.62), "3m.", True, 0.0, 0.0),
        (4, "4f", (1.0 / 3.0, 2.0 / 3.0, 0.63), "3m.", True, 0.03706588728197406, 0.03706588728197406),
        (5, "6g", (0.5, 0.0, 0.0), ".2/m.", True, 0.0, 0.0),
        (6, "12k", (0.2, 0.1, 0.3), ".m.", True, 0.0, 0.0),
    ],
    ("rutile", "connected"): [
        (1, "4c", (0.5, 0.0, 0.0), "2/m..", True, 0.05464477742657577, 0.027322388713287887),
        (2, "4d", (0.0, 0.5, 0.25), "-4..", True, 0.0, 0.0),
        (3, "8i", (0.3, 0.0, 0.5), "m..", True, 0.0, 0.0),
        (4, "16k", (0.12, 0.25, 0.41), "1", False, 0.09188000000000009, 0.09188000000000009),
    ],
    ("rutile", "singles"): [
        (1, "4c", (0.5, 0.0, 0.0), "2/m..", True, 0.0, 0.0),
        (2, "4c", (0.5, 0.0, 0.5), "2/m..", True, 0.05464477742657574, 0.05464477742657574),
        (3, "4d", (0.0, 0.5, 0.25), "-4..", True, 0.0, 0.0),
        (4, "8i", (0.3, 0.0, 0.5), "m..", True, 0.0, 0.0),
        (5, "16k", (0.12, 0.25, 0.41), "1", False, 0.09188000000000009, 0.09188000000000009),
    ],
    ("fcc", "connected"): [
        (1, "1b", (0.5, 0.5, 0.5), "m-3m", True, 0.0, 0.0),
        (2, "2c", (0.25, 0.25, 0.25), "-43m", True, 0.025526554800834385, 0.012763277400417192),
        (3, "12g", (1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0), "2.mm", True, 0.19026370588679758, 0.19026370588679758),
    ],
    ("fcc", "singles"): [
        (1, "1b", (0.5, 0.5, 0.5), "m-3m", True, 0.0, 0.0),
        (2, "2c", (0.25, 0.25, 0.25), "-43m", True, 0.0, 0.0),
        (3, "2c", (0.25, 0.25, 0.25), "-43m", True, 0.025526554800834385, 0.025526554800834385),
        (4, "12g", (1.0 / 6.0, 1.0 / 3.0, 1.0 / 6.0), "2.mm", True, 0.19026370588679758, 0.19026370588679758),
    ],
}

_SPACE_GROUPS = {"bcc": (229, "Im-3m"), "triclinic": (1, "P1"), "hcp": (194, "P6_3/mmc"), "rutile": (136, "P4_2/mnm"), "fcc": (225, "Fm-3m")}


@pytest.mark.parametrize(("host_name", "grouping"), sorted(_EXPECTED), ids=lambda value: str(value))
def test_wyckoff_assignments_match_recorded_golden_values(host_name: str, grouping: str) -> None:
    host, sites, clusters = _case(host_name, grouping)

    assignments = assign_cluster_wyckoff_sites(sites, clusters, host, match_tolerance_ang=0.25)

    expected = _EXPECTED[(host_name, grouping)]
    assert [assignment.cluster_index for assignment in assignments] == [row[0] for row in expected]
    for assignment, (_, label, position, site_symbol, special, max_distance, mean_distance) in zip(
        assignments, expected, strict=True
    ):
        match = assignment.match
        assert match.matched is True
        assert match.label == label
        assert match.letter == label.lstrip("0123456789")
        assert match.multiplicity == int(label.rstrip("abcdefghijklmnopqrstuvwxyz"))
        assert match.fractional_position == pytest.approx(position, abs=1e-12)
        assert match.cartesian_position_ang == pytest.approx(lattice_frac_to_cart(position, host.lattice_vectors), abs=1e-9)
        assert match.site_symmetry_symbol == site_symbol
        assert (match.space_group_number, match.space_group_symbol) == _SPACE_GROUPS[host_name]
        assert match.special is special
        assert match.occupied is False
        assert match.max_distance_ang == pytest.approx(max_distance, abs=1e-12)
        assert match.mean_distance_ang == pytest.approx(mean_distance, abs=1e-12)
        assert match.warnings == ()


def test_wyckoff_assignment_reports_cluster_with_no_consistent_site_as_unmatched() -> None:
    host, sites, _ = _case("bcc", "singles")
    # A 6b-like pair plus a 16f-like outlier: no single orbit is within 0.05 Å of all three.
    merged = RelaxationClusterSummary(
        cluster_index=1,
        size=3,
        representative_index=8,
        representative_job_id=sites[8].job_id,
        representative_fractional_position=sites[8].folded_fractional_position,
        representative_cartesian_position_ang=sites[8].cartesian_position_ang,
        minimum_relative_energy_ev=0.0,
        average_relative_energy_ev=0.0,
        standard_deviation_ev=0.0,
        member_indexes=(0, 8, 9),
    )

    assignments = assign_cluster_wyckoff_sites(sites, [merged], host, match_tolerance_ang=0.05)

    assert assignments[0].match.matched is False
    assert assignments[0].match.label == ""
    assert "No unoccupied Wyckoff position" in assignments[0].match.warnings[0]


def test_repeated_wyckoff_assignment_returns_identical_results() -> None:
    host, sites, clusters = _case("bcc", "connected")

    first = assign_cluster_wyckoff_sites(sites, clusters, host, match_tolerance_ang=0.25)
    second = assign_cluster_wyckoff_sites(sites, clusters, host, match_tolerance_ang=0.25)
    tighter = assign_cluster_wyckoff_sites(sites, clusters, host, match_tolerance_ang=0.06)

    assert second == first
    # The tolerance participates in the result, so a different tolerance must not reuse it.
    assert [assignment.match.matched for assignment in tighter] != [assignment.match.matched for assignment in first]


def _host_operations(host: Structure) -> tuple[list[np.ndarray], list[np.ndarray]]:
    symmetry = get_crystal_symmetry(_host_cell(host), symprec=1e-5)
    assert symmetry is not None
    rotations = [np.asarray(rotation, dtype=float) for rotation in symmetry["rotations"]]
    translations = [np.asarray(translation, dtype=float) for translation in symmetry["translations"]]
    return rotations, translations


def _brute_force_special_candidates(reference, rotations, translations, lattice, *, tolerance_ang):
    """Project onto every non-identity operation and every pair of them, as originally implemented."""

    identity = np.eye(3)
    operations = [
        (rotation, translation)
        for rotation, translation in zip(rotations, translations, strict=True)
        if not (np.allclose(rotation, identity) and np.allclose(translation - np.round(translation), 0.0))
    ]
    candidates = [_project_to_constraints(reference, (operation,)) for operation in operations]
    candidates += [_project_to_constraints(reference, pair) for pair in combinations(operations, 2)]
    cutoff = max(tolerance_ang * 2.0, tolerance_ang + 0.1)
    nearby = [
        candidate
        for candidate in candidates
        if candidate is not None and _pbc_distance(candidate, reference, lattice) <= cutoff
    ]
    return _deduplicate_positions(nearby, lattice, tolerance_ang=1e-4)


@pytest.mark.parametrize(
    ("host_factory", "reference"),
    [
        (_bcc_host, (0.51, 0.005, 0.01)),
        (_bcc_host, (0.103, 0.101, 0.099)),
        (_hcp_host, (0.34, 0.66, 0.63)),
        (_rutile_host, (0.49, 0.01, 0.02)),
        (_rutile_host, (0.12, 0.27, 0.41)),
        (_fcc_primitive_host, (0.26, 0.25, 0.24)),
    ],
)
def test_special_candidates_match_brute_force_projection_over_all_operation_pairs(host_factory, reference) -> None:
    host = host_factory()
    rotations, translations = _host_operations(host)

    expected = _brute_force_special_candidates(reference, rotations, translations, host.lattice_vectors, tolerance_ang=0.25)
    actual = _project_nearby_special_candidates(reference, rotations, translations, host.lattice_vectors, tolerance_ang=0.25)

    assert len(expected) > 1
    assert actual == expected


@pytest.mark.parametrize("host_factory", [_bcc_host, _hcp_host, _rutile_host, _fcc_primitive_host])
def test_symmetry_orbit_matches_greedy_pairwise_deduplication(host_factory) -> None:
    host = host_factory()
    rotations, translations = _host_operations(host)
    lattice = host.lattice_vectors

    for point in ((0.5, 0.0, 0.0), (0.25, 0.25, 0.25), (0.12, 0.27, 0.41), (0.0, 0.0, 0.999999999)):
        images = [np.dot(rotation, np.asarray(point)) + translation for rotation, translation in zip(rotations, translations, strict=True)]
        expected: list[tuple[float, float, float]] = []
        for image in images:
            # Wrapped twice, as originally: ``-1e-17 % 1.0`` is ``1.0``, and the second wrap folds it to 0.
            wrapped = _wrap_fractional(_wrap_fractional(image))
            if not any(_pbc_distance(wrapped, existing, lattice) <= 1e-5 for existing in expected):
                expected.append(wrapped)

        orbit = _orbit_from_images(
            _operation_images(point, rotations, translations), lattice, tolerance_ang=1e-5
        )

        assert len(orbit) == len(expected)
        for actual_position, expected_position in zip(orbit, expected, strict=True):
            assert actual_position == pytest.approx(expected_position, abs=1e-12)


@pytest.fixture
def fresh_wyckoff_caches():
    """Keep results computed with monkeypatched helpers out of the module-level caches."""

    wyckoff_analysis._match_cluster.cache_clear()
    wyckoff_analysis._host_symmetry_operations.cache_clear()
    yield
    wyckoff_analysis._match_cluster.cache_clear()
    wyckoff_analysis._host_symmetry_operations.cache_clear()


def _bcc_six_b_cluster():
    host, sites, clusters = _case("bcc", "connected")
    six_b = [cluster for cluster in clusters if cluster.member_indexes == (8, 9)]
    assert len(six_b) == 1
    return host, sites, six_b


@pytest.mark.usefixtures("fresh_wyckoff_caches")
def test_wyckoff_assignment_labels_only_the_best_ranked_candidate(monkeypatch) -> None:
    host, sites, cluster = _bcc_six_b_cluster()
    calls: list[int] = []
    original_label = wyckoff_analysis._label_candidate

    def counting_label(host_cell, orbit, *, symprec):
        calls.append(len(orbit))
        return original_label(host_cell, orbit, symprec=symprec)

    monkeypatch.setattr(wyckoff_analysis, "_label_candidate", counting_label)

    assignments = assign_cluster_wyckoff_sites(sites, cluster, host, match_tolerance_ang=0.25)

    assert assignments[0].match.label == "6b"
    assert calls == [6]


@pytest.mark.usefixtures("fresh_wyckoff_caches")
def test_wyckoff_assignment_falls_back_to_next_ranked_candidate_when_labelling_fails(monkeypatch) -> None:
    host, sites, cluster = _bcc_six_b_cluster()
    original_label = wyckoff_analysis._label_candidate

    def refuse_six_fold_orbits(host_cell, orbit, *, symprec):
        if len(orbit) == 6:
            return None
        return original_label(host_cell, orbit, symprec=symprec)

    monkeypatch.setattr(wyckoff_analysis, "_label_candidate", refuse_six_fold_orbits)

    assignments = assign_cluster_wyckoff_sites(sites, cluster, host, match_tolerance_ang=0.25)

    match = assignments[0].match
    assert match.matched is True
    assert match.multiplicity is not None and match.multiplicity > 6
    assert match.max_distance_ang is not None and match.max_distance_ang <= 0.25


@pytest.mark.usefixtures("fresh_wyckoff_caches")
def test_wyckoff_assignment_is_unmatched_when_no_candidate_can_be_labelled(monkeypatch) -> None:
    host, sites, cluster = _bcc_six_b_cluster()
    monkeypatch.setattr(wyckoff_analysis, "_label_candidate", lambda host_cell, orbit, *, symprec: None)

    assignments = assign_cluster_wyckoff_sites(sites, cluster, host, match_tolerance_ang=0.25)

    assert assignments[0].match.matched is False
    assert "No unoccupied Wyckoff position" in assignments[0].match.warnings[0]


@pytest.mark.usefixtures("fresh_wyckoff_caches")
def test_wyckoff_assignment_reports_unavailable_symmetry_when_spglib_fails(monkeypatch) -> None:
    host, sites, cluster = _bcc_six_b_cluster()
    monkeypatch.setattr(wyckoff_analysis, "get_crystal_symmetry", lambda cell, *, symprec: None)

    assignments = assign_cluster_wyckoff_sites(sites, cluster, host, match_tolerance_ang=0.25)

    assert assignments[0].match.matched is False
    assert "spglib" in assignments[0].match.warnings[0]

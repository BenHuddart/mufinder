"""Tests for periodic muon site-generation algorithms."""

from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pytest

from mufinder.science import site_generation
from mufinder.science.site_generation import (
    SiteGenerationError,
    closest_symmetry_equivalent_position,
    estimate_muon_muon_separation_for_count,
    estimate_random_site_count_from_separation,
    generate_attached_element_sites,
    generate_random_sites,
    generate_random_sites_by_distance,
    lattice_cart_to_frac,
    lattice_frac_to_cart,
    symmetry_equivalent_positions,
)


def _pbc_distance(a: list[float], b: list[float], lattice: list[list[float]]) -> float:
    delta = [a[i] - b[i] for i in range(3)]
    delta = [d - round(d) for d in delta]
    cart = lattice_frac_to_cart(delta, lattice)
    return math.sqrt(sum(v * v for v in cart))


def test_fractional_cartesian_roundtrip() -> None:
    lattice = [[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]
    frac = [0.25, 0.5, 0.75]
    cart = lattice_frac_to_cart(frac, lattice)
    recovered = lattice_cart_to_frac(cart, lattice)

    assert recovered == pytest.approx(frac)


def test_generate_random_sites_respects_distance_constraints() -> None:
    lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    host_positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    min_muon_atom = 0.8
    min_muon_muon = 0.8

    points, attempts = generate_random_sites(
        lattice_vectors=lattice,
        host_positions_frac=host_positions,
        count=20,
        min_muon_muon=min_muon_muon,
        min_muon_atom=min_muon_atom,
        max_attempts=50000,
        seed=7,
    )

    assert len(points) == 20
    assert attempts > 0

    for idx, point in enumerate(points):
        for host in host_positions:
            assert _pbc_distance(point.fractional, host, lattice) >= min_muon_atom
        for other in points[idx + 1 :]:
            assert _pbc_distance(point.fractional, other.fractional, lattice) >= min_muon_muon


def test_symmetry_equivalent_random_sites_reserve_full_orbits() -> None:
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    host_species = ["Si"]
    host_positions = [[0.0, 0.0, 0.0]]

    equivalents = symmetry_equivalent_positions(
        [0.1, 0.2, 0.3],
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions,
    )
    assert len(equivalents) > 1

    points, _ = generate_random_sites(
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions,
        count=3,
        min_muon_muon=1.0,
        min_muon_atom=0.5,
        max_attempts=50000,
        seed=11,
        include_symmetry_equivalents=True,
    )

    orbits = [
        symmetry_equivalent_positions(
            point.fractional,
            lattice_vectors=lattice,
            host_species=host_species,
            host_positions_frac=host_positions,
        )
        for point in points
    ]
    for index, orbit in enumerate(orbits):
        for other_orbit in orbits[index + 1 :]:
            assert min(_pbc_distance(a, b, lattice) for a in orbit for b in other_orbit) >= 1.0


def test_closest_symmetry_equivalent_position_selects_representative_near_anchor() -> None:
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    clustered = closest_symmetry_equivalent_position(
        [0.1, 0.2, 0.3],
        anchor_frac=[0.9, 0.8, 0.7],
        lattice_vectors=lattice,
        host_species=["Si"],
        host_positions_frac=[[0.0, 0.0, 0.0]],
    )

    assert clustered == pytest.approx([0.9, 0.8, 0.7])


def test_generate_attached_element_sites_uses_host_geometry_per_atom() -> None:
    lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    host_species = ["O", "Si", "Si", "O"]
    host_positions = [[0.5, 0.5, 0.5], [0.65, 0.5, 0.5], [0.5, 0.65, 0.5], [0.25, 0.25, 0.25]]

    points, attempts = generate_attached_element_sites(
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions,
        target_element="O",
        muons_per_atom=2,
        bond_distance=1.0,
        min_muon_atom=0.5,
    )

    assert len(points) == 4
    assert attempts >= 4
    for anchor in (host_positions[0], host_positions[3]):
        distances = [_pbc_distance(point.fractional, anchor, lattice) for point in points]
        assert sum(distance == pytest.approx(1.0) for distance in distances) == 2


def test_generate_attached_element_sites_missing_element_error() -> None:
    lattice = [[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]]
    with pytest.raises(SiteGenerationError, match="No host atoms"):
        generate_attached_element_sites(
            lattice_vectors=lattice,
            host_species=["O", "Si"],
            host_positions_frac=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            target_element="Fe",
            muons_per_atom=1,
            bond_distance=1.0,
            min_muon_atom=0.6,
        )


def test_invalid_generation_arguments_raise() -> None:
    lattice = [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]
    with pytest.raises(SiteGenerationError):
        generate_random_sites(
            lattice_vectors=lattice,
            host_positions_frac=[],
            count=0,
            min_muon_muon=0.5,
            min_muon_atom=0.5,
        )

    with pytest.raises(SiteGenerationError):
        generate_attached_element_sites(
            lattice_vectors=lattice,
            host_species=["O"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
            target_element="O",
            muons_per_atom=0,
            bond_distance=1.0,
            min_muon_atom=0.5,
        )


def test_generate_random_sites_raises_when_requested_count_cannot_be_met() -> None:
    lattice = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    with pytest.raises(SiteGenerationError, match="Could only generate"):
        generate_random_sites(
            lattice_vectors=lattice,
            host_positions_frac=[],
            count=200,
            min_muon_muon=0.8,
            min_muon_atom=0.0,
            max_attempts=5000,
            seed=7,
        )


def test_generate_random_sites_can_return_partial_results_when_enabled() -> None:
    lattice = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

    points, attempts = generate_random_sites(
        lattice_vectors=lattice,
        host_positions_frac=[],
        count=200,
        min_muon_muon=0.8,
        min_muon_atom=0.0,
        max_attempts=5000,
        seed=7,
        allow_partial=True,
    )

    assert 0 < len(points) < 200
    assert attempts == 5000


def test_random_generation_reports_progress_as_sites_are_accepted() -> None:
    updates: list[tuple[int, int]] = []

    points, _attempts = generate_random_sites(
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        host_positions_frac=[],
        count=4,
        min_muon_muon=0.8,
        min_muon_atom=0.0,
        max_attempts=5000,
        seed=4,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert len(points) == 4
    assert updates[0] == (0, 4)
    assert updates[-1] == (4, 4)
    assert [completed for completed, _total in updates] == [0, 1, 2, 3, 4]


def test_generate_random_sites_by_distance_fills_until_attempt_limit() -> None:
    lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]
    host_positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    min_muon_atom = 0.8
    min_muon_muon = 1.2

    points, attempts = generate_random_sites_by_distance(
        lattice_vectors=lattice,
        host_positions_frac=host_positions,
        min_muon_muon=min_muon_muon,
        min_muon_atom=min_muon_atom,
        max_attempts=5000,
        seed=7,
    )

    assert len(points) > 1
    assert attempts == 5000

    for idx, point in enumerate(points):
        for host in host_positions:
            assert _pbc_distance(point.fractional, host, lattice) >= min_muon_atom
        for other in points[idx + 1 :]:
            assert _pbc_distance(point.fractional, other.fractional, lattice) >= min_muon_muon


def test_random_generation_by_distance_uses_predicted_progress_total() -> None:
    updates: list[tuple[int, int]] = []

    points, _attempts = generate_random_sites_by_distance(
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        host_positions_frac=[],
        min_muon_muon=1.4,
        min_muon_atom=0.0,
        max_attempts=500,
        seed=5,
        expected_count=5,
        progress_callback=lambda completed, total: updates.append((completed, total)),
    )

    assert len(points) > 0
    assert updates[0] == (0, 5)
    assert updates[-1][0] == len(points)
    assert updates[-1][1] >= 5


def test_estimate_muon_muon_separation_inverts_expected_random_count() -> None:
    lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    target_count = 30

    separation = estimate_muon_muon_separation_for_count(
        lattice_vectors=lattice,
        host_positions_frac=[],
        target_count=target_count,
        min_muon_atom=1.0,
        include_symmetry_equivalents=False,
    )
    estimated_count = estimate_random_site_count_from_separation(
        lattice_vectors=lattice,
        host_positions_frac=[],
        min_muon_muon=separation,
        min_muon_atom=1.0,
        include_symmetry_equivalents=False,
    )

    assert separation == pytest.approx(2.82, rel=0.03)
    assert estimated_count == pytest.approx(target_count, rel=0.02)


def test_generate_attached_element_sites_raises_when_requested_count_cannot_be_met() -> None:
    lattice = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    host_species = ["O"]
    host_positions = [[0.0, 0.0, 0.0]]

    with pytest.raises(SiteGenerationError, match="attached-element candidate"):
        generate_attached_element_sites(
            lattice_vectors=lattice,
            host_species=host_species,
            host_positions_frac=host_positions,
            target_element="O",
            muons_per_atom=1,
            bond_distance=0.5,
            min_muon_atom=0.8,
        )


_CUBIC = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]


def test_deduplication_tolerance_is_inclusive_at_the_exact_pair_distance() -> None:
    first, second = [0.1, 0.2, 0.3], [0.1 + 3e-7, 0.2 - 1e-7, 0.3 + 2e-7]
    pair_distance = site_generation._pbc_distance(second, first, _CUBIC)

    def deduplicate(tolerance: float) -> list[list[float]]:
        return site_generation._deduplicate_fractional_positions([first, second], _CUBIC, tolerance_ang=tolerance)

    # Broadcast norms can be an ulp off, so decisions this close are re-measured with ``_pbc_distance``.
    assert deduplicate(pair_distance) == [first]
    assert deduplicate(math.nextafter(pair_distance, 0.0)) == [first, second]
    assert deduplicate(math.nextafter(pair_distance, 1.0)) == [first]
    assert site_generation._deduplicate_fractional_positions([], _CUBIC) == []


def test_deduplication_keeps_first_occurrence_and_drops_exact_repeats() -> None:
    positions = [[1.25, -0.5, 0.0], [0.5, 0.5, 0.5], [0.25, 0.5, 0.0], [0.5, 0.5, 0.5 + 1e-7]]

    unique = site_generation._deduplicate_fractional_positions(positions, _CUBIC)

    assert unique == [[0.25, 0.5, 0.0], [0.5, 0.5, 0.5]]


def test_host_symmetry_is_detected_once_per_host(monkeypatch: pytest.MonkeyPatch) -> None:
    site_generation._cached_symmetry_operations.cache_clear()
    calls: list[float] = []
    real = site_generation.get_crystal_symmetry

    def counting(cell, *, symprec):
        calls.append(symprec)
        return real(cell, symprec=symprec)

    monkeypatch.setattr(site_generation, "get_crystal_symmetry", counting)
    host = {"lattice_vectors": _CUBIC, "host_species": ["Si", "Si"], "host_positions_frac": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]}
    try:
        first = symmetry_equivalent_positions([0.1, 0.1, 0.1], **host)
        second = symmetry_equivalent_positions([0.3, 0.2, 0.1], **host)
        operations = site_generation._symmetry_operations(**host)
    finally:
        site_generation._cached_symmetry_operations.cache_clear()

    assert len(first) == 16
    assert len(second) == 96
    assert calls == [1e-5]
    assert operations is not None
    # Shared cached arrays must not be writable by callers.
    assert all(not rotation.flags.writeable for rotation in operations[0])


def test_symmetry_equivalents_fall_back_to_input_when_spglib_finds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    site_generation._cached_symmetry_operations.cache_clear()
    monkeypatch.setattr(site_generation, "get_crystal_symmetry", lambda cell, *, symprec: None)
    try:
        equivalents = symmetry_equivalent_positions(
            [1.1, -0.2, 0.3],
            lattice_vectors=_CUBIC,
            host_species=["Si"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
        )
    finally:
        site_generation._cached_symmetry_operations.cache_clear()

    assert equivalents == [pytest.approx([0.1, 0.8, 0.3])]


def test_broadcast_spacing_check_decides_exactly_at_the_limit() -> None:
    point, reference = [0.1, 0.2, 0.3], [0.4, 0.45, 0.9]
    limit = site_generation._pbc_distance(point, reference, _CUBIC)
    points, references = np.asarray([point]), np.asarray([reference])

    # ``_pbc_distance(point, reference) < limit`` decides, so the exact distance is not "closer".
    assert site_generation._any_closer_than(points, references, _CUBIC, limit) is False
    assert site_generation._any_closer_than(points, references, _CUBIC, math.nextafter(limit, 10.0)) is True
    assert site_generation._any_closer_than(points, np.empty((0, 3)), _CUBIC, limit) is False


def test_broadcast_spacing_check_scans_large_inputs_in_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(site_generation, "_DISTANCE_CHECK_BLOCK", 4)
    references = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    far = np.asarray([[0.25, 0.25, 0.0]] * 9)
    near = np.vstack([far, [[0.51, 0.5, 0.5]]])

    assert site_generation._any_closer_than(far, references, _CUBIC, 1.0) is False
    assert site_generation._any_closer_than(near, references, _CUBIC, 1.0) is True


def test_identity_leads_requires_identity_rotation_and_zero_translation() -> None:
    identity, mirror = np.eye(3), np.diag([-1.0, 1.0, 1.0])
    zero, shift = np.zeros(3), np.asarray([0.5, 0.0, 0.0])

    assert site_generation._identity_leads(([identity, mirror], [zero, shift])) is True
    assert site_generation._identity_leads(([mirror, identity], [zero, zero])) is False
    assert site_generation._identity_leads(([identity], [shift])) is False
    assert site_generation._identity_leads(([], [])) is False
    assert site_generation._identity_leads(None) is False


def test_random_generation_checks_whole_orbits_when_identity_is_not_first(monkeypatch: pytest.MonkeyPatch) -> None:
    host = {"lattice_vectors": _CUBIC, "host_species": ["Si", "Si"], "host_positions_frac": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]}
    real = site_generation._symmetry_operations

    def identity_last(**kwargs):
        rotations, translations = real(**kwargs)
        return rotations[::-1], translations[::-1]

    monkeypatch.setattr(site_generation, "_symmetry_operations", identity_last)

    points, _attempts = generate_random_sites_by_distance(
        **host, min_muon_muon=0.8, min_muon_atom=1.0, max_attempts=300, seed=3, include_symmetry_equivalents=True
    )

    assert points
    orbits = [
        site_generation._equivalent_positions_from_operations(
            point.fractional, lattice_vectors=_CUBIC, operations=identity_last(**host)
        )
        for point in points
    ]
    for orbit in orbits:
        for image in orbit:
            assert all(_pbc_distance(image, atom, _CUBIC) >= 1.0 for atom in host["host_positions_frac"])
    for first, second in combinations(range(len(orbits)), 2):
        assert all(_pbc_distance(a, b, _CUBIC) >= 0.8 for a in orbits[first] for b in orbits[second])


def test_random_generation_rejects_malformed_host_positions() -> None:
    with pytest.raises(SiteGenerationError, match="triples"):
        generate_random_sites(
            lattice_vectors=_CUBIC,
            host_positions_frac=[[0.0, 0.0], [0.5, 0.5]],
            count=1,
            min_muon_muon=0.8,
            min_muon_atom=1.0,
        )

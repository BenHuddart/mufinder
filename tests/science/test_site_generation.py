"""Tests for periodic muon site-generation algorithms."""

from __future__ import annotations

import math

import pytest

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

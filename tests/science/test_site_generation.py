"""Tests for periodic muon site-generation algorithms."""

from __future__ import annotations

import math

import pytest

from mufinder.science.site_generation import (
    SiteGenerationError,
    generate_near_element_sites,
    generate_random_sites,
    generate_random_sites_by_distance,
    lattice_cart_to_frac,
    lattice_frac_to_cart,
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


def test_generate_near_element_sites_and_missing_element_error() -> None:
    lattice = [[7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 7.0]]
    host_species = ["O", "Si", "O"]
    host_positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]

    points, _ = generate_near_element_sites(
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions,
        target_element="O",
        count=8,
        shell_min=0.6,
        shell_max=1.2,
        min_muon_muon=0.6,
        min_muon_atom=0.6,
        max_attempts=50000,
        seed=3,
    )
    assert len(points) == 8

    with pytest.raises(SiteGenerationError, match="No host atoms"):
        generate_near_element_sites(
            lattice_vectors=lattice,
            host_species=host_species,
            host_positions_frac=host_positions,
            target_element="Fe",
            count=1,
            shell_min=0.6,
            shell_max=1.2,
            min_muon_muon=0.6,
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
        generate_near_element_sites(
            lattice_vectors=lattice,
            host_species=["O"],
            host_positions_frac=[[0.0, 0.0, 0.0]],
            target_element="O",
            count=1,
            shell_min=1.2,
            shell_max=1.0,
            min_muon_muon=0.5,
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


def test_generate_near_element_sites_raises_when_requested_count_cannot_be_met() -> None:
    lattice = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    host_species = ["O"]
    host_positions = [[0.0, 0.0, 0.0]]

    with pytest.raises(SiteGenerationError, match="Could only generate"):
        generate_near_element_sites(
            lattice_vectors=lattice,
            host_species=host_species,
            host_positions_frac=host_positions,
            target_element="O",
            count=100,
            shell_min=0.9,
            shell_max=1.2,
            min_muon_muon=0.8,
            min_muon_atom=0.8,
            max_attempts=5000,
            seed=3,
        )

"""Tests for reciprocal-space Monkhorst-Pack grid generation helpers."""

from __future__ import annotations

import pytest

from mufinder.science.kpoint_grids import (
    equivalent_grid_for_supercell,
    format_kpoint_grid,
    generate_kpoint_grids,
    grid_from_spacing,
    kpoint_count,
    lattice_parameters,
    lattice_matches,
    reciprocal_lattice_vectors,
    spacing_from_grid,
    supercell_lattice_vectors,
)


def test_reciprocal_lattice_vectors_cubic_cell() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    b1, b2, b3 = reciprocal_lattice_vectors(lattice)

    assert b1 == pytest.approx((1.57079632679, 0.0, 0.0))
    assert b2 == pytest.approx((0.0, 1.57079632679, 0.0))
    assert b3 == pytest.approx((0.0, 0.0, 1.57079632679))


def test_grid_from_spacing_uses_reciprocal_magnitudes_formula() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]
    spacing = 0.1

    grid = grid_from_spacing(lattice, spacing)

    # n_i = floor(|b_i| / (2*pi*spacing)) + 1
    # |b| = [2*pi/4, 2*pi/6, 2*pi/8] => [1.5708, 1.0472, 0.7854]
    # denominator = 2*pi*0.1 = 0.6283
    # n = [3, 2, 2]
    assert grid == (3, 2, 2)


def test_generate_kpoint_grids_increases_density_up_to_max_grid() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]

    grids = generate_kpoint_grids(lattice, 0.1)

    assert grids == ((1, 1, 1), (2, 2, 2), (3, 2, 2))


def test_kpoint_helpers_format_and_count() -> None:
    grid = (4, 5, 6)

    assert kpoint_count(grid) == 120
    assert format_kpoint_grid(grid) == "4 x 5 x 6"


def test_lattice_matches_with_tolerance() -> None:
    lhs = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    rhs = [[4.0 + 1e-7, 0.0, 0.0], [0.0, 5.0 - 1e-7, 0.0], [0.0, 0.0, 6.0]]

    assert lattice_matches(lhs, rhs)
    assert not lattice_matches(lhs, [[4.01, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])


def test_lattice_parameters_return_lengths_and_angles() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]

    a, b, c, alpha, beta, gamma = lattice_parameters(lattice)

    assert a == pytest.approx(4.0)
    assert b == pytest.approx(5.0)
    assert c == pytest.approx(6.0)
    assert alpha == pytest.approx(90.0)
    assert beta == pytest.approx(90.0)
    assert gamma == pytest.approx(90.0)


def test_equivalent_grid_for_supercell_preserves_density() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    reference_grid = (4, 4, 4)
    doubled_a = ((2, 0, 0), (0, 1, 0), (0, 0, 1))

    super_lattice = supercell_lattice_vectors(lattice, doubled_a)
    equivalent = equivalent_grid_for_supercell(lattice, reference_grid, super_lattice)

    assert equivalent == (2, 4, 4)


def test_spacing_from_grid_roundtrip_for_nontrivial_grid() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]
    grid = (3, 2, 2)

    spacing = spacing_from_grid(lattice, grid)
    regenerated = grid_from_spacing(lattice, spacing)

    assert regenerated == grid

"""Tests for reciprocal-space Monkhorst-Pack grid generation helpers."""

from __future__ import annotations

import pytest

from mufinder.science.kpoint_grids import (
    equivalent_grid_for_supercell,
    format_kpoint_grid,
    generate_kpoint_grids,
    grid_from_spacing,
    kpoint_count,
    kpoint_count_for_spacing,
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


def test_equivalent_grid_for_supercell_scales_each_axis_for_unequal_lattice_lengths() -> None:
    lattice = [[3.6, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 5.4]]
    reference_grid = (4, 4, 4)
    doubled_all = ((2, 0, 0), (0, 2, 0), (0, 0, 2))

    super_lattice = supercell_lattice_vectors(lattice, doubled_all)
    equivalent = equivalent_grid_for_supercell(lattice, reference_grid, super_lattice)

    assert equivalent == (2, 2, 2)


def test_equivalent_grid_for_supercell_rounds_up_to_avoid_undersampling() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    reference_grid = (4, 4, 4)
    supercell = ((2, 0, 0), (0, 2, 0), (0, 0, 3))

    super_lattice = supercell_lattice_vectors(lattice, supercell)
    equivalent = equivalent_grid_for_supercell(lattice, reference_grid, super_lattice)

    assert equivalent == (2, 2, 2)


def test_spacing_from_grid_roundtrip_for_nontrivial_grid() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]
    grid = (3, 2, 2)

    spacing = spacing_from_grid(lattice, grid)
    regenerated = grid_from_spacing(lattice, spacing)

    assert regenerated == grid


def test_kpoint_count_for_spacing_matches_grid_product_at_integer_grids() -> None:
    # For a cubic cell with a = 4.0 the reciprocal magnitude is 2*pi / a, so
    # spacing = 1/a should map to n_i_cont = 2 in every direction and a
    # continuous k-point count of 8.0, matching the discrete (2,2,2) grid.
    a = 4.0
    lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    assert kpoint_count_for_spacing(lattice, 1.0 / a) == pytest.approx(8.0)
    # spacing 1/(2a) -> n_i_cont = 3 in each direction -> 27.
    assert kpoint_count_for_spacing(lattice, 1.0 / (2.0 * a)) == pytest.approx(27.0)


def test_kpoint_count_for_spacing_is_monotone_decreasing() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    spacings = [0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02]
    counts = [kpoint_count_for_spacing(lattice, s) for s in spacings]
    # Smaller spacing -> denser grid -> larger continuous count.
    assert counts == sorted(counts)


def test_kpoint_count_for_spacing_tends_to_one_for_huge_spacing() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    # Spacing >> max |b_i| / (2*pi) makes every n_i_cont approach 1; the
    # continuous count then approaches 1.
    assert kpoint_count_for_spacing(lattice, 100.0) == pytest.approx(1.0, abs=0.05)


def test_kpoint_count_for_spacing_rejects_nonpositive_spacing() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError):
        kpoint_count_for_spacing(lattice, 0.0)
    with pytest.raises(ValueError):
        kpoint_count_for_spacing(lattice, -0.1)


def test_kpoint_count_for_spacing_agrees_with_grid_from_spacing_at_castep_boundary() -> None:
    # CASTEP uses floor((|b|/(2*pi*s)) + 1, so at the exact boundary s = 1/a
    # the discrete grid is (2,2,2). The continuous helper should report
    # exactly 8 there too (no floor difference at the integer).
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    grid = grid_from_spacing(lattice, 0.25)
    assert grid == (2, 2, 2)
    assert kpoint_count_for_spacing(lattice, 0.25) == pytest.approx(
        float(grid[0] * grid[1] * grid[2])
    )

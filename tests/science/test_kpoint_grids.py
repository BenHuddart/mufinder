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


def test_spacing_from_grid_uses_castep_max_formula() -> None:
    # CASTEP convention: spacing = max_i( |b_i| / (2*pi * N_i) ).
    # For orthorhombic a=4, b=6, c=8 with grid (3, 2, 2):
    #   max( 1/(4*3), 1/(6*2), 1/(8*2) ) = max(1/12, 1/12, 1/16) = 1/12.
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]
    assert spacing_from_grid(lattice, (3, 2, 2)) == pytest.approx(1.0 / 12.0)


def test_spacing_from_grid_cubic_lif_5x5x5() -> None:
    # Cubic LiF (a=4.0834 Ang), 5x5x5 grid: spacing = 1/(a*5).
    a = 4.0834
    lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    assert spacing_from_grid(lattice, (5, 5, 5)) == pytest.approx(1.0 / (a * 5.0))


def test_spacing_from_grid_hexagonal_alf3_3x3x3() -> None:
    # Hexagonal AlF3 (a=4.872, c=12.466 Ang), 3x3x3 grid.
    # |b_ab| = 4*pi/(a*sqrt(3)); spacing = |b_ab|/(2*pi*3) = 2/(3*a*sqrt(3)).
    import math as _math

    a, c = 4.872, 12.466
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * _math.sqrt(3) / 2.0, 0.0], [0.0, 0.0, c]]
    expected = 2.0 / (3.0 * a * _math.sqrt(3))  # ≈ 0.0790 1/Ang
    assert spacing_from_grid(lattice, (3, 3, 3)) == pytest.approx(expected, rel=1e-4)


def test_spacing_from_grid_hexagonal_alf3_anisotropic_grids_decrease() -> None:
    # Previously 6x6x5, 7x7x5, 12x12x5 all reported the same spacing because
    # the old formula ignored the in-plane direction.  They must now differ.
    import math as _math

    a, c = 4.872, 12.466
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * _math.sqrt(3) / 2.0, 0.0], [0.0, 0.0, c]]
    s6 = spacing_from_grid(lattice, (6, 6, 5))
    s7 = spacing_from_grid(lattice, (7, 7, 5))
    s12 = spacing_from_grid(lattice, (12, 12, 5))
    assert s6 > s7 > s12


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

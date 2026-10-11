"""Tests for reciprocal-space Monkhorst-Pack grid generation helpers.

The spacing<->grid round-trip cases are the contract described in
``docs/castep/kpoint_spacing_convention.rst``.
"""

from __future__ import annotations

import math

import pytest

from mufinder.science.kpoint_grids import (
    most_uniform_grid,
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


# Lattice geometry: reciprocal vectors, parameters, matching and supercells


def test_reciprocal_lattice_vectors_cubic_cell() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    b1, b2, b3 = reciprocal_lattice_vectors(lattice)

    assert b1 == pytest.approx((1.57079632679, 0.0, 0.0))
    assert b2 == pytest.approx((0.0, 1.57079632679, 0.0))
    assert b3 == pytest.approx((0.0, 0.0, 1.57079632679))


def test_reciprocal_lattice_vectors_rejects_wrong_vector_count() -> None:
    with pytest.raises(ValueError, match="three vectors"):
        reciprocal_lattice_vectors([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_reciprocal_lattice_vectors_rejects_wrong_component_count() -> None:
    with pytest.raises(ValueError, match="three components"):
        reciprocal_lattice_vectors([[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_reciprocal_lattice_vectors_rejects_zero_volume_cell() -> None:
    # Coplanar lattice vectors -> zero triple product -> zero volume.
    coplanar = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    with pytest.raises(ValueError, match="non-zero cell volume"):
        reciprocal_lattice_vectors(coplanar)


def test_lattice_parameters_return_lengths_and_angles() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]

    a, b, c, alpha, beta, gamma = lattice_parameters(lattice)

    assert a == pytest.approx(4.0)
    assert b == pytest.approx(5.0)
    assert c == pytest.approx(6.0)
    assert alpha == pytest.approx(90.0)
    assert beta == pytest.approx(90.0)
    assert gamma == pytest.approx(90.0)


def test_lattice_parameters_reports_non_orthogonal_angles() -> None:
    # A monoclinic-ish cell where the b axis is tilted into the a direction
    # yields a gamma angle below 90 degrees, exercising the angle branch with
    # a non-trivial cosine.
    lattice = [[4.0, 0.0, 0.0], [2.0, 2.0 * math.sqrt(3.0), 0.0], [0.0, 0.0, 6.0]]
    a, b, c, alpha, beta, gamma = lattice_parameters(lattice)
    assert a == pytest.approx(4.0)
    assert c == pytest.approx(6.0)
    assert alpha == pytest.approx(90.0)
    assert beta == pytest.approx(90.0)
    assert gamma == pytest.approx(60.0)


def test_lattice_parameters_rejects_wrong_vector_count() -> None:
    with pytest.raises(ValueError, match="three vectors"):
        lattice_parameters([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]])


def test_lattice_parameters_rejects_wrong_component_count() -> None:
    with pytest.raises(ValueError, match="three components"):
        lattice_parameters([[4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])


def test_lattice_parameters_rejects_zero_length_vector() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        lattice_parameters([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])


def test_lattice_matches_with_tolerance() -> None:
    lhs = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    rhs = [[4.0 + 1e-7, 0.0, 0.0], [0.0, 5.0 - 1e-7, 0.0], [0.0, 0.0, 6.0]]

    assert lattice_matches(lhs, rhs)
    assert not lattice_matches(lhs, [[4.01, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])


def test_lattice_matches_rejects_wrong_outer_length() -> None:
    lhs = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    assert not lattice_matches(lhs, [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    assert not lattice_matches([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]], lhs)


def test_lattice_matches_rejects_wrong_row_length() -> None:
    lhs = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    rhs = [[4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    assert not lattice_matches(lhs, rhs)


def test_supercell_lattice_vectors_identity_is_passthrough() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    result = supercell_lattice_vectors(lattice, identity)
    assert result == (
        (4.0, 0.0, 0.0),
        (0.0, 5.0, 0.0),
        (0.0, 0.0, 6.0),
    )


def test_supercell_lattice_vectors_rejects_wrong_matrix_row_count() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="three rows"):
        supercell_lattice_vectors(lattice, [[1, 0, 0], [0, 1, 0]])


def test_supercell_lattice_vectors_rejects_wrong_matrix_row_length() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="three values"):
        supercell_lattice_vectors(lattice, [[1, 0], [0, 1, 0], [0, 0, 1]])


def test_supercell_lattice_vectors_rejects_wrong_lattice_vector_count() -> None:
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="three vectors"):
        supercell_lattice_vectors([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]], identity)


def test_supercell_lattice_vectors_rejects_wrong_lattice_component_count() -> None:
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="three components"):
        supercell_lattice_vectors(
            [[4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], identity
        )


# Spacing to grid (CASTEP's floor + 1 formula)


def test_grid_from_spacing_uses_reciprocal_magnitudes_formula() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]
    spacing = 0.1

    grid = grid_from_spacing(lattice, spacing)

    # n_i = floor(|b_i| / (2*pi*spacing)) + 1
    # |b| = [2*pi/4, 2*pi/6, 2*pi/8] => [1.5708, 1.0472, 0.7854]
    # denominator = 2*pi*0.1 = 0.6283
    # n = [3, 2, 2]
    assert grid == (3, 2, 2)


def test_grid_from_spacing_clamps_to_gamma_for_huge_spacing() -> None:
    # A spacing far larger than |b_i|/(2*pi) drives floor(...) to 0, so each
    # axis count clamps up to 1 via max(1, count).
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    assert grid_from_spacing(lattice, 1000.0) == (1, 1, 1)


def test_grid_from_spacing_rejects_zero_spacing() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive finite"):
        grid_from_spacing(lattice, 0.0)


def test_grid_from_spacing_rejects_negative_spacing() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive finite"):
        grid_from_spacing(lattice, -0.05)


def test_grid_from_spacing_rejects_non_finite_spacing() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive finite"):
        grid_from_spacing(lattice, math.inf)


# Convergence sweep grids


def test_generate_kpoint_grids_walks_castep_spacing_trajectory() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]

    grids = generate_kpoint_grids(lattice, 0.1)

    # Each axis transitions at |b_i|/(2*pi*n) independently, so the sweep
    # follows CASTEP's spacing curve rather than an isotropic level ramp.
    assert grids == ((1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 2))
    assert grids[-1] == grid_from_spacing(lattice, 0.1)


def test_generate_kpoint_grids_emits_only_castep_reachable_grids() -> None:
    import math as _math

    a = 4.872
    c = 12.466
    lattice = [
        [a, 0.0, 0.0],
        [-0.5 * a, a * _math.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, c],
    ]

    grids = generate_kpoint_grids(lattice, 0.02)

    magnitudes = tuple(
        _math.sqrt(sum(component ** 2 for component in vector))
        for vector in reciprocal_lattice_vectors(lattice)
    )
    two_pi = 2.0 * _math.pi

    for grid in grids:
        lo = max(magnitude / (two_pi * n) for magnitude, n in zip(magnitudes, grid))
        hi_candidates = [
            magnitude / (two_pi * (n - 1))
            for magnitude, n in zip(magnitudes, grid)
            if n > 1
        ]
        hi = min(hi_candidates) if hi_candidates else _math.inf
        assert lo < hi, f"grid {grid} not reachable from any CASTEP spacing"


def test_generate_kpoint_grids_isotropic_cell_matches_naive_ramp() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    grids = generate_kpoint_grids(lattice, 0.1)

    # CASTEP's curve and an isotropic level ramp coincide for cubic cells.
    assert grids == ((1, 1, 1), (2, 2, 2), (3, 3, 3))


def test_kpoint_helpers_format_and_count() -> None:
    grid = (4, 5, 6)

    assert kpoint_count(grid) == 120
    assert format_kpoint_grid(grid) == "4 x 5 x 6"


# Grid to spacing and the round-trip contract


def test_spacing_from_grid_returns_round_trip_upper_bound() -> None:
    # Return value is min_i( |b_i| / (2*pi * (N_i - 1)) ) so feeding it to
    # CASTEP's floor+1 formula reproduces the grid exactly.
    # For orthorhombic a=4, b=6, c=8 with grid (3, 2, 2):
    #   min( 1/(4*2), 1/(6*1), 1/(8*1) ) = min(0.125, 0.1667, 0.125) = 0.125.
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]
    assert spacing_from_grid(lattice, (3, 2, 2)) == pytest.approx(0.125)


def test_spacing_from_grid_round_trips_through_castep_formula() -> None:
    # Every spacing, fed back to grid_from_spacing, must reproduce its grid.
    # Use the sweep generator so we only test CASTEP-reachable grids — the
    # bounds of an unreachable grid (e.g. (2,2,2) on a 4x6x8 cell) are not
    # consistent with each other and could not round-trip with any value.
    lattice = [[4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 8.0]]
    for grid in generate_kpoint_grids(lattice, 0.05):
        if grid == (1, 1, 1):
            continue
        spacing = spacing_from_grid(lattice, grid)
        assert grid_from_spacing(lattice, spacing) == grid


def test_grid_from_spacing_round_trips_lif_high_n() -> None:
    # LiF cubic (a=4.0834 Ang) with fine spacings reaching N=8 through N=12.
    # Previously, two-step floating-point division made |b|/(2π·s) evaluate to
    # N-1-ε (e.g. 6.9999…9991 instead of 7.0), causing floor to snap down by
    # one and the recovered grid component to be N-1 instead of N.
    a = 4.0834
    lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    for n in range(2, 14):
        grid = (n, n, n)
        spacing = spacing_from_grid(lattice, grid)
        assert grid_from_spacing(lattice, spacing) == grid, (
            f"round-trip failed for ({n},{n},{n}): spacing={spacing}"
        )


def test_spacing_from_grid_cubic_lif_5x5x5() -> None:
    # Cubic LiF (a=4.0834 Ang), 5x5x5 grid: round-trip spacing = 1/(a*4).
    a = 4.0834
    lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    assert spacing_from_grid(lattice, (5, 5, 5)) == pytest.approx(1.0 / (a * 4.0))


def test_spacing_from_grid_hexagonal_alf3_3x3x3() -> None:
    # Hexagonal AlF3 (a=4.872, c=12.466 Ang), 3x3x3 grid.
    # Smallest |b|/(2*pi*(N-1)) is along c with N-1 = 2: spacing = 1/(2*c).
    import math as _math

    a, c = 4.872, 12.466
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * _math.sqrt(3) / 2.0, 0.0], [0.0, 0.0, c]]
    expected = 1.0 / (2.0 * c)  # ≈ 0.0401 1/Ang
    assert spacing_from_grid(lattice, (3, 3, 3)) == pytest.approx(expected, rel=1e-4)


def test_spacing_from_grid_strictly_decreasing_along_castep_reachable_sweep() -> None:
    # For the CASTEP-reachable grids produced by generate_kpoint_grids, the
    # spacing annotation must be strictly decreasing so users can read
    # convergence progress from the labels. Previously this was tested with
    # the old isotropic-ramp grids (6x6x5, 7x7x5, 12x12x5) on hexagonal
    # AlF3, but those grids are NOT CASTEP-reachable (no single
    # KPOINT_MP_SPACING reproduces them) and all return the same spacing
    # under min/(N-1). The new test uses the actual new-convention sweep.
    import math as _math

    a, c = 4.872, 12.466
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * _math.sqrt(3) / 2.0, 0.0], [0.0, 0.0, c]]
    reachable_grids = [g for g in generate_kpoint_grids(lattice, 0.02) if g != (1, 1, 1)]
    spacings = [spacing_from_grid(lattice, g) for g in reachable_grids]

    for i in range(1, len(spacings)):
        assert spacings[i] < spacings[i - 1], (
            f"spacing not strictly decreasing: {reachable_grids[i]} gives {spacings[i]:.6f} "
            f"but {reachable_grids[i-1]} gives {spacings[i-1]:.6f}"
        )


def test_spacing_from_grid_rejects_gamma_only_grid() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="Gamma-only"):
        spacing_from_grid(lattice, (1, 1, 1))


def test_spacing_from_grid_rejects_non_positive_grid_values() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive integers"):
        spacing_from_grid(lattice, (0, 2, 2))
    with pytest.raises(ValueError, match="positive integers"):
        spacing_from_grid(lattice, (2, -1, 2))
    with pytest.raises(ValueError, match="positive integers"):
        spacing_from_grid(lattice, (2, 2, 0))


def test_spacing_from_grid_handles_partial_gamma_axes() -> None:
    # Slab-like grid with one frozen axis: the spacing only depends on the
    # non-frozen axes that contribute to the round-trip bound.
    lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 30.0]]
    spacing = spacing_from_grid(lattice, (5, 5, 1))
    # |b_a|/(2*pi*4) = 1/(3*4); the c axis with N=1 does not contribute.
    assert spacing == pytest.approx(1.0 / (3.0 * 4.0))
    assert grid_from_spacing(lattice, spacing) == (5, 5, 1)


# Continuous k-point count


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


def test_kpoint_count_for_spacing_rejects_non_finite_spacing() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive finite"):
        kpoint_count_for_spacing(lattice, math.nan)


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


# Supercell-equivalent grids


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


def test_equivalent_grid_for_supercell_rejects_non_positive_reference_count() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive integers"):
        equivalent_grid_for_supercell(lattice, (0, 4, 4), lattice)


# Grid choice


def test_most_uniform_grid_prefers_fewer_kpoints_along_the_long_axis() -> None:
    lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 6.0]]

    assert most_uniform_grid(lattice, [(2, 2, 4), (4, 2, 2)]) == (4, 2, 2)


def test_most_uniform_grid_keeps_the_earlier_grid_on_a_tie() -> None:
    cubic = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    assert most_uniform_grid(cubic, [(2, 2, 4), (4, 2, 2)]) == (2, 2, 4)


def test_most_uniform_grid_requires_a_grid() -> None:
    with pytest.raises(ValueError):
        most_uniform_grid([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [])

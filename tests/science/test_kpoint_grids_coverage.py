"""Coverage-focused edge-case and guard tests for ``kpoint_grids``.

These exercise the validation guards and error paths in the spacing<->grid
helpers that the primary ``test_kpoint_grids`` suite does not reach. Pure
arithmetic only: no network, GUI, filesystem, sleep, or randomness.
"""

from __future__ import annotations

import math

import pytest

from mufinder.science.kpoint_grids import (
    equivalent_grid_for_supercell,
    grid_from_spacing,
    kpoint_count_for_spacing,
    lattice_matches,
    lattice_parameters,
    reciprocal_lattice_vectors,
    spacing_from_grid,
    supercell_lattice_vectors,
)


# --- reciprocal_lattice_vectors guards ---------------------------------------


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


# --- grid_from_spacing guards ------------------------------------------------


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


def test_grid_from_spacing_clamps_to_gamma_for_huge_spacing() -> None:
    # A spacing far larger than |b_i|/(2*pi) drives floor(...) to 0, so each
    # axis count clamps up to 1 via max(1, count).
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    assert grid_from_spacing(lattice, 1000.0) == (1, 1, 1)


def test_kpoint_count_for_spacing_rejects_non_finite_spacing() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive finite"):
        kpoint_count_for_spacing(lattice, math.nan)


# --- spacing_from_grid guards ------------------------------------------------


def test_spacing_from_grid_rejects_non_positive_grid_values() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive integers"):
        spacing_from_grid(lattice, (0, 2, 2))
    with pytest.raises(ValueError, match="positive integers"):
        spacing_from_grid(lattice, (2, -1, 2))
    with pytest.raises(ValueError, match="positive integers"):
        spacing_from_grid(lattice, (2, 2, 0))


# --- lattice_matches edge cases ----------------------------------------------


def test_lattice_matches_rejects_wrong_outer_length() -> None:
    lhs = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    assert not lattice_matches(lhs, [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    assert not lattice_matches([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]], lhs)


def test_lattice_matches_rejects_wrong_row_length() -> None:
    lhs = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    rhs = [[4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    assert not lattice_matches(lhs, rhs)


# --- lattice_parameters guards -----------------------------------------------


def test_lattice_parameters_rejects_wrong_vector_count() -> None:
    with pytest.raises(ValueError, match="three vectors"):
        lattice_parameters([[4.0, 0.0, 0.0], [0.0, 5.0, 0.0]])


def test_lattice_parameters_rejects_wrong_component_count() -> None:
    with pytest.raises(ValueError, match="three components"):
        lattice_parameters([[4.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])


def test_lattice_parameters_rejects_zero_length_vector() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        lattice_parameters([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]])


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


# --- supercell_lattice_vectors guards ----------------------------------------


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


def test_supercell_lattice_vectors_identity_is_passthrough() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    result = supercell_lattice_vectors(lattice, identity)
    assert result == (
        (4.0, 0.0, 0.0),
        (0.0, 5.0, 0.0),
        (0.0, 0.0, 6.0),
    )


# --- equivalent_grid_for_supercell guards ------------------------------------


def test_equivalent_grid_for_supercell_rejects_non_positive_reference_count() -> None:
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    with pytest.raises(ValueError, match="positive integers"):
        equivalent_grid_for_supercell(lattice, (0, 4, 4), lattice)

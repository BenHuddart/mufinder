"""Tests for supercell expansion with magnetic spin propagation."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.supercell import (
    PROPAGATION_FIT_TOLERANCE,
    build_supercell_structure,
    supercell_fits_propagation_vector,
)


def _base_structure() -> Structure:
    return Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties={"pbc": [True, True, True]},
    )


def test_build_supercell_structure_identity_matrix() -> None:
    structure, spins = build_supercell_structure(
        _base_structure(),
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        atomic_spins=[2.0],
        propagation_vector=[0.0, 0.0, 0.0],
    )

    assert len(structure.species) == 1
    assert spins == pytest.approx([2.0])
    assert structure.fractional_positions[0] == pytest.approx([0.0, 0.0, 0.0])


def test_build_supercell_structure_applies_propagation_phase() -> None:
    structure, spins = build_supercell_structure(
        _base_structure(),
        [[1, 0, 0], [0, 1, 0], [0, 0, 2]],
        atomic_spins=[2.0],
        propagation_vector=[0.0, 0.0, 0.5],
    )

    assert len(structure.species) == 2
    assert sorted(round(value, 6) for value in spins) == pytest.approx([-2.0, 2.0])


def test_build_supercell_structure_rejects_non_invertible_matrix() -> None:
    with pytest.raises(ValueError, match="invertible"):
        build_supercell_structure(
            _base_structure(),
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
            atomic_spins=[2.0],
            propagation_vector=[0.0, 0.0, 0.0],
        )


def test_a_doubled_cell_fits_a_half_propagation_vector() -> None:
    k = [0.5, 0.0, 0.0]

    assert supercell_fits_propagation_vector([[2, 0, 0], [0, 1, 0], [0, 0, 1]], k)
    assert supercell_fits_propagation_vector([[4, 0, 0], [0, 3, 0], [0, 0, 5]], k)
    assert not supercell_fits_propagation_vector([[3, 0, 0], [0, 1, 0], [0, 0, 1]], k)
    assert not supercell_fits_propagation_vector([[1, 0, 0], [0, 2, 0], [0, 0, 2]], k)


def test_a_general_matrix_fits_when_every_row_takes_a_whole_phase() -> None:
    # Rows (1, 1, 0), (-1, 1, 0), (0, 0, 1): k·A is 1, 0 and 0 for k = (½, ½, 0).
    general = [[1, 1, 0], [-1, 1, 0], [0, 0, 1]]

    assert supercell_fits_propagation_vector(general, [0.5, 0.5, 0.0])
    # For k = (½, 0, 0) the first two rows take half a turn.
    assert not supercell_fits_propagation_vector(general, [0.5, 0.0, 0.0])


def test_a_zero_or_missing_propagation_vector_always_fits() -> None:
    for matrix in ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[3, 0, 0], [0, 1, 0], [0, 0, 1]]):
        assert supercell_fits_propagation_vector(matrix, [0.0, 0.0, 0.0])
        assert supercell_fits_propagation_vector(matrix, None)


def test_a_third_fits_a_tripled_cell_within_the_tolerance() -> None:
    tripled = [[3, 0, 0], [0, 1, 0], [0, 0, 1]]

    assert supercell_fits_propagation_vector(tripled, [1 / 3, 0.0, 0.0])
    assert supercell_fits_propagation_vector(tripled, [0.333, 0.0, 0.0])
    assert not supercell_fits_propagation_vector(tripled, [0.33, 0.0, 0.0])
    assert PROPAGATION_FIT_TOLERANCE == pytest.approx(1e-3)


def test_a_rounded_third_still_fits_a_longer_cell() -> None:
    # k·A = 1.998 for a sixfold cell: 0.002 from 2, within 0.001 for each of the six cells.
    assert supercell_fits_propagation_vector([[6, 0, 0], [0, 1, 0], [0, 0, 1]], [0.333, 0, 0])
    assert not supercell_fits_propagation_vector([[5, 0, 0], [0, 1, 0], [0, 0, 1]], [0.333, 0, 0])


def test_the_fit_does_not_depend_on_the_basis_the_matrix_writes() -> None:
    # [[1, 32, 0], [0, 1, 0], [0, 0, 1]] is the unit cell in another basis.
    sheared = [[1, 32, 0], [0, 1, 0], [0, 0, 1]]
    identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    for k in ([0.03, 0, 0], [0, 0.5, 0], [0.0005, 0, 0]):
        assert supercell_fits_propagation_vector(sheared, k) == supercell_fits_propagation_vector(
            identity, k
        )


def test_a_propagation_vector_within_the_tolerance_of_zero_fits_any_cell() -> None:
    assert supercell_fits_propagation_vector([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0005, 0, 0])
    assert supercell_fits_propagation_vector([[3, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0005, 0, 0])


def test_a_cell_that_fits_has_spins_that_repeat_across_its_boundary() -> None:
    """The check agrees with the spins the supercell build gives."""

    half = [0.5, 0.0, 0.0]
    for scale, fits in ((2, True), (3, False)):
        matrix = [[scale, 0, 0], [0, 1, 0], [0, 0, 1]]
        _structure, spins = build_supercell_structure(
            _base_structure(), matrix, atomic_spins=[1.0], propagation_vector=half
        )
        # The next copy along a, one supercell over, has the spin of translation `scale`.
        wrapped = float(np.cos(2 * np.pi * half[0] * scale))
        assert (wrapped == pytest.approx(spins[0])) is fits
        assert supercell_fits_propagation_vector(matrix, half) is fits

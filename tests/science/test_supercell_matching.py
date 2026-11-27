"""Tests for supercell detection and normalization."""

from __future__ import annotations

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.supercell_matching import (
    detect_supercell_matrix,
    host_cell_lattice,
    normalize_for_comparison,
)


def _experimental_structure() -> Structure:
    return Structure(
        label="Si",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _dft_supercell() -> Structure:
    return Structure(
        label="Si-super",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
    )


def test_detect_supercell_matrix_identifies_diagonal_supercell() -> None:
    detection = detect_supercell_matrix(_experimental_structure(), _dft_supercell())

    assert detection.is_supercell is True
    assert detection.matrix == ((2, 0, 0), (0, 1, 0), (0, 0, 1))
    assert detection.determinant == 2


def test_normalize_for_comparison_tiles_experimental_for_supercell() -> None:
    normalized = normalize_for_comparison(_experimental_structure(), _dft_supercell())

    assert normalized.is_supercell is True
    assert len(normalized.experimental.species) == 2
    assert normalized.warnings == ()


def test_host_cell_lattice_divides_a_diagonal_supercell_by_its_tiling() -> None:
    supercell = [[9.39, 0.0, 0.0], [0.0, 9.39, 0.0], [0.0, 0.0, 6.36]]

    host = host_cell_lattice(supercell, ((2, 0, 0), (0, 2, 0), (0, 0, 2)))

    assert np.allclose(host, [[4.695, 0.0, 0.0], [0.0, 4.695, 0.0], [0.0, 0.0, 3.18]])


def test_host_cell_lattice_recovers_the_host_setting_of_a_non_diagonal_supercell() -> None:
    # A sqrt(2) x sqrt(2) x 1 cell, rotated 45° from the host axes, of a host
    # strained 1 % along a only: M^-1 L gives back that strained host cell.
    host = np.diag([4.695 * 1.01, 4.695, 3.18])
    matrix = ((1, 1, 0), (-1, 1, 0), (0, 0, 1))
    supercell = np.asarray(matrix, dtype=float) @ host

    assert np.allclose(host_cell_lattice(supercell.tolist(), matrix), host)

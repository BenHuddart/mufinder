"""Tests for supercell detection and normalization."""

from __future__ import annotations

from mufinder.domain.models import Structure
from mufinder.science.supercell_matching import detect_supercell_matrix, normalize_for_comparison


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

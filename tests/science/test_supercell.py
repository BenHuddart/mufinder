"""Tests for supercell expansion with magnetic spin propagation."""

from __future__ import annotations

import pytest

from mufinder.domain.models import Structure
from mufinder.science.supercell import build_supercell_structure


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

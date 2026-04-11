"""Tests for bond/angle/coordination comparison helpers."""

from __future__ import annotations

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import match_structures
from mufinder.science.bonding_analysis import (
    compare_bond_angles,
    compare_coordination_numbers,
    compare_nearest_neighbour_bond_lengths,
)


def _exp() -> Structure:
    return Structure(
        label="exp",
        lattice_vectors=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        species=["Si", "Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.33, 0.0, 0.0], [0.0, 0.33, 0.0]],
    )


def _dft() -> Structure:
    return Structure(
        label="dft",
        lattice_vectors=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        species=["Si", "Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.35, 0.0, 0.0], [0.0, 0.34, 0.0]],
    )


def test_compare_bonding_metrics_returns_values() -> None:
    matching = match_structures(_exp(), _dft())

    bond_lengths = compare_nearest_neighbour_bond_lengths(_exp(), _dft(), matching)
    bond_angles = compare_bond_angles(_exp(), _dft(), matching)
    coordination = compare_coordination_numbers(_exp(), _dft(), matching)

    assert bond_lengths.mean_difference is not None
    assert bond_lengths.max_difference is not None
    assert bond_angles.mean_difference is not None
    assert coordination.changed_sites >= 0

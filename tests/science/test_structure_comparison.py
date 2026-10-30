"""Tests for end-to-end structure comparison orchestration."""

from __future__ import annotations

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.structure_comparison import DftComparisonInput, compare_structures
from tests.support.structures import host_structure


def _exp() -> Structure:
    return Structure(
        label="exp",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _dft() -> Structure:
    return Structure(
        label="dft",
        lattice_vectors=[[5.2, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.01, 0.0, 0.0]],
    )


def test_compare_structures_builds_metric_payload() -> None:
    experimental = _exp()
    dft = _dft()

    result = compare_structures(
        experimental,
        [DftComparisonInput(label="DFT-1", structure=dft, structure_id=dft.id)],
    )

    assert result.definitions
    assert result.experimental_values["a"].display
    assert len(result.dft_columns) == 1
    assert result.dft_columns[0].values["a"].display == "5.2 (+4.00 %)"
    assert [definition.key for definition in result.definitions][-2:] == [
        "space_group",
        "space_group_match",
    ]


def test_compare_structures_matches_a_162_atom_supercell_column() -> None:
    # A 3x3x3 supercell of a rutile cell, strained 0.3 %, against the unit cell.
    # Without per-pair bonding metrics this column takes about 60 ms (measured,
    # not asserted: a wall-clock bound would flake on a loaded runner).
    unit = host_structure(
        label="unit",
        lattice=[[4.695, 0, 0], [0, 4.695, 0], [0, 0, 3.18]],
        species=("Co", "Co", "F", "F", "F", "F"),
        positions=(
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.306, 0.306, 0.0),
            (0.694, 0.694, 0.0),
            (0.806, 0.194, 0.5),
            (0.194, 0.806, 0.5),
        ),
    )
    cells = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    supercell = host_structure(
        label="supercell",
        lattice=(np.array(unit.lattice_vectors) * 3.0 * 1.003).tolist(),
        species=tuple(unit.species) * len(cells),
        positions=tuple(
            tuple((np.array(position) + np.array(cell)) / 3.0)
            for cell in cells
            for position in unit.fractional_positions
        ),
    )

    result = compare_structures(
        unit, [DftComparisonInput(label="DFT", structure=supercell, structure_id=supercell.id)]
    )

    assert len(result.dft_columns[0].matching.pairs) == 162
    assert result.dft_columns[0].values["a"].display == "14.1273 (+0.30 %)"

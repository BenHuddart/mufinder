"""Tests for end-to-end structure comparison orchestration."""

from __future__ import annotations

import copy

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.structure_comparison import DftComparisonInput, compare_structures
from mufinder.science.supercell_matching import tile_to_supercell
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

    assert result.experimental_values["a"].display
    assert len(result.dft_columns) == 1
    assert result.dft_columns[0].values["a"].display == "5.2 (+4.00 %)"


def _rutile_unit() -> Structure:
    return host_structure(
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


def _column(structure: Structure) -> DftComparisonInput:
    return DftComparisonInput(label=structure.label, structure=structure, structure_id=structure.id)


def _strained(structure: Structure, strain: np.ndarray, label: str) -> Structure:
    strained = copy.deepcopy(structure)
    strained.label = label
    strained.lattice_vectors = (np.asarray(structure.lattice_vectors) @ strain).tolist()
    return strained


def test_compare_structures_matches_a_162_atom_supercell_column() -> None:
    # A 3x3x3 supercell of a rutile cell, strained 0.3 %, against the unit cell.
    # Without per-pair bonding metrics this column takes about 60 ms (measured,
    # not asserted: a wall-clock bound would flake on a loaded runner).
    unit = _rutile_unit()
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

    result = compare_structures(unit, [_column(supercell)])

    assert len(result.dft_columns[0].matching.pairs) == 162
    # Shown per host cell: 4.695 Å x 1.003, not the supercell's 14.1273 Å.
    assert result.dft_columns[0].values["a"].display == "4.70908 (+0.30 %)"


def test_supercell_columns_read_per_host_cell_beside_a_primitive_column() -> None:
    unit = _rutile_unit()
    strain = np.diag([1.004, 1.004, 0.998])
    primitive = _strained(unit, np.diag([1.02, 1.02, 1.01]), "DFT (unit)")
    cubic = _strained(
        tile_to_supercell(unit, ((2, 0, 0), (0, 2, 0), (0, 0, 2))), strain, "DFT (2x2x2)"
    )
    rotated = _strained(
        tile_to_supercell(unit, ((1, 1, 0), (-1, 1, 0), (0, 0, 1))), strain, "DFT (rotated)"
    )

    result = compare_structures(unit, [_column(primitive), _column(cubic), _column(rotated)])

    for column in result.dft_columns[1:]:
        values = column.values
        assert np.isclose(values["a"].value, 4.695 * 1.004)
        assert np.isclose(values["c"].value, 3.18 * 0.998)
        assert np.isclose(values["gamma"].value, 90.0)
        assert np.isclose(values["cell_volume"].value, 4.695**2 * 3.18 * 1.004**2 * 0.998)
        assert values["a"].display == "4.71378 (+0.40 %)"
        assert len(column.matching.pairs) == len(column.normalization.dft.species)
    # The primitive column has no note; each supercell column has one.
    assert result.warnings == (
        "DFT (2x2x2): 2 × 2 × 2 supercell; its lattice parameters are shown per host cell.",
        "DFT (rotated): Supercell of 2 host cells; its lattice parameters are shown per host cell.",
    )
    # The notes are table-wide; a column's own warnings stay the problems found.
    assert all(column.warnings == () for column in result.dft_columns)


def test_a_column_in_another_setting_of_the_host_reads_in_the_host_axes() -> None:
    # The same rutile cell with its axes listed c, a, b: det M = 1, so it is no
    # supercell and needs no note, but its rows still read along the host's a, b, c.
    unit = _rutile_unit()
    matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    relabelled = _strained(unit, np.diag([1.004, 1.004, 0.998]), "DFT (cab)")
    relabelled.lattice_vectors = (matrix @ np.asarray(relabelled.lattice_vectors)).tolist()
    relabelled.fractional_positions = (
        np.asarray(relabelled.fractional_positions) @ np.linalg.inv(matrix)
    ).tolist()

    result = compare_structures(unit, [_column(relabelled)])

    values = result.dft_columns[0].values
    assert values["a"].display == "4.71378 (+0.40 %)"
    assert values["c"].display == "3.17364 (-0.20 %)"
    assert result.warnings == ()

"""High-level orchestration for experimental-vs-DFT structure comparison."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np

from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.science.atom_matching import AtomMatching, match_structures
from mufinder.science.structure_metrics import (
    MetricValue,
    dft_metric_values,
    experimental_metric_values,
)
from mufinder.science.supercell_matching import NormalizedStructurePair, normalize_for_comparison


@dataclass(frozen=True, slots=True)
class DftComparisonInput:
    """One DFT structure selected for comparison."""

    label: str
    structure: Structure
    structure_id: UUID


@dataclass(frozen=True, slots=True)
class DftComparisonColumn:
    """Computed metrics and metadata for one DFT column."""

    label: str
    structure_id: UUID
    values: dict[str, MetricValue]
    matching: AtomMatching
    normalization: NormalizedStructurePair
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Complete comparison payload for table rendering."""

    experimental_values: dict[str, MetricValue]
    dft_columns: tuple[DftComparisonColumn, ...]
    warnings: tuple[str, ...] = ()


def _per_host_cell_note(normalization: NormalizedStructurePair) -> str:
    """The banner note for a supercell column: its tiling, and that it is read per host cell."""

    matrix = normalization.supercell_matrix
    diagonal = all(matrix[i][j] == 0 for i in range(3) for j in range(3) if i != j)
    if diagonal and all(matrix[i][i] > 0 for i in range(3)):
        supercell = f"{matrix[0][0]} × {matrix[1][1]} × {matrix[2][2]} supercell"
    else:
        host_cells = round(abs(float(np.linalg.det(np.asarray(matrix, dtype=float)))))
        supercell = f"Supercell of {plural(host_cells, 'host cell')}"
    return f"{supercell}; its lattice parameters are shown per host cell."


def compare_structures(
    experimental_structure: Structure,
    dft_structures: list[DftComparisonInput],
) -> ComparisonResult:
    """Compare one experimental structure against one or more DFT structures."""

    if not dft_structures:
        raise ValueError("At least one DFT structure is required for comparison.")

    reference_values = experimental_metric_values(experimental_structure)

    columns: list[DftComparisonColumn] = []
    warnings: list[str] = []

    for dft_input in dft_structures:
        normalization = normalize_for_comparison(experimental_structure, dft_input.structure)
        matching = match_structures(normalization.experimental, normalization.dft)
        values = dft_metric_values(
            normalization.experimental,
            normalization.dft,
            matching,
            host=experimental_structure,
            supercell_matrix=normalization.supercell_matrix,
        )

        column_warnings = [*normalization.warnings, *matching.warnings]
        if column_warnings:
            warnings.extend(f"{dft_input.label}: {warning}" for warning in column_warnings)
        if normalization.is_supercell:
            warnings.append(f"{dft_input.label}: {_per_host_cell_note(normalization)}")

        columns.append(
            DftComparisonColumn(
                label=dft_input.label,
                structure_id=dft_input.structure_id,
                values=values,
                matching=matching,
                normalization=normalization,
                warnings=tuple(column_warnings),
            )
        )

    return ComparisonResult(
        experimental_values=reference_values,
        dft_columns=tuple(columns),
        warnings=tuple(warnings),
    )

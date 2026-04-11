"""High-level orchestration for experimental-vs-DFT structure comparison."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import AtomMatching, match_structures
from mufinder.science.structure_metrics import (
    MetricDefinition,
    MetricValue,
    dft_metric_values,
    experimental_metric_values,
    metric_definitions,
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

    definitions: tuple[MetricDefinition, ...]
    experimental_values: dict[str, MetricValue]
    dft_columns: tuple[DftComparisonColumn, ...]
    warnings: tuple[str, ...] = ()


def compare_structures(
    experimental_structure: Structure,
    dft_structures: list[DftComparisonInput],
) -> ComparisonResult:
    """Compare one experimental structure against one or more DFT structures."""

    if not dft_structures:
        raise ValueError("At least one DFT structure is required for comparison.")

    definitions = metric_definitions()
    reference_values = experimental_metric_values(experimental_structure)

    columns: list[DftComparisonColumn] = []
    warnings: list[str] = []

    for dft_input in dft_structures:
        normalization = normalize_for_comparison(experimental_structure, dft_input.structure)
        matching = match_structures(normalization.experimental, normalization.dft)
        values = dft_metric_values(normalization.experimental, normalization.dft, matching)

        column_warnings = [*normalization.warnings, *matching.warnings]
        if column_warnings:
            warnings.extend(f"{dft_input.label}: {warning}" for warning in column_warnings)

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
        definitions=definitions,
        experimental_values=reference_values,
        dft_columns=tuple(columns),
        warnings=tuple(warnings),
    )

"""Build a saved comparison table from completed pristine relaxations.

The science layer computes the comparison; this module turns the result into
the values a :class:`~mufinder.domain.models.ComparisonTable` stores. A table
stores values only: each metric's key, raw value, display text and warning,
the column labels and structure ids, the warnings and the parameter
differences. A metric's label, unit, group and tooltip are looked up from
:mod:`mufinder.science.structure_metrics` by key when the table is shown.
"""

from __future__ import annotations

from collections.abc import Sequence

from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.domain.models import ComparisonTable, Structure
from mufinder.domain.text import plural
from mufinder.science.structure_comparison import (
    ComparisonResult,
    DftComparisonInput,
    compare_structures,
)
from mufinder.science.structure_metrics import MetricValue
from mufinder.services.project_document import ProjectDocument
from mufinder.services.source_candidates import CompletedPristineJob


def _format_parameter_value(value: object) -> str:
    if isinstance(value, ValueWithUnit):
        return f"{value.value} {value.unit}".strip()
    if isinstance(value, BlockSection):
        return f"%BLOCK {value.name} ({plural(len(value.rows), 'row')})"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(entry) for entry in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={entry}" for key, entry in value.items())
    if value is None:
        return ""
    return str(value)


def parameter_difference_rows(jobs: Sequence[CompletedPristineJob]) -> list[dict[str, object]]:
    """One row per CASTEP parameter that differs between ``jobs``, with each job's value."""

    if len(jobs) < 2:
        return []

    rows: list[dict[str, object]] = []
    for name in sorted({name for job in jobs for name in job.parameter_values}):
        displays = [_format_parameter_value(job.parameter_values.get(name)) for job in jobs]
        if len(set(displays)) > 1:
            rows.append({"name": name, "values": displays})
    return rows


def _serialize_metric_values(values: dict[str, MetricValue]) -> dict[str, dict[str, object]]:
    return {
        key: {"value": value.value, "display": value.display, "warning": value.warning}
        for key, value in values.items()
    }


def comparison_result_data(
    result: ComparisonResult,
    *,
    parameter_differences: list[dict[str, object]],
) -> dict[str, object]:
    """The values of ``result`` as ``ComparisonTable.result_data`` stores them."""

    return {
        "experimental_values": _serialize_metric_values(result.experimental_values),
        "dft_columns": [
            {
                "label": column.label,
                "structure_id": str(column.structure_id),
                "warnings": list(column.warnings),
                "supercell_matrix": [
                    [int(entry) for entry in row] for row in column.normalization.supercell_matrix
                ],
                "values": _serialize_metric_values(column.values),
            }
            for column in result.dft_columns
        ],
        "parameter_differences": parameter_differences,
        "warnings": list(result.warnings),
    }


def comparison_inputs(
    document: ProjectDocument, jobs: Sequence[CompletedPristineJob]
) -> list[DftComparisonInput]:
    """The relaxed structures of ``jobs``, skipping a job whose structure is gone."""

    inputs: list[DftComparisonInput] = []
    for job in jobs:
        structure = document.structures.get(job.structure_id)
        if structure is not None:
            inputs.append(
                DftComparisonInput(
                    label=f"DFT ({job.job_label})",
                    structure=structure,
                    structure_id=structure.id,
                )
            )
    return inputs


def build_comparison_table(
    *,
    label: str,
    host: Structure,
    inputs: list[DftComparisonInput],
    jobs: Sequence[CompletedPristineJob],
) -> ComparisonTable:
    """Compare ``inputs`` against ``host`` and return the unsaved table.

    Raises :class:`ValueError` when the structures cannot be compared.
    """

    result = compare_structures(host, inputs)
    return ComparisonTable(
        label=label,
        experimental_structure_id=host.id,
        dft_structure_ids=[item.structure_id for item in inputs],
        dft_labels=[item.label for item in inputs],
        result_data=comparison_result_data(
            result, parameter_differences=parameter_difference_rows(jobs)
        ),
    )


__all__ = [
    "build_comparison_table",
    "comparison_inputs",
    "comparison_result_data",
    "parameter_difference_rows",
]

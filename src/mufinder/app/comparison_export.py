"""CSV export support for pristine-relaxation comparison tables."""

from __future__ import annotations

import csv
from pathlib import Path

from mufinder.domain.models import ComparisonTable
from mufinder.science.structure_metrics import MetricDefinition, metric_definitions


def table_metric_definitions(table: ComparisonTable) -> list[MetricDefinition]:
    """The defined metrics a saved table holds, in the order of ``metric_definitions()``.

    A table names its metrics by key in ``metric_definitions`` and in its
    experimental values. Keys that are no longer defined are skipped, and the
    label, unit, group and tooltip always come from the current definitions.
    """

    payload = table.result_data
    keys: set[str] = set()
    for definition in payload.get("metric_definitions", []):
        if isinstance(definition, dict):
            keys.add(str(definition.get("key", "")))
    experimental_values = payload.get("experimental_values", {})
    if isinstance(experimental_values, dict):
        keys.update(str(key) for key in experimental_values)
    return [definition for definition in metric_definitions() if definition.key in keys]


def export_comparison_csv(table: ComparisonTable, output_path: Path) -> None:
    """Export one comparison table to a CSV file."""

    payload = table.result_data
    dft_columns = payload.get("dft_columns", [])
    experimental_values = payload.get("experimental_values", {})

    headers = ["Metric", "Unit", "Experimental"] + [
        str(column.get("label", "DFT")) for column in dft_columns if isinstance(column, dict)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)

        for definition in table_metric_definitions(table):
            key = definition.key
            experimental = experimental_values.get(key, {})
            exp_display = ""
            if isinstance(experimental, dict):
                exp_display = str(experimental.get("display", ""))

            row = [definition.label, definition.unit, exp_display]
            for column in dft_columns:
                display = ""
                if isinstance(column, dict):
                    values = column.get("values", {})
                    if isinstance(values, dict):
                        metric_payload = values.get(key, {})
                        if isinstance(metric_payload, dict):
                            display = str(metric_payload.get("display", ""))
                row.append(display)
            writer.writerow(row)

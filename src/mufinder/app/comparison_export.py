"""CSV export support for pristine-relaxation comparison tables."""

from __future__ import annotations

import csv
from pathlib import Path

from mufinder.domain.models import ComparisonTable


def export_comparison_csv(table: ComparisonTable, output_path: Path) -> None:
    """Export one comparison table to a CSV file."""

    payload = table.result_data
    definitions = payload.get("metric_definitions", [])
    dft_columns = payload.get("dft_columns", [])
    experimental_values = payload.get("experimental_values", {})

    headers = ["Metric", "Unit", "Experimental"] + [
        str(column.get("label", "DFT")) for column in dft_columns if isinstance(column, dict)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)

        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            key = str(definition.get("key", ""))
            label = str(definition.get("label", key))
            unit = str(definition.get("unit", ""))
            experimental = experimental_values.get(key, {})
            exp_display = ""
            if isinstance(experimental, dict):
                exp_display = str(experimental.get("display", ""))

            row = [label, unit, exp_display]
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

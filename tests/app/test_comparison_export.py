"""Tests for comparison table CSV export."""

from __future__ import annotations

import csv
from pathlib import Path

from mufinder.app.comparison_export import export_comparison_csv
from mufinder.domain.models import ComparisonTable


def test_export_comparison_csv_writes_headers_and_metric_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "comparison.csv"
    table = ComparisonTable(
        label="Comparison",
        result_data={
            "metric_definitions": [
                {"key": "a", "label": "a", "unit": "Ang"},
                "skip-me",
                {"key": "b", "label": "b", "unit": ""},
            ],
            "experimental_values": {
                "a": {"display": "5.0"},
                "b": "ignored",
            },
            "dft_columns": [
                {"label": "job_1", "values": {"a": {"display": "5.1"}, "b": {"display": "6.1"}}},
                {"label": "job_2", "values": {"a": {"display": "4.9"}}},
                "skip-me",
            ],
        },
    )

    export_comparison_csv(table, output_path)

    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["Metric", "Unit", "Experimental", "job_1", "job_2"]
    # Label and unit come from the current definitions, not the saved table.
    assert rows[1] == ["a", "Å", "5.0", "5.1", "4.9", ""]
    assert rows[2] == ["b", "Å", "", "6.1", "", ""]


def test_export_comparison_csv_skips_metrics_that_are_no_longer_defined(tmp_path: Path) -> None:
    output_path = tmp_path / "old.csv"
    table = ComparisonTable(
        label="Old comparison",
        result_data={
            "metric_definitions": [
                {"key": "cell_volume", "label": "Cell Volume", "unit": "Ang^3"},
                {"key": "delta_volume_pct", "label": "Change in volume vs experiment"},
                {"key": "mean_bond_angle_diff", "label": "Mean bond-angle difference"},
                {"key": "similarity_score", "label": "Similarity score", "unit": "0-1"},
            ],
            "experimental_values": {
                "cell_volume": {"display": "70.0968"},
                "similarity_score": {"display": "1"},
            },
            "dft_columns": [
                {
                    "label": "DFT (PBE)",
                    "values": {
                        "cell_volume": {"display": "70.6915"},
                        "similarity_score": {"display": "1"},
                    },
                }
            ],
        },
    )

    export_comparison_csv(table, output_path)

    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows == [
        ["Metric", "Unit", "Experimental", "DFT (PBE)"],
        ["V", "Å³", "70.0968", "70.6915"],
    ]

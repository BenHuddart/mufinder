"""Tests for comparison table rendering widget."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QScrollArea, QWidget

from mufinder.app.comparison_table_widget import (
    ComparisonMetricHelpDialog,
    ComparisonTableWidget,
    _build_metric_help_plain_text,
)
from mufinder.app.math_label import MathLabel
from mufinder.domain.models import ComparisonTable


def _table() -> ComparisonTable:
    return ComparisonTable(
        label="Comparison A",
        result_data={
            "metric_definitions": [
                {
                    "key": "a",
                    "label": "a",
                    "unit": "Ang",
                    "group": "Lattice Parameters",
                    "meaning": "x",
                    "calculation": "x",
                    "interpretation": "x",
                    "caveats": "x",
                }
            ],
            "experimental_values": {"a": {"display": "5"}},
            "dft_columns": [{"label": "pristine_relax_01", "values": {"a": {"display": "5.1"}}}],
            "parameter_differences": [{"name": "cut_off_energy", "values": ["600"]}],
            "warnings": ["warning text"],
        },
    )


def test_comparison_table_widget_renders_headers_and_warning(qapp) -> None:
    widget = ComparisonTableWidget(_table())

    headers = [widget._grid.horizontalHeaderItem(index).text() for index in range(widget._grid.columnCount())]

    assert headers == ["Metric", "Experimental", "DFT (pristine_relax_01)"]
    assert widget._grid.verticalHeader().isVisible() is False
    assert widget._parameter_diff_section_row == 0
    assert widget._parameter_diff_toggle is not None
    assert widget._parameter_diff_toggle.objectName() == "sectionToggleButton"
    assert widget._warning_banner.isHidden() is False


def test_comparison_table_widget_toggles_parameter_diff_rows(qapp) -> None:
    widget = ComparisonTableWidget(_table())

    assert widget._parameter_diff_rows
    widget._set_parameter_diff_visibility(False)

    assert all(widget._grid.isRowHidden(row) for row in widget._parameter_diff_rows)
    assert widget._parameter_diff_section_row is not None
    assert widget._grid.isRowHidden(widget._parameter_diff_section_row) is False


def test_comparison_table_widget_highlights_unique_best_agreement(qapp) -> None:
    table = ComparisonTable(
        label="Comparison B",
        result_data={
            "metric_definitions": [
                {
                    "key": "a",
                    "label": "a",
                    "unit": "Ang",
                    "group": "Lattice Parameters",
                    "meaning": "x",
                    "calculation": "x",
                    "interpretation": "x",
                    "caveats": "x",
                }
            ],
            "experimental_values": {"a": {"display": "5", "value": 5.0}},
            "dft_columns": [
                {"label": "job_1", "values": {"a": {"display": "5.2", "value": 5.2}}},
                {"label": "job_2", "values": {"a": {"display": "5.04", "value": 5.04}}},
                {"label": "job_3", "values": {"a": {"display": "5.3", "value": 5.3}}},
            ],
            "parameter_differences": [],
            "warnings": [],
        },
    )

    widget = ComparisonTableWidget(table)

    best_item = widget._grid.item(1, 3)
    assert best_item is not None
    assert best_item.text() == "5.04"
    assert best_item.font().bold() is True
    assert best_item.foreground().color() == widget._best_agreement_color()


def test_comparison_table_widget_does_not_highlight_identical_values(qapp) -> None:
    table = ComparisonTable(
        label="Comparison C",
        result_data={
            "metric_definitions": [
                {
                    "key": "a",
                    "label": "a",
                    "unit": "Ang",
                    "group": "Lattice Parameters",
                    "meaning": "x",
                    "calculation": "x",
                    "interpretation": "x",
                    "caveats": "x",
                }
            ],
            "experimental_values": {"a": {"display": "5", "value": 5.0}},
            "dft_columns": [
                {"label": "job_1", "values": {"a": {"display": "5.1", "value": 5.1}}},
                {"label": "job_2", "values": {"a": {"display": "5.1", "value": 5.1}}},
            ],
            "parameter_differences": [],
            "warnings": [],
        },
    )

    widget = ComparisonTableWidget(table)

    first_item = widget._grid.item(1, 2)
    second_item = widget._grid.item(1, 3)
    assert first_item is not None
    assert second_item is not None
    assert first_item.font().bold() is False
    assert second_item.font().bold() is False


def test_metric_help_plain_text_includes_equation_and_symbols() -> None:
    text = _build_metric_help_plain_text(
        {
            "label": "RMSD (matched atoms)",
            "group": "Atomic Displacements",
            "unit": "Ang",
            "meaning": "Root-mean-square displacement over matched atoms.",
            "physics": "Measures the characteristic amplitude of atomic relaxation.",
            "calculation": "sqrt(mean(d_i^2)).",
            "method": "Atoms are matched by a species-resolved Hungarian assignment.",
            "equation": r"\mathrm{RMSD} = \sqrt{\frac{1}{N}\sum_i d_i^2}",
            "symbols": [
                {"symbol": r"d_i", "meaning": "Matched-atom displacement for pair i."},
                {"symbol": r"N", "meaning": "Number of matched atoms."},
            ],
            "interpretation": "Lower is better.",
            "caveats": "Depends on atom matching quality.",
        }
    )

    assert "Equation" in text
    assert "Equation (LaTeX)" not in text
    assert "Matched-atom displacement for pair i." in text
    assert "Atomic Displacements" in text


def test_metric_help_plain_text_merges_registry_details_for_sparse_payloads() -> None:
    text = _build_metric_help_plain_text(
        {
            "key": "rmsd",
            "label": "RMSD (matched atoms)",
            "unit": "Ang",
            "group": "Atomic Displacements",
            "meaning": "Root-mean-square displacement over matched atoms.",
            "calculation": "sqrt(mean(d_i^2)).",
            "interpretation": "Lower is better.",
            "caveats": "Depends on atom matching quality.",
        }
    )

    assert "Equation" in text
    assert "Equation (LaTeX)" not in text
    assert "Physics" in text
    assert "Symbols" in text


def test_metric_help_dialog_renders_equation_with_math_label(qapp) -> None:
    dialog = ComparisonMetricHelpDialog(
        {
            "label": "RMSD (matched atoms)",
            "group": "Atomic Displacements",
            "unit": "Ang",
            "meaning": "Root-mean-square displacement over matched atoms.",
            "physics": "Measures the characteristic amplitude of atomic relaxation.",
            "calculation": "sqrt(mean(d_i^2)).",
            "method": "Atoms are matched by a species-resolved Hungarian assignment.",
            "equation": r"\mathrm{RMSD} = \sqrt{\frac{1}{N}\sum_i d_i^2}",
            "symbols": [
                {"symbol": r"d_i", "meaning": "Matched-atom displacement for pair i."},
                {"symbol": r"N", "meaning": "Number of matched atoms."},
            ],
            "interpretation": "Lower is better.",
            "caveats": "Depends on atom matching quality.",
        }
    )

    math_label = dialog.findChild(MathLabel, "metricEquationLabel")
    scroll = dialog.findChild(QScrollArea, "metricHelpScrollArea")
    symbol_labels = dialog.findChildren(MathLabel, "metricSymbolLabel")

    assert math_label is not None
    assert scroll is not None
    assert math_label.findChildren(QLabel, "mathRenderedLine")
    assert len(symbol_labels) == 2
    assert all(label.findChildren(QLabel, "mathRenderedLine") for label in symbol_labels)

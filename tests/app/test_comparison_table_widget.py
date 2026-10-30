"""Tests for comparison table rendering widget."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from mufinder.app.comparison_export import export_comparison_csv
from mufinder.app.comparison_table_widget import COMPARISON_METRICS_URL, ComparisonTableWidget
from mufinder.branding import DOCUMENTATION_URL
from mufinder.domain.models import ComparisonTable
from mufinder.science.structure_comparison import (
    ComparisonResult,
    DftComparisonInput,
    compare_structures,
)
from mufinder.science.structure_metrics import MetricValue, metric_definitions
from mufinder.science.supercell_matching import tile_to_supercell
from tests.support.structures import host_structure

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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

    headers = [
        widget._grid.horizontalHeaderItem(index).text()
        for index in range(widget._grid.columnCount())
    ]

    assert widget._title_edit.text() == "Comparison A"
    assert headers == ["Metric", "Experimental", "DFT (pristine_relax_01)"]
    assert widget._grid.verticalHeader().isVisible() is False
    assert widget._parameter_diff_section_row == 0
    assert widget._parameter_diff_toggle is not None
    assert widget._parameter_diff_toggle.objectName() == "sectionToggleButton"
    assert widget._warning_banner.isHidden() is False


def test_comparison_table_widget_emits_normalized_title_edits(qapp) -> None:
    table = _table()
    widget = ComparisonTableWidget(table)
    emitted: list[tuple[object, str]] = []
    widget.title_changed.connect(lambda table_id, title: emitted.append((table_id, title)))

    widget._title_edit.setText("  Renamed comparison  ")
    widget._title_edit.editingFinished.emit()

    assert emitted == [(table.id, "Renamed comparison")]
    assert widget._title_edit.text() == "Renamed comparison"


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


def _metric_labels(widget: ComparisonTableWidget) -> list[str]:
    grid = widget._grid
    labels: list[str] = []
    for row in range(grid.rowCount()):
        item = grid.item(row, 0)
        if item is not None and grid.columnSpan(row, 0) == 1:
            labels.append(item.text())
    return labels


def _full_table(column_count: int) -> ComparisonTable:
    """A table with every defined metric, as a comparison saves it."""

    keys = [definition.key for definition in metric_definitions()]
    return ComparisonTable(
        label="Comparison D",
        result_data={
            "metric_definitions": [{"key": key} for key in keys],
            "experimental_values": {key: {"display": "4.695", "value": 4.695} for key in keys},
            "dft_columns": [
                {
                    "label": f"DFT (PBE {index})",
                    "values": {
                        key: {"display": "4.7091 (+0.30 %)", "value": 4.7091 + index}
                        for key in keys
                    },
                }
                for index in range(column_count)
            ],
            "parameter_differences": [],
            "warnings": [],
        },
    )


def test_metric_rows_are_labelled_items_with_a_tooltip_and_no_info_button(qapp) -> None:
    widget = ComparisonTableWidget(_full_table(1))
    grid = widget._grid

    assert _metric_labels(widget) == [
        "a (Å)",
        "b (Å)",
        "c (Å)",
        "α (°)",
        "β (°)",
        "γ (°)",
        "V (Å³)",
        "RMSD (Å)",
        "Max displacement (Å)",
        "Space group",
        "Same space group",
    ]
    meanings = {definition.label: definition.meaning for definition in metric_definitions()}
    for row in range(grid.rowCount()):
        assert grid.cellWidget(row, 0) is None
        item = grid.item(row, 0)
        if grid.columnSpan(row, 0) == 1:
            label = item.text().split(" (")[0]
            assert item.toolTip() == meanings[label]
    assert not [button for button in widget.findChildren(QPushButton) if button.text() == "Info"]


def test_metric_rows_are_grouped_under_the_three_groups(qapp) -> None:
    widget = ComparisonTableWidget(_full_table(1))
    grid = widget._grid

    groups = [
        grid.item(row, 0).text()
        for row in range(grid.rowCount())
        if grid.columnSpan(row, 0) > 1 and grid.item(row, 0) is not None
    ]
    assert groups == ["Lattice parameters", "Atomic displacements", "Symmetry"]


def test_about_link_opens_the_comparison_page_of_the_user_guide(qapp) -> None:
    widget = ComparisonTableWidget(_full_table(1))
    link = widget._about_link

    assert "About these metrics" in link.text()
    assert COMPARISON_METRICS_URL in link.text()
    assert link.openExternalLinks() is True
    assert link.accessibleName() == "About these metrics"
    assert link.textInteractionFlags() & Qt.TextInteractionFlag.LinksAccessibleByKeyboard
    assert link.toolTip().endswith(".")
    # The link names a page the documentation build publishes.
    page = COMPARISON_METRICS_URL.removeprefix(DOCUMENTATION_URL).removesuffix(".html")
    assert (REPOSITORY_ROOT / "docs" / f"{page}.rst").is_file()


def test_saved_table_with_old_metric_keys_opens_and_skips_undefined_rows(qapp) -> None:
    table = ComparisonTable(
        label="Old comparison",
        result_data={
            "metric_definitions": [
                {
                    "key": "cell_volume",
                    "label": "Cell Volume",
                    "unit": "Ang^3",
                    "group": "Lattice Parameters",
                },
                {"key": "delta_a_pct", "label": "Change in a vs experiment", "unit": "%"},
                {"key": "mean_displacement", "label": "Mean displacement", "unit": "Ang"},
                {"key": "rmsd", "label": "RMSD (matched atoms)", "unit": "Ang"},
                {"key": "topology_preserved", "label": "Topology/connectivity preserved"},
                {"key": "space_group", "label": "Space group comparison", "group": "Symmetry"},
                {"key": "similarity_score", "label": "Similarity score", "unit": "0-1"},
            ],
            "experimental_values": {
                "cell_volume": {"display": "70.0968", "value": 70.0968},
                "rmsd": {"display": "0", "value": 0.0},
                "space_group": {"display": "Reference", "value": "reference"},
                "similarity_score": {"display": "1", "value": 1.0},
            },
            "dft_columns": [
                {
                    "label": "DFT (PBE)",
                    "values": {
                        "cell_volume": {"display": "70.6915", "value": 70.6915},
                        "rmsd": {"display": "0.00706888", "value": 0.00706888},
                        "space_group": {"display": "Match (P4_2/mnm)", "value": "Match (P4_2/mnm)"},
                        "similarity_score": {"display": "1", "value": 1.0},
                    },
                }
            ],
            "parameter_differences": [],
            "warnings": [],
        },
    )

    widget = ComparisonTableWidget(table)

    # Labels, units and groups come from the current definitions.
    assert _metric_labels(widget) == ["V (Å³)", "RMSD (Å)", "Space group"]
    grid = widget._grid
    rows = {grid.item(row, 0).text(): row for row in range(grid.rowCount())}
    assert grid.item(rows["V (Å³)"], 2).text() == "70.6915"
    assert grid.item(rows["Space group"], 2).text() == "Match (P4_2/mnm)"


def test_table_fits_the_page_without_scrolling_for_two_dft_columns(qapp) -> None:
    for column_count in (1, 2):
        widget = ComparisonTableWidget(_full_table(column_count))
        widget.resize(900, 660)
        widget.show()
        qapp.processEvents()
        grid = widget._grid

        assert grid.verticalScrollBar().maximum() == 0
        assert grid.horizontalScrollBar().maximum() == 0
        assert grid.columnWidth(0) < 200
        widget.close()


def _computed_table(result: ComparisonResult) -> ComparisonTable:
    """A table holding ``result``'s values and displays, as generating one saves them."""

    def payload(values: dict[str, MetricValue]) -> dict[str, dict[str, object]]:
        return {
            key: {"value": value.value, "display": value.display} for key, value in values.items()
        }

    return ComparisonTable(
        label="Computed",
        result_data={
            "metric_definitions": [{"key": definition.key} for definition in result.definitions],
            "experimental_values": payload(result.experimental_values),
            "dft_columns": [
                {"label": column.label, "values": payload(column.values)}
                for column in result.dft_columns
            ],
            "parameter_differences": [],
            "warnings": list(result.warnings),
        },
    )


def _supercell_comparison() -> ComparisonResult:
    """CoF₂ against a primitive cell 2 % off, a 2 × 2 × 2 and a rotated supercell."""

    host = host_structure(
        label="CoF2",
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
    columns = []
    for label, matrix, scale in (
        ("unit", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), 1.02),
        ("2x2x2", ((2, 0, 0), (0, 2, 0), (0, 0, 2)), 1.004),
        ("rotated", ((1, 1, 0), (-1, 1, 0), (0, 0, 1)), 1.006),
    ):
        cell = tile_to_supercell(host, matrix)
        cell.lattice_vectors = (np.asarray(cell.lattice_vectors) * scale).tolist()
        columns.append(DftComparisonInput(label=label, structure=cell, structure_id=cell.id))
    return compare_structures(host, columns)


def _row(widget: ComparisonTableWidget, label: str) -> int:
    grid = widget._grid
    return next(row for row in range(grid.rowCount()) if grid.item(row, 0).text() == label)


def _row_texts(widget: ComparisonTableWidget, label: str) -> list[str]:
    grid = widget._grid
    row = _row(widget, label)
    return [grid.item(row, column).text() for column in range(1, grid.columnCount())]


def test_supercell_columns_show_host_cell_values_and_can_win_the_highlight(qapp) -> None:
    widget = ComparisonTableWidget(_computed_table(_supercell_comparison()))

    assert _row_texts(widget, "a (Å)") == [
        "4.695",
        "4.7889 (+2.00 %)",
        "4.71378 (+0.40 %)",
        "4.72317 (+0.60 %)",
    ]
    assert _row_texts(widget, "γ (°)")[1:] == ["90 (+0.00 %)"] * 3
    # Per host cell the 2 × 2 × 2 column is closest, so it wins the a and V rows;
    # the rotated supercell is next and the primitive cell last.
    grid = widget._grid
    for label in ("a (Å)", "V (Å³)"):
        row = _row(widget, label)
        assert [grid.item(row, column).font().bold() for column in range(2, 5)] == [
            False,
            True,
            False,
        ]
    banner = widget._warning_banner.text()
    assert "2x2x2: 2 × 2 × 2 supercell; its lattice parameters are shown per host cell." in banner
    assert "rotated: Supercell of 2 host cells" in banner


def test_csv_export_of_supercell_columns_matches_the_table(qapp, tmp_path: Path) -> None:
    table = _computed_table(_supercell_comparison())
    widget = ComparisonTableWidget(table)
    output_path = tmp_path / "comparison.csv"

    export_comparison_csv(table, output_path)

    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = {row[0]: row[2:] for row in csv.reader(handle)}
    for definition in metric_definitions():
        label = f"{definition.label} ({definition.unit})" if definition.unit else definition.label
        assert rows[definition.label] == _row_texts(widget, label), definition.key

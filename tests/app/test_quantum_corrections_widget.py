"""The Quantum Corrections page: three steps above one table of sites."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QGroupBox, QHeaderView, QLabel

from mufinder.app import theme
from mufinder.app.quantum_corrections_widget import (
    QuantumCorrectionsWidget,
    QuantumSourceRow,
    SITE_COLUMNS,
)
from mufinder.app.theme import status_brush
from tests.support.reset_menu import install_fake_menu, open_reset_menu

_STATUS_COLUMN = 8


def _row(index: int = 0, status: str = "ready", **fields) -> QuantumSourceRow:
    values = {
        "key": f"site-{index}",
        "label": f"Site {index}",
        "charge_state": "positive muon",
        "relative_energy": "0.000",
        "displacements": "9",
        "source_job": f"relax_{index}",
        "num_farms_display": "1",
        "status": status,
    }
    return QuantumSourceRow(**{**values, **fields})


def _completed(index: int = 0, **fields) -> QuantumSourceRow:
    values = {
        "zpe": "0.558",
        "corrected_energy": "0.000",
        "widths": "0.36 0.21 0.18",
        "witness": "-0.008",
        "muon_modes": "1136.3, 3225.0, 4632.8",
        "result_id": f"result-{index}",
    }
    return _row(index, "completed", **{**values, **fields})


def _cells(widget: QuantumCorrectionsWidget, row: int) -> list[str]:
    table = widget._sources_table
    return [table.item(row, column).text() for column in range(table.columnCount())]


# --- the table ---------------------------------------------------------------------------


def test_one_table_lists_a_site_with_its_result_and_a_dash_without(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            _completed(0, pinned_label="8g · site 1"),
            _row(1, "ready", relative_energy="+0.257", pinned_label="16m · site 2"),
        ]
    )
    table = widget._sources_table

    assert [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())] == [
        "Site",
        "Charge",
        "dE (eV)",
        "ZPE (eV)",
        "dE + ZPE (eV)",
        "Widths (Å)",
        "w1",
        "Farms",
        "Status",
    ]
    assert _cells(widget, 0) == [
        "8g · site 1",
        "positive muon",
        "0.000",
        "0.558",
        "0.000",
        "0.36 0.21 0.18",
        "-0.008",
        "1",
        "completed",
    ]
    assert _cells(widget, 1) == [
        "16m · site 2",
        "positive muon",
        "+0.257",
        "–",
        "–",
        "–",
        "–",
        "1",
        "ready",
    ]


def test_every_header_has_a_one_line_tooltip(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    table = widget._sources_table

    for column, (title, tooltip) in enumerate(SITE_COLUMNS):
        item = table.horizontalHeaderItem(column)
        assert item.text() == title
        assert item.toolTip() == tooltip
        assert tooltip and "\n" not in tooltip


def test_site_column_takes_the_spare_width(qapp) -> None:
    header = QuantumCorrectionsWidget()._sources_table.horizontalHeader()

    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    for column in range(1, len(SITE_COLUMNS)):
        assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.ResizeToContents


def test_displacements_source_job_and_modes_move_to_the_row_tooltip(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_completed(0, pinned_label="8g · site 1"), _row(1)])
    table = widget._sources_table

    for column in range(table.columnCount()):
        assert table.item(0, column).toolTip() == (
            "Relaxed site: Site 0\n"
            "Displacements: 9\n"
            "Source job: relax_0\n"
            "Muon modes (cm⁻¹): 1136.3, 3225.0, 4632.8"
        )
    assert table.item(1, 0).toolTip() == "Displacements: 9\nSource job: relax_1"


def test_page_has_no_physics_prose_or_second_table(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    text = " ".join(label.text() for label in widget.findChildren(QLabel))

    for phrase in ("zero-point", "harmonic", "entangled", "Completed", "witness"):
        assert phrase not in text
    assert widget.findChildren(QGroupBox) == []
    assert not hasattr(widget, "_results_table")


def test_set_sources_skips_rebuild_when_unchanged(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_row()])
    set_counts: list[int] = []
    original = widget._sources_table.setRowCount
    widget._sources_table.setRowCount = lambda count: (set_counts.append(count), original(count))[1]

    # Equal-by-value rows must not rebuild the table.
    widget.set_sources([_row()])
    assert set_counts == []

    # A changed row must rebuild.
    widget.set_sources([_row(status="running")])
    assert set_counts != []


def test_set_sources_empty_then_populated(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([])
    assert widget._sources_table.rowCount() == 0
    widget.set_sources([_row()])
    assert widget._sources_table.rowCount() == 1
    widget.set_sources([])
    assert widget._sources_table.rowCount() == 0


def test_status_uses_the_shared_colours_and_follows_a_theme_change(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_row(status="failed")])
    status_item = widget._sources_table.item(0, _STATUS_COLUMN)
    assert status_item.foreground().color().name() == status_brush("failed").color().name()

    status_item.setForeground(QBrush(QColor("#123456")))
    widget.changeEvent(QEvent(QEvent.Type.PaletteChange))

    assert status_item.foreground().color().name() == status_brush("failed").color().name()


def test_a_rejected_rename_restores_the_site_cell(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_row()])
    renames: list[tuple[str, str]] = []
    widget.source_label_renamed.connect(lambda key, label: renames.append((key, label)))

    widget._sources_table.item(0, 0).setText("")

    assert renames == [("site-0", "")]
    assert widget._sources_table.item(0, 0).text() == "Site 0"


# --- step 1: sites -----------------------------------------------------------------------


def test_sites_step_counts_the_pinned_rows_and_any_kept_extras(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    assert widget._source_label.text() == "0 pinned from Muon Sites"

    widget.set_sources([_row(0, pinned_label="8g · site 1"), _row(1, pinned_label="16m · site 2")])
    assert widget._source_label.text() == "2 pinned from Muon Sites"

    widget.set_sources([_row(0, pinned_label="8g · site 1"), _row(1, "running")])
    assert widget._source_label.text() == "1 pinned from Muon Sites, 1 more already set up"

    widget.set_site_scope(show_every_site=True)
    assert widget._source_label.text() == "2 relaxed sites"


def test_sites_step_shows_the_notice_when_nothing_is_pinned(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_site_scope(show_every_site=False, notice="Nothing is pinned yet.")

    assert widget._site_scope.notice_text() == "Nothing is pinned yet."


# --- step 2: run -------------------------------------------------------------------------


def test_run_step_counts_completed_rows(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    assert widget._run_status_label.text() == "No sites to run"

    widget.set_sources([_completed(0), _row(1, "running"), _row(2, "failed")])
    assert widget._run_status_label.text() == "1 of 3 completed"

    widget.set_sources([_completed(0), _completed(1)])
    assert widget._run_status_label.text() == "2 of 2 completed"


def test_run_buttons_run_all_rows_when_nothing_is_selected(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_row(0), _row(1)])
    local: list[list[str]] = []
    cluster: list[list[str]] = []
    widget.submit_local_requested.connect(local.append)
    widget.submit_cluster_requested.connect(cluster.append)

    assert widget._submit_local_button.text() == "Run All Locally"
    assert widget._submit_local_button.isEnabled()
    widget._submit_local_button.click()
    widget._submit_cluster_button.click()
    assert local == [["site-0", "site-1"]]
    assert cluster == [["site-0", "site-1"]]

    widget._sources_table.selectRow(1)
    assert widget._submit_local_button.text() == "Run Selected Locally"
    widget._submit_local_button.click()
    assert local[-1] == ["site-1"]


def test_configure_targets_the_selection_then_rows_not_yet_run_then_every_row(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    configured: list[object] = []
    widget.configure_requested.connect(configured.append)
    assert widget._batch_configure_button.text() == "Configure…"
    assert widget._batch_configure_button.isEnabled() is False

    widget.set_sources([_completed(0), _row(1, "ready"), _row(2, "defined")])
    assert widget._batch_configure_button.isEnabled() is True
    widget._batch_configure_button.click()
    assert configured[-1] == ["site-1", "site-2"]

    widget._sources_table.selectRow(0)
    widget._batch_configure_button.click()
    assert configured[-1] == ["site-0"]

    widget._sources_table.clearSelection()
    widget.set_sources([_completed(0), _completed(1)])
    widget._batch_configure_button.click()
    assert configured[-1] == ["site-0", "site-1"]


def test_there_is_no_per_row_configure_button(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_row()])
    table = widget._sources_table

    assert all(table.cellWidget(0, column) is None for column in range(table.columnCount()))


def test_reset_failed_button_follows_the_selection(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_row(0, "failed"), _row(1, "completed"), _row(2, "stopped")])
    emitted: list[object] = []
    widget.reset_failed_requested.connect(emitted.append)
    install_fake_menu(monkeypatch)

    assert widget._reset_failed_button.text() == "Reset Failed Jobs…"
    assert widget._reset_failed_button.isEnabled() is False

    widget._sources_table.selectRow(1)
    assert widget._reset_failed_button.isEnabled() is False
    menu = open_reset_menu(widget._sources_table, row=1)
    assert menu is not None and menu.action("Reset Failed Jobs…").enabled is False

    widget._sources_table.selectAll()
    assert widget._reset_failed_button.isEnabled() is True
    widget._reset_failed_button.click()
    assert emitted == [["site-0", "site-2"]]


def test_row_menu_offers_the_shared_failed_job_reset_and_no_remove(
    qapp, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_row(0, "failed"), _row(1, "running"), _row(2, "cancelled")])
    cancelled: list[object] = []
    reset: list[object] = []
    configured: list[object] = []
    widget.cancel_requested.connect(cancelled.append)
    widget.reset_failed_requested.connect(reset.append)
    widget.configure_requested.connect(configured.append)
    install_fake_menu(monkeypatch)
    widget._sources_table.selectAll()

    menu = open_reset_menu(widget._sources_table)

    texts = [action.text for action in menu.actions]
    assert [text.split(" (")[0] for text in texts] == [
        "Run Local",
        "Run on Cluster",
        "Cancel",
        "Inspect Failure",
        "Reset Failed Jobs…",
        "Configure…",
        "Open Containing Folder",
        "Export Inputs/Outputs…",
    ]
    assert "Reset Failed Jobs… (2 of 3)" in texts
    menu.action("Cancel (1 of 3)").triggered.emit()
    assert cancelled == [["site-1"]]
    # Opening the menu fired the enabled reset: only the failed and cancelled rows go.
    assert reset == [["site-0", "site-2"]]


# --- step 3: results ---------------------------------------------------------------------


def test_results_step_hands_off_once_a_result_exists(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_row(0), _row(1)])
    assert widget._results_status_label.text() == "No results yet"
    assert widget._next_button.isHidden()

    widget.set_sources([_completed(0), _row(1)])
    assert widget._results_status_label.text() == "Select a site"
    assert not widget._next_button.isHidden()
    assert widget._next_button.text() == "Next: Energy Barriers →"

    opened: list[bool] = []
    widget.open_energy_barriers_requested.connect(lambda: opened.append(True))
    widget._next_button.click()
    assert opened == [True]


def test_open_analysis_acts_on_the_one_selected_row_with_a_result(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_completed(0, pinned_label="8g · site 1"), _row(1), _completed(2)])
    opened: list[str] = []
    widget.analysis_requested.connect(opened.append)
    button = widget._open_analysis_button
    assert button.text() == "Open Analysis"
    assert button.isEnabled() is False

    widget._sources_table.selectRow(1)
    assert button.isEnabled() is False
    assert widget._results_status_label.text() == "No result for this site yet"

    widget._sources_table.selectRow(0)
    assert button.isEnabled() is True
    assert widget._results_status_label.text() == "Site selected"
    button.click()
    assert opened == ["result-0"]

    widget._sources_table.selectAll()
    assert button.isEnabled() is False
    assert widget._results_status_label.text() == "Select a site"


# --- layout ------------------------------------------------------------------------------


def test_page_needs_less_width_than_the_default_window(qapp) -> None:
    theme.apply_theme(qapp, theme.THEME_LIGHT)
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            _completed(0, pinned_label="16m · cluster 12"),
            _row(1, "running", pinned_label="8g · cluster 2"),
        ]
    )
    widget.set_site_scope(show_every_site=False, notice="Nothing is pinned yet. " * 10)

    # The default window is 1333 px wide, and the navigator takes its share of it.
    assert widget.minimumSizeHint().width() <= 1000
    widget.close()

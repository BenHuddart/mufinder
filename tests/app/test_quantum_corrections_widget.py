"""Flush-path guard tests for the Quantum Corrections widget."""

from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QHeaderView, QLabel

from mufinder.app.quantum_corrections_widget import (
    QuantumCorrectionsWidget,
    QuantumResultRow,
    QuantumSourceRow,
)
from mufinder.app.theme import status_brush


def _source_row(key: str = "site-a", status: str = "ready") -> QuantumSourceRow:
    return QuantumSourceRow(
        key=key,
        label="Site A",
        charge_state="neutral",
        relative_energy="0.00",
        displacements="9",
        source_job="relax-a",
        stage="Stage 1",
        num_farms_display="",
        status=status,
    )


def _result_row(key: str = "res-a", status: str = "completed") -> QuantumResultRow:
    return QuantumResultRow(
        key=key,
        label="Quantum phonon",
        zpe="0.580",
        muon_modes="2742.0, 3226.0, 3226.0",
        widths="0.23, 0.21, 0.21",
        witness="-0.14",
        status=status,
    )


def test_set_sources_skips_rebuild_when_unchanged(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    rows = [_source_row()]
    widget.set_sources(rows)
    assert widget._sources_table.rowCount() == 1

    set_counts: list[int] = []
    original = widget._sources_table.setRowCount
    widget._sources_table.setRowCount = lambda count: (set_counts.append(count), original(count))[1]

    # Equal-by-value rows must not rebuild the table.
    widget.set_sources([_source_row()])
    assert set_counts == []

    # A changed row must rebuild.
    widget.set_sources([_source_row(status="running")])
    assert set_counts != []


def test_set_results_skips_rebuild_when_unchanged(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_results([_result_row()])
    assert widget._results_table.rowCount() == 1

    set_counts: list[int] = []
    original = widget._results_table.setRowCount
    widget._results_table.setRowCount = lambda count: (set_counts.append(count), original(count))[1]

    widget.set_results([_result_row()])
    assert set_counts == []

    widget.set_results([_result_row(status="failed")])
    assert set_counts != []


def test_theme_change_reapplies_status_colors(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([_source_row(status="running")])
    widget.set_results([_result_row(status="completed")])

    # A palette change must re-colour status cells without a table rebuild,
    # and must not raise.
    widget.changeEvent(QEvent(QEvent.Type.PaletteChange))

    assert widget._sources_table.item(0, 7) is not None
    assert widget._results_table.item(0, 5) is not None


def test_results_table_columns_and_header_tooltips(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    table = widget._results_table

    headers = [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]
    assert headers == [
        "Job",
        "ZPE (eV)",
        "Muon modes (cm⁻¹)",
        "Width (Å)",
        "w1",
        "Status",
        "Analysis",
    ]
    tooltips = {
        headers[column]: table.horizontalHeaderItem(column).toolTip()
        for column in range(table.columnCount())
    }
    assert tooltips["ZPE (eV)"] == (
        "Total muon zero-point energy, ½ħω summed over the three muon-dominated Γ modes"
    )
    assert tooltips["Muon modes (cm⁻¹)"] == (
        "Frequencies of the three Γ modes with the largest muon participation, in frequency order"
    )
    assert tooltips["Width (Å)"] == "Zero-point width √(ħ/2m_μω) of the muon along each muon mode"
    assert tooltips["w1"].startswith("Mode-mixing witness P − 3: 0 when")
    assert "down to −3" in tooltips["w1"]

    header = table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    for column in range(1, table.columnCount()):
        assert header.sectionResizeMode(column) == QHeaderView.ResizeMode.ResizeToContents


def test_results_row_fills_each_column(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_results([_result_row()])
    table = widget._results_table

    assert [table.item(0, column).text() for column in range(6)] == [
        "Quantum phonon",
        "0.580",
        "2742.0, 3226.0, 3226.0",
        "0.23, 0.21, 0.21",
        "-0.14",
        "completed",
    ]
    assert table.item(0, 5).foreground().color().name() == status_brush("completed").color().name()
    assert table.cellWidget(0, 6).text() == "Launch Analysis"


def test_theme_change_recolours_the_results_status_column(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_results([_result_row(status="failed")])
    status_item = widget._results_table.item(0, 5)
    status_item.setForeground(QBrush(QColor("#123456")))

    widget.changeEvent(QEvent(QEvent.Type.PaletteChange))

    assert status_item.foreground().color().name() == status_brush("failed").color().name()


def test_set_sources_empty_then_populated(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([])
    assert widget._sources_table.rowCount() == 0
    widget.set_sources([_source_row()])
    assert widget._sources_table.rowCount() == 1
    widget.set_sources([])
    assert widget._sources_table.rowCount() == 0


def test_page_guidance_explains_purpose_and_source_headers(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    text = " ".join(label.text() for label in widget.findChildren(QLabel))
    assert "zero-point energy" in text
    assert "entangled" in text
    assert "harmonic" in text
    assert "relaxed endpoint" not in text
    assert "locked once the row's first job has been submitted" in text
    assert "mode-mixing witness w1" in text

    table = widget._sources_table
    tooltips = {}
    for column in range(table.columnCount()):
        item = table.horizontalHeaderItem(column)
        tooltips[item.text()] = item.toolTip()
    for header in ("Site", "Charge", "dE (eV)", "Displacements", "Source Job", "Status", "Parameters"):
        assert tooltips[header].strip(), header
    assert "farm count" in tooltips["Displacements"]

"""Flush-path guard tests for the Quantum Corrections widget."""

from __future__ import annotations

from PySide6.QtCore import QEvent

from mufinder.app.quantum_corrections_widget import (
    QuantumCorrectionsWidget,
    QuantumResultRow,
    QuantumSourceRow,
)


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
        zpe="0.10",
        eigenvalues="0:120.0",
        witness="0.5",
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
    assert widget._results_table.item(0, 4) is not None


def test_set_sources_empty_then_populated(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources([])
    assert widget._sources_table.rowCount() == 0
    widget.set_sources([_source_row()])
    assert widget._sources_table.rowCount() == 1
    widget.set_sources([])
    assert widget._sources_table.rowCount() == 0

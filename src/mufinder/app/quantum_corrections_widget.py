"""Quantum Corrections workflow surface."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import style_section_title, use_square_top_table_corners


@dataclass(frozen=True, slots=True)
class QuantumSourceRow:
    """One relaxed-site source displayed in the Quantum Corrections view."""

    label: str
    charge_state: str
    relative_energy: str
    source_job: str
    status: str


@dataclass(frozen=True, slots=True)
class QuantumResultRow:
    """One completed quantum-corrections result row."""

    label: str
    zpe: str
    modes: str
    witness: str
    phonon_file: str


class QuantumCorrectionsWidget(QWidget):
    """Display eligible relaxed muon sources and completed phonon analyses."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Quantum Corrections workflow")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Quantum Corrections")
        style_section_title(title)
        layout.addWidget(title)

        self._source_label = QLabel("Eligible relaxed sites")
        style_section_title(self._source_label)
        layout.addWidget(self._source_label)
        self._sources_table = _table(("Site", "Charge", "Relative Energy", "Source Job", "Status"))
        layout.addWidget(self._sources_table)

        self._result_label = QLabel("Completed phonon analyses")
        style_section_title(self._result_label)
        layout.addWidget(self._result_label)
        self._results_table = _table(("Job", "ZPE", "Modes", "w1", "Phonon File"))
        layout.addWidget(self._results_table)

        layout.addStretch(1)

    def set_sources(self, rows: list[QuantumSourceRow]) -> None:
        """Replace the displayed eligible-source rows."""

        self._source_label.setText(f"Eligible relaxed sites ({len(rows)})")
        _fill_table(
            self._sources_table,
            [
                (row.label, row.charge_state, row.relative_energy, row.source_job, row.status)
                for row in rows
            ],
        )

    def set_results(self, rows: list[QuantumResultRow]) -> None:
        """Replace the displayed quantum-result rows."""

        self._result_label.setText(f"Completed phonon analyses ({len(rows)})")
        _fill_table(
            self._results_table,
            [
                (row.label, row.zpe, row.modes, row.witness, row.phonon_file)
                for row in rows
            ],
        )


def _table(headers: tuple[str, ...]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setFrameShape(QFrame.Shape.NoFrame)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setStretchLastSection(True)
    use_square_top_table_corners(table)
    return table


def _fill_table(table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row_index, column, item)
    table.resizeRowsToContents()

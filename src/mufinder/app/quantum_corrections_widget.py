"""Quantum Corrections workflow surface."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import style_hint_text, style_section_subtitle, style_section_title
from mufinder.app.workflow_table_utils import (
    create_workflow_table,
    set_workflow_table_item,
    style_workflow_configure_button,
    workflow_row_is_locked,
)


@dataclass(frozen=True, slots=True)
class QuantumSourceRow:
    """One relaxed-site source displayed in the Quantum Corrections view."""

    key: str
    label: str
    charge_state: str
    relative_energy: str
    source_job: str
    status: str
    configured: bool = False


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

    configure_requested = Signal(str)
    submit_local_requested = Signal(list)
    submit_cluster_requested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Quantum Corrections workflow")
        self._source_rows: list[QuantumSourceRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header_row = QVBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(3)

        title = QLabel("Quantum Corrections")
        style_section_title(title)
        header_row.addWidget(title)

        subtitle = QLabel(
            "Estimate local muon zero-point motion and phonon-derived witnesses from completed structural relaxations."
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        header_row.addWidget(subtitle)

        summary = QLabel(
            "Choose relaxed muon sites, optionally refine the key phonon settings, and submit only the rows you want to run."
        )
        summary.setWordWrap(True)
        style_hint_text(summary)
        header_row.addWidget(summary)
        layout.addLayout(header_row)

        sources_group = QGroupBox("Quantum Phonon Jobs")
        sources_layout = QVBoxLayout(sources_group)
        sources_layout.setContentsMargins(10, 8, 10, 8)
        sources_layout.setSpacing(6)

        self._source_label = QLabel("Eligible relaxed sites")
        style_section_title(self._source_label)
        sources_layout.addWidget(self._source_label)
        source_hint = QLabel(
            "Each row inherits a relaxed endpoint structure. Configure the default phonon controls directly, or add other CASTEP parameters from the searchable list."
        )
        source_hint.setWordWrap(True)
        style_hint_text(source_hint)
        sources_layout.addWidget(source_hint)
        self._sources_table = create_workflow_table(
            ("Site", "Charge", "Relative Energy", "Source Job", "Status", "Parameters"),
            resize_to_contents_columns=(5,),
        )
        self._sources_table.itemSelectionChanged.connect(self._update_actions)
        sources_layout.addWidget(self._sources_table)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._submit_local_button = QPushButton("Run Local Calculations")
        self._submit_cluster_button = QPushButton("Run on Cluster")
        self._submit_local_button.clicked.connect(self._emit_submit_local)
        self._submit_cluster_button.clicked.connect(self._emit_submit_cluster)
        actions.addWidget(self._submit_local_button)
        actions.addWidget(self._submit_cluster_button)
        actions.addStretch(1)
        sources_layout.addLayout(actions)
        layout.addWidget(sources_group)

        results_group = QGroupBox("Completed Results")
        results_layout = QVBoxLayout(results_group)
        results_layout.setContentsMargins(10, 8, 10, 8)
        results_layout.setSpacing(6)

        self._result_label = QLabel("Completed phonon analyses")
        style_section_title(self._result_label)
        results_layout.addWidget(self._result_label)
        result_hint = QLabel(
            "MuFinder reports the parsed phonon file, selected-mode ZPE, and witness values for completed Gamma-point phonon jobs."
        )
        result_hint.setWordWrap(True)
        style_hint_text(result_hint)
        results_layout.addWidget(result_hint)
        self._results_table = create_workflow_table(("Job", "ZPE", "Modes", "w1", "Phonon File"))
        results_layout.addWidget(self._results_table)
        layout.addWidget(results_group, 1)

        layout.addStretch(1)

    def set_sources(self, rows: list[QuantumSourceRow]) -> None:
        """Replace the displayed eligible-source rows."""

        self._source_rows = list(rows)
        self._source_label.setText(f"Eligible relaxed sites ({len(rows)})")
        _fill_source_table(self._sources_table, rows, configure_requested=self.configure_requested.emit)
        self._update_actions()

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

    def selected_source_keys(self) -> list[str]:
        """Return selected source keys in table order."""

        indexes = sorted({index.row() for index in self._sources_table.selectedIndexes()})
        keys: list[str] = []
        for row in indexes:
            if 0 <= row < len(self._source_rows):
                keys.append(self._source_rows[row].key)
        return keys

    def _emit_submit_local(self) -> None:
        self.submit_local_requested.emit(self.selected_source_keys())

    def _emit_submit_cluster(self) -> None:
        self.submit_cluster_requested.emit(self.selected_source_keys())

    def _update_actions(self) -> None:
        enabled = bool(self.selected_source_keys())
        self._submit_local_button.setEnabled(enabled)
        self._submit_cluster_button.setEnabled(enabled)

def _fill_source_table(
    table: QTableWidget,
    rows: list[QuantumSourceRow],
    *,
    configure_requested,
) -> None:
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column, value in enumerate((row.label, row.charge_state, row.relative_energy, row.source_job, row.status)):
            set_workflow_table_item(table, row_index, column, value)
        configure_button = QPushButton(table)
        style_workflow_configure_button(
            configure_button,
            workflow_label="Quantum Corrections",
            configured=row.configured,
            locked=workflow_row_is_locked(row.status),
        )
        configure_button.clicked.connect(lambda _checked=False, key=row.key: configure_requested(key))
        table.setCellWidget(row_index, 5, configure_button)
    table.resizeRowsToContents()


def _fill_table(table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column, value in enumerate(values):
            set_workflow_table_item(table, row_index, column, value)
    table.resizeRowsToContents()

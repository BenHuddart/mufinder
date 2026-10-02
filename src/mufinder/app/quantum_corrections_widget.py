"""Quantum Corrections workflow surface."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.selection_snapshot import SelectionSnapshotPushButton, resolve_target_keys
from mufinder.app.theme import status_brush, style_hint_text, style_section_subtitle, style_section_title
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
    displacements: str
    source_job: str
    stage: str
    num_farms_display: str
    status: str
    configured: bool = False
    resettable: bool = False


@dataclass(frozen=True, slots=True)
class QuantumResultRow:
    """One completed quantum-corrections result row."""

    key: str
    label: str
    zpe: str
    eigenvalues: str
    witness: str
    status: str


class QuantumCorrectionsWidget(QWidget):
    """Display eligible relaxed muon sources and completed phonon analyses."""

    configure_requested = Signal(object)
    source_label_renamed = Signal(str, str)
    submit_local_requested = Signal(list)
    submit_cluster_requested = Signal(list)
    reset_stage_requested = Signal(list)
    analysis_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Quantum Corrections workflow")
        self._source_rows: list[QuantumSourceRow] = []
        # Flush-path guards: cache the last-rendered rows so set_sources /
        # set_results can skip the table rebuild when nothing changed. The rows
        # are frozen dataclasses, so list equality is an exact value compare.
        self._source_rows_valid = False
        self._result_rows: list[QuantumResultRow] = []
        self._result_rows_valid = False
        self._updating_sources = False

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
            "Each relaxed site runs as a two-stage workflow: a singlepoint first, then a phonon continuation. "
            "Use the displacement count to choose an appropriate farm count for stage 2."
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
            "Each row inherits a relaxed endpoint structure. Shared electronic settings apply to both stages, "
            "while phonon method, pseudopotentials, and farms apply only to the stage-2 continuation."
        )
        source_hint.setWordWrap(True)
        style_hint_text(source_hint)
        sources_layout.addWidget(source_hint)
        self._sources_table = create_workflow_table(
            ("Site", "Charge", "Relative Energy", "Displacements", "Source Job", "Stage", "Farms", "Status", "Parameters"),
            resize_to_contents_columns=(8,),
        )
        self._sources_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._sources_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._sources_table.itemSelectionChanged.connect(self._update_actions)
        self._sources_table.itemChanged.connect(self._on_source_item_changed)
        sources_layout.addWidget(self._sources_table)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._submit_local_button = QPushButton("Run Local Calculations")
        self._submit_cluster_button = QPushButton("Run on Cluster")
        self._reset_stage_button = QPushButton("Reset to Stage 1")
        self._batch_configure_button.clicked.connect(self._emit_batch_configure)
        self._submit_local_button.clicked.connect(self._emit_submit_local)
        self._submit_cluster_button.clicked.connect(self._emit_submit_cluster)
        self._reset_stage_button.clicked.connect(self._emit_reset_stage)
        actions.addWidget(self._submit_local_button)
        actions.addWidget(self._submit_cluster_button)
        actions.addWidget(self._reset_stage_button)
        actions.addStretch(1)
        actions.addWidget(self._batch_configure_button)
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
            "MuFinder reports Gamma-point phonon eigenvalues, selected-mode ZPE, and witness values for completed phonon jobs."
        )
        result_hint.setWordWrap(True)
        style_hint_text(result_hint)
        results_layout.addWidget(result_hint)
        self._results_table = create_workflow_table(
            ("Job", "ZPE", "Eigenvalues (cm⁻¹)", "w1", "Status", "Analysis")
        )
        results_header = self._results_table.horizontalHeader()
        results_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        results_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        results_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        results_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        results_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        results_header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        results_layout.addWidget(self._results_table)
        layout.addWidget(results_group, 1)

        layout.addStretch(1)

    def set_sources(self, rows: list[QuantumSourceRow]) -> None:
        """Replace the displayed eligible-source rows.

        Skips the table rebuild when the rows are unchanged since the last
        render; ``QuantumSourceRow`` is a frozen dataclass so list equality is an
        exact value compare. Skipping the rebuild also preserves the user's
        current selection across dashboard flushes.
        """

        if self._source_rows_valid and list(rows) == self._source_rows:
            return
        self._source_rows = list(rows)
        self._source_rows_valid = True
        self._source_label.setText(f"Eligible relaxed sites ({len(rows)})")
        self._updating_sources = True
        try:
            _fill_source_table(self._sources_table, rows, configure_requested=self.configure_requested.emit)
        finally:
            self._updating_sources = False
        self._update_actions()

    def set_results(self, rows: list[QuantumResultRow]) -> None:
        """Replace the displayed quantum-result rows.

        Skips the table rebuild (and the per-row ``QPushButton`` creation and
        column-resize passes) when the rows are unchanged. ``QuantumResultRow``
        is a frozen dataclass, so list equality is exact. Status-cell colours are
        palette-derived and re-applied on theme change by :meth:`changeEvent`.
        """

        if self._result_rows_valid and list(rows) == self._result_rows:
            return
        self._result_rows = list(rows)
        self._result_rows_valid = True
        self._result_label.setText(f"Completed phonon analyses ({len(rows)})")
        _fill_table(
            self._results_table,
            [
                (row.label, row.zpe, row.eigenvalues, row.witness, row.status)
                for row in rows
            ],
        )
        for row_index, row in enumerate(rows):
            # Colour the Status cell with the project-standard per-status palette, matching the
            # eligible-sources table and the relaxation/convergence tables.
            status_item = self._results_table.item(row_index, 4)
            if status_item is not None:
                status_item.setForeground(status_brush(row.status))
            button = QPushButton("Launch Analysis")
            button.clicked.connect(lambda _checked=False, key=row.key: self.analysis_requested.emit(key))
            self._results_table.setCellWidget(row_index, 5, button)
        if self._results_table.rowCount() > 0:
            self._results_table.resizeColumnToContents(1)
            self._results_table.resizeColumnToContents(3)
            self._results_table.resizeColumnToContents(4)
            self._results_table.resizeColumnToContents(5)
            self._results_table.setColumnWidth(3, max(56, self._results_table.columnWidth(3)))

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt API naming.
        """Re-apply palette-derived status colours on a theme change.

        ``set_sources`` / ``set_results`` now skip rebuilds when their rows are
        unchanged, so the status-cell foreground brushes would otherwise stay
        frozen at the previous theme's colours after a palette flip.
        """

        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        }:
            self._reapply_status_colors()

    def _reapply_status_colors(self) -> None:
        for row_index, row in enumerate(self._source_rows):
            status_item = self._sources_table.item(row_index, 7)
            if status_item is not None:
                status_item.setForeground(status_brush(row.status))
        for row_index, row in enumerate(self._result_rows):
            status_item = self._results_table.item(row_index, 4)
            if status_item is not None:
                status_item.setForeground(status_brush(row.status))

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

    def _emit_batch_configure(self) -> None:
        keys = self.selected_source_keys()
        if keys:
            self.configure_requested.emit(keys)

    def _emit_submit_cluster(self) -> None:
        self.submit_cluster_requested.emit(self.selected_source_keys())

    def _emit_reset_stage(self) -> None:
        self.reset_stage_requested.emit(self.selected_source_keys())

    def _update_actions(self) -> None:
        keys = self.selected_source_keys()
        enabled = bool(keys)
        self._batch_configure_button.setEnabled(enabled)
        self._submit_local_button.setEnabled(enabled)
        self._submit_cluster_button.setEnabled(enabled)
        selected_rows = [
            self._source_rows[index]
            for index in sorted({item.row() for item in self._sources_table.selectedIndexes()})
            if 0 <= index < len(self._source_rows)
        ]
        self._reset_stage_button.setEnabled(any(row.resettable for row in selected_rows))

    def _on_source_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_sources or item.column() != 0:
            return
        row = item.row()
        if not (0 <= row < len(self._source_rows)):
            return
        key = self._source_rows[row].key
        self.source_label_renamed.emit(key, item.text())


def _fill_source_table(
    table: QTableWidget,
    rows: list[QuantumSourceRow],
    *,
    configure_requested,
) -> None:
    previous = table.blockSignals(True)
    table.setRowCount(len(rows))
    key_by_row: dict[int, str] = {i: row.key for i, row in enumerate(rows)}
    for row_index, row in enumerate(rows):
        for column, value in enumerate(
            (
                row.label,
                row.charge_state,
                row.relative_energy,
                row.displacements,
                row.source_job,
                row.stage,
                row.num_farms_display,
                row.status,
            )
        ):
            set_workflow_table_item(table, row_index, column, value)
        status_item = table.item(row_index, 7)
        if status_item is not None:
            status_item.setForeground(status_brush(row.status))
        label_item = table.item(row_index, 0)
        if label_item is not None:
            label_item.setFlags(label_item.flags() | Qt.ItemFlag.ItemIsEditable)
        configure_button = SelectionSnapshotPushButton(table)
        configure_button.selection_snapshot_key = row.key
        style_workflow_configure_button(
            configure_button,
            workflow_label="Quantum Corrections",
            configured=row.configured,
            locked=workflow_row_is_locked(row.status),
        )
        def _on_configure_clicked(_checked: bool = False, key=row.key, btn=configure_button) -> None:
            keys = resolve_target_keys(
                btn,
                clicked_key=key,
                snapshot_keys=[key_by_row[r] for r in btn.selection_snapshot_rows if r in key_by_row],
                eligibility=lambda k: True,
                parent=table,
                action="Configure parameters for",
            )
            if keys:
                configure_requested(list(keys))

        configure_button.clicked.connect(_on_configure_clicked)
        table.setCellWidget(row_index, 8, configure_button)
    table.resizeRowsToContents()
    table.blockSignals(previous)


def _fill_table(table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column, value in enumerate(values):
            set_workflow_table_item(table, row_index, column, value)
    table.resizeRowsToContents()

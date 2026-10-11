"""Quantum Corrections workflow surface."""

from __future__ import annotations

from collections.abc import Sequence
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

from mufinder.app.theme import status_brush, style_hint_text, style_section_subtitle, style_section_title
from mufinder.app.workflow_row_widget import SourceTableWidget
from mufinder.app.workflow_table_utils import (
    RowMenuAction,
    SelectionFeedback,
    create_workflow_table,
    set_workflow_table_item,
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


class QuantumCorrectionsWidget(SourceTableWidget):
    """Display eligible relaxed muon sources and completed phonon analyses.

    The source table and its targeting come from :class:`SourceTableWidget`;
    this widget adds the per-row stage and farm columns, the editable site
    label and the "Reset to Stage 1" action.
    """

    configure_requested = Signal(object)
    source_label_renamed = Signal(str, str)
    submit_local_requested = Signal(list)
    submit_cluster_requested = Signal(list)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_stage_requested = Signal(list)
    analysis_requested = Signal(str)

    _SOURCE_STATUS_COLUMN = 7
    _SOURCE_CONFIGURE_COLUMN = 8
    _SOURCE_WORKFLOW_LABEL = "Quantum Corrections"
    _CONFIGURE_SIGNAL = "configure_requested"
    _SUBMIT_LOCAL_SIGNAL = "submit_local_requested"
    _SUBMIT_CLUSTER_SIGNAL = "submit_cluster_requested"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Quantum Corrections workflow")
        self._source_rows: list[QuantumSourceRow]
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
            ("Site", "Charge", "dE (eV)", "Displacements", "Source Job", "Stage", "Farms", "Status", "Parameters"),
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
        self._selection_count_label = QLabel("")
        style_hint_text(self._selection_count_label)
        # SelectionFeedback sets the run buttons' text ("Run All ..." / "Run
        # Selected ...") from the selection.
        self._submit_local_button = QPushButton()
        self._submit_cluster_button = QPushButton()
        self._reset_stage_button = QPushButton("Reset to Stage 1")
        self._batch_configure_button.clicked.connect(self._emit_batch_configure)
        self._submit_local_button.clicked.connect(self._emit_submit_local)
        self._submit_cluster_button.clicked.connect(self._emit_submit_cluster)
        self._reset_stage_button.clicked.connect(self._emit_reset_stage)
        actions.addWidget(self._submit_local_button)
        actions.addWidget(self._submit_cluster_button)
        actions.addWidget(self._reset_stage_button)
        actions.addStretch(1)
        actions.addWidget(self._selection_count_label)
        actions.addWidget(self._batch_configure_button)
        sources_layout.addLayout(actions)

        # Rows here are derived from completed relaxations, so there is no
        # Remove, and the recovery action is "Reset to Stage 1" rather than the
        # shared failed-job reset.
        self._sources_table.setToolTip(
            "Right-click selected rows to run, cancel, inspect, configure, or reset them to stage 1."
        )
        self._bind_source_row_menu(
            extra_actions=(
                RowMenuAction(
                    text="Reset to Stage 1",
                    is_enabled=lambda rows: any(
                        self._source_rows[int(row)].resettable
                        for row in rows
                        if 0 <= int(row) < len(self._source_rows)
                    ),
                    trigger=lambda _rows: self._emit_reset_stage(),
                    on_rows=True,
                    separator_before=True,
                ),
            ),
        )
        self._sources_feedback = SelectionFeedback(
            self._sources_table,
            count_label=self._selection_count_label,
            run_local_button=self._submit_local_button,
            run_cluster_button=self._submit_cluster_button,
        )
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
        self._source_rows_valid = True
        self._source_label.setText(f"Eligible relaxed sites ({len(rows)})")
        self._updating_sources = True
        try:
            self._set_source_rows(rows)
        finally:
            self._updating_sources = False

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
        self._reapply_source_status_colors()
        for row_index, row in enumerate(self._result_rows):
            status_item = self._results_table.item(row_index, 4)
            if status_item is not None:
                status_item.setForeground(status_brush(row.status))

    def source_row_values(self, row: QuantumSourceRow) -> tuple[str, ...]:
        return (
            row.label,
            row.charge_state,
            row.relative_energy,
            row.displacements,
            row.source_job,
            row.stage,
            row.num_farms_display,
            row.status,
        )

    def create_source_row_cells(self, row_index: int, row: QuantumSourceRow) -> None:
        # The site label is renamed in place (``source_label_renamed``).
        label_item = self._sources_table.item(row_index, 0)
        if label_item is not None:
            label_item.setFlags(label_item.flags() | Qt.ItemFlag.ItemIsEditable)

    def refresh_source_actions(self, selected_rows: Sequence[QuantumSourceRow]) -> None:
        self._reset_stage_button.setEnabled(any(row.resettable for row in selected_rows))

    def _emit_reset_stage(self) -> None:
        self.reset_stage_requested.emit(self.selected_source_keys())

    def _on_source_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_sources or item.column() != 0:
            return
        row = item.row()
        if not (0 <= row < len(self._source_rows)):
            return
        key = self._source_rows[row].key
        self.source_label_renamed.emit(key, item.text())


def _fill_table(table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column, value in enumerate(values):
            set_workflow_table_item(table, row_index, column, value)
    table.resizeRowsToContents()

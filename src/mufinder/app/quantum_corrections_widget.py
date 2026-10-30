"""Quantum Corrections workflow surface."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.pinned_sites_scope import PinnedSitesScopeBar
from mufinder.app.theme import (
    status_brush,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
)
from mufinder.app.workflow_row_widget import RelaxedSiteSourceTableWidget
from mufinder.app.workflow_table_utils import (
    RESET_FAILED_ACTION_TEXT,
    SelectionFeedback,
    create_workflow_table,
    fill_workflow_table,
    name_row_control,
    set_header_tooltips,
)


# Header tooltips for the source table, keyed by header label so the mapping survives
# column changes.
_SOURCE_HEADER_TOOLTIPS: dict[str, str] = {
    "Site": "Pinned summary label, or the relaxed site's label (double-click to rename)",
    "Charge": "Charge state of the source relaxation: positive muon or muonium",
    "dE (eV)": (
        "Relaxed total energy relative to the lowest-energy eligible site with the same charge state and cell"
    ),
    "Displacements": (
        "3 \u00d7 atom count: the displaced calculations the finite-displacement phonon run needs, "
        "a guide to its farm count"
    ),
    "Source Job": "The muon relaxation the row was built from",
    "Status": "Status of the row's current job",
    "Parameters": "Configure the row; locked once its first job has been submitted",
}


@dataclass(frozen=True, slots=True)
class QuantumSourceRow:
    """One relaxed-site source displayed in the Quantum Corrections view."""

    key: str
    label: str
    charge_state: str
    relative_energy: str
    displacements: str
    source_job: str
    num_farms_display: str
    status: str
    configured: bool = False
    pinned_label: str = ""


@dataclass(frozen=True, slots=True)
class QuantumResultRow:
    """One completed quantum-corrections result row.

    Every field is display text built by the controller: ``zpe`` is the total
    muon ZPE in eV, ``muon_modes`` the three selected Γ-mode frequencies in
    cm⁻¹ in frequency order, ``widths`` the muon's zero-point width in Å along
    each of those modes in the same order, and ``witness`` the ``w1``
    mode-mixing witness.
    """

    key: str
    label: str
    zpe: str
    muon_modes: str
    widths: str
    witness: str
    status: str


#: Completed Results column headers and their tooltips, in column order.
RESULT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Job", ""),
    (
        "ZPE (eV)",
        "Total muon zero-point energy, ½ħω summed over the three muon-dominated Γ modes",
    ),
    (
        "Muon modes (cm⁻¹)",
        "Frequencies of the three Γ modes with the largest muon participation, in frequency order",
    ),
    ("Width (Å)", "Zero-point width √(ħ/2m_μω) of the muon along each muon mode"),
    (
        "w1",
        "Mode-mixing witness P − 3: 0 when the muon's motion is contained in its three modes, "
        "more negative as it mixes with host modes (down to −3)",
    ),
    ("Status", ""),
    ("Analysis", ""),
)
RESULT_STATUS_COLUMN = 5
RESULT_ANALYSIS_COLUMN = 6


class QuantumCorrectionsWidget(RelaxedSiteSourceTableWidget):
    """Display eligible relaxed muon sources and completed phonon analyses.

    The source table, its targeting, the editable site label, the pinned-sites
    scope and the "Reset Failed Jobs..." button come from
    :class:`RelaxedSiteSourceTableWidget`; this widget adds the farm column and
    the completed phonon results.
    """

    configure_requested = Signal(object)
    source_label_renamed = Signal(str, str)
    submit_local_requested = Signal(list)
    submit_cluster_requested = Signal(list)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    analysis_requested = Signal(str)
    show_every_site_toggled = Signal(bool)
    open_muon_sites_requested = Signal()

    _SOURCE_STATUS_COLUMN = 6
    _SOURCE_CONFIGURE_COLUMN = 7
    _SOURCE_WORKFLOW_LABEL = "Quantum Corrections"
    _CONFIGURE_SIGNAL = "configure_requested"
    _SUBMIT_LOCAL_SIGNAL = "submit_local_requested"
    _SUBMIT_CLUSTER_SIGNAL = "submit_cluster_requested"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Quantum Corrections workflow")
        self._source_rows: list[QuantumSourceRow]
        # Flush-path guard: cache the last-rendered results so set_results can
        # skip the table rebuild when nothing changed.
        self._result_rows: list[QuantumResultRow] = []
        self._result_rows_valid = False

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
            "How much zero-point energy each distinct relaxed site carries, how far the muon spreads, "
            "and whether its motion is entangled with the host."
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        header_row.addWidget(subtitle)

        summary = QLabel(
            "Gamma-point phonons of each relaxed site in the harmonic approximation, with the muon mass applied. "
            "Compare the ZPE, widths and w1 between sites computed with the same settings; see the Quantum "
            "Corrections theory page in the user guide for what the numbers mean."
        )
        summary.setWordWrap(True)
        style_hint_text(summary)
        header_row.addWidget(summary)
        layout.addLayout(header_row)

        sources_group = QGroupBox("Quantum Phonon Jobs")
        sources_layout = QVBoxLayout(sources_group)
        sources_layout.setContentsMargins(10, 8, 10, 8)
        sources_layout.setSpacing(6)

        self._source_label = QLabel("Pinned muon sites")
        style_section_title(self._source_label)
        sources_layout.addWidget(self._source_label)
        source_hint = QLabel(
            "Each row starts from a completed muon relaxation and inherits its electronic settings. The "
            "electronic settings and farm count apply to the row's finite-displacement phonon job, and "
            "every setting is locked once the row's job has been submitted."
        )
        source_hint.setWordWrap(True)
        style_hint_text(source_hint)
        sources_layout.addWidget(source_hint)
        self._site_scope = PinnedSitesScopeBar(sources_group)
        self._site_scope.show_every_site_toggled.connect(self.show_every_site_toggled)
        self._site_scope.open_muon_sites_requested.connect(self.open_muon_sites_requested)
        sources_layout.addWidget(self._site_scope)
        self._sources_table = create_workflow_table(
            (
                "Site",
                "Charge",
                "dE (eV)",
                "Displacements",
                "Source Job",
                "Farms",
                "Status",
                "Parameters",
            ),
            resize_to_contents_columns=(7,),
            accessible_name="Quantum Phonon Jobs",
        )
        set_header_tooltips(self._sources_table, _SOURCE_HEADER_TOOLTIPS)
        self._sources_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._sources_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
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
        self._reset_failed_button = QPushButton(RESET_FAILED_ACTION_TEXT)
        self._reset_failed_button.setToolTip(
            "Return the selected failed, stopped or cancelled phonon jobs to 'defined' for a rerun."
        )
        self._batch_configure_button.clicked.connect(self._emit_batch_configure)
        self._submit_local_button.clicked.connect(self._emit_submit_local)
        self._submit_cluster_button.clicked.connect(self._emit_submit_cluster)
        self._reset_failed_button.clicked.connect(self._emit_reset_failed_selection)
        actions.addWidget(self._submit_local_button)
        actions.addWidget(self._submit_cluster_button)
        actions.addWidget(self._reset_failed_button)
        actions.addStretch(1)
        actions.addWidget(self._selection_count_label)
        actions.addWidget(self._batch_configure_button)
        sources_layout.addLayout(actions)

        # Rows here are derived from completed relaxations, so there is no
        # Remove; a failed row recovers through the shared failed-job reset.
        self._sources_table.setToolTip(
            "Right-click selected rows to run, cancel, inspect, reset, or configure them."
        )
        self._bind_source_row_menu(emit_reset=self.reset_failed_requested.emit)
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
            "One row per completed phonon run: the total muon ZPE, the muon modes and widths, and the "
            "mode-mixing witness w1. Launch Analysis opens the modes and atom-pair coupling."
        )
        result_hint.setWordWrap(True)
        style_hint_text(result_hint)
        results_layout.addWidget(result_hint)
        self._results_table = create_workflow_table(
            tuple(title for title, _tip in RESULT_COLUMNS),
            stretch_last_section=False,
            accessible_name="Completed Results",
        )
        results_header = self._results_table.horizontalHeader()
        for column, (_title, tooltip) in enumerate(RESULT_COLUMNS):
            mode = (
                QHeaderView.ResizeMode.Stretch
                if column == 0
                else QHeaderView.ResizeMode.ResizeToContents
            )
            results_header.setSectionResizeMode(column, mode)
            header_item = self._results_table.horizontalHeaderItem(column)
            if tooltip and header_item is not None:
                header_item.setToolTip(tooltip)
        results_layout.addWidget(self._results_table)
        layout.addWidget(results_group, 1)

        layout.addStretch(1)

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
        fill_workflow_table(
            self._results_table,
            [
                (row.label, row.zpe, row.muon_modes, row.widths, row.witness, row.status)
                for row in rows
            ],
        )
        for row_index, row in enumerate(rows):
            # Colour the Status cell with the project-standard per-status palette, matching the
            # eligible-sources table and the relaxation/convergence tables.
            status_item = self._results_table.item(row_index, RESULT_STATUS_COLUMN)
            if status_item is not None:
                status_item.setForeground(status_brush(row.status))
            button = QPushButton("Launch Analysis")
            label_item = self._results_table.item(row_index, 0)
            name_row_control(
                button, "Launch Analysis for", "" if label_item is None else label_item.text()
            )
            button.clicked.connect(
                lambda _checked=False, key=row.key: self.analysis_requested.emit(key)
            )
            self._results_table.setCellWidget(row_index, RESULT_ANALYSIS_COLUMN, button)
        if self._results_table.rowCount() > 0:
            for column in range(1, len(RESULT_COLUMNS)):
                self._results_table.resizeColumnToContents(column)

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
            status_item = self._results_table.item(row_index, RESULT_STATUS_COLUMN)
            if status_item is not None:
                status_item.setForeground(status_brush(row.status))

    def source_row_values(self, row: QuantumSourceRow) -> tuple[str, ...]:
        return (
            row.pinned_label or row.label,
            row.charge_state,
            row.relative_energy,
            row.displacements,
            row.source_job,
            row.num_farms_display,
            row.status,
        )

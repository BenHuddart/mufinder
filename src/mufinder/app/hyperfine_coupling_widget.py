"""Hyperfine Coupling workflow surface."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEvent, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.pinned_sites_scope import PinnedSitesScopeBar
from mufinder.app.theme import style_hint_text, style_section_subtitle, style_section_title
from mufinder.app.workflow_row_widget import RelaxedSiteSourceTableWidget
from mufinder.app.workflow_table_utils import (
    RESET_FAILED_ACTION_TEXT,
    SelectionFeedback,
    create_workflow_table,
    fill_workflow_table,
    selected_table_rows,
    set_header_tooltips,
)

# Header tooltips for the source table, keyed by header label.
_SOURCE_HEADER_TOOLTIPS: dict[str, str] = {
    "Site": "Pinned summary label, or the relaxed site's label (double-click to rename)",
    "Charge": "Charge state of the source relaxation: positive muon or muonium",
    "Source Job": "The muon relaxation the row was built from",
    "Status": "Status of the row's magres job",
    "Parameters": "Configure the row; locked once its job has been submitted",
}

USE_IN_MAGNETIC_FIELDS_TEXT = "Use in Magnetic Fields"


@dataclass(frozen=True, slots=True)
class HyperfineSourceRow:
    """One relaxed-site source displayed in the Hyperfine Coupling view."""

    key: str
    label: str
    charge_state: str
    source_job: str
    status: str
    configured: bool = False
    pinned_label: str = ""


@dataclass(frozen=True, slots=True)
class HyperfineResultRow:
    """One completed hyperfine result, as display text built by the controller.

    ``handoff_reason`` is empty when the result can be handed to Magnetic
    Fields and otherwise says why not.
    """

    key: str
    site: str
    charge_state: str
    isotropic: str
    contact_field: str
    pseudo_contact_field: str
    ratio: str
    dipolar: str
    handoff_reason: str = ""


#: Completed Results column headers and their tooltips, in column order.
RESULT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Site", "The relaxed site the magres job ran on"),
    ("Charge state", "Charge state of the source relaxation"),
    (
        "A_iso (MHz)",
        "Isotropic (Fermi-contact) muon hyperfine coupling from CASTEP's PAW-reconstructed "
        "spin density, in the physical sign convention",
    ),
    ("B_c (T)", "Contact field at the muon, A_iso / (g_e γ_μ/2π)"),
    (
        "B_c pseudo (T)",
        "Contact field from the pseudo spin density at the muon, without the core "
        "reconstruction; empty when the formatted density could not be read",
    ),
    ("Reconstructed / pseudo", "B_c divided by B_c pseudo"),
    (
        "Dipolar (MHz)",
        "Principal values of the anisotropic (dipolar) part of the tensor, ascending",
    ),
)


class HyperfineCouplingWidget(RelaxedSiteSourceTableWidget):
    """Display eligible relaxed muon sources and their completed hyperfine couplings.

    The source table comes from :class:`RelaxedSiteSourceTableWidget`. The
    results table lists one row per completed magres job; "Use in Magnetic
    Fields" asks for the selected result's coupling to be handed over.
    """

    configure_requested = Signal(object)
    source_label_renamed = Signal(str, str)
    submit_local_requested = Signal(list)
    submit_cluster_requested = Signal(list)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    show_every_site_toggled = Signal(bool)
    open_muon_sites_requested = Signal()
    use_in_magnetic_fields_requested = Signal(str)

    _SOURCE_STATUS_COLUMN = 3
    _SOURCE_CONFIGURE_COLUMN = 4
    _SOURCE_WORKFLOW_LABEL = "Hyperfine Coupling"
    _CONFIGURE_SIGNAL = "configure_requested"
    _SUBMIT_LOCAL_SIGNAL = "submit_local_requested"
    _SUBMIT_CLUSTER_SIGNAL = "submit_cluster_requested"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Hyperfine Coupling workflow")
        self._source_rows: list[HyperfineSourceRow]
        self._result_rows: list[HyperfineResultRow] = []
        self._result_rows_valid = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel("Hyperfine Coupling")
        style_section_title(title)
        layout.addWidget(title)
        subtitle = QLabel(
            "The contact hyperfine field of each distinct relaxed site, and the hyperfine "
            "tensor of a muonium state."
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(subtitle)
        summary = QLabel(
            "Each site runs as one spin-polarised CASTEP magres job. CASTEP reconstructs the "
            "spin density at the muon from its projectors; the pseudo spin density from the "
            "same job is shown beside it as a check. See the Hyperfine Coupling theory page in "
            "the user guide."
        )
        summary.setWordWrap(True)
        style_hint_text(summary)
        layout.addWidget(summary)

        sources_group = QGroupBox("Hyperfine Jobs")
        sources_layout = QVBoxLayout(sources_group)
        sources_layout.setContentsMargins(10, 8, 10, 8)
        sources_layout.setSpacing(6)
        self._source_label = QLabel("Pinned muon sites")
        style_section_title(self._source_label)
        sources_layout.addWidget(self._source_label)
        source_hint = QLabel(
            "Each row starts from a completed muon relaxation and inherits its electronic "
            "settings. The job is always spin polarised and writes its formatted density; "
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
            ("Site", "Charge", "Source Job", "Status", "Parameters"),
            resize_to_contents_columns=(4,),
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
        self._submit_local_button = QPushButton()
        self._submit_cluster_button = QPushButton()
        self._reset_failed_button = QPushButton(RESET_FAILED_ACTION_TEXT)
        self._reset_failed_button.setToolTip(
            "Return the selected failed, stopped or cancelled magres jobs to 'defined' for a rerun."
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

        # Rows come from completed relaxations, so there is no Remove; a failed
        # row recovers through the shared failed-job reset.
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
        self._result_label = QLabel("Completed hyperfine calculations")
        style_section_title(self._result_label)
        results_layout.addWidget(self._result_label)
        result_hint = QLabel(
            "One row per completed magres job. Select a row and use Use in Magnetic Fields to "
            "fill the Hyperfine coupling of every Magnetic Fields form that shows the same site."
        )
        result_hint.setWordWrap(True)
        style_hint_text(result_hint)
        results_layout.addWidget(result_hint)
        self._results_table = create_workflow_table(
            tuple(title for title, _tip in RESULT_COLUMNS), stretch_last_section=False
        )
        self._results_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        results_header = self._results_table.horizontalHeader()
        for column, (_title, tooltip) in enumerate(RESULT_COLUMNS):
            mode = (
                QHeaderView.ResizeMode.Stretch
                if column == 0
                else QHeaderView.ResizeMode.ResizeToContents
            )
            results_header.setSectionResizeMode(column, mode)
            header_item = self._results_table.horizontalHeaderItem(column)
            if header_item is not None:
                header_item.setToolTip(tooltip)
        self._results_table.itemSelectionChanged.connect(self._update_handoff_button)
        results_layout.addWidget(self._results_table)
        result_actions = QHBoxLayout()
        result_actions.addStretch(1)
        self._use_in_magnetic_fields_button = QPushButton(USE_IN_MAGNETIC_FIELDS_TEXT)
        self._use_in_magnetic_fields_button.clicked.connect(self._emit_use_in_magnetic_fields)
        result_actions.addWidget(self._use_in_magnetic_fields_button)
        results_layout.addLayout(result_actions)
        layout.addWidget(results_group, 1)
        layout.addStretch(1)
        self._update_handoff_button()

    def set_results(self, rows: list[HyperfineResultRow]) -> None:
        """Replace the displayed result rows, skipping the rebuild when unchanged."""

        if self._result_rows_valid and list(rows) == self._result_rows:
            return
        self._result_rows = list(rows)
        self._result_rows_valid = True
        self._result_label.setText(f"Completed hyperfine calculations ({len(rows)})")
        fill_workflow_table(
            self._results_table,
            [
                (
                    row.site,
                    row.charge_state,
                    row.isotropic,
                    row.contact_field,
                    row.pseudo_contact_field,
                    row.ratio,
                    row.dipolar,
                )
                for row in rows
            ],
        )
        if self._results_table.rowCount() > 0:
            for column in range(1, len(RESULT_COLUMNS)):
                self._results_table.resizeColumnToContents(column)
        self._update_handoff_button()

    def source_row_values(self, row: HyperfineSourceRow) -> tuple[str, ...]:
        return (row.pinned_label or row.label, row.charge_state, row.source_job, row.status)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt API naming.
        """Re-apply the palette-derived status colours on a theme change."""

        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        }:
            self._reapply_source_status_colors()

    def _selected_result_row(self) -> HyperfineResultRow | None:
        rows = selected_table_rows(self._results_table)
        if len(rows) != 1 or not (0 <= rows[0] < len(self._result_rows)):
            return None
        return self._result_rows[rows[0]]

    def _update_handoff_button(self) -> None:
        row = self._selected_result_row()
        if row is None:
            reason = "Select one completed result."
        else:
            reason = row.handoff_reason
        self._use_in_magnetic_fields_button.setEnabled(not reason)
        self._use_in_magnetic_fields_button.setToolTip(
            reason
            or "Fill the Hyperfine coupling of every Magnetic Fields form showing this site, "
            "using that form's contact shell."
        )

    def _emit_use_in_magnetic_fields(self) -> None:
        row = self._selected_result_row()
        if row is not None and not row.handoff_reason:
            self.use_in_magnetic_fields_requested.emit(row.key)

"""Quantum Corrections workflow surface: three numbered steps above one table of sites."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QEvent, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.pinned_sites_scope import PinnedSitesScopeBar
from mufinder.app.theme import form_spacing, page_margins, style_section_title
from mufinder.app.workflow_row_widget import RelaxedSiteSourceTableWidget
from mufinder.app.workflow_steps import WorkflowSteps
from mufinder.app.workflow_submission import READY_WORKFLOW_ROW_STATUSES
from mufinder.app.workflow_table_utils import (
    RESET_FAILED_ACTION_TEXT,
    SelectionFeedback,
    create_workflow_table,
    set_header_tooltips,
)

#: What a cell shows for a site without the number.
NO_VALUE_TEXT = "–"

#: Follows a number whose result was stored with a warning; the row tooltip says which.
WARNING_MARK = "⚠"

#: The table's column headers and one-line tooltips, in column order.
SITE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("Site", "Pinned summary label, or the relaxed site's label (double-click to rename)"),
    ("Charge", "Charge state of the source relaxation: positive muon or muonium"),
    (
        "dE (eV)",
        "Relaxed total energy relative to the lowest eligible site with the same charge state "
        "and cell",
    ),
    ("ZPE (eV)", "Muon zero-point energy, ½ħω summed over the three muon-dominated Γ modes"),
    (
        "dE + ZPE (eV)",
        "Relaxed energy plus muon ZPE, relative to the lowest such sum among sites with a "
        "result and no non-positive muon mode, same charge state and cell",
    ),
    ("Widths (Å)", "Zero-point width of the muon along each of its three modes"),
    ("w1", "Mode-mixing witness: 0 when the muon's motion stays in its three modes, down to −3"),
    ("Farms", "Task farms the phonon job runs with"),
    ("Status", "Status of the row's current job"),
)


@dataclass(frozen=True, slots=True)
class QuantumSourceRow:
    """One site in the Quantum Corrections table: its job and, once complete, its result.

    Every field is display text built by the controller. ``zpe``,
    ``corrected_energy``, ``widths`` and ``witness`` are the result's columns;
    ``muon_modes`` lists the three selected Γ-mode frequencies for the row tooltip;
    ``result_id`` names the result ``Open Analysis`` opens, empty without one;
    ``warnings`` are those stored with that result. ``zpe_mark`` says when the ZPE may
    not belong to the site's current geometry (``earlier`` or ``unconfirmed``), and
    ``notes`` are further lines for the row tooltip.

    An empty ``key`` marks a result with no eligible site: a read-only row
    that no job action reaches and only ``Open Analysis`` acts on.
    """

    key: str
    label: str
    charge_state: str
    relative_energy: str
    displacements: str
    source_job: str
    num_farms_display: str
    status: str
    pinned_label: str = ""
    zpe: str = NO_VALUE_TEXT
    corrected_energy: str = NO_VALUE_TEXT
    widths: str = NO_VALUE_TEXT
    witness: str = NO_VALUE_TEXT
    muon_modes: str = ""
    result_id: str = ""
    warnings: tuple[str, ...] = ()
    zpe_mark: str = ""
    notes: tuple[str, ...] = ()


class QuantumCorrectionsWidget(RelaxedSiteSourceTableWidget):
    """The flow as three steps (Sites, Run, Results) above one table of sites.

    The table, its targeting, the editable site label and the row menu come from
    :class:`RelaxedSiteSourceTableWidget`; this widget adds the steps and the
    result columns.
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
    open_energy_barriers_requested = Signal()

    _SOURCE_STATUS_COLUMN = 8
    _SOURCE_CONFIGURE_COLUMN = None
    _SOURCE_WORKFLOW_LABEL = "Quantum Corrections"
    _CONFIGURE_SIGNAL = "configure_requested"
    _SUBMIT_LOCAL_SIGNAL = "submit_local_requested"
    _SUBMIT_CLUSTER_SIGNAL = "submit_cluster_requested"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Quantum Corrections workflow")
        self._source_rows: list[QuantumSourceRow]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(page_margins())
        layout.setSpacing(form_spacing())

        title = QLabel("Quantum Corrections")
        style_section_title(title)
        layout.addWidget(title)

        steps = WorkflowSteps(self)

        sites_step = steps.add_step("Sites")
        self._source_label = sites_step.status
        self._site_scope = PinnedSitesScopeBar(self, link_always_shown=True)
        self._site_scope.show_every_site_toggled.connect(self.show_every_site_toggled)
        self._site_scope.open_muon_sites_requested.connect(self.open_muon_sites_requested)
        sites_step.actions.addWidget(self._site_scope, 1)

        run_step = steps.add_step("Run")
        self._run_status_label = run_step.status
        self._batch_configure_button = QPushButton("Configure…")
        self._batch_configure_button.setToolTip(
            "Configure the selected sites, or with nothing selected every site not yet run."
        )
        # SelectionFeedback sets the run buttons' text ("Run All ..." / "Run
        # Selected ...") from the selection.
        self._submit_local_button = QPushButton()
        self._submit_cluster_button = QPushButton()
        self._reset_failed_button = QPushButton(RESET_FAILED_ACTION_TEXT)
        self._reset_failed_button.setToolTip(
            "Return the selected failed, stopped or cancelled phonon jobs to 'defined' for a rerun."
        )
        self._batch_configure_button.clicked.connect(self._emit_configure_targets)
        self._submit_local_button.clicked.connect(self._emit_submit_local)
        self._submit_cluster_button.clicked.connect(self._emit_submit_cluster)
        self._reset_failed_button.clicked.connect(self._emit_reset_failed_selection)
        run_step.actions.addWidget(self._batch_configure_button)
        run_step.actions.addWidget(self._submit_local_button)
        run_step.actions.addWidget(self._submit_cluster_button)
        run_step.actions.addWidget(self._reset_failed_button)
        run_step.actions.addStretch(1)

        results_step = steps.add_step("Results")
        self._results_status_label = results_step.status
        self._open_analysis_button = QPushButton("Open Analysis")
        self._open_analysis_button.setToolTip("Open the modes and coupling of the selected site.")
        self._open_analysis_button.clicked.connect(self._emit_open_analysis)
        self._next_button = QPushButton("Next: Energy Barriers →")
        self._next_button.setObjectName("overviewActionButton")
        self._next_button.clicked.connect(self.open_energy_barriers_requested)
        results_step.actions.addWidget(self._open_analysis_button)
        results_step.actions.addStretch(1)
        results_step.actions.addWidget(self._next_button)
        layout.addWidget(steps)

        self._sources_table = create_workflow_table(
            tuple(title for title, _tip in SITE_COLUMNS),
            default_resize_mode=QHeaderView.ResizeMode.ResizeToContents,
            stretch_last_section=False,
            accessible_name="Quantum Corrections Sites",
        )
        self._sources_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        set_header_tooltips(self._sources_table, dict(SITE_COLUMNS))
        self._sources_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._sources_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._sources_table.itemSelectionChanged.connect(self._update_actions)
        self._sources_table.itemChanged.connect(self._on_source_item_changed)
        # Rows here are derived from completed relaxations, so there is no
        # Remove; a failed row recovers through the shared failed-job reset.
        self._bind_source_row_menu(emit_reset=self.reset_failed_requested.emit)
        self._sources_feedback = SelectionFeedback(
            self._sources_table,
            run_local_button=self._submit_local_button,
            run_cluster_button=self._submit_cluster_button,
        )
        layout.addWidget(self._sources_table, 1)

        self._update_source_label(0)
        self._update_actions()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt API naming.
        """Re-apply palette-derived status colours on a theme change.

        ``set_sources`` skips rebuilds when its rows are unchanged, so the
        status-cell foreground brushes would otherwise stay frozen at the previous
        theme's colours after a palette flip.
        """

        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
        }:
            self._reapply_source_status_colors()

    def source_row_values(self, row: QuantumSourceRow) -> tuple[str, ...]:
        return (
            row.pinned_label or row.label,
            row.charge_state,
            row.relative_energy,
            f"{_marked(row.zpe, row)} ({row.zpe_mark})" if row.zpe_mark else _marked(row.zpe, row),
            _marked(row.corrected_energy, row),
            row.widths,
            row.witness,
            row.num_farms_display,
            row.status,
        )

    def create_source_row_cells(self, row_index: int, row: QuantumSourceRow) -> None:
        if row.key:
            super().create_source_row_cells(row_index, row)
        tooltip = _row_tooltip(row)
        for column in range(self._sources_table.columnCount()):
            item = self._sources_table.item(row_index, column)
            if item is not None:
                item.setToolTip(tooltip)

    def refresh_source_actions(self, selected_rows: Sequence[QuantumSourceRow]) -> None:
        super().refresh_source_actions(selected_rows)
        sites = self._site_rows()
        self._batch_configure_button.setEnabled(
            bool(sites) and (not selected_rows or any(row.key for row in selected_rows))
        )
        completed = sum(row.status == "completed" for row in sites)
        self._run_status_label.setText(
            f"{completed} of {len(sites)} completed" if sites else "No sites to run"
        )

        has_results = any(row.result_id for row in self._source_rows)
        chosen = selected_rows[0] if len(selected_rows) == 1 else None
        self._open_analysis_button.setEnabled(bool(chosen and chosen.result_id))
        self._next_button.setVisible(has_results)
        if not has_results:
            text = "No results yet"
        elif chosen is None:
            text = "Select a site"
        elif chosen.result_id:
            text = "Site selected"
        else:
            text = "No result for this site yet"
        self._results_status_label.setText(text)

    def selected_source_keys(self) -> list[str]:
        return [row.key for row in self._selected_source_rows() if row.key]

    def _selected_or_all_source_keys(self) -> list[str]:
        """Return the sites the run buttons act on: those selected, else every site."""

        return [row.key for row in self._selected_source_rows() or self._source_rows if row.key]

    def _site_rows(self) -> list[QuantumSourceRow]:
        return [row for row in self._source_rows if row.key]

    def _update_source_label(self, _count: int) -> None:
        count = len(self._site_rows())
        if self._show_every_site:
            self._source_label.setText(f"{count} relaxed site{'' if count == 1 else 's'}")
            return
        pinned = sum(bool(row.pinned_label) for row in self._source_rows)
        text = f"{pinned} pinned from Muon Sites"
        if count > pinned:
            text += f", {count - pinned} more already set up"
        self._source_label.setText(text)

    def _emit_configure_targets(self) -> None:
        """Emit the selected sites, else those not yet run, else every site."""

        if self._selected_source_rows():
            keys = self.selected_source_keys()
        else:
            keys = [
                row.key for row in self._site_rows() if row.status in READY_WORKFLOW_ROW_STATUSES
            ] or [row.key for row in self._site_rows()]
        if keys:
            self.configure_requested.emit(keys)

    def _emit_open_analysis(self) -> None:
        selected = self._selected_source_rows()
        if len(selected) == 1 and selected[0].result_id:
            self.analysis_requested.emit(selected[0].result_id)


def _marked(text: str, row: QuantumSourceRow) -> str:
    """Return a number's cell text, with the mark when its result carries a warning."""

    return f"{text} {WARNING_MARK}" if row.warnings and text != NO_VALUE_TEXT else text


def _row_tooltip(row: QuantumSourceRow) -> str:
    """Return what a row's cells show on hover: what the table has no column for."""

    lines = [f"{WARNING_MARK} {warning}" for warning in row.warnings]
    lines.extend(row.notes)
    if not row.key:
        lines.append("No eligible site stands behind this result, so it is kept but not ranked.")
    if row.pinned_label:
        lines.append(f"Relaxed site: {row.label}")
    if row.displacements:
        lines.append(f"Displacements: {row.displacements}")
    if row.source_job:
        lines.append(f"Source job: {row.source_job}")
    if row.muon_modes:
        lines.append(f"Muon modes (cm⁻¹): {row.muon_modes}")
    return "\n".join(lines)

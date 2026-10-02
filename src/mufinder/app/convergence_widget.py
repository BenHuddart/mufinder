"""Cutoff-convergence configuration panel with workflow-aware CASTEP settings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.convergence_types import ConvergenceConfig, SweepPointRunState
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.project_defaults_widget import ProjectDefaultsStrip
from mufinder.app.workflow_parameters import (
    kpoint_exclusive_hint,
    reorder_workflow_parameter_specs,
    seed_xc_functional_default,
)
from mufinder.app.workflow_submission import READY_WORKFLOW_ROW_STATUSES
from mufinder.app import workflow_table_utils
from mufinder.app.workflow_table_utils import (
    confirm_row_removal,
    count_phrase,
    RESETTABLE_ROW_STATUSES,
    add_reset_failed_action,
    bind_delete_shortcuts,
    create_workflow_table,
    style_workflow_configure_button,
    workflow_row_status_text,
)
from mufinder.app.theme import (
    status_brush,
    style_section_subtitle,
    style_section_title,
)
from mufinder.castep_io.gui_metadata import (
    get_gui_keyword_specs_for_files,
    infer_task,
    keyword_locked_in_workflow,
    keyword_visible_in_workflow,
    normalize_gui_value,
    validate_gui_values,
)
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.app.selection_snapshot import (
    SelectionSnapshotPushButton,
    SelectionSnapshotToolButton,
    resolve_target_keys,
)

_STRUCTURE_CONTROLLED_CELL_KEYWORDS = {
    "lattice_cart",
    "lattice_abc",
    "positions_frac",
    "positions_abs",
}
_ACTIVE_JOB_STATUSES = {"submitting", "queued", "running", "retrieving"}
_PROGRESSED_JOB_STATUSES = _ACTIVE_JOB_STATUSES | {"stopped", "completed", "failed", "cancelled"}
# Submit-enablement gate for the row context menu. It shares the single source
# of truth used by the submit predicate (workflow_row_state_is_ready) so the
# context-menu Submit action agrees with the always-enabled Run buttons, which
# already submit "not generated" rows through the same materialize path.
_SUBMITTABLE_JOB_STATUSES = set(READY_WORKFLOW_ROW_STATUSES)
# Shared with every other workflow table's row reset action; see
# ``workflow_table_utils.RESETTABLE_ROW_STATUSES``.
_RESETTABLE_JOB_STATUSES = set(RESETTABLE_ROW_STATUSES)
_LOCKED_CUTOFF_EDIT_STATUSES = _PROGRESSED_JOB_STATUSES - {"cancelled"}


def _reorder_specs_with_kpoint_anchor(specs: list[KeywordSpec]) -> list[KeywordSpec]:
    return reorder_workflow_parameter_specs(specs)


class CutoffRangeDialog(QDialog):
    """Compact dialog for generating multiple cutoff rows from a numeric range."""

    def __init__(self, defaults: Mapping[str, object] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate Cutoffs From Range")

        defaults = defaults or {}
        start = float(defaults.get("start", 300.0))
        stop = float(defaults.get("stop", 1400.0))
        step = float(defaults.get("step", 100.0))
        replace_existing = bool(defaults.get("replace_existing", False))

        self._start = DecimalLineEdit(self)
        self._start.setRange(1.0, 10_000.0)
        self._start.setValue(start)

        self._stop = DecimalLineEdit(self)
        self._stop.setRange(1.0, 10_000.0)
        self._stop.setValue(stop)

        self._step = DecimalLineEdit(self)
        self._step.setRange(1.0, 5000.0)
        self._step.setValue(step)

        self._replace_existing = QCheckBox("Replace existing cutoff list", self)
        self._replace_existing.setChecked(replace_existing)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Start (eV)", self._start)
        form.addRow("Stop (eV)", self._stop)
        form.addRow("Step (eV)", self._step)
        layout.addLayout(form)
        layout.addWidget(self._replace_existing)
        layout.addWidget(self._buttons)

        self._buttons.accepted.connect(self._accept_if_valid)
        self._buttons.rejected.connect(self.reject)

    def values(self) -> tuple[float, ...]:
        """Return the generated cutoff list implied by the current controls."""
        start = self._start.value()
        stop = self._stop.value()
        step = self._step.value()

        values: list[float] = []
        current = start
        guard = 0
        while current <= stop + (step * 1e-6):
            values.append(round(current, 8))
            current += step
            guard += 1
            if guard > 5000:
                raise ValueError("Generated range exceeded 5000 rows.")
        return tuple(values)

    def replace_existing(self) -> bool:
        """Return whether generated cutoffs should replace the current list."""
        return self._replace_existing.isChecked()

    def defaults_state(self) -> dict[str, object]:
        """Persistable dialog state for later reopenings."""
        return {
            "start": self._start.value(),
            "stop": self._stop.value(),
            "step": self._step.value(),
            "replace_existing": self._replace_existing.isChecked(),
        }

    def _accept_if_valid(self) -> None:
        if self._step.value() <= 0:
            QMessageBox.warning(self, "Invalid range", "Step must be positive.")
            return
        if self._stop.value() < self._start.value():
            QMessageBox.warning(self, "Invalid range", "Stop must be greater than or equal to start.")
            return
        try:
            self.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Range too large", str(exc))
            return
        self.accept()


class CutoffConvergenceWidget(QWidget):
    """UI for configuring cutoff convergence sweeps and per-cutoff parameters."""

    _CUTOFF_COLUMN = 0
    _STATUS_COLUMN = 1
    _TOTAL_ENERGY_COLUMN = 2
    _MAX_FORCE_COLUMN = 3
    _CONFIGURE_COLUMN = 4
    _REMOVE_COLUMN = 5
    _ROW_HEIGHT = 32
    _CUTOFF_COLUMN_WIDTH = 132
    _CONFIGURE_COLUMN_MIN_WIDTH = 136
    _REMOVE_COLUMN_WIDTH = 34

    configure_requested = Signal(object)
    submit_local_requested = Signal(object)
    submit_cluster_requested = Signal(object)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    remove_cutoff_requested = Signal(object)
    examine_requested = Signal()
    form_state_changed = Signal(dict)
    simulation_cell_create_requested = Signal()
    simulation_cell_selection_requested = Signal(object)
    simulation_cell_use_unit_cell_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._add_cutoff_button = QPushButton("Add Cutoff")
        self._generate_range_button = QPushButton("Generate From Range...")
        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._batch_configure_button.setEnabled(False)
        self._examine_button = QPushButton("Examine Cutoff Convergence")
        self._examine_button.setEnabled(False)
        self.project_defaults_strip = ProjectDefaultsStrip()
        self._range_state: dict[str, object] = {
            "start": 300.0,
            "stop": 1400.0,
            "step": 100.0,
            "replace_existing": False,
        }
        self._parameter_species_options: tuple[str, ...] = ()

        self._cutoff_table = create_workflow_table(
            (
                "Cutoff Energy (eV)",
                "Status",
                "Total Energy (eV)",
                "Max |Force| (eV/Ang)",
                "Parameters",
                "",
            )
        )
        # Per-widget settings the shared helper does not cover (or that this
        # editable table overrides): extended multi-select, in-place editing,
        # a fixed row height, a custom context menu, per-column resize modes,
        # and a non-stretching last (Remove) column.
        self._cutoff_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._cutoff_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._cutoff_table.verticalHeader().setDefaultSectionSize(self._ROW_HEIGHT)
        self._cutoff_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header = self._cutoff_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(self._CUTOFF_COLUMN, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(self._STATUS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._TOTAL_ENERGY_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._MAX_FORCE_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._CONFIGURE_COLUMN, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(self._REMOVE_COLUMN, QHeaderView.ResizeMode.Fixed)
        self._cutoff_table.setColumnWidth(self._CUTOFF_COLUMN, self._CUTOFF_COLUMN_WIDTH)
        self._cutoff_table.setColumnWidth(self._CONFIGURE_COLUMN, self._CONFIGURE_COLUMN_MIN_WIDTH)
        self._cutoff_table.setColumnWidth(self._REMOVE_COLUMN, self._REMOVE_COLUMN_WIDTH)

        self._cutoff_runtime_states: dict[float, SweepPointRunState] = {}
        self._cutoff_parameter_values: dict[float, dict[str, Any]] = {}
        self._cutoff_added_advanced: dict[float, set[str]] = {}
        self._restoring_form_state = False
        self._updating_simulation_cell = False

        self._name_prefix = QLineEdit("cutoff_convergence")
        self._simulation_cell_summary = QLabel("Using single unit cell.")
        self._simulation_cell_summary.setWordWrap(True)
        self._simulation_cell_combo = QComboBox()
        self._simulation_cell_create_button = QPushButton("Manage Supercells...")
        self._simulation_cell_use_unit_button = QPushButton("Use Unit Cell")

        specs = get_gui_keyword_specs_for_files(
            files=("param", "cell"),
            workflow_stage="convergence_cutoff",
            include_advanced=True,
        )
        self._specs = [spec for spec in specs if spec.name not in {"task", "spin_polarised"}]
        self._specs = _reorder_specs_with_kpoint_anchor(self._specs)
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        self._default_params = self._default_shared_params()
        self._schema_xc_functional = self._default_params.get("xc_functional")

        self._build_layout()
        self._connect_signals()
        self._delete_shortcut, self._backspace_shortcut = bind_delete_shortcuts(
            self, self._remove_selected_or_current_cutoffs
        )

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)
        title = QLabel("Cutoff Convergence")
        style_section_title(title)
        title_column.addWidget(title)

        subtitle = QLabel("Check whether cutoff settings are accurate enough before production runs.")
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        title_column.addWidget(subtitle)

        header_row.addLayout(title_column, 1)
        header_row.addWidget(self.project_defaults_strip, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        simulation_group = QGroupBox("Simulation Cell")
        simulation_layout = QVBoxLayout(simulation_group)
        simulation_layout.setContentsMargins(10, 8, 10, 8)
        simulation_layout.setSpacing(6)
        simulation_layout.addWidget(self._simulation_cell_summary)

        simulation_row = QHBoxLayout()
        simulation_row.setSpacing(8)
        simulation_row.addWidget(QLabel("Saved supercells"))
        simulation_row.addWidget(self._simulation_cell_combo, 1)
        simulation_row.addWidget(self._simulation_cell_create_button)
        simulation_row.addWidget(self._simulation_cell_use_unit_button)
        simulation_layout.addLayout(simulation_row)

        layout.addWidget(simulation_group)

        sweep_group = QGroupBox("Cutoff Sweep")
        sweep_layout = QVBoxLayout(sweep_group)
        sweep_layout.setContentsMargins(10, 8, 10, 8)
        sweep_layout.setSpacing(6)
        sweep_layout.addWidget(self._cutoff_table)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(6)
        controls_left = QHBoxLayout()
        controls_left.setSpacing(6)
        controls_left.addWidget(self._add_cutoff_button)
        controls_left.addWidget(self._generate_range_button)
        controls_row.addLayout(controls_left)
        controls_row.addStretch(1)
        controls_row.addWidget(self._batch_configure_button)
        sweep_layout.addLayout(controls_row)

        examine_row = QHBoxLayout()
        examine_row.addStretch(1)
        examine_row.addWidget(self._examine_button)
        sweep_layout.addLayout(examine_row)

        layout.addWidget(sweep_group, 1)

        run_group = QGroupBox("Run Jobs")
        run_layout = QVBoxLayout(run_group)
        run_layout.setContentsMargins(10, 8, 10, 8)
        run_layout.setSpacing(6)

        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        run_row.addWidget(QLabel("Job name prefix"))
        run_row.addWidget(self._name_prefix, 1)
        self._run_local_button = QPushButton("Run Local Calculations")
        self._run_cluster_button = QPushButton("Run on Cluster")
        run_row.addWidget(self._run_local_button)
        run_row.addWidget(self._run_cluster_button)
        run_layout.addLayout(run_row)

        layout.addWidget(run_group)

    def _connect_signals(self) -> None:
        self._add_cutoff_button.clicked.connect(self._add_cutoff)
        self._generate_range_button.clicked.connect(self._open_range_dialog)
        self._batch_configure_button.clicked.connect(self._configure_selected_cutoffs)

        self._cutoff_table.itemChanged.connect(self._on_cutoff_table_changed)
        self._cutoff_table.customContextMenuRequested.connect(self._open_cutoff_context_menu)
        selection_model = self._cutoff_table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda *_: self._refresh_batch_button_state())
        self._name_prefix.textChanged.connect(self._emit_state)
        self._examine_button.clicked.connect(self.examine_requested.emit)
        self._run_local_button.clicked.connect(self._run_local_selected_cutoffs)
        self._run_cluster_button.clicked.connect(self._run_cluster_selected_cutoffs)
        self._simulation_cell_combo.currentIndexChanged.connect(self._on_simulation_cell_combo_changed)
        self._simulation_cell_create_button.clicked.connect(self.simulation_cell_create_requested.emit)
        self._simulation_cell_use_unit_button.clicked.connect(
            self.simulation_cell_use_unit_cell_requested.emit
        )

    def set_xc_functional_default(self, xc_functional: str | None) -> None:
        """Use the project default XC functional for rows without their own value."""
        seed_xc_functional_default(
            self._default_params,
            xc_functional,
            schema_default=self._schema_xc_functional,
        )

    def set_simulation_cell_options(
        self,
        *,
        summary: str,
        assets: list[dict[str, str]],
        selected_structure_id: str | None,
        enabled: bool,
    ) -> None:
        self._updating_simulation_cell = True
        try:
            self._simulation_cell_summary.setText(summary)
            self._simulation_cell_combo.clear()
            self._simulation_cell_combo.addItem("Single unit cell", None)
            for asset in assets:
                label = str(asset.get("label", "")).strip() or "Saved supercell"
                structure_id = str(asset.get("structure_id", "")).strip()
                self._simulation_cell_combo.addItem(label, structure_id)

            selected_index = 0
            if selected_structure_id:
                matched = self._simulation_cell_combo.findData(str(selected_structure_id))
                if matched >= 0:
                    selected_index = matched
            self._simulation_cell_combo.setCurrentIndex(selected_index)

            self._simulation_cell_combo.setEnabled(enabled)
            self._simulation_cell_create_button.setEnabled(enabled)
            self._simulation_cell_use_unit_button.setEnabled(enabled)
        finally:
            self._updating_simulation_cell = False

    def set_parameter_species_options(self, species_options: Sequence[str]) -> None:
        self._parameter_species_options = tuple(
            dict.fromkeys(option for item in species_options if (option := str(item).strip()))
        )

    def _on_simulation_cell_combo_changed(self, index: int) -> None:
        if self._updating_simulation_cell:
            return
        data = self._simulation_cell_combo.itemData(index)
        if not data:
            self.simulation_cell_use_unit_cell_requested.emit()
            return
        self.simulation_cell_selection_requested.emit(str(data))

    def _shared_locked_values(self) -> dict[str, str]:
        locked: dict[str, str] = {}
        for spec in self._specs:
            if keyword_locked_in_workflow(spec, workflow_stage="convergence_cutoff"):
                locked[spec.name] = "Variable (set by workflow)"

        if "cut_off_energy" in {spec.name for spec in self._specs}:
            locked["cut_off_energy"] = "Variable (set by sweep)"

        return locked

    def _spec_is_blocked_by_locked(self, spec: KeywordSpec, locked_names: set[str]) -> bool:
        if spec.name in locked_names:
            return False
        return any(conflict in locked_names for conflict in spec.mutually_exclusive_with)

    def _default_shared_params(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {"task": infer_task("convergence_cutoff")}
        locked_names = set(self._shared_locked_values())

        for spec in self._specs:
            if spec.default_value is None:
                continue
            if not keyword_visible_in_workflow(
                spec,
                workflow_stage="convergence_cutoff",
                include_advanced=False,
            ):
                continue
            if self._spec_is_blocked_by_locked(spec, locked_names):
                continue
            if spec.name in locked_names:
                continue
            defaults[spec.name] = normalize_gui_value(spec, spec.default_value)

        return defaults

    def _make_table_item(self, text: str, *, editable: bool) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        return item

    @staticmethod
    def _status_brush(status_value: str):
        return status_brush(status_value)

    def _next_cutoff_guess(self) -> float:
        guess = 300.0
        if self._cutoff_table.rowCount() == 0:
            return guess

        previous = self._cutoff_value_at_row(self._cutoff_table.rowCount() - 1, strict=False)
        if previous is None:
            return guess
        return previous + 20.0

    def _insert_cutoff_row(self, value: float) -> None:
        row = self._cutoff_table.rowCount()
        self._cutoff_table.insertRow(row)
        self._cutoff_table.setItem(
            row,
            self._CUTOFF_COLUMN,
            self._make_table_item(f"{float(value):.8g}", editable=True),
        )
        self._cutoff_table.setItem(
            row,
            self._STATUS_COLUMN,
            self._make_table_item("not generated", editable=False),
        )
        self._cutoff_table.setItem(
            row,
            self._TOTAL_ENERGY_COLUMN,
            self._make_table_item("", editable=False),
        )
        self._cutoff_table.setItem(
            row,
            self._MAX_FORCE_COLUMN,
            self._make_table_item("", editable=False),
        )

        configure_button = SelectionSnapshotPushButton(self._cutoff_table)
        configure_button.selection_snapshot_key = value
        self._style_configure_button(configure_button)
        configure_button.clicked.connect(
            lambda _checked=False, key=value, btn=configure_button: self._configure_cutoff_button_clicked(btn, key)
        )
        self._cutoff_table.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

        remove_button = SelectionSnapshotToolButton(self._cutoff_table)
        remove_button.selection_snapshot_key = value
        remove_button.setText("x")
        remove_button.setAutoRaise(True)
        remove_button.setToolTip("Remove this cutoff")
        remove_button.clicked.connect(
            lambda _checked=False, key=value, btn=remove_button: self._remove_cutoff_button_clicked(btn, key)
        )
        self._cutoff_table.setCellWidget(row, self._REMOVE_COLUMN, remove_button)
        self._refresh_cutoff_row(row)

    def _style_configure_button(self, button: QPushButton) -> None:
        metrics = button.fontMetrics()
        label_width = max(
            metrics.horizontalAdvance("Configure..."),
            metrics.horizontalAdvance("Configured..."),
        )
        button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        button.setMinimumWidth(max(self._CONFIGURE_COLUMN_MIN_WIDTH, label_width + 24))
        button.setMinimumHeight(max(24, metrics.height() + 10))

    def _set_cutoff_values(self, values: tuple[float, ...]) -> None:
        self._cutoff_table.blockSignals(True)
        self._cutoff_table.setRowCount(0)
        for value in values:
            self._insert_cutoff_row(value)
        self._cutoff_table.blockSignals(False)
        self._prune_cutoff_configs()
        self._refresh_all_cutoff_rows()
        self._refresh_batch_button_state()

    def _add_cutoff(self, value: float | None = None) -> None:
        self._cutoff_table.blockSignals(True)
        self._insert_cutoff_row(self._next_cutoff_guess() if value is None else float(value))
        self._cutoff_table.blockSignals(False)
        self._refresh_all_cutoff_rows()
        self._emit_state()

    def _open_range_dialog(self) -> None:
        dialog = CutoffRangeDialog(self._range_state, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._range_state = dialog.defaults_state()
            return

        self._range_state = dialog.defaults_state()
        generated = dialog.values()
        if dialog.replace_existing():
            merged = generated
        else:
            merged = tuple(dict.fromkeys(self._collect_cutoffs(strict=False) + generated))

        self._set_cutoff_values(merged)
        self._emit_state()

    def _format_cutoff_list(self, cutoffs: tuple[float, ...]) -> str:
        return ", ".join(f"{value:.8g} eV" for value in cutoffs)

    def _configure_cutoff_button_clicked(self, button: SelectionSnapshotPushButton, clicked_cutoff: float) -> None:
        snapshot_keys = [
            self._cutoff_value_at_row(r, strict=False)
            for r in button.selection_snapshot_rows
        ]
        cutoffs = resolve_target_keys(
            button,
            clicked_key=clicked_cutoff,
            snapshot_keys=[c for c in snapshot_keys if c is not None],
            eligibility=lambda c: True,
            parent=self,
            action="Configure parameters for",
        )
        if cutoffs:
            self._open_cutoff_parameter_dialog(cutoffs)

    def _configure_selected_cutoffs(self) -> None:
        cutoffs = self._selected_cutoffs()
        if not cutoffs:
            QMessageBox.information(
                self,
                "No cutoffs selected",
                "Select one or more cutoff rows first.",
            )
            return
        self._open_cutoff_parameter_dialog(cutoffs)

    def _open_cutoff_parameter_dialog(self, cutoffs: tuple[float, ...]) -> None:
        if not cutoffs:
            return

        intro_text = f"Included cutoffs: {self._format_cutoff_list(tuple(sorted(cutoffs)))}"
        read_only = self._cutoff_parameters_locked(cutoffs)
        dialog_kwargs = {
            "specs": self._specs,
            "workflow_stage": "convergence_cutoff",
            "locked_values": self._shared_locked_values(),
            "intro_text": intro_text,
            "species_options": self._parameter_species_options,
            "parent": self,
        }
        if read_only:
            dialog_kwargs["read_only"] = True
        dialog = ParameterDialog(**dialog_kwargs)
        dialog.setWindowTitle(
            (
                "Inspect Cutoff Parameters"
                if len(cutoffs) > 1
                else f"Inspect Parameters: {cutoffs[0]:.8g} eV"
            )
            if read_only
            else (
                "Configure Cutoff Parameters"
                if len(cutoffs) > 1
                else f"Configure Parameters: {cutoffs[0]:.8g} eV"
            )
        )
        kpoint_hint = kpoint_exclusive_hint(
            self._effective_params_for_cutoff(c) for c in cutoffs
        )
        dialog.set_values(
            self._common_parameter_values_for_cutoffs(cutoffs),
            preferred_exclusive=kpoint_hint,
        )
        dialog.set_advanced_names(self._advanced_names_for_cutoffs(cutoffs))

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if read_only:
            return

        values = self._sanitize_parameter_mapping(dialog.collect_values())
        advanced_names = set(dialog.advanced_names())
        for cutoff in cutoffs:
            self._set_cutoff_parameter_values(cutoff, values, advanced_names)

        self._refresh_all_cutoff_rows()
        self._emit_state()

    def _effective_params_for_cutoff(self, cutoff: float) -> dict[str, Any]:
        values = self._cutoff_parameter_values.get(cutoff)
        if values is None:
            return dict(self._default_params)
        return dict(values)

    def _set_cutoff_parameter_values(
        self, cutoff: float, values: dict[str, Any], advanced_names: set[str]
    ) -> None:
        normalized = self._sanitize_parameter_mapping(values)
        if normalized == self._default_params and not advanced_names:
            self._cutoff_parameter_values.pop(cutoff, None)
            self._cutoff_added_advanced.pop(cutoff, None)
            return

        self._cutoff_parameter_values[cutoff] = dict(normalized)
        if advanced_names:
            self._cutoff_added_advanced[cutoff] = set(advanced_names)
        else:
            self._cutoff_added_advanced.pop(cutoff, None)

    def _common_parameter_values_for_cutoffs(self, cutoffs: tuple[float, ...]) -> dict[str, Any]:
        effective_maps = [self._effective_params_for_cutoff(cutoff) for cutoff in cutoffs]
        if not effective_maps:
            return dict(self._default_params)

        common = dict(effective_maps[0])
        for candidate in effective_maps[1:]:
            common = {
                name: value
                for name, value in common.items()
                if name in candidate and candidate[name] == value
            }

        if "task" not in common:
            common["task"] = infer_task("convergence_cutoff")
        return common

    def _advanced_names_for_cutoffs(self, cutoffs: tuple[float, ...]) -> set[str]:
        names: set[str] = set()
        for cutoff in cutoffs:
            names.update(self._cutoff_added_advanced.get(cutoff, set()))
        return names

    def _is_cutoff_customized(self, cutoff: float) -> bool:
        return cutoff in self._cutoff_parameter_values or cutoff in self._cutoff_added_advanced

    def _cutoff_parameters_locked(self, cutoffs: tuple[float, ...]) -> bool:
        for cutoff in cutoffs:
            runtime = self._cutoff_runtime_states.get(float(cutoff))
            status = "" if runtime is None else runtime.status.strip().lower()
            if status in _LOCKED_CUTOFF_EDIT_STATUSES:
                return True
        return False

    def _remove_cutoff_button_clicked(self, button: SelectionSnapshotToolButton, clicked_cutoff: float) -> None:
        snapshot_keys = [
            self._cutoff_value_at_row(r, strict=False)
            for r in button.selection_snapshot_rows
        ]
        asked_single: list[bool] = []
        cutoffs = resolve_target_keys(
            button,
            clicked_key=clicked_cutoff,
            snapshot_keys=[c for c in snapshot_keys if c is not None],
            eligibility=lambda c: True,
            parent=self,
            action="Remove",
            confirm_unprompted=lambda: asked_single.append(True)
            or self._confirm_cutoff_removal(self._rows_for_cutoffs((clicked_cutoff,))),
        )
        rows = self._rows_for_cutoffs(cutoffs)
        if rows:
            # The three-way prompt is the confirmation for a multi-row removal,
            # but it says nothing about running jobs, so it is not consent to
            # cancel them: those rows are confirmed again with that spelled out.
            self._request_cutoff_removals(
                rows,
                confirmed=bool(asked_single) or not self._active_cutoffs(rows),
            )

    def _rows_for_cutoffs(self, cutoffs: Sequence[float]) -> tuple[int, ...]:
        return tuple(
            r
            for r in range(self._cutoff_table.rowCount())
            if self._cutoff_value_at_row(r, strict=False) in cutoffs
        )

    def _cutoff_status_at_row(self, row: int) -> str:
        cutoff = self._cutoff_value_at_row(row, strict=False)
        runtime = self._cutoff_runtime_states.get(cutoff) if cutoff is not None else None
        return (runtime.status if runtime is not None else "").strip().lower()

    def _active_cutoffs(self, rows: Sequence[int]) -> list[float]:
        active: list[float] = []
        for row in rows:
            cutoff = self._cutoff_value_at_row(row, strict=False)
            if cutoff is not None and self._cutoff_status_at_row(row) in _ACTIVE_JOB_STATUSES:
                active.append(cutoff)
        return active

    def _confirm_cutoff_removal(self, rows: Sequence[int]) -> bool:
        submitted = sum(
            1 for row in rows if self._cutoff_status_at_row(row) in _PROGRESSED_JOB_STATUSES
        )
        return confirm_row_removal(
            self,
            len(rows),
            noun=("cutoff", "cutoffs"),
            submitted_count=submitted,
            cancelling_count=len(self._active_cutoffs(rows)),
        )

    def _request_cutoff_removals(self, rows: tuple[int, ...], *, confirmed: bool = False) -> None:
        """Remove ``rows`` after one confirmation covering the whole batch.

        Rows whose job is still queued or running have it cancelled first; the
        confirmation says how many, so nothing is cancelled unannounced.
        ``confirmed`` skips the prompt for a caller that has already asked.
        """

        if not rows:
            return
        if not confirmed and not self._confirm_cutoff_removal(rows):
            return

        active = self._active_cutoffs(rows)
        if active:
            self.cancel_requested.emit(active)

        cutoffs = [
            cutoff
            for row in rows
            if (cutoff := self._cutoff_value_at_row(row, strict=False)) is not None
        ]
        invalid_rows = [row for row in rows if self._cutoff_value_at_row(row, strict=False) is None]
        for row in sorted(invalid_rows, reverse=True):
            self._remove_cutoff_row(row)

        if cutoffs:
            self.remove_cutoff_requested.emit(cutoffs)

    def _remove_selected_or_current_cutoffs(self) -> None:
        if self._cutoff_table.state() == QAbstractItemView.State.EditingState:
            return
        selected_rows = self._selected_rows()
        if selected_rows:
            self._request_cutoff_removals(selected_rows)
            return

        current_row = self._cutoff_table.currentRow()
        if current_row >= 0:
            self._request_cutoff_removals((current_row,))

    def _remove_cutoff_row(self, row: int) -> None:
        cutoff = self._cutoff_value_at_row(row, strict=False)
        if cutoff is not None:
            self._cutoff_parameter_values.pop(cutoff, None)
            self._cutoff_added_advanced.pop(cutoff, None)
        self._cutoff_table.removeRow(row)
        self._prune_cutoff_configs()
        self._refresh_batch_button_state()
        self._emit_state()

    def remove_cutoffs(self, cutoffs: list[float] | tuple[float, ...]) -> None:
        targets = {float(value) for value in cutoffs}
        if not targets:
            return
        for row in reversed(range(self._cutoff_table.rowCount())):
            cutoff = self._cutoff_value_at_row(row, strict=False)
            if cutoff is None or cutoff not in targets:
                continue
            self._remove_cutoff_row(row)

    def _cutoff_value_at_row(self, row: int, *, strict: bool) -> float | None:
        item = self._cutoff_table.item(row, self._CUTOFF_COLUMN)
        if item is None:
            return None
        text = item.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            if strict:
                raise
            return None

    def _collect_cutoffs(self, *, strict: bool = True) -> tuple[float, ...]:
        values: list[float] = []
        for row in range(self._cutoff_table.rowCount()):
            value = self._cutoff_value_at_row(row, strict=strict)
            if value is not None:
                values.append(value)
        return tuple(values)

    def _selected_cutoffs(self) -> tuple[float, ...]:
        selection_model = self._cutoff_table.selectionModel()
        if selection_model is None:
            return ()
        rows = sorted(index.row() for index in selection_model.selectedRows())
        selected: list[float] = []
        for row in rows:
            value = self._cutoff_value_at_row(row, strict=False)
            if value is not None:
                selected.append(value)
        return tuple(selected)

    def _selected_rows(self) -> tuple[int, ...]:
        selection_model = self._cutoff_table.selectionModel()
        if selection_model is None:
            return ()
        rows = sorted(index.row() for index in selection_model.selectedRows())
        return tuple(rows)

    def _selected_or_current_cutoffs(self) -> tuple[float, ...]:
        selected = self._selected_cutoffs()
        if selected:
            return selected

        current_row = self._cutoff_table.currentRow()
        if current_row >= 0:
            cutoff = self._cutoff_value_at_row(current_row, strict=False)
            if cutoff is not None:
                return (cutoff,)

        if self._cutoff_table.rowCount() == 1:
            cutoff = self._cutoff_value_at_row(0, strict=False)
            if cutoff is not None:
                return (cutoff,)

        return ()

    def _row_status(self, row: int) -> str:
        return workflow_row_status_text(self._cutoff_table, row, self._STATUS_COLUMN)

    def _open_cutoff_context_menu(self, position) -> None:
        row = self._cutoff_table.rowAt(position.y())
        if row < 0:
            return

        selected_rows = set(self._selected_rows())
        if row not in selected_rows:
            self._cutoff_table.selectRow(row)

        cutoffs = self._selected_cutoffs()
        selected_rows = self._selected_rows()
        statuses = [self._row_status(selected_row) for selected_row in selected_rows]
        can_submit = bool(cutoffs) and all(status in _SUBMITTABLE_JOB_STATUSES for status in statuses)
        can_cancel = bool(cutoffs) and all(status in _ACTIVE_JOB_STATUSES for status in statuses)
        can_failed_actions = bool(cutoffs) and all(status == "failed" for status in statuses)

        menu = QMenu(self)
        submit_action = menu.addAction("Submit Local Calculation")
        submit_action.setEnabled(can_submit)
        submit_action.triggered.connect(
            lambda _checked=False, selected=cutoffs: self.submit_local_requested.emit(list(selected))
        )

        submit_cluster_action = menu.addAction("Submit Cluster Calculation")
        submit_cluster_action.setEnabled(can_submit)
        submit_cluster_action.triggered.connect(
            lambda _checked=False, selected=cutoffs: self.submit_cluster_requested.emit(list(selected))
        )

        cancel_action = menu.addAction("Cancel Calculation")
        cancel_action.setEnabled(can_cancel)
        cancel_action.triggered.connect(
            lambda _checked=False, selected=cutoffs: self._confirm_and_cancel(selected)
        )

        if hasattr(menu, "addSeparator"):
            menu.addSeparator()

        inspect_failed_action = menu.addAction("Inspect Failure Files")
        inspect_failed_action.setEnabled(can_failed_actions)
        inspect_failed_action.triggered.connect(
            lambda _checked=False, selected=cutoffs: self.inspect_failed_requested.emit(list(selected))
        )

        add_reset_failed_action(
            menu,
            keys=cutoffs,
            statuses=statuses,
            emit_reset=self.reset_failed_requested.emit,
        )

        menu.exec(self._cutoff_table.viewport().mapToGlobal(position))

    def _confirm_and_cancel(self, cutoffs: Sequence[float]) -> None:
        phrase = count_phrase(len(cutoffs), ("calculation", "calculations"))
        # Looked up on the module so tests can replace the modal dialog.
        if workflow_table_utils.confirm_destructive(
            self,
            "Cancel calculations",
            f"Cancel {phrase}?",
            "",
            f"Cancel {phrase}",
            "Keep running",
        ):
            self.cancel_requested.emit(list(cutoffs))

    def set_runtime_states(self, states: Mapping[float, SweepPointRunState]) -> None:
        """Update row status/energy display using project job/result state."""
        self._cutoff_runtime_states = {float(key): value for key, value in states.items()}
        self._refresh_all_cutoff_rows()
        self._refresh_examine_button_state()

    def _refresh_all_cutoff_rows(self) -> None:
        self._cutoff_table.blockSignals(True)
        for row in range(self._cutoff_table.rowCount()):
            self._refresh_cutoff_row(row)
        self._cutoff_table.blockSignals(False)

    def _refresh_cutoff_row(self, row: int) -> None:
        cutoff = self._cutoff_value_at_row(row, strict=False)
        runtime = self._cutoff_runtime_states.get(cutoff) if cutoff is not None else None

        status_text = "not generated"
        energy_text = ""
        max_force_text = ""
        if runtime is not None:
            if runtime.status:
                status_text = runtime.status
            if runtime.total_energy is not None:
                energy_text = f"{runtime.total_energy:.12g}"
            if runtime.max_force is not None:
                max_force_text = f"{runtime.max_force:.12g}"

        status_normalized = status_text.strip().lower()

        cutoff_item = self._cutoff_table.item(row, self._CUTOFF_COLUMN)
        if cutoff_item is not None:
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if status_normalized not in _LOCKED_CUTOFF_EDIT_STATUSES:
                flags |= Qt.ItemFlag.ItemIsEditable
                cutoff_item.setToolTip("")
            else:
                cutoff_item.setToolTip(
                    "Cutoff value is locked once this row has been queued for execution or has entered execution history."
                )
            cutoff_item.setFlags(flags)

        status_item = self._cutoff_table.item(row, self._STATUS_COLUMN)
        if status_item is None:
            status_item = self._make_table_item(status_text, editable=False)
            self._cutoff_table.setItem(row, self._STATUS_COLUMN, status_item)
        else:
            status_item.setText(status_text)
        status_item.setForeground(self._status_brush(status_text))

        energy_item = self._cutoff_table.item(row, self._TOTAL_ENERGY_COLUMN)
        if energy_item is None:
            energy_item = self._make_table_item(energy_text, editable=False)
            self._cutoff_table.setItem(row, self._TOTAL_ENERGY_COLUMN, energy_item)
        else:
            energy_item.setText(energy_text)

        max_force_item = self._cutoff_table.item(row, self._MAX_FORCE_COLUMN)
        if max_force_item is None:
            max_force_item = self._make_table_item(max_force_text, editable=False)
            self._cutoff_table.setItem(row, self._MAX_FORCE_COLUMN, max_force_item)
        else:
            max_force_item.setText(max_force_text)

        configure_button = self._cutoff_table.cellWidget(row, self._CONFIGURE_COLUMN)
        if isinstance(configure_button, QPushButton):
            customized = cutoff is not None and self._is_cutoff_customized(cutoff)
            locked = status_normalized in _LOCKED_CUTOFF_EDIT_STATUSES
            style_workflow_configure_button(
                configure_button,
                workflow_label="CASTEP",
                configured=customized,
                locked=locked,
                noun="cutoff",
                used=True,
            )

    def _refresh_batch_button_state(self) -> None:
        self._batch_configure_button.setEnabled(bool(self._selected_cutoffs()))

    def _refresh_examine_button_state(self) -> None:
        completed_with_results = 0
        for runtime in self._cutoff_runtime_states.values():
            if runtime.status.strip().lower() != "completed":
                continue
            if runtime.total_energy is None or runtime.max_force is None:
                continue
            completed_with_results += 1

        enabled = completed_with_results >= 2
        self._examine_button.setEnabled(enabled)
        if enabled:
            self._examine_button.setToolTip("Analyse completed cutoff energy and force trends.")
        else:
            self._examine_button.setToolTip(
                "Need at least two completed cutoffs with both total energy and max-force results."
            )

    def _prune_cutoff_configs(self) -> None:
        available = set(self._collect_cutoffs(strict=False))
        stale = [cutoff for cutoff in self._cutoff_parameter_values if cutoff not in available]
        for cutoff in stale:
            self._cutoff_parameter_values.pop(cutoff, None)
            self._cutoff_added_advanced.pop(cutoff, None)

        stale_advanced = [cutoff for cutoff in self._cutoff_added_advanced if cutoff not in available]
        for cutoff in stale_advanced:
            self._cutoff_added_advanced.pop(cutoff, None)

    def _on_cutoff_table_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self._CUTOFF_COLUMN:
            return
        self._prune_cutoff_configs()
        self._refresh_all_cutoff_rows()
        self._emit_state()

    def _emit_state(self) -> None:
        if self._restoring_form_state:
            return
        self.form_state_changed.emit(self.collect_state())

    def _restore_structured_value(self, name: str, value: object) -> object:
        spec = self._specs_by_name.get(name)
        if spec is None:
            return value

        if isinstance(value, dict):
            if spec.kind == "block" and "rows" in value:
                rows = tuple(tuple(row) for row in value.get("rows", ()))
                return BlockSection(
                    name=str(value.get("name", spec.name)),
                    rows=rows,
                    unit=str(value["unit"]) if value.get("unit") is not None else None,
                )
            if spec.value_type in {"physical", "positive_float_with_optional_unit"} and {"value", "unit"} <= value.keys():
                return ValueWithUnit(value=value["value"], unit=str(value["unit"]))

        if isinstance(value, list) and spec.value_type in {"int_triplet", "real_vector"}:
            value = tuple(value)

        try:
            return normalize_gui_value(spec, value)
        except (TypeError, ValueError):
            return value

    def _sanitize_parameter_mapping(self, values: dict[str, object]) -> dict[str, object]:
        cleaned: dict[str, object] = {}
        for name, value in values.items():
            if name in _STRUCTURE_CONTROLLED_CELL_KEYWORDS:
                continue
            if name == "task" or name in self._specs_by_name:
                cleaned[name] = self._restore_structured_value(name, value)
        return cleaned

    def _restore_legacy_cutoff_configs(self, state: dict[str, Any], cutoffs: tuple[float, ...]) -> None:
        shared = state.get("shared_params", {})
        shared_clean = (
            self._sanitize_parameter_mapping({str(name): raw for name, raw in shared.items()})
            if isinstance(shared, dict)
            else dict(self._default_params)
        )
        if "task" not in shared_clean:
            shared_clean["task"] = infer_task("convergence_cutoff")

        shared_advanced = {
            str(name) for name in state.get("shared_added_advanced", []) if isinstance(name, str)
        }

        overrides_by_cutoff: dict[float, dict[str, object]] = {}
        override_advanced: dict[float, set[str]] = {}
        for entry in state.get("overrides", []):
            if not isinstance(entry, dict):
                continue
            params = self._sanitize_parameter_mapping(
                {str(name): raw for name, raw in dict(entry.get("params", {})).items()}
            )
            advanced = {name for name in params if name in self._specs_by_name and params[name] is not None}
            for raw_cutoff in entry.get("cutoffs", []):
                try:
                    cutoff = float(raw_cutoff)
                except (TypeError, ValueError):
                    continue
                overrides_by_cutoff[cutoff] = dict(params)
                override_advanced[cutoff] = set(advanced)

        for cutoff in cutoffs:
            merged = dict(shared_clean)
            merged.update(overrides_by_cutoff.get(cutoff, {}))
            advanced_names = set(shared_advanced).union(override_advanced.get(cutoff, set()))
            self._set_cutoff_parameter_values(cutoff, merged, advanced_names)

    def collect_state(self) -> dict[str, Any]:
        return {
            "mode": "manual",
            "name_prefix": self._name_prefix.text().strip() or "cutoff_convergence",
            "range": dict(self._range_state),
            "cutoff_energies": list(self._collect_cutoffs(strict=False)),
            "cutoff_parameter_rows": [
                {
                    "cutoff": cutoff,
                    "params": dict(self._cutoff_parameter_values[cutoff]),
                    "added_advanced": sorted(self._cutoff_added_advanced.get(cutoff, set())),
                }
                for cutoff in sorted(self._cutoff_parameter_values)
            ],
        }

    @staticmethod
    def _coerce_cutoff_values(raw_values: object) -> tuple[float, ...]:
        values: list[float] = []
        if isinstance(raw_values, list):
            for value in raw_values:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
        return tuple(values)

    @staticmethod
    def _coerce_cutoffs_from_rows(raw_rows: object) -> tuple[float, ...]:
        values: list[float] = []
        if isinstance(raw_rows, list):
            for entry in raw_rows:
                if not isinstance(entry, dict):
                    continue
                try:
                    values.append(float(entry.get("cutoff")))
                except (TypeError, ValueError):
                    continue
        return tuple(values)

    def set_form_state(self, state: dict[str, Any]) -> None:
        self._restoring_form_state = True
        try:
            self._range_state = {
                "start": 300.0,
                "stop": 1400.0,
                "step": 100.0,
                "replace_existing": False,
            }
            self._cutoff_parameter_values = {}
            self._cutoff_added_advanced = {}

            previous_block = self._name_prefix.blockSignals(True)
            self._name_prefix.setText("cutoff_convergence")
            self._name_prefix.blockSignals(previous_block)

            cutoffs: tuple[float, ...] = ()
            cutoff_rows: list[dict[str, Any]] = []
            if state:
                previous_block = self._name_prefix.blockSignals(True)
                self._name_prefix.setText(str(state.get("name_prefix", "cutoff_convergence")))
                self._name_prefix.blockSignals(previous_block)

                range_state = state.get("range", {})
                if isinstance(range_state, dict):
                    for key in ("start", "stop", "step"):
                        try:
                            self._range_state[key] = float(range_state.get(key, self._range_state[key]))
                        except (TypeError, ValueError):
                            continue
                    self._range_state["replace_existing"] = bool(
                        range_state.get("replace_existing", self._range_state["replace_existing"])
                    )

                from_list = self._coerce_cutoff_values(state.get("cutoff_energies", []))
                from_rows = self._coerce_cutoffs_from_rows(state.get("cutoff_parameter_rows", []))
                cutoffs = tuple(dict.fromkeys(from_list + from_rows))

                raw_cutoff_rows = state.get("cutoff_parameter_rows", [])
                if isinstance(raw_cutoff_rows, list):
                    cutoff_rows = [entry for entry in raw_cutoff_rows if isinstance(entry, dict)]

            self._set_cutoff_values(cutoffs)

            if cutoff_rows:
                for entry in cutoff_rows:
                    try:
                        cutoff = float(entry.get("cutoff"))
                    except (TypeError, ValueError):
                        continue
                    params = self._sanitize_parameter_mapping(
                        {str(name): raw for name, raw in dict(entry.get("params", {})).items()}
                    )
                    advanced_names = {
                        str(name) for name in entry.get("added_advanced", []) if isinstance(name, str)
                    }
                    self._set_cutoff_parameter_values(cutoff, params, advanced_names)
            elif state:
                self._restore_legacy_cutoff_configs(state, cutoffs)

            self._refresh_all_cutoff_rows()
            self._refresh_batch_button_state()
        finally:
            self._restoring_form_state = False

        self._emit_state()

    def _effective_params_by_cutoff(self, cutoffs: tuple[float, ...]) -> dict[float, dict[str, Any]]:
        return {cutoff: self._effective_params_for_cutoff(cutoff) for cutoff in cutoffs}

    def _shared_params_from_effective(self, effective: dict[float, dict[str, Any]]) -> dict[str, Any]:
        if not effective:
            return dict(self._default_params)

        maps = list(effective.values())
        shared = dict(maps[0])
        for candidate in maps[1:]:
            shared = {
                name: value
                for name, value in shared.items()
                if name in candidate and candidate[name] == value
            }
        if "task" not in shared:
            shared["task"] = infer_task("convergence_cutoff")
        return shared

    def _overrides_from_effective(
        self, cutoffs: tuple[float, ...], effective: dict[float, dict[str, Any]], shared: dict[str, Any]
    ) -> dict[frozenset[float], dict[str, object]]:
        grouped: list[tuple[set[float], dict[str, object]]] = []
        for cutoff in cutoffs:
            diff = {
                name: value
                for name, value in effective[cutoff].items()
                if shared.get(name) != value
            }
            if not diff:
                continue
            for members, payload in grouped:
                if payload == diff:
                    members.add(cutoff)
                    break
            else:
                grouped.append(({cutoff}, diff))

        return {frozenset(members): payload for members, payload in grouped}

    def build_config(self) -> ConvergenceConfig | None:
        try:
            cutoffs = self._collect_cutoffs()
        except ValueError:
            QMessageBox.warning(self, "Invalid cutoff list", "Cutoff values must be numeric.")
            return None

        if not cutoffs:
            QMessageBox.warning(self, "No cutoffs", "Add one or more cutoff energies first.")
            return None

        if len(set(cutoffs)) != len(cutoffs):
            QMessageBox.warning(self, "Duplicate cutoffs", "Cutoff energies must be unique.")
            return None

        effective = self._effective_params_by_cutoff(cutoffs)
        for cutoff, values in effective.items():
            task = str(values.get("task", infer_task("convergence_cutoff")))
            errors = validate_gui_values(values, self._specs_by_name, task=task)
            if errors:
                QMessageBox.warning(
                    self,
                    "Invalid parameters",
                    f"Cutoff {cutoff:.8g} eV:\n" + "\n".join(errors),
                )
                return None

        shared_params = self._shared_params_from_effective(effective)
        overrides = self._overrides_from_effective(cutoffs, effective, shared_params)

        return ConvergenceConfig(
            sweep_parameter="cut_off_energy",
            sweep_values=cutoffs,
            cutoff_energies=cutoffs,
            shared_params=shared_params,
            overrides=overrides,
            name_prefix=self._name_prefix.text().strip() or "convergence",
        )

    def _emit_config(self) -> None:
        config = self.build_config()
        if config is None:
            return
        self.configure_requested.emit(config)
        self._emit_state()


    def _run_local_selected_cutoffs(self) -> None:
        config = self.build_config()
        if config is None:
            return

        selected_cutoffs = tuple(dict.fromkeys(self._selected_or_current_cutoffs()))
        if not selected_cutoffs:
            QMessageBox.information(
                self,
                "No cutoffs selected",
                "Select one or more cutoff rows first.",
            )
            return

        self.configure_requested.emit(config)
        self.submit_local_requested.emit(list(selected_cutoffs))
        self._emit_state()

    def _run_cluster_selected_cutoffs(self) -> None:
        config = self.build_config()
        if config is None:
            return

        selected_cutoffs = tuple(dict.fromkeys(self._selected_or_current_cutoffs()))
        if not selected_cutoffs:
            QMessageBox.information(
                self,
                "No cutoffs selected",
                "Select one or more cutoff rows first.",
            )
            return

        self.configure_requested.emit(config)
        self.submit_cluster_requested.emit(list(selected_cutoffs))
        self._emit_state()


# Backward-compatible export used by existing call sites.
ConvergenceWidget = CutoffConvergenceWidget

"""K-point convergence configuration panel with workflow-aware CASTEP settings."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIntValidator
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
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.convergence_display import (
    format_convergence_criteria,
    format_convergence_method,
    format_cutoff_energy,
    format_kpoint_grid,
)
from mufinder.app.convergence_types import ConvergenceConfig, SweepPointRunState
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.theme import (
    status_brush,
    style_section_subtitle,
    style_section_title,
    style_warning_text,
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
from mufinder.science.kpoint_grids import generate_kpoint_grids, lattice_matches

_STRUCTURE_CONTROLLED_CELL_KEYWORDS = {
    "lattice_cart",
    "lattice_abc",
    "positions_frac",
    "positions_abs",
}
_KPOINT_SORT_ANCHOR_NAMES = ("basis_precision", "cut_off_energy")
_ACTIVE_JOB_STATUSES = {"submitting", "queued", "running", "retrieving"}
_PROGRESSED_JOB_STATUSES = _ACTIVE_JOB_STATUSES | {"stopped", "completed", "failed", "cancelled"}
_LOCKED_GRID_EDIT_STATUSES = _PROGRESSED_JOB_STATUSES
_DEFAULT_KPOINT_SPACING = 0.02


def _reorder_specs_with_kpoint_anchor(specs: list[KeywordSpec]) -> list[KeywordSpec]:
    kpoint_name = "kpoint_mp_grid"
    kpoint_index = next((index for index, spec in enumerate(specs) if spec.name == kpoint_name), None)
    if kpoint_index is None:
        return specs

    anchor_indices = [
        index
        for index, spec in enumerate(specs)
        if spec.name in _KPOINT_SORT_ANCHOR_NAMES
    ]
    if not anchor_indices:
        return specs

    target_index = max(anchor_indices) + 1
    if kpoint_index == target_index:
        return specs

    kpoint_spec = specs.pop(kpoint_index)
    if kpoint_index < target_index:
        target_index -= 1
    specs.insert(target_index, kpoint_spec)
    return specs


class KPointSpacingDialog(QDialog):
    """Dialog for generating Monkhorst-Pack grids from a spacing target."""

    def __init__(self, defaults: Mapping[str, object] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Generate K-Point Grids")

        defaults = defaults or {}
        spacing = float(defaults.get("spacing", _DEFAULT_KPOINT_SPACING))
        replace_existing = bool(defaults.get("replace_existing", False))

        self._spacing = DecimalLineEdit(self)
        self._spacing.setDecimals(4)
        self._spacing.setRange(0.001, 10.0)
        self._spacing.setSingleStep(0.01)
        self._spacing.setValue(spacing)

        self._replace_existing = QCheckBox("Replace existing grid list", self)
        self._replace_existing.setChecked(replace_existing)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Minimum spacing (1/Ang)", self._spacing)
        layout.addLayout(form)
        layout.addWidget(self._replace_existing)
        layout.addWidget(self._buttons)

        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

    def spacing(self) -> float:
        return float(self._spacing.value())

    def replace_existing(self) -> bool:
        return self._replace_existing.isChecked()

    def defaults_state(self) -> dict[str, object]:
        return {
            "spacing": float(self._spacing.value()),
            "replace_existing": self._replace_existing.isChecked(),
        }


class KPointGridDialog(QDialog):
    """Small helper dialog for entering one Monkhorst-Pack grid triplet."""

    def __init__(self, grid: tuple[int, int, int] = (1, 1, 1), parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add K-Point Grid")

        validator = QIntValidator(1, 9999, self)

        self._nx = QLineEdit(str(int(grid[0])), self)
        self._ny = QLineEdit(str(int(grid[1])), self)
        self._nz = QLineEdit(str(int(grid[2])), self)
        for field in (self._nx, self._ny, self._nz):
            field.setValidator(validator)
            field.setMaxLength(4)
            field.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Enter Monkhorst-Pack grid integers", self))

        row = QHBoxLayout()
        row.addWidget(self._nx)
        row.addWidget(QLabel("x", self))
        row.addWidget(self._ny)
        row.addWidget(QLabel("x", self))
        row.addWidget(self._nz)
        layout.addLayout(row)
        layout.addWidget(self._buttons)

        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._nx.textChanged.connect(self._refresh_ok_state)
        self._ny.textChanged.connect(self._refresh_ok_state)
        self._nz.textChanged.connect(self._refresh_ok_state)

        self._refresh_ok_state()

    def _refresh_ok_state(self) -> None:
        ok_button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is None:
            return
        ok_button.setEnabled(
            bool(self._nx.hasAcceptableInput())
            and bool(self._ny.hasAcceptableInput())
            and bool(self._nz.hasAcceptableInput())
        )

    def grid(self) -> tuple[int, int, int]:
        return (int(self._nx.text()), int(self._ny.text()), int(self._nz.text()))


class KPointConvergenceWidget(QWidget):
    """UI for configuring k-point convergence sweeps and per-grid parameters."""

    _GRID_COLUMN = 0
    _STATUS_COLUMN = 1
    _TOTAL_ENERGY_COLUMN = 2
    _MAX_FORCE_COLUMN = 3
    _CONFIGURE_COLUMN = 4
    _REMOVE_COLUMN = 5
    _ROW_HEIGHT = 32
    _GRID_COLUMN_WIDTH = 128
    _CONFIGURE_COLUMN_MIN_WIDTH = 136
    _REMOVE_COLUMN_WIDTH = 34

    configure_requested = Signal(object)
    submit_local_requested = Signal(object)
    submit_cluster_requested = Signal(object)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    remove_grid_requested = Signal(object)
    examine_requested = Signal()
    form_state_changed = Signal(dict)
    simulation_cell_create_requested = Signal()
    simulation_cell_selection_requested = Signal(object)
    simulation_cell_use_unit_cell_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._warning_label = QLabel("")
        self._warning_label.setWordWrap(True)
        style_warning_text(self._warning_label)
        self._warning_label.hide()

        self._project_default_grid = QLabel("Not set")
        self._project_default_method = QLabel("Not set")
        self._project_default_criteria = QLabel("Not set")
        self._project_default_cutoff = QLabel("Not set")

        self._range_state: dict[str, object] = {
            "spacing": _DEFAULT_KPOINT_SPACING,
            "replace_existing": False,
        }

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            [
                "K-Point Grid",
                "Status",
                "Total Energy (eV)",
                "Max |Force| (eV/Ang)",
                "Parameters",
                "",
            ]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(self._ROW_HEIGHT)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self._GRID_COLUMN, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(self._STATUS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._TOTAL_ENERGY_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._MAX_FORCE_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._CONFIGURE_COLUMN, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(self._REMOVE_COLUMN, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._GRID_COLUMN, self._GRID_COLUMN_WIDTH)
        self._table.setColumnWidth(self._CONFIGURE_COLUMN, self._CONFIGURE_COLUMN_MIN_WIDTH)
        self._table.setColumnWidth(self._REMOVE_COLUMN, self._REMOVE_COLUMN_WIDTH)

        self._add_grid_button = QPushButton("Add Grid")
        self._generate_button = QPushButton("Generate From Spacing...")
        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._batch_configure_button.setEnabled(False)
        self._examine_button = QPushButton("Examine K-Point Convergence")
        self._examine_button.setEnabled(False)

        self._run_local_button = QPushButton("Run Local Calculations")
        self._run_cluster_button = QPushButton("Run on Cluster")

        self._name_prefix = QLineEdit("kpoint_convergence")
        self._simulation_cell_summary = QLabel("Using single unit cell.")
        self._simulation_cell_summary.setWordWrap(True)
        self._simulation_cell_combo = QComboBox()
        self._simulation_cell_create_button = QPushButton("Manage Supercells...")
        self._simulation_cell_use_unit_button = QPushButton("Use Unit Cell")

        self._runtime_states: dict[tuple[int, int, int], SweepPointRunState] = {}
        self._parameter_values: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._added_advanced: dict[tuple[int, int, int], set[str]] = {}
        self._restoring_form_state = False
        self._updating_simulation_cell = False

        self._generation_cell_id: str | None = None
        self._generation_lattice: list[list[float]] | None = None
        self._current_cell_id: str | None = None
        self._current_lattice: list[list[float]] | None = None

        specs = get_gui_keyword_specs_for_files(
            files=("param", "cell"),
            workflow_stage="convergence_kpoint",
            include_advanced=True,
        )
        self._specs = [spec for spec in specs if spec.name not in {"task", "spin_polarised"}]
        self._specs = _reorder_specs_with_kpoint_anchor(self._specs)
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        self._default_params = self._default_shared_params()

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        defaults_group = QGroupBox("Project Defaults")
        defaults_layout = QVBoxLayout(defaults_group)
        defaults_layout.setContentsMargins(10, 8, 10, 8)
        defaults_layout.setSpacing(4)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row1.addWidget(QLabel("Default k-point grid:"))
        row1.addWidget(self._project_default_grid)
        row1.addSpacing(8)
        row1.addWidget(QLabel("Method:"))
        row1.addWidget(self._project_default_method)
        row1.addStretch(1)
        defaults_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("Criteria:"))
        row2.addWidget(self._project_default_criteria)
        row2.addStretch(1)
        defaults_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.setSpacing(8)
        row3.addWidget(QLabel("Applied cutoff default:"))
        row3.addWidget(self._project_default_cutoff)
        row3.addStretch(1)
        defaults_layout.addLayout(row3)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)
        title = QLabel("K-Point Convergence")
        style_section_title(title)
        title_column.addWidget(title)

        subtitle = QLabel("Refine k-point sampling so energies and forces are stable.")
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        title_column.addWidget(subtitle)

        header_row.addLayout(title_column, 1)
        header_row.addWidget(defaults_group)
        layout.addLayout(header_row)

        layout.addWidget(self._warning_label)

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

        sweep_group = QGroupBox("K-Point Grid Sweep")
        sweep_layout = QVBoxLayout(sweep_group)
        sweep_layout.setContentsMargins(10, 8, 10, 8)
        sweep_layout.setSpacing(6)
        sweep_layout.addWidget(self._table)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(6)
        controls_row.addWidget(self._add_grid_button)
        controls_row.addWidget(self._generate_button)
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
        run_row.addWidget(self._run_local_button)
        run_row.addWidget(self._run_cluster_button)
        run_layout.addLayout(run_row)
        layout.addWidget(run_group)

    def _connect_signals(self) -> None:
        self._add_grid_button.clicked.connect(self._open_add_grid_dialog)
        self._generate_button.clicked.connect(self._open_spacing_dialog)
        self._batch_configure_button.clicked.connect(self._configure_selected_grids)

        self._table.itemChanged.connect(self._on_table_changed)
        selection_model = self._table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda *_: self._refresh_batch_button_state())

        self._name_prefix.textChanged.connect(self._emit_state)
        self._examine_button.clicked.connect(self.examine_requested.emit)
        self._run_local_button.clicked.connect(self._run_local_selected_grids)
        self._run_cluster_button.clicked.connect(self._run_cluster_selected_grids)

        self._simulation_cell_combo.currentIndexChanged.connect(self._on_simulation_cell_combo_changed)
        self._simulation_cell_create_button.clicked.connect(self.simulation_cell_create_requested.emit)
        self._simulation_cell_use_unit_button.clicked.connect(self.simulation_cell_use_unit_cell_requested.emit)

    def _make_item(self, text: str, *, editable: bool) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        return item

    @staticmethod
    def _status_brush(status_value: str):
        return status_brush(status_value)

    @staticmethod
    def _format_grid(grid: tuple[int, int, int]) -> str:
        return f"{grid[0]} x {grid[1]} x {grid[2]}"

    @staticmethod
    def _parse_grid(text: str) -> tuple[int, int, int]:
        parts = [part for part in re.split(r"[^0-9]+", text.strip()) if part]
        if len(parts) != 3:
            raise ValueError("K-point grid must contain exactly three integers.")
        grid = (int(parts[0]), int(parts[1]), int(parts[2]))
        if grid[0] <= 0 or grid[1] <= 0 or grid[2] <= 0:
            raise ValueError("K-point grid values must be positive integers.")
        return grid

    def set_project_defaults(
        self,
        *,
        kpoint_grid: tuple[int, int, int] | None,
        kpoint_method: str | None,
        energy_threshold: float | None,
        force_threshold: float | None,
    ) -> None:
        self._project_default_grid.setText(format_kpoint_grid(kpoint_grid))
        self._project_default_method.setText(format_convergence_method(kpoint_method))
        self._project_default_criteria.setText(
            format_convergence_criteria(
                energy_threshold=energy_threshold,
                force_threshold=force_threshold,
            )
        )

    def set_cutoff_default(self, cutoff_energy: float | None) -> None:
        self._project_default_cutoff.setText(format_cutoff_energy(cutoff_energy))
        if cutoff_energy is None:
            self._default_params.pop("cut_off_energy", None)
        else:
            self._default_params["cut_off_energy"] = ValueWithUnit(float(cutoff_energy), "eV")

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

    def _on_simulation_cell_combo_changed(self, index: int) -> None:
        if self._updating_simulation_cell:
            return
        data = self._simulation_cell_combo.itemData(index)
        if not data:
            self.simulation_cell_use_unit_cell_requested.emit()
            return
        self.simulation_cell_selection_requested.emit(str(data))

    def set_generation_context(
        self,
        *,
        structure_id: str | None,
        lattice_vectors: list[list[float]] | None,
    ) -> None:
        self._current_cell_id = structure_id
        self._current_lattice = lattice_vectors
        self._refresh_generation_warning()

    def _refresh_generation_warning(self) -> None:
        if self._generation_lattice is None or self._current_lattice is None:
            self._warning_label.hide()
            return
        if lattice_matches(self._generation_lattice, self._current_lattice):
            self._warning_label.hide()
            return

        self._warning_label.setText(
            "K-point grids were generated for a different simulation cell. Regenerate grids before running new jobs."
        )
        self._warning_label.show()

    def _shared_locked_values(self) -> dict[str, str]:
        locked: dict[str, str] = {}
        for spec in self._specs:
            if keyword_locked_in_workflow(spec, workflow_stage="convergence_kpoint"):
                locked[spec.name] = "Variable (set by workflow)"

        if "kpoint_mp_grid" in {spec.name for spec in self._specs}:
            locked["kpoint_mp_grid"] = "Variable (set by sweep)"

        return locked

    def _spec_is_blocked_by_locked(self, spec: KeywordSpec, locked_names: set[str]) -> bool:
        if spec.name in locked_names:
            return False
        return any(conflict in locked_names for conflict in spec.mutually_exclusive_with)

    def _default_shared_params(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {"task": infer_task("convergence_kpoint")}
        locked_names = set(self._shared_locked_values())

        for spec in self._specs:
            if spec.default_value is None:
                continue
            if not keyword_visible_in_workflow(
                spec,
                workflow_stage="convergence_kpoint",
                include_advanced=False,
            ):
                continue
            if self._spec_is_blocked_by_locked(spec, locked_names):
                continue
            if spec.name in locked_names:
                continue
            defaults[spec.name] = normalize_gui_value(spec, spec.default_value)

        return defaults

    def _grid_value_at_row(self, row: int, *, strict: bool) -> tuple[int, int, int] | None:
        item = self._table.item(row, self._GRID_COLUMN)
        if item is None:
            return None
        text = item.text().strip()
        if not text:
            return None
        try:
            return self._parse_grid(text)
        except ValueError:
            if strict:
                raise
            return None

    def _collect_grids(self, *, strict: bool = True) -> tuple[tuple[int, int, int], ...]:
        values: list[tuple[int, int, int]] = []
        for row in range(self._table.rowCount()):
            value = self._grid_value_at_row(row, strict=strict)
            if value is not None:
                values.append(value)
        return tuple(values)

    def _insert_grid_row(self, grid: tuple[int, int, int]) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, self._GRID_COLUMN, self._make_item(self._format_grid(grid), editable=True))
        self._table.setItem(row, self._STATUS_COLUMN, self._make_item("not generated", editable=False))
        self._table.setItem(row, self._TOTAL_ENERGY_COLUMN, self._make_item("", editable=False))
        self._table.setItem(row, self._MAX_FORCE_COLUMN, self._make_item("", editable=False))

        configure_button = QPushButton(self._table)
        self._style_configure_button(configure_button)
        configure_button.clicked.connect(
            lambda _checked=False, button=configure_button: self._configure_grid_button_clicked(button)
        )
        self._table.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

        remove_button = QToolButton(self._table)
        remove_button.setText("x")
        remove_button.setAutoRaise(True)
        remove_button.setToolTip("Remove this k-point grid")
        remove_button.clicked.connect(
            lambda _checked=False, button=remove_button: self._remove_grid_button_clicked(button)
        )
        self._table.setCellWidget(row, self._REMOVE_COLUMN, remove_button)
        self._refresh_row(row)

    def _style_configure_button(self, button: QPushButton) -> None:
        metrics = button.fontMetrics()
        label_width = max(
            metrics.horizontalAdvance("Configure..."),
            metrics.horizontalAdvance("Configured..."),
        )
        button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        button.setMinimumWidth(max(self._CONFIGURE_COLUMN_MIN_WIDTH, label_width + 24))
        button.setMinimumHeight(max(24, metrics.height() + 10))

    def _set_grid_values(self, values: tuple[tuple[int, int, int], ...]) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for value in values:
            self._insert_grid_row(value)
        self._table.blockSignals(False)
        self._prune_configs()
        self._refresh_all_rows()
        self._refresh_batch_button_state()

    def _add_grid(self, grid: tuple[int, int, int] | None = None) -> None:
        target = (1, 1, 1) if grid is None else grid
        self._table.blockSignals(True)
        self._insert_grid_row(target)
        self._table.blockSignals(False)
        self._refresh_all_rows()
        self._emit_state()

    def _open_add_grid_dialog(self) -> None:
        grid = (1, 1, 1)
        if self._table.rowCount() > 0:
            last = self._grid_value_at_row(self._table.rowCount() - 1, strict=False)
            if last is not None:
                grid = last

        dialog = KPointGridDialog(grid=grid, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._add_grid(dialog.grid())

    def _open_spacing_dialog(self) -> None:
        if self._current_lattice is None:
            QMessageBox.warning(self, "No simulation cell", "Select or create a simulation cell first.")
            return

        dialog = KPointSpacingDialog(self._range_state, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._range_state = dialog.defaults_state()
            return

        self._range_state = dialog.defaults_state()
        try:
            generated = generate_kpoint_grids(self._current_lattice, dialog.spacing())
        except ValueError as exc:
            QMessageBox.warning(self, "Grid generation failed", str(exc))
            return

        if dialog.replace_existing():
            merged = generated
        else:
            merged = tuple(dict.fromkeys(self._collect_grids(strict=False) + generated))

        self._set_grid_values(merged)
        self._generation_cell_id = self._current_cell_id
        self._generation_lattice = [list(row) for row in self._current_lattice]
        self._refresh_generation_warning()
        self._emit_state()

    def _selected_rows(self) -> tuple[int, ...]:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return ()
        rows = sorted(index.row() for index in selection_model.selectedRows())
        return tuple(rows)

    def _selected_grids(self) -> tuple[tuple[int, int, int], ...]:
        selected: list[tuple[int, int, int]] = []
        for row in self._selected_rows():
            grid = self._grid_value_at_row(row, strict=False)
            if grid is not None:
                selected.append(grid)
        return tuple(selected)

    def _selected_or_current_grids(self) -> tuple[tuple[int, int, int], ...]:
        selected = self._selected_grids()
        if selected:
            return selected

        current_row = self._table.currentRow()
        if current_row >= 0:
            grid = self._grid_value_at_row(current_row, strict=False)
            if grid is not None:
                return (grid,)

        if self._table.rowCount() == 1:
            grid = self._grid_value_at_row(0, strict=False)
            if grid is not None:
                return (grid,)

        return ()

    def _configure_selected_grids(self) -> None:
        grids = self._selected_grids()
        if not grids:
            QMessageBox.information(self, "No grids selected", "Select one or more k-point rows first.")
            return
        self._open_grid_parameter_dialog(grids)

    def _configure_grid_button_clicked(self, button: QPushButton) -> None:
        for row in range(self._table.rowCount()):
            if self._table.cellWidget(row, self._CONFIGURE_COLUMN) is button:
                grid = self._grid_value_at_row(row, strict=False)
                if grid is not None:
                    self._open_grid_parameter_dialog((grid,))
                return

    def _open_grid_parameter_dialog(self, grids: tuple[tuple[int, int, int], ...]) -> None:
        if not grids:
            return

        intro = "Included grids: " + ", ".join(self._format_grid(grid) for grid in sorted(grids))
        read_only = self._grid_parameters_locked(grids)
        dialog_kwargs = {
            "specs": self._specs,
            "workflow_stage": "convergence_kpoint",
            "locked_values": self._shared_locked_values(),
            "intro_text": intro,
            "parent": self,
        }
        if read_only:
            dialog_kwargs["read_only"] = True
        dialog = ParameterDialog(**dialog_kwargs)
        dialog.setWindowTitle("Inspect K-Point Parameters" if read_only else "Configure K-Point Parameters")
        dialog.set_values(self._common_parameter_values_for_grids(grids))
        dialog.set_advanced_names(self._advanced_names_for_grids(grids))

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if read_only:
            return

        values = self._sanitize_parameter_mapping(dialog.collect_values())
        advanced_names = set(dialog.advanced_names())
        for grid in grids:
            self._set_grid_parameter_values(grid, values, advanced_names)

        self._refresh_all_rows()
        self._emit_state()

    def _effective_params_for_grid(self, grid: tuple[int, int, int]) -> dict[str, Any]:
        values = self._parameter_values.get(grid)
        if values is None:
            return dict(self._default_params)
        return dict(values)

    def _set_grid_parameter_values(
        self,
        grid: tuple[int, int, int],
        values: dict[str, Any],
        advanced_names: set[str],
    ) -> None:
        normalized = self._sanitize_parameter_mapping(values)
        if normalized == self._default_params and not advanced_names:
            self._parameter_values.pop(grid, None)
            self._added_advanced.pop(grid, None)
            return

        self._parameter_values[grid] = dict(normalized)
        if advanced_names:
            self._added_advanced[grid] = set(advanced_names)
        else:
            self._added_advanced.pop(grid, None)

    def _common_parameter_values_for_grids(
        self,
        grids: tuple[tuple[int, int, int], ...],
    ) -> dict[str, Any]:
        effective_maps = [self._effective_params_for_grid(grid) for grid in grids]
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
            common["task"] = infer_task("convergence_kpoint")
        return common

    def _advanced_names_for_grids(self, grids: tuple[tuple[int, int, int], ...]) -> set[str]:
        names: set[str] = set()
        for grid in grids:
            names.update(self._added_advanced.get(grid, set()))
        return names

    def _is_grid_customized(self, grid: tuple[int, int, int]) -> bool:
        return grid in self._parameter_values or grid in self._added_advanced

    def _grid_parameters_locked(self, grids: tuple[tuple[int, int, int], ...]) -> bool:
        for grid in grids:
            runtime = self._runtime_states.get(tuple(grid))
            status = "" if runtime is None else runtime.status.strip().lower()
            if status in _LOCKED_GRID_EDIT_STATUSES:
                return True
        return False

    def _remove_grid_button_clicked(self, button: QToolButton) -> None:
        for row in range(self._table.rowCount()):
            if self._table.cellWidget(row, self._REMOVE_COLUMN) is button:
                self._request_grid_removals((row,))
                return

    def _request_grid_removals(self, rows: tuple[int, ...]) -> None:
        if not rows:
            return

        grids: list[tuple[int, int, int]] = []
        for row in rows:
            grid = self._grid_value_at_row(row, strict=False)
            if grid is not None:
                grids.append(grid)

        if grids:
            self.remove_grid_requested.emit(grids)

    def remove_grids(self, grids: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...]) -> None:
        targets = {tuple(int(component) for component in grid) for grid in grids}
        if not targets:
            return
        for row in reversed(range(self._table.rowCount())):
            grid = self._grid_value_at_row(row, strict=False)
            if grid is None or grid not in targets:
                continue
            self._parameter_values.pop(grid, None)
            self._added_advanced.pop(grid, None)
            self._table.removeRow(row)
        self._prune_configs()
        self._refresh_batch_button_state()
        self._emit_state()

    def set_runtime_states(self, states: Mapping[tuple[int, int, int], SweepPointRunState]) -> None:
        self._runtime_states = {
            (int(key[0]), int(key[1]), int(key[2])): value
            for key, value in states.items()
        }
        self._refresh_all_rows()
        self._refresh_examine_button_state()

    def _refresh_all_rows(self) -> None:
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            self._refresh_row(row)
        self._table.blockSignals(False)

    def _refresh_row(self, row: int) -> None:
        grid = self._grid_value_at_row(row, strict=False)
        runtime = self._runtime_states.get(grid) if grid is not None else None

        status_text = "not generated"
        energy_text = ""
        force_text = ""
        if runtime is not None:
            if runtime.status:
                status_text = runtime.status
            if runtime.total_energy is not None:
                energy_text = f"{runtime.total_energy:.12g}"
            if runtime.max_force is not None:
                force_text = f"{runtime.max_force:.12g}"

        status_normalized = status_text.strip().lower()

        grid_item = self._table.item(row, self._GRID_COLUMN)
        if grid_item is not None:
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if status_normalized not in _LOCKED_GRID_EDIT_STATUSES:
                flags |= Qt.ItemFlag.ItemIsEditable
            grid_item.setFlags(flags)

        status_item = self._table.item(row, self._STATUS_COLUMN)
        if status_item is None:
            status_item = self._make_item(status_text, editable=False)
            self._table.setItem(row, self._STATUS_COLUMN, status_item)
        else:
            status_item.setText(status_text)
        status_item.setForeground(self._status_brush(status_text))

        energy_item = self._table.item(row, self._TOTAL_ENERGY_COLUMN)
        if energy_item is None:
            energy_item = self._make_item(energy_text, editable=False)
            self._table.setItem(row, self._TOTAL_ENERGY_COLUMN, energy_item)
        else:
            energy_item.setText(energy_text)

        force_item = self._table.item(row, self._MAX_FORCE_COLUMN)
        if force_item is None:
            force_item = self._make_item(force_text, editable=False)
            self._table.setItem(row, self._MAX_FORCE_COLUMN, force_item)
        else:
            force_item.setText(force_text)

        configure_button = self._table.cellWidget(row, self._CONFIGURE_COLUMN)
        if isinstance(configure_button, QPushButton):
            customized = grid is not None and self._is_grid_customized(grid)
            locked = status_normalized in _LOCKED_GRID_EDIT_STATUSES
            configure_button.setText("Configured..." if customized or locked else "Configure...")
            configure_button.setToolTip(
                "Inspect the CASTEP parameters used for this grid."
                if locked
                else (
                    "Edit custom parameters for this grid."
                    if customized
                    else "Configure CASTEP parameters for this grid."
                )
            )

    def _refresh_batch_button_state(self) -> None:
        self._batch_configure_button.setEnabled(bool(self._selected_grids()))

    def _refresh_examine_button_state(self) -> None:
        completed_with_results = 0
        for runtime in self._runtime_states.values():
            if runtime.status.strip().lower() != "completed":
                continue
            if runtime.total_energy is None or runtime.max_force is None:
                continue
            completed_with_results += 1

        self._examine_button.setEnabled(completed_with_results >= 2)

    def _prune_configs(self) -> None:
        available = set(self._collect_grids(strict=False))
        stale = [grid for grid in self._parameter_values if grid not in available]
        for grid in stale:
            self._parameter_values.pop(grid, None)
            self._added_advanced.pop(grid, None)

        stale_advanced = [grid for grid in self._added_advanced if grid not in available]
        for grid in stale_advanced:
            self._added_advanced.pop(grid, None)

    def _on_table_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self._GRID_COLUMN:
            return
        self._prune_configs()
        self._refresh_all_rows()
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

    def collect_state(self) -> dict[str, Any]:
        return {
            "mode": "manual",
            "name_prefix": self._name_prefix.text().strip() or "kpoint_convergence",
            "range": dict(self._range_state),
            "kpoint_grids": [list(grid) for grid in self._collect_grids(strict=False)],
            "kpoint_parameter_rows": [
                {
                    "grid": list(grid),
                    "params": dict(self._parameter_values[grid]),
                    "added_advanced": sorted(self._added_advanced.get(grid, set())),
                }
                for grid in sorted(self._parameter_values)
            ],
            "generated_for_cell_id": self._generation_cell_id,
            "generated_for_lattice": self._generation_lattice,
        }

    @staticmethod
    def _coerce_grids(raw_values: object) -> tuple[tuple[int, int, int], ...]:
        values: list[tuple[int, int, int]] = []
        if isinstance(raw_values, list):
            for value in raw_values:
                if isinstance(value, list) and len(value) == 3:
                    try:
                        grid = (int(value[0]), int(value[1]), int(value[2]))
                    except (TypeError, ValueError):
                        continue
                    if grid[0] > 0 and grid[1] > 0 and grid[2] > 0:
                        values.append(grid)
        return tuple(values)

    def set_form_state(self, state: dict[str, Any]) -> None:
        self._restoring_form_state = True
        try:
            self._range_state = {
                "spacing": _DEFAULT_KPOINT_SPACING,
                "replace_existing": False,
            }
            self._parameter_values = {}
            self._added_advanced = {}

            previous_block = self._name_prefix.blockSignals(True)
            self._name_prefix.setText("kpoint_convergence")
            self._name_prefix.blockSignals(previous_block)

            grids: tuple[tuple[int, int, int], ...] = ()
            if state:
                previous_block = self._name_prefix.blockSignals(True)
                self._name_prefix.setText(str(state.get("name_prefix", "kpoint_convergence")))
                self._name_prefix.blockSignals(previous_block)

                range_state = state.get("range", {})
                if isinstance(range_state, dict):
                    try:
                        self._range_state["spacing"] = float(range_state.get("spacing", self._range_state["spacing"]))
                    except (TypeError, ValueError):
                        pass
                    self._range_state["replace_existing"] = bool(
                        range_state.get("replace_existing", self._range_state["replace_existing"])
                    )

                grids = self._coerce_grids(state.get("kpoint_grids", []))

                raw_rows = state.get("kpoint_parameter_rows", [])
                if isinstance(raw_rows, list):
                    for entry in raw_rows:
                        if not isinstance(entry, dict):
                            continue
                        grid_values = self._coerce_grids([entry.get("grid")])
                        if not grid_values:
                            continue
                        grid = grid_values[0]
                        params = self._sanitize_parameter_mapping(
                            {str(name): raw for name, raw in dict(entry.get("params", {})).items()}
                        )
                        advanced_names = {
                            str(name)
                            for name in entry.get("added_advanced", [])
                            if isinstance(name, str)
                        }
                        self._set_grid_parameter_values(grid, params, advanced_names)

                generated_cell = state.get("generated_for_cell_id")
                self._generation_cell_id = str(generated_cell) if isinstance(generated_cell, str) else None

                generated_lattice = state.get("generated_for_lattice")
                if isinstance(generated_lattice, list) and len(generated_lattice) == 3:
                    self._generation_lattice = [list(row) for row in generated_lattice if isinstance(row, list)]
                else:
                    self._generation_lattice = None

            self._set_grid_values(grids)
            self._refresh_all_rows()
            self._refresh_batch_button_state()
            self._refresh_generation_warning()
        finally:
            self._restoring_form_state = False

        self._emit_state()

    def build_config(self) -> ConvergenceConfig | None:
        try:
            grids = self._collect_grids()
        except ValueError:
            QMessageBox.warning(self, "Invalid grid list", "K-point grids must be integer triplets.")
            return None

        if not grids:
            QMessageBox.warning(self, "No grids", "Add one or more k-point grids first.")
            return None

        if len(set(grids)) != len(grids):
            QMessageBox.warning(self, "Duplicate grids", "K-point grids must be unique.")
            return None

        effective = {grid: self._effective_params_for_grid(grid) for grid in grids}
        for grid, values in effective.items():
            task = str(values.get("task", infer_task("convergence_kpoint")))
            errors = validate_gui_values(values, self._specs_by_name, task=task)
            if errors:
                QMessageBox.warning(
                    self,
                    "Invalid parameters",
                    f"Grid {self._format_grid(grid)}:\n" + "\n".join(errors),
                )
                return None

        shared_params = dict(next(iter(effective.values())))
        for candidate in list(effective.values())[1:]:
            shared_params = {
                name: value
                for name, value in shared_params.items()
                if name in candidate and candidate[name] == value
            }
        if "task" not in shared_params:
            shared_params["task"] = infer_task("convergence_kpoint")

        grouped: list[tuple[set[tuple[int, int, int]], dict[str, object]]] = []
        for grid in grids:
            diff = {
                name: value
                for name, value in effective[grid].items()
                if shared_params.get(name) != value
            }
            if not diff:
                continue
            for members, payload in grouped:
                if payload == diff:
                    members.add(grid)
                    break
            else:
                grouped.append(({grid}, diff))

        overrides = {frozenset(members): payload for members, payload in grouped}

        return ConvergenceConfig(
            sweep_parameter="kpoint_mp_grid",
            sweep_values=grids,
            shared_params=shared_params,
            overrides=overrides,
            name_prefix=self._name_prefix.text().strip() or "kpoint_convergence",
        )

    def _emit_config(self) -> None:
        config = self.build_config()
        if config is None:
            return
        self.configure_requested.emit(config)
        self._emit_state()

    def _run_local_selected_grids(self) -> None:
        config = self.build_config()
        if config is None:
            return

        selected = tuple(dict.fromkeys(self._selected_or_current_grids()))
        if not selected:
            QMessageBox.information(self, "No grids selected", "Select one or more k-point rows first.")
            return

        self.configure_requested.emit(config)
        self.submit_local_requested.emit(list(selected))
        self._emit_state()

    def _run_cluster_selected_grids(self) -> None:
        config = self.build_config()
        if config is None:
            return

        selected = tuple(dict.fromkeys(self._selected_or_current_grids()))
        if not selected:
            QMessageBox.information(self, "No grids selected", "Select one or more k-point rows first.")
            return

        self.configure_requested.emit(config)
        self.submit_cluster_requested.emit(list(selected))
        self._emit_state()

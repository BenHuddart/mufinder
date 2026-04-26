"""UEP source-job configuration panel and completed-source launch controls."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
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

from mufinder.app.convergence_display import format_cutoff_energy
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.relaxation_adapter import suggested_grid_for_row
from mufinder.app.theme import status_brush, style_hint_text, style_section_subtitle, style_section_title
from mufinder.app.uep_types import SupercellMatrix, UepConfig, UepRunState, UepSourceRow
from mufinder.castep_io.gui_metadata import (
    get_gui_keyword_specs_for_files,
    normalize_gui_value,
    validate_gui_values,
)
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import BlockSection, ValueWithUnit


_STRUCTURE_CONTROLLED_CELL_KEYWORDS = {
    "lattice_cart",
    "lattice_abc",
    "positions_frac",
    "positions_abs",
}
_KPOINT_SORT_ANCHOR_NAMES = ("basis_precision", "cut_off_energy")
_PREFERRED_DEFAULT_PARAMETER_NAMES = (
    "cut_off_energy",
    "kpoint_mp_grid",
    "kpoint_mp_spacing",
    "kpoints_list",
)
_UNLOCKED_RUNTIME_STATUSES = {"", "not generated", "defined", "ready"}


def _normalize_uep_name_prefix(value: object) -> str:
    prefix = str(value).strip() or "uep"
    return "uep" if prefix == "uep_source" else prefix


def _append_workflow_stage(values: tuple[str, ...], workflow_stage: str) -> tuple[str, ...]:
    if workflow_stage in values:
        return values
    return (*values, workflow_stage)


def _clone_spec_for_uep_workflow(
    spec: KeywordSpec,
    *,
    basic: bool,
    advanced: bool,
) -> KeywordSpec:
    basic_workflows = spec.basic_workflows
    advanced_workflows = spec.advanced_workflows

    if basic:
        basic_workflows = _append_workflow_stage(basic_workflows, "uep")
    if advanced:
        advanced_workflows = _append_workflow_stage(advanced_workflows, "uep")

    return replace(
        spec,
        basic_workflows=basic_workflows,
        advanced_workflows=advanced_workflows,
    )


def _reorder_specs(specs: list[KeywordSpec]) -> list[KeywordSpec]:
    kpoint_name = "kpoint_mp_grid"
    kpoint_index = next((index for index, spec in enumerate(specs) if spec.name == kpoint_name), None)
    if kpoint_index is None:
        return specs

    anchor_indices = [index for index, spec in enumerate(specs) if spec.name in _KPOINT_SORT_ANCHOR_NAMES]
    if not anchor_indices:
        return specs

    target_index = max(anchor_indices) + 1
    if kpoint_index == target_index:
        return specs

    spec = specs.pop(kpoint_index)
    if kpoint_index < target_index:
        target_index -= 1
    specs.insert(target_index, spec)
    return specs


class UEPWidget(QWidget):
    """UI for configuring UEP source CASTEP jobs and launching analysis for completed rows."""

    _LABEL_COLUMN = 0
    _SUPERCELL_COLUMN = 1
    _STATUS_COLUMN = 2
    _TOTAL_ENERGY_COLUMN = 3
    _POTENTIAL_GRID_COLUMN = 4
    _CONFIGURE_COLUMN = 5
    _WIZARD_COLUMN = 6
    _REMOVE_COLUMN = 7

    configure_requested = Signal(object)
    submit_local_requested = Signal(object)
    submit_cluster_requested = Signal(object)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    remove_requested = Signal(object)
    row_supercell_edit_requested = Signal(object)
    source_structure_selection_requested = Signal(object)
    wizard_requested = Signal(object)
    form_state_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._restoring_form_state = False
        self._suppress_state_emission = False
        self._next_row_id = 1

        self._host_lattice: list[list[float]] | None = None
        self._reference_lattice: list[list[float]] | None = None
        self._reference_grid: tuple[int, int, int] | None = None

        self._row_supercells: dict[int, SupercellMatrix | None] = {}
        self._row_parameter_values: dict[int, dict[str, object]] = {}
        self._row_added_advanced: dict[int, set[str]] = {}
        self._runtime_states: dict[str, UepRunState] = {}
        self._supercell_options: list[tuple[str, SupercellMatrix]] = []
        self._source_structure_options: list[tuple[str, str]] = []
        self._updating_source_structure = False

        self._name_prefix = QLineEdit("uep")
        self._project_default_cutoff = QLabel("Not set")
        self._project_default_kpoint = QLabel("Not set")
        self._source_structure_summary = QLabel("Using the host structure.")
        self._source_structure_summary.setWordWrap(True)
        style_hint_text(self._source_structure_summary)
        self._source_structure_combo = QComboBox()
        self._kpoint_message = QLabel("")
        self._kpoint_message.setWordWrap(True)
        style_hint_text(self._kpoint_message)
        self._kpoint_message.hide()

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            [
                "Label",
                "Supercell",
                "Status",
                "Total Energy (eV)",
                "3D Potential",
                "Parameters",
                "Analysis",
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
        self._table.verticalHeader().setDefaultSectionSize(32)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self._LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._SUPERCELL_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._STATUS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._TOTAL_ENERGY_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._POTENTIAL_GRID_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._CONFIGURE_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._WIZARD_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._REMOVE_COLUMN, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._REMOVE_COLUMN, 34)

        self._add_row_button = QPushButton("Add Job")
        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._batch_configure_button.setEnabled(False)
        self._run_local_button = QPushButton("Run Local Calculations")
        self._run_cluster_button = QPushButton("Run on Cluster")

        self._specs = self._build_parameter_specs()
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        self._default_params = self._default_shared_params()

        self._build_layout()
        self._connect_signals()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(3)

        title = QLabel("UEP")
        style_section_title(title)
        subtitle = QLabel(
            "Create CASTEP single-point source jobs that write formatted densities for "
            "unperturbed electrostatic potential analysis."
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        title_column.addWidget(title)
        title_column.addWidget(subtitle)

        defaults_group = QGroupBox("Project Defaults")
        defaults_layout = QVBoxLayout(defaults_group)
        defaults_layout.setContentsMargins(10, 8, 10, 8)
        cutoff_row = QHBoxLayout()
        cutoff_row.addWidget(QLabel("Cutoff"))
        cutoff_row.addWidget(self._project_default_cutoff)
        cutoff_row.addStretch(1)
        defaults_layout.addLayout(cutoff_row)
        kpoint_row = QHBoxLayout()
        kpoint_row.addWidget(QLabel("K-point grid"))
        kpoint_row.addWidget(self._project_default_kpoint)
        kpoint_row.addStretch(1)
        defaults_layout.addLayout(kpoint_row)

        header_row.addLayout(title_column, 1)
        header_row.addWidget(defaults_group)
        layout.addLayout(header_row)

        source_group = QGroupBox("Source Structure")
        source_layout = QVBoxLayout(source_group)
        source_layout.setContentsMargins(10, 8, 10, 8)
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Structure"))
        selector_row.addWidget(self._source_structure_combo, 1)
        source_layout.addLayout(selector_row)
        source_layout.addWidget(self._source_structure_summary)
        layout.addWidget(source_group)

        layout.addWidget(self._kpoint_message)

        table_group = QGroupBox("UEP Source Jobs")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 8, 10, 8)
        table_layout.addWidget(self._table)

        control_row = QHBoxLayout()
        control_row.addWidget(self._add_row_button)
        control_row.addWidget(self._batch_configure_button)
        control_row.addStretch(1)
        control_row.addWidget(self._run_local_button)
        control_row.addWidget(self._run_cluster_button)
        table_layout.addLayout(control_row)
        layout.addWidget(table_group, 1)

    def _connect_signals(self) -> None:
        self._add_row_button.clicked.connect(self._add_row)
        self._batch_configure_button.clicked.connect(self._on_batch_configure_clicked)
        self._run_local_button.clicked.connect(self._emit_submit_local)
        self._run_cluster_button.clicked.connect(self._emit_submit_cluster)
        self._table.itemSelectionChanged.connect(self._refresh_batch_button_state)
        self._name_prefix.textChanged.connect(lambda *_: self._emit_state())
        self._source_structure_combo.currentIndexChanged.connect(self._on_source_structure_changed)

    def _build_parameter_specs(self) -> list[KeywordSpec]:
        specs = get_gui_keyword_specs_for_files(
            files=("param", "cell"),
            workflow_stage="uep",
            include_advanced=True,
        )
        adjusted: list[KeywordSpec] = []
        for spec in specs:
            if spec.file == "cell" and spec.name in _STRUCTURE_CONTROLLED_CELL_KEYWORDS:
                continue
            if spec.name == "write_formatted_density":
                adjusted.append(_clone_spec_for_uep_workflow(spec, basic=True, advanced=True))
                continue
            adjusted.append(spec)
        return _reorder_specs(adjusted)

    def _default_shared_params(self) -> dict[str, object]:
        defaults: dict[str, object] = {
            "task": "SINGLEPOINT",
            "write_formatted_density": True,
        }
        return defaults

    def set_project_defaults(
        self,
        *,
        cutoff_energy: float | None,
        kpoint_grid: tuple[int, int, int] | None,
        kpoint_reference_lattice: list[list[float]] | None = None,
    ) -> None:
        with self._suspend_state_emission():
            self._reference_grid = kpoint_grid
            self._reference_lattice = [list(row) for row in kpoint_reference_lattice] if kpoint_reference_lattice else None
            self._project_default_cutoff.setText(format_cutoff_energy(cutoff_energy))
            self._project_default_kpoint.setText(
                " x ".join(str(int(value)) for value in kpoint_grid) if kpoint_grid is not None else "Not set"
            )
            if cutoff_energy is not None:
                self._default_params["cut_off_energy"] = ValueWithUnit(float(cutoff_energy), "eV")
            else:
                self._default_params.pop("cut_off_energy", None)
            if kpoint_grid is not None:
                self._default_params["kpoint_mp_grid"] = tuple(int(value) for value in kpoint_grid)
            else:
                self._default_params.pop("kpoint_mp_grid", None)
            self._refresh_table_rows()

    def set_host_lattice(self, lattice_vectors: list[list[float]] | None) -> None:
        self._host_lattice = [list(row) for row in lattice_vectors] if lattice_vectors is not None else None
        self._refresh_table_rows()

    def set_source_structure_options(
        self,
        options: list[dict[str, object]],
        *,
        selected_id: str | None = None,
        summary: str | None = None,
    ) -> None:
        with self._suspend_state_emission():
            self._source_structure_options = []
            self._updating_source_structure = True
            try:
                self._source_structure_combo.clear()
                for option in options:
                    identifier = str(option.get("id", "")).strip()
                    label = str(option.get("label", "")).strip()
                    if not identifier or not label:
                        continue
                    self._source_structure_options.append((identifier, label))
                    self._source_structure_combo.addItem(label, identifier)
                index = 0
                if selected_id:
                    found = self._source_structure_combo.findData(selected_id)
                    if found >= 0:
                        index = found
                if self._source_structure_combo.count():
                    self._source_structure_combo.setCurrentIndex(index)
            finally:
                self._updating_source_structure = False

            self._source_structure_combo.setEnabled(self._source_structure_combo.count() > 0)
            self._source_structure_summary.setText(summary or "Using the host structure.")

    def selected_source_structure_id(self) -> str | None:
        data = self._source_structure_combo.currentData()
        if isinstance(data, str) and data.strip():
            return data.strip()
        return None

    def set_runtime_states(self, states: Mapping[str, UepRunState]) -> None:
        self._runtime_states = {str(label): state for label, state in states.items()}
        self._refresh_table_rows()

    def set_supercell_options(self, options: list[dict[str, object]]) -> None:
        parsed: list[tuple[str, SupercellMatrix]] = []
        for option in options:
            label = str(option.get("label", "")).strip()
            matrix_payload = option.get("matrix")
            if not label or not isinstance(matrix_payload, list) or len(matrix_payload) != 3:
                continue
            try:
                matrix: SupercellMatrix = (
                    (int(matrix_payload[0][0]), int(matrix_payload[0][1]), int(matrix_payload[0][2])),
                    (int(matrix_payload[1][0]), int(matrix_payload[1][1]), int(matrix_payload[1][2])),
                    (int(matrix_payload[2][0]), int(matrix_payload[2][1]), int(matrix_payload[2][2])),
                )
            except (TypeError, ValueError, IndexError):
                continue
            parsed.append((label, matrix))
        self._supercell_options = parsed
        self._refresh_table_rows()

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
            if name in self._specs_by_name:
                cleaned[name] = self._restore_structured_value(name, value)
        return cleaned

    def row_supercell_matrix(self, row_id: int) -> SupercellMatrix | None:
        return self._row_supercells.get(int(row_id))

    def set_row_supercell(self, row_id: int, matrix: SupercellMatrix | None) -> None:
        self._row_supercells[int(row_id)] = matrix
        self._refresh_table_rows()
        self._emit_state()

    def _refresh_batch_button_state(self) -> None:
        self._batch_configure_button.setEnabled(bool(self._selected_row_ids()))

    def _selected_row_ids(self) -> tuple[int, ...]:
        row_ids: list[int] = []
        for index in sorted({item.row() for item in self._table.selectedItems()}):
            row_id = self._row_id_at(index)
            if row_id is not None:
                row_ids.append(row_id)
        return tuple(row_ids)

    def _row_id_at(self, table_row: int) -> int | None:
        item = self._table.item(table_row, self._LABEL_COLUMN)
        if item is None:
            return None
        raw = item.data(Qt.ItemDataRole.UserRole)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def _new_row_label(self, row_number: int | None = None) -> str:
        number = self._table.rowCount() + 1 if row_number is None else int(row_number)
        return f"source_{number:02d}"

    def _add_row(self) -> None:
        row_id = self._next_row_id
        self._next_row_id += 1
        self._row_supercells.setdefault(row_id, None)
        self._row_parameter_values.setdefault(row_id, {})
        self._row_added_advanced.setdefault(row_id, set())
        self._insert_row(row_id=row_id, label=self._new_row_label())
        self._emit_state()

    def _insert_row(self, *, row_id: int, label: str) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        label_item = QTableWidgetItem(label)
        label_item.setData(Qt.ItemDataRole.UserRole, row_id)
        self._table.setItem(row, self._LABEL_COLUMN, label_item)

        supercell_combo = QComboBox(self._table)
        supercell_combo.addItem("Unit cell", None)
        for option_label, matrix in self._supercell_options:
            supercell_combo.addItem(option_label, matrix)
        supercell_combo.addItem("Manage Supercells...", "__create__")
        supercell_combo.currentIndexChanged.connect(
            lambda _index, row_id=row_id, combo=supercell_combo: self._on_supercell_changed(row_id, combo)
        )
        self._table.setCellWidget(row, self._SUPERCELL_COLUMN, supercell_combo)

        self._table.setItem(row, self._STATUS_COLUMN, QTableWidgetItem("ready"))
        self._table.setItem(row, self._TOTAL_ENERGY_COLUMN, QTableWidgetItem(""))
        self._table.setItem(row, self._POTENTIAL_GRID_COLUMN, QTableWidgetItem(""))

        configure_button = QPushButton("Configure...")
        configure_button.clicked.connect(lambda _checked=False, row_id=row_id: self._configure_row(row_id))
        self._table.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

        wizard_button = QPushButton("Open Wizard")
        wizard_button.clicked.connect(lambda _checked=False, row_id=row_id: self._emit_wizard_for_row(row_id))
        self._table.setCellWidget(row, self._WIZARD_COLUMN, wizard_button)

        remove_button = QToolButton(self._table)
        remove_button.setText("×")
        remove_button.clicked.connect(lambda _checked=False, row_id=row_id: self._remove_row(row_id))
        self._table.setCellWidget(row, self._REMOVE_COLUMN, remove_button)

        self._apply_row_state(row_id)

    def _row_index_for_id(self, row_id: int) -> int | None:
        for row in range(self._table.rowCount()):
            if self._row_id_at(row) == row_id:
                return row
        return None

    def _on_supercell_changed(self, row_id: int, combo: QComboBox) -> None:
        data = combo.currentData()
        if data == "__create__":
            self.row_supercell_edit_requested.emit(row_id)
            current = self._row_supercells.get(row_id)
            self._set_combo_to_supercell(combo, current)
            return
        self._row_supercells[row_id] = data if isinstance(data, tuple) else None
        self._refresh_table_rows()
        self._emit_state()

    @staticmethod
    def _runtime_controls_locked(status: str) -> bool:
        return str(status).strip().casefold() not in _UNLOCKED_RUNTIME_STATUSES

    def _runtime_state_for_row(self, row_id: int) -> UepRunState:
        row = self._row_index_for_id(row_id)
        if row is None:
            return UepRunState(status="ready")
        item = self._table.item(row, self._LABEL_COLUMN)
        label = "" if item is None else item.text().strip()
        return self._runtime_states.get(label, UepRunState(status="ready"))

    def _set_combo_to_supercell(self, combo: QComboBox, matrix: SupercellMatrix | None) -> None:
        for index in range(combo.count()):
            if combo.itemData(index) == matrix:
                combo.setCurrentIndex(index)
                return
        combo.setCurrentIndex(0)

    def _configure_row(self, row_id: int) -> None:
        read_only = self._runtime_controls_locked(self._runtime_state_for_row(row_id).status)
        dialog_kwargs = {
            "specs": self._specs,
            "workflow_stage": "uep",
            "locked_values": {
                "task": "SINGLEPOINT",
                "write_formatted_density": "TRUE (required for UEP source analysis)",
            },
            "intro_text": (
                "UEP source jobs are CASTEP single-point calculations. "
                "Formatted density output stays enabled because the UEP analysis requires `.den_fmt`."
            ),
            "parent": self,
        }
        if read_only:
            dialog_kwargs["read_only"] = True
        dialog = ParameterDialog(**dialog_kwargs)
        if hasattr(dialog, "setWindowTitle"):
            dialog.setWindowTitle("Inspect UEP Parameters" if read_only else "Configure UEP Parameters")
        values = self._effective_params_for_row(row_id)
        dialog.set_values(values)
        dialog.set_advanced_names(self._row_added_advanced.get(row_id, set()))
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        if read_only:
            return

        resolved = self._sanitize_parameter_mapping(dialog.collect_values())
        resolved["write_formatted_density"] = True
        resolved["task"] = "SINGLEPOINT"
        self._row_parameter_values[row_id] = resolved
        self._row_added_advanced[row_id] = set(dialog.advanced_names())
        self._emit_state()

    def _on_batch_configure_clicked(self) -> None:
        row_ids = self._selected_row_ids()
        if not row_ids:
            return
        values = self._common_params_for_rows(row_ids)
        read_only = any(
            self._runtime_controls_locked(self._runtime_state_for_row(row_id).status)
            for row_id in row_ids
        )
        dialog_kwargs = {
            "specs": self._specs,
            "workflow_stage": "uep",
            "locked_values": {
                "task": "SINGLEPOINT",
                "write_formatted_density": "TRUE (required for UEP source analysis)",
            },
            "parent": self,
        }
        if read_only:
            dialog_kwargs["read_only"] = True
        dialog = ParameterDialog(**dialog_kwargs)
        if hasattr(dialog, "setWindowTitle"):
            dialog.setWindowTitle("Inspect UEP Parameters" if read_only else "Configure UEP Parameters")
        dialog.set_values(values)
        names: set[str] = set()
        for row_id in row_ids:
            names.update(self._row_added_advanced.get(row_id, set()))
        dialog.set_advanced_names(names)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        if read_only:
            return
        collected = self._sanitize_parameter_mapping(dialog.collect_values())
        for row_id in row_ids:
            row_values = dict(self._effective_params_for_row(row_id))
            row_values.update(collected)
            row_values["write_formatted_density"] = True
            row_values["task"] = "SINGLEPOINT"
            self._row_parameter_values[row_id] = row_values
            self._row_added_advanced.setdefault(row_id, set()).update(dialog.advanced_names())
        self._emit_state()

    def _common_params_for_rows(self, row_ids: tuple[int, ...]) -> dict[str, object]:
        values_by_row = [self._effective_params_for_row(row_id) for row_id in row_ids]
        if not values_by_row:
            return {}
        names = set(values_by_row[0])
        for values in values_by_row[1:]:
            names &= set(values)
        common: dict[str, object] = {}
        for name in names:
            first = values_by_row[0][name]
            if all(values.get(name) == first for values in values_by_row[1:]):
                common[name] = first
        return common

    def _effective_params_for_row(self, row_id: int) -> dict[str, object]:
        effective = dict(self._default_params)
        effective.update(self._row_parameter_values.get(row_id, {}))
        if (
            self._host_lattice is not None
            and (matrix := self._row_supercells.get(row_id)) is not None
            and self._reference_grid is not None
        ):
            suggested = suggested_grid_for_row(
                host_lattice=self._host_lattice,
                row_matrix=matrix,
                reference_lattice=self._reference_lattice,
                reference_grid=self._reference_grid,
            )
            if suggested is not None and "kpoint_mp_grid" not in self._row_parameter_values.get(row_id, {}):
                effective["kpoint_mp_grid"] = suggested
        effective["task"] = "SINGLEPOINT"
        effective["write_formatted_density"] = True
        return effective

    def _remove_row(self, row_id: int) -> None:
        row = self._row_index_for_id(row_id)
        if row is None:
            return
        label_item = self._table.item(row, self._LABEL_COLUMN)
        label = label_item.text().strip() if label_item is not None else ""
        self._table.removeRow(row)
        self._row_supercells.pop(row_id, None)
        self._row_parameter_values.pop(row_id, None)
        self._row_added_advanced.pop(row_id, None)
        self.remove_requested.emit([label] if label else [])
        self._emit_state()

    def _emit_submit_local(self) -> None:
        config = self._build_config_or_warn()
        if config is None:
            return
        labels = self._selected_or_all_labels()
        if not labels:
            QMessageBox.information(self, "No jobs selected", "Select one or more rows first.")
            return
        self.configure_requested.emit(config)
        self.submit_local_requested.emit(labels)

    def _emit_submit_cluster(self) -> None:
        config = self._build_config_or_warn()
        if config is None:
            return
        labels = self._selected_or_all_labels()
        if not labels:
            QMessageBox.information(self, "No jobs selected", "Select one or more rows first.")
            return
        self.configure_requested.emit(config)
        self.submit_cluster_requested.emit(labels)

    def _selected_or_all_labels(self) -> list[str]:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return []
        selected_rows = [index.row() for index in selection_model.selectedRows()]
        if not selected_rows:
            selected_rows = list(range(self._table.rowCount()))
        labels: list[str] = []
        for row in selected_rows:
            item = self._table.item(row, self._LABEL_COLUMN)
            if item is None:
                continue
            label = item.text().strip()
            if label:
                labels.append(label)
        return labels

    def _build_config_or_warn(self) -> UepConfig | None:
        rows: list[UepSourceRow] = []
        seen: set[str] = set()
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._LABEL_COLUMN)
            row_id = self._row_id_at(row)
            if item is None or row_id is None:
                continue
            label = item.text().strip()
            if not label:
                QMessageBox.warning(self, "Missing label", "Every UEP row needs a label.")
                return None
            if label in seen:
                QMessageBox.warning(self, "Duplicate label", f"Duplicate UEP row label '{label}'.")
                return None
            seen.add(label)

            values = self._effective_params_for_row(row_id)
            try:
                errors = validate_gui_values(values, self._specs_by_name, task="SINGLEPOINT")
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid UEP parameters", str(exc))
                return None
            if errors:
                QMessageBox.warning(self, "Invalid UEP parameters", "\n".join(errors))
                return None

            rows.append(
                UepSourceRow(
                    label=label,
                    supercell_matrix=self._row_supercells.get(row_id),
                    parameter_values=values,
                    added_advanced=tuple(sorted(self._row_added_advanced.get(row_id, set()))),
                )
            )

        if not rows:
            QMessageBox.warning(self, "No jobs configured", "Add at least one UEP source job.")
            return None

        return UepConfig(
            name_prefix=_normalize_uep_name_prefix(self._name_prefix.text()),
            rows=tuple(rows),
        )

    def _refresh_table_rows(self) -> None:
        for row in range(self._table.rowCount()):
            row_id = self._row_id_at(row)
            if row_id is None:
                continue
            combo = self._table.cellWidget(row, self._SUPERCELL_COLUMN)
            if isinstance(combo, QComboBox):
                current = self._row_supercells.get(row_id)
                combo.blockSignals(True)
                labels = [combo.itemText(index) for index in range(combo.count())]
                if labels != ["Unit cell", *[label for label, _ in self._supercell_options], "Manage Supercells..."]:
                    combo.clear()
                    combo.addItem("Unit cell", None)
                    for option_label, matrix in self._supercell_options:
                        combo.addItem(option_label, matrix)
                    combo.addItem("Manage Supercells...", "__create__")
                self._set_combo_to_supercell(combo, current)
                combo.blockSignals(False)
            self._apply_row_state(row_id)

    def _apply_row_state(self, row_id: int) -> None:
        row = self._row_index_for_id(row_id)
        if row is None:
            return
        label_item = self._table.item(row, self._LABEL_COLUMN)
        if label_item is None:
            return
        label = label_item.text().strip()
        state = self._runtime_states.get(label, UepRunState(status="ready"))
        status_item = self._table.item(row, self._STATUS_COLUMN)
        if status_item is None:
            status_item = QTableWidgetItem("")
            self._table.setItem(row, self._STATUS_COLUMN, status_item)
        status_text = state.status or "ready"
        locked = self._runtime_controls_locked(status_text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if not locked:
            flags |= Qt.ItemFlag.ItemIsEditable
            label_item.setToolTip("")
        else:
            label_item.setToolTip("Job label is locked once this row has been queued for execution.")
        label_item.setFlags(flags)
        supercell_combo = self._table.cellWidget(row, self._SUPERCELL_COLUMN)
        if isinstance(supercell_combo, QComboBox):
            supercell_combo.setEnabled(not locked)
        remove_button = self._table.cellWidget(row, self._REMOVE_COLUMN)
        if isinstance(remove_button, QToolButton):
            remove_button.setEnabled(True)
        status_item.setText(status_text)
        status_item.setForeground(status_brush(status_text))

        total_energy = "" if state.total_energy is None else f"{state.total_energy:.6f}"
        if state.potential_grid_available is True:
            potential_text = "Saved"
        elif status_text == "completed":
            potential_text = "Not calculated"
        else:
            potential_text = ""
        self._table.item(row, self._TOTAL_ENERGY_COLUMN).setText(total_energy)
        self._table.item(row, self._POTENTIAL_GRID_COLUMN).setText(potential_text)

        configure_button = self._table.cellWidget(row, self._CONFIGURE_COLUMN)
        if isinstance(configure_button, QPushButton):
            customized = bool(self._row_parameter_values.get(row_id, {}))
            configure_button.setText("Configured..." if customized or locked else "Configure...")
            configure_button.setToolTip(
                "Inspect the CASTEP parameters used for this UEP source job."
                if locked
                else (
                    "Edit custom CASTEP parameters for this UEP source job."
                    if customized
                    else "Configure CASTEP parameters for this UEP source job."
                )
            )

        wizard_button = self._table.cellWidget(row, self._WIZARD_COLUMN)
        if isinstance(wizard_button, QPushButton):
            wizard_button.setEnabled(status_text == "completed" and bool(state.density_available))
            wizard_button.setToolTip(
                ""
                if wizard_button.isEnabled()
                else "Complete the source job and ensure `.den_fmt`/`.castep` outputs are available first."
            )

    def _emit_wizard_for_row(self, row_id: int) -> None:
        row = self._row_index_for_id(row_id)
        if row is None:
            return
        item = self._table.item(row, self._LABEL_COLUMN)
        if item is None:
            return
        self.wizard_requested.emit(item.text().strip())

    def collect_state(self) -> dict[str, Any]:
        rows: list[dict[str, object]] = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._LABEL_COLUMN)
            row_id = self._row_id_at(row)
            if item is None or row_id is None:
                continue
            matrix = self._row_supercells.get(row_id)
            rows.append(
                {
                    "label": item.text().strip(),
                    "supercell_matrix": [list(vector) for vector in matrix] if matrix is not None else None,
                    "parameter_values": dict(self._row_parameter_values.get(row_id, {})),
                    "added_advanced": sorted(self._row_added_advanced.get(row_id, set())),
                }
            )
        return {
            "name_prefix": _normalize_uep_name_prefix(self._name_prefix.text()),
            "rows": rows,
            "selected_source_structure_id": self.selected_source_structure_id(),
        }

    def set_form_state(self, state: dict[str, Any]) -> None:
        self._restoring_form_state = True
        try:
            self._table.setRowCount(0)
            self._row_supercells.clear()
            self._row_parameter_values.clear()
            self._row_added_advanced.clear()
            self._next_row_id = 1

            self._name_prefix.setText(_normalize_uep_name_prefix(state.get("name_prefix", "uep")))
            rows_payload = state.get("rows", [])
            if not isinstance(rows_payload, list):
                rows_payload = []

            for entry in rows_payload:
                if not isinstance(entry, dict):
                    continue
                row_id = self._next_row_id
                self._next_row_id += 1
                default_label = self._new_row_label(row_id)
                label = str(entry.get("label", default_label)).strip() or default_label
                matrix_payload = entry.get("supercell_matrix")
                matrix = None
                if isinstance(matrix_payload, list) and len(matrix_payload) == 3:
                    try:
                        matrix = (
                            (int(matrix_payload[0][0]), int(matrix_payload[0][1]), int(matrix_payload[0][2])),
                            (int(matrix_payload[1][0]), int(matrix_payload[1][1]), int(matrix_payload[1][2])),
                            (int(matrix_payload[2][0]), int(matrix_payload[2][1]), int(matrix_payload[2][2])),
                        )
                    except (TypeError, ValueError, IndexError):
                        matrix = None
                self._row_supercells[row_id] = matrix
                params = entry.get("parameter_values", {})
                if isinstance(params, dict):
                    self._row_parameter_values[row_id] = self._sanitize_parameter_mapping(
                        {str(name): raw for name, raw in params.items()}
                    )
                else:
                    self._row_parameter_values[row_id] = {}
                advanced = entry.get("added_advanced", [])
                self._row_added_advanced[row_id] = {str(name) for name in advanced} if isinstance(advanced, list) else set()
                self._insert_row(row_id=row_id, label=label)

            selected_source_structure_id = state.get("selected_source_structure_id")
            if isinstance(selected_source_structure_id, str) and selected_source_structure_id:
                index = self._source_structure_combo.findData(selected_source_structure_id)
                if index >= 0:
                    self._source_structure_combo.setCurrentIndex(index)
        finally:
            self._restoring_form_state = False
        self._refresh_table_rows()
        self._emit_state()

    def _on_source_structure_changed(self) -> None:
        if self._updating_source_structure:
            return
        selected = self.selected_source_structure_id()
        if selected is not None:
            self.source_structure_selection_requested.emit(selected)
        self._emit_state()

    @contextmanager
    def _suspend_state_emission(self) -> Generator[None, None, None]:
        previous = self._suppress_state_emission
        self._suppress_state_emission = True
        try:
            yield
        finally:
            self._suppress_state_emission = previous

    def _emit_state(self) -> None:
        if self._restoring_form_state or self._suppress_state_emission:
            return
        self.form_state_changed.emit(self.collect_state())

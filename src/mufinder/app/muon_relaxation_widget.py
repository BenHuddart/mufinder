"""Muon structural-relaxation configuration panel."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import UUID

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.convergence_display import format_cutoff_energy, format_kpoint_grid
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.relaxation_adapter import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    muon_charge_for_state,
    muon_charge_state_label,
    suggested_grid_for_row,
)
from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow, RelaxationRunState, SupercellMatrix
from mufinder.app.theme import current_palette, status_brush, style_hint_text, style_section_subtitle, style_section_title
from mufinder.castep_io.gui_metadata import (
    get_gui_keyword_specs_for_files,
    infer_task,
    keyword_visible_in_workflow,
    normalize_gui_value,
    validate_gui_values,
)
from mufinder.castep_io.schema import KeywordSpec, get_keyword_spec
from mufinder.castep_io.types import BlockSection, ValueWithUnit


_WORKFLOW_STAGE = "muon_relaxation"
_EXCLUDED_PARAMETER_NAMES = {"task", "fix_all_cell", "write_cell_structure", "charge"}
_EXTRA_GEOMETRY_PARAMETER_NAMES = (
    "geom_disp_tol",
    "geom_energy_tol",
    "geom_force_tol",
    "geom_frequency_est",
    "geom_max_iter",
    "geom_stress_tol",
)
_PREFERRED_DEFAULT_PARAMETER_NAMES = (
    "cut_off_energy",
    "kpoint_mp_grid",
    "kpoint_mp_spacing",
    "kpoints_list",
)
_BASIC_BASIS_CONTROL_NAMES = {"basis_precision", "cut_off_energy"}
_UNLOCKED_RUNTIME_STATUSES = {"", "not generated", "defined", "ready"}


def _append_workflow_stage(values: tuple[str, ...], workflow_stage: str) -> tuple[str, ...]:
    if workflow_stage in values:
        return values
    return (*values, workflow_stage)


def _clone_spec_for_muon_workflow(spec: KeywordSpec) -> KeywordSpec:
    if spec.name not in _BASIC_BASIS_CONTROL_NAMES:
        return spec
    return replace(
        spec,
        basic_workflows=_append_workflow_stage(spec.basic_workflows, _WORKFLOW_STAGE),
        advanced_workflows=_append_workflow_stage(spec.advanced_workflows, _WORKFLOW_STAGE),
    )


class _GroupedCoordinateHeader(QHeaderView):
    """Horizontal header that groups coordinate components under one label."""

    def __init__(
        self,
        groups: tuple[tuple[int, int, str], ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._groups = groups
        self._coordinate_columns = {
            column
            for start, end, _label in groups
            for column in range(start, end + 1)
        }
        self.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

    def sizeHint(self) -> QSize:
        size = super().sizeHint()
        size.setHeight(max(size.height(), 46))
        return size

    def sectionSizeFromContents(self, logical_index: int) -> QSize:
        size = super().sectionSizeFromContents(logical_index)
        if logical_index in self._coordinate_columns:
            size.setHeight(max(size.height(), 46))
        return size

    def paintSection(self, painter: QPainter, rect: QRect, logical_index: int) -> None:
        if logical_index not in self._coordinate_columns:
            super().paintSection(painter, rect, logical_index)
            return
        if not rect.isValid():
            return

        palette = current_palette()
        painter.save()
        painter.fillRect(rect, QColor(palette.bg_surface_alt))
        painter.setPen(QColor(palette.border))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.setPen(QColor(palette.text_primary))

        text = self.model().headerData(logical_index, self.orientation(), Qt.ItemDataRole.DisplayRole)
        label_rect = QRect(rect)
        label_rect.setTop(rect.top() + rect.height() // 2)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, "" if text is None else str(text))
        painter.restore()

    def paintEvent(self, event) -> None:  # noqa: ANN001 - Qt override signature.
        super().paintEvent(event)
        painter = QPainter(self.viewport())
        palette = current_palette()
        painter.setPen(QColor(palette.text_primary))
        for start, end, label in self._groups:
            left = self.sectionViewportPosition(start)
            right = self.sectionViewportPosition(end) + self.sectionSize(end)
            if right <= 0 or left >= self.viewport().width():
                continue
            visible_left = max(left, 0)
            group_rect = QRect(
                visible_left,
                0,
                min(right, self.viewport().width()) - visible_left,
                self.height() // 2,
            )
            painter.drawText(group_rect, Qt.AlignmentFlag.AlignCenter, label)
            painter.setPen(QColor(palette.border))
            underline_y = group_rect.bottom()
            painter.drawLine(group_rect.left() + 6, underline_y, group_rect.right() - 6, underline_y)
            painter.setPen(QColor(palette.text_primary))


class MuonRelaxationWidget(QWidget):
    """UI for staging and running fixed-cell muon relaxation jobs."""

    _LABEL_COLUMN = 0
    _SUPERCELL_COLUMN = 1
    _CHARGE_COLUMN = 2
    _INITIAL_FX_COLUMN = 3
    _INITIAL_FY_COLUMN = 4
    _INITIAL_FZ_COLUMN = 5
    _FINAL_FX_COLUMN = 6
    _FINAL_FY_COLUMN = 7
    _FINAL_FZ_COLUMN = 8
    _TOTAL_ENERGY_COLUMN = 9
    _STATUS_COLUMN = 10
    _CONFIGURE_COLUMN = 11
    _REMOVE_COLUMN = 12

    configure_requested = Signal(object)
    submit_local_requested = Signal(object)
    submit_cluster_requested = Signal(object)
    row_supercell_edit_requested = Signal(object)
    remove_requested = Signal(object)
    form_state_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._restoring_form_state = False
        self._next_row_id = 1
        self._host_lattice: list[list[float]] | None = None
        self._reference_lattice: list[list[float]] | None = None
        self._reference_grid: tuple[int, int, int] | None = None

        self._row_site_ids: dict[int, UUID] = {}
        self._row_positions: dict[int, tuple[float, float, float]] = {}
        self._row_supercells: dict[int, SupercellMatrix | None] = {}
        self._row_parameter_values: dict[int, dict[str, object]] = {}
        self._row_added_advanced: dict[int, set[str]] = {}
        self._runtime_states: dict[str, RelaxationRunState] = {}
        self._supercell_options: list[tuple[str, SupercellMatrix]] = []

        self._project_default_cutoff = QLabel("Not set")
        self._project_default_kpoint = QLabel("Not set")
        self._kpoint_message = QLabel("")
        self._kpoint_message.setWordWrap(True)
        style_hint_text(self._kpoint_message)
        self._kpoint_message.hide()

        self._table = QTableWidget(0, 13)
        self._table.setHorizontalHeader(
            _GroupedCoordinateHeader(
                (
                    (self._INITIAL_FX_COLUMN, self._INITIAL_FZ_COLUMN, "Initial"),
                    (self._FINAL_FX_COLUMN, self._FINAL_FZ_COLUMN, "Final"),
                ),
                self._table,
            )
        )
        self._table.setHorizontalHeaderLabels(
            [
                "Label",
                "Supercell",
                "Charge",
                "fX",
                "fY",
                "fZ",
                "fX",
                "fY",
                "fZ",
                "Total Energy\n(eV)",
                "Status",
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
        self._table.verticalHeader().setDefaultSectionSize(32)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(self._LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._SUPERCELL_COLUMN, QHeaderView.ResizeMode.Stretch)
        for column in (
            self._INITIAL_FX_COLUMN,
            self._INITIAL_FY_COLUMN,
            self._INITIAL_FZ_COLUMN,
            self._FINAL_FX_COLUMN,
            self._FINAL_FY_COLUMN,
            self._FINAL_FZ_COLUMN,
            self._TOTAL_ENERGY_COLUMN,
            self._STATUS_COLUMN,
            self._CONFIGURE_COLUMN,
        ):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._CHARGE_COLUMN, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._CHARGE_COLUMN, self._charge_column_width())
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setSectionResizeMode(self._REMOVE_COLUMN, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._REMOVE_COLUMN, 34)

        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._batch_configure_button.setEnabled(False)
        self._run_local_button = QPushButton("Run Local Calculations")
        self._run_cluster_button = QPushButton("Run on Cluster")

        self._specs = self._build_parameter_specs()
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        self._default_params = self._default_shared_params()

        self._build_layout()
        self._connect_signals()

    def _charge_column_width(self) -> int:
        metrics = self._table.fontMetrics()
        content_width = max(
            metrics.horizontalAdvance("positive muon"),
            metrics.horizontalAdvance("muonium"),
            metrics.horizontalAdvance("Charge"),
        )
        return content_width + 72

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        title_column = QVBoxLayout()
        title = QLabel("Structural Relaxations")
        style_section_title(title)
        subtitle = QLabel("Configure and run geometry optimization calculations from staged initial muon positions.")
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
        layout.addWidget(self._kpoint_message)

        table_group = QGroupBox("Relaxation Jobs")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 8, 10, 8)
        table_layout.addWidget(self._table)
        control_row = QHBoxLayout()
        control_row.addStretch(1)
        control_row.addWidget(self._batch_configure_button)
        table_layout.addLayout(control_row)
        layout.addWidget(table_group, 1)

        run_group = QGroupBox("Run Jobs")
        run_layout = QVBoxLayout(run_group)
        run_layout.setContentsMargins(10, 8, 10, 8)
        run_row = QHBoxLayout()
        run_row.addStretch(1)
        run_row.addWidget(self._run_local_button)
        run_row.addWidget(self._run_cluster_button)
        run_layout.addLayout(run_row)
        layout.addWidget(run_group)

    def _connect_signals(self) -> None:
        self._batch_configure_button.clicked.connect(self._configure_selected_rows)
        self._table.itemChanged.connect(self._on_table_item_changed)
        selection_model = self._table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda *_: self._refresh_batch_button_state())
        self._run_local_button.clicked.connect(self._run_local_selected_rows)
        self._run_cluster_button.clicked.connect(self._run_cluster_selected_rows)

    def _build_parameter_specs(self) -> list[KeywordSpec]:
        specs = [
            _clone_spec_for_muon_workflow(spec)
            for spec in get_gui_keyword_specs_for_files(
                files=("param", "cell"),
                workflow_stage=_WORKFLOW_STAGE,
                include_advanced=True,
            )
            if spec.name not in _EXCLUDED_PARAMETER_NAMES
        ]
        seen = {spec.name for spec in specs}
        for name in _EXTRA_GEOMETRY_PARAMETER_NAMES:
            spec = get_keyword_spec("param", name) or get_keyword_spec("cell", name)
            if spec is not None and spec.name not in seen and spec.name not in _EXCLUDED_PARAMETER_NAMES:
                specs.append(spec)
                seen.add(spec.name)
        return specs

    @staticmethod
    def _spec_is_blocked_by_locked(spec: KeywordSpec, locked_names: set[str]) -> bool:
        if spec.name in locked_names:
            return False
        return any(conflict in locked_names for conflict in spec.mutually_exclusive_with)

    @staticmethod
    def _preferred_default_name(left: str, right: str) -> str:
        pair = {left, right}
        for preferred in _PREFERRED_DEFAULT_PARAMETER_NAMES:
            if preferred in pair:
                return preferred
        return min(left, right)

    def _drop_default_conflicts(self, values: dict[str, Any], *, locked_names: set[str] | None = None) -> dict[str, Any]:
        locked = locked_names or set()
        for name in tuple(values):
            spec = self._specs_by_name.get(name)
            if spec is not None and self._spec_is_blocked_by_locked(spec, locked):
                values.pop(name, None)

        # Prefer deterministic defaults inside mutually exclusive groups.
        changed = True
        while changed:
            changed = False
            for name in tuple(values):
                if name in locked:
                    continue
                spec = self._specs_by_name.get(name)
                if spec is None or name not in values:
                    continue
                for conflict in spec.mutually_exclusive_with:
                    if conflict not in values:
                        continue
                    keep = self._preferred_default_name(name, conflict)
                    drop = conflict if keep == name else name
                    if drop in locked:
                        drop = name
                    if drop in values:
                        values.pop(drop, None)
                        changed = True
                        break
                if changed:
                    break
        return values

    def _drop_conflicts_with_explicit_values(
        self,
        values: dict[str, object],
        *,
        explicit_names: set[str],
    ) -> dict[str, object]:
        for name in tuple(explicit_names):
            if name not in values:
                continue
            spec = self._specs_by_name.get(name)
            if spec is None:
                continue
            for conflict in spec.mutually_exclusive_with:
                if conflict in values and conflict not in explicit_names:
                    values.pop(conflict, None)
        return values

    def _default_shared_params(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {"task": infer_task(_WORKFLOW_STAGE)}
        locked_names = {"task"}
        for spec in self._specs:
            if spec.default_value is None:
                continue
            if spec.name == "kpoint_mp_grid":
                continue
            if not keyword_visible_in_workflow(spec, workflow_stage=_WORKFLOW_STAGE, include_advanced=False):
                continue
            if self._spec_is_blocked_by_locked(spec, locked_names):
                continue
            defaults[spec.name] = normalize_gui_value(spec, spec.default_value)
        return self._drop_default_conflicts(defaults, locked_names=locked_names)

    def set_project_defaults(
        self,
        *,
        cutoff_energy: float | None,
        kpoint_grid: tuple[int, int, int] | None,
        kpoint_reference_lattice: list[list[float]] | None,
    ) -> None:
        self._project_default_cutoff.setText(format_cutoff_energy(cutoff_energy))
        self._project_default_kpoint.setText(format_kpoint_grid(kpoint_grid))
        self._reference_grid = kpoint_grid
        self._reference_lattice = [list(row) for row in kpoint_reference_lattice] if isinstance(kpoint_reference_lattice, list) else None
        if cutoff_energy is None:
            self._default_params.pop("cut_off_energy", None)
        else:
            self._default_params["cut_off_energy"] = ValueWithUnit(float(cutoff_energy), "eV")
        self._default_params = self._drop_default_conflicts(self._default_params, locked_names={"task"})
        self._normalize_all_row_parameter_values()
        self._refresh_all_rows()

    def set_host_lattice(self, lattice_vectors: list[list[float]] | None) -> None:
        self._host_lattice = [list(row) for row in lattice_vectors] if lattice_vectors is not None else None
        self._normalize_all_row_parameter_values()
        self._refresh_all_rows()

    def set_runtime_states(self, states: dict[str, RelaxationRunState]) -> None:
        self._runtime_states = {str(label): state for label, state in states.items()}
        self._refresh_all_rows()

    def set_supercell_options(self, options: list[dict[str, object]]) -> None:
        parsed: list[tuple[str, SupercellMatrix]] = []
        for entry in options:
            label = str(entry.get("label", "")).strip() if isinstance(entry, dict) else ""
            matrix = self._coerce_supercell_matrix(entry.get("matrix")) if isinstance(entry, dict) else None
            if label and matrix is not None:
                parsed.append((label, matrix))
        self._supercell_options = parsed
        self._refresh_all_rows()

    def add_staged_sites(self, rows: list[dict[str, object]]) -> None:
        for row_data in rows:
            raw_site_id = row_data.get("id")
            try:
                site_id = UUID(str(raw_site_id))
            except (TypeError, ValueError):
                continue
            fractional = row_data.get("fractional_position", row_data.get("fractional", (0.0, 0.0, 0.0)))
            if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                continue
            base_label = str(row_data.get("label", "")).strip() or f"site_{self._table.rowCount() + 1}"
            self._add_row(
                site_id=site_id,
                label=self._unique_label(base_label),
                fractional=(float(fractional[0]), float(fractional[1]), float(fractional[2])),
                charge_state=POSITIVE_MUON_CHARGE_STATE,
            )
        self._emit_state()

    def row_supercell_matrix(self, row_id: int) -> SupercellMatrix | None:
        return self._row_supercells.get(int(row_id))

    def set_row_supercell(self, row_id: int, matrix: SupercellMatrix | None, *, message: str | None = None) -> None:
        self._row_supercells[int(row_id)] = matrix
        row = self._row_index_for_id(int(row_id))
        if row is not None:
            self._refresh_row(row)
        if message:
            self._kpoint_message.setText(message)
            self._kpoint_message.show()
        else:
            self._set_kpoint_message_for_row(int(row_id))
        self._emit_state()

    def _make_item(self, text: str, *, editable: bool) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        return item

    def _unique_label(self, base_label: str) -> str:
        existing = {self._label_at_row(row) for row in range(self._table.rowCount())}
        if base_label not in existing:
            return base_label
        suffix = 2
        while f"{base_label} {suffix}" in existing:
            suffix += 1
        return f"{base_label} {suffix}"

    def _add_row(
        self,
        *,
        site_id: UUID,
        label: str,
        fractional: tuple[float, float, float],
        charge_state: str,
    ) -> None:
        row = self._table.rowCount()
        row_id = self._next_row_id
        self._next_row_id += 1
        self._row_site_ids[row_id] = site_id
        self._row_positions[row_id] = fractional
        self._row_supercells[row_id] = None
        self._row_parameter_values[row_id] = {}
        self._row_added_advanced[row_id] = set()

        self._table.blockSignals(True)
        self._table.insertRow(row)
        label_item = self._make_item(label, editable=True)
        label_item.setData(Qt.ItemDataRole.UserRole, row_id)
        self._table.setItem(row, self._LABEL_COLUMN, label_item)

        supercell_combo = QComboBox(self._table)
        supercell_combo.currentIndexChanged.connect(lambda _index, rid=row_id: self._on_supercell_combo_changed(rid))
        self._table.setCellWidget(row, self._SUPERCELL_COLUMN, supercell_combo)

        charge_combo = QComboBox(self._table)
        charge_combo.addItem("positive muon", POSITIVE_MUON_CHARGE_STATE)
        charge_combo.addItem("muonium", MUONIUM_CHARGE_STATE)
        charge_index = charge_combo.findData(charge_state)
        charge_combo.setCurrentIndex(charge_index if charge_index >= 0 else 0)
        charge_combo.currentIndexChanged.connect(lambda *_: self._emit_state())
        self._table.setCellWidget(row, self._CHARGE_COLUMN, charge_combo)

        for column, value in zip(
            (self._INITIAL_FX_COLUMN, self._INITIAL_FY_COLUMN, self._INITIAL_FZ_COLUMN),
            fractional,
            strict=True,
        ):
            self._table.setItem(row, column, self._make_item(self._format_fractional(value), editable=False))
        self._table.setItem(row, self._FINAL_FX_COLUMN, self._make_item("", editable=False))
        self._table.setItem(row, self._FINAL_FY_COLUMN, self._make_item("", editable=False))
        self._table.setItem(row, self._FINAL_FZ_COLUMN, self._make_item("", editable=False))
        self._table.setItem(row, self._TOTAL_ENERGY_COLUMN, self._make_item("", editable=False))
        self._table.setItem(row, self._STATUS_COLUMN, self._make_item("not generated", editable=False))

        configure_button = QPushButton("Configure...", self._table)
        configure_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        configure_button.clicked.connect(lambda _checked=False, rid=row_id: self._open_row_parameter_dialog((rid,)))
        self._table.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

        remove_button = QToolButton(self._table)
        remove_button.setText("x")
        remove_button.setAutoRaise(True)
        remove_button.clicked.connect(lambda _checked=False, rid=row_id: self._remove_rows_by_id((rid,)))
        self._table.setCellWidget(row, self._REMOVE_COLUMN, remove_button)

        self._refresh_row(row)
        self._table.blockSignals(False)
        self._refresh_batch_button_state()

    def _row_index_for_id(self, row_id: int) -> int | None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._LABEL_COLUMN)
            if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) == int(row_id):
                return row
        return None

    def _row_id_at(self, row: int) -> int | None:
        item = self._table.item(row, self._LABEL_COLUMN)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    def _label_at_row(self, row: int) -> str:
        item = self._table.item(row, self._LABEL_COLUMN)
        return item.text().strip() if item is not None else ""

    def _charge_state_at_row(self, row: int) -> str:
        combo = self._table.cellWidget(row, self._CHARGE_COLUMN)
        if isinstance(combo, QComboBox):
            return str(combo.currentData() or POSITIVE_MUON_CHARGE_STATE)
        return POSITIVE_MUON_CHARGE_STATE

    @staticmethod
    def _runtime_key(label: str, charge_state: str) -> str:
        prefix = "muonium" if charge_state == MUONIUM_CHARGE_STATE else "positivemuon"
        return f"{prefix}:{label}"

    @staticmethod
    def _runtime_controls_locked(status: str) -> bool:
        return str(status).strip().casefold() not in _UNLOCKED_RUNTIME_STATUSES

    @staticmethod
    def _coerce_supercell_matrix(value: object) -> SupercellMatrix | None:
        if value is None or not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            return (
                (int(value[0][0]), int(value[0][1]), int(value[0][2])),
                (int(value[1][0]), int(value[1][1]), int(value[1][2])),
                (int(value[2][0]), int(value[2][1]), int(value[2][2])),
            )
        except (TypeError, ValueError, IndexError):
            return None

    @staticmethod
    def _supercell_label(matrix: SupercellMatrix | None) -> str:
        if matrix is None or matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            return "Unit cell"
        if all(matrix[i][j] == 0 for i in range(3) for j in range(3) if i != j):
            return f"{matrix[0][0]}x{matrix[1][1]}x{matrix[2][2]}"
        return "Custom 3x3"

    def _populate_supercell_combo(self, combo: QComboBox, row_id: int) -> None:
        current = self._row_supercells.get(row_id)
        previous = combo.blockSignals(True)
        combo.clear()
        combo.addItem("Unit cell", ("matrix", None))
        selected_index = 0
        for label, matrix in self._supercell_options:
            combo.addItem(label, ("matrix", matrix))
            if current is not None and matrix == current:
                selected_index = combo.count() - 1
        if current is not None and selected_index == 0:
            combo.addItem(self._supercell_label(current), ("matrix", current))
            selected_index = combo.count() - 1
        combo.addItem("Create new...", ("create", None))
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(previous)

    def _on_supercell_combo_changed(self, row_id: int) -> None:
        row = self._row_index_for_id(row_id)
        if row is None:
            return
        combo = self._table.cellWidget(row, self._SUPERCELL_COLUMN)
        if not isinstance(combo, QComboBox):
            return
        data = combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        mode = data[0]
        matrix = self._coerce_supercell_matrix(data[1])
        if mode == "create":
            self.row_supercell_edit_requested.emit(row_id)
            self._refresh_row(row)
            return
        self._row_supercells[row_id] = matrix
        self._row_parameter_values[row_id] = self._strip_implicit_row_defaults(row_id, self._row_parameter_values.get(row_id, {}))
        self._set_kpoint_message_for_row(row_id)
        self._refresh_row(row)
        self._emit_state()

    def _suggested_grid_for_row(self, row_id: int) -> tuple[int, int, int] | None:
        if self._host_lattice is None:
            return None
        return suggested_grid_for_row(
            host_lattice=self._host_lattice,
            row_matrix=self._row_supercells.get(row_id),
            reference_lattice=self._reference_lattice,
            reference_grid=self._reference_grid,
        )

    def _set_kpoint_message_for_row(self, row_id: int) -> None:
        current_matrix = self._row_supercells.get(row_id)
        if current_matrix is None or self._reference_grid is None or self._host_lattice is None:
            self._kpoint_message.hide()
            self._kpoint_message.setText("")
            return
        suggested = self._suggested_grid_for_row(row_id)
        if suggested is None or suggested == self._reference_grid:
            self._kpoint_message.hide()
            self._kpoint_message.setText("")
            return
        self._kpoint_message.setText(
            "K-point grid adjusted from "
            f"{self._reference_grid[0]}x{self._reference_grid[1]}x{self._reference_grid[2]} to "
            f"{suggested[0]}x{suggested[1]}x{suggested[2]} for equivalent density."
        )
        self._kpoint_message.show()

    def _effective_params_for_row(self, row_id: int) -> dict[str, object]:
        values = dict(self._default_params)
        row_values = self._row_parameter_values.get(row_id, {})
        values.update(row_values)
        self._drop_conflicts_with_explicit_values(values, explicit_names=set(row_values))
        values["task"] = infer_task(_WORKFLOW_STAGE)
        if "kpoint_mp_grid" not in values:
            suggested = self._suggested_grid_for_row(row_id)
            if suggested is not None:
                values["kpoint_mp_grid"] = suggested
        self._drop_default_conflicts(values, locked_names={"task"})
        return values

    def _implicit_params_for_row(self, row_id: int) -> dict[str, object]:
        implicit = dict(self._default_params)
        implicit["task"] = infer_task(_WORKFLOW_STAGE)
        suggested = self._suggested_grid_for_row(row_id)
        if suggested is not None:
            implicit["kpoint_mp_grid"] = suggested
        return self._drop_default_conflicts(implicit, locked_names={"task"})

    def _strip_implicit_row_defaults(self, row_id: int, values: dict[str, object]) -> dict[str, object]:
        cleaned = self._sanitize_parameter_mapping(values)
        implicit = self._implicit_params_for_row(row_id)
        stripped: dict[str, object] = {}
        for name, value in cleaned.items():
            if name not in self._specs_by_name:
                continue
            if name in implicit and value == implicit[name]:
                continue
            stripped[name] = value
        return stripped

    def _normalize_all_row_parameter_values(self) -> None:
        for row_id in tuple(self._row_parameter_values):
            self._row_parameter_values[row_id] = self._strip_implicit_row_defaults(row_id, self._row_parameter_values.get(row_id, {}))

    def _refresh_row(self, row: int) -> None:
        row_id = self._row_id_at(row)
        if row_id is None:
            return
        supercell_combo = self._table.cellWidget(row, self._SUPERCELL_COLUMN)
        if isinstance(supercell_combo, QComboBox):
            self._populate_supercell_combo(supercell_combo, row_id)
        label = self._label_at_row(row)
        charge_state = self._charge_state_at_row(row)
        runtime = self._runtime_states.get(self._runtime_key(label, charge_state))
        if runtime is None:
            runtime = self._runtime_states.get(label)
        status = "not generated" if runtime is None else runtime.status or "not generated"
        locked = self._runtime_controls_locked(status)
        if isinstance(supercell_combo, QComboBox):
            supercell_combo.setEnabled(not locked)
        charge_combo = self._table.cellWidget(row, self._CHARGE_COLUMN)
        if isinstance(charge_combo, QComboBox):
            charge_combo.setEnabled(not locked)
        if runtime is None:
            self._set_status_item(row, status)
            self._set_fractional_items(row, None)
            self._set_float_item(row, self._TOTAL_ENERGY_COLUMN, None)
        else:
            self._set_status_item(row, status)
            self._set_fractional_items(row, runtime.final_fractional_position)
            self._set_float_item(row, self._TOTAL_ENERGY_COLUMN, runtime.total_energy)
        configure_button = self._table.cellWidget(row, self._CONFIGURE_COLUMN)
        if isinstance(configure_button, QPushButton):
            configure_button.setText("Configured..." if self._row_parameter_values.get(row_id, {}) else "Configure...")

    def _set_status_item(self, row: int, status: str) -> None:
        item = self._table.item(row, self._STATUS_COLUMN)
        if item is None:
            item = self._make_item(status, editable=False)
            self._table.setItem(row, self._STATUS_COLUMN, item)
        else:
            item.setText(status)
        item.setForeground(status_brush(status))

    @staticmethod
    def _format_fractional(value: float) -> str:
        return f"{float(value):.3f}"

    def _set_float_item(
        self,
        row: int,
        column: int,
        value: float | None,
        *,
        formatter: str = ".8g",
    ) -> None:
        item = self._table.item(row, column)
        text = "" if value is None else format(float(value), formatter)
        if item is None:
            self._table.setItem(row, column, self._make_item(text, editable=False))
        else:
            item.setText(text)

    def _set_fractional_items(self, row: int, values: tuple[float, float, float] | None) -> None:
        for column, value in zip(
            (self._FINAL_FX_COLUMN, self._FINAL_FY_COLUMN, self._FINAL_FZ_COLUMN),
            values or (None, None, None),
            strict=True,
        ):
            self._set_float_item(row, column, value, formatter=".3f")

    def _refresh_all_rows(self) -> None:
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            self._refresh_row(row)
        self._table.blockSignals(False)

    def _selected_row_ids(self) -> tuple[int, ...]:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return ()
        ids: list[int] = []
        for index in selection_model.selectedRows():
            row_id = self._row_id_at(index.row())
            if row_id is not None:
                ids.append(row_id)
        return tuple(ids)

    def _refresh_batch_button_state(self) -> None:
        self._batch_configure_button.setEnabled(bool(self._selected_row_ids()))

    def _configure_selected_rows(self) -> None:
        row_ids = self._selected_row_ids()
        if not row_ids:
            QMessageBox.information(self, "No rows selected", "Select one or more rows first.")
            return
        self._open_row_parameter_dialog(row_ids)

    def _common_params_for_rows(self, row_ids: tuple[int, ...]) -> dict[str, object]:
        if not row_ids:
            return dict(self._default_params)
        maps = [self._effective_params_for_row(row_id) for row_id in row_ids]
        common = dict(maps[0])
        for candidate in maps[1:]:
            common = {name: value for name, value in common.items() if name in candidate and candidate[name] == value}
        common["task"] = infer_task(_WORKFLOW_STAGE)
        return common

    def _advanced_names_for_rows(self, row_ids: tuple[int, ...]) -> set[str]:
        names: set[str] = set()
        for row_id in row_ids:
            names.update(self._row_added_advanced.get(row_id, set()))
        return names

    def _restore_structured_value(self, name: str, value: object) -> object:
        spec = self._specs_by_name.get(name)
        if spec is None:
            return value
        if isinstance(value, dict):
            if spec.kind == "block" and "rows" in value:
                return BlockSection(
                    name=str(value.get("name", spec.name)),
                    rows=tuple(tuple(row) for row in value.get("rows", ())),
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

    def _open_row_parameter_dialog(self, row_ids: tuple[int, ...]) -> None:
        intro_labels = [self._label_at_row(row) for row_id in row_ids if (row := self._row_index_for_id(row_id)) is not None]
        charges = {
            self._charge_state_at_row(row)
            for row_id in row_ids
            if (row := self._row_index_for_id(row_id)) is not None
        }
        charge_text = "Variable (set by row control)"
        if len(charges) == 1:
            charge_state = next(iter(charges))
            charge_text = f"{muon_charge_for_state(charge_state):+d} ({muon_charge_state_label(charge_state)})"
        dialog = ParameterDialog(
            specs=self._specs,
            workflow_stage=_WORKFLOW_STAGE,
            locked_values={
                "task": "GEOMETRYOPTIMIZATION",
                "fix_all_cell": "TRUE (required for muon relaxation)",
                "write_cell_structure": "TRUE (required for relaxed-cell outputs)",
                "charge": charge_text,
            },
            intro_text="Included rows: " + ", ".join(intro_labels),
            parent=self,
        )
        dialog.setWindowTitle("Configure Muon Relaxation Parameters")
        dialog.set_values(self._common_params_for_rows(row_ids))
        dialog.set_advanced_names(self._advanced_names_for_rows(row_ids))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = self._sanitize_parameter_mapping(dialog.collect_values())
        advanced = set(dialog.advanced_names())
        for row_id in row_ids:
            self._row_parameter_values[row_id] = self._strip_implicit_row_defaults(row_id, values)
            self._row_added_advanced[row_id] = set(advanced)
        self._refresh_all_rows()
        self._emit_state()

    def _remove_rows_by_id(self, row_ids: tuple[int, ...]) -> None:
        removed_labels: list[str] = []
        for row_id in row_ids:
            row = self._row_index_for_id(row_id)
            if row is None:
                continue
            removed_labels.append(self._label_at_row(row))
            self._table.removeRow(row)
            self._row_site_ids.pop(row_id, None)
            self._row_positions.pop(row_id, None)
            self._row_supercells.pop(row_id, None)
            self._row_parameter_values.pop(row_id, None)
            self._row_added_advanced.pop(row_id, None)
        if removed_labels:
            self.remove_requested.emit(removed_labels)
            self._refresh_batch_button_state()
            self._emit_state()

    def _build_config_or_warn(self) -> MuonRelaxationConfig | None:
        if self._table.rowCount() == 0:
            QMessageBox.warning(self, "No jobs", "Create one or more jobs from generated muon sites first.")
            return None
        rows: list[MuonRelaxationRow] = []
        seen: set[str] = set()
        for row_index in range(self._table.rowCount()):
            row_id = self._row_id_at(row_index)
            if row_id is None:
                continue
            label = self._label_at_row(row_index)
            if not label:
                QMessageBox.warning(self, "Invalid label", "Job labels must not be empty.")
                return None
            charge_state = self._charge_state_at_row(row_index)
            unique = f"{charge_state}:{label}"
            if unique in seen:
                QMessageBox.warning(self, "Duplicate label", f"Duplicate {muon_charge_state_label(charge_state)} job label '{label}'.")
                return None
            seen.add(unique)
            effective_values = self._effective_params_for_row(row_id)
            errors = validate_gui_values(effective_values, self._specs_by_name, task=infer_task(_WORKFLOW_STAGE))
            if errors:
                QMessageBox.warning(self, "Invalid parameters", f"Row '{label}':\n" + "\n".join(errors))
                return None
            rows.append(
                MuonRelaxationRow(
                    site_id=self._row_site_ids[row_id],
                    label=label,
                    initial_fractional_position=self._row_positions[row_id],
                    charge_state=charge_state,
                    supercell_matrix=self._row_supercells.get(row_id),
                    parameter_values=dict(self._row_parameter_values.get(row_id, {})),
                    added_advanced=tuple(sorted(self._row_added_advanced.get(row_id, set()))),
                )
            )
        return MuonRelaxationConfig(rows=tuple(rows))

    def _submission_key_at_row(self, row: int) -> str:
        return self._runtime_key(self._label_at_row(row), self._charge_state_at_row(row))

    def _selected_or_all_labels(self) -> list[str]:
        selection_model = self._table.selectionModel()
        selected_rows = [index.row() for index in selection_model.selectedRows()] if selection_model is not None else []
        if not selected_rows:
            selected_rows = list(range(self._table.rowCount()))
        return [self._submission_key_at_row(row) for row in selected_rows if self._label_at_row(row)]

    def _run_local_selected_rows(self) -> None:
        config = self._build_config_or_warn()
        if config is None:
            return
        labels = self._selected_or_all_labels()
        if not labels:
            QMessageBox.information(self, "No jobs selected", "Select one or more rows first.")
            return
        self.configure_requested.emit(config)
        self.submit_local_requested.emit(labels)
        self._emit_state()

    def _run_cluster_selected_rows(self) -> None:
        config = self._build_config_or_warn()
        if config is None:
            return
        labels = self._selected_or_all_labels()
        if not labels:
            QMessageBox.information(self, "No jobs selected", "Select one or more rows first.")
            return
        self.configure_requested.emit(config)
        self.submit_cluster_requested.emit(labels)
        self._emit_state()

    def collect_state(self) -> dict[str, Any]:
        rows: list[dict[str, object]] = []
        for row in range(self._table.rowCount()):
            row_id = self._row_id_at(row)
            if row_id is None:
                continue
            rows.append(
                {
                    "site_id": str(self._row_site_ids[row_id]),
                    "label": self._label_at_row(row),
                    "fractional_position": list(self._row_positions[row_id]),
                    "charge_state": self._charge_state_at_row(row),
                    "supercell_matrix": (
                        [list(line) for line in self._row_supercells.get(row_id)]
                        if self._row_supercells.get(row_id) is not None
                        else None
                    ),
                    "parameter_values": dict(self._row_parameter_values.get(row_id, {})),
                    "added_advanced": sorted(self._row_added_advanced.get(row_id, set())),
                }
            )
        return {"rows": rows}

    def set_form_state(self, state: dict[str, Any]) -> None:
        self._restoring_form_state = True
        try:
            self._table.blockSignals(True)
            self._table.setRowCount(0)
            self._next_row_id = 1
            self._row_site_ids.clear()
            self._row_positions.clear()
            self._row_supercells.clear()
            self._row_parameter_values.clear()
            self._row_added_advanced.clear()
            raw_rows = state.get("rows", []) if isinstance(state, dict) else []
            if not isinstance(raw_rows, list):
                raw_rows = []
            for entry in raw_rows:
                if not isinstance(entry, dict):
                    continue
                try:
                    site_id = UUID(str(entry.get("site_id")))
                except (TypeError, ValueError):
                    continue
                fractional = entry.get("fractional_position", (0.0, 0.0, 0.0))
                if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                    continue
                self._add_row(
                    site_id=site_id,
                    label=str(entry.get("label", "")).strip() or f"site_{self._table.rowCount() + 1}",
                    fractional=(float(fractional[0]), float(fractional[1]), float(fractional[2])),
                    charge_state=str(entry.get("charge_state", POSITIVE_MUON_CHARGE_STATE)),
                )
                row_id = self._row_id_at(self._table.rowCount() - 1)
                if row_id is None:
                    continue
                self._row_supercells[row_id] = self._coerce_supercell_matrix(entry.get("supercell_matrix"))
                params = entry.get("parameter_values", {})
                if isinstance(params, dict):
                    self._row_parameter_values[row_id] = self._sanitize_parameter_mapping({str(name): value for name, value in params.items()})
                added = entry.get("added_advanced", [])
                if isinstance(added, list):
                    self._row_added_advanced[row_id] = {str(name) for name in added if isinstance(name, str)}
            self._normalize_all_row_parameter_values()
            self._refresh_all_rows()
            self._refresh_batch_button_state()
        finally:
            self._table.blockSignals(False)
            self._restoring_form_state = False
        self._emit_state()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self._LABEL_COLUMN:
            return
        self._refresh_all_rows()
        self._emit_state()

    def _emit_state(self) -> None:
        if self._restoring_form_state:
            return
        self.form_state_changed.emit(self.collect_state())

"""Muon structural-relaxation configuration panel."""

from __future__ import annotations

from collections.abc import Sequence
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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.active_host_badge import ActiveHostBadge
from mufinder.app.convergence_display import format_cutoff_energy, format_kpoint_grid
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.workflow_parameters import kpoint_exclusive_hint, reorder_workflow_parameter_specs
from mufinder.app.relaxation_adapter import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    muon_charge_for_state,
    muon_charge_state_label,
    suggested_grid_for_row,
)
from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow, RelaxationRunState, SupercellMatrix
from mufinder.app.workflow_table_utils import (
    bind_delete_shortcuts,
    create_workflow_table,
    style_workflow_configure_button,
)
from mufinder.app.theme import (
    current_palette,
    status_brush,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
    use_tabbed_table_corners,
)
from mufinder.science.kpoint_grids import spacing_from_grid
from mufinder.castep_io.gui_metadata import (
    get_gui_keyword_specs_for_files,
    infer_task,
    keyword_visible_in_workflow,
    normalize_gui_value,
    validate_gui_values,
)
from mufinder.castep_io.schema import KeywordSpec, get_keyword_spec
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.app.selection_snapshot import (
    SelectionSnapshotComboBox,
    SelectionSnapshotPushButton,
    SelectionSnapshotToolButton,
    resolve_target_keys,
)


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
_UNLOCKED_RUNTIME_STATUSES = {"", "not generated", "defined", "ready", "cancelled"}
_CHARGE_STATE_TABS = (
    (POSITIVE_MUON_CHARGE_STATE, "Positive muon"),
    (MUONIUM_CHARGE_STATE, "Muonium"),
)


def _append_workflow_stage(values: tuple[str, ...], workflow_stage: str) -> tuple[str, ...]:
    if workflow_stage in values:
        return values
    return (*values, workflow_stage)


def _clone_spec_for_muon_workflow(
    spec: KeywordSpec,
    *,
    basic: bool,
    advanced: bool,
) -> KeywordSpec:
    basic_workflows = spec.basic_workflows
    advanced_workflows = spec.advanced_workflows

    if basic:
        basic_workflows = _append_workflow_stage(basic_workflows, _WORKFLOW_STAGE)
    if advanced:
        advanced_workflows = _append_workflow_stage(advanced_workflows, _WORKFLOW_STAGE)

    default_value = spec.default_value
    if spec.name == "continuation":
        default_value = "DEFAULT"
    elif spec.name == "geom_max_iter":
        default_value = 300
    elif spec.name == "symmetry_generate":
        default_value = None

    return replace(
        spec,
        basic_workflows=basic_workflows,
        advanced_workflows=advanced_workflows,
        default_value=default_value,
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
    _INITIAL_FX_COLUMN = 2
    _INITIAL_FY_COLUMN = 3
    _INITIAL_FZ_COLUMN = 4
    _FINAL_FX_COLUMN = 5
    _FINAL_FY_COLUMN = 6
    _FINAL_FZ_COLUMN = 7
    _TOTAL_ENERGY_COLUMN = 8
    _STATUS_COLUMN = 9
    _CONFIGURE_COLUMN = 10
    _REMOVE_COLUMN = 11

    configure_requested = Signal(object)
    submit_local_requested = Signal(object)
    submit_cluster_requested = Signal(object)
    row_supercell_edit_requested = Signal(object)
    remove_requested = Signal(object)
    analysis_workspace_requested = Signal(object)
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
        self._row_charge_states: dict[int, str] = {}
        self._row_supercells: dict[int, SupercellMatrix | None] = {}
        self._row_parameter_values: dict[int, dict[str, object]] = {}
        self._row_added_advanced: dict[int, set[str]] = {}
        self._runtime_states: dict[str, RelaxationRunState] = {}
        self._supercell_options: list[tuple[str, SupercellMatrix]] = []
        self._parameter_species_options: tuple[str, ...] = ()

        self._active_host_badge = ActiveHostBadge()
        self._project_default_cutoff = QLabel("Not set")
        self._project_default_kpoint = QLabel("Not set")
        self._kpoint_message = QLabel("")
        self._kpoint_message.setWordWrap(True)
        style_hint_text(self._kpoint_message)
        self._kpoint_message.hide()

        self._charge_tabs = QTabWidget()
        self._charge_tabs.setObjectName("muonRelaxationChargeTabs")
        self._tables_by_charge_state: dict[str, QTableWidget] = {}
        for charge_state, tab_label in _CHARGE_STATE_TABS:
            table = self._create_table()
            self._tables_by_charge_state[charge_state] = table
            self._charge_tabs.addTab(table, tab_label)
        self._table = self._tables_by_charge_state[POSITIVE_MUON_CHARGE_STATE]

        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._batch_configure_button.setEnabled(False)
        self._create_analysis_button = QPushButton("Create Analysis Workspace from Selection")
        self._run_local_button = QPushButton("Run Local Calculations")
        self._run_cluster_button = QPushButton("Run on Cluster")

        self._specs = self._build_parameter_specs()
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        self._default_params = self._default_shared_params()

        self._build_layout()
        self._connect_signals()
        self._delete_shortcut, self._backspace_shortcut = bind_delete_shortcuts(
            self, self._delete_selected_rows
        )

    def _create_table(self) -> QTableWidget:
        column_labels = (
            "Label",
            "Supercell",
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
        )
        table = create_workflow_table(column_labels)
        # This table carries a custom grouped header spanning the coordinate
        # columns; install it and re-apply the labels the helper set on the
        # default header it replaces.
        table.setHorizontalHeader(
            _GroupedCoordinateHeader(
                (
                    (self._INITIAL_FX_COLUMN, self._INITIAL_FZ_COLUMN, "Initial"),
                    (self._FINAL_FX_COLUMN, self._FINAL_FZ_COLUMN, "Final"),
                ),
                table,
            )
        )
        table.setHorizontalHeaderLabels(list(column_labels))
        # Per-widget settings the shared helper does not cover (or that this
        # editable table overrides): extended multi-select, in-place editing,
        # a fixed row height, centered header text, per-column resize modes,
        # a non-stretching last (Remove) column, and tabbed-pane corners.
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        table.verticalHeader().setDefaultSectionSize(32)
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
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
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setSectionResizeMode(self._REMOVE_COLUMN, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(self._REMOVE_COLUMN, 34)
        use_tabbed_table_corners(table)
        return table

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
        layout.addWidget(self._active_host_badge)
        layout.addWidget(self._kpoint_message)

        table_group = QGroupBox("Relaxation Jobs")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 8, 10, 8)
        table_layout.addWidget(self._charge_tabs)
        control_row = QHBoxLayout()
        control_row.addWidget(self._create_analysis_button)
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
        self._create_analysis_button.clicked.connect(self._create_analysis_workspace_from_selection)
        self._charge_tabs.currentChanged.connect(lambda *_: self._refresh_batch_button_state())
        for table in self._tables_by_charge_state.values():
            table.itemChanged.connect(self._on_table_item_changed)
            selection_model = table.selectionModel()
            if selection_model is not None:
                selection_model.selectionChanged.connect(lambda *_: self._refresh_batch_button_state())
        self._run_local_button.clicked.connect(self._run_local_selected_rows)
        self._run_cluster_button.clicked.connect(self._run_cluster_selected_rows)

    def _build_parameter_specs(self) -> list[KeywordSpec]:
        specs: list[KeywordSpec] = []
        seen_names: set[str] = set()

        convergence_specs = get_gui_keyword_specs_for_files(
            files=("param", "cell"),
            workflow_stage="convergence_cutoff",
            include_advanced=True,
        )
        for spec in convergence_specs:
            if spec.name in _EXCLUDED_PARAMETER_NAMES or spec.name in seen_names:
                continue
            basic_in_convergence = keyword_visible_in_workflow(
                spec,
                workflow_stage="convergence_cutoff",
                include_advanced=False,
            )
            cloned = _clone_spec_for_muon_workflow(
                spec,
                basic=(
                    basic_in_convergence
                    or spec.name in ("kpoint_mp_grid", "kpoint_mp_spacing")
                    or spec.name in _EXTRA_GEOMETRY_PARAMETER_NAMES
                ),
                advanced=keyword_visible_in_workflow(
                    spec,
                    workflow_stage="convergence_cutoff",
                    include_advanced=True,
                )
                and not basic_in_convergence,
            )
            specs.append(cloned)
            seen_names.add(cloned.name)

        for name in _EXTRA_GEOMETRY_PARAMETER_NAMES:
            spec = get_keyword_spec("param", name) or get_keyword_spec("cell", name)
            if spec is None or spec.name in _EXCLUDED_PARAMETER_NAMES or spec.name in seen_names:
                continue
            specs.append(
                _clone_spec_for_muon_workflow(
                    spec,
                    basic=True,
                    advanced=True,
                )
            )
            seen_names.add(spec.name)

        return reorder_workflow_parameter_specs(specs)

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
            # kpoint_mp_grid and kpoint_mp_spacing are project-default-aware:
            # _effective_params_for_row injects them from the adopted project
            # grid (and its equivalent spacing) so the dialog reflects the
            # user's project default rather than the bare schema default.
            if spec.name in ("kpoint_mp_grid", "kpoint_mp_spacing"):
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

    def set_active_host(self, structure: object | None) -> None:
        """Update the active-host badge."""
        from mufinder.domain.models import Structure as _Structure
        self._active_host_badge.set_active_host(structure if isinstance(structure, _Structure) else None)

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

    def set_parameter_species_options(self, species_options: Sequence[str]) -> None:
        self._parameter_species_options = tuple(
            dict.fromkeys(option for item in species_options if (option := str(item).strip()))
        )

    def add_staged_sites(
        self,
        rows: list[dict[str, object]],
        *,
        charge_states: tuple[str, ...] = (POSITIVE_MUON_CHARGE_STATE,),
    ) -> None:
        normalized_charge_states = tuple(
            charge_state
            for charge_state in charge_states
            if charge_state in self._tables_by_charge_state
        ) or (POSITIVE_MUON_CHARGE_STATE,)
        for row_data in rows:
            raw_site_id = row_data.get("id")
            try:
                site_id = UUID(str(raw_site_id))
            except (TypeError, ValueError):
                continue
            fractional = row_data.get("fractional_position", row_data.get("fractional", (0.0, 0.0, 0.0)))
            if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                continue
            for charge_state in normalized_charge_states:
                table = self._table_for_charge_state(charge_state)
                base_label = str(row_data.get("label", "")).strip() or f"site_{table.rowCount() + 1}"
                self._add_row(
                    site_id=site_id,
                    label=self._unique_label(base_label, charge_state=charge_state),
                    fractional=(float(fractional[0]), float(fractional[1]), float(fractional[2])),
                    charge_state=charge_state,
                )
        self._emit_state()

    def _table_for_charge_state(self, charge_state: str) -> QTableWidget:
        return self._tables_by_charge_state.get(charge_state, self._table)

    def _active_table(self) -> QTableWidget:
        current = self._charge_tabs.currentWidget()
        return current if isinstance(current, QTableWidget) else self._table

    def _all_tables(self) -> tuple[QTableWidget, ...]:
        return tuple(self._tables_by_charge_state[charge_state] for charge_state, _label in _CHARGE_STATE_TABS)

    def row_supercell_matrix(self, row_id: int) -> SupercellMatrix | None:
        return self._row_supercells.get(int(row_id))

    def set_row_supercell(self, row_id: int, matrix: SupercellMatrix | None, *, message: str | None = None) -> None:
        self._row_supercells[int(row_id)] = matrix
        location = self._row_location_for_id(int(row_id))
        if location is not None:
            table, row = location
            self._refresh_row(row, table=table)
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

    def _unique_label(self, base_label: str, *, charge_state: str) -> str:
        table = self._table_for_charge_state(charge_state)
        existing = {self._label_at_row(row, table=table) for row in range(table.rowCount())}
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
        if charge_state not in self._tables_by_charge_state:
            charge_state = POSITIVE_MUON_CHARGE_STATE
        table = self._table_for_charge_state(charge_state)
        row = table.rowCount()
        row_id = self._next_row_id
        self._next_row_id += 1
        self._row_site_ids[row_id] = site_id
        self._row_positions[row_id] = fractional
        self._row_charge_states[row_id] = charge_state
        self._row_supercells[row_id] = None
        self._row_parameter_values[row_id] = {}
        self._row_added_advanced[row_id] = set()

        table.blockSignals(True)
        table.insertRow(row)
        label_item = self._make_item(label, editable=True)
        label_item.setData(Qt.ItemDataRole.UserRole, row_id)
        table.setItem(row, self._LABEL_COLUMN, label_item)

        supercell_combo = SelectionSnapshotComboBox(table)
        supercell_combo.selection_snapshot_key = row_id
        supercell_combo.currentIndexChanged.connect(
            lambda _index, rid=row_id, combo=supercell_combo: self._on_supercell_combo_changed(rid, combo)
        )
        table.setCellWidget(row, self._SUPERCELL_COLUMN, supercell_combo)

        for column, value in zip(
            (self._INITIAL_FX_COLUMN, self._INITIAL_FY_COLUMN, self._INITIAL_FZ_COLUMN),
            fractional,
            strict=True,
        ):
            table.setItem(row, column, self._make_item(self._format_fractional(value), editable=False))
        table.setItem(row, self._FINAL_FX_COLUMN, self._make_item("", editable=False))
        table.setItem(row, self._FINAL_FY_COLUMN, self._make_item("", editable=False))
        table.setItem(row, self._FINAL_FZ_COLUMN, self._make_item("", editable=False))
        table.setItem(row, self._TOTAL_ENERGY_COLUMN, self._make_item("", editable=False))
        table.setItem(row, self._STATUS_COLUMN, self._make_item("not generated", editable=False))

        configure_button = SelectionSnapshotPushButton("Configure...", table)
        configure_button.selection_snapshot_key = row_id
        configure_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        configure_button.clicked.connect(
            lambda _checked=False, rid=row_id, btn=configure_button: self._dispatch_row_action(btn, rid, "configure", table)
        )
        table.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

        remove_button = SelectionSnapshotToolButton(table)
        remove_button.selection_snapshot_key = row_id
        remove_button.setText("x")
        remove_button.setAutoRaise(True)
        remove_button.clicked.connect(
            lambda _checked=False, rid=row_id, btn=remove_button: self._dispatch_row_action(btn, rid, "remove", table)
        )
        table.setCellWidget(row, self._REMOVE_COLUMN, remove_button)

        self._refresh_row(row, table=table)
        table.blockSignals(False)
        self._refresh_batch_button_state()

    def _row_location_for_id(self, row_id: int) -> tuple[QTableWidget, int] | None:
        for table in self._all_tables():
            for row in range(table.rowCount()):
                item = table.item(row, self._LABEL_COLUMN)
                if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) == int(row_id):
                    return table, row
        return None

    def _row_index_for_id(self, row_id: int) -> int | None:
        location = self._row_location_for_id(row_id)
        return None if location is None else location[1]

    def _row_id_at(self, row: int, *, table: QTableWidget | None = None) -> int | None:
        target = table or self._table
        item = target.item(row, self._LABEL_COLUMN)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    def _label_at_row(self, row: int, *, table: QTableWidget | None = None) -> str:
        target = table or self._table
        item = target.item(row, self._LABEL_COLUMN)
        return item.text().strip() if item is not None else ""

    def _charge_state_at_row(self, row: int, *, table: QTableWidget | None = None) -> str:
        row_id = self._row_id_at(row, table=table)
        if row_id is None:
            return POSITIVE_MUON_CHARGE_STATE
        return self._row_charge_states.get(row_id, POSITIVE_MUON_CHARGE_STATE)

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
        combo.addItem("Manage Supercells...", ("create", None))
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(previous)

    def _on_supercell_combo_changed(self, row_id: int, combo: SelectionSnapshotComboBox) -> None:
        location = self._row_location_for_id(row_id)
        if location is None:
            return
        table, row = location
        if not isinstance(combo, QComboBox):
            return
        data = combo.currentData()
        if not isinstance(data, tuple) or len(data) != 2:
            return
        mode = data[0]
        matrix = self._coerce_supercell_matrix(data[1])
        if mode == "create":
            self.row_supercell_edit_requested.emit(row_id)
            self._refresh_row(row, table=table)
            return
        snapshot_keys = [
            self._row_id_at(r, table=table)
            for r in combo.selection_snapshot_rows
        ]
        target_row_ids = resolve_target_keys(
            combo,
            clicked_key=row_id,
            snapshot_keys=[k for k in snapshot_keys if k is not None],
            eligibility=self._row_supercell_is_editable,
            parent=self,
            title="Apply supercell to selected rows?",
            message="Apply this supercell choice to all selected editable relaxation rows?",
        )
        for target_row_id in target_row_ids:
            self._row_supercells[target_row_id] = matrix
            self._row_parameter_values[target_row_id] = self._strip_implicit_row_defaults(
                target_row_id,
                self._row_parameter_values.get(target_row_id, {}),
            )
            loc = self._row_location_for_id(target_row_id)
            if loc is not None:
                target_table, target_row = loc
                self._refresh_row(target_row, table=target_table)
        self._set_kpoint_message_for_row(row_id)
        self._emit_state()

    def _dispatch_row_action(
        self,
        widget: SelectionSnapshotPushButton | SelectionSnapshotToolButton,
        row_id: int,
        kind: str,
        table: QTableWidget,
    ) -> None:
        snapshot_keys = [
            self._row_id_at(r, table=table)
            for r in widget.selection_snapshot_rows
        ]
        if kind == "configure":
            target_row_ids = resolve_target_keys(
                widget,
                clicked_key=row_id,
                snapshot_keys=[k for k in snapshot_keys if k is not None],
                eligibility=lambda rid: True,
                parent=self,
                title="Configure selected rows?",
                message="Open a shared parameter dialog for all selected relaxation rows?",
            )
            self._open_row_parameter_dialog(target_row_ids)
        elif kind == "remove":
            target_row_ids = resolve_target_keys(
                widget,
                clicked_key=row_id,
                snapshot_keys=[k for k in snapshot_keys if k is not None],
                eligibility=lambda rid: True,
                parent=self,
                title="Remove selected rows?",
                message="Remove all selected relaxation rows?",
            )
            self._remove_rows_by_id(target_row_ids)

    def _row_supercell_is_editable(self, row_id: int) -> bool:
        location = self._row_location_for_id(row_id)
        if location is None:
            return False
        table, row = location
        label = self._label_at_row(row, table=table)
        charge_state = self._charge_state_at_row(row, table=table)
        runtime = self._runtime_states.get(self._runtime_key(label, charge_state))
        if runtime is None:
            runtime = self._runtime_states.get(label)
        status = "not generated" if runtime is None else runtime.status or "not generated"
        return not self._runtime_controls_locked(status)

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

    def _suggested_spacing_for_row(self, row_id: int) -> ValueWithUnit | None:
        # Mirror the project's adopted k-point grid as a k-point spacing so
        # the configure-parameters dialog has a sensible default when the
        # user toggles to "Use spacing".
        if self._reference_grid is None or self._reference_lattice is None:
            return None
        if tuple(int(v) for v in self._reference_grid) == (1, 1, 1):
            return None
        try:
            spacing = spacing_from_grid(self._reference_lattice, self._reference_grid)
        except ValueError:
            return None
        return ValueWithUnit(float(spacing), "1/ang")

    def _effective_params_for_row(self, row_id: int) -> dict[str, object]:
        # See PristineRelaxationWidget._effective_params_for_row for the
        # rationale: kpoint_mp_grid and kpoint_mp_spacing are mutually
        # exclusive, so the row carries only one of them. The dialog
        # seeding path (_common_params_for_rows) layers the other half in
        # as a UX-only hint.
        values = dict(self._default_params)
        row_values = self._row_parameter_values.get(row_id, {})
        values.update(row_values)
        self._drop_conflicts_with_explicit_values(values, explicit_names=set(row_values))
        values["task"] = infer_task(_WORKFLOW_STAGE)
        if (
            "kpoint_mp_grid" not in values
            and "kpoint_mp_spacing" not in values
        ):
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

    def _refresh_row(self, row: int, *, table: QTableWidget | None = None) -> None:
        target = table or self._table
        row_id = self._row_id_at(row, table=target)
        if row_id is None:
            return
        supercell_combo = target.cellWidget(row, self._SUPERCELL_COLUMN)
        if isinstance(supercell_combo, QComboBox):
            self._populate_supercell_combo(supercell_combo, row_id)
        label = self._label_at_row(row, table=target)
        charge_state = self._charge_state_at_row(row, table=target)
        runtime = self._runtime_states.get(self._runtime_key(label, charge_state))
        if runtime is None:
            runtime = self._runtime_states.get(label)
        status = "not generated" if runtime is None else runtime.status or "not generated"
        locked = self._runtime_controls_locked(status)
        label_item = target.item(row, self._LABEL_COLUMN)
        if label_item is not None:
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if not locked:
                flags |= Qt.ItemFlag.ItemIsEditable
                label_item.setToolTip("")
            else:
                label_item.setToolTip("Job label is locked once this row has been queued for execution.")
            label_item.setFlags(flags)
        if isinstance(supercell_combo, QComboBox):
            supercell_combo.setEnabled(not locked)
        remove_button = target.cellWidget(row, self._REMOVE_COLUMN)
        if isinstance(remove_button, QToolButton):
            remove_button.setEnabled(True)
        if runtime is None:
            self._set_status_item(row, status, table=target)
            self._set_fractional_items(row, None, table=target)
            self._set_float_item(row, self._TOTAL_ENERGY_COLUMN, None, table=target)
        else:
            self._set_status_item(row, status, table=target)
            self._set_fractional_items(row, runtime.final_fractional_position, table=target)
            self._set_float_item(row, self._TOTAL_ENERGY_COLUMN, runtime.total_energy, table=target)
        configure_button = target.cellWidget(row, self._CONFIGURE_COLUMN)
        if isinstance(configure_button, QPushButton):
            customized = bool(self._row_parameter_values.get(row_id, {}))
            style_workflow_configure_button(
                configure_button,
                workflow_label="CASTEP",
                configured=customized,
                locked=locked,
                noun="row",
                used=True,
            )

    def _set_status_item(self, row: int, status: str, *, table: QTableWidget | None = None) -> None:
        target = table or self._table
        item = target.item(row, self._STATUS_COLUMN)
        if item is None:
            item = self._make_item(status, editable=False)
            target.setItem(row, self._STATUS_COLUMN, item)
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
        table: QTableWidget | None = None,
    ) -> None:
        target = table or self._table
        item = target.item(row, column)
        text = "" if value is None else format(float(value), formatter)
        if item is None:
            target.setItem(row, column, self._make_item(text, editable=False))
        else:
            item.setText(text)

    def _set_fractional_items(
        self,
        row: int,
        values: tuple[float, float, float] | None,
        *,
        table: QTableWidget | None = None,
    ) -> None:
        for column, value in zip(
            (self._FINAL_FX_COLUMN, self._FINAL_FY_COLUMN, self._FINAL_FZ_COLUMN),
            values or (None, None, None),
            strict=True,
        ):
            self._set_float_item(row, column, value, formatter=".3f", table=table)

    def _refresh_all_rows(self) -> None:
        for table in self._all_tables():
            table.blockSignals(True)
            for row in range(table.rowCount()):
                self._refresh_row(row, table=table)
            table.blockSignals(False)

    def _selected_row_ids(self) -> tuple[int, ...]:
        table = self._active_table()
        selection_model = table.selectionModel()
        if selection_model is None:
            return ()
        ids: list[int] = []
        for index in selection_model.selectedRows():
            row_id = self._row_id_at(index.row(), table=table)
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

    def _delete_selected_rows(self, table: QTableWidget | None = None) -> None:
        target = table or self._active_table()
        if target.state() == QAbstractItemView.State.EditingState:
            return
        selection_model = target.selectionModel()
        if selection_model is None:
            return
        row_ids: list[int] = []
        for index in selection_model.selectedRows():
            row_id = self._row_id_at(index.row(), table=target)
            if row_id is not None:
                row_ids.append(row_id)
        if not row_ids and target.currentRow() >= 0:
            row_id = self._row_id_at(target.currentRow(), table=target)
            if row_id is not None:
                row_ids.append(row_id)
        if not row_ids:
            return
        self._remove_rows_by_id(tuple(row_ids))

    def _common_params_for_rows(self, row_ids: tuple[int, ...]) -> dict[str, object]:
        if not row_ids:
            return dict(self._default_params)
        maps = [self._effective_params_for_row(row_id) for row_id in row_ids]
        common = dict(maps[0])
        for candidate in maps[1:]:
            common = {name: value for name, value in common.items() if name in candidate and candidate[name] == value}
        common["task"] = infer_task(_WORKFLOW_STAGE)
        # Dialog-only pre-fill of the inactive half of the kpoint pair. See
        # PristineRelaxationWidget._common_params_for_rows for the rationale.
        has_grid = "kpoint_mp_grid" in common
        has_spacing = "kpoint_mp_spacing" in common
        if has_grid and not has_spacing:
            suggested_spacing = self._suggested_spacing_for_row(row_ids[0])
            if suggested_spacing is not None:
                common["kpoint_mp_spacing"] = suggested_spacing
        elif has_spacing and not has_grid:
            suggested_grid = self._suggested_grid_for_row(row_ids[0])
            if suggested_grid is not None:
                common["kpoint_mp_grid"] = suggested_grid
        return common

    def _advanced_names_for_rows(self, row_ids: tuple[int, ...]) -> set[str]:
        names: set[str] = set()
        for row_id in row_ids:
            names.update(self._row_added_advanced.get(row_id, set()))
        return names

    def _row_parameters_locked(self, row_ids: tuple[int, ...]) -> bool:
        for row_id in row_ids:
            location = self._row_location_for_id(row_id)
            if location is None:
                continue
            table, row = location
            label = self._label_at_row(row, table=table)
            charge_state = self._charge_state_at_row(row, table=table)
            runtime = self._runtime_states.get(self._runtime_key(label, charge_state))
            if runtime is None:
                runtime = self._runtime_states.get(label)
            status = "" if runtime is None else runtime.status
            if self._runtime_controls_locked(status):
                return True
        return False

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
        row_locations = [
            location
            for row_id in row_ids
            if (location := self._row_location_for_id(row_id)) is not None
        ]
        intro_labels = [
            f"{self._label_at_row(row, table=table)} ({muon_charge_state_label(self._charge_state_at_row(row, table=table))})"
            for table, row in row_locations
        ]
        charges = {self._charge_state_at_row(row, table=table) for table, row in row_locations}
        charge_text = "Variable (selected charge-state tabs)"
        if len(charges) == 1:
            charge_state = next(iter(charges))
            charge_text = f"{muon_charge_for_state(charge_state):+d} ({muon_charge_state_label(charge_state)})"
        read_only = self._row_parameters_locked(row_ids)
        dialog_kwargs = {
            "specs": self._specs,
            "workflow_stage": _WORKFLOW_STAGE,
            "locked_values": {
                "task": "GEOMETRYOPTIMIZATION",
                "fix_all_cell": "TRUE (required for muon relaxation)",
                "write_cell_structure": "TRUE (required for relaxed-cell outputs)",
                "charge": charge_text,
            },
            "intro_text": "Included rows: " + ", ".join(intro_labels),
            "species_options": self._parameter_species_options,
            "parent": self,
        }
        if read_only:
            dialog_kwargs["read_only"] = True
        dialog = ParameterDialog(**dialog_kwargs)
        dialog.setWindowTitle(
            "Inspect Muon Relaxation Parameters" if read_only else "Configure Muon Relaxation Parameters"
        )
        kpoint_hint = kpoint_exclusive_hint(
            self._effective_params_for_row(r) for r in row_ids
        )
        dialog.set_values(self._common_params_for_rows(row_ids), preferred_exclusive=kpoint_hint)
        dialog.set_advanced_names(self._advanced_names_for_rows(row_ids))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        if read_only:
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
            location = self._row_location_for_id(row_id)
            if location is None:
                continue
            table, row = location
            removed_labels.append(self._submission_key_at_row(row, table=table))
            table.removeRow(row)
            self._row_site_ids.pop(row_id, None)
            self._row_positions.pop(row_id, None)
            self._row_charge_states.pop(row_id, None)
            self._row_supercells.pop(row_id, None)
            self._row_parameter_values.pop(row_id, None)
            self._row_added_advanced.pop(row_id, None)
        if removed_labels:
            self.remove_requested.emit(removed_labels)
            self._refresh_batch_button_state()
            self._emit_state()

    def _build_config_or_warn(self) -> MuonRelaxationConfig | None:
        if not any(table.rowCount() for table in self._all_tables()):
            QMessageBox.warning(self, "No jobs", "Create one or more jobs from generated muon sites first.")
            return None
        rows: list[MuonRelaxationRow] = []
        seen: set[str] = set()
        for table in self._all_tables():
            for row_index in range(table.rowCount()):
                row_id = self._row_id_at(row_index, table=table)
                if row_id is None:
                    continue
                label = self._label_at_row(row_index, table=table)
                if not label:
                    QMessageBox.warning(self, "Invalid label", "Job labels must not be empty.")
                    return None
                charge_state = self._charge_state_at_row(row_index, table=table)
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

    def _submission_key_at_row(self, row: int, *, table: QTableWidget | None = None) -> str:
        target = table or self._table
        return self._runtime_key(self._label_at_row(row, table=target), self._charge_state_at_row(row, table=target))

    def _selected_or_all_labels(self) -> list[str]:
        selected: list[str] = []
        table = self._active_table()
        selection_model = table.selectionModel()
        if selection_model is not None:
            for index in selection_model.selectedRows():
                label = self._label_at_row(index.row(), table=table)
                if label:
                    selected.append(self._submission_key_at_row(index.row(), table=table))
        if selected:
            return selected

        labels: list[str] = []
        for row in range(table.rowCount()):
            label = self._label_at_row(row, table=table)
            if label:
                labels.append(self._submission_key_at_row(row, table=table))
        return labels

    def _selected_labels_only(self) -> list[str]:
        selected: list[str] = []
        table = self._active_table()
        selection_model = table.selectionModel()
        if selection_model is None:
            return selected
        for index in selection_model.selectedRows():
            label = self._label_at_row(index.row(), table=table)
            if label:
                selected.append(self._submission_key_at_row(index.row(), table=table))
        return selected

    def _create_analysis_workspace_from_selection(self) -> None:
        labels = self._selected_labels_only()
        if not labels:
            QMessageBox.information(
                self,
                "No relaxations selected",
                "Select one or more completed structural relaxation rows first.",
            )
            return
        self.analysis_workspace_requested.emit(labels)

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
        for table in self._all_tables():
            for row in range(table.rowCount()):
                row_id = self._row_id_at(row, table=table)
                if row_id is None:
                    continue
                rows.append(
                    {
                        "site_id": str(self._row_site_ids[row_id]),
                        "label": self._label_at_row(row, table=table),
                        "fractional_position": list(self._row_positions[row_id]),
                        "charge_state": self._charge_state_at_row(row, table=table),
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
            for table in self._all_tables():
                table.blockSignals(True)
                table.setRowCount(0)
            self._next_row_id = 1
            self._row_site_ids.clear()
            self._row_positions.clear()
            self._row_charge_states.clear()
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
                    label=str(entry.get("label", "")).strip() or f"site_{self._table_for_charge_state(str(entry.get('charge_state', POSITIVE_MUON_CHARGE_STATE))).rowCount() + 1}",
                    fractional=(float(fractional[0]), float(fractional[1]), float(fractional[2])),
                    charge_state=str(entry.get("charge_state", POSITIVE_MUON_CHARGE_STATE)),
                )
                location = self._row_location_for_id(self._next_row_id - 1)
                row_id = None if location is None else self._row_id_at(location[1], table=location[0])
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
            for table in self._all_tables():
                table.blockSignals(False)
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

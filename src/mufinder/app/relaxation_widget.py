"""Pristine-relaxation configuration panel.

Rows whose job has left the unlocked runtime statuses keep their inline
parameter lock (``_runtime_controls_locked``): a submitted row's inputs must not
drift away from the directory that was rendered from them. Editing a terminal
row's parameters goes through the "Continue relaxation…" path instead, which
mutates the active definition's row in place via
:meth:`~mufinder.app.workflow_controllers.pristine_relaxation.PristineRelaxationController.update_row_parameters`
and mirrors the edit back into this form, preserving the row→job binding.
"""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from PySide6.QtCore import Qt, Signal
from mufinder.app.selection_snapshot import (
    SelectionSnapshotComboBox,
    SelectionSnapshotPushButton,
    SelectionSnapshotToolButton,
    resolve_target_keys,
)

from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
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

from mufinder.app.convergence_display import format_cutoff_energy, format_kpoint_grid
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.relaxation_adapter import suggested_grid_for_row
from mufinder.app.workflow_parameters import kpoint_exclusive_hint, reorder_workflow_parameter_specs
from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow, RelaxationRunState, SupercellMatrix
from mufinder.app.theme import status_brush, style_hint_text, style_section_subtitle, style_section_title
from mufinder.app.workflow_table_utils import (
    RowMenuAction,
    bind_delete_shortcuts,
    bind_reset_failed_context_menu,
    create_workflow_table,
    style_workflow_configure_button,
    workflow_row_status_text,
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


_EXCLUDED_PARAMETER_NAMES = {"task", "fix_all_cell"}
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


def _append_workflow_stage(values: tuple[str, ...], workflow_stage: str) -> tuple[str, ...]:
    if workflow_stage in values:
        return values
    return (*values, workflow_stage)


def _clone_spec_for_pristine_workflow(
    spec: KeywordSpec,
    *,
    basic: bool,
    advanced: bool,
) -> KeywordSpec:
    basic_workflows = spec.basic_workflows
    advanced_workflows = spec.advanced_workflows

    if basic:
        basic_workflows = _append_workflow_stage(basic_workflows, "pristine_relaxation")
    if advanced:
        advanced_workflows = _append_workflow_stage(advanced_workflows, "pristine_relaxation")

    default_value = 300 if spec.name == "geom_max_iter" else spec.default_value
    if spec.name == "continuation":
        default_value = "DEFAULT"

    return replace(
        spec,
        basic_workflows=basic_workflows,
        advanced_workflows=advanced_workflows,
        default_value=default_value,
    )


def _reorder_pristine_specs(specs: list[KeywordSpec]) -> list[KeywordSpec]:
    return reorder_workflow_parameter_specs(specs)


class PristineRelaxationWidget(QWidget):
    """UI for configuring pristine-relaxation jobs with per-row supercells."""

    _LABEL_COLUMN = 0
    _SUPERCELL_COLUMN = 1
    _FIX_CELL_COLUMN = 2
    _STATUS_COLUMN = 3
    _LATTICE_A_COLUMN = 4
    _LATTICE_B_COLUMN = 5
    _LATTICE_C_COLUMN = 6
    _CONFIGURE_COLUMN = 7
    _REMOVE_COLUMN = 8

    configure_requested = Signal(object)
    submit_local_requested = Signal(object)
    submit_cluster_requested = Signal(object)
    generate_comparison_requested = Signal()
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    remove_requested = Signal(object)
    row_supercell_edit_requested = Signal(object)
    form_state_changed = Signal(dict)
    #: Emits the row's label when "Continue relaxation…" is chosen from the
    #: row context menu. See :meth:`set_continuable_predicate` for how the
    #: menu decides whether that action is enabled for a given row.
    continue_relaxation_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._restoring_form_state = False
        self._next_row_id = 1

        # Injected by the controller wiring (see MainWindow's
        # ``set_continuable_predicate`` call): resolves a row's job through the
        # controller and checks its real status, never the row's displayed
        # status text (see the "Continue relaxation…" menu-gating rule in
        # ``mufinder.app.workflow_table_utils.RowMenuAction``). Defaults to
        # "never continuable" so an unwired widget (e.g. in a test) never
        # offers an action it cannot fulfil.
        self._continuable_predicate: Callable[[object], bool] = lambda _key: False

        self._host_lattice: list[list[float]] | None = None
        self._reference_lattice: list[list[float]] | None = None
        self._reference_grid: tuple[int, int, int] | None = None

        self._row_supercells: dict[int, SupercellMatrix | None] = {}
        self._row_parameter_values: dict[int, dict[str, object]] = {}
        self._row_added_advanced: dict[int, set[str]] = {}
        self._runtime_states: dict[str, RelaxationRunState] = {}
        self._supercell_options: list[tuple[str, SupercellMatrix]] = []
        self._parameter_species_options: tuple[str, ...] = ()

        self._name_prefix = QLineEdit("pristine_relax")
        self._project_default_cutoff = QLabel("Not set")
        self._project_default_kpoint = QLabel("Not set")
        self._kpoint_message = QLabel("")
        self._kpoint_message.setWordWrap(True)
        style_hint_text(self._kpoint_message)
        self._kpoint_message.hide()

        self._table = create_workflow_table(
            (
                "Label",
                "Supercell",
                "Fix Cell",
                "Status",
                "a (Ang)",
                "b (Ang)",
                "c (Ang)",
                "Parameters",
                "",
            )
        )
        # Per-widget settings the shared helper does not cover (or that this
        # editable table overrides): extended multi-select, in-place editing,
        # a fixed row height, per-column resize modes, and a non-stretching
        # last (Remove) column.
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._table.verticalHeader().setDefaultSectionSize(32)
        self._table_delete_shortcut, self._table_backspace_shortcut = bind_delete_shortcuts(
            self, self._delete_selected_rows
        )

        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(self._LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._SUPERCELL_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._FIX_CELL_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._STATUS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._LATTICE_A_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._LATTICE_B_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._LATTICE_C_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._CONFIGURE_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._REMOVE_COLUMN, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(self._REMOVE_COLUMN, 34)

        self._add_row_button = QPushButton("Add Job")
        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._batch_configure_button.setEnabled(False)

        self._run_local_button = QPushButton("Run Local Calculations")
        self._run_cluster_button = QPushButton("Run on Cluster")
        self._generate_comparison_button = QPushButton("Generate Comparison Table...")

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

        title = QLabel("Pristine Relaxation")
        style_section_title(title)
        subtitle = QLabel(
            "Create labelled geometry-optimization jobs with per-job supercells and cell constraints."
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

        layout.addWidget(self._kpoint_message)

        table_group = QGroupBox("Relaxation Jobs")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 8, 10, 8)
        table_layout.addWidget(self._table)

        control_row = QHBoxLayout()
        control_row.addWidget(self._add_row_button)
        control_row.addStretch(1)
        control_row.addWidget(self._batch_configure_button)
        table_layout.addLayout(control_row)
        layout.addWidget(table_group, 1)

        run_group = QGroupBox("Run Jobs")
        run_layout = QVBoxLayout(run_group)
        run_layout.setContentsMargins(10, 8, 10, 8)
        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Job name prefix"))
        run_row.addWidget(self._name_prefix, 1)
        run_row.addWidget(self._run_local_button)
        run_row.addWidget(self._run_cluster_button)
        run_row.addWidget(self._generate_comparison_button)
        run_layout.addLayout(run_row)
        layout.addWidget(run_group)

    def _connect_signals(self) -> None:
        self._add_row_button.clicked.connect(self._add_row)
        self._batch_configure_button.clicked.connect(self._configure_selected_rows)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._table.setToolTip(
            "Right-click a failed, stopped, or cancelled row to reset it back to ready for "
            "rerun, or a completed/failed/stopped/cancelled row to continue its relaxation."
        )
        bind_reset_failed_context_menu(
            self._table,
            key_for_row=self._reset_key_at_row,
            status_for_row=self._row_status,
            emit_reset=self.reset_failed_requested.emit,
            additional_actions=(
                RowMenuAction(
                    text="Continue relaxation…",
                    is_enabled=lambda keys: len(keys) == 1
                    and self._continuable_predicate(keys[0]),
                    trigger=lambda keys: self.continue_relaxation_requested.emit(keys[0]),
                ),
            ),
        )

        selection_model = self._table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda *_: self._refresh_batch_button_state())

        self._name_prefix.textChanged.connect(self._emit_state)
        self._run_local_button.clicked.connect(self._run_local_selected_rows)
        self._run_cluster_button.clicked.connect(self._run_cluster_selected_rows)
        self._generate_comparison_button.clicked.connect(self.generate_comparison_requested.emit)

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
            cloned = _clone_spec_for_pristine_workflow(
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
                _clone_spec_for_pristine_workflow(
                    spec,
                    basic=True,
                    advanced=True,
                )
            )
            seen_names.add(spec.name)

        return _reorder_pristine_specs(specs)

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

    def _default_shared_params(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {"task": infer_task("pristine_relaxation")}
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
            if not keyword_visible_in_workflow(
                spec,
                workflow_stage="pristine_relaxation",
                include_advanced=False,
            ):
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
        self._reference_lattice = (
            [list(row) for row in kpoint_reference_lattice]
            if isinstance(kpoint_reference_lattice, list)
            else None
        )
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

    def set_runtime_states(self, states: Mapping[str, RelaxationRunState]) -> None:
        self._runtime_states = {str(label): state for label, state in states.items()}
        self._refresh_all_rows()

    def set_continuable_predicate(self, predicate: Callable[[object], bool]) -> None:
        """Inject the live "can this row's job be continued?" check.

        Called by the controller wiring with ``PristineRelaxationController.row_is_continuable``.
        Evaluated fresh every time the row context menu opens, not cached, so a
        status change since the last refresh (e.g. a dashboard dismissal) is
        always reflected in whether "Continue relaxation…" is enabled.
        """

        self._continuable_predicate = predicate

    def set_supercell_options(self, options: list[dict[str, object]]) -> None:
        parsed: list[tuple[str, SupercellMatrix]] = []
        for entry in options:
            if not isinstance(entry, dict):
                continue
            label = str(entry.get("label", "")).strip()
            if not label:
                continue
            matrix = self._coerce_supercell_matrix(entry.get("matrix"))
            if matrix is None:
                continue
            parsed.append((label, matrix))
        self._supercell_options = parsed
        self._refresh_all_rows()

    def set_parameter_species_options(self, species_options: Sequence[str]) -> None:
        self._parameter_species_options = tuple(
            dict.fromkeys(option for item in species_options if (option := str(item).strip()))
        )

    def row_supercell_matrix(self, row_id: int) -> SupercellMatrix | None:
        return self._row_supercells.get(int(row_id))

    def set_row_supercell(
        self,
        row_id: int,
        matrix: SupercellMatrix | None,
        *,
        message: str | None = None,
    ) -> None:
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

    def clear_kpoint_message(self) -> None:
        self._kpoint_message.hide()
        self._kpoint_message.setText("")

    def _make_item(self, text: str, *, editable: bool) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        return item

    def _new_row_label(self) -> str:
        return f"relax_{self._table.rowCount() + 1}"

    def _add_row(self) -> None:
        row = self._table.rowCount()
        row_id = self._next_row_id
        self._next_row_id += 1

        self._row_supercells[row_id] = None
        self._row_parameter_values[row_id] = {}
        self._row_added_advanced[row_id] = set()

        self._table.blockSignals(True)
        self._table.insertRow(row)

        label_item = self._make_item(self._new_row_label(), editable=True)
        label_item.setData(Qt.ItemDataRole.UserRole, row_id)
        self._table.setItem(row, self._LABEL_COLUMN, label_item)

        supercell_combo = SelectionSnapshotComboBox(self._table)
        supercell_combo.selection_snapshot_key = row_id
        supercell_combo.currentIndexChanged.connect(
            lambda _index, rid=row_id, combo=supercell_combo: self._on_supercell_combo_changed(rid, combo)
        )
        self._table.setCellWidget(row, self._SUPERCELL_COLUMN, supercell_combo)

        fix_combo = QComboBox(self._table)
        fix_combo.addItem("Yes (fixed)", True)
        fix_combo.addItem("No (relax)", False)
        default_index = fix_combo.findData(False)
        if default_index >= 0:
            fix_combo.setCurrentIndex(default_index)
        fix_combo.currentIndexChanged.connect(lambda *_: self._emit_state())
        self._table.setCellWidget(row, self._FIX_CELL_COLUMN, fix_combo)

        self._table.setItem(row, self._STATUS_COLUMN, self._make_item("not generated", editable=False))
        self._table.setItem(row, self._LATTICE_A_COLUMN, self._make_item("", editable=False))
        self._table.setItem(row, self._LATTICE_B_COLUMN, self._make_item("", editable=False))
        self._table.setItem(row, self._LATTICE_C_COLUMN, self._make_item("", editable=False))

        configure_button = SelectionSnapshotPushButton("Configure...", self._table)
        configure_button.selection_snapshot_key = row_id
        configure_button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        configure_button.clicked.connect(
            lambda _checked=False, rid=row_id, btn=configure_button: self._dispatch_row_action(btn, rid, "configure")
        )
        self._table.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

        remove_button = SelectionSnapshotToolButton(self._table)
        remove_button.selection_snapshot_key = row_id
        remove_button.setText("x")
        remove_button.setAutoRaise(True)
        remove_button.clicked.connect(
            lambda _checked=False, rid=row_id, btn=remove_button: self._dispatch_row_action(btn, rid, "remove")
        )
        self._table.setCellWidget(row, self._REMOVE_COLUMN, remove_button)

        self._refresh_row(row)
        self._table.blockSignals(False)
        self._refresh_batch_button_state()
        self._emit_state()

    def _row_index_for_id(self, row_id: int) -> int | None:
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._LABEL_COLUMN)
            if item is None:
                continue
            if int(item.data(Qt.ItemDataRole.UserRole)) == int(row_id):
                return row
        return None

    def _row_id_at(self, row: int) -> int | None:
        item = self._table.item(row, self._LABEL_COLUMN)
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        if value is None:
            return None
        return int(value)

    def _label_at_row(self, row: int) -> str:
        item = self._table.item(row, self._LABEL_COLUMN)
        return item.text().strip() if item is not None else ""

    def _reset_key_at_row(self, row: int) -> str | None:
        """Return the row's submission key, or ``None`` for an unlabeled row."""

        return self._label_at_row(row) or None

    def _row_id_for_label(self, label: str) -> int | None:
        for row in range(self._table.rowCount()):
            if self._label_at_row(row) == label:
                return self._row_id_at(row)
        return None

    def _row_status(self, row: int) -> str:
        return workflow_row_status_text(self._table, row, self._STATUS_COLUMN)

    def _fix_all_cell_at_row(self, row: int) -> bool:
        combo = self._table.cellWidget(row, self._FIX_CELL_COLUMN)
        if isinstance(combo, QComboBox):
            return bool(combo.currentData())
        return False

    @staticmethod
    def _coerce_supercell_matrix(value: object) -> SupercellMatrix | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            matrix = (
                (int(value[0][0]), int(value[0][1]), int(value[0][2])),
                (int(value[1][0]), int(value[1][1]), int(value[1][2])),
                (int(value[2][0]), int(value[2][1]), int(value[2][2])),
            )
        except (TypeError, ValueError, IndexError):
            return None
        return matrix

    def _set_kpoint_message_for_row(self, row_id: int) -> None:
        current_matrix = self._row_supercells.get(row_id)
        if current_matrix is None or self._reference_grid is None or self._host_lattice is None:
            self.clear_kpoint_message()
            return
        suggested = suggested_grid_for_row(
            host_lattice=self._host_lattice,
            row_matrix=current_matrix,
            reference_lattice=self._reference_lattice,
            reference_grid=self._reference_grid,
        )
        if suggested is None or suggested == self._reference_grid:
            self.clear_kpoint_message()
            return
        self._kpoint_message.setText(
            "K-point grid adjusted from "
            f"{self._reference_grid[0]}x{self._reference_grid[1]}x{self._reference_grid[2]} to "
            f"{suggested[0]}x{suggested[1]}x{suggested[2]} for equivalent density."
        )
        self._kpoint_message.show()

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
        row = self._row_index_for_id(row_id)
        if row is None:
            return
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

        snapshot_keys = [
            self._row_id_at(r)
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
            target_row = self._row_index_for_id(target_row_id)
            if target_row is not None:
                self._refresh_row(target_row)
        self._set_kpoint_message_for_row(row_id)
        self._emit_state()

    def _dispatch_row_action(
        self,
        widget: SelectionSnapshotPushButton | SelectionSnapshotToolButton,
        row_id: int,
        kind: str,
    ) -> None:
        snapshot_keys = [
            self._row_id_at(r)
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
        row = self._row_index_for_id(row_id)
        if row is None:
            return False
        label = self._label_at_row(row)
        runtime = self._runtime_states.get(label)
        status = "not generated" if runtime is None else runtime.status or "not generated"
        return not self._runtime_controls_locked(status)

    def _set_status_item(self, row: int, status: str) -> None:
        item = self._table.item(row, self._STATUS_COLUMN)
        if item is None:
            item = self._make_item(status, editable=False)
            self._table.setItem(row, self._STATUS_COLUMN, item)
        else:
            item.setText(status)
        item.setForeground(status_brush(status))

    def _supercell_label(self, matrix: SupercellMatrix | None) -> str:
        if matrix is None or matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            return "Unit cell"
        is_diagonal = (
            matrix[0][1] == 0
            and matrix[0][2] == 0
            and matrix[1][0] == 0
            and matrix[1][2] == 0
            and matrix[2][0] == 0
            and matrix[2][1] == 0
        )
        if is_diagonal:
            return f"{matrix[0][0]}x{matrix[1][1]}x{matrix[2][2]}"
        return "Custom 3x3"

    def _suggested_grid_for_row(self, row_id: int) -> tuple[int, int, int] | None:
        if self._host_lattice is None:
            return None
        grid = suggested_grid_for_row(
            host_lattice=self._host_lattice,
            row_matrix=self._row_supercells.get(row_id),
            reference_lattice=self._reference_lattice,
            reference_grid=self._reference_grid,
        )
        return grid

    @staticmethod
    def _runtime_controls_locked(status: str) -> bool:
        return str(status).strip().casefold() not in _UNLOCKED_RUNTIME_STATUSES

    def _suggested_spacing_for_row(self, row_id: int) -> ValueWithUnit | None:
        # When the project has adopted a k-point grid, mirror it as a k-point
        # spacing so the configure-parameters dialog's spacing field gets a
        # sensible default when the user toggles to "Use spacing".
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
        # The row's "effective" values reflect what would land in the rendered
        # .cell: only ONE of kpoint_mp_grid / kpoint_mp_spacing, never both,
        # since they are mutually exclusive at the CASTEP level. The
        # configure-parameters dialog renders both rows with a radio toggle
        # and drops the inactive choice in collect_values(), so the row's
        # stored parameter_values is also single-choice. When neither has
        # been set yet we default to grid (matching the radio default).
        values = dict(self._default_params)
        values.update(self._row_parameter_values.get(row_id, {}))
        values.setdefault("task", infer_task("pristine_relaxation"))
        if (
            "kpoint_mp_grid" not in values
            and "kpoint_mp_spacing" not in values
        ):
            suggested = self._suggested_grid_for_row(row_id)
            if suggested is not None:
                values["kpoint_mp_grid"] = suggested
        return values

    def _implicit_params_for_row(self, row_id: int) -> dict[str, object]:
        implicit = dict(self._default_params)
        implicit.setdefault("task", infer_task("pristine_relaxation"))
        # Mirror _effective_params_for_row's single-choice contract: only
        # inject kpoint_mp_grid as the implicit default when the row has not
        # been explicitly configured to use kpoint_mp_spacing instead.
        row_values = self._row_parameter_values.get(row_id, {})
        if "kpoint_mp_spacing" not in row_values:
            suggested = self._suggested_grid_for_row(row_id)
            if suggested is not None:
                implicit["kpoint_mp_grid"] = suggested
        return implicit

    def _strip_implicit_row_defaults(self, row_id: int, values: dict[str, object]) -> dict[str, object]:
        cleaned = self._sanitize_parameter_mapping(dict(values))
        implicit = self._implicit_params_for_row(row_id)

        stripped: dict[str, object] = {}
        for name, value in cleaned.items():
            if name not in self._specs_by_name and name != "task":
                continue
            if name in implicit and value == implicit[name]:
                continue
            stripped[name] = value

        return stripped

    def _normalize_all_row_parameter_values(self) -> None:
        for row_id in tuple(self._row_parameter_values):
            self._row_parameter_values[row_id] = self._strip_implicit_row_defaults(
                row_id,
                self._row_parameter_values.get(row_id, {}),
            )

    def _refresh_row(self, row: int) -> None:
        row_id = self._row_id_at(row)
        if row_id is None:
            return

        supercell_combo = self._table.cellWidget(row, self._SUPERCELL_COLUMN)
        if isinstance(supercell_combo, QComboBox):
            self._populate_supercell_combo(supercell_combo, row_id)

        label = self._label_at_row(row)
        runtime = self._runtime_states.get(label)
        status = "not generated" if runtime is None else runtime.status or "not generated"
        locked = self._runtime_controls_locked(status)
        label_item = self._table.item(row, self._LABEL_COLUMN)
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
        fix_combo = self._table.cellWidget(row, self._FIX_CELL_COLUMN)
        if isinstance(fix_combo, QComboBox):
            fix_combo.setEnabled(not locked)
        remove_button = self._table.cellWidget(row, self._REMOVE_COLUMN)
        if isinstance(remove_button, QToolButton):
            remove_button.setEnabled(True)
        if runtime is None:
            self._set_status_item(row, "not generated")
            self._set_float_item(row, self._LATTICE_A_COLUMN, None)
            self._set_float_item(row, self._LATTICE_B_COLUMN, None)
            self._set_float_item(row, self._LATTICE_C_COLUMN, None)
        else:
            self._set_status_item(row, runtime.status or "not generated")
            self._set_float_item(row, self._LATTICE_A_COLUMN, runtime.lattice_a)
            self._set_float_item(row, self._LATTICE_B_COLUMN, runtime.lattice_b)
            self._set_float_item(row, self._LATTICE_C_COLUMN, runtime.lattice_c)

        configure_button = self._table.cellWidget(row, self._CONFIGURE_COLUMN)
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

    def _set_float_item(self, row: int, column: int, value: float | None) -> None:
        item = self._table.item(row, column)
        text = "" if value is None else f"{float(value):.8g}"
        if item is None:
            item = self._make_item(text, editable=False)
            self._table.setItem(row, column, item)
        else:
            item.setText(text)

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

    def _delete_selected_rows(self) -> None:
        if self._table.state() == QAbstractItemView.State.EditingState:
            return
        row_ids = self._selected_row_ids()
        if not row_ids and self._table.currentRow() >= 0:
            row_id = self._row_id_at(self._table.currentRow())
            if row_id is not None:
                row_ids = (row_id,)
        if not row_ids:
            return
        self._remove_rows_by_id(row_ids)

    def _common_params_for_rows(self, row_ids: tuple[int, ...]) -> dict[str, object]:
        if not row_ids:
            return dict(self._default_params)
        maps = [self._effective_params_for_row(row_id) for row_id in row_ids]
        common = dict(maps[0])
        for candidate in maps[1:]:
            common = {
                name: value
                for name, value in common.items()
                if name in candidate and candidate[name] == value
            }
        if "task" not in common:
            common["task"] = infer_task("pristine_relaxation")
        # Configure-parameters dialog seeding only: pre-fill the inactive
        # half of the kpoint_mp_grid / kpoint_mp_spacing pair so the editor
        # for whichever radio the user toggles to shows a sensible value
        # derived from the project default. The dialog drops the inactive
        # choice in collect_values(), so this never reaches the rendered
        # .cell or the validation path.
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
            row = self._row_index_for_id(row_id)
            if row is None:
                continue
            runtime = self._runtime_states.get(self._label_at_row(row))
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

    def continue_relaxation_dialog_seed(self, label: object) -> dict[str, object] | None:
        """Return the ``ContinueRelaxationDialog`` seed kwargs for one row.

        Mirrors :meth:`_open_row_parameter_dialog`'s single-row seeding (the
        row's effective parameter values, its added-advanced set, and the same
        locked task/fix-cell/write-cell-structure keywords), for the one row
        the Continue action targets. MainWindow supplies everything this
        widget cannot know (the checkpoint probe, the rendered old/new
        ``.param`` text, and the job's execution target/directory) alongside
        these kwargs when constructing the dialog. Returns ``None`` when
        ``label`` does not resolve to a row.
        """

        row_id = self._row_id_for_label(str(label))
        if row_id is None:
            return None
        return {
            "specs": self._specs,
            "workflow_stage": "pristine_relaxation",
            "parameter_values": self._effective_params_for_row(row_id),
            "added_advanced": tuple(sorted(self._row_added_advanced.get(row_id, set()))),
            "locked_values": {
                "task": "GEOMETRYOPTIMIZATION",
                "fix_all_cell": "Variable (set by row control)",
                "write_cell_structure": "TRUE (required for relaxed-cell outputs)",
            },
            "species_options": self._parameter_species_options,
        }

    def _open_row_parameter_dialog(self, row_ids: tuple[int, ...]) -> None:
        intro_labels: list[str] = []
        for row_id in row_ids:
            row = self._row_index_for_id(row_id)
            if row is None:
                continue
            intro_labels.append(self._label_at_row(row))

        read_only = self._row_parameters_locked(row_ids)
        dialog_kwargs = {
            "specs": self._specs,
            "workflow_stage": "pristine_relaxation",
            "locked_values": {
                "task": "GEOMETRYOPTIMIZATION",
                "fix_all_cell": "Variable (set by row control)",
                "write_cell_structure": "TRUE (required for relaxed-cell outputs)",
            },
            "intro_text": "Included rows: " + ", ".join(intro_labels),
            "species_options": self._parameter_species_options,
            "parent": self,
        }
        if read_only:
            dialog_kwargs["read_only"] = True
        dialog = ParameterDialog(**dialog_kwargs)
        dialog.setWindowTitle("Inspect Relaxation Parameters" if read_only else "Configure Relaxation Parameters")
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
            row = self._row_index_for_id(row_id)
            if row is None:
                continue
            removed_labels.append(self._label_at_row(row))
            self._table.removeRow(row)
            self._row_supercells.pop(row_id, None)
            self._row_parameter_values.pop(row_id, None)
            self._row_added_advanced.pop(row_id, None)

        if removed_labels:
            self.remove_requested.emit(removed_labels)
            self._refresh_batch_button_state()
            self._emit_state()

    def _build_config_or_warn(self) -> PristineRelaxationConfig | None:
        if self._table.rowCount() == 0:
            QMessageBox.warning(self, "No jobs", "Add one or more relaxation rows first.")
            return None

        rows: list[PristineRelaxationRow] = []
        seen_labels: set[str] = set()
        for row_index in range(self._table.rowCount()):
            row_id = self._row_id_at(row_index)
            if row_id is None:
                continue
            label = self._label_at_row(row_index)
            if not label:
                QMessageBox.warning(self, "Invalid label", "Job labels must not be empty.")
                return None
            if label in seen_labels:
                QMessageBox.warning(self, "Duplicate label", f"Duplicate job label '{label}'.")
                return None
            seen_labels.add(label)

            effective_values = self._effective_params_for_row(row_id)
            task = str(effective_values.get("task", infer_task("pristine_relaxation")))
            errors = validate_gui_values(effective_values, self._specs_by_name, task=task)
            if errors:
                QMessageBox.warning(
                    self,
                    "Invalid parameters",
                    f"Row '{label}':\n" + "\n".join(errors),
                )
                return None

            rows.append(
                PristineRelaxationRow(
                    label=label,
                    fix_all_cell=self._fix_all_cell_at_row(row_index),
                    supercell_matrix=self._row_supercells.get(row_id),
                    parameter_values=dict(self._row_parameter_values.get(row_id, {})),
                    added_advanced=tuple(sorted(self._row_added_advanced.get(row_id, set()))),
                )
            )

        return PristineRelaxationConfig(
            name_prefix=self._name_prefix.text().strip() or "pristine_relax",
            rows=tuple(rows),
        )

    def _selected_or_all_labels(self) -> list[str]:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return []
        selected_rows = [index.row() for index in selection_model.selectedRows()]
        if not selected_rows:
            selected_rows = list(range(self._table.rowCount()))
        labels: list[str] = []
        for row in selected_rows:
            label = self._label_at_row(row)
            if label:
                labels.append(label)
        return labels

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
                    "label": self._label_at_row(row),
                    "fix_all_cell": self._fix_all_cell_at_row(row),
                    "supercell_matrix": (
                        [list(line) for line in self._row_supercells.get(row_id)]
                        if self._row_supercells.get(row_id) is not None
                        else None
                    ),
                    "params": dict(self._row_parameter_values.get(row_id, {})),
                    "added_advanced": sorted(self._row_added_advanced.get(row_id, set())),
                }
            )

        return {
            "name_prefix": self._name_prefix.text().strip() or "pristine_relax",
            "rows": rows,
        }

    def set_form_state(self, state: dict[str, Any]) -> None:
        self._restoring_form_state = True
        try:
            self._table.blockSignals(True)
            self._table.setRowCount(0)
            self._next_row_id = 1
            self._row_supercells = {}
            self._row_parameter_values = {}
            self._row_added_advanced = {}

            previous_block = self._name_prefix.blockSignals(True)
            self._name_prefix.setText(str(state.get("name_prefix", "pristine_relax")))
            self._name_prefix.blockSignals(previous_block)

            raw_rows = state.get("rows", [])
            if not isinstance(raw_rows, list):
                raw_rows = []

            for entry in raw_rows:
                if not isinstance(entry, dict):
                    continue
                self._add_row()
                row = self._table.rowCount() - 1
                row_id = self._row_id_at(row)
                if row_id is None:
                    continue

                label = str(entry.get("label", self._new_row_label())).strip()
                label_item = self._table.item(row, self._LABEL_COLUMN)
                if label_item is not None:
                    label_item.setText(label)

                fix_combo = self._table.cellWidget(row, self._FIX_CELL_COLUMN)
                if isinstance(fix_combo, QComboBox):
                    target = bool(entry.get("fix_all_cell", False))
                    index = fix_combo.findData(target)
                    fix_combo.setCurrentIndex(index if index >= 0 else 0)

                raw_matrix = entry.get("supercell_matrix")
                matrix: SupercellMatrix | None = None
                if isinstance(raw_matrix, list) and len(raw_matrix) == 3:
                    try:
                        matrix = (
                            (int(raw_matrix[0][0]), int(raw_matrix[0][1]), int(raw_matrix[0][2])),
                            (int(raw_matrix[1][0]), int(raw_matrix[1][1]), int(raw_matrix[1][2])),
                            (int(raw_matrix[2][0]), int(raw_matrix[2][1]), int(raw_matrix[2][2])),
                        )
                    except (TypeError, ValueError, IndexError):
                        matrix = None
                self._row_supercells[row_id] = matrix

                params = entry.get("params", {})
                if isinstance(params, dict):
                    self._row_parameter_values[row_id] = self._sanitize_parameter_mapping(
                        {str(name): raw for name, raw in params.items()}
                    )
                else:
                    self._row_parameter_values[row_id] = {}

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

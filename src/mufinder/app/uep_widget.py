"""UEP source-job configuration panel and completed-source launch controls."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
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
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import status_brush, style_hint_text, style_section_subtitle, style_section_title
from mufinder.app.uep_types import SupercellMatrix, UepConfig, UepRunState, UepSourceRow
from mufinder.app.project_defaults_widget import ProjectDefaultsStrip
from mufinder.app.workflow_parameters import (
    reorder_workflow_parameter_specs,
    seed_xc_functional_default,
)
from mufinder.app.workflow_table_utils import (
    REMOVE_SELECTED_TEXT,
    SelectionFeedback,
    bind_delete_shortcuts,
    bind_reset_failed_context_menu,
    confirm_row_removal,
    create_workflow_table,
    standard_row_actions,
    style_workflow_configure_button,
    workflow_row_status_text,
)
from mufinder.castep_io.gui_metadata import (
    get_gui_keyword_specs_for_files,
    validate_gui_values,
)
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import ValueWithUnit
from mufinder.app.selection_snapshot import (
    SelectionSnapshotPushButton,
    SelectionSnapshotToolButton,
    resolve_target_keys,
)
from mufinder.app.workflow_dialogs import RowParameterRequest, run_row_parameter_dialog
from mufinder.app.workflow_parameters import STRUCTURE_CONTROLLED_CELL_KEYWORDS
from mufinder.app.workflow_parameters import clone_spec_for_workflow
from mufinder.app.workflow_parameters import sanitize_parameter_mapping
from mufinder.app.workflow_table_utils import workflow_row_is_locked
from mufinder.app.adapter_common import coerce_supercell_matrix
from mufinder.app.workflow_table_utils import SupercellColumn


def _normalize_uep_name_prefix(value: object) -> str:
    prefix = str(value).strip() or "uep"
    return "uep" if prefix == "uep_source" else prefix


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

        self._row_parameter_values: dict[int, dict[str, object]] = {}
        self._row_added_advanced: dict[int, set[str]] = {}
        self._runtime_states: dict[str, UepRunState] = {}
        self._source_structure_options: list[tuple[str, str]] = []
        self._updating_source_structure = False
        self._parameter_species_options: tuple[str, ...] = ()

        self._name_prefix = QLineEdit("uep")
        self.project_defaults_strip = ProjectDefaultsStrip()
        self._source_structure_summary = QLabel("Using the host structure.")
        self._source_structure_summary.setWordWrap(True)
        style_hint_text(self._source_structure_summary)
        self._source_structure_combo = QComboBox()
        self._kpoint_message = QLabel("")
        self._kpoint_message.setWordWrap(True)
        style_hint_text(self._kpoint_message)
        self._kpoint_message.hide()
        self._supercells = SupercellColumn(
            self,
            column=self._SUPERCELL_COLUMN,
            manage_requested=self.row_supercell_edit_requested.emit,
            refresh_rows=self._refresh_table_rows,
            state_changed=self._emit_state,
        )

        self._table = create_workflow_table(
            (
                "Label",
                "Supercell",
                "Status",
                "Total Energy (eV)",
                "3D Potential",
                "Parameters",
                "Analysis",
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

        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
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
        self._remove_selected_button = QPushButton(REMOVE_SELECTED_TEXT)
        self._selection_count_label = QLabel("")
        style_hint_text(self._selection_count_label)
        # SelectionFeedback sets the run buttons' text ("Run All ..." / "Run
        # Selected ...") from the selection.
        self._run_local_button = QPushButton()
        self._run_cluster_button = QPushButton()

        self._specs = self._build_parameter_specs()
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        self._default_params: dict[str, object] = {
            "task": "SINGLEPOINT",
            "symmetry_generate": True,
            "write_formatted_density": True,
        }

        self._build_layout()
        self._connect_signals()
        self._delete_shortcut, self._backspace_shortcut = bind_delete_shortcuts(
            self, self._delete_selected_rows
        )

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

        header_row.addLayout(title_column, 1)
        header_row.addWidget(self.project_defaults_strip, 0, Qt.AlignmentFlag.AlignTop)
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

        # Two rows, as in the relaxation tables: one row cannot hold the
        # selection controls, the job-name prefix and both run buttons at
        # laptop width without crushing the prefix field.
        control_row = QHBoxLayout()
        control_row.addWidget(self._add_row_button)
        control_row.addStretch(1)
        control_row.addWidget(self._selection_count_label)
        control_row.addWidget(self._remove_selected_button)
        control_row.addWidget(self._batch_configure_button)
        table_layout.addLayout(control_row)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Job name prefix"))
        run_row.addWidget(self._name_prefix, 1)
        run_row.addWidget(self._run_local_button)
        run_row.addWidget(self._run_cluster_button)
        table_layout.addLayout(run_row)
        layout.addWidget(table_group, 1)

    def _connect_signals(self) -> None:
        self._add_row_button.clicked.connect(self._add_row)
        self._batch_configure_button.clicked.connect(self._on_batch_configure_clicked)
        self._run_local_button.clicked.connect(self._emit_submit_local)
        self._run_cluster_button.clicked.connect(self._emit_submit_cluster)
        self._remove_selected_button.clicked.connect(self._delete_selected_rows)
        self._table.itemSelectionChanged.connect(self._refresh_batch_button_state)
        self._table.setToolTip(
            "Right-click selected rows to run, cancel, inspect, reset, configure, or remove them."
        )
        # A right-click makes the rows under it the selection, so the menu's
        # run/configure/remove entries reuse the selection-based handlers.
        leading_actions, trailing_actions = standard_row_actions(
            status_for_row=self._row_status,
            run_local=lambda _rows: self._emit_submit_local(),
            run_cluster=lambda _rows: self._emit_submit_cluster(),
            cancel=lambda rows: self.cancel_requested.emit(self._labels_at_rows(rows)),
            inspect=lambda rows: self.inspect_failed_requested.emit(self._labels_at_rows(rows)),
            configure=lambda _rows: self._on_batch_configure_clicked(),
            remove=lambda _rows: self._delete_selected_rows(),
            confirm_parent=self,
        )
        bind_reset_failed_context_menu(
            self._table,
            key_for_row=self._label_at_row,
            status_for_row=self._row_status,
            emit_reset=self.reset_failed_requested.emit,
            leading_actions=leading_actions,
            additional_actions=trailing_actions,
        )
        self._selection_feedback = SelectionFeedback(
            self._table,
            count_label=self._selection_count_label,
            run_local_button=self._run_local_button,
            run_cluster_button=self._run_cluster_button,
            selection_buttons=(self._remove_selected_button,),
        )
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
            if spec.file == "cell" and spec.name in STRUCTURE_CONTROLLED_CELL_KEYWORDS:
                continue
            if spec.name == "write_formatted_density":
                adjusted.append(clone_spec_for_workflow(spec, workflow_stage="uep", basic=True, advanced=True))
                continue
            # kpoint_mp_grid / kpoint_mp_spacing render as a mutually-exclusive
            # radio pair in the configure-parameters dialog. Force both into
            # the basic view so the toggle is visible without expanding the
            # advanced section, matching pristine/muon relaxation.
            if spec.name in ("kpoint_mp_grid", "kpoint_mp_spacing"):
                adjusted.append(clone_spec_for_workflow(spec, workflow_stage="uep", basic=True, advanced=True))
                continue
            adjusted.append(spec)
        return reorder_workflow_parameter_specs(adjusted)

    def set_project_defaults(
        self,
        *,
        cutoff_energy: float | None,
        kpoint_grid: tuple[int, int, int] | None,
        kpoint_reference_lattice: list[list[float]] | None = None,
        xc_functional: str | None = None,
    ) -> None:
        """Seed row defaults from the project defaults (shown by the strip)."""
        with self._suspend_state_emission():
            seed_xc_functional_default(
                self._default_params, xc_functional, schema_default=None
            )
            self._supercells.set_kpoint_reference(kpoint_grid, kpoint_reference_lattice)
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
        self._supercells.set_host_lattice(lattice_vectors)
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

    def set_parameter_species_options(self, species_options: Sequence[str]) -> None:
        self._parameter_species_options = tuple(
            dict.fromkeys(option for item in species_options if (option := str(item).strip()))
        )

    def set_runtime_states(self, states: Mapping[str, UepRunState]) -> None:
        self._runtime_states = {str(label): state for label, state in states.items()}
        self._refresh_table_rows()

    def set_supercell_options(self, options: list[dict[str, object]]) -> None:
        self._supercells.set_options(options)
        self._refresh_table_rows()

    def row_supercell_matrix(self, row_id: int) -> SupercellMatrix | None:
        return self._supercells.matrices.get(int(row_id))

    def set_row_supercell(self, row_id: int, matrix: SupercellMatrix | None) -> None:
        self._supercells.matrices[int(row_id)] = matrix
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

    def _selected_or_current_row_ids(self) -> tuple[int, ...]:
        row_ids = self._selected_row_ids()
        if row_ids:
            return row_ids
        current_row = self._table.currentRow()
        if current_row < 0:
            return ()
        row_id = self._row_id_at(current_row)
        return () if row_id is None else (row_id,)

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
        self._supercells.matrices.setdefault(row_id, None)
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

        self._table.setCellWidget(row, self._SUPERCELL_COLUMN, self._supercells.create_combo(self._table, row_id))

        self._table.setItem(row, self._STATUS_COLUMN, QTableWidgetItem("ready"))
        self._table.setItem(row, self._TOTAL_ENERGY_COLUMN, QTableWidgetItem(""))
        self._table.setItem(row, self._POTENTIAL_GRID_COLUMN, QTableWidgetItem(""))

        configure_button = SelectionSnapshotPushButton("Configure...", self._table)
        configure_button.selection_snapshot_key = row_id
        configure_button.clicked.connect(
            lambda _checked=False, rid=row_id, btn=configure_button: self._dispatch_row_action(btn, rid, "configure")
        )
        self._table.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

        wizard_button = SelectionSnapshotPushButton("Open Wizard", self._table)
        wizard_button.selection_snapshot_key = row_id
        # The wizard runs on one row at a time, so it is never a multi-row target.
        wizard_button.propagates_to_selection = False
        wizard_button.clicked.connect(
            lambda _checked=False, rid=row_id, btn=wizard_button: self._dispatch_row_action(btn, rid, "wizard")
        )
        self._table.setCellWidget(row, self._WIZARD_COLUMN, wizard_button)

        remove_button = SelectionSnapshotToolButton(self._table)
        remove_button.selection_snapshot_key = row_id
        remove_button.setText("×")
        remove_button.clicked.connect(
            lambda _checked=False, rid=row_id, btn=remove_button: self._dispatch_row_action(btn, rid, "remove")
        )
        self._table.setCellWidget(row, self._REMOVE_COLUMN, remove_button)

        self._apply_row_state(row_id)

    def _row_index_for_id(self, row_id: int) -> int | None:
        for row in range(self._table.rowCount()):
            if self._row_id_at(row) == row_id:
                return row
        return None

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
        if kind == "wizard":
            # The wizard runs on one row at a time, so it always opens for the
            # clicked row and never asks about the rest of the selection.
            widget.selection_snapshot_rows = ()
            widget.selection_snapshot_key = None
            self._emit_wizard_for_row(row_id)
        elif kind == "configure":
            target_row_ids = resolve_target_keys(
                widget,
                clicked_key=row_id,
                snapshot_keys=[k for k in snapshot_keys if k is not None],
                eligibility=lambda rid: True,
                parent=self,
                action="Configure parameters for",
            )
            if not target_row_ids:
                return
            self._configure_rows(target_row_ids)
        elif kind == "remove":
            target_row_ids = resolve_target_keys(
                widget,
                clicked_key=row_id,
                snapshot_keys=[k for k in snapshot_keys if k is not None],
                eligibility=lambda rid: True,
                parent=self,
                action="Remove",
                confirm_unprompted=lambda: self._confirm_removal((row_id,)),
            )
            if target_row_ids:
                self._remove_rows_by_id(target_row_ids)

    def _confirm_removal(self, row_ids: tuple[int, ...]) -> bool:
        submitted = sum(
            1
            for row_id in row_ids
            if workflow_row_is_locked(self._runtime_state_for_row(row_id).status)
        )
        return confirm_row_removal(self, len(row_ids), submitted_count=submitted)

    def _runtime_state_for_row(self, row_id: int) -> UepRunState:
        row = self._row_index_for_id(row_id)
        if row is None:
            return UepRunState(status="ready")
        item = self._table.item(row, self._LABEL_COLUMN)
        label = "" if item is None else item.text().strip()
        return self._runtime_states.get(label, UepRunState(status="ready"))


    def _configure_rows(self, row_ids: tuple[int, ...]) -> None:
        result = run_row_parameter_dialog(
            RowParameterRequest(
                specs=self._specs,
                workflow_stage="uep",
                subject="UEP Parameters",
                values=self._common_params_for_rows(row_ids),
                row_values={row_id: self._effective_params_for_row(row_id) for row_id in row_ids},
                row_advanced={row_id: self._row_added_advanced.get(row_id, set()) for row_id in row_ids},
                sanitize=lambda values: sanitize_parameter_mapping(values, self._specs_by_name),
                locked_values={
                    "task": "SINGLEPOINT",
                    "write_formatted_density": "TRUE (required for UEP source analysis)",
                },
                read_only=any(
                    workflow_row_is_locked(self._runtime_state_for_row(row_id).status)
                    for row_id in row_ids
                ),
                intro_text=(
                    "UEP source jobs are CASTEP single-point calculations. "
                    "Formatted density output stays enabled because the UEP analysis requires `.den_fmt`."
                ),
                species_options=self._parameter_species_options,
            ),
            parent=self,
        )
        if result is None:
            return
        for row_id, values in result.row_values.items():
            values["write_formatted_density"] = True
            values["task"] = "SINGLEPOINT"
            self._row_parameter_values[row_id] = values
            self._row_added_advanced[row_id] = result.row_advanced[row_id]
        self._emit_state()

    def _on_batch_configure_clicked(self) -> None:
        self._configure_rows(self._selected_row_ids())

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
        # Dialog-only pre-fill of the inactive half of the kpoint pair so
        # the radio toggle shows a sensible value when the user switches.
        # The param dialog drops the inactive choice in collect_values(),
        # so this hint never reaches validation or the rendered .cell.
        has_grid = "kpoint_mp_grid" in common
        has_spacing = "kpoint_mp_spacing" in common
        if has_grid and not has_spacing and row_ids:
            suggested_spacing = self._supercells.suggested_spacing()
            if suggested_spacing is not None:
                common["kpoint_mp_spacing"] = suggested_spacing
        elif has_spacing and not has_grid and row_ids:
            suggested_grid = self._supercells.suggested_grid(row_ids[0])
            if suggested_grid is not None:
                common["kpoint_mp_grid"] = suggested_grid
        return common

    def _effective_params_for_row(self, row_id: int) -> dict[str, object]:
        # The row's effective values reflect what would land in the rendered
        # .cell: only ONE of kpoint_mp_grid / kpoint_mp_spacing, never both.
        # The configure-parameters dialog renders both rows with a radio
        # toggle and drops the inactive choice in collect_values(), so the
        # row's stored parameter_values is also single-choice. If the user
        # explicitly chose spacing, drop the grid inherited from
        # _default_params to avoid a CASTEP mutual-exclusion conflict.
        effective = dict(self._default_params)
        row_values = self._row_parameter_values.get(row_id, {})
        effective.update(row_values)
        if "kpoint_mp_spacing" in row_values:
            # User explicitly chose spacing: drop any grid inherited from defaults.
            effective.pop("kpoint_mp_grid", None)
        elif "kpoint_mp_grid" in row_values:
            # User explicitly chose grid: drop any spacing inherited from defaults.
            effective.pop("kpoint_mp_spacing", None)
        elif (
            self._supercells.matrices.get(row_id) is not None
            and (suggested := self._supercells.suggested_grid(row_id)) is not None
        ):
            effective["kpoint_mp_grid"] = suggested
        effective["task"] = "SINGLEPOINT"
        effective["write_formatted_density"] = True
        return effective

    def _remove_row(self, row_id: int) -> None:
        self._remove_rows_by_id((row_id,))

    def _remove_rows_by_id(self, row_ids: tuple[int, ...]) -> None:
        removed_labels: list[str] = []
        for row_id in row_ids:
            row = self._row_index_for_id(row_id)
            if row is None:
                continue
            label_item = self._table.item(row, self._LABEL_COLUMN)
            label = label_item.text().strip() if label_item is not None else ""
            self._table.removeRow(row)
            self._supercells.matrices.pop(row_id, None)
            self._row_parameter_values.pop(row_id, None)
            self._row_added_advanced.pop(row_id, None)
            if label:
                removed_labels.append(label)
        if removed_labels:
            self.remove_requested.emit(removed_labels)
            self._refresh_batch_button_state()
            self._emit_state()

    def _delete_selected_rows(self) -> None:
        if self._table.state() == QAbstractItemView.State.EditingState:
            return
        row_ids = self._selected_or_current_row_ids()
        if row_ids and self._confirm_removal(row_ids):
            self._remove_rows_by_id(row_ids)

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

    def _labels_at_rows(self, rows: Sequence[int]) -> list[str]:
        return [label for row in rows if (label := self._label_at_row(row))]

    def _label_at_row(self, row: int) -> str | None:
        item = self._table.item(row, self._LABEL_COLUMN)
        if item is None:
            return None
        return item.text().strip() or None

    def _row_status(self, row: int) -> str:
        return workflow_row_status_text(self._table, row, self._STATUS_COLUMN)

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
                    supercell_matrix=self._supercells.matrices.get(row_id),
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
                self._supercells.populate_combo(combo, row_id)
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
        locked = workflow_row_is_locked(status_text)
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
            style_workflow_configure_button(
                configure_button,
                workflow_label="CASTEP",
                configured=customized,
                locked=locked,
                noun="UEP source job",
                used=True,
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
            matrix = self._supercells.matrices.get(row_id)
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
            self._supercells.matrices.clear()
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
                self._supercells.matrices[row_id] = coerce_supercell_matrix(entry.get("supercell_matrix"))
                params = entry.get("parameter_values", {})
                if isinstance(params, dict):
                    self._row_parameter_values[row_id] = sanitize_parameter_mapping(
                        {str(name): raw for name, raw in params.items()},
                        self._specs_by_name,
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

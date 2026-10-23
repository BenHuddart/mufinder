"""Pristine-relaxation configuration panel.

Rows whose job has left the unlocked runtime statuses keep their inline
parameter lock (``workflow_row_is_locked``): a submitted row's inputs must not
drift away from the directory that was rendered from them. Editing a terminal
row's parameters goes through the "Continue relaxation…" path instead, which
mutates the active definition's row in place via
:meth:`~mufinder.app.workflow_controllers.pristine_relaxation.PristineRelaxationController.update_row_parameters`
and mirrors the edit back into this form, preserving the row→job binding.

The row scaffold is :class:`~mufinder.app.workflow_row_widget.RelaxationRowTableWidget`;
this module adds the per-row cell constraint and the lattice-parameter columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QPushButton, QTableWidget, QWidget

from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
from mufinder.app.selection_snapshot import SelectionSnapshotComboBox, resolve_target_keys
from mufinder.app.workflow_row_widget import RelaxationRowTableWidget, RowEntry
from mufinder.app.workflow_table_utils import (
    multi_row_outcome_text,
    set_workflow_table_item,
    show_status_message,
)


class PristineRelaxationWidget(RelaxationRowTableWidget):
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
    _COLUMN_HEADERS = (
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

    _WORKFLOW_STAGE = "pristine_relaxation"
    _DIALOG_SUBJECT = "Relaxation Parameters"
    _LOCKED_DIALOG_VALUES = {
        "task": "GEOMETRYOPTIMIZATION",
        "fix_all_cell": "Variable (set by row control)",
        "write_cell_structure": "TRUE (required for relaxed-cell outputs)",
    }
    _ROW_LABEL_TEMPLATE = "relax_{number}"
    _NAME_PREFIX_DEFAULT = "pristine_relax"
    _NO_ROWS_MESSAGE = "Add one or more relaxation rows first."

    generate_comparison_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._generate_comparison_button = QPushButton("Generate Comparison Table...")
        self._build_layout(
            title="Pristine Relaxation",
            subtitle="Create labelled geometry-optimization jobs with per-job supercells and cell constraints.",
            table_group_title="Relaxation Jobs",
            run_extras=(self._generate_comparison_button,),
        )
        self._connect_signals()
        self._generate_comparison_button.clicked.connect(self.generate_comparison_requested.emit)

    def create_row_cells(self, table: QTableWidget, row: int, row_id: int) -> None:
        fix_combo = SelectionSnapshotComboBox(table)
        fix_combo.selection_snapshot_key = row_id
        fix_combo.addItem("Yes (fixed)", True)
        fix_combo.addItem("No (relax)", False)
        fix_combo.setCurrentIndex(fix_combo.findData(False))
        fix_combo.currentIndexChanged.connect(
            lambda _index, combo=fix_combo: self._on_fix_cell_combo_changed(row_id, combo)
        )
        table.setCellWidget(row, self._FIX_CELL_COLUMN, fix_combo)
        for column in (self._LATTICE_A_COLUMN, self._LATTICE_B_COLUMN, self._LATTICE_C_COLUMN):
            set_workflow_table_item(table, row, column, "")

    def refresh_row_cells(
        self,
        table: QTableWidget,
        row: int,
        row_id: int,
        runtime: Any,
        *,
        status: str,
        locked: bool,
    ) -> None:
        del row_id, status
        fix_combo = table.cellWidget(row, self._FIX_CELL_COLUMN)
        if isinstance(fix_combo, QComboBox):
            fix_combo.setEnabled(not locked)
        lattice = (
            (None, None, None)
            if runtime is None
            else (runtime.lattice_a, runtime.lattice_b, runtime.lattice_c)
        )
        for column, value in zip(
            (self._LATTICE_A_COLUMN, self._LATTICE_B_COLUMN, self._LATTICE_C_COLUMN),
            lattice,
            strict=True,
        ):
            self._set_float_item(row, column, value, table=table)

    def _fix_all_cell_at_row(self, row: int, *, table: QTableWidget | None = None) -> bool:
        combo = (table or self._table).cellWidget(row, self._FIX_CELL_COLUMN)
        if isinstance(combo, QComboBox):
            return bool(combo.currentData())
        return False

    def _on_fix_cell_combo_changed(self, row_id: int, combo: SelectionSnapshotComboBox) -> None:
        snapshot_keys = [
            key
            for row in combo.selection_snapshot_rows
            if (key := self._row_id_at(row)) is not None
        ]
        target_row_ids = resolve_target_keys(
            combo,
            clicked_key=row_id,
            snapshot_keys=snapshot_keys,
            eligibility=self._row_is_editable,
            parent=self,
            action="Set this cell constraint on",
        )
        if not target_row_ids:
            combo.restore_pre_interaction_index()
            return
        if len(target_row_ids) > 1:
            show_status_message(
                self,
                multi_row_outcome_text(
                    "Cell constraint set on", len(target_row_ids), len(snapshot_keys)
                ),
            )
        for target_row_id in target_row_ids:
            location = self._row_location_for_id(target_row_id)
            if target_row_id == row_id or location is None:
                continue
            table, target_row = location
            target_combo = table.cellWidget(target_row, self._FIX_CELL_COLUMN)
            if isinstance(target_combo, QComboBox):
                previous = target_combo.blockSignals(True)
                target_combo.setCurrentIndex(combo.currentIndex())
                target_combo.blockSignals(previous)
        self._emit_state()

    def config_for_rows(self, rows: Sequence[RowEntry]) -> PristineRelaxationConfig:
        return PristineRelaxationConfig(
            name_prefix=self._name_prefix_text(),
            rows=tuple(
                PristineRelaxationRow(
                    label=entry.label,
                    fix_all_cell=self._fix_all_cell_at_row(entry.row, table=entry.table),
                    supercell_matrix=self._supercells.matrices.get(entry.row_id),
                    parameter_values=dict(self._row_parameter_values.get(entry.row_id, {})),
                    added_advanced=tuple(sorted(self._row_added_advanced.get(entry.row_id, set()))),
                )
                for entry in rows
            ),
        )

    def collect_state(self) -> dict[str, Any]:
        return {
            "name_prefix": self._name_prefix_text(),
            "rows": [
                {
                    "label": entry.label,
                    "fix_all_cell": self._fix_all_cell_at_row(entry.row, table=entry.table),
                    "supercell_matrix": self._supercell_matrix_state(entry.row_id),
                    "params": dict(self._row_parameter_values.get(entry.row_id, {})),
                    "added_advanced": sorted(self._row_added_advanced.get(entry.row_id, set())),
                }
                for entry in self._row_entries()
            ],
        }

    def set_form_state(self, state: dict[str, Any]) -> None:
        with self._restoring_form():
            self._set_name_prefix(str(state.get("name_prefix", self._NAME_PREFIX_DEFAULT)))
            raw_rows = state.get("rows", [])
            for entry in raw_rows if isinstance(raw_rows, list) else []:
                if not isinstance(entry, dict):
                    continue
                row_id = self._allocate_row_id()
                self._restore_row_fields(row_id, entry, params_key="params")
                row = self._insert_row(
                    row_id, str(entry.get("label", self._new_row_label())).strip()
                )
                fix_combo = self._table.cellWidget(row, self._FIX_CELL_COLUMN)
                if isinstance(fix_combo, QComboBox):
                    index = fix_combo.findData(bool(entry.get("fix_all_cell", False)))
                    fix_combo.setCurrentIndex(index if index >= 0 else 0)

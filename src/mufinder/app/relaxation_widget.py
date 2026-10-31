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
from PySide6.QtWidgets import QCheckBox, QPushButton, QTableWidget, QWidget

from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
from mufinder.app.selection_snapshot import SelectionSnapshotCheckBox, resolve_target_keys
from mufinder.app.workflow_row_widget import RelaxationRowTableWidget, RowEntry
from mufinder.app.workflow_table_utils import (
    multi_row_outcome_text,
    name_row_control,
    set_workflow_table_item,
    show_status_message,
)
from mufinder.science.structure_metrics import format_with_change


#: Significant figures of a lattice cell, which leaves the three of them room
#: for the row's other columns at the default window size.
_LATTICE_DIGITS = 5


class PristineRelaxationWidget(RelaxationRowTableWidget):
    """UI for configuring pristine-relaxation jobs with per-row supercells."""

    _LABEL_COLUMN = 0
    _LABEL_COLUMN_WIDTH = 106
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
        "Fix cell",
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
    _EMPTY_ROWS_TEXT = "Add a job to relax the host unit cell or a saved supercell."

    generate_comparison_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._generate_comparison_button = QPushButton("Generate Comparison Table…")
        self._build_layout(
            title="Pristine Relaxation",
            subtitle=(
                "Relax the host without a muon, to check the settings against experiment "
                "and give the later stages a reference structure."
            ),
            table_group_title="Relaxation Jobs",
            control_extras=(self._generate_comparison_button,),
        )
        self._connect_signals()
        self._generate_comparison_button.clicked.connect(self.generate_comparison_requested.emit)

    def create_row_cells(self, table: QTableWidget, row: int, row_id: int) -> None:
        fix_check = SelectionSnapshotCheckBox(table)
        fix_check.setObjectName("rowCheckBox")
        fix_check.selection_snapshot_key = row_id
        fix_check.toggled.connect(
            lambda _checked, check=fix_check: self._on_fix_cell_toggled(row_id, check)
        )
        table.setCellWidget(row, self._FIX_CELL_COLUMN, fix_check)
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
        fix_check = table.cellWidget(row, self._FIX_CELL_COLUMN)
        if isinstance(fix_check, QCheckBox):
            fix_check.setEnabled(not locked)
            name_row_control(fix_check, "Fix cell for", self._row_display_label(row, table=table))
        lattice = (
            (None, None, None)
            if runtime is None
            else (runtime.lattice_a, runtime.lattice_b, runtime.lattice_c)
        )
        references = (
            runtime.host_lattice_lengths
            if runtime is not None and runtime.host_lattice_lengths is not None
            else (None, None, None)
        )
        for column, value, reference in zip(
            (self._LATTICE_A_COLUMN, self._LATTICE_B_COLUMN, self._LATTICE_C_COLUMN),
            lattice,
            references,
            strict=True,
        ):
            text = (
                ""
                if value is None
                else format_with_change(value, reference, digits=_LATTICE_DIGITS)
            )
            set_workflow_table_item(table, row, column, text)

    def _fix_all_cell_at_row(self, row: int, *, table: QTableWidget | None = None) -> bool:
        check = (table or self._table).cellWidget(row, self._FIX_CELL_COLUMN)
        return isinstance(check, QCheckBox) and check.isChecked()

    def _on_fix_cell_toggled(self, row_id: int, check: SelectionSnapshotCheckBox) -> None:
        snapshot_keys = [
            key
            for row in check.selection_snapshot_rows
            if (key := self._row_id_at(row)) is not None
        ]
        target_row_ids = resolve_target_keys(
            check,
            clicked_key=row_id,
            snapshot_keys=snapshot_keys,
            eligibility=self._row_is_editable,
            parent=self,
            action="Set this cell constraint on",
        )
        if not target_row_ids:
            check.restore_pre_interaction_state()
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
            target_check = table.cellWidget(target_row, self._FIX_CELL_COLUMN)
            if isinstance(target_check, QCheckBox):
                previous = target_check.blockSignals(True)
                target_check.setChecked(check.isChecked())
                target_check.blockSignals(previous)
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
                fix_check = self._table.cellWidget(row, self._FIX_CELL_COLUMN)
                if isinstance(fix_check, QCheckBox):
                    fix_check.setChecked(bool(entry.get("fix_all_cell", False)))

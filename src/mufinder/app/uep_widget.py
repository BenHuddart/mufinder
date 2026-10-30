"""UEP source-job configuration panel and completed-source launch controls.

The row scaffold is :class:`~mufinder.app.workflow_row_widget.RowTableWidget`;
this module adds the source-structure selector, the UEP parameter surface, the
result columns and the per-row analysis wizard button.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.selection_snapshot import SelectionSnapshotPushButton
from mufinder.app.theme import style_hint_text
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.app.workflow_parameters import (
    STRUCTURE_CONTROLLED_CELL_KEYWORDS,
    clone_spec_for_workflow,
    reorder_workflow_parameter_specs,
    seed_xc_functional_default,
)
from mufinder.app.workflow_row_widget import RowEntry, RowTableWidget
from mufinder.app.workflow_table_utils import set_workflow_table_item
from mufinder.castep_io.gui_metadata import get_gui_keyword_specs_for_files
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import ValueWithUnit


def _normalize_uep_name_prefix(value: object) -> str:
    prefix = str(value).strip() or "uep"
    return "uep" if prefix == "uep_source" else prefix


class UEPWidget(RowTableWidget):
    """UI for configuring UEP source CASTEP jobs and launching analysis for completed rows.

    A UEP row stores the parameter dialog's values with the source-job task and
    the formatted-density flag pinned, rather than the diff from its defaults.
    """

    _LABEL_COLUMN = 0
    _SUPERCELL_COLUMN = 1
    _STATUS_COLUMN = 2
    _TOTAL_ENERGY_COLUMN = 3
    _POTENTIAL_GRID_COLUMN = 4
    _CONFIGURE_COLUMN = 5
    _WIZARD_COLUMN = 6
    _REMOVE_COLUMN = 7
    _COLUMN_HEADERS = (
        "Label",
        "Supercell",
        "Status",
        "Total Energy (eV)",
        "3D Potential",
        "Parameters",
        "Analysis",
        "",
    )

    _WORKFLOW_STAGE = "uep"
    _PINNED_PARAMS = {"write_formatted_density": True}
    _DIALOG_SUBJECT = "UEP Parameters"
    _DIALOG_INTRO = (
        "UEP source jobs are CASTEP single-point calculations. "
        "Formatted density output stays enabled because the UEP analysis requires `.den_fmt`."
    )
    _LOCKED_DIALOG_VALUES = {
        "task": "SINGLEPOINT",
        "write_formatted_density": "TRUE (required for UEP source analysis)",
    }
    _EMPTY_STATUS = "ready"
    _CONFIGURE_NOUN = "UEP source job"
    _ROW_LABEL_TEMPLATE = "source_{number:02d}"
    _NAME_PREFIX_DEFAULT = "uep"
    _NO_ROWS_MESSAGE = "Add at least one UEP source job."

    source_structure_selection_requested = Signal(object)
    wizard_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_structure_options: list[tuple[str, str]] = []
        self._updating_source_structure = False
        self._source_structure_summary = QLabel("Using the host structure.")
        self._source_structure_summary.setWordWrap(True)
        style_hint_text(self._source_structure_summary)
        self._source_structure_combo = QComboBox()

        source_group = QGroupBox("Source Structure")
        source_layout = QVBoxLayout(source_group)
        source_layout.setContentsMargins(10, 8, 10, 8)
        selector_row = QHBoxLayout()
        source_structure_label = QLabel("Structure")
        source_structure_label.setBuddy(self._source_structure_combo)
        selector_row.addWidget(source_structure_label)
        selector_row.addWidget(self._source_structure_combo, 1)
        source_layout.addLayout(selector_row)
        source_layout.addWidget(self._source_structure_summary)

        self._build_layout(
            title="UEP",
            subtitle=(
                "Create CASTEP single-point source jobs that write formatted densities for "
                "unperturbed electrostatic potential analysis."
            ),
            table_group_title="UEP Source Jobs",
            preamble=(source_group,),
            separate_run_group=False,
        )
        self._connect_signals()
        self._source_structure_combo.currentIndexChanged.connect(self._on_source_structure_changed)

    def _build_parameter_specs(self) -> list[KeywordSpec]:
        adjusted: list[KeywordSpec] = []
        for spec in get_gui_keyword_specs_for_files(
            files=("param", "cell"),
            workflow_stage="uep",
            include_advanced=True,
        ):
            if spec.file == "cell" and spec.name in STRUCTURE_CONTROLLED_CELL_KEYWORDS:
                continue
            # write_formatted_density is locked on but shown; kpoint_mp_grid /
            # kpoint_mp_spacing render as a mutually exclusive radio pair, so
            # both sit in the basic view, as in the relaxation workflows.
            if spec.name in ("write_formatted_density", "kpoint_mp_grid", "kpoint_mp_spacing"):
                adjusted.append(
                    clone_spec_for_workflow(spec, workflow_stage="uep", basic=True, advanced=True)
                )
                continue
            adjusted.append(spec)
        return reorder_workflow_parameter_specs(adjusted)

    def _initial_default_params(self) -> dict[str, object]:
        return {
            "task": "SINGLEPOINT",
            "symmetry_generate": True,
            "write_formatted_density": True,
        }

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
            seed_xc_functional_default(self._default_params, xc_functional, schema_default=None)
            self._supercells.set_kpoint_reference(kpoint_grid, kpoint_reference_lattice)
            if cutoff_energy is not None:
                self._default_params["cut_off_energy"] = ValueWithUnit(float(cutoff_energy), "eV")
            else:
                self._default_params.pop("cut_off_energy", None)
            if kpoint_grid is not None:
                self._default_params["kpoint_mp_grid"] = tuple(int(value) for value in kpoint_grid)
            else:
                self._default_params.pop("kpoint_mp_grid", None)
            self._refresh_all_rows()

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

    def _effective_params_for_row(self, row_id: int) -> dict[str, object]:
        # The row's effective values reflect what would land in the rendered
        # .cell: only ONE of kpoint_mp_grid / kpoint_mp_spacing, never both.
        # The defaults carry the project grid; a row that chose spacing drops
        # it, and a supercell row without its own choice gets the grid rescaled
        # to its supercell.
        effective = dict(self._default_params)
        row_values = self._row_parameter_values.get(row_id, {})
        effective.update(row_values)
        if "kpoint_mp_spacing" in row_values:
            effective.pop("kpoint_mp_grid", None)
        elif "kpoint_mp_grid" in row_values:
            effective.pop("kpoint_mp_spacing", None)
        elif (
            self._supercells.matrices.get(row_id) is not None
            and (suggested := self._supercells.suggested_grid(row_id)) is not None
        ):
            effective["kpoint_mp_grid"] = suggested
        effective.update(self._pinned_params())
        return effective

    def create_row_cells(self, table: QTableWidget, row: int, row_id: int) -> None:
        set_workflow_table_item(table, row, self._TOTAL_ENERGY_COLUMN, "")
        set_workflow_table_item(table, row, self._POTENTIAL_GRID_COLUMN, "")
        wizard_button = SelectionSnapshotPushButton("Open Wizard", table)
        wizard_button.selection_snapshot_key = row_id
        # The wizard runs on one row at a time, so it is never a multi-row target.
        wizard_button.propagates_to_selection = False
        wizard_button.clicked.connect(
            lambda _checked=False, btn=wizard_button: self._open_wizard(btn, row_id)
        )
        table.setCellWidget(row, self._WIZARD_COLUMN, wizard_button)

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
        del row_id, locked
        self._set_float_item(
            row,
            self._TOTAL_ENERGY_COLUMN,
            None if runtime is None else runtime.total_energy,
            formatter=".6f",
            table=table,
        )
        if runtime is not None and runtime.potential_grid_available is True:
            potential_text = "Saved"
        elif status == "completed":
            potential_text = "Not calculated"
        else:
            potential_text = ""
        set_workflow_table_item(table, row, self._POTENTIAL_GRID_COLUMN, potential_text)

        wizard_button = table.cellWidget(row, self._WIZARD_COLUMN)
        if isinstance(wizard_button, QPushButton):
            wizard_button.setEnabled(
                status == "completed" and runtime is not None and bool(runtime.density_available)
            )
            wizard_button.setToolTip(
                ""
                if wizard_button.isEnabled()
                else "Complete the source job and ensure `.den_fmt`/`.castep` outputs are available first."
            )

    def _open_wizard(self, button: SelectionSnapshotPushButton, row_id: int) -> None:
        # The wizard runs on one row at a time, so it always opens for the
        # clicked row and never asks about the rest of the selection.
        button.selection_snapshot_rows = ()
        button.selection_snapshot_key = None
        location = self._row_location_for_id(row_id)
        if location is not None:
            table, row = location
            self.wizard_requested.emit(self._label_at_row(row, table=table))

    def config_for_rows(self, rows: Sequence[RowEntry]) -> UepConfig:
        return UepConfig(
            name_prefix=_normalize_uep_name_prefix(self._name_prefix.text()),
            rows=tuple(
                UepSourceRow(
                    label=entry.label,
                    supercell_matrix=self._supercells.matrices.get(entry.row_id),
                    parameter_values=self._effective_params_for_row(entry.row_id),
                    added_advanced=tuple(sorted(self._row_added_advanced.get(entry.row_id, set()))),
                )
                for entry in rows
            ),
        )

    def collect_state(self) -> dict[str, Any]:
        return {
            "name_prefix": _normalize_uep_name_prefix(self._name_prefix.text()),
            "rows": [
                {
                    "label": entry.label,
                    "supercell_matrix": self._supercell_matrix_state(entry.row_id),
                    "parameter_values": dict(self._row_parameter_values.get(entry.row_id, {})),
                    "added_advanced": sorted(self._row_added_advanced.get(entry.row_id, set())),
                }
                for entry in self._row_entries()
            ],
            "selected_source_structure_id": self.selected_source_structure_id(),
        }

    def set_form_state(self, state: dict[str, Any]) -> None:
        with self._restoring_form():
            self._set_name_prefix(_normalize_uep_name_prefix(state.get("name_prefix", "uep")))
            rows_payload = state.get("rows", [])
            for entry in rows_payload if isinstance(rows_payload, list) else []:
                if not isinstance(entry, dict):
                    continue
                row_id = self._allocate_row_id()
                default_label = self._new_row_label(row_id)
                label = str(entry.get("label", default_label)).strip() or default_label
                self._restore_row_fields(row_id, entry, params_key="parameter_values")
                self._insert_row(row_id, label)

            selected_source_structure_id = state.get("selected_source_structure_id")
            if isinstance(selected_source_structure_id, str) and selected_source_structure_id:
                index = self._source_structure_combo.findData(selected_source_structure_id)
                if index >= 0:
                    self._source_structure_combo.setCurrentIndex(index)

    def _on_source_structure_changed(self) -> None:
        if self._updating_source_structure:
            return
        selected = self.selected_source_structure_id()
        if selected is not None:
            self.source_structure_selection_requested.emit(selected)
        self._emit_state()

"""Muon structural-relaxation configuration panel.

Rows whose job has left the unlocked runtime statuses keep their inline
parameter lock: a submitted row's inputs must not drift away from the directory
that was rendered from them. Editing a terminal row's parameters goes through
the "Continue relaxation…" path instead, which mutates the active definition's
row in place via
:meth:`~mufinder.app.workflow_controllers.muon_relaxation.MuonRelaxationController.update_row_parameters`
(keyed by the charge-qualified row key) and mirrors the edit back into this
form, preserving the row→job binding.

The row scaffold is :class:`~mufinder.app.workflow_row_widget.RelaxationRowTableWidget`;
this module adds one table per charge state, the staged-site position columns
and the read-only rows imported from calculations run outside MuFinder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QWidget,
)

from mufinder.app.active_host_badge import ActiveHostBadge
from mufinder.app.relaxation_adapter import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    muon_charge_for_state,
    muon_charge_state_label,
)
from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow
from mufinder.app.theme import current_palette, use_tabbed_table_corners
from mufinder.app.workflow_row_widget import RelaxationRowTableWidget, RowEntry
from mufinder.app.workflow_table_utils import set_workflow_table_item


_IMPORTED_ROW_TOOLTIP = (
    "Imported from a calculation run outside MuFinder. Its parameters and results are read-only."
)
_CHARGE_STATE_TABS = (
    (POSITIVE_MUON_CHARGE_STATE, "Positive muon"),
    (MUONIUM_CHARGE_STATE, "Muonium"),
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
            column for start, end, _label in groups for column in range(start, end + 1)
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

        text = self.model().headerData(
            logical_index, self.orientation(), Qt.ItemDataRole.DisplayRole
        )
        label_rect = QRect(rect)
        label_rect.setTop(rect.top() + rect.height() // 2)
        painter.drawText(
            label_rect, Qt.AlignmentFlag.AlignCenter, "" if text is None else str(text)
        )
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
            painter.drawLine(
                group_rect.left() + 6, underline_y, group_rect.right() - 6, underline_y
            )
            painter.setPen(QColor(palette.text_primary))


class MuonRelaxationWidget(RelaxationRowTableWidget):
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
    _COLUMN_HEADERS = (
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

    _WORKFLOW_STAGE = "muon_relaxation"
    _EXCLUDED_PARAMETER_NAMES = frozenset(
        {"task", "fix_all_cell", "write_cell_structure", "charge"}
    )
    # Starting values that differ from the schema default in this workflow.
    _PARAMETER_DEFAULT_OVERRIDES = {
        "geom_max_iter": 300,
        "continuation": "DEFAULT",
        "symmetry_generate": None,
    }
    _DIALOG_SUBJECT = "Muon Relaxation Parameters"
    _LOCKED_DIALOG_VALUES = {
        "task": "GEOMETRYOPTIMIZATION",
        "fix_all_cell": "TRUE (required for muon relaxation)",
        "write_cell_structure": "TRUE (required for relaxed-cell outputs)",
    }
    _ADD_ROW_TEXT = None
    _NO_ROWS_MESSAGE = "Create one or more jobs from generated muon sites first."
    _READ_ONLY_ROW_TOOLTIP = _IMPORTED_ROW_TOOLTIP

    analysis_workspace_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._row_site_ids: dict[int, UUID] = {}
        self._row_positions: dict[int, tuple[float, float, float]] = {}
        self._row_charge_states: dict[int, str] = {}
        # Rows ingested from a calculation performed outside MuFinder. Membership
        # is the read-only marker; the batch id groups one import's rows.
        self._row_imported: set[int] = set()
        self._row_import_batch_ids: dict[int, UUID] = {}

        self._active_host_badge = ActiveHostBadge()
        self._create_analysis_button = QPushButton("Create Analysis Workspace from Selection")
        self._build_layout(
            title="Structural Relaxations",
            subtitle="Configure and run geometry optimization calculations from staged initial muon positions.",
            table_group_title="Relaxation Jobs",
            preamble=(self._active_host_badge,),
            table_view=self._charge_tabs,
            control_leading=self._create_analysis_button,
        )
        self._connect_signals()
        # The count label and run/remove buttons follow the table on show.
        self._charge_tabs.currentChanged.connect(lambda *_: self._refresh_batch_button_state())
        self._create_analysis_button.clicked.connect(self._create_analysis_workspace_from_selection)

    def _create_tables(self) -> tuple[QTableWidget, ...]:
        self._charge_tabs = QTabWidget()
        self._charge_tabs.setObjectName("muonRelaxationChargeTabs")
        self._tables_by_charge_state: dict[str, QTableWidget] = {}
        groups = (
            (self._INITIAL_FX_COLUMN, self._INITIAL_FZ_COLUMN, "Initial"),
            (self._FINAL_FX_COLUMN, self._FINAL_FZ_COLUMN, "Final"),
        )
        for charge_state, tab_label in _CHARGE_STATE_TABS:
            # The coordinate columns sit under a grouped "Initial"/"Final" header.
            table = self._create_table(lambda parent: _GroupedCoordinateHeader(groups, parent))
            use_tabbed_table_corners(table)
            self._tables_by_charge_state[charge_state] = table
            self._charge_tabs.addTab(table, tab_label)
        self._table = self._tables_by_charge_state[POSITIVE_MUON_CHARGE_STATE]
        return tuple(
            self._tables_by_charge_state[charge_state]
            for charge_state, _label in _CHARGE_STATE_TABS
        )

    def _active_table(self) -> QTableWidget:
        current = self._charge_tabs.currentWidget()
        return current if isinstance(current, QTableWidget) else self._table

    def _table_for_charge_state(self, charge_state: str) -> QTableWidget:
        return self._tables_by_charge_state.get(charge_state, self._table)

    def set_active_host(self, structure: object | None) -> None:
        """Update the active-host badge."""
        from mufinder.domain.models import Structure as _Structure

        self._active_host_badge.set_active_host(
            structure if isinstance(structure, _Structure) else None
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
            fractional = row_data.get(
                "fractional_position", row_data.get("fractional", (0.0, 0.0, 0.0))
            )
            if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                continue
            for charge_state in normalized_charge_states:
                table = self._table_for_charge_state(charge_state)
                base_label = (
                    str(row_data.get("label", "")).strip() or f"site_{table.rowCount() + 1}"
                )
                row_id = self._new_site_row_id(
                    site_id=site_id,
                    fractional=(float(fractional[0]), float(fractional[1]), float(fractional[2])),
                    charge_state=charge_state,
                )
                self._insert_row(row_id, self._unique_label(base_label, table=table), table=table)
        self._emit_state()

    def _unique_label(self, base_label: str, *, table: QTableWidget) -> str:
        existing = {self._label_at_row(row, table=table) for row in range(table.rowCount())}
        if base_label not in existing:
            return base_label
        suffix = 2
        while f"{base_label} {suffix}" in existing:
            suffix += 1
        return f"{base_label} {suffix}"

    def _new_site_row_id(
        self,
        *,
        site_id: UUID,
        fractional: tuple[float, float, float],
        charge_state: str,
    ) -> int:
        """Allocate a row id for a staged site in ``charge_state``'s table."""

        row_id = self._allocate_row_id()
        self._row_site_ids[row_id] = site_id
        self._row_positions[row_id] = fractional
        self._row_charge_states[row_id] = (
            charge_state
            if charge_state in self._tables_by_charge_state
            else POSITIVE_MUON_CHARGE_STATE
        )
        return row_id

    def _forget_row(self, row_id: int) -> None:
        super()._forget_row(row_id)
        self._row_site_ids.pop(row_id, None)
        self._row_positions.pop(row_id, None)
        self._row_charge_states.pop(row_id, None)
        self._row_imported.discard(row_id)
        self._row_import_batch_ids.pop(row_id, None)

    def _charge_state_at_row(self, row: int, *, table: QTableWidget | None = None) -> str:
        row_id = self._row_id_at(row, table=table)
        if row_id is None:
            return POSITIVE_MUON_CHARGE_STATE
        return self._row_charge_states.get(row_id, POSITIVE_MUON_CHARGE_STATE)

    @staticmethod
    def _runtime_key(label: str, charge_state: str) -> str:
        prefix = "muonium" if charge_state == MUONIUM_CHARGE_STATE else "positivemuon"
        return f"{prefix}:{label}"

    def _submission_key_at_row(self, row: int, *, table: QTableWidget | None = None) -> str:
        return self._runtime_key(
            self._label_at_row(row, table=table),
            self._charge_state_at_row(row, table=table),
        )

    def _row_display_label(self, row: int, *, table: QTableWidget) -> str:
        charge_label = muon_charge_state_label(self._charge_state_at_row(row, table=table))
        return f"{self._label_at_row(row, table=table)} ({charge_label})"

    def _row_is_read_only(self, row_id: int) -> bool:
        # An imported row's parameters describe a run that already happened;
        # editing them would misdescribe it.
        return row_id in self._row_imported

    def _row_default_overrides(self, row_id: int) -> dict[str, object]:
        # relaxation_adapter runs muonium spin-polarised unless the row says
        # otherwise, so show that rather than the unticked default.
        if self._row_charge_states.get(row_id) == MUONIUM_CHARGE_STATE:
            return {"spin_polarized": True}
        return {}

    def _dialog_locked_values(self, row_ids: tuple[int, ...]) -> dict[str, str]:
        charges = {
            self._row_charge_states.get(row_id, POSITIVE_MUON_CHARGE_STATE)
            for row_id in row_ids
            if self._row_location_for_id(row_id) is not None
        }
        charge_text = "Variable (selected charge-state tabs)"
        if len(charges) == 1:
            charge_state = next(iter(charges))
            charge_text = f"{muon_charge_for_state(charge_state):+d} ({muon_charge_state_label(charge_state)})"
        return {**self._LOCKED_DIALOG_VALUES, "charge": charge_text}

    def _configurable_row_ids(self, row_ids: tuple[int, ...]) -> tuple[int, ...]:
        editable = tuple(row_id for row_id in row_ids if row_id not in self._row_imported)
        excluded = len(row_ids) - len(editable)
        if not editable:
            QMessageBox.information(
                self,
                "Imported rows are read-only",
                "The selected row(s) were imported from calculations run outside "
                "MuFinder, so their parameters cannot be edited. Use Configure on a "
                "single row to view them.",
            )
            return ()
        if excluded:
            QMessageBox.information(
                self,
                "Imported rows excluded",
                f"{excluded} imported row(s) were left out of this edit because their "
                "parameters describe calculations that have already run.",
            )
        return editable

    def _removable_row_ids(self, row_ids: tuple[int, ...]) -> tuple[int, ...]:
        # Removing an imported row must go through the dedicated cascade, which
        # first checks whether anything downstream still references it.
        protected = tuple(row_id for row_id in row_ids if row_id in self._row_imported)
        if protected:
            QMessageBox.information(
                self,
                "Imported rows are read-only",
                f"{len(protected)} imported row(s) were not removed. Imported "
                "relaxations own the only copy of their results, so they are "
                "removed through the Remove Imported Relaxations action instead.",
            )
        return tuple(row_id for row_id in row_ids if row_id not in protected)

    def create_row_cells(self, table: QTableWidget, row: int, row_id: int) -> None:
        for column, value in zip(
            (self._INITIAL_FX_COLUMN, self._INITIAL_FY_COLUMN, self._INITIAL_FZ_COLUMN),
            self._row_positions[row_id],
            strict=True,
        ):
            set_workflow_table_item(table, row, column, f"{float(value):.3f}")
        for column in (
            self._FINAL_FX_COLUMN,
            self._FINAL_FY_COLUMN,
            self._FINAL_FZ_COLUMN,
            self._TOTAL_ENERGY_COLUMN,
        ):
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
        del row_id, status, locked
        final = None if runtime is None else runtime.final_fractional_position
        for column, value in zip(
            (self._FINAL_FX_COLUMN, self._FINAL_FY_COLUMN, self._FINAL_FZ_COLUMN),
            final or (None, None, None),
            strict=True,
        ):
            self._set_float_item(row, column, value, formatter=".3f", table=table)
        self._set_float_item(
            row,
            self._TOTAL_ENERGY_COLUMN,
            None if runtime is None else runtime.total_energy,
            table=table,
        )

    def _create_analysis_workspace_from_selection(self) -> None:
        keys = self._selected_keys()
        if not keys:
            QMessageBox.information(
                self,
                "No relaxations selected",
                "Select one or more completed structural relaxation rows first.",
            )
            return
        self.analysis_workspace_requested.emit(keys)

    def config_for_rows(self, rows: Sequence[RowEntry]) -> MuonRelaxationConfig:
        return MuonRelaxationConfig(
            rows=tuple(
                MuonRelaxationRow(
                    site_id=self._row_site_ids[entry.row_id],
                    label=entry.label,
                    initial_fractional_position=self._row_positions[entry.row_id],
                    charge_state=self._charge_state_at_row(entry.row, table=entry.table),
                    supercell_matrix=self._supercells.matrices.get(entry.row_id),
                    parameter_values=dict(self._row_parameter_values.get(entry.row_id, {})),
                    added_advanced=tuple(sorted(self._row_added_advanced.get(entry.row_id, set()))),
                    imported=entry.row_id in self._row_imported,
                    import_batch_id=self._row_import_batch_ids.get(entry.row_id),
                )
                for entry in rows
            )
        )

    def collect_state(self) -> dict[str, Any]:
        rows: list[dict[str, object]] = []
        for entry in self._row_entries():
            row_id = entry.row_id
            rows.append(
                {
                    "site_id": str(self._row_site_ids[row_id]),
                    "label": entry.label,
                    "fractional_position": list(self._row_positions[row_id]),
                    "charge_state": self._charge_state_at_row(entry.row, table=entry.table),
                    "supercell_matrix": self._supercell_matrix_state(row_id),
                    "parameter_values": dict(self._row_parameter_values.get(row_id, {})),
                    "added_advanced": sorted(self._row_added_advanced.get(row_id, set())),
                    "imported": row_id in self._row_imported,
                    "import_batch_id": (
                        str(self._row_import_batch_ids[row_id])
                        if row_id in self._row_import_batch_ids
                        else None
                    ),
                }
            )
        return {"rows": rows}

    def set_form_state(self, state: dict[str, Any]) -> None:
        with self._restoring_form():
            raw_rows = state.get("rows", []) if isinstance(state, dict) else []
            for entry in raw_rows if isinstance(raw_rows, list) else []:
                if not isinstance(entry, dict):
                    continue
                try:
                    site_id = UUID(str(entry.get("site_id")))
                except (TypeError, ValueError):
                    continue
                fractional = entry.get("fractional_position", (0.0, 0.0, 0.0))
                if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                    continue
                charge_state = str(entry.get("charge_state", POSITIVE_MUON_CHARGE_STATE))
                table = self._table_for_charge_state(charge_state)
                label = str(entry.get("label", "")).strip() or f"site_{table.rowCount() + 1}"
                row_id = self._new_site_row_id(
                    site_id=site_id,
                    fractional=(float(fractional[0]), float(fractional[1]), float(fractional[2])),
                    charge_state=charge_state,
                )
                # The imported marker is read *before* the parameter values,
                # because sanitising them consults it: an imported row keeps
                # names the current keyword catalog does not know.
                if bool(entry.get("imported", False)):
                    self._row_imported.add(row_id)
                    raw_batch_id = entry.get("import_batch_id")
                    if raw_batch_id is not None:
                        try:
                            self._row_import_batch_ids[row_id] = UUID(str(raw_batch_id))
                        except (TypeError, ValueError):
                            # A malformed batch id must not silently unlock the
                            # row: keep it imported, just ungrouped.
                            pass
                self._restore_row_fields(
                    row_id,
                    entry,
                    params_key="parameter_values",
                    keep_unknown=row_id in self._row_imported,
                )
                self._insert_row(row_id, label, table=table)

"""Structured block-parameter editing dialogs for CASTEP keywords."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import style_section_title, track_item_view_corners
from mufinder.castep_io.schema import BlockColumnSpec, BlockSchema, KeywordSpec
from mufinder.castep_io.types import BlockSection


@dataclass(slots=True)
class _ParsedBlock:
    rows: tuple[tuple[object, ...], ...]
    unit: str | None


def summarize_block_value(spec: KeywordSpec, value: BlockSection | None) -> str:
    """Return a compact human-readable summary for a block value."""

    if value is None:
        return "Not set"

    schema = spec.block_schema
    if schema is not None and schema.summary_mode == "matrix3":
        prefix = f"{value.unit} " if value.unit else ""
        return f"{prefix}3x3 matrix"
    if schema is not None and schema.summary_mode == "matrix2":
        prefix = f"{value.unit} " if value.unit else ""
        return f"{prefix}2-row matrix"

    row_count = len(value.rows)
    prefix = f"{value.unit} " if value.unit else ""
    noun = "row" if row_count == 1 else "rows"
    return f"{prefix}{row_count} {noun}"


class BlockParameterDialog(QDialog):
    """Schema-driven editor for block-style CASTEP keywords."""

    def __init__(
        self,
        spec: KeywordSpec,
        *,
        initial_value: BlockSection | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self._schema = spec.block_schema or BlockSchema()
        self._initial_value = initial_value

        self.setWindowTitle(spec.label or spec.name.upper())
        self.resize(760, 460)

        root = QVBoxLayout(self)

        title = QLabel(spec.label or spec.name.upper(), self)
        style_section_title(title)
        root.addWidget(title)

        if spec.description:
            summary = QLabel(spec.description, self)
            summary.setWordWrap(True)
            root.addWidget(summary)

        self._unit_edit = QLineEdit(self)
        unit_row = QWidget(self)
        unit_layout = QFormLayout(unit_row)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.addRow("Unit", self._unit_edit)
        root.addWidget(unit_row)
        unit_row.setVisible(self._schema.allow_unit)

        columns = self._schema.columns or (BlockColumnSpec(name="Entry"),)
        self._table = QTableWidget(self)
        track_item_view_corners(self._table)
        self._table.setColumnCount(len(columns))
        self._table.setHorizontalHeaderLabels([column.name for column in columns])
        root.addWidget(self._table, 1)

        controls = QHBoxLayout()
        self._add_row_button = QPushButton("Add Row", self)
        self._remove_row_button = QPushButton("Remove Selected", self)
        self._add_row_button.clicked.connect(self._add_row)
        self._remove_row_button.clicked.connect(self._remove_selected_rows)
        controls.addWidget(self._add_row_button)
        controls.addWidget(self._remove_row_button)
        controls.addStretch(1)
        root.addLayout(controls)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._populate_from_initial()
        self._update_row_controls()

    def block_value(self) -> BlockSection:
        parsed = self._parse()
        return BlockSection(
            name=self._spec.name,
            rows=tuple(tuple(row) for row in parsed.rows),
            unit=parsed.unit,
        )

    def _populate_from_initial(self) -> None:
        if self._schema.allow_unit:
            default_unit = self._schema.default_unit or self._spec.unit or ""
            self._unit_edit.setText(
                (self._initial_value.unit if self._initial_value else default_unit) or ""
            )

        if self._initial_value is not None:
            for row in self._initial_value.rows:
                self._append_row(tuple(str(item) for item in row))
        else:
            initial_rows = self._schema.fixed_row_count or max(self._schema.min_rows, 1)
            for _ in range(initial_rows):
                self._append_row()

        if self._schema.row_labels:
            for index, label in enumerate(self._schema.row_labels):
                if index < self._table.rowCount():
                    header_item = QTableWidgetItem(label)
                    self._table.setVerticalHeaderItem(index, header_item)

    def _append_row(self, values: tuple[str, ...] | None = None) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        columns = self._schema.columns or (BlockColumnSpec(name="Entry"),)
        for column_index in range(len(columns)):
            text = values[column_index] if values and column_index < len(values) else ""
            self._table.setItem(row, column_index, QTableWidgetItem(text))

    def _add_row(self) -> None:
        self._append_row()
        self._update_row_controls()

    def _remove_selected_rows(self) -> None:
        rows = sorted({item.row() for item in self._table.selectedItems()}, reverse=True)
        minimum = self._schema.fixed_row_count or self._schema.min_rows
        for row in rows:
            if self._table.rowCount() <= minimum:
                break
            self._table.removeRow(row)
        self._update_row_controls()

    def _update_row_controls(self) -> None:
        fixed = self._schema.fixed_row_count is not None
        self._add_row_button.setEnabled(not fixed)
        self._remove_row_button.setEnabled(
            not fixed and self._table.rowCount() > self._schema.min_rows
        )

    def _accept_if_valid(self) -> None:
        try:
            self._parse()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid block data", str(exc))
            return
        self.accept()

    def _parse(self) -> _ParsedBlock:
        columns = self._schema.columns or (BlockColumnSpec(name="Entry"),)
        row_count = self._table.rowCount()

        if self._schema.fixed_row_count is not None and row_count != self._schema.fixed_row_count:
            raise ValueError(
                f"{self._spec.name.upper()} requires exactly {self._schema.fixed_row_count} rows."
            )
        if row_count < self._schema.min_rows:
            raise ValueError(
                f"{self._spec.name.upper()} requires at least {self._schema.min_rows} rows."
            )
        if self._schema.row_group_size is not None and row_count % self._schema.row_group_size != 0:
            raise ValueError(
                f"{self._spec.name.upper()} row count must be a multiple of {self._schema.row_group_size}."
            )

        parsed_rows: list[tuple[object, ...]] = []
        for row_index in range(row_count):
            parsed_row: list[object] = []
            for column_index, column in enumerate(columns):
                item = self._table.item(row_index, column_index)
                text = item.text().strip() if item is not None else ""
                if column.required and not text:
                    raise ValueError(f"Row {row_index + 1} column '{column.name}' is required.")
                if not text:
                    parsed_row.append("")
                    continue
                parsed_row.append(self._parse_cell(column, text))
            parsed_rows.append(tuple(parsed_row))

        unit: str | None = None
        if self._schema.allow_unit:
            unit_text = self._unit_edit.text().strip()
            unit = unit_text or None

        return _ParsedBlock(rows=tuple(parsed_rows), unit=unit)

    def _parse_cell(self, column: BlockColumnSpec, text: str) -> object:
        if column.allowed_values and text not in column.allowed_values:
            allowed = ", ".join(column.allowed_values)
            raise ValueError(f"'{text}' is not valid for {column.name}. Allowed values: {allowed}.")
        if column.value_kind == "int":
            return int(text)
        if column.value_kind == "float":
            return float(text)
        if column.value_kind == "bool":
            upper = text.upper()
            if upper not in {"TRUE", "FALSE"}:
                raise ValueError(f"{column.name} must be TRUE or FALSE.")
            return upper == "TRUE"
        return text

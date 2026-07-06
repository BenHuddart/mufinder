"""Renderable comparison table view for pristine-relaxation structure comparisons."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from uuid import UUID

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.comparison_export import export_comparison_csv
from mufinder.app.math_label import MathLabel
from mufinder.app.theme import adjust_font_size, style_section_title
from mufinder.domain.models import ComparisonTable
from mufinder.science.structure_metrics import metric_definitions


_METRIC_REGISTRY = {definition.key: definition for definition in metric_definitions()}


def _definition_text(definition: dict, key: str) -> str:
    value = definition.get(key, "")
    return str(value).strip() if value is not None else ""


def _definition_symbols(definition: dict) -> list[tuple[str, str]]:
    raw_symbols = definition.get("symbols", [])
    if not isinstance(raw_symbols, list):
        return []

    symbols: list[tuple[str, str]] = []
    for entry in raw_symbols:
        if isinstance(entry, dict):
            symbol = str(entry.get("symbol", "")).strip()
            meaning = str(entry.get("meaning", "")).strip()
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            symbol = str(entry[0]).strip()
            meaning = str(entry[1]).strip()
        else:
            continue
        if symbol and meaning:
            symbols.append((symbol, meaning))
    return symbols


def _merge_metric_definition(definition: dict) -> dict:
    key = _definition_text(definition, "key")
    base = _METRIC_REGISTRY.get(key)
    if base is None:
        return dict(definition)

    merged = {
        "key": base.key,
        "label": base.label,
        "unit": base.unit,
        "group": base.group,
        "meaning": base.meaning,
        "calculation": base.calculation,
        "interpretation": base.interpretation,
        "caveats": base.caveats,
        "physics": base.physics,
        "method": base.method,
        "equation": base.equation,
        "symbols": [{"symbol": symbol, "meaning": meaning} for symbol, meaning in base.symbols],
    }
    merged.update(definition)
    return merged


def _metric_help_sections(definition: dict) -> list[tuple[str, str]]:
    definition = _merge_metric_definition(definition)
    return [
        ("Meaning", _definition_text(definition, "meaning")),
        ("Physics", _definition_text(definition, "physics")),
        ("Implementation Summary", _definition_text(definition, "calculation")),
        ("Method", _definition_text(definition, "method")),
        ("Interpretation", _definition_text(definition, "interpretation")),
        ("Caveats", _definition_text(definition, "caveats")),
    ]


def _metric_help_summary_lines(definition: dict) -> list[str]:
    definition = _merge_metric_definition(definition)
    lines: list[str] = []
    group = _definition_text(definition, "group")
    unit = _definition_text(definition, "unit")
    if group:
        lines.append(f"Group: {group}")
    if unit:
        lines.append(f"Unit: {unit}")
    return lines


def _build_metric_help_plain_text(definition: dict) -> str:
    definition = _merge_metric_definition(definition)
    lines = _metric_help_summary_lines(definition)
    equation = _definition_text(definition, "equation")
    symbols = _definition_symbols(definition)

    for title, body in _metric_help_sections(definition):
        if body:
            if lines:
                lines.append("")
            lines.append(title)
            lines.append(body)

    if equation:
        if lines:
            lines.append("")
        lines.append("Equation")
        lines.append(equation)

    if symbols:
        if lines:
            lines.append("")
        lines.append("Symbols")
        for symbol, meaning in symbols:
            lines.append(f"- {symbol}: {meaning}")

    return "\n".join(lines)


def _create_help_paragraph(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def _create_symbol_row(symbol: str, meaning: str, parent: QWidget) -> QWidget:
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    bullet = QLabel("•", row)
    bullet.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
    bullet.setFixedWidth(12)
    layout.addWidget(bullet, 0, Qt.AlignmentFlag.AlignTop)

    symbol_widget = MathLabel(symbol, display_mode=False, parent=row)
    symbol_widget.setObjectName("metricSymbolLabel")
    layout.addWidget(symbol_widget, 0, Qt.AlignmentFlag.AlignTop)

    meaning_label = _create_help_paragraph(meaning, row)
    layout.addWidget(meaning_label, 1)
    return row


class ComparisonMetricHelpDialog(QDialog):
    """Read-only dialog describing one comparison metric."""

    def __init__(self, definition: dict, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        definition = _merge_metric_definition(definition)
        title_text = _definition_text(definition, "label") or "Metric"
        self.setWindowTitle(title_text)
        self.resize(680, 560)

        layout = QVBoxLayout(self)

        title = QLabel(title_text)
        style_section_title(title)
        layout.addWidget(title)

        scroll = QScrollArea(self)
        scroll.setObjectName("metricHelpScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget(scroll)
        content.setObjectName("metricHelpContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        for line in _metric_help_summary_lines(definition):
            summary_label = _create_help_paragraph(line, content)
            summary_label.setObjectName("metricSummaryLabel")
            content_layout.addWidget(summary_label)

        for title_text, body_text in _metric_help_sections(definition):
            if not body_text:
                continue
            section_title = QLabel(title_text, content)
            style_section_title(section_title)
            content_layout.addWidget(section_title)
            content_layout.addWidget(_create_help_paragraph(body_text, content))

        equation = _definition_text(definition, "equation")
        if equation:
            equation_title = QLabel("Equation", content)
            style_section_title(equation_title)
            content_layout.addWidget(equation_title)

            equation_label = MathLabel(equation, display_mode=True, parent=content)
            equation_label.setObjectName("metricEquationLabel")
            content_layout.addWidget(equation_label, 0, Qt.AlignmentFlag.AlignLeft)

        symbols = _definition_symbols(definition)
        if symbols:
            symbols_title = QLabel("Symbols", content)
            style_section_title(symbols_title)
            content_layout.addWidget(symbols_title)
            for symbol, meaning in symbols:
                content_layout.addWidget(_create_symbol_row(symbol, meaning, content))

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ComparisonTableWidget(QWidget):
    """Display one persisted comparison table with metric help and export/delete actions."""

    delete_requested = Signal(object)
    title_changed = Signal(object, str)

    @staticmethod
    def _column_header_label(raw_label: object) -> str:
        label = str(raw_label).strip()
        return label if label.startswith("DFT (") else f"DFT ({label})"

    @staticmethod
    def _as_numeric(value: object) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _best_agreement_color() -> QColor:
        return QColor("#15803D")

    def _highlight_best_agreement(
        self,
        *,
        experimental_payload: object,
        dft_values: list[tuple[QTableWidgetItem, object]],
    ) -> None:
        if not isinstance(experimental_payload, dict):
            return
        experimental_value = self._as_numeric(experimental_payload.get("value"))
        if experimental_value is None or len(dft_values) < 2:
            return

        comparable: list[tuple[QTableWidgetItem, float]] = []
        for item, payload in dft_values:
            if not isinstance(payload, dict):
                continue
            value = self._as_numeric(payload.get("value"))
            if value is None:
                continue
            comparable.append((item, value))

        if len(comparable) < 2:
            return

        values = [value for _item, value in comparable]
        if max(values) - min(values) <= 1e-12:
            return

        deltas = [abs(value - experimental_value) for _item, value in comparable]
        best_delta = min(deltas)
        best_indices = [index for index, delta in enumerate(deltas) if abs(delta - best_delta) <= 1e-12]
        if len(best_indices) != 1:
            return

        best_item = comparable[best_indices[0]][0]
        best_item.setForeground(QBrush(self._best_agreement_color()))
        best_font = best_item.font()
        best_font.setBold(True)
        best_item.setFont(best_font)

    def __init__(self, table: ComparisonTable, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._table = table
        self._parameter_diff_rows: list[int] = []
        self._parameter_diff_section_row: int | None = None
        self._parameter_diff_toggle: QToolButton | None = None
        self._parameter_diff_expanded = True
        self._restoring_title = False

        self._title_edit = QLineEdit(self._table.label or "Comparison Table")
        self._title_edit.setObjectName("comparisonTableTitleEdit")
        self._title_edit.setMaxLength(80)
        title_font = self._title_edit.font()
        title_font.setBold(True)
        adjust_font_size(title_font, 2)
        self._title_edit.setFont(title_font)
        self._warning_banner = QLabel("")
        self._warning_banner.setWordWrap(True)
        self._warning_banner.hide()

        self._grid = QTableWidget(self)
        self._grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._grid.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._grid.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._grid.verticalHeader().setVisible(False)

        self._export_button = QPushButton("Export CSV...")
        self._delete_button = QPushButton("Remove")

        self._build_layout()
        self._connect_signals()
        self._populate_table()

    @property
    def comparison_table_id(self) -> UUID:
        return self._table.id

    def set_table_label(self, label: str) -> None:
        normalized = str(label).strip() or "Comparison Table"
        self._restoring_title = True
        try:
            self._title_edit.setText(normalized)
        finally:
            self._restoring_title = False

    def _build_layout(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(self._title_edit)
        header.addStretch(1)
        header.addWidget(self._export_button)
        header.addWidget(self._delete_button)
        root.addLayout(header)

        root.addWidget(self._warning_banner)
        root.addWidget(self._grid, 1)

    def _connect_signals(self) -> None:
        self._title_edit.editingFinished.connect(self._on_title_edit_finished)
        self._export_button.clicked.connect(self._export_csv)
        self._delete_button.clicked.connect(self._confirm_delete)

    def _on_title_edit_finished(self) -> None:
        normalized = self._title_edit.text().strip() or "Comparison Table"
        if self._title_edit.text() != normalized:
            self._title_edit.setText(normalized)
        if self._restoring_title:
            return
        current_label = self._table.label or "Comparison Table"
        if normalized == current_label:
            return
        self.title_changed.emit(self._table.id, normalized)

    def _populate_table(self) -> None:
        payload = self._table.result_data
        definitions = payload.get("metric_definitions", [])
        dft_columns = payload.get("dft_columns", [])
        experimental_values = payload.get("experimental_values", {})

        column_count = 2 + len([value for value in dft_columns if isinstance(value, dict)])
        self._grid.setColumnCount(column_count)
        headers = ["Metric", "Experimental"] + [
            self._column_header_label(column.get("label", "DFT"))
            for column in dft_columns
            if isinstance(column, dict)
        ]
        self._grid.setHorizontalHeaderLabels(headers)

        grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            group = str(definition.get("group", "Metrics"))
            grouped.setdefault(group, []).append(definition)

        self._grid.setRowCount(0)
        self._parameter_diff_rows = []
        self._parameter_diff_section_row = None
        self._parameter_diff_toggle = None

        parameter_rows = payload.get("parameter_differences", [])
        if isinstance(parameter_rows, list) and parameter_rows:
            section_row = self._grid.rowCount()
            self._grid.insertRow(section_row)
            self._parameter_diff_section_row = section_row
            self._set_parameter_diff_header_row(section_row, column_count)

            for parameter_row in parameter_rows:
                if not isinstance(parameter_row, dict):
                    continue
                row = self._grid.rowCount()
                self._grid.insertRow(row)
                self._parameter_diff_rows.append(row)

                name = str(parameter_row.get("name", ""))
                unit = str(parameter_row.get("unit", ""))
                label = name if not unit else f"{name} ({unit})"
                self._grid.setItem(row, 0, QTableWidgetItem(label))
                self._grid.setItem(row, 1, QTableWidgetItem("-"))

                values = parameter_row.get("values", [])
                output_column = 2
                if isinstance(values, list):
                    for value in values[: column_count - 2]:
                        self._grid.setItem(row, output_column, QTableWidgetItem(str(value)))
                        output_column += 1

            self._set_parameter_diff_visibility(self._parameter_diff_expanded)

        for group, metrics in grouped.items():
            group_row = self._grid.rowCount()
            self._grid.insertRow(group_row)
            item = QTableWidgetItem(group)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._grid.setItem(group_row, 0, item)
            self._grid.setSpan(group_row, 0, 1, column_count)

            for definition in metrics:
                metric_row = self._grid.rowCount()
                self._grid.insertRow(metric_row)
                self._set_metric_header_cell(metric_row, definition)

                key = str(definition.get("key", ""))
                experimental = experimental_values.get(key, {})
                exp_display = ""
                if isinstance(experimental, dict):
                    exp_display = str(experimental.get("display", ""))
                self._grid.setItem(metric_row, 1, QTableWidgetItem(exp_display))

                output_column = 2
                dft_metric_values: list[tuple[QTableWidgetItem, object]] = []
                for dft_column in dft_columns:
                    if not isinstance(dft_column, dict):
                        continue
                    values = dft_column.get("values", {})
                    metric_payload = values.get(key, {}) if isinstance(values, dict) else {}
                    display = ""
                    if isinstance(metric_payload, dict):
                        display = str(metric_payload.get("display", ""))
                    item = QTableWidgetItem(display)
                    self._grid.setItem(metric_row, output_column, item)
                    dft_metric_values.append((item, metric_payload))
                    output_column += 1

                self._highlight_best_agreement(
                    experimental_payload=experimental,
                    dft_values=dft_metric_values,
                )

        warnings = payload.get("warnings", [])
        if isinstance(warnings, list) and warnings:
            warning_text = "\n".join(str(value) for value in warnings)
            self._warning_banner.setText(warning_text)
            self._warning_banner.show()
        else:
            self._warning_banner.hide()

        self._grid.resizeColumnsToContents()

    def _set_parameter_diff_header_row(self, row: int, column_count: int) -> None:
        self._grid.setSpan(row, 0, 1, column_count)

        container = QWidget(self._grid)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(2, 0, 4, 0)
        layout.setSpacing(0)

        toggle = QToolButton(container)
        toggle.setObjectName("sectionToggleButton")
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setText("DFT Parameter Differences")
        toggle.toggled.connect(self._set_parameter_diff_visibility)

        layout.addWidget(toggle)
        layout.addStretch(1)

        self._grid.setCellWidget(row, 0, container)
        self._parameter_diff_toggle = toggle
        self._sync_parameter_diff_header()

    def _sync_parameter_diff_header(self) -> None:
        if self._parameter_diff_toggle is None:
            return
        self._parameter_diff_toggle.blockSignals(True)
        self._parameter_diff_toggle.setChecked(self._parameter_diff_expanded)
        self._parameter_diff_toggle.setArrowType(
            Qt.ArrowType.DownArrow if self._parameter_diff_expanded else Qt.ArrowType.RightArrow
        )
        self._parameter_diff_toggle.blockSignals(False)

    def _set_metric_header_cell(self, row: int, definition: dict) -> None:
        container = QWidget(self._grid)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(6)

        label = str(definition.get("label", ""))
        unit = str(definition.get("unit", "")).strip()
        text = label if not unit else f"{label} ({unit})"
        text_label = QLabel(text, container)
        text_label.setWordWrap(True)

        info_button = QPushButton("Info", container)
        info_button.setMaximumWidth(52)
        info_button.clicked.connect(
            lambda _checked=False, definition=definition: self._show_metric_info(definition)
        )

        layout.addWidget(text_label, 1)
        layout.addWidget(info_button)
        self._grid.setCellWidget(row, 0, container)

    def _show_metric_info(self, definition: dict) -> None:
        dialog = ComparisonMetricHelpDialog(definition, parent=self)
        dialog.exec()

    def _set_parameter_diff_visibility(self, visible: bool) -> None:
        self._parameter_diff_expanded = visible
        for row in self._parameter_diff_rows:
            self._grid.setRowHidden(row, not visible)
        self._sync_parameter_diff_header()

    def _export_csv(self) -> None:
        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Comparison CSV",
            f"{self._table.label or 'comparison'}.csv",
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            export_comparison_csv(self._table, Path(file_path))
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return

        QMessageBox.information(self, "Export complete", f"Saved CSV to:\n{file_path}")

    def _confirm_delete(self) -> None:
        answer = QMessageBox.question(
            self,
            "Delete comparison table",
            "Delete this comparison table?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.delete_requested.emit(self._table.id)

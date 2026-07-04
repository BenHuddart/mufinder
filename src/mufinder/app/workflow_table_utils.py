"""Shared helpers for workflow job/result tables."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QFrame, QHeaderView, QPushButton, QTableWidget, QTableWidgetItem

from mufinder.app.theme import use_square_top_table_corners


def create_workflow_table(
    headers: tuple[str, ...],
    *,
    resize_to_contents_columns: tuple[int, ...] = (),
) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.setFrameShape(QFrame.Shape.NoFrame)
    # Long values (file paths, labels) must elide instead of wrapping, so
    # resizeRowsToContents() cannot balloon row heights.
    table.setWordWrap(False)
    table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    for column in resize_to_contents_columns:
        if 0 <= column < len(headers):
            table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    use_square_top_table_corners(table)
    return table


def set_workflow_table_item(table: QTableWidget, row: int, column: int, text: str) -> None:
    item = table.item(row, column)
    if item is None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(row, column, item)
        return
    item.setText(text)


def workflow_row_is_locked(status: str) -> bool:
    return str(status).strip().casefold() not in {"", "ready", "defined", "not generated"}


def style_workflow_configure_button(
    button: QPushButton,
    *,
    workflow_label: str,
    configured: bool,
    locked: bool,
    noun: str = "row",
    used: bool = False,
) -> None:
    """Apply the shared Configure/Configured text and tooltip to ``button``.

    ``noun`` names the per-row entity in the tooltip (default ``"row"``; e.g.
    ``"UEP source job"``). ``used`` selects the ``"parameters used for"`` phrasing
    of the locked (inspect) tooltip; the default ``"parameters for"`` phrasing is
    preserved for callers that do not opt in.
    """

    button.setText("Configured..." if configured or locked else "Configure...")
    inspect_phrase = (
        f"Inspect the {workflow_label} parameters used for this {noun}."
        if used
        else f"Inspect the {workflow_label} parameters for this {noun}."
    )
    button.setToolTip(
        inspect_phrase
        if locked
        else (
            f"Edit custom {workflow_label} parameters for this {noun}."
            if configured
            else f"Configure {workflow_label} parameters for this {noun}."
        )
    )

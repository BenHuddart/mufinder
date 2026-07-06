"""Shared helpers for workflow job/result tables."""

from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from mufinder.app.theme import use_square_top_table_corners


def bind_delete_shortcuts(
    widget: QWidget,
    slot: Callable[[], None],
    *,
    context: Qt.ShortcutContext = Qt.ShortcutContext.WidgetWithChildrenShortcut,
) -> tuple[QShortcut, QShortcut]:
    """Bind a delete/remove ``slot`` to both Delete and Backspace on ``widget``.

    macOS delivers the physical Delete key as ``Qt.Key.Key_Backspace``, so
    CLAUDE.md requires every delete/remove/cancel/reset shortcut to bind both
    ``QKeySequence.StandardKey.Delete`` and ``Qt.Key.Key_Backspace`` — binding
    only the standard "Delete" key sequence leaves macOS users unable to
    trigger the action from the keyboard. Returns the two ``QShortcut``
    instances so callers can keep references alive (Qt shortcuts are owned by
    their parent widget, but callers historically stash them on ``self`` as
    well).
    """

    delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, widget)
    delete_shortcut.setContext(context)
    delete_shortcut.activated.connect(slot)
    backspace_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backspace), widget)
    backspace_shortcut.setContext(context)
    backspace_shortcut.activated.connect(slot)
    return delete_shortcut, backspace_shortcut


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

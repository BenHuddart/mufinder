"""Snapshot-aware in-cell table widgets and multi-row action resolver.

On Windows, clicking a widget embedded in a QTableWidget cell via setCellWidget
collapses the table's multi-row selection to the clicked row *before* any
clicked/currentIndexChanged slot fires.  Handlers that call
selectionModel().selectedRows() inside the slot therefore always see a single
row, breaking multi-row batch actions.

The widgets in this module capture the selection at mousePressEvent /
keyPressEvent time — the last moment before the platform's focus-management
can collapse it — and stash the snapshot on the widget itself.  Handlers then
read from the snapshot via resolve_target_keys() rather than querying the live
selection model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QComboBox, QMessageBox, QPushButton, QTableWidget, QToolButton, QWidget

KeyT = TypeVar("KeyT")


class _SnapshotMixin:
    """Mixin that captures the enclosing QTableWidget's selected row indexes.

    Mix this in before the Qt base class so MRO ensures our overrides fire
    first, then delegate to the Qt implementation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.selection_snapshot_rows: tuple[int, ...] = ()
        self.selection_snapshot_key: object = None

    def _capture_selection_snapshot(self) -> None:
        table = self._owning_table()
        if table is None:
            self.selection_snapshot_rows = ()
            return
        model = table.selectionModel()
        if model is None:
            self.selection_snapshot_rows = ()
            return
        self.selection_snapshot_rows = tuple(
            sorted(idx.row() for idx in model.selectedRows())
        )

    def _owning_table(self) -> QTableWidget | None:
        # Cell widgets are reparented under the viewport, so parent() is the
        # viewport, not the table.  Walk up at most a few hops.
        parent: QWidget | None = self.parent()  # type: ignore[attr-defined]
        for _ in range(4):
            if parent is None:
                return None
            if isinstance(parent, QTableWidget):
                return parent
            parent = parent.parent()
        return None


class SelectionSnapshotPushButton(_SnapshotMixin, QPushButton):
    """QPushButton that snapshots its table's selection on mouse/keyboard activation."""

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._capture_selection_snapshot()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._capture_selection_snapshot()
        super().keyPressEvent(event)


class SelectionSnapshotToolButton(_SnapshotMixin, QToolButton):
    """QToolButton that snapshots its table's selection on mouse/keyboard activation."""

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._capture_selection_snapshot()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._capture_selection_snapshot()
        super().keyPressEvent(event)


class SelectionSnapshotComboBox(_SnapshotMixin, QComboBox):
    """QComboBox that snapshots its table's selection on mouse/keyboard activation.

    Snapshots at mousePressEvent (which opens the popup) and on navigation
    keys that can change currentIndex without opening the popup.
    """

    _NAV_KEYS = frozenset((
        Qt.Key.Key_Up,
        Qt.Key.Key_Down,
        Qt.Key.Key_PageUp,
        Qt.Key.Key_PageDown,
    ))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._capture_selection_snapshot()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in self._NAV_KEYS:
            self._capture_selection_snapshot()
        super().keyPressEvent(event)


def resolve_target_keys(
    widget: _SnapshotMixin,
    *,
    clicked_key: KeyT,
    snapshot_keys: Sequence[KeyT],
    eligibility: Callable[[KeyT], bool],
    parent: QWidget,
    title: str = "Apply to all selected rows?",
    message: str | None = None,
) -> tuple[KeyT, ...]:
    """Return the row keys the action should apply to, prompting when appropriate.

    Rules
    -----
    1. If *clicked_key* is not in *snapshot_keys* the user clicked a button in
       a row that isn't part of the current selection → return (clicked_key,),
       no prompt.
    2. Filter *snapshot_keys* through *eligibility*.  If the eligible subset
       has ≤ 1 entry (effectively only the clicked row) → return (clicked_key,),
       no prompt.
    3. Otherwise show a QMessageBox.question.
       Yes → return the eligible subset (snapshot order preserved).
       No  → return (clicked_key,).

    The snapshot fields on *widget* are cleared after each call so stale-read
    bugs become obvious (the next call without a new press returns the
    single-row fallback).
    """
    snapshot = tuple(snapshot_keys)

    # Clear the snapshot immediately — consumed.
    widget.selection_snapshot_rows = ()
    widget.selection_snapshot_key = None

    if clicked_key not in snapshot:
        return (clicked_key,)

    eligible = tuple(k for k in snapshot if eligibility(k))

    if len(eligible) <= 1:
        return (clicked_key,)

    n_snapshot = len(snapshot)
    n_eligible = len(eligible)

    if message is None:
        if n_eligible < n_snapshot:
            body = (
                f"{n_eligible} of {n_snapshot} selected rows are eligible. "
                f"Apply to those {n_eligible} rows?"
            )
        else:
            body = f"Apply to all {n_eligible} selected rows?"
    else:
        body = message

    response = QMessageBox.question(
        parent,
        title,
        body,
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if response == QMessageBox.StandardButton.Yes:
        return eligible
    return (clicked_key,)

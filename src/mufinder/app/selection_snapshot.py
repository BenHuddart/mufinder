"""Snapshot-aware in-cell table widgets and multi-row action resolver.

Moving focus to a widget embedded in a QTableWidget cell via setCellWidget makes
the view treat that cell as current, which collapses the table's multi-row
selection to the widget's row.  It happens on every platform, before the click
that caused it is delivered, so a handler that reads
selectionModel().selectedRows() always sees a single row and a multi-row batch
action silently runs on one.

The widgets in this module deal with that in two steps.  They record the
selection before the focus change collapses it and put it back once they have
focus (see the notes on _SnapshotMixin), so a click on a selected row leaves the
selection as the user made it.  They then capture the selection at
mousePressEvent / keyPressEvent time and stash the snapshot on the widget
itself; handlers read it via resolve_target_keys() rather than querying the
live selection model.

The targeting rule the widgets and resolver implement together is "highlighted
rows are what is affected":

- a control on a row inside a multi-row selection targets the selection, after
  a three-way prompt (all selected rows / this row only / cancel);
- a control on a row *outside* the selection first makes that row the
  selection, then acts on it alone;
- once the prompt is answered the table's selection is put back to match the
  rows the action applies to, undoing any platform collapse.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import Enum
from typing import TypeVar

from PySide6.QtCore import QEvent, QItemSelection, QItemSelectionModel, QObject, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
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
        self._pre_focus_rows: tuple[int, ...] = ()

    # -- keeping the selection across focus changes --------------------------
    #
    # When focus moves to a cell widget the view makes that cell current, which
    # collapses a multi-row selection to the widget's row.  It happens before a
    # click's press event is delivered, so by the time anything could snapshot
    # the selection it is already gone.  There are two routes:
    #
    # - focus leaves the table itself for the widget: the table collapses the
    #   selection while handling its own FocusOut, before the widget hears of
    #   the change.  A per-table _SelectionKeeper records the selection first.
    # - focus arrives from anywhere else (another cell widget, a combo popup
    #   closing): the view collapses it from its event filter on the widget.
    #   A filter of our own, installed later and therefore run first, records
    #   the selection.
    #
    # focusInEvent then puts the recorded selection back when this widget's row
    # was part of it.  A widget on a row outside the selection is left alone,
    # so the highlight moves to its row.

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        # setCellWidget installs the view's filter and then shows the widget;
        # (re)installing here keeps ours ahead of it.
        self.installEventFilter(self)  # type: ignore[attr-defined]
        table = self._owning_table()
        if table is not None:
            _SelectionKeeper.ensure(table)
        super().showEvent(event)  # type: ignore[misc]

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        if watched is self and event.type() == QEvent.Type.FocusIn:
            self._pre_focus_rows = self._table_selected_rows()
        return super().eventFilter(watched, event)  # type: ignore[misc]

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().focusInEvent(event)  # type: ignore[misc]
        rows = self._pre_focus_rows
        self._pre_focus_rows = ()
        table = self._owning_table()
        if table is None:
            return
        if len(rows) < 2:
            keeper = _SelectionKeeper.ensure(table)
            rows = keeper.take_rows()
        if len(rows) >= 2 and self._own_table_row(table) in rows:
            _select_table_rows(table, rows)

    def _table_selected_rows(self) -> tuple[int, ...]:
        table = self._owning_table()
        return _selected_rows(table) if table is not None else ()

    def _capture_selection_snapshot(self) -> None:
        table = self._owning_table()
        if table is None:
            self.selection_snapshot_rows = ()
            return
        model = table.selectionModel()
        if model is None:
            self.selection_snapshot_rows = ()
            return
        rows = tuple(sorted(idx.row() for idx in model.selectedRows()))
        own_row = self._own_table_row(table)
        if own_row is not None and own_row not in rows:
            # A control on a row outside the selection acts on its own row
            # only, so move the highlight there before the action runs: what
            # is highlighted is always what is affected.
            table.selectRow(own_row)
            rows = (own_row,)
        self.selection_snapshot_rows = rows

    def _own_table_row(self, table: QTableWidget) -> int | None:
        """Return the row of *table* this widget is installed in as a cell widget."""
        for row in range(table.rowCount()):
            for column in range(table.columnCount()):
                if table.cellWidget(row, column) is self:
                    return row
        return None

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


def _selected_rows(table: QTableWidget) -> tuple[int, ...]:
    model = table.selectionModel()
    if model is None:
        return ()
    return tuple(sorted(idx.row() for idx in model.selectedRows()))


class _SelectionKeeper(QObject):
    """Remembers a table's selection as it was when focus began to leave it.

    One per table, created by the first snapshot widget shown in it.  The
    record only describes the focus change in progress, so it is dropped as
    soon as control returns to the event loop.
    """

    _OBJECT_NAME = "mufinderSelectionKeeper"

    def __init__(self, table: QTableWidget) -> None:
        super().__init__(table)
        self.setObjectName(self._OBJECT_NAME)
        self._table = table
        self._rows: tuple[int, ...] = ()
        table.installEventFilter(self)

    @classmethod
    def ensure(cls, table: QTableWidget) -> "_SelectionKeeper":
        existing = table.findChild(
            cls, cls._OBJECT_NAME, Qt.FindChildOption.FindDirectChildrenOnly
        )
        return existing if existing is not None else cls(table)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        if watched is self._table and event.type() in (
            QEvent.Type.FocusAboutToChange,
            QEvent.Type.FocusOut,
        ):
            if not self._rows:
                self._rows = _selected_rows(self._table)
                QTimer.singleShot(0, self._forget)
        return False

    def _forget(self) -> None:
        self._rows = ()

    def take_rows(self) -> tuple[int, ...]:
        rows = self._rows
        self._rows = ()
        return rows


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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pre_interaction_index: int = -1

    def _capture_selection_snapshot(self) -> None:
        # Remember the value the user started from, so a cancelled multi-row
        # prompt can put the combo back (see restore_pre_interaction_index).
        self.pre_interaction_index = self.currentIndex()
        super()._capture_selection_snapshot()

    def restore_pre_interaction_index(self) -> None:
        """Put the combo back to the index it showed before the user changed it.

        Signals are blocked so the owning widget's change handler does not run
        again for what is an undo, not a new choice.
        """
        if not 0 <= self.pre_interaction_index < self.count():
            return
        previous = self.blockSignals(True)
        self.setCurrentIndex(self.pre_interaction_index)
        self.blockSignals(previous)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._capture_selection_snapshot()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in self._NAV_KEYS:
            self._capture_selection_snapshot()
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        # The wheel changes currentIndex without a press, so without this the
        # change handler would read a snapshot left over from an earlier press.
        self._capture_selection_snapshot()
        super().wheelEvent(event)


class SelectionScope(Enum):
    """The user's answer to the apply-to-selection prompt."""

    ALL_SELECTED = "all_selected"
    CLICKED_ONLY = "clicked_only"
    CANCEL = "cancel"


def ask_selection_scope(
    parent: QWidget,
    title: str,
    body: str,
    all_label: str,
    one_label: str,
) -> SelectionScope:
    """Ask whether an in-row action applies to the selection, one row, or nothing.

    Kept as a module-level function so tests can replace it instead of driving
    a modal dialog.
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(body)
    all_button = box.addButton(all_label, QMessageBox.ButtonRole.AcceptRole)
    one_button = box.addButton(one_label, QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(all_button)
    box.exec()
    clicked = box.clickedButton()
    if clicked is all_button:
        return SelectionScope.ALL_SELECTED
    if clicked is one_button:
        return SelectionScope.CLICKED_ONLY
    return SelectionScope.CANCEL


def _select_table_rows(table: QTableWidget, rows: Sequence[int]) -> None:
    """Make *rows* the table's selection, leaving it alone when already so."""
    model = table.selectionModel()
    if model is None:
        return
    wanted = sorted({row for row in rows if 0 <= row < table.rowCount()})
    if wanted == sorted(idx.row() for idx in model.selectedRows()):
        return
    selection = QItemSelection()
    for row in wanted:
        index = table.model().index(row, 0)
        selection.select(index, index)
    model.select(
        selection,
        QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
    )


def resolve_target_keys(
    widget: _SnapshotMixin,
    *,
    clicked_key: KeyT,
    snapshot_keys: Sequence[KeyT],
    eligibility: Callable[[KeyT], bool],
    parent: QWidget,
    action: str = "Apply to",
    title: str | None = None,
    noun: tuple[str, str] = ("row", "rows"),
    confirm_unprompted: Callable[[], bool] | None = None,
) -> tuple[KeyT, ...]:
    """Return the row keys the action should apply to, prompting when appropriate.

    Rules
    -----
    1. If *clicked_key* is not in *snapshot_keys* the user clicked a control in
       a row that isn't part of the current selection → return (clicked_key,),
       no prompt.
    2. Filter *snapshot_keys* through *eligibility*.  If the eligible subset
       has ≤ 1 entry (effectively only the clicked row) → return (clicked_key,),
       no prompt.
    3. Otherwise ask via :func:`ask_selection_scope`.
       All selected → return the eligible subset (snapshot order preserved).
       This row only → return (clicked_key,).
       Cancel → return (), so the caller does nothing (and a combo should put
       its previous value back).

    *confirm_unprompted* is for destructive actions, which must always be
    confirmed.  The three-way prompt is that confirmation when it is shown; in
    rules 1 and 2, where it is not, this callable is asked instead and a False
    answer returns () just as Cancel does.

    *action* is the verb phrase the prompt is built from ("Remove", "Set this
    supercell on", …) and *noun* the singular/plural name of a row, so every
    table words the question the same way and reports rows it will skip.

    The snapshot fields on *widget* are cleared after each call so stale-read
    bugs become obvious (the next call without a new press returns the
    single-row fallback).  When the prompt was shown, the owning table's
    selection is then set to match the answer: the snapshot rows for "all" and
    for cancel, the clicked row alone for "this row only".
    """
    snapshot = tuple(snapshot_keys)
    snapshot_rows = tuple(widget.selection_snapshot_rows)

    # Clear the snapshot immediately — consumed.
    widget.selection_snapshot_rows = ()
    widget.selection_snapshot_key = None

    def _clicked_only() -> tuple[KeyT, ...]:
        if confirm_unprompted is not None and not confirm_unprompted():
            return ()
        return (clicked_key,)

    if clicked_key not in snapshot:
        return _clicked_only()

    eligible = tuple(k for k in snapshot if eligibility(k))

    if len(eligible) <= 1:
        return _clicked_only()

    singular, plural = noun
    n_snapshot = len(snapshot)
    n_eligible = len(eligible)

    if n_eligible < n_snapshot:
        n_skipped = n_snapshot - n_eligible
        body = (
            f"{action} {n_eligible} of {n_snapshot} selected {plural}? "
            f"{n_skipped} {'is' if n_skipped == 1 else 'are'} not eligible and will be skipped."
        )
        all_label = f"{n_eligible} eligible {plural}"
    else:
        body = f"{action} all {n_eligible} selected {plural}?"
        all_label = f"All {n_eligible} selected {plural}"

    scope = ask_selection_scope(
        parent,
        title if title is not None else f"Apply to selected {plural}?",
        body,
        all_label,
        f"This {singular} only",
    )

    table = widget._owning_table()
    if table is not None:
        own_row = widget._own_table_row(table)
        if scope is SelectionScope.CLICKED_ONLY and own_row is not None:
            _select_table_rows(table, (own_row,))
        elif scope is not SelectionScope.CLICKED_ONLY and snapshot_rows:
            _select_table_rows(table, snapshot_rows)

    if scope is SelectionScope.ALL_SELECTED:
        return eligible
    if scope is SelectionScope.CLICKED_ONLY:
        return (clicked_key,)
    return ()

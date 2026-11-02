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

import shiboken6
from PySide6.QtCore import QEvent, QItemSelection, QItemSelectionModel, QObject, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTabWidget,
    QToolButton,
    QWidget,
)

from mufinder.app.eliding_label import ElidingComboBox
from mufinder.app.theme import SPACING

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
        #: Whether this control acts on the whole selection when its row is
        #: part of a multi-row selection. Set False on a control that only ever
        #: acts on its own row, so it is not highlighted as a multi-row target.
        self.propagates_to_selection: bool = True
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
        """Return the row of *table* this widget is in, directly or inside a controls cell."""
        position = _cell_position(table, self, self.parent())  # type: ignore[attr-defined]
        return position[0] if position is not None else None

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


def _cell_position(table: QTableWidget, *widgets: QWidget | None) -> tuple[int, int] | None:
    """Return the ``(row, column)`` of the cell whose widget is one of ``widgets``."""

    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            cell = table.cellWidget(row, column)
            if cell is not None and any(cell is widget for widget in widgets):
                return row, column
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
        existing = table.findChild(cls, cls._OBJECT_NAME, Qt.FindChildOption.FindDirectChildrenOnly)
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


class RowControlsCell(QWidget):
    """A cell of a ``RowControlsTable`` holding several snapshot controls side by side.

    Each control keeps the targeting rule of a control installed directly in the
    cell. ``row_controls`` is what ``SelectionFeedback`` marks as multi-row
    targets. Focus on a control in here is not focus on the cell widget, so it
    does not collapse a multi-row selection.

    The cell never takes focus itself. The view calls ``setFocus`` on the cell
    widget of whichever cell becomes current, including when a row removed from
    the table moves the current cell, wherever focus is at the time; it skips a
    cell widget that has no focus policy. The controls take focus from a click,
    and from the keys handled here and in ``RowControlsTable``. They are not on
    the window's own Tab path, which would visit every row.
    """

    def __init__(self, *controls: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.row_controls: tuple[QWidget, ...] = controls
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.xs)
        for control in controls:
            control.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            control.installEventFilter(self)
            layout.addWidget(control)

    def reachable_controls(self) -> tuple[QWidget, ...]:
        """The controls that can take focus now: enabled and not hidden."""

        return tuple(
            control
            for control in self.row_controls
            if control.isEnabled() and control.isVisibleTo(self)
        )

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        # A button left alone treats the arrow keys as Tab and Shift+Tab.
        if event.type() == QEvent.Type.KeyPress and watched in self.row_controls:
            key = event.key()
            plain = not event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                table = self._controls_table()
                if plain and table is not None:
                    table.move_focus_across_rows(self, watched, -1 if key == Qt.Key.Key_Up else 1)
                return True
            if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                neighbour = self._next_reachable_control(watched, key == Qt.Key.Key_Right)
                if plain and neighbour is not None:
                    neighbour.setFocus(Qt.FocusReason.ShortcutFocusReason)
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API naming.
        # A key a control has no use for stops here. In the table, Page Down and the
        # like would move the highlight away from the row that has focus.
        event.accept()

    def focusNextPrevChild(self, next: bool) -> bool:  # noqa: A002, N802 - Qt API naming.
        focused = QApplication.focusWidget()
        if focused in self.row_controls:
            neighbour = self._next_reachable_control(focused, next)
            if neighbour is None and focused not in self.reachable_controls():
                # Qt is moving focus off a control just disabled or hidden, not the user.
                neighbour = self._next_reachable_control(focused, not next)
            if neighbour is not None:
                neighbour.setFocus(
                    Qt.FocusReason.TabFocusReason if next else Qt.FocusReason.BacktabFocusReason
                )
                return True
        table = self._controls_table()
        if table is not None:
            return table.tab_out_of_row(self, next)
        return super().focusNextPrevChild(next)

    def _next_reachable_control(self, control: QWidget, forward: bool) -> QWidget | None:
        position = self.row_controls.index(control)
        candidates = (
            self.row_controls[position + 1 :] if forward else reversed(self.row_controls[:position])
        )
        reachable = self.reachable_controls()
        return next((other for other in candidates if other in reachable), None)

    def _controls_table(self) -> RowControlsTable | None:
        parent = self.parentWidget()
        while parent is not None and not isinstance(parent, RowControlsTable):
            parent = parent.parentWidget()
        return parent


class RowControlsTable(QTableWidget):
    """A table with a ``RowControlsCell`` in each row, and one Tab stop for all of them.

    Tab goes from the table to the current row's controls and then on to the next
    widget in the window; Shift+Tab comes back the same way. Up and Down on a control
    move to the same control on the next row, so a table of any length costs the
    keyboard the same few presses to pass. Tab on a plain ``QTableWidget`` walks its
    cells and wraps round, so it never leaves.

    When a row is removed with focus on one of its controls, focus moves to the same
    control on the row below, or above when none below can take it, and to the table
    when no row can. Focus anywhere else is not touched, and nor is the selection,
    since a row can leave without the user doing anything.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setTabKeyNavigation(False)

    def focusNextPrevChild(self, next: bool) -> bool:  # noqa: A002, N802 - Qt API naming.
        if next and self.isVisible() and QApplication.focusWidget() is self:
            row = self.currentRow()
            cell = self._controls_cell_in_row(row)
            reachable = cell.reachable_controls() if cell is not None else ()
            if reachable:
                self.scrollTo(self.currentIndex())
                reachable[0].setFocus(Qt.FocusReason.TabFocusReason)
                return True
        return super().focusNextPrevChild(next)

    def tab_out_of_row(self, cell: RowControlsCell, forward: bool) -> bool:
        """Leave ``cell``'s row: Shift+Tab to the table, Tab on past it."""

        # A click focuses a control without moving the view's current cell.
        position = _cell_position(self, cell)
        if position is not None:
            self._set_current_cell_keeping_selection(*position)
        if forward:
            # The window's Tab path runs on from the widget that has focus.
            self.setFocus(Qt.FocusReason.TabFocusReason)
            return super().focusNextPrevChild(True)
        self.setFocus(Qt.FocusReason.BacktabFocusReason)
        return True

    def move_focus_across_rows(self, cell: RowControlsCell, control: QWidget, step: int) -> None:
        """Move focus to ``control``'s counterpart ``step`` rows on, past rows with none.

        This is the arrow keys in the table for a row's controls: a row outside the
        selection becomes the selection.
        """

        position = _cell_position(self, cell)
        if position is None:
            return
        slot = cell.row_controls.index(control)
        row = position[0] + step
        while 0 <= row < self.rowCount():
            counterpart = self._counterpart(row, slot)
            if counterpart is not None:
                if row not in _selected_rows(self):
                    _select_table_rows(self, (row,))
                self._set_current_cell_keeping_selection(row, position[1])
                self.scrollTo(self.currentIndex())
                # Focus moved by a handler is keyboard use only with this reason.
                counterpart.setFocus(Qt.FocusReason.ShortcutFocusReason)
                return
            row += step

    def rowsAboutToBeRemoved(self, parent, start: int, end: int) -> None:  # noqa: N802 - Qt API.
        self._pass_focus_from_rows(start, end)
        super().rowsAboutToBeRemoved(parent, start, end)

    def _pass_focus_from_rows(self, start: int, end: int) -> None:
        # The window's focus widget, not the application's: a dialog or menu can be open
        # over the window, and focus comes back to whatever the window holds.
        focused = self.window().focusWidget()
        cell = focused.parentWidget() if focused is not None else None
        if not isinstance(cell, RowControlsCell):
            return
        position = _cell_position(self, cell)
        if position is None or not start <= position[0] <= end:
            return
        slot = cell.row_controls.index(focused)
        for row in (*range(end + 1, self.rowCount()), *range(start - 1, -1, -1)):
            counterpart = self._counterpart(row, slot)
            if counterpart is not None:
                self._set_current_cell_keeping_selection(row, position[1])
                counterpart.setFocus(Qt.FocusReason.OtherFocusReason)
                return
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def _counterpart(self, row: int, slot: int) -> QWidget | None:
        """The control at ``slot`` on ``row``, or its first that can take focus."""

        cell = self._controls_cell_in_row(row)
        if cell is None:
            return None
        reachable = cell.reachable_controls()
        if cell.row_controls[slot] in reachable:
            return cell.row_controls[slot]
        return reachable[0] if reachable else None

    def _set_current_cell_keeping_selection(self, row: int, column: int) -> None:
        self.selectionModel().setCurrentIndex(
            self.model().index(row, column), QItemSelectionModel.SelectionFlag.NoUpdate
        )

    def _controls_cell_in_row(self, row: int) -> RowControlsCell | None:
        for column in range(self.columnCount()):
            cell = self.cellWidget(row, column)
            if isinstance(cell, RowControlsCell):
                return cell
        return None


class SelectionSnapshotComboBox(_SnapshotMixin, ElidingComboBox):
    """QComboBox that snapshots its table's selection on mouse/keyboard activation.

    Snapshots at mousePressEvent (which opens the popup) and on navigation
    keys that can change currentIndex without opening the popup.
    """

    _NAV_KEYS = frozenset(
        (
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
            Qt.Key.Key_PageUp,
            Qt.Key.Key_PageDown,
        )
    )

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


class SelectionSnapshotCheckBox(_SnapshotMixin, QCheckBox):
    """QCheckBox that snapshots its table's selection on mouse/keyboard activation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pre_interaction_checked: bool = False

    def _capture_selection_snapshot(self) -> None:
        # Remember the state the user started from, so a cancelled multi-row
        # prompt can put the box back (see restore_pre_interaction_state).
        self.pre_interaction_checked = self.isChecked()
        super()._capture_selection_snapshot()

    def restore_pre_interaction_state(self) -> None:
        """Put the box back to the state it showed before the user changed it.

        Signals are blocked so the owning widget's change handler does not run
        again for what is an undo, not a new choice.
        """
        previous = self.blockSignals(True)
        self.setChecked(self.pre_interaction_checked)
        self.blockSignals(previous)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._capture_selection_snapshot()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._capture_selection_snapshot()
        super().keyPressEvent(event)


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


def select_and_reveal_table_rows(table: QTableWidget, rows: Sequence[int]) -> bool:
    """Select *rows* of *table* and bring the first into view; False when there are none.

    A table on a hidden tab (a charge-state tab, say) has that tab shown first.
    """
    if not rows:
        return False
    ancestor = table.parentWidget()
    while ancestor is not None:
        if isinstance(ancestor, QTabWidget):
            for index in range(ancestor.count()):
                page = ancestor.widget(index)
                if page is table or page.isAncestorOf(table):
                    ancestor.setCurrentIndex(index)
                    break
        ancestor = ancestor.parentWidget()
    first = table.model().index(min(rows), 0)
    table.setCurrentIndex(first)
    _select_table_rows(table, rows)
    table.scrollTo(first)
    return True


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

    # A refresh during the prompt can have removed the control's row.
    table = widget._owning_table() if shiboken6.isValid(widget) else None
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

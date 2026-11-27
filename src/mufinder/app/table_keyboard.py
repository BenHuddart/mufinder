"""The keyboard path through a table: one Tab stop, however many rows it has.

Every table in ``mufinder.app`` is a ``TabStopTable``. A plain ``QTableWidget`` walks
its cells on Tab and wraps round from the last cell to the first, so Tab never leaves
it and the controls after it cannot be reached from the keyboard.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QEvent, QItemSelectionModel, QObject, Qt
from PySide6.QtGui import QInputMethodEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QTableWidget, QWidget


def cell_position(table: QTableWidget, *widgets: QWidget | None) -> tuple[int, int] | None:
    """Return the ``(row, column)`` of the cell whose widget is one of ``widgets``."""

    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            cell = table.cellWidget(row, column)
            if cell is not None and any(cell is widget for widget in widgets):
                return row, column
    return None


def _take_off_tab_path(widget: QWidget, wheel_focus: _WheelFocus) -> None:
    """Stop ``widget`` and everything in it being a stop on the window's Tab path.

    A widget that Tab could reach is marked as one of its row's controls first, since
    its focus policy no longer says so: a tool button has Tab focus alone, and keeps
    none. Clicks and ``setFocus`` still give a control focus as before, and so does the
    mouse wheel, through ``wheel_focus``, for one that had wheel focus. Qt counts
    wheel focus only together with Tab focus. ``children()`` rather than
    ``findChildren``: the latter ties the children's Python wrappers to ``widget``'s,
    which can close a reference cycle with Qt-owned widgets.
    """

    policy = widget.focusPolicy()
    if policy.value & Qt.FocusPolicy.TabFocus.value:
        widget.setProperty(_ROW_CONTROL_PROPERTY, True)
        if policy.value & _WHEEL_FOCUS == _WHEEL_FOCUS:
            widget.installEventFilter(wheel_focus)
        widget.setFocusPolicy(Qt.FocusPolicy(policy.value & ~Qt.FocusPolicy.TabFocus.value))
    for child in widget.children():
        if isinstance(child, QWidget):
            _take_off_tab_path(child, wheel_focus)


_WHEEL_FOCUS = Qt.FocusPolicy.WheelFocus.value


class _WheelFocus(QObject):
    """Give a cell control focus when the wheel turns over it, as its wheel focus did."""

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        if (
            event.type() == QEvent.Type.Wheel
            and isinstance(watched, QWidget)
            and watched.isEnabled()
            and not watched.hasFocus()
        ):
            watched.setFocus(Qt.FocusReason.MouseFocusReason)
        return False


class TabStopTable(QTableWidget):
    """A table that is one stop on the way round the window, however many rows it has.

    Tab goes from the table to the controls in the current row's cells, in column
    order, and then on to the next control in the window; Shift+Tab comes back the
    same way, from a row's first control to the table and from the table to the
    control before it. The arrow keys move between cells. Tab while a cell is being
    edited is the item delegate's: it commits the edit and edits the next cell, as
    in any Qt table.

    The controls in cells are taken off the window's own Tab path when they are set,
    so Tab from elsewhere never lands in the middle of the table; a click, the arrow
    keys and this table's Tab handling still give them focus. A control added to a
    cell widget after ``setCellWidget``, or given Tab focus again, is back on the
    window's path, so build a cell completely before setting it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setTabKeyNavigation(False)
        self._wheel_focus = _WheelFocus(self)
        self._moving_by_key = False

    def setCellWidget(self, row: int, column: int, widget: QWidget) -> None:  # noqa: N802 - Qt API.
        _take_off_tab_path(widget, self._wheel_focus)
        # ``cellWidget`` also returns an item editor that is open; only these count.
        widget.setProperty(_CELL_WIDGET_PROPERTY, True)
        super().setCellWidget(row, column, widget)

    def row_tab_controls(self, row: int) -> tuple[QWidget, ...]:
        """The controls in ``row``'s cells in the order Tab visits them, reachable or not.

        A cell widget that Tab reached when it was set is one control. One that did not,
        a container, stands for the controls in it; a ``RowControlsCell`` lists its own.
        """

        header = self.horizontalHeader()
        controls: list[QWidget] = []
        for visual in range(self.columnCount()):
            column = header.logicalIndex(visual)
            if self.isColumnHidden(column):
                continue
            cell = self._set_cell_widget(row, column)
            if cell is not None:
                controls.extend(_focusable_controls(cell))
        return tuple(controls)

    def focusNextPrevChild(self, next: bool) -> bool:  # noqa: A002, N802 - Qt API naming.
        focused = QApplication.focusWidget()
        if not self.isVisible() or focused is None:
            return super().focusNextPrevChild(next)
        if focused is self:
            if next:
                controls = reachable_controls(self.row_tab_controls(self.currentRow()), self)
                if controls:
                    self.scrollTo(self.currentIndex())
                    controls[0].setFocus(Qt.FocusReason.TabFocusReason)
                    return True
            return super().focusNextPrevChild(next)
        located = self._locate_control(focused)
        if located is None:
            if self.isAncestorOf(focused):
                # Not a row control now: a cell widget leaving with its row, or one in
                # a hidden column. The table takes focus.
                self.setFocus(Qt.FocusReason.OtherFocusReason)
                return True
            return super().focusNextPrevChild(next)
        row, column, control = located
        controls = self.row_tab_controls(row)
        neighbour = next_reachable_control(controls, control, next, self)
        if control not in reachable_controls(controls, self):
            # Qt is moving focus off a control just disabled or hidden, not the user:
            # to another control in the row, or to the table when none can take it.
            neighbour = neighbour or next_reachable_control(controls, control, not next, self)
            if neighbour is None:
                self.setFocus(Qt.FocusReason.OtherFocusReason)
                return True
        if neighbour is not None:
            neighbour.setFocus(
                Qt.FocusReason.TabFocusReason if next else Qt.FocusReason.BacktabFocusReason
            )
            return True
        return self._tab_out_of_row(row, column, next)

    def edit(self, *args) -> bool:
        # Qt edits the current cell when it changes and when its delayed edit after a
        # click on a selected cell runs out, and for a cell widget editing means taking
        # focus. Only a key moving the current cell may move focus into a cell widget:
        # the delayed edit would otherwise pull focus back into the row's last control
        # after Tab has left it.
        if len(args) == 3 and not self._moving_by_key:
            index = args[0]
            if self._set_cell_widget(index.row(), index.column()) is not None:
                return False
        return super().edit(*args)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API naming.
        self._moving_by_key = True
        try:
            super().keyPressEvent(event)
        finally:
            self._moving_by_key = False

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API naming.
        index = self.indexAt(event.position().toPoint())
        super().mousePressEvent(event)
        # The delegate takes a click on a check box without moving the current cell;
        # the clicked row is still the one Tab goes through next.
        if index.isValid() and index != self.currentIndex() and self.isEnabled():
            self.set_current_cell_keeping_selection(index.row(), index.column())

    def inputMethodEvent(self, event: QInputMethodEvent) -> None:  # noqa: N802 - Qt API.
        # A line edit in a cell that loses focus to the table sends it an empty input
        # method event a moment later. Qt's view takes any input method event for
        # typing and edits the current cell, which moves focus back into that cell.
        if not event.commitString() and not event.preeditString():
            event.ignore()
            return
        super().inputMethodEvent(event)

    def _tab_out_of_row(self, row: int, column: int, forward: bool) -> bool:
        """Leave a row's controls: Shift+Tab to the table, Tab on past it."""

        # A click focuses a control without moving the view's current cell.
        self.set_current_cell_keeping_selection(row, column)
        if forward:
            # The window's Tab path runs on from the widget that has focus.
            self.setFocus(Qt.FocusReason.TabFocusReason)
            return super().focusNextPrevChild(True)
        self.setFocus(Qt.FocusReason.BacktabFocusReason)
        return True

    def set_current_cell_keeping_selection(self, row: int, column: int) -> None:
        self.selectionModel().setCurrentIndex(
            self.model().index(row, column), QItemSelectionModel.SelectionFlag.NoUpdate
        )

    def _locate_control(self, focused: QWidget) -> tuple[int, int, QWidget] | None:
        """The row, column and row control holding ``focused``, if a cell widget does."""

        for row in range(self.rowCount()):
            for column in range(self.columnCount()):
                cell = self._set_cell_widget(row, column)
                if cell is None or not (cell is focused or cell.isAncestorOf(focused)):
                    continue
                if self.isColumnHidden(column):
                    return None
                for control in _focusable_controls(cell):
                    if control is focused or control.isAncestorOf(focused):
                        return row, column, control
                return None
        return None

    def _set_cell_widget(self, row: int, column: int) -> QWidget | None:
        cell = self.cellWidget(row, column)
        return cell if cell is not None and cell.property(_CELL_WIDGET_PROPERTY) else None


_CELL_WIDGET_PROPERTY = "tabStopTableCell"
_ROW_CONTROL_PROPERTY = "tabStopTableRowControl"


def reachable_controls(controls: Sequence[QWidget], within: QWidget) -> tuple[QWidget, ...]:
    """The ``controls`` that can take focus now: enabled and not hidden inside ``within``."""

    return tuple(
        control for control in controls if control.isEnabled() and control.isVisibleTo(within)
    )


def next_reachable_control(
    controls: Sequence[QWidget], control: QWidget, forward: bool, within: QWidget
) -> QWidget | None:
    """The first control after ``control`` in ``controls``, or before it, that can take focus."""

    position = list(controls).index(control)
    candidates = controls[position + 1 :] if forward else tuple(reversed(controls[:position]))
    reachable = reachable_controls(controls, within)
    return next((other for other in candidates if other in reachable), None)


def _focusable_controls(widget: QWidget) -> tuple[QWidget, ...]:
    row_controls = getattr(widget, "row_controls", None)
    if row_controls is not None:
        return tuple(row_controls)
    if widget.property(_ROW_CONTROL_PROPERTY):
        return (widget,)
    controls: list[QWidget] = []
    for child in widget.children():
        if isinstance(child, QWidget) and not child.isWindow():
            controls.extend(_focusable_controls(child))
    return tuple(controls)

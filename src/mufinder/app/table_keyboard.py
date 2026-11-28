"""The keyboard path: through a window panel by panel, and through a table as one stop.

Every table in ``mufinder.app`` is a ``TabStopTable``. A plain ``QTableWidget`` walks
its cells on Tab and wraps round from the last cell to the first, so Tab never leaves
it and the controls after it cannot be reached from the keyboard.

Qt's Tab order is the order controls were created in. ``set_screen_tab_order`` sets
the order from where the controls are on screen instead. Every MuFinder dialog gets
that order from ``install_screen_tab_order``, so no dialog sets its own; a widget
marked with ``keep_own_tab_order`` keeps the order it has.
"""

from __future__ import annotations

from collections.abc import Sequence
import itertools

import shiboken6
from PySide6.QtCore import QEvent, QItemSelectionModel, QObject, QRect, Qt
from PySide6.QtGui import QInputMethodEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLayout,
    QScrollBar,
    QTableWidget,
    QWidget,
)

# Two panels may overlap by this many pixels and still read as one after the other.
_OVERLAP_TOLERANCE = 3
# Controls Tab takes whole, whatever widgets they are made of.
_WHOLE_CONTROLS = (QAbstractSpinBox, QComboBox, QAbstractItemView)
# How far back from a control to look for the one before it in the focus chain.
_CHAIN_STEPS = 200


def is_tab_stop(widget: QWidget) -> bool:
    """Whether Tab stops on ``widget`` itself, taken whole without its inner widgets.

    A button counts whatever its Tab focus is now: Qt takes Tab focus from the
    unchosen options of an exclusive group and gives it back once one is chosen.
    """

    policy = widget.focusPolicy()
    return bool(policy.value & Qt.FocusPolicy.TabFocus.value) or (
        isinstance(widget, QAbstractButton) and policy != Qt.FocusPolicy.NoFocus
    )


def screen_tab_order(root: QWidget) -> list[QWidget]:
    """Return the Tab stops in ``root`` in the order they read on screen, panel by panel.

    The widget tree is read container by container: a container's children are
    put in reading order by where they are, top to bottom and then left to right
    within a band, and each child's own stops follow in their turn. So a panel, a
    form column, a viewer's toolbar, a splitter pane or a scroll area, is read whole
    before the panel beside it. A control that takes Tab is one stop, whole: a spin
    box or combo box with its parts, a table with its cell controls, which the table
    leads Tab through. A stop that holds stops of its own, such as a scroll area or a
    checkable group box, comes before them; a tab widget, like any control whose
    focus goes to a part of it, is one stop. A ``QDialogButtonBox`` keeps its own order, the order
    of its layout, and a panel marked with :func:`keep_own_tab_order` the order its
    focus chain has. Hidden widgets are left out.
    """

    return _stops_in(root)


def set_screen_tab_order(root: QWidget, *, followed_by: QWidget | None = None) -> None:
    """Chain Tab through ``root`` in :func:`screen_tab_order`.

    A window keeps the control it opens on: the chain is turned round to start at
    the stop Tab reached first before. A part of a window goes just before
    ``followed_by``, the control outside it that Tab reaches next.
    """

    order = _stops_in(root)
    if not order:
        return
    if followed_by is not None:
        previous = _focusable_before(followed_by, root)
        # With nothing found before it, the part is at least led into ``followed_by``.
        order = [*order, followed_by] if previous is None else [previous, *order]
    else:
        start = order.index(_first_in_focus_chain(root.window(), order))
        order = order[start:] + order[:start]
    for earlier, later in itertools.pairwise(order):
        QWidget.setTabOrder(earlier, later)


KEEP_OWN_TAB_ORDER = "keepOwnTabOrder"


def keep_own_tab_order(widget: QWidget, reason: str) -> None:
    """Leave ``widget``'s Tab order as it is: a panel's stops, or a whole dialog's.

    ``reason`` says why; the keyboard baseline test reads the same mark.
    """

    widget.setProperty(KEEP_OWN_TAB_ORDER, reason)


def install_screen_tab_order(app: QApplication) -> None:
    """Chain Tab through every dialog class defined in MuFinder in screen order.

    A dialog is ordered when it first becomes active, when it is shown again, and
    when its layouts change while it is shown, so a control it reveals takes its
    place. Nothing is installed on the application itself. Qt's own dialogs, a plain
    ``QDialog`` and a dialog marked with :func:`keep_own_tab_order` are left alone.
    """

    if isinstance(app, QApplication) and getattr(app, "_mufinder_screen_tab_order", None) is None:
        hook = _ScreenTabOrderHook(app)
        app.focusWindowChanged.connect(hook.order_active_dialog)
        app.focusChanged.connect(hook.follow_focus)
        app._mufinder_screen_tab_order = hook


def _ordered_by_hook(window: QWidget | None) -> bool:
    return (
        isinstance(window, QDialog)
        and type(window).__module__.startswith("mufinder.")
        and not window.property(KEEP_OWN_TAB_ORDER)
    )


def _key(widget: QWidget) -> int:
    return shiboken6.getCppPointer(widget)[0]


class _ScreenTabOrderHook(QObject):
    """Watches each shown dialog's layout tops (:func:`_layout_tops`), which receive
    its ``LayoutRequest`` events, and orders the dialog when the first arrives. The
    requests still queued then, its own among them, are delivered at once and start
    no second run. A hidden dialog keeps only the watch on itself, for its next show.
    """

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        self._watched: dict[int, list[QWidget]] = {}  # shown dialogs: their layout tops
        self._hidden: set[int] = set()
        self._ordering = False

    def order_active_dialog(self, _window: object = None) -> None:
        dialog = QApplication.activeWindow()
        key = _key(dialog) if dialog is not None else None
        if _ordered_by_hook(dialog) and key not in self._watched and key not in self._hidden:
            dialog.destroyed.connect(lambda _=None, key=key: self._forget(key))
            self._watched[key] = []
            self._order(dialog)
            self.follow_focus(None, QApplication.focusWidget())

    def follow_focus(self, old: QWidget | None, now: QWidget | None) -> None:
        # Watch the focused control's keys, so Tab never runs on an order not yet set.
        if old is not None and shiboken6.isValid(old) and not old.isWindow():
            if not any(old is top for top in self._watched.get(_key(old.window()), [])):
                old.removeEventFilter(self)
        if now is not None and _key(now.window()) in self._watched:
            now.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        kind = event.type()
        if kind == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            # A change not yet ordered orders the dialog now, before Tab moves.
            for top in self._live_tops(_key(watched.window())):
                QApplication.sendPostedEvents(top, QEvent.Type.LayoutRequest.value)
        elif kind == QEvent.Type.LayoutRequest and not self._ordering:
            dialog = watched.window()
            if _key(dialog) in self._watched and dialog.isVisible():
                self._order(dialog)
        elif kind == QEvent.Type.Hide and watched.isWindow():
            self._unwatch(watched)
        elif kind == QEvent.Type.Show and _key(watched) in self._hidden:
            self._hidden.discard(_key(watched))
            self._watched[_key(watched)] = [watched]
            self._order(watched)
        return False

    def _order(self, dialog: QWidget) -> None:
        self._ordering = True
        try:
            set_screen_tab_order(dialog)
            key = _key(dialog)
            watched = self._live_tops(key)
            for top in _layout_tops(dialog):
                if top not in watched:
                    top.installEventFilter(self)
                    if top is not dialog:
                        top.destroyed.connect(lambda _=None, key=key: self._top_gone(key))
                    watched.append(top)
            for top in watched:
                QApplication.sendPostedEvents(top, QEvent.Type.LayoutRequest.value)
        finally:
            self._ordering = False

    def _top_gone(self, key: int) -> None:
        # A watched widget was deleted, as a scroll area's replaced content is.
        watched = self._watched.get(key)
        if watched and shiboken6.isValid(watched[0]):
            QApplication.postEvent(watched[0], QEvent(QEvent.Type.LayoutRequest))

    def _live_tops(self, key: int) -> list[QWidget]:
        watched = self._watched.get(key)
        if watched is None:
            return []
        watched[:] = [top for top in watched if shiboken6.isValid(top)]
        return watched

    def _unwatch(self, dialog: QWidget) -> None:
        self._hidden.add(_key(dialog))
        for top in self._watched.pop(_key(dialog), [])[1:]:
            if shiboken6.isValid(top):
                top.removeEventFilter(self)

    def _forget(self, key: int) -> None:
        self._watched.pop(key, None)
        self._hidden.discard(key)


def _layout_tops(dialog: QWidget) -> list[QWidget]:
    """``dialog`` and each widget in it with a layout that no layout places.

    A control Tab takes whole (an item view, a tab widget) is not entered: the
    order never reaches inside it, so a change there needs no new order.
    """

    tops = [dialog]
    pending = [dialog]
    while pending:
        parent = pending.pop()
        placed = _placed_by(parent.layout())
        for child in parent.children():
            if not isinstance(child, QWidget) or child.isWindow():
                continue
            if child.layout() is not None and _key(child) not in placed:
                tops.append(child)
            if not _taken_whole(child):
                pending.append(child)
    return tops


def _taken_whole(widget: QWidget) -> bool:
    """Whether :func:`screen_tab_order` takes ``widget`` as one stop, or none, unread."""

    return isinstance(widget, _WHOLE_CONTROLS) or (
        is_tab_stop(widget) and widget.focusProxy() is not None
    )


def _keeps_a_place(widget: QWidget) -> bool:
    """A scroll area keeps its place while Tab passes over it: a reading one becomes
    a stop when its content overflows, with no layout change to reorder on."""

    return isinstance(widget, QAbstractScrollArea) and widget.focusPolicy() != (
        Qt.FocusPolicy.NoFocus
    )


def _placed_by(layout: QLayout | None) -> set[int]:
    """The widgets ``layout`` and the layouts in it place."""

    placed: set[int] = set()
    layouts = [layout] if layout is not None else []
    while layouts:
        layout = layouts.pop()
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item.widget() is not None:
                placed.add(_key(item.widget()))
            elif item.layout() is not None:
                layouts.append(item.layout())
    return placed


def _own_order(panel: QWidget) -> list[QWidget]:
    """``panel``'s stops in the order its focus chain has them now.

    A stop the walk does not find comes after the others, in screen order.
    """

    stops = _stops_in(panel)
    ordered: list[QWidget] = []
    # Other widgets can sit between the panel's own in the chain: the walk goes on
    # until every stop is found, or ``_CHAIN_STEPS`` widgets pass without one.
    widget, passed = panel, 0
    while len(ordered) < len(stops) and passed < _CHAIN_STEPS:
        widget = widget.nextInFocusChain()
        if widget is panel:
            break
        if widget in stops and widget not in ordered:
            ordered.append(widget)
            passed = 0
        else:
            passed += 1
    lead = [panel] if is_tab_stop(panel) else []
    return [*lead, *ordered, *(stop for stop in stops if stop not in ordered)]


def _stops_in(container: QWidget) -> list[QWidget]:
    if container.layout() is not None:
        # Controls just added or shown have their places only once the layout has run.
        container.layout().activate()
    panels: list[tuple[QRect, list[QWidget]]] = []
    for child in container.children():
        if not isinstance(child, QWidget) or child.isWindow() or child.isHidden():
            continue
        if isinstance(child, QScrollBar):
            continue
        if child.property(KEEP_OWN_TAB_ORDER):
            stops = _own_order(child)
        elif _taken_whole(child):
            # Qt leads Tab through it, and through a tab widget's pages, itself.
            stops = [child] if is_tab_stop(child) else []
        elif is_tab_stop(child) or _keeps_a_place(child):
            stops = [child, *_stops_in(child)]
        else:
            stops = _stops_in(child)
        if stops:
            panels.append((child.geometry(), stops))
    if isinstance(container, QDialogButtonBox):
        layout = container.layout()
        laid_out = [layout.itemAt(index).widget() for index in range(layout.count())]
        panels = [panel for widget in laid_out for panel in panels if panel[1][0] is widget]
        return [stop for _rect, stops in panels for stop in stops]
    return [stop for _rect, stops in _reading_order(panels) for stop in stops]


def _reading_order(panels: list[tuple[QRect, list[QWidget]]]) -> list:
    """Cut ``panels`` into bands top to bottom, each band into columns, and recurse."""

    if len(panels) <= 1:
        return panels
    for start, end in (
        (lambda rect: rect.top(), lambda rect: rect.bottom() + 1),
        (lambda rect: rect.left(), lambda rect: rect.right() + 1),
    ):
        bands = _bands(panels, start, end)
        if len(bands) > 1:
            return [panel for band in bands for panel in _reading_order(band)]
    # Overlapping panels cannot be cut apart: read them by their top left corners.
    return sorted(panels, key=lambda panel: (panel[0].top(), panel[0].left()))


def _bands(panels: list, start, end) -> list[list]:
    """Group ``panels`` into bands along one axis, cut where no panel spans the gap."""

    panels = sorted(panels, key=lambda panel: (start(panel[0]), end(panel[0])))
    bands = [[panels[0]]]
    reach = end(panels[0][0])
    for panel in panels[1:]:
        if start(panel[0]) >= reach - _OVERLAP_TOLERANCE:
            bands.append([panel])
            reach = end(panel[0])
        else:
            bands[-1].append(panel)
            reach = max(reach, end(panel[0]))
    return bands


def _first_in_focus_chain(window: QWidget, stops: list[QWidget]) -> QWidget:
    """The one of ``stops`` that Tab reaches first from ``window``.

    It walks only as far as that stop, and never more than ``_CHAIN_STEPS``:
    walking the whole chain of a main window that holds 3D viewers has crashed
    the application.
    """

    first = None
    widget = window.nextInFocusChain()
    for _ in range(_CHAIN_STEPS):
        if widget is window:
            break
        if widget in stops:
            # Qt opens a window on the first control that can take Tab focus now.
            if widget.isEnabled() and widget.focusPolicy().value & Qt.FocusPolicy.TabFocus.value:
                return widget
            first = first or widget
        widget = widget.nextInFocusChain()
    return first or stops[0]


def _focusable_before(widget: QWidget, root: QWidget) -> QWidget | None:
    """The nearest widget before ``widget`` in the focus chain that is outside ``root``.

    Only a widget that can take focus can be put before another in the chain.
    """

    previous = widget.previousInFocusChain()
    for _ in range(_CHAIN_STEPS):
        if previous is widget:
            break
        if previous.focusPolicy() != Qt.FocusPolicy.NoFocus and not (
            previous is root or root.isAncestorOf(previous)
        ):
            return previous
        previous = previous.previousInFocusChain()
    return None


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

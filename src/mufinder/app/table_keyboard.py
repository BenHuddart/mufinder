"""The keyboard path: through a window panel by panel, and through a table as one stop.

Every table in ``mufinder.app`` is a ``TabStopTable``. A plain ``QTableWidget`` walks
its cells on Tab and wraps round from the last cell to the first, so Tab never leaves
it and the controls after it cannot be reached from the keyboard.

Qt's Tab order is the order controls were created in. ``set_screen_tab_order`` sets it
from the layouts, and ``install_screen_tab_order`` applies it to every MuFinder dialog.
``screen_tab_order`` reads where the controls are on screen: the reference the
keyboard baseline test checks the layout order against. ``keep_tab_on_chosen_option``
makes an exclusive group's chosen option its one stop.
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
    QBoxLayout,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QLayout,
    QScrollBar,
    QSplitter,
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


def keep_tab_on_chosen_option(options: QButtonGroup | Sequence[QAbstractButton]) -> None:
    """Make the chosen option the one Tab stop of an exclusive group, from the first Tab.

    Qt keeps Tab on one option of an exclusive group only once one of them has had
    focus; until then the first Tab into the group stops on its first option, chosen
    or not. ``options`` is a ``QButtonGroup`` with all its buttons added, or radio
    buttons that exclude each other as siblings; they are removed together. The stop
    follows the choice, by click, key or code, and comes back to it when focus leaves
    an option that was not chosen; while the chosen option is disabled or hidden, the
    first option that is neither takes it. With nothing chosen the Tab stops are left
    as Qt has them.
    """

    _ChosenOptionStop(options)


class _ChosenOptionStop(QObject):
    """Puts a group's Tab stop on its chosen option whenever its options change."""

    _EVENTS = frozenset(
        {QEvent.Type.EnabledChange, QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.FocusOut}
    )

    def __init__(self, options: QButtonGroup | Sequence[QAbstractButton]) -> None:
        grouped = isinstance(options, QButtonGroup)
        self._options = list(options.buttons() if grouped else options)
        if not self._options:
            raise ValueError("keep_tab_on_chosen_option needs the group's options added first")
        # It lives as long as the options, made with or without a parent yet.
        super().__init__(self._options[0])
        for option in self._options:
            option.toggled.connect(self.place_stop)
            option.installEventFilter(self)
        self.place_stop()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        if event.type() in self._EVENTS:
            self.place_stop()
        return False

    def place_stop(self) -> None:
        options = [option for option in self._options if shiboken6.isValid(option)]
        stop = next((option for option in options if option.isChecked()), None)
        if stop is None:
            return
        shown = [option for option in options if option.isEnabled() and option.isVisible()]
        if shown and not any(option is stop for option in shown):
            stop = shown[0]
        for option in options:
            policy = option.focusPolicy()
            option.setFocusPolicy(
                policy | Qt.FocusPolicy.TabFocus
                if option is stop
                else policy & ~Qt.FocusPolicy.TabFocus
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
    of its layout. Hidden widgets are left out. It reads where controls are, not
    their layouts: the reference the layout order is tested against.
    """

    return _stops_in(root, by_layout=False)


def set_screen_tab_order(root: QWidget, *, followed_by: QWidget | None = None) -> None:
    """Chain Tab through ``root`` in the order its layouts place its controls.

    Hidden controls are placed too: a box by its items, as drawn; a grid by row then
    column; a form by row; a splitter by its panes; a nested layout in its place;
    children no layout places by where they are. A control created later needs this
    to run again. A window keeps the control it opens on; a part goes just before
    ``followed_by``.
    """

    order = [] if root.property(KEEP_OWN_TAB_ORDER) else _stops_in(root, by_layout=True)
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
    """Leave ``widget``'s Tab order as it is: a panel's stops, or a whole dialog's."""

    if not reason:
        raise ValueError("keep_own_tab_order needs the reason the order is kept")
    widget.setProperty(KEEP_OWN_TAB_ORDER, reason)


def install_screen_tab_order(app: QApplication) -> None:
    """Order each MuFinder dialog class's dialog as it becomes the active window.

    It is one signal connection, no event filter; a marked dialog is left alone.
    """

    if isinstance(app, QApplication):
        app.focusWindowChanged.connect(_order_active_dialog)


def _order_active_dialog(window: object = None) -> None:
    dialog = QApplication.activeWindow()
    if (
        isinstance(dialog, QDialog)
        and window is dialog.windowHandle()  # not a popup, such as a combo box's list
        and type(dialog).__module__.startswith("mufinder.")
        and not dialog.property(KEEP_OWN_TAB_ORDER)
    ):
        set_screen_tab_order(dialog)


def _children_in_order(container: QWidget, by_layout: bool) -> tuple[list, list]:
    """The children ``container``'s layout places, hidden ones too when ``by_layout``
    (otherwise only a button box's), and the shown rest."""

    children = [
        child
        for child in container.children()
        if isinstance(child, QWidget) and not child.isWindow() and not isinstance(child, QScrollBar)
    ]
    placed: list[QWidget] = []
    if by_layout and isinstance(container, QSplitter):
        placed = [container.widget(index) for index in range(container.count())]
    elif container.layout() is not None and (by_layout or isinstance(container, QDialogButtonBox)):
        own = {id(child) for child in children}
        placed = [widget for widget in _laid_out(container.layout()) if id(widget) in own]
    if not by_layout:
        placed = [child for child in placed if not child.isHidden()]
    taken = {id(child) for child in placed}
    rest = [child for child in children if id(child) not in taken and not child.isHidden()]
    return placed, rest


def _laid_out(layout: QLayout) -> list[QWidget]:
    """The widgets ``layout`` places, in its order, the layouts in it read in their place."""

    # A widget laid out right to left mirrors its layouts, but not a button box's
    # platform order; the screen reads left to right.
    owner = layout.parentWidget()
    across = -1 if owner is not None and owner.isRightToLeft() else 1
    if isinstance(owner, QDialogButtonBox):
        across = 1
    if isinstance(layout, (QGridLayout, QFormLayout)):
        # A grid by (row, column), a form by (row, side): label side first.
        place = [layout.getItemPosition(index)[:2] for index in range(layout.count())]
        indexes = sorted(
            range(layout.count()),
            key=lambda i: (place[i][0], across * getattr(place[i][1], "value", place[i][1])),
        )
    else:
        indexes = list(range(layout.count()))
        backwards = {QBoxLayout.Direction.RightToLeft, QBoxLayout.Direction.BottomToTop}
        if isinstance(layout, QBoxLayout) and (layout.direction() in backwards) != (
            across < 0 and layout.direction() in _HORIZONTAL
        ):
            indexes.reverse()
    widgets: list[QWidget] = []
    for index in indexes:
        item = layout.itemAt(index)
        if item.widget() is not None:
            widgets.append(item.widget())
        elif item.layout() is not None:
            widgets.extend(_laid_out(item.layout()))
    return widgets


_HORIZONTAL = {QBoxLayout.Direction.LeftToRight, QBoxLayout.Direction.RightToLeft}


def _stops_in(container: QWidget, *, by_layout: bool) -> list[QWidget]:
    if container.layout() is not None:
        # Controls just added or shown have their places only once the layout has run.
        container.layout().activate()
    placed, rest = _children_in_order(container, by_layout)
    entries = [(child, _child_stops(child, by_layout)) for child in placed]
    unplaced = [(child.geometry(), _child_stops(child, by_layout)) for child in rest]
    for rect, stops in _reading_order([panel for panel in unplaced if panel[1]]):
        # It goes before the first shown laid-out child that reads after it on screen.
        shown = [
            (child.geometry(), [index])
            for index, (child, _stops) in enumerate(entries)
            if child is not None and not child.isHidden()
        ]
        reading = [marks for _rect, marks in _reading_order([*shown, (rect, [None])])]
        after = reading[reading.index([None]) + 1 :]
        entries.insert(after[0][0] if after else len(entries), (None, stops))
    return [stop for _child, stops in entries for stop in stops]


def _child_stops(child: QWidget, by_layout: bool) -> list[QWidget]:
    if child.property(KEEP_OWN_TAB_ORDER):
        return _own_order(child, by_layout)
    if isinstance(child, _WHOLE_CONTROLS) or (
        is_tab_stop(child) and child.focusProxy() is not None
    ):
        # Qt leads Tab through it, and through a tab widget's pages, itself.
        return [child] if is_tab_stop(child) else []
    if is_tab_stop(child) or (
        by_layout
        and isinstance(child, QAbstractScrollArea)
        and child.focusPolicy() != Qt.FocusPolicy.NoFocus
    ):
        # A reading scroll area is a stop only while its content overflows.
        return [child, *_stops_in(child, by_layout=by_layout)]
    return _stops_in(child, by_layout=by_layout)


def _own_order(panel: QWidget, by_layout: bool) -> list[QWidget]:
    """``panel``'s stops in the order its focus chain has them now, found within
    ``_CHAIN_STEPS`` widgets of each other; one not found comes after the others."""

    stops = _stops_in(panel, by_layout=by_layout)
    ordered: list[QWidget] = []
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
    walking the whole chain of a main window that holds 3D viewers has crashed the
    application. With none found by then, the first stop on screen opens it.
    """

    def opens_on(widget: QWidget) -> bool:
        # Qt opens a window on the first control that can take Tab focus now.
        return (
            widget.isEnabled()
            and widget.isVisibleTo(window)
            and bool(widget.focusPolicy().value & Qt.FocusPolicy.TabFocus.value)
        )

    first = None
    widget = window.nextInFocusChain()
    for _ in range(_CHAIN_STEPS):
        if widget is window:
            break
        if widget in stops:
            if opens_on(widget):
                return widget
            first = first or widget
        widget = widget.nextInFocusChain()
    return next((stop for stop in stops if opens_on(stop)), first or stops[0])


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

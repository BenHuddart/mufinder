"""Walk a window with real Tab presses and check it against the keyboard baseline.

The baseline (``docs/architecture.rst``, Keyboard baseline) every dialog and page
meets: Tab and Shift+Tab reach every enabled, shown control; Tab follows the
screen panel by panel, as ``screen_tab_order`` reads it; Enter presses the
default button and Escape cancels.
"""

from __future__ import annotations

from dataclasses import dataclass
import types

import shiboken6
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QHeaderView,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollBar,
    QTabBar,
    QTextEdit,
    QWidget,
)

from mufinder.app.table_keyboard import TabStopTable, screen_tab_order
from tests.support.qt import open_and_take_focus, press_key

#: Presses after which a walk that has not come back round is reported as a trap.
WALK_LIMIT = 300

#: 3D viewer canvases take focus by a click, never by Tab.
_VIEWER_CANVASES = {
    "FakePlotter",
    "FakePyVistaCanvas",
    "FakePyVistaTriadCanvas",
    "QtInteractor",
    "QVTKRenderWindowInteractor",
}

_CONTROLS = (
    QAbstractButton,
    QLineEdit,
    QAbstractSpinBox,
    QComboBox,
    QAbstractItemView,
    QAbstractSlider,
    QTabBar,
    QTextEdit,
    QPlainTextEdit,
)


def address(widget: QWidget) -> int:
    """The C++ object's address: wrapper ids can be reused once a wrapper is collected."""

    return shiboken6.getCppPointer(widget)[0]


def describe(widget: QWidget | None) -> str:
    """A short name for ``widget`` in a failure message."""

    if widget is None or not shiboken6.isValid(widget):
        return repr(widget)
    text = widget.text().replace("&", "") if isinstance(widget, QAbstractButton) else ""
    label = widget.accessibleName() or text or widget.objectName()
    return f"{type(widget).__name__}({label!r})" if label else type(widget).__name__


@dataclass
class Walk:
    """The stops one walk round a window visited, and why it stopped if not back at its start."""

    stops: list[QWidget]
    fault: str = ""


def tab_walk(window: QWidget, *, backward: bool = False) -> Walk:
    """Press Tab, or Shift+Tab, until focus comes back to the first stop."""

    modifiers = Qt.KeyboardModifier.ShiftModifier if backward else Qt.KeyboardModifier.NoModifier
    stops: list[QWidget] = []
    seen: set[int] = set()
    start = QApplication.focusWidget()
    if start is None:
        # A window with no control that takes Tab has nothing to walk.
        return Walk(stops)
    if start is not window and not window.isAncestorOf(start):
        return Walk(stops, f"the walk starts outside the window, on {describe(start)}")
    for _ in range(WALK_LIMIT):
        if QApplication.focusWidget() is None:
            return Walk(stops, "focus went to nothing")
        widget = press_key(Qt.Key.Key_Tab, modifiers)
        if widget is None:
            return Walk(stops, "focus went to nothing")
        if widget is not window and not window.isAncestorOf(widget):
            return Walk(stops, f"focus left the window for {describe(widget)}")
        if stops and address(widget) == address(stops[0]):
            return Walk(stops)
        if address(widget) in seen:
            return Walk(stops, f"trapped: the walk came back to {describe(widget)}")
        seen.add(address(widget))
        stops.append(widget)
    return Walk(stops, f"focus never came back within {WALK_LIMIT} presses")


def cell_table(widget: QWidget) -> TabStopTable | None:
    """The table whose cell holds ``widget``, if one does."""

    parent = widget.parentWidget()
    while parent is not None and not isinstance(parent, TabStopTable):
        parent = parent.parentWidget()
    if parent is None or not parent.viewport().isAncestorOf(widget):
        return None
    return parent


def _cell_row(table: TabStopTable, widget: QWidget) -> int | None:
    for row in range(table.rowCount()):
        for column in range(table.columnCount()):
            cell = table.cellWidget(row, column)
            if cell is not None and (cell is widget or cell.isAncestorOf(widget)):
                return row
    return None


def _exclusive_group(button: QAbstractButton) -> list[QAbstractButton]:
    """The shown, enabled members of ``button``'s exclusive group, when it has partners."""

    if not button.isCheckable():
        return []
    group = button.group()
    if group is not None and group.exclusive():
        members = list(group.buttons())
    elif group is None and button.autoExclusive() and button.parentWidget() is not None:
        members = [
            other
            for other in button.parentWidget().findChildren(
                QAbstractButton, options=Qt.FindChildOption.FindDirectChildrenOnly
            )
            if other.autoExclusive() and other.isCheckable() and other.group() is None
        ]
    else:
        return []
    members = [member for member in members if member.isVisible() and member.isEnabled()]
    return members if len(members) > 1 else []


def _ancestors(widget: QWidget):
    while widget is not None:
        yield widget
        widget = widget.parentWidget()


def controls_to_reach(root: QWidget) -> list[list[QWidget]]:
    """The shown, enabled controls in ``root`` that Tab must reach, each with its stand-ins.

    An exclusive group is one entry: reaching any member reaches it. Not required:
    viewer canvases (click focus), read-only text, scroll bars, headers, the parts
    of a spin box or combo box, and the cell controls of a table's rows other than
    the current one.
    """

    required: list[list[QWidget]] = []
    groups: set[int] = set()
    for widget in root.findChildren(QWidget):
        if not isinstance(widget, _CONTROLS) or widget.isWindow():
            continue
        if not widget.isVisible() or not widget.isEnabled():
            continue
        if isinstance(widget, (QScrollBar, QHeaderView)):
            continue
        parent = widget.parentWidget()
        if isinstance(parent, (QComboBox, QAbstractSpinBox, QAbstractItemView, QTabBar)):
            continue
        if any(isinstance(outer, QComboBox) for outer in _ancestors(parent)):
            continue
        if any(type(outer).__name__ in _VIEWER_CANVASES for outer in _ancestors(widget)):
            continue
        if isinstance(widget, (QTextEdit, QPlainTextEdit)) and widget.isReadOnly():
            continue
        table = cell_table(widget)
        if table is not None and _cell_row(table, widget) != table.currentRow():
            continue
        group = _exclusive_group(widget) if isinstance(widget, QAbstractButton) else []
        if group:
            key = min(address(member) for member in group)
            if key not in groups:
                groups.add(key)
                required.append(group)
            continue
        required.append([widget])
    return required


def unreached(required: list[list[QWidget]], stops: list[QWidget]) -> list[QWidget]:
    """The required controls no stop is, or is part of."""

    keys = {address(stop) for stop in stops}
    missing = []
    for entry in required:
        reached = any(
            address(member) in keys
            or any(member.isAncestorOf(stop) for stop in stops)
            or (member.focusProxy() is not None and address(member.focusProxy()) in keys)
            for member in entry
        )
        if not reached:
            missing.append(entry[0])
    return missing


def screen_order_faults(stops: list[QWidget], root: QWidget, *, cycle: bool = True) -> list[str]:
    """Where the Tab walk ``stops`` leaves the screen's reading order of ``root``.

    Each stop counts as the whole control it is part of, and a table's cell
    controls as the table. A ``cycle`` walk may start anywhere on the way round;
    otherwise it must start at the first control on screen.
    """

    reading = {address(widget): index for index, widget in enumerate(screen_tab_order(root))}
    ranks: list[tuple[int, QWidget]] = []
    faults = []
    for stop in stops:
        rank = next((reading[address(w)] for w in _ancestors(stop) if address(w) in reading), None)
        if rank is None:
            faults.append(f"{describe(stop)} is not in the screen reading")
        elif not ranks or ranks[-1][0] != rank:
            ranks.append((rank, stop))
    if cycle and ranks:
        start = min(range(len(ranks)), key=lambda index: ranks[index][0])
        ranks = ranks[start:] + ranks[:start]
    faults += [
        f"{describe(earlier)} -> {describe(later)}"
        for (rank_a, earlier), (rank_b, later) in zip(ranks, ranks[1:], strict=False)
        if rank_b < rank_a
    ]
    return faults


def option_group_faults(stops: list[QWidget]) -> list[tuple[QWidget, str]]:
    """Where a walk's ``stops`` take an exclusive group other than at its chosen option.

    Each exclusive group a stop belongs to must take exactly one stop, and that stop
    is the chosen option whenever it is shown and enabled.
    """

    faults = []
    seen: set[int] = set()
    for stop in stops:
        group = _exclusive_group(stop) if isinstance(stop, QAbstractButton) else []
        key = min((address(member) for member in group), default=None)
        if key is None or key in seen:
            continue
        seen.add(key)
        taken = [member for member in group if any(member is other for other in stops)]
        chosen = next((member for member in group if member.isChecked()), None)
        if len(taken) != 1:
            names = ", ".join(describe(member) for member in taken)
            faults.append((stop, f"Tab stops on {len(taken)} options of one group: {names}"))
        elif chosen is not None and taken[0] is not chosen:
            faults.append(
                (stop, f"Tab stops on {describe(taken[0])}, not the chosen {describe(chosen)}")
            )
    return faults


def reverse_faults(forward: list[QWidget], backward: list[QWidget]) -> str:
    """Whether Shift+Tab visits the Tab walk in reverse; cell controls are Tab's alone."""

    ahead = [address(stop) for stop in forward if cell_table(stop) is None]
    back = [address(stop) for stop in backward if cell_table(stop) is None]
    if sorted(ahead) != sorted(back):
        return "Shift+Tab stops on other controls than Tab"
    if not ahead:
        return ""
    reverse = list(reversed(ahead))
    start = reverse.index(back[0])
    return "" if reverse[start:] + reverse[:start] == back else "Shift+Tab is not Tab reversed"


@dataclass
class KeyResult:
    """What one key press did in a dialog: the buttons it pressed and how the dialog closed."""

    pressed: list[str]
    closed: str


class DialogKeys:
    """Press Enter or Escape in a dialog and record what it did, without doing the work.

    A pressed button's signals are blocked as it is pressed, so ``clicked`` and the
    button box's ``accepted`` never fire. ``accept``, ``reject`` and ``done`` are
    replaced on the instance, so the dialog stays open and only records the close.
    """

    def __init__(self, dialog: QDialog) -> None:
        self.dialog = dialog
        self._recording = False
        self._pressed: list[QAbstractButton] = []
        self._closed = ""
        for button in dialog.findChildren(QAbstractButton):
            button.pressed.connect(lambda button=button: self._on_pressed(button))
        dialog.accept = types.MethodType(lambda _self: self._close("accept"), dialog)
        dialog.reject = types.MethodType(lambda _self: self._close("reject"), dialog)
        dialog.done = types.MethodType(lambda _self, code: self._close(f"done({code})"), dialog)

    def default_buttons(self) -> list[str]:
        return [
            describe(button)
            for button in self.dialog.findChildren(QPushButton)
            if button.isDefault() and button.isVisible()
        ]

    def press(self, key: Qt.Key, on: QWidget) -> KeyResult:
        self._pressed, self._closed, self._recording = [], "", True
        try:
            on.setFocus(Qt.FocusReason.OtherFocusReason)
            QApplication.processEvents()
            QTest.keyClick(QApplication.focusWidget() or self.dialog, key)
            if any(button.isDown() for button in self._pressed):
                QTest.qWait(150)  # an animated click finishes after 100 ms
        finally:
            self._recording = False
            for button in self._pressed:
                button.setDown(False)
                button.blockSignals(False)
        if not self.dialog.isVisible():
            open_and_take_focus(self.dialog)
        return KeyResult([describe(button) for button in self._pressed], self._closed)

    def _on_pressed(self, button: QAbstractButton) -> None:
        if self._recording:
            button.blockSignals(True)
            self._pressed.append(button)

    def _close(self, how: str) -> None:
        if self._recording:
            self._closed = self._closed or how

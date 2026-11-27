"""Qt event-loop and widget-tree helpers shared by the app tests."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import TypeVar

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QLabel,
    QTableView,
    QWidget,
)

W = TypeVar("W", bound=QWidget)


def release_keyboard_modifiers() -> None:
    """Put the application's keyboard modifiers back to none.

    ``QTest.keyClick`` with a modifier (Shift+Tab, say) sends the modifier's
    release carrying the modifier itself, so ``QApplication.keyboardModifiers()``
    still reports it held afterwards. Anything that reads the modifiers then
    behaves as if the key were down: ``QTableView.selectRow`` extends the
    selection instead of replacing it. One release with no modifiers, sent to a
    widget that is never shown, clears the record without reaching any other
    widget.
    """

    if QApplication.keyboardModifiers() == Qt.KeyboardModifier.NoModifier:
        return
    sink = QWidget()
    QTest.keyRelease(sink, Qt.Key.Key_Shift, Qt.KeyboardModifier.NoModifier)
    sink.deleteLater()


def pump_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> bool:
    """Pump the GUI event loop until ``predicate`` holds or ``timeout`` seconds elapse.

    Work that runs on a worker thread reports back through queued signals, which
    are only delivered while the event loop spins. Returns the final value of
    ``predicate`` so a test can ``assert pump_until(...)``.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        QApplication.processEvents()
        time.sleep(0.005)
    return predicate()


def click_header_section(table: QTableView, column: int) -> None:
    """Click the middle of ``table``'s header over ``column``, as a user does."""

    header = table.horizontalHeader()
    x = header.sectionViewportPosition(column) + header.sectionSize(column) // 2
    QTest.mouseClick(
        header.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        QPoint(x, header.height() // 2),
    )


def click_table_row(table: QTableView, row: int) -> None:
    """Click the first cell of ``row`` in ``table``, as a user does."""

    table.scrollTo(table.model().index(row, 0))
    QTest.mouseClick(
        table.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        table.visualRect(table.model().index(row, 0)).center(),
    )


def find_control(parent: QWidget, name: str, kind: type[W]) -> W:
    """The one ``kind`` control in ``parent`` that a user would know as ``name``.

    A control matches by its accessible name, by its text (a button or check
    box, ignoring ``&`` mnemonics), or by the text of the label it is the buddy
    of (a form-layout row label). Fails unless exactly one control matches, so a
    test reaches a page's controls the way a user or a screen reader does,
    without the page's private attributes.
    """

    matches: list[W] = []
    for widget in parent.findChildren(kind):
        text = widget.text().replace("&", "") if isinstance(widget, QAbstractButton) else None
        if widget.accessibleName() == name or text == name:
            matches.append(widget)
    for label in parent.findChildren(QLabel):
        buddy = label.buddy()
        if label.text() == name and isinstance(buddy, kind) and buddy not in matches:
            matches.append(buddy)
    assert len(matches) == 1, f"{len(matches)} {kind.__name__} controls are named {name!r}"
    return matches[0]


def answer_next_message_box(*, click: str | None = None) -> dict[str, object]:
    """Answer the next modal ``QMessageBox`` and record how it was laid out.

    Polls the event loop for the box, records its button labels and which
    button is the default and the Escape choice, then clicks the button labelled
    ``click`` or, when ``click`` is ``None``, presses Escape. The returned dict
    fills in once the box has been answered: ``buttons``, ``default``,
    ``escape``, ``text``, ``informative`` and ``parent``.
    """

    from PySide6.QtCore import QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox

    seen: dict[str, object] = {}
    timer = QTimer()
    timer.setInterval(10)

    def _poll() -> None:
        box = QApplication.activeModalWidget()
        if not isinstance(box, QMessageBox) or not box.isVisible():
            return
        timer.stop()
        default = box.defaultButton()
        escape = box.escapeButton()
        seen["buttons"] = [button.text() for button in box.buttons()]
        seen["default"] = default.text() if default is not None else None
        seen["escape"] = escape.text() if escape is not None else None
        seen["text"] = box.text()
        seen["informative"] = box.informativeText()
        seen["parent"] = box.parentWidget()
        for button in box.buttons():
            if click is not None and button.text() == click:
                button.click()
                return
        if click is not None:
            # An exception raised in a slot would leave the box open and hang the
            # test, so record the miss and dismiss the box instead.
            seen["missing"] = click
        QTest.keyClick(box, Qt.Key.Key_Escape)

    timer.timeout.connect(_poll)
    timer.start()
    seen["_timer"] = timer
    return seen


def press_key(
    key: Qt.Key, modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
) -> QWidget | None:
    """Press ``key`` on whatever has keyboard focus, as a user would; return the new focus.

    The modifier is let go before anything reads it, as a real key would be.
    """

    target = QApplication.focusWidget()
    assert target is not None, "nothing has keyboard focus"
    QTest.keyClick(target, key, modifiers)
    release_keyboard_modifiers()
    QApplication.processEvents()
    return QApplication.focusWidget()


def assert_tab_leaves_table(table, *, presses: int = 12) -> list[QWidget]:
    """Real Tab and Shift+Tab presses take keyboard focus out of ``table``.

    ``table`` is shown in its active window first, with the tab, stack page or hidden
    group holding it brought forward. A table with no rows gets three rows of text,
    since a table with none has no cells to walk. Tab may visit the current row's
    controls on the way; Shift+Tab from the table leaves at once. Returns the
    controls in the table that Tab stopped on, in order.
    """

    from PySide6.QtWidgets import QStackedWidget, QTableWidgetItem, QTabWidget

    from mufinder.app.table_keyboard import TabStopTable

    assert isinstance(table, TabStopTable)
    if table.rowCount() == 0:
        table.setRowCount(3)
        for row in range(3):
            for column in range(table.columnCount()):
                table.setItem(row, column, QTableWidgetItem(f"{row}.{column}"))
    widget = table
    while not widget.isWindow():
        parent = widget.parentWidget()
        if isinstance(parent, QStackedWidget):
            # A tab widget's pages sit in a stack of its own; switch through the tab
            # widget, as a click on the tab does, so a page built on first show is built.
            tabs = parent.parentWidget()
            (tabs if isinstance(tabs, QTabWidget) else parent).setCurrentWidget(widget)
        if widget.isHidden():
            widget.show()
        widget = parent
    window = table.window()
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    if table.currentRow() < 0:
        column = next(c for c in range(table.columnCount()) if not table.isColumnHidden(c))
        table.setCurrentCell(0, column)
    QApplication.processEvents()
    assert table.isVisible()

    def _inside(focus: QWidget | None) -> bool:
        return focus is table or (focus is not None and table.isAncestorOf(focus))

    table.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()
    assert QApplication.focusWidget() is table
    stops = []
    for _ in range(presses):
        stops.append(press_key(Qt.Key.Key_Tab))
        if not _inside(stops[-1]):
            break
    assert not _inside(stops[-1]), f"{presses} Tab presses stayed in {table.accessibleName()!r}"
    # The controls in the rows are off the window's Tab path: back from the control
    # after the table never lands on a control in some row. (It is the table itself
    # unless that control keeps Shift+Tab, as a label's links do, or the window's
    # tab order runs another way.)
    back = press_key(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
    assert back is table or not _inside(back)

    table.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()
    focus = press_key(Qt.Key.Key_Tab, Qt.KeyboardModifier.ShiftModifier)
    assert not _inside(focus), f"Shift+Tab stayed in {table.accessibleName()!r}"
    return stops[:-1]

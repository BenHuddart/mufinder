"""Tests for the paste shortcut on editable text widgets."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QLineEdit,
    QPlainTextEdit,
    QTextEdit,
    QWidget,
)
import pytest

from mufinder.app import text_editing
from mufinder.app.text_editing import TextEditingShortcutFilter, install_text_editing_shortcuts


def _focus(qapp: QApplication, widget: QWidget) -> None:
    widget.show()
    widget.activateWindow()
    widget.setFocus()
    qapp.processEvents()


def test_text_editing_filter_pastes_with_command_v(qapp) -> None:
    edit = QLineEdit()
    _focus(qapp, edit)
    QApplication.clipboard().setText("MnSi")
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.MetaModifier)

    handled = TextEditingShortcutFilter().eventFilter(edit, event)

    assert handled is True
    assert edit.text() == "MnSi"
    assert event.isAccepted()


def test_text_editing_filter_does_not_paste_into_read_only_text(qapp) -> None:
    edit = QTextEdit()
    edit.setReadOnly(True)
    _focus(qapp, edit)
    QApplication.clipboard().setText("blocked")
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.MetaModifier)

    handled = TextEditingShortcutFilter().eventFilter(edit, event)

    assert handled is False
    assert edit.toPlainText() == ""


def test_install_text_editing_shortcuts_is_idempotent(qapp) -> None:
    install_text_editing_shortcuts(qapp)
    installed = getattr(qapp, "_mufinder_text_editing_shortcut_filter")

    install_text_editing_shortcuts(qapp)

    assert getattr(qapp, "_mufinder_text_editing_shortcut_filter") is installed


def _editable_combo() -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    combo.lineEdit().clear()
    return combo


def _text_of(widget: QWidget) -> str:
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text()
    return widget.toPlainText()  # type: ignore[attr-defined]


@pytest.mark.parametrize("factory", [QLineEdit, QPlainTextEdit, QTextEdit, _editable_combo])
def test_command_v_pastes_into_the_focused_text_widget(qapp, factory) -> None:
    install_text_editing_shortcuts(qapp)
    widget = factory()
    _focus(qapp, widget)
    QApplication.clipboard().setText("CoF2")

    QTest.keyClick(qapp.focusWidget(), Qt.Key.Key_V, Qt.KeyboardModifier.MetaModifier)

    assert _text_of(widget) == "CoF2"


def test_command_v_follows_focus_to_the_next_text_widget(qapp) -> None:
    install_text_editing_shortcuts(qapp)
    window = QWidget()
    first = QLineEdit(window)
    second = QPlainTextEdit(window)
    window.show()
    _focus(qapp, first)
    _focus(qapp, second)
    QApplication.clipboard().setText("Fe")

    QTest.keyClick(second, Qt.Key.Key_V, Qt.KeyboardModifier.MetaModifier)

    assert second.toPlainText() == "Fe"
    assert first.text() == ""


def test_command_v_leaves_read_only_text_unchanged(qapp) -> None:
    install_text_editing_shortcuts(qapp)
    edit = QPlainTextEdit()
    edit.setReadOnly(True)
    _focus(qapp, edit)
    QApplication.clipboard().setText("blocked")

    QTest.keyClick(edit, Qt.Key.Key_V, Qt.KeyboardModifier.MetaModifier)

    assert edit.toPlainText() == ""


def test_command_v_pastes_while_the_completer_popup_has_the_keys(qapp) -> None:
    install_text_editing_shortcuts(qapp)
    edit = QLineEdit()
    edit.setCompleter(QCompleter(["Mn", "MnSi"], edit))
    _focus(qapp, edit)
    edit.completer().complete()
    qapp.processEvents()
    popup = edit.completer().popup()
    assert popup.isVisible()
    QApplication.clipboard().setText("MnSi")

    QTest.keyClick(popup, Qt.Key.Key_V, Qt.KeyboardModifier.MetaModifier)

    assert edit.text() == "MnSi"
    popup.hide()


class _CountingFilter(TextEditingShortcutFilter):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.seen: list[QObject] = []

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API naming.
        self.seen.append(watched)
        return super().eventFilter(watched, event)


def test_filter_watches_only_the_focused_text_widget(qapp) -> None:
    shortcut_filter = _CountingFilter()
    first = QLineEdit()
    second = QLineEdit()
    button_area = QWidget()
    shortcut_filter.follow_focus(None, first)

    QApplication.sendEvent(button_area, QEvent(QEvent.Type.User))
    QApplication.sendEvent(second, QEvent(QEvent.Type.User))
    assert shortcut_filter.seen == []
    QApplication.sendEvent(first, QEvent(QEvent.Type.User))
    assert shortcut_filter.seen == [first]

    shortcut_filter.seen.clear()
    shortcut_filter.follow_focus(first, button_area)
    QApplication.sendEvent(first, QEvent(QEvent.Type.User))
    QApplication.sendEvent(button_area, QEvent(QEvent.Type.User))
    assert shortcut_filter.seen == []


class _FakeApplication(QObject):
    focusChanged = Signal(object, object)  # noqa: N815 - mirrors QApplication.

    def __init__(self, focus: QWidget | None) -> None:
        super().__init__()
        self._focus = focus
        self.filters: list[QObject] = []

    def focusWidget(self) -> QWidget | None:  # noqa: N802 - mirrors QApplication.
        return self._focus

    def installEventFilter(self, event_filter: QObject) -> None:  # noqa: N802 - mirrors Qt.
        self.filters.append(event_filter)


def test_install_puts_no_event_filter_on_the_application(qapp, monkeypatch) -> None:
    monkeypatch.setattr(text_editing, "TextEditingShortcutFilter", _CountingFilter)
    focused = QLineEdit()
    later = QPlainTextEdit()
    app = _FakeApplication(focused)

    install_text_editing_shortcuts(app)  # type: ignore[arg-type]

    assert app.filters == []
    shortcut_filter = getattr(app, "_mufinder_text_editing_shortcut_filter")
    QApplication.sendEvent(focused, QEvent(QEvent.Type.User))
    assert shortcut_filter.seen == [focused]

    shortcut_filter.seen.clear()
    app.focusChanged.emit(focused, later)
    QApplication.sendEvent(focused, QEvent(QEvent.Type.User))
    QApplication.sendEvent(later, QEvent(QEvent.Type.User))
    assert shortcut_filter.seen == [later]

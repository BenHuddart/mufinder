"""Tests for application-level text editing shortcuts."""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QTextEdit

from mufinder.app.text_editing import TextEditingShortcutFilter, install_text_editing_shortcuts


def test_text_editing_filter_pastes_with_command_v(qapp) -> None:
    edit = QLineEdit()
    edit.show()
    edit.setFocus()
    qapp.processEvents()
    QApplication.clipboard().setText("MnSi")
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_V, Qt.KeyboardModifier.MetaModifier)

    handled = TextEditingShortcutFilter().eventFilter(edit, event)

    assert handled is True
    assert edit.text() == "MnSi"
    assert event.isAccepted()


def test_text_editing_filter_does_not_paste_into_read_only_text(qapp) -> None:
    edit = QTextEdit()
    edit.setReadOnly(True)
    edit.show()
    edit.setFocus()
    qapp.processEvents()
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

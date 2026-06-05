"""Application-level text editing shortcut support."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QWidget


class TextEditingShortcutFilter(QObject):
    """Route standard text-editing shortcuts to focused editable text widgets."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API naming.
        if event.type() == QEvent.Type.ShortcutOverride and _is_paste_event(event):
            if _paste_target() is not None:
                event.accept()
            return False
        if event.type() != QEvent.Type.KeyPress or not _is_paste_event(event):
            return super().eventFilter(watched, event)

        target = _paste_target()
        if target is None:
            return super().eventFilter(watched, event)
        target.paste()
        event.accept()
        return True


def install_text_editing_shortcuts(app: QApplication) -> None:
    """Install standard text-editing shortcut handling once per QApplication."""

    if getattr(app, "_mufinder_text_editing_shortcut_filter", None) is not None:
        return
    install_filter = getattr(app, "installEventFilter", None)
    if not callable(install_filter):
        return
    shortcut_filter = TextEditingShortcutFilter(app)
    install_filter(shortcut_filter)
    setattr(app, "_mufinder_text_editing_shortcut_filter", shortcut_filter)


def _is_paste_event(event: QEvent) -> bool:
    matches = getattr(event, "matches", None)
    if callable(matches) and matches(QKeySequence.StandardKey.Paste):
        return True

    key = getattr(event, "key", lambda: None)()
    modifiers = getattr(event, "modifiers", lambda: Qt.KeyboardModifier.NoModifier)()
    if key != Qt.Key.Key_V:
        return False
    if modifiers & Qt.KeyboardModifier.AltModifier:
        return False
    command_or_control = modifiers & (
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
    )
    return bool(command_or_control)


def _paste_target() -> QLineEdit | QPlainTextEdit | QTextEdit | None:
    app = QApplication.instance()
    if app is None:
        return None
    widget = app.focusWidget()
    if isinstance(widget, QComboBox) and widget.isEditable():
        widget = widget.lineEdit()
    if not isinstance(widget, (QLineEdit, QPlainTextEdit, QTextEdit)):
        return None
    if not _is_editable_text_widget(widget):
        return None
    return widget


def _is_editable_text_widget(widget: QWidget) -> bool:
    if not widget.isEnabled():
        return False
    is_read_only = getattr(widget, "isReadOnly", None)
    if callable(is_read_only) and bool(is_read_only()):
        return False
    return True

"""Paste shortcut support for editable text widgets.

The filter watches only the focused editable text widget, never the whole
application: an application-wide Python event filter would run for every
paint, timer and layout event in the app.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QPlainTextEdit, QTextEdit, QWidget
from shiboken6 import isValid

# Widgets whose key events the filter watches while they hold focus; an
# editable combo box receives the keys its line edit edits.
_WATCHED_WIDGET_TYPES = (QLineEdit, QPlainTextEdit, QTextEdit, QComboBox)


class TextEditingShortcutFilter(QObject):
    """Route the paste shortcut to the focused editable text widget.

    Besides ``QKeySequence.StandardKey.Paste``, V with the Control or Meta
    modifier and without Alt pastes, so both Command-V and Control-V work on
    macOS. The filter is installed on the focused widget only
    (``follow_focus``).
    """

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

    def follow_focus(self, old: QWidget | None, now: QWidget | None) -> None:
        """Move the filter from the widget that lost focus to the one that gained it.

        A completer's popup takes the keys while it is open, so the filter
        watches the focused widget's completer popup too.
        """

        for watched in _watched_objects(old):
            watched.removeEventFilter(self)
        for watched in _watched_objects(now):
            watched.installEventFilter(self)


def install_text_editing_shortcuts(app: QApplication) -> None:
    """Install paste shortcut handling once per QApplication.

    The filter follows ``QApplication.focusChanged`` onto each focused text
    widget; nothing is installed on the application itself.
    """

    if getattr(app, "_mufinder_text_editing_shortcut_filter", None) is not None:
        return
    focus_changed = getattr(app, "focusChanged", None)
    if focus_changed is None or not callable(getattr(app, "focusWidget", None)):
        return
    shortcut_filter = TextEditingShortcutFilter(app)
    focus_changed.connect(shortcut_filter.follow_focus)
    shortcut_filter.follow_focus(None, app.focusWidget())
    setattr(app, "_mufinder_text_editing_shortcut_filter", shortcut_filter)


def _watched_objects(widget: QWidget | None) -> list[QWidget]:
    if not isinstance(widget, _WATCHED_WIDGET_TYPES) or not isValid(widget):
        return []
    watched: list[QWidget] = [widget]
    completer = widget.completer() if isinstance(widget, (QLineEdit, QComboBox)) else None
    if completer is not None and isValid(completer):
        watched.append(completer.popup())
    return watched


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

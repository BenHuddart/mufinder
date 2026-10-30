"""A single-line label that elides instead of widening its parent."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLabel, QWidget


class ElidingLabel(QLabel):
    """Single-line label that elides with an ellipsis instead of widening its parent.

    A plain ``QLabel`` reports its whole text as its minimum width, so a long
    text would stop whatever holds it from shrinking. ``text()`` still returns
    the full text, always shown as plain text. With word wrap on, it is a plain
    wrapping ``QLabel``.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        # Eliding paints the characters as they are; wrapping must not read them as HTML.
        self.setTextFormat(Qt.TextFormat.PlainText)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        hint = super().minimumSizeHint()
        return hint if self.wordWrap() else QSize(0, hint.height())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self.wordWrap():
            super().paintEvent(event)
            return
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, rect.width())
        painter = QPainter(self)
        self.style().drawItemText(
            painter,
            rect,
            int(self.alignment()),
            self.palette(),
            self.isEnabled(),
            text,
            self.foregroundRole(),
        )

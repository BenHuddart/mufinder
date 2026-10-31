"""A single-line label and a combo box that elide instead of clipping or widening."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)


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


class ElidingComboBox(QComboBox):
    """Combo box whose current text ends in an ellipsis when it does not fit.

    Qt clips a long current text at the edit field, part-way through a letter.
    ``currentText()`` still returns the full text; :meth:`displayed_text` is what
    is painted. The width stays whatever the size-adjust policy gives, so use
    ``AdjustToMinimumContentsLengthWithIcon`` to stop a long item widening the
    parent, and put the full text somewhere reachable such as the tooltip.
    """

    def _style_option(self) -> QStyleOptionComboBox:
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        return option

    def displayed_text(self) -> str:
        """Return the current text as painted: whole if it fits, else elided."""

        option = self._style_option()
        field = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        return self.fontMetrics().elidedText(
            option.currentText, Qt.TextElideMode.ElideRight, field.width()
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        del event
        painter = QStylePainter(self)
        option = self._style_option()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        option.currentText = self.displayed_text()
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)

"""A help icon that explains a control in place: a tooltip on hover, a bubble on click.

``help_button`` makes the icon. Hovering shows its text as a tooltip; clicking
it, or Space or Enter while it has keyboard focus, opens the same text in a
small bubble (``HelpBubble``) anchored under the icon. The bubble is a popup:
it stays until Escape, a click elsewhere or the window losing focus closes it,
and its text can be selected and copied. Longer explanations that need a
dialog keep their own buttons.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtWidgets import QFrame, QLabel, QToolButton, QVBoxLayout, QWidget

from mufinder.app.icons import icon
from mufinder.app.theme import (
    INLINE_HELP_ICON_SIZE,
    card_margins,
    scaled_to_text,
    style_inline_help_button,
)


class HelpBubble(QFrame):
    """A popup holding a help text, which the user can select; closed like any popup."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("mufinderHelpBubble")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        # A click on the icon that closes the bubble must not reach the icon and
        # open it again.
        self.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(card_margins())
        self._label = QLabel(text, self)
        self._label.setObjectName("mufinderHelpBubbleText")
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self._label.setMaximumWidth(scaled_to_text(360))
        layout.addWidget(self._label)

    def text(self) -> str:
        return self._label.text()

    def show_below(self, anchor: QWidget) -> None:
        """Open under ``anchor``, its left edge at the anchor's, kept on the anchor's screen."""

        self.adjustSize()
        position = anchor.mapToGlobal(QPoint(0, anchor.height()))
        screen = anchor.screen()
        if screen is not None:
            area = screen.availableGeometry()
            right = area.x() + area.width()
            position.setX(max(area.x(), min(position.x(), right - self.width())))
            if position.y() + self.height() > area.y() + area.height():
                position.setY(anchor.mapToGlobal(QPoint(0, 0)).y() - self.height())
        self.move(position)
        self.show()


class HelpButton(QToolButton):
    """The help icon: ``text`` as its tooltip, and in a bubble when clicked."""

    def __init__(self, text: str, *, accessible_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        # A small muted glyph at body-text height; the padding, not the glyph,
        # makes the hit area comfortable.
        style_inline_help_button(self)
        self.setIcon(icon("help_quiet"))
        side = scaled_to_text(INLINE_HELP_ICON_SIZE)
        self.setIconSize(QSize(side, side))
        self.setAutoRaise(True)
        self.setToolTip(text)
        self.setAccessibleName(accessible_name)
        self.setAccessibleDescription(text)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._text = text
        self._bubble: HelpBubble | None = None
        self.clicked.connect(self.show_bubble)

    def help_text(self) -> str:
        return self._text

    def bubble(self) -> HelpBubble | None:
        """The bubble while it is open, otherwise ``None``."""

        bubble = self._bubble
        if bubble is None:
            return None
        try:
            return bubble if bubble.isVisible() else None
        except RuntimeError:  # deleted on close
            return None

    def show_bubble(self) -> None:
        """Open the bubble under the icon, or leave the open one as it is."""

        if self.bubble() is not None:
            return
        self._bubble = HelpBubble(self._text, self)
        self._bubble.show_below(self)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        # A tool button clicks on Space only; Enter opens the bubble too.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.show_bubble()
            event.accept()
            return
        super().keyPressEvent(event)


def help_button(text: str, *, accessible_name: str, parent: QWidget | None = None) -> HelpButton:
    """Make a help icon explaining a control: ``text`` on hover and in a bubble on click.

    ``accessible_name`` says what it explains (``"About the propagation vector"``).
    """

    return HelpButton(text, accessible_name=accessible_name, parent=parent)

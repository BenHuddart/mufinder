"""A rounded card of facts: groups of captioned values on one line when they fit."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QMargins, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QLabel,
    QLayout,
    QLayoutItem,
    QMenu,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from mufinder.app.theme import (
    card_margins,
    form_spacing,
    refresh_widget_style,
    scaled_to_text,
)

#: One fact: its caption and its value as shown (``("a", "4.6950 Å")``).
Fact = tuple[str, str]


class _FactValue(QLabel):
    """A fact's value: selectable, and shortened in the middle when it is too wide.

    A long value, such as a file path, would otherwise set the narrowest the
    page can go. Below its full width it shows ``start…end`` with the whole
    value as its tooltip; ``value()`` is always the whole value, and the
    right-click ``Copy`` copies it whole (selecting the shortened text copies
    what is shown). It is not an ``ElidingLabel``: that label paints its text
    itself, so the text could not be selected.
    """

    def __init__(self, value: str, parent: QWidget | None = None) -> None:
        super().__init__(value, parent)
        self._value = value
        self.setObjectName("factValue")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def value(self) -> str:
        return self._value

    def copy_value(self) -> None:
        """Put the whole value on the clipboard, however it is shown."""

        QApplication.clipboard().setText(self._value)

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        menu = QMenu(self)
        menu.addAction("Copy", self.copy_value)
        menu.exec(event.globalPos())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        hint = super().sizeHint()
        return QSize(self.fontMetrics().horizontalAdvance(self._value) + 1, hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        hint = self.sizeHint()
        return QSize(min(hint.width(), scaled_to_text(_SHORTEST_VALUE_WIDTH)), hint.height())

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        shown = self.fontMetrics().elidedText(
            self._value, Qt.TextElideMode.ElideMiddle, self.contentsRect().width()
        )
        if shown != self.text():
            self.setText(shown)
        self.setToolTip("" if shown == self._value else self._value)


#: The narrowest a long fact value goes before it is shortened, in pixels of
#: the designed text size.
_SHORTEST_VALUE_WIDTH = 160
#: Spare width widens each gap between facts up to this many times its normal size.
_MOST_GAP_STRETCH = 2


@dataclass(slots=True)
class _Placed:
    item: QLayoutItem
    group: int
    hairline: bool


class _FactGroupsLayout(QLayout):
    """Lay groups of facts out on one line, a hairline between groups, when they fit.

    On one line, spare width widens every gap evenly, up to twice the normal
    gap (the hairline keeps equal space on either side); width beyond that is
    left empty at the right end. A fact wider than it needs to be, such as a
    long path, is shortened (down to its minimum) before the line breaks. When
    the facts do not fit on one line even so, each group starts a line of its
    own and the hairlines are hidden; a group too wide for the width wraps
    fact by fact.
    """

    def __init__(self, parent: QWidget | None = None, *, spacing: int) -> None:
        super().__init__(parent)
        self._placed: list[_Placed] = []
        self._gap = spacing
        self._one_line = True

    def add_fact(self, widget: QWidget, group: int) -> None:
        self.addChildWidget(widget)
        self._placed.append(_Placed(self._widget_item(widget), group, hairline=False))
        self.invalidate()

    def add_hairline(self, widget: QWidget, group: int) -> None:
        self.addChildWidget(widget)
        self._placed.append(_Placed(self._widget_item(widget), group, hairline=True))
        self.invalidate()

    @staticmethod
    def _widget_item(widget: QWidget) -> QLayoutItem:
        return QWidgetItem(widget)

    def is_one_line(self) -> bool:
        """Whether the last arrangement put every group on one line."""

        return self._one_line

    def spacing(self) -> int:
        return self._gap

    # QLayout API --------------------------------------------------------------

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt API naming.
        self._placed.append(_Placed(item, 0, hairline=False))

    def count(self) -> int:
        return len(self._placed)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt API naming.
        return self._placed[index].item if 0 <= index < len(self._placed) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt API naming.
        return self._placed.pop(index).item if 0 <= index < len(self._placed) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt API naming.
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API naming.
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API naming.
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API naming.
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        facts = self._facts()
        width = sum(item.sizeHint().width() for item, _group in facts) + self._one_line_gaps()
        height = max((item.sizeHint().height() for item, _group in facts), default=0)
        margins = self.contentsMargins()
        return QSize(
            width + margins.left() + margins.right(), height + margins.top() + margins.bottom()
        )

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API naming.
        size = QSize()
        for item, _group in self._facts():
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    # Arrangement --------------------------------------------------------------

    def _facts(self) -> list[tuple[QLayoutItem, int]]:
        return [(p.item, p.group) for p in self._placed if not p.hairline]

    def _hairlines(self) -> list[QLayoutItem]:
        return [p.item for p in self._placed if p.hairline]

    def _groups(self) -> list[list[QLayoutItem]]:
        groups: dict[int, list[QLayoutItem]] = {}
        for item, group in self._facts():
            groups.setdefault(group, []).append(item)
        return [groups[key] for key in sorted(groups)]

    def _hairline_width(self) -> int:
        hairlines = self._hairlines()
        return hairlines[0].sizeHint().width() if hairlines else 0

    def _one_line_gaps(self, gap: int | None = None) -> int:
        """The width of every gap on one line: between facts, and either side of each hairline."""

        gap = self._gap if gap is None else gap
        groups = self._groups()
        facts = sum(len(group) for group in groups)
        boundaries = max(len(groups) - 1, 0)
        # Between two groups: a gap, the hairline and a gap, in place of one gap.
        return gap * (max(facts - 1, 0) + boundaries) + boundaries * self._hairline_width()

    def _gap_count(self) -> int:
        groups = self._groups()
        facts = sum(len(group) for group in groups)
        return max(facts - 1, 0) + max(len(groups) - 1, 0)

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        groups = self._groups()
        if not groups:
            return margins.top() + margins.bottom()
        hints = {id(item): item.sizeHint().width() for group in groups for item in group}
        minimums = {id(item): item.minimumSize().width() for group in groups for item in group}
        gaps = self._one_line_gaps()
        natural = sum(hints.values()) + gaps
        narrowest = sum(minimums.values()) + gaps
        one_line = narrowest <= area.width()
        if apply and one_line != self._one_line:
            self._one_line = one_line
        if one_line:
            height = self._arrange_one_line(area, groups, hints, minimums, natural, apply)
        else:
            height = self._arrange_by_group(area, groups, hints, minimums, apply)
        return height + margins.top() + margins.bottom()

    def _arrange_one_line(self, area, groups, hints, minimums, natural, apply) -> int:
        widths = dict(hints)
        gap = self._gap
        if natural > area.width():
            # Shorten the facts that can be shortened, the widest first, down to
            # their minimum, until the line fits.
            deficit = natural - area.width()
            for key in sorted(widths, key=lambda k: hints[k] - minimums[k], reverse=True):
                cut = min(deficit, hints[key] - minimums[key])
                widths[key] -= cut
                deficit -= cut
                if deficit <= 0:
                    break
        else:
            spare = area.width() - natural
            count = self._gap_count()
            if count:
                gap += min(spare // count, self._gap * (_MOST_GAP_STRETCH - 1))
        height = max(item.sizeHint().height() for group in groups for item in group)
        if apply:
            x = area.x()
            hairlines = self._hairlines()
            for index, group in enumerate(groups):
                if index:
                    hairline = hairlines[index - 1] if index - 1 < len(hairlines) else None
                    if hairline is not None:
                        _show(hairline, True)
                        hairline.setGeometry(
                            QRect(x + gap, area.y(), self._hairline_width(), height)
                        )
                    x += 2 * gap + self._hairline_width()
                for position, item in enumerate(group):
                    if position:
                        x += gap
                    width = widths[id(item)]
                    item.setGeometry(QRect(x, area.y(), width, height))
                    x += width
        return height

    def _arrange_by_group(self, area, groups, hints, minimums, apply) -> int:
        y = area.y()
        if apply:
            for hairline in self._hairlines():
                _show(hairline, False)
        for group_index, group in enumerate(groups):
            if group_index:
                y += self._gap
            lines: list[list[tuple[QLayoutItem, int]]] = []
            line: list[tuple[QLayoutItem, int]] = []
            line_width = 0
            for item in group:
                width = min(hints[id(item)], max(area.width(), minimums[id(item)]))
                needed = width if not line else line_width + self._gap + width
                if line and needed > area.width():
                    lines.append(line)
                    line = []
                    needed = width
                line.append((item, width))
                line_width = needed
            if line:
                lines.append(line)
            for line_index, line_items in enumerate(lines):
                if line_index:
                    y += self._gap
                height = max(item.sizeHint().height() for item, _width in line_items)
                if apply:
                    x = area.x()
                    for item, width in line_items:
                        item.setGeometry(QRect(x, y, width, height))
                        x += width + self._gap
                y += height
        return y - area.y()


def _show(item: QLayoutItem, shown: bool) -> None:
    widget = item.widget()
    if widget is not None and widget.isHidden() == shown:
        widget.setVisible(shown)


class FactCard(QFrame):
    """Show groups of facts, each a small muted caption over its value.

    ``set_groups`` takes the facts in groups. The groups share one line, a thin
    hairline between each two, when they fit; spare width widens the gaps
    evenly up to twice their size and the rest is left at the right end. When
    they do not fit, each group takes a line of its own with no hairlines, and
    a group too wide for the card wraps fact by fact. A long value is
    shortened in the middle, down to a minimum, before the line breaks.
    ``set_message`` shows a line of text alone instead, without the frame, for
    when there are no facts. Values can be selected and copied; each value's
    accessible description is its caption and value (``"a, 4.6950 Å"``), so a
    screen reader hears both.
    """

    def __init__(self, *, accessible_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("factCard")
        self.setAccessibleName(accessible_name)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(card_margins())
        self._layout.setSpacing(0)
        self._message = QLabel()
        self._message.setObjectName("factCardMessage")
        self._message.setTextFormat(Qt.TextFormat.PlainText)
        self._message.setWordWrap(True)
        self._layout.addWidget(self._message)
        self._facts_area: QWidget | None = None
        self._facts_layout: _FactGroupsLayout | None = None
        self._groups: list[list[QWidget]] = []
        self._hairlines: list[QFrame] = []
        self._values: dict[str, _FactValue] = {}

    def set_message(self, text: str) -> None:
        """Show ``text`` alone, with no facts and without the card's frame."""

        self._clear_facts()
        self._message.setText(text)
        self._message.show()
        self._set_framed(False)

    def set_groups(self, groups: Sequence[Sequence[Fact]]) -> None:
        """Show ``groups`` of facts, replacing whatever the card showed."""

        self._clear_facts()
        self._message.hide()
        self._set_framed(True)
        area = QWidget(self)
        # Transparent through the theme, not a style sheet of its own, which
        # would make the hairlines inside it transparent too.
        area.setObjectName("factCardFacts")
        layout = _FactGroupsLayout(area, spacing=form_spacing() * 2)
        layout.setContentsMargins(0, 0, 0, 0)
        for index, facts in enumerate(groups):
            if index:
                hairline = QFrame(area)
                hairline.setObjectName("factCardHairline")
                hairline.setFixedWidth(1)
                layout.add_hairline(hairline, index)
                self._hairlines.append(hairline)
            widgets = []
            for caption, value in facts:
                fact = self._fact(area, caption, value)
                layout.add_fact(fact, index)
                widgets.append(fact)
            self._groups.append(widgets)
        self._layout.addWidget(area)
        self._facts_area = area
        self._facts_layout = layout

    def value_label(self, caption: str) -> _FactValue | None:
        """The label showing the value captioned ``caption``, or ``None``."""

        return self._values.get(caption)

    def facts(self) -> list[list[Fact]]:
        """The facts shown, group by group; empty while the card shows a message."""

        return [
            [
                (fact.findChild(QLabel, "factCaption").text(), label.value())
                for fact in group
                for label in [fact.findChild(_FactValue, "factValue")]
            ]
            for group in self._groups
        ]

    def hairlines(self) -> list[QFrame]:
        """The hairlines between groups, shown only while the groups share one line."""

        return list(self._hairlines)

    def is_one_line(self) -> bool:
        """Whether every group shares one line."""

        return self._facts_layout is not None and self._facts_layout.is_one_line()

    def message(self) -> str:
        """The text shown in place of facts, or ``""`` when facts are shown."""

        return "" if self._message.isHidden() else self._message.text()

    def _fact(self, parent: QWidget, caption: str, value: str) -> QWidget:
        fact = QWidget(parent)
        fact.setObjectName("fact")
        layout = QVBoxLayout(fact)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        caption_label = QLabel(caption, fact)
        caption_label.setObjectName("factCaption")
        caption_label.setTextFormat(Qt.TextFormat.PlainText)
        value_label = _FactValue(value, fact)
        value_label.setAccessibleName(caption)
        value_label.setAccessibleDescription(f"{caption}, {value}")
        layout.addWidget(caption_label)
        layout.addWidget(value_label)
        self._values[caption] = value_label
        return fact

    def _set_framed(self, framed: bool) -> None:
        """Draw the rounded frame round facts; a message alone reads as plain text."""

        self._layout.setContentsMargins(card_margins() if framed else QMargins())
        self.setProperty("factCardState", "facts" if framed else "message")
        refresh_widget_style(self)

    def _clear_facts(self) -> None:
        if self._facts_area is not None:
            self._layout.removeWidget(self._facts_area)
            self._facts_area.hide()
            self._facts_area.deleteLater()
        self._facts_area = None
        self._facts_layout = None
        self._groups = []
        self._hairlines = []
        self._values = {}

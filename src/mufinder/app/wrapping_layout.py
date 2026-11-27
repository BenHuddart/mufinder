"""A row layout that wraps its items onto further lines when the width runs out."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QLayoutItem, QWidget


class WrappingRowLayout(QLayout):
    """Lay items out left to right, wrapping onto further lines when the width runs out.

    The minimum width is the widest single item, so a parent can shrink the row
    that far and the items move onto extra lines instead of being clipped. The
    last item on a line takes the rest of that line when it expands
    horizontally, so a right-aligned control inside it stays at the edge.
    """

    def __init__(self, parent: QWidget | None = None, *, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[QLayoutItem] = []
        self._item_spacing = spacing

    def spacing(self) -> int:
        """The gap between items on a line and between lines."""

        return self._item_spacing

    def setSpacing(self, spacing: int) -> None:  # noqa: N802 - Qt API naming.
        self._item_spacing = int(spacing)
        self.invalidate()

    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt API naming.
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt API naming.
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int) -> QLayoutItem | None:  # noqa: N802 - Qt API naming.
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt API naming.
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API naming.
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API naming.
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API naming.
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        visible = [item.sizeHint() for item in self._items if not item.isEmpty()]
        width = sum(hint.width() for hint in visible) + self._item_spacing * max(
            len(visible) - 1, 0
        )
        height = max((hint.height() for hint in visible), default=0)
        margins = self.contentsMargins()
        return QSize(
            width + margins.left() + margins.right(), height + margins.top() + margins.bottom()
        )

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API naming.
        size = QSize()
        for item in self._items:
            if not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        """Place the items inside ``rect`` (when ``apply``) and return the height used."""

        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        lines: list[list[tuple[QLayoutItem, int]]] = []
        line: list[tuple[QLayoutItem, int]] = []
        line_width = 0
        for item in self._items:
            if item.isEmpty():
                continue
            width = min(item.sizeHint().width(), max(area.width(), item.minimumSize().width()))
            needed = width if not line else line_width + self._item_spacing + width
            if line and needed > area.width():
                lines.append(line)
                line = []
                needed = width
            line.append((item, width))
            line_width = needed
        if line:
            lines.append(line)

        y = area.y()
        for line_index, line_items in enumerate(lines):
            if line_index:
                y += self._item_spacing
            line_height = max(item.sizeHint().height() for item, _width in line_items)
            if apply:
                x = area.x()
                for position, (item, width) in enumerate(line_items):
                    is_last = position == len(line_items) - 1
                    if is_last and item.expandingDirections() & Qt.Orientation.Horizontal:
                        width = max(width, area.right() + 1 - x)
                    item.setGeometry(QRect(x, y, width, line_height))
                    x += width + self._item_spacing
            y += line_height
        return y - rect.y() + margins.bottom()

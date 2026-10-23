"""Reusable native Qt widget for LaTeX-style math rendering."""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import re

from matplotlib import rc_context
from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

try:
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image as _math_to_image
except Exception:  # pragma: no cover - optional runtime dependency fallback.
    FontProperties = None
    _math_to_image = None


def _normalize_math_expression(expression: str) -> str:
    # Matplotlib mathtext accepts a strict LaTeX subset, so normalize a few
    # symbols used by MuFinder's scientific help text to mathtext-friendly forms.
    normalized = expression
    normalized = re.sub(r"\\\\+", r"\\", normalized)
    normalized = normalized.replace(r"\left\lVert", r"\left\Vert")
    normalized = normalized.replace(r"\right\rVert", r"\right\Vert")
    normalized = normalized.replace(r"\lVert", r"\Vert")
    normalized = normalized.replace(r"\rVert", r"\Vert")
    normalized = re.sub(r"\\le(?![A-Za-z])", r"\\leq", normalized)
    normalized = re.sub(r"\\ge(?![A-Za-z])", r"\\geq", normalized)
    normalized = re.sub(r"\\ne(?![A-Za-z])", r"\\neq", normalized)
    return normalized.strip()


@lru_cache(maxsize=256)
def _render_png(
    expression: str, display_mode: bool, color: str, device_pixel_ratio_x100: int
) -> bytes | None:
    if _math_to_image is None or FontProperties is None:
        return None

    if not expression:
        return None

    delimited = expression
    if not (delimited.startswith("$") and delimited.endswith("$")):
        delimited = f"${delimited}$"

    buffer = BytesIO()
    font_size = 15 if display_mode else 12
    properties = FontProperties(size=font_size)
    device_pixel_ratio = max(1.0, device_pixel_ratio_x100 / 100.0)
    dpi = 96 * device_pixel_ratio
    try:
        with rc_context(
            {
                "savefig.transparent": True,
                "savefig.facecolor": "none",
                "figure.facecolor": "none",
            }
        ):
            _math_to_image(
                delimited,
                buffer,
                format="png",
                dpi=dpi,
                prop=properties,
                color=color,
            )
    except Exception:
        return None
    return buffer.getvalue()


class _MathPixmapLabel(QLabel):
    """Display one rendered equation pixmap at its device-independent size."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("mathRenderedLine")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def set_png_data(self, png_data: bytes, device_pixel_ratio: float) -> None:
        pixmap = QPixmap()
        if not pixmap.loadFromData(png_data, "PNG"):
            self.clear()
            return
        pixmap.setDevicePixelRatio(max(1.0, device_pixel_ratio))
        self.setPixmap(pixmap)
        self.updateGeometry()


class MathLabel(QWidget):
    """Render one or more LaTeX-style math expressions using native Qt widgets."""

    def __init__(
        self,
        math: str = "",
        *,
        display_mode: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._math = ""
        self._display_mode = display_mode
        self._line_widgets: list[QWidget] = []
        self._render_device_pixel_ratio = 1.0
        self._render_color = "#000000"

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

        self.set_math(math)

    def math(self) -> str:
        return self._math

    def set_math(self, math: str, display_mode: bool | None = None) -> None:
        self._math = str(math or "")
        if display_mode is not None:
            self._display_mode = bool(display_mode)
        self._rerender_lines()

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API naming.
        return False

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API naming.
        return self.sizeHint().height()

    def sizeHint(self) -> QSize:
        if not self._line_widgets:
            return QSize(0, 0)

        margins = self.contentsMargins()
        content_width = max(widget.sizeHint().width() for widget in self._line_widgets)
        content_height = sum(widget.sizeHint().height() for widget in self._line_widgets)
        if len(self._line_widgets) > 1:
            content_height += self._layout.spacing() * (len(self._line_widgets) - 1)
        return QSize(
            margins.left() + content_width + margins.right(),
            margins.top() + content_height + margins.bottom(),
        )

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._rerender_lines_if_context_changed()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
            QEvent.Type.StyleChange,
            QEvent.Type.FontChange,
            QEvent.Type.ScreenChangeInternal,
        }:
            self._rerender_lines_if_context_changed()

    def _clear_lines(self) -> None:
        self._line_widgets.clear()
        while self._layout.count():
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _rerender_lines_if_context_changed(self) -> None:
        current_dpr = self._current_device_pixel_ratio()
        current_color = self._current_text_color()
        if (
            abs(current_dpr - self._render_device_pixel_ratio) > 0.01
            or current_color != self._render_color
        ):
            self._rerender_lines()

    def _rerender_lines(self) -> None:
        self._render_device_pixel_ratio = self._current_device_pixel_ratio()
        self._render_color = self._current_text_color()

        self._clear_lines()
        for line in [segment.strip() for segment in self._math.splitlines() if segment.strip()]:
            normalized = _normalize_math_expression(line)
            png_data = _render_png(
                normalized,
                self._display_mode,
                self._render_color,
                round(self._render_device_pixel_ratio * 100),
            )
            if png_data is None:
                widget = self._build_fallback_label(line)
            else:
                widget = _MathPixmapLabel(self)
                widget.set_png_data(png_data, self._render_device_pixel_ratio)
            self._line_widgets.append(widget)
            self._layout.addWidget(widget, 0, Qt.AlignmentFlag.AlignLeft)

        self.updateGeometry()

    def _current_device_pixel_ratio(self) -> float:
        screen = self.screen()
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return 1.0
        return max(1.0, screen.devicePixelRatio())

    def _current_text_color(self) -> str:
        return self.palette().windowText().color().name()

    def _build_fallback_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setObjectName("mathFallbackLine")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setToolTip("Unable to render this expression; showing source text instead.")
        return label

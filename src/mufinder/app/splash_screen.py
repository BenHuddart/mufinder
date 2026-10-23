"""Branded splash screen shown while the main window is initializing."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QGuiApplication,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSplashScreen

from mufinder.app.theme import ensure_fonts_loaded
from mufinder.branding import logo_path

_TITLE_FONT_CANDIDATES = ("Fraunces", "Playfair Display", "Libre Baskerville", "Georgia", "Serif")
_BODY_FONT_CANDIDATES = ("Inter", "Helvetica Neue", "Arial", "Helvetica")


class MuFinderSplashScreen(QSplashScreen):
    """Simple, branded splash screen with progress indicator."""

    _WIDTH = 680
    _HEIGHT = 360

    def __init__(self) -> None:
        super().__init__(self._make_canvas(self._fallback_device_pixel_ratio()))
        ensure_fonts_loaded()
        self._progress = 0
        self._message = "Starting MuFinder..."
        self._logo = self._load_logo()
        self._title_font_family = self._best_font_family(_TITLE_FONT_CANDIDATES, fallback="Serif")
        self._body_font_family = self._best_font_family(_BODY_FONT_CANDIDATES, fallback="Helvetica")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self._paint()

    def set_progress(self, value: int, message: str) -> None:
        """Update progress bar and status text."""

        self._progress = max(0, min(100, int(value)))
        self._message = message.strip() or self._message
        self._paint()
        self.showMessage("", Qt.AlignmentFlag.AlignLeft, QColor("transparent"))

    def _paint(self) -> None:
        dpr = self._device_pixel_ratio()
        pixmap = self._make_canvas(dpr)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        gradient = QLinearGradient(0.0, 0.0, float(self._WIDTH), float(self._HEIGHT))
        gradient.setColorAt(0.0, QColor("#F5EDE0"))
        gradient.setColorAt(1.0, QColor("#EDE0CE"))
        painter.fillRect(0, 0, self._WIDTH, self._HEIGHT, QBrush(gradient))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#E02020"))
        painter.drawRoundedRect(0, 0, self._WIDTH, 8, 4, 4)

        logo_scaled = self._scaled_logo(dpr)
        if logo_scaled is not None:
            logo_x = (
                self._WIDTH - int(logo_scaled.width() / max(0.1, logo_scaled.devicePixelRatio()))
            ) // 2
            painter.drawPixmap(logo_x, 50, logo_scaled)

        title_font = QFont(self._title_font_family)
        title_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        title_font.setPixelSize(26)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor("#2C2418")))
        painter.drawText(0, 218, self._WIDTH, 40, int(Qt.AlignmentFlag.AlignHCenter), "MuFinder")

        subtitle_font = QFont(self._body_font_family)
        subtitle_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        subtitle_font.setPixelSize(12)
        painter.setFont(subtitle_font)
        painter.setPen(QPen(QColor("#7A6B59")))
        painter.drawText(
            0, 246, self._WIDTH, 24, int(Qt.AlignmentFlag.AlignHCenter), "Preparing your expedition"
        )

        bar_x = 84
        bar_y = 294
        bar_w = 512
        bar_h = 10
        painter.setPen(QPen(QColor("#C4AD8F"), 1))
        painter.setBrush(QColor("#FDF6EC"))
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 5, 5)

        fill_w = int((bar_w - 2) * (self._progress / 100.0))
        painter.setPen(QPen(QColor("#2C2418"), 2))
        painter.drawLine(
            bar_x + 1, bar_y + (bar_h // 2), bar_x + max(1, fill_w), bar_y + (bar_h // 2)
        )

        message_font = QFont(self._body_font_family)
        message_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        message_font.setPixelSize(11)
        painter.setFont(message_font)
        painter.setPen(QPen(QColor("#7A6B59")))
        painter.drawText(0, 318, self._WIDTH, 26, int(Qt.AlignmentFlag.AlignHCenter), self._message)

        painter.end()
        self.setPixmap(pixmap)

    @staticmethod
    def _load_logo() -> QPixmap | None:
        logo = QPixmap(str(Path(logo_path())))
        if logo.isNull():
            return None
        return logo

    @classmethod
    def _make_canvas(cls, dpr: float) -> QPixmap:
        physical_width = max(1, int(round(cls._WIDTH * dpr)))
        physical_height = max(1, int(round(cls._HEIGHT * dpr)))
        pixmap = QPixmap(physical_width, physical_height)
        pixmap.setDevicePixelRatio(dpr)
        pixmap.fill(Qt.GlobalColor.transparent)
        return pixmap

    @staticmethod
    def _best_font_family(candidates: tuple[str, ...], *, fallback: str) -> str:
        available = {family.casefold(): family for family in QFontDatabase.families()}
        for candidate in candidates:
            matched = available.get(candidate.casefold())
            if matched is not None:
                return matched
        return fallback

    @staticmethod
    def _fallback_device_pixel_ratio() -> float:
        app = QGuiApplication.instance()
        if app is None:
            return 1.0
        screen = app.primaryScreen()
        if screen is None:
            return 1.0
        return max(1.0, float(screen.devicePixelRatio()))

    def _device_pixel_ratio(self) -> float:
        handle = self.windowHandle()
        if handle is not None and handle.screen() is not None:
            return max(1.0, float(handle.screen().devicePixelRatio()))
        return self._fallback_device_pixel_ratio()

    def _scaled_logo(self, dpr: float) -> QPixmap | None:
        if self._logo is None:
            return None
        physical_size = max(1, int(round(138 * dpr)))
        logo = self._logo.scaled(
            physical_size,
            physical_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        logo.setDevicePixelRatio(dpr)
        return logo

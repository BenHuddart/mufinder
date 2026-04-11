"""Icon lookup helpers for MuFinder widgets and actions."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QStyle

_ICON_MAP: dict[str, QStyle.StandardPixmap] = {
    "new_project": QStyle.StandardPixmap.SP_FileDialogNewFolder,
    "open_project": QStyle.StandardPixmap.SP_DialogOpenButton,
    "save_project": QStyle.StandardPixmap.SP_DialogSaveButton,
    "demo": QStyle.StandardPixmap.SP_ComputerIcon,
    "refresh": QStyle.StandardPixmap.SP_BrowserReload,
    "cluster": QStyle.StandardPixmap.SP_DriveNetIcon,
    "settings": QStyle.StandardPixmap.SP_FileDialogDetailedView,
    "appearance_light": QStyle.StandardPixmap.SP_DialogYesButton,
    "appearance_dark": QStyle.StandardPixmap.SP_DialogNoButton,
    "appearance_system": QStyle.StandardPixmap.SP_BrowserReload,
    "host_structure": QStyle.StandardPixmap.SP_DirIcon,
    "convergence": QStyle.StandardPixmap.SP_MediaPlay,
    "kpoint": QStyle.StandardPixmap.SP_ArrowRight,
    "muon": QStyle.StandardPixmap.SP_DialogApplyButton,
    "results": QStyle.StandardPixmap.SP_DialogHelpButton,
}


def icon(name: str) -> QIcon:
    """Return a standard icon for the requested semantic name."""

    app = QApplication.instance()
    if app is None:
        return QIcon()

    style = app.style()
    standard = _ICON_MAP.get(name)
    if standard is None:
        return QIcon()
    return style.standardIcon(standard)


def _icon_device_pixel_ratio() -> float:
    app = QGuiApplication.instance()
    if app is None:
        return 1.0
    screen = app.primaryScreen()
    if screen is None:
        return 1.0
    return max(1.0, float(screen.devicePixelRatio()))


def _icon_canvas(logical_size: int = 16) -> QPixmap:
    dpr = _icon_device_pixel_ratio()
    physical_size = max(1, int(round(logical_size * dpr)))
    pixmap = QPixmap(physical_size, physical_size)
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    return pixmap


def workflow_stage_icon(state: str) -> QIcon:
    """Return icon for workflow stage state: completed, active, or incomplete."""

    from mufinder.app.theme import current_palette

    palette = current_palette()
    edge = QColor(palette.border)
    fill = QColor(palette.bg_surface)
    complete = QColor(palette.seal_gold)
    active = QColor(palette.accent)
    pending = QColor(palette.text_muted)

    normalized = state.strip().lower()
    if normalized not in {"completed", "active", "incomplete"}:
        normalized = "incomplete"

    pixmap = _icon_canvas(16)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    if normalized == "completed":
        painter.setPen(QPen(edge, 1.1))
        painter.setBrush(complete)
        painter.drawEllipse(2, 2, 12, 12)
        painter.setPen(QPen(fill, 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        seal = QPainterPath()
        seal.moveTo(5.0, 8.0)
        seal.lineTo(7.0, 10.2)
        seal.lineTo(11.0, 5.8)
        painter.drawPath(seal)
    else:
        painter.setPen(QPen(pending, 1.3, Qt.PenStyle.DotLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(fill)
        painter.drawEllipse(3, 3, 10, 10)

    painter.end()
    return QIcon(pixmap)


def workflow_tick_icon(completed: bool) -> QIcon:
    """Return legacy workflow icon for backward-compatible call sites."""

    return workflow_stage_icon("completed" if completed else "incomplete")

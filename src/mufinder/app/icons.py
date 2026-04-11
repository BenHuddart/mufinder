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
    candidate = style.standardIcon(standard)
    if not candidate.isNull():
        preview = candidate.pixmap(18, 18)
        if not preview.isNull():
            return candidate
    return _semantic_icon(name)


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


def _semantic_icon(name: str) -> QIcon:
    pixmap = _icon_canvas(18)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    palette = QApplication.palette()
    ink = palette.buttonText().color()
    accent = palette.highlight().color()
    muted = palette.mid().color()
    paper = palette.base().color()

    painter.setPen(QPen(ink, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

    if name == "new_project":
        _draw_document_icon(painter, ink, paper)
        painter.setPen(QPen(accent, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(9, 5, 9, 11)
        painter.drawLine(6, 8, 12, 8)
    elif name == "open_project":
        _draw_folder_icon(painter, ink, muted)
    elif name == "save_project":
        _draw_save_icon(painter, ink, paper, accent)
    elif name == "demo":
        _draw_monitor_icon(painter, ink, accent)
    elif name == "cluster":
        _draw_cluster_icon(painter, ink, accent, paper)
    elif name == "settings":
        _draw_settings_icon(painter, ink, accent)
    elif name == "appearance_light":
        _draw_sun_icon(painter, ink, accent)
    elif name == "appearance_dark":
        _draw_moon_icon(painter, ink, paper)
    elif name == "appearance_system":
        _draw_system_icon(painter, ink, accent, paper)
    else:
        painter.end()
        return QIcon()

    painter.end()
    return QIcon(pixmap)


def _draw_document_icon(painter: QPainter, ink: QColor, paper: QColor) -> None:
    path = QPainterPath()
    path.moveTo(4, 2.5)
    path.lineTo(10.5, 2.5)
    path.lineTo(14, 6)
    path.lineTo(14, 15)
    path.lineTo(4, 15)
    path.closeSubpath()
    painter.setBrush(paper)
    painter.drawPath(path)
    painter.drawLine(10.5, 2.5, 10.5, 6)
    painter.drawLine(10.5, 6, 14, 6)


def _draw_folder_icon(painter: QPainter, ink: QColor, muted: QColor) -> None:
    painter.setBrush(muted.lighter(140))
    painter.drawRoundedRect(2.5, 6.0, 13.0, 8.0, 2.0, 2.0)
    painter.drawRoundedRect(3.5, 4.0, 6.0, 3.5, 1.4, 1.4)


def _draw_save_icon(painter: QPainter, ink: QColor, paper: QColor, accent: QColor) -> None:
    painter.setBrush(ink.lighter(175))
    painter.drawRoundedRect(3.0, 3.0, 12.0, 12.0, 1.8, 1.8)
    painter.setBrush(paper)
    painter.drawRect(5.0, 4.5, 7.0, 3.5)
    painter.setBrush(accent)
    painter.drawRect(6.0, 10.0, 6.0, 3.5)


def _draw_monitor_icon(painter: QPainter, ink: QColor, accent: QColor) -> None:
    painter.setBrush(accent.lighter(170))
    painter.drawRoundedRect(2.5, 3.0, 13.0, 9.0, 1.6, 1.6)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawLine(7.0, 13.5, 11.0, 13.5)
    painter.drawLine(9.0, 12.0, 9.0, 15.5)


def _draw_cluster_icon(painter: QPainter, ink: QColor, accent: QColor, paper: QColor) -> None:
    painter.setPen(QPen(ink, 1.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawLine(5.0, 6.0, 9.0, 9.0)
    painter.drawLine(13.0, 6.0, 9.0, 9.0)
    painter.drawLine(9.0, 9.0, 9.0, 13.0)
    painter.setBrush(paper)
    painter.drawEllipse(3.0, 3.0, 4.0, 4.0)
    painter.drawEllipse(11.0, 3.0, 4.0, 4.0)
    painter.setBrush(accent)
    painter.drawEllipse(7.0, 11.0, 4.0, 4.0)


def _draw_settings_icon(painter: QPainter, ink: QColor, accent: QColor) -> None:
    painter.setBrush(accent.lighter(165))
    for x1, y1, x2, y2 in (
        (9, 2.5, 9, 5),
        (9, 13, 9, 15.5),
        (2.5, 9, 5, 9),
        (13, 9, 15.5, 9),
        (4.3, 4.3, 5.8, 5.8),
        (12.2, 12.2, 13.7, 13.7),
        (12.2, 5.8, 13.7, 4.3),
        (4.3, 13.7, 5.8, 12.2),
    ):
        painter.drawLine(x1, y1, x2, y2)
    painter.setBrush(accent.lighter(150))
    painter.drawEllipse(5.0, 5.0, 8.0, 8.0)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(7.2, 7.2, 3.6, 3.6)


def _draw_sun_icon(painter: QPainter, ink: QColor, accent: QColor) -> None:
    painter.setBrush(accent.lighter(135))
    painter.drawEllipse(5.0, 5.0, 8.0, 8.0)
    for x1, y1, x2, y2 in (
        (9, 1.5, 9, 4),
        (9, 14, 9, 16.5),
        (1.5, 9, 4, 9),
        (14, 9, 16.5, 9),
        (3.2, 3.2, 4.8, 4.8),
        (13.2, 13.2, 14.8, 14.8),
        (13.2, 4.8, 14.8, 3.2),
        (3.2, 14.8, 4.8, 13.2),
    ):
        painter.drawLine(x1, y1, x2, y2)


def _draw_moon_icon(painter: QPainter, ink: QColor, paper: QColor) -> None:
    painter.setBrush(ink)
    painter.drawEllipse(4.0, 3.0, 10.0, 10.0)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(paper)
    painter.drawEllipse(7.5, 2.5, 8.0, 10.5)
    painter.setPen(QPen(ink, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))


def _draw_system_icon(painter: QPainter, ink: QColor, accent: QColor, paper: QColor) -> None:
    painter.setBrush(paper)
    painter.drawRoundedRect(2.5, 3.0, 13.0, 9.0, 1.6, 1.6)
    painter.drawLine(7.0, 13.5, 11.0, 13.5)
    painter.drawLine(9.0, 12.0, 9.0, 15.5)
    painter.setBrush(accent.lighter(150))
    painter.drawRect(4.0, 4.5, 4.5, 6.0)
    painter.setBrush(ink.lighter(185))
    painter.drawRect(8.5, 4.5, 5.0, 6.0)

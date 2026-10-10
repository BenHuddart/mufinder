"""Project home page: the clickable trail map and its workflow landmarks.

``WorkflowDefinition`` describes the workflow tree that the project navigator
and the home trail are built from, ``ProjectTrailDestination`` one landmark on
the trail, ``ProjectHomeTrailWidget`` paints the map and emits the route of the
landmark the user clicks, and ``ConstructionSignWidget`` stands in for a
workflow page that is not yet available.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from mufinder.app.theme import THEME_LIGHT, body_font_family, current_palette, heading_font_family, light_palette
from mufinder.branding import logo_path, project_home_color_map_path, project_home_map_path

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """Describe a workflow node and its completion rule."""

    key: str
    title: str
    subtitle: str = ""
    route: str | None = None
    completion_goal: str | None = None
    completion_check: Callable[[ProjectDocument], bool] | None = None
    children: tuple["WorkflowDefinition", ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectTrailDestination:
    """Describe one clickable landmark on the project home trail."""

    title: str
    route: str
    theme: str
    completed: bool = False
    completion_goal: str | None = None
    placement: str = "trail"


class ProjectHomeTrailWidget(QWidget):
    """Paint the clickable project-home trail map."""

    route_requested = Signal(str)

    _SOURCE_WIDTH = 1536.0
    _SOURCE_HEIGHT = 1024.0
    _ARTWORK_SOURCE_REGION = QRectF(80.0, 44.0, 1421.0, 800.0)
    _SIGN_REGIONS = {
        "host_structure": QRectF(114.0, 340.0, 196.0, 46.0),
        "supercell_editor": QRectF(422.0, 315.0, 262.0, 46.0),
        "convergence_cutoff": QRectF(356.0, 503.0, 252.0, 46.0),
        "convergence_kpoint": QRectF(763.0, 311.0, 220.0, 46.0),
        "pristine_relaxation": QRectF(835.0, 502.0, 250.0, 46.0),
        "uep": QRectF(1078.0, 276.0, 164.0, 47.0),
        "muon_candidates": QRectF(1275.0, 382.0, 201.0, 47.0),
        "magnetic_fields": QRectF(1128.0, 586.0, 196.0, 47.0),
        "quantum_corrections": QRectF(711.0, 758.0, 225.0, 47.0),
        "energy_barriers": QRectF(224.0, 747.0, 245.0, 47.0),
    }
    _SIGN_TEXT_INSETS = {
        "host_structure": (12.0, 1.0, 14.0, 10.0),
        "supercell_editor": (14.0, 1.0, 16.0, 10.0),
        "convergence_cutoff": (12.0, 1.0, 14.0, 10.0),
        "convergence_kpoint": (12.0, 1.0, 14.0, 10.0),
        "pristine_relaxation": (14.0, 1.0, 16.0, 10.0),
        "uep": (14.0, 1.0, 16.0, 10.0),
        "muon_candidates": (8.0, 1.0, 10.0, 10.0),
        "magnetic_fields": (6.0, 1.0, 8.0, 10.0),
        "quantum_corrections": (6.0, 1.0, 8.0, 10.0),
        "energy_barriers": (14.0, 1.0, 16.0, 10.0),
    }
    _SIGN_TEXT_OFFSETS = {
        "host_structure": (0.0, 6.0),
        "supercell_editor": (-16.0, 3.0),
        "convergence_cutoff": (-20.0, 6.0),
        "convergence_kpoint": (-5.0, 5.0),
        "pristine_relaxation": (-12.0, 6.0),
        "muon_candidates": (9.0, -3.0),
        "magnetic_fields": (0.0, 12.0),
        "quantum_corrections": (12.0, 3.0),
    }
    _SIGN_HIGHLIGHT_OFFSETS = {
        "supercell_editor": (-14.0, 0.0),
        "convergence_cutoff": (-14.0, 0.0),
        "muon_candidates": (8.0, -5.0),
    }
    _SIGN_TEXT_MAX_SIZES = {
        "magnetic_fields": 28,
        "quantum_corrections": 28,
    }
    _LANDMARK_REGIONS = {
        "host_structure": QRectF(70.0, 100.0, 260.0, 230.0),
        "supercell_editor": QRectF(385.0, 30.0, 330.0, 275.0),
        "convergence_cutoff": QRectF(280.0, 370.0, 465.0, 165.0),
        "convergence_kpoint": QRectF(720.0, 70.0, 260.0, 220.0),
        "pristine_relaxation": QRectF(780.0, 360.0, 330.0, 150.0),
        "uep": QRectF(965.0, 35.0, 295.0, 210.0),
        "muon_candidates": QRectF(1288.0, 130.0, 225.0, 240.0),
        "magnetic_fields": QRectF(1070.0, 420.0, 290.0, 180.0),
        "quantum_corrections": QRectF(690.0, 590.0, 285.0, 200.0),
        "energy_barriers": QRectF(210.0, 585.0, 360.0, 195.0),
        "provenance_log": QRectF(1080.0, 675.0, 355.0, 170.0),
    }
    _CHECK_POINTS = {
        "host_structure": QPointF(294.0, 331.0),
        "supercell_editor": QPointF(660.0, 304.0),
        "convergence_cutoff": QPointF(580.0, 492.0),
        "convergence_kpoint": QPointF(982.0, 301.0),
        "pristine_relaxation": QPointF(1062.0, 492.0),
        "uep": QPointF(1223.0, 274.0),
        "muon_candidates": QPointF(1468.0, 364.0),
        "magnetic_fields": QPointF(1320.0, 578.0),
        "quantum_corrections": QPointF(913.0, 756.0),
        "energy_barriers": QPointF(450.0, 745.0),
    }
    _LOGO_COVER_REGION = QRectF(34.0, 780.0, 174.0, 190.0)
    # Banner height (logical px) at a typical window; signpost label sizes are
    # scaled relative to this so they stay proportionate as the map grows/shrinks.
    _SIGN_REF_BANNER_HEIGHT = 34.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._destinations: list[ProjectTrailDestination] = []
        self._hit_rects: dict[str, QRectF] = {}
        self._hovered_route: str | None = None
        self._logo_pixmap = QPixmap(logo_path())
        self._map_pixmap = QPixmap(project_home_map_path())
        self._color_map_pixmap = QPixmap(project_home_color_map_path())
        self.setObjectName("projectHomeTrail")
        self.setMinimumSize(720, 480)
        self.setMouseTracking(True)
        self.setAccessibleName("Project workflow home trail")

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(960, 640)

    def set_destinations(self, destinations: list[ProjectTrailDestination]) -> None:
        self._destinations = list(destinations)
        self.setAccessibleName(
            "Project workflow home trail: "
            + ", ".join(destination.title for destination in self._destinations)
        )
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if event.button() == Qt.MouseButton.LeftButton:
            point = event.position()
            for route, rect in self._destination_hit_rects().items():
                if rect.contains(point):
                    self.route_requested.emit(route)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        hovered_route = self._route_at(event.position())
        if hovered_route != self._hovered_route:
            self._hovered_route = hovered_route
            self.setCursor(Qt.CursorShape.PointingHandCursor if hovered_route else Qt.CursorShape.ArrowCursor)
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._hovered_route is not None:
            self._hovered_route = None
            self.unsetCursor()
            self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        painter.fillRect(self.rect(), QColor(palette.bg_primary))
        map_rect = self._map_rect()
        if self._map_pixmap.isNull():
            self._paint_map_background(painter, palette, map_rect)
            self._paint_terrain(painter, palette, map_rect)
            self._paint_trail_path(painter, palette, map_rect)
            for destination in self._destinations:
                self._paint_destination(painter, palette, destination, map_rect)
        else:
            self._paint_map_artwork(painter, map_rect)
        self._hit_rects = self._destination_hit_rects(map_rect)

        self._paint_hover_region(painter, palette, map_rect)
        if not self._map_pixmap.isNull():
            self._paint_sign_titles(painter, palette, map_rect)
        self._paint_completion_seals(painter, palette, map_rect)

        painter.end()

    def _map_rect(self) -> QRectF:
        available = QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -8.0)
        if available.isEmpty():
            return QRectF()
        source_region = self._active_source_region()
        source_ratio = source_region.width() / source_region.height()
        available_ratio = available.width() / max(1.0, available.height())
        if available_ratio > source_ratio:
            height = available.height()
            width = height * source_ratio
        else:
            width = available.width()
            height = width / source_ratio
        return QRectF(
            available.center().x() - width / 2.0,
            available.center().y() - height / 2.0,
            width,
            height,
        )

    def _active_source_region(self) -> QRectF:
        if self._map_pixmap.isNull():
            return QRectF(0.0, 0.0, self._SOURCE_WIDTH, self._SOURCE_HEIGHT)
        return self._ARTWORK_SOURCE_REGION

    def _map_rect_from_source(self, source_rect: QRectF, map_rect: QRectF | None = None) -> QRectF:
        target = map_rect if map_rect is not None else self._map_rect()
        source_region = self._active_source_region()
        return QRectF(
            target.left() + target.width() * ((source_rect.left() - source_region.left()) / source_region.width()),
            target.top() + target.height() * ((source_rect.top() - source_region.top()) / source_region.height()),
            target.width() * (source_rect.width() / source_region.width()),
            target.height() * (source_rect.height() / source_region.height()),
        )

    def _map_point_from_source(self, point: QPointF, map_rect: QRectF | None = None) -> QPointF:
        target = map_rect if map_rect is not None else self._map_rect()
        source_region = self._active_source_region()
        return QPointF(
            target.left() + target.width() * ((point.x() - source_region.left()) / source_region.width()),
            target.top() + target.height() * ((point.y() - source_region.top()) / source_region.height()),
        )

    def _destination_hit_rects(self, map_rect: QRectF | None = None) -> dict[str, QRectF]:
        hit_rects: dict[str, QRectF] = {}
        for destination in self._destinations:
            region = self._LANDMARK_REGIONS.get(destination.route)
            if region is None:
                region = self._SIGN_REGIONS.get(destination.route)
            if region is not None:
                hit_rects[destination.route] = self._map_rect_from_source(region, map_rect).adjusted(
                    -8.0,
                    -8.0,
                    8.0,
                    8.0,
                )
                sign_region = self._SIGN_REGIONS.get(destination.route)
                if sign_region is not None:
                    hit_rects[destination.route] = hit_rects[destination.route].united(
                        self._map_rect_from_source(sign_region, map_rect).adjusted(-6.0, -6.0, 6.0, 6.0)
                    )
        return hit_rects

    def _route_at(self, point: QPointF) -> str | None:
        for route, rect in self._destination_hit_rects().items():
            if rect.contains(point):
                return route
        return None

    def _paint_map_background(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        if map_rect.isEmpty():
            return

        painter.setPen(QPen(QColor(palette.border).darker(135), 2.0))
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawRoundedRect(map_rect, 18.0, 18.0)

        inner = map_rect.adjusted(10.0, 10.0, -10.0, -10.0)
        painter.setPen(QPen(QColor(palette.border), 1.0, Qt.PenStyle.DotLine))
        painter.setBrush(QColor(palette.bg_primary))
        painter.drawRoundedRect(inner, 12.0, 12.0)

        painter.setPen(QPen(QColor(palette.border), 1.0, Qt.PenStyle.DotLine))
        for index in range(4):
            y = map_rect.top() + map_rect.height() * (0.18 + index * 0.14)
            path = QPainterPath(QPointF(map_rect.left() + 24.0, y))
            path.cubicTo(
                QPointF(map_rect.left() + map_rect.width() * 0.28, y - 16.0),
                QPointF(map_rect.left() + map_rect.width() * 0.58, y + 18.0),
                QPointF(map_rect.right() - 24.0, y - 6.0),
            )
            painter.drawPath(path)

    def _source_rect(self, left: float, top: float, width: float, height: float, map_rect: QRectF) -> QRectF:
        return self._map_rect_from_source(QRectF(left, top, width, height), map_rect)

    def _paint_map_artwork(self, painter: QPainter, map_rect: QRectF) -> None:
        if map_rect.isEmpty():
            return
        painter.drawPixmap(map_rect, self._map_pixmap, self._active_source_region())

    def _paint_terrain(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        water = QColor("#2E7D8A" if palette.mode == THEME_LIGHT else "#2A6472")
        water.setAlpha(34 if palette.mode == THEME_LIGHT else 48)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(water)
        painter.drawEllipse(self._source_rect(-166.0, 430.0, 250.0, 190.0, map_rect))
        painter.drawEllipse(self._source_rect(1460.0, 430.0, 210.0, 212.0, map_rect))
        painter.drawEllipse(self._source_rect(740.0, 982.0, 430.0, 72.0, map_rect))

        wave_pen = QColor(palette.bg_surface)
        wave_pen.setAlpha(120)
        painter.setPen(QPen(wave_pen, 1.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for offset in (0.0, 20.0):
            wave = QPainterPath(self._map_point_from_source(QPointF(22.0, 486.0 + offset), map_rect))
            wave.cubicTo(
                self._map_point_from_source(QPointF(64.0, 462.0 + offset), map_rect),
                self._map_point_from_source(QPointF(96.0, 512.0 + offset), map_rect),
                self._map_point_from_source(QPointF(132.0, 488.0 + offset), map_rect),
            )
            painter.drawPath(wave)

        for source_rect in (
            QRectF(260.0, 72.0, 112.0, 136.0),
            QRectF(675.0, 92.0, 96.0, 86.0),
            QRectF(1298.0, 86.0, 118.0, 92.0),
            QRectF(280.0, 638.0, 250.0, 188.0),
            QRectF(1018.0, 508.0, 95.0, 76.0),
        ):
            painter.save()
            painter.setOpacity(0.28)
            self._draw_mountain_group(painter, palette, self._map_rect_from_source(source_rect, map_rect))
            painter.restore()

        for point in (
            QPointF(210.0, 410.0),
            QPointF(182.0, 556.0),
            QPointF(258.0, 585.0),
            QPointF(710.0, 862.0),
            QPointF(1260.0, 500.0),
            QPointF(1372.0, 532.0),
            QPointF(690.0, 230.0),
            QPointF(1186.0, 164.0),
        ):
            painter.save()
            painter.setOpacity(0.5)
            self._draw_tree(painter, palette, self._map_point_from_source(point, map_rect), map_rect.width() * 0.018)
            painter.restore()

    def _paint_trail_path(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        trail_points = [
            QPointF(300.0, 370.0),
            QPointF(378.0, 320.0),
            QPointF(520.0, 338.0),
            QPointF(748.0, 338.0),
            QPointF(842.0, 414.0),
            QPointF(860.0, 560.0),
            QPointF(650.0, 730.0),
            QPointF(480.0, 824.0),
            QPointF(735.0, 778.0),
            QPointF(978.0, 636.0),
            QPointF(1150.0, 628.0),
            QPointF(1230.0, 518.0),
            QPointF(1228.0, 378.0),
            QPointF(1335.0, 392.0),
        ]
        if len(trail_points) < 2:
            return

        path = QPainterPath(self._map_point_from_source(trail_points[0], map_rect))
        for start, end in zip(trail_points, trail_points[1:]):
            start_point = self._map_point_from_source(start, map_rect)
            end_point = self._map_point_from_source(end, map_rect)
            mid_x = (start_point.x() + end_point.x()) / 2.0
            path.cubicTo(QPointF(mid_x, start_point.y()), QPointF(mid_x, end_point.y()), end_point)

        painter.setPen(
            QPen(
                QColor(palette.text_muted).darker(130),
                max(2.0, map_rect.width() * 0.008),
                Qt.PenStyle.DotLine,
                Qt.PenCapStyle.RoundCap,
            )
        )
        painter.drawPath(path)

    def _draw_mountain_group(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.1))
        painter.setBrush(QColor(palette.bg_surface_alt))
        for x_frac, peak_frac, width_frac in ((0.0, 0.18, 0.48), (0.28, 0.02, 0.52), (0.58, 0.24, 0.42)):
            left = rect.left() + rect.width() * x_frac
            width = rect.width() * width_frac
            peak = QPointF(left + width * 0.5, rect.top() + rect.height() * peak_frac)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(left, rect.bottom()),
                        peak,
                        QPointF(left + width, rect.bottom()),
                    ]
                )
            )
            painter.setPen(QPen(QColor(palette.bg_surface), 1.0))
            painter.drawLine(peak, QPointF(peak.x() - width * 0.12, peak.y() + rect.height() * 0.18))
            painter.drawLine(peak, QPointF(peak.x() + width * 0.14, peak.y() + rect.height() * 0.22))
            painter.setPen(QPen(QColor(palette.border), 1.1))

    def _draw_tree(self, painter: QPainter, palette, base: QPointF, size: float) -> None:
        painter.setPen(QPen(QColor(palette.border).darker(135), 1.0))
        painter.setBrush(QColor("#5D9A57" if palette.mode == THEME_LIGHT else "#3D7661"))
        for index in range(3):
            y = base.y() - size * (0.8 + index * 0.55)
            half_width = size * (0.75 - index * 0.12)
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(base.x() - half_width, y + size * 0.7),
                        QPointF(base.x(), y),
                        QPointF(base.x() + half_width, y + size * 0.7),
                    ]
                )
            )
        painter.setPen(QPen(QColor(palette.border).darker(150), 1.4))
        painter.drawLine(base, QPointF(base.x(), base.y() - size * 0.7))

    def _paint_logo_overlay(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        cover_rect = self._map_rect_from_source(self._LOGO_COVER_REGION, map_rect)
        painter.setPen(QPen(QColor(palette.border).darker(135), 1.2))
        painter.setBrush(QColor("#D6AE6B"))
        painter.drawEllipse(cover_rect.adjusted(6.0, 10.0, -6.0, -10.0))

        if self._logo_pixmap.isNull():
            return
        logo_rect = cover_rect.adjusted(32.0, 40.0, -32.0, -40.0)
        painter.drawPixmap(logo_rect, self._logo_pixmap, QRectF(self._logo_pixmap.rect()))

    def _paint_hover_region(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        if self._hovered_route is None:
            return
        rect = self._hit_rects.get(self._hovered_route)
        if rect is None:
            return
        if not self._map_pixmap.isNull() and not self._color_map_pixmap.isNull() and not map_rect.isEmpty():
            self._paint_colored_destination_artwork(painter, map_rect, self._hovered_route)

        sign_source = self._SIGN_REGIONS.get(self._hovered_route)
        if sign_source is None:
            return
        rect = self._map_rect_from_source(sign_source, map_rect)
        highlight_offset = self._SIGN_HIGHLIGHT_OFFSETS.get(self._hovered_route)
        if highlight_offset is not None:
            scale = rect.height() / 46.0
            rect.translate(highlight_offset[0] * scale, highlight_offset[1] * scale)

        painter.save()
        clip_path = QPainterPath()
        clip_path.addRoundedRect(rect, 6.0, 6.0)
        painter.setClipPath(clip_path)

        glow = QColor(palette.seal_gold)
        glow.setAlpha(70)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRoundedRect(rect, 6.0, 6.0)

        wash = QColor("#FFFFFF")
        wash.setAlpha(46)
        painter.setBrush(wash)
        painter.drawRoundedRect(rect.adjusted(3.0, 3.0, -3.0, -3.0), 4.5, 4.5)
        painter.restore()

    def _paint_colored_destination_artwork(self, painter: QPainter, map_rect: QRectF, route: str) -> None:
        reveal_path = QPainterPath()
        landmark_source = self._LANDMARK_REGIONS.get(route)
        if landmark_source is not None:
            landmark_rect = self._map_rect_from_source(landmark_source, map_rect).adjusted(-8.0, -8.0, 8.0, 8.0)
            radius = min(28.0, max(12.0, landmark_rect.height() * 0.18))
            reveal_path.addRoundedRect(landmark_rect, radius, radius)

        sign_source = self._SIGN_REGIONS.get(route)
        if sign_source is not None:
            sign_rect = self._map_rect_from_source(sign_source, map_rect).adjusted(-4.0, -4.0, 4.0, 4.0)
            reveal_path.addRoundedRect(sign_rect, 8.0, 8.0)

        if reveal_path.isEmpty():
            return

        painter.save()
        painter.setClipPath(reveal_path)
        painter.drawPixmap(map_rect, self._color_map_pixmap, self._active_source_region())
        painter.restore()

    def _paint_sign_titles(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        for destination in self._destinations:
            sign_source = self._SIGN_REGIONS.get(destination.route)
            if sign_source is None:
                continue
            sign_rect = self._map_rect_from_source(sign_source, map_rect)
            self._draw_sign_title(painter, palette, sign_rect, destination.title, destination.route)

    @staticmethod
    def _workflow_scroll_label_color() -> QColor:
        return QColor(light_palette().text_primary)

    @classmethod
    def _sign_title_pixel_size(cls, route: str, title: str, banner_height: float) -> int:
        """Pixel size for a signpost label, scaled to the rendered banner.

        A fixed size made labels look generous on small windows but shrink
        to near-illegible on large or high-DPI displays. Tying the size to the
        banner height keeps the same proportion at every window size. The
        reference height matches the banner at a typical window, so the
        already-tuned default appearance is preserved; the result is floored for
        legibility and capped so it never overwhelms its banner.
        """

        base = float(cls._SIGN_TEXT_MAX_SIZES.get(route, 13 if len(title) <= 16 else 11))
        scale = banner_height / cls._SIGN_REF_BANNER_HEIGHT
        return round(max(11.0, min(base * scale, base * 2.0)))

    def _draw_sign_title(self, painter: QPainter, palette, rect: QRectF, title: str, route: str) -> None:
        left, top, right, bottom = self._SIGN_TEXT_INSETS.get(route, (18.0, 5.0, 20.0, 8.0))
        text_rect = rect.adjusted(left, top, -right, -bottom)
        offset_x, offset_y = self._SIGN_TEXT_OFFSETS.get(route, (0.0, 0.0))
        offset_scale = rect.height() / 46.0
        text_rect.translate(offset_x * offset_scale, offset_y * offset_scale)

        font = QFont(heading_font_family())
        font.setBold(True)
        font.setPixelSize(self._sign_title_pixel_size(route, title, rect.height()))
        while font.pixelSize() > 8 and QFontMetricsF(font).horizontalAdvance(title) > text_rect.width() - 8.0:
            font.setPixelSize(font.pixelSize() - 1)

        painter.setFont(font)
        painter.setPen(self._workflow_scroll_label_color())
        painter.drawText(
            text_rect,
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextSingleLine),
            title,
        )

    def _paint_completion_seals(self, painter: QPainter, palette, map_rect: QRectF) -> None:
        completed_routes = {destination.route for destination in self._destinations if destination.completed}
        for route in completed_routes:
            source_point = self._CHECK_POINTS.get(route)
            if source_point is None:
                continue
            center = self._map_point_from_source(source_point, map_rect)
            self._draw_completion_seal(painter, palette, center)

    def _paint_destination(
        self,
        painter: QPainter,
        palette,
        destination: ProjectTrailDestination,
        map_rect: QRectF,
    ) -> None:
        landmark_source = self._LANDMARK_REGIONS.get(destination.route)
        sign_source = self._SIGN_REGIONS.get(destination.route)
        if landmark_source is None or sign_source is None:
            return

        landmark_rect = self._map_rect_from_source(landmark_source, map_rect)
        sign_rect = self._map_rect_from_source(sign_source, map_rect)

        self._draw_scaled_landmark(painter, palette, destination.theme, landmark_rect)
        self._draw_sign(painter, palette, sign_rect, destination.title)

    def _draw_scaled_landmark(self, painter: QPainter, palette, theme: str, rect: QRectF) -> None:
        if rect.isEmpty():
            return
        painter.save()
        painter.translate(rect.left(), rect.top())
        painter.scale(rect.width() / 68.0, rect.height() / 54.0)
        self._draw_landmark(painter, palette, theme, QRectF(0.0, 0.0, 68.0, 54.0))
        painter.restore()

    def _draw_sign(self, painter: QPainter, palette, rect: QRectF, title: str) -> None:
        post_x = rect.center().x() - 3.0
        painter.setPen(QPen(QColor(palette.border).darker(145), 1.0))
        painter.setBrush(QColor(palette.border).darker(115))
        painter.drawRoundedRect(QRectF(post_x, rect.bottom() - 2.0, 6.0, 28.0), 2.0, 2.0)

        painter.setPen(QPen(QColor(palette.border).darker(140), 1.1))
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        font = QFont(body_font_family())
        font.setPixelSize(8 if len(title) > 16 else 9)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(palette.text_primary))
        painter.drawText(
            rect.adjusted(7.0, 3.0, -7.0, -3.0),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap),
            title,
        )

    def _draw_completion_seal(self, painter: QPainter, palette, center: QPointF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.1))
        painter.setBrush(QColor(palette.seal_gold))
        painter.drawEllipse(center, 14.0, 14.0)
        painter.setPen(QPen(QColor(palette.bg_surface), 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(center.x() - 6.0, center.y()), QPointF(center.x() - 2.0, center.y() + 5.0))
        painter.drawLine(QPointF(center.x() - 2.0, center.y() + 5.0), QPointF(center.x() + 7.0, center.y() - 6.0))

    def _draw_landmark(self, painter: QPainter, palette, theme: str, rect: QRectF) -> None:
        if theme == "house":
            self._draw_house(painter, palette, rect)
        elif theme == "castle":
            self._draw_castle(painter, palette, rect)
        elif theme == "ocean":
            self._draw_ocean(painter, palette, rect)
        elif theme == "cube":
            self._draw_cube(painter, palette, rect)
        elif theme == "hammock":
            self._draw_hammock(painter, palette, rect)
        elif theme == "storm":
            self._draw_storm(painter, palette, rect)
        elif theme == "muon":
            self._draw_muon(painter, palette, rect)
        elif theme == "magnet":
            self._draw_magnet(painter, palette, rect)
        elif theme == "cat":
            self._draw_cat(painter, palette, rect)
        elif theme == "mountains":
            self._draw_mountains(painter, palette, rect)
        elif theme == "journal":
            self._draw_journal(painter, palette, rect)

    def _draw_house(self, painter: QPainter, palette, rect: QRectF) -> None:
        body = QRectF(rect.left() + 14, rect.top() + 24, 40, 26)
        roof = QPolygonF(
            [
                QPointF(body.left() - 4, body.top() + 2),
                QPointF(body.center().x(), rect.top() + 8),
                QPointF(body.right() + 4, body.top() + 2),
            ]
        )
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawRect(body)
        painter.setBrush(QColor(palette.accent))
        painter.drawPolygon(roof)
        painter.setBrush(QColor(palette.bg_surface_alt))
        painter.drawRect(QRectF(body.center().x() - 5, body.bottom() - 15, 10, 15))

    def _draw_castle(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface_alt))
        for x in (rect.left() + 10, rect.center().x() - 10, rect.right() - 30):
            painter.drawRect(QRectF(x, rect.top() + 14, 20, 36))
        painter.drawRect(QRectF(rect.left() + 16, rect.top() + 28, 36, 22))
        painter.setBrush(QColor(palette.seal_gold))
        for x in (rect.left() + 10, rect.center().x() - 10, rect.right() - 30):
            painter.drawRect(QRectF(x, rect.top() + 8, 20, 8))

    def _draw_ocean(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.accent), 2.3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        for index in range(4):
            y = rect.top() + 16 + index * 8
            path = QPainterPath(QPointF(rect.left() + 8, y))
            path.cubicTo(
                QPointF(rect.left() + 22, y - 10),
                QPointF(rect.left() + 34, y + 10),
                QPointF(rect.left() + 52, y),
            )
            painter.drawPath(path)

    def _draw_cube(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.1))
        colors = (palette.accent, palette.seal_gold, palette.bg_surface_alt)
        size = 12
        left = rect.left() + 14
        top = rect.top() + 10
        for row in range(3):
            for col in range(3):
                painter.setBrush(QColor(colors[(row + col) % len(colors)]))
                painter.drawRect(QRectF(left + col * size, top + row * size, size, size))

    def _draw_hammock(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border).darker(140), 2.0))
        painter.drawLine(QPointF(rect.left() + 12, rect.bottom() - 6), QPointF(rect.left() + 20, rect.top() + 6))
        painter.drawLine(QPointF(rect.right() - 12, rect.bottom() - 6), QPointF(rect.right() - 20, rect.top() + 6))
        painter.setPen(QPen(QColor(palette.accent), 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        path = QPainterPath(QPointF(rect.left() + 19, rect.top() + 17))
        path.cubicTo(
            QPointF(rect.left() + 28, rect.top() + 42),
            QPointF(rect.right() - 28, rect.top() + 42),
            QPointF(rect.right() - 19, rect.top() + 17),
        )
        painter.drawPath(path)

    def _draw_storm(self, painter: QPainter, palette, rect: QRectF) -> None:
        cloud = QColor(palette.bg_surface_alt)
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(cloud)
        painter.drawEllipse(QRectF(rect.left() + 10, rect.top() + 16, 26, 22))
        painter.drawEllipse(QRectF(rect.left() + 26, rect.top() + 9, 26, 28))
        painter.drawEllipse(QRectF(rect.left() + 38, rect.top() + 18, 20, 20))
        painter.setBrush(QColor(palette.seal_gold))
        bolt = QPolygonF(
            [
                QPointF(rect.center().x(), rect.top() + 30),
                QPointF(rect.center().x() - 8, rect.bottom()),
                QPointF(rect.center().x() + 2, rect.bottom() - 11),
                QPointF(rect.center().x() - 2, rect.bottom() - 11),
                QPointF(rect.center().x() + 8, rect.top() + 30),
            ]
        )
        painter.drawPolygon(bolt)

    def _draw_muon(self, painter: QPainter, palette, rect: QRectF) -> None:
        font = QFont(body_font_family())
        font.setPixelSize(34)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(palette.accent))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), "mu")

    def _draw_magnet(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.accent), 8.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(QRectF(rect.left() + 14, rect.top() + 10, 40, 42), 0, 180 * 16)
        painter.setPen(QPen(QColor(palette.border), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(QPointF(rect.left() + 15, rect.top() + 30), QPointF(rect.left() + 15, rect.bottom() - 4))
        painter.drawLine(QPointF(rect.right() - 15, rect.top() + 30), QPointF(rect.right() - 15, rect.bottom() - 4))

    def _draw_cat(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface_alt))
        face = QRectF(rect.left() + 16, rect.top() + 16, 36, 30)
        painter.drawEllipse(face)
        ears = QPolygonF(
            [
                QPointF(face.left() + 4, face.top() + 7),
                QPointF(face.left() + 10, face.top() - 8),
                QPointF(face.left() + 17, face.top() + 6),
                QPointF(face.right() - 17, face.top() + 6),
                QPointF(face.right() - 10, face.top() - 8),
                QPointF(face.right() - 4, face.top() + 7),
            ]
        )
        painter.drawPolygon(ears)
        painter.setBrush(QColor(palette.text_primary))
        painter.drawEllipse(QPointF(face.left() + 13, face.top() + 14), 2.0, 2.0)
        painter.drawEllipse(QPointF(face.right() - 13, face.top() + 14), 2.0, 2.0)

    def _draw_mountains(self, painter: QPainter, palette, rect: QRectF) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface_alt))
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(rect.left() + 4, rect.bottom() - 4),
                    QPointF(rect.left() + 24, rect.top() + 8),
                    QPointF(rect.left() + 42, rect.bottom() - 4),
                ]
            )
        )
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(rect.left() + 28, rect.bottom() - 4),
                    QPointF(rect.left() + 48, rect.top() + 14),
                    QPointF(rect.right() - 4, rect.bottom() - 4),
                ]
            )
        )

    def _draw_journal(self, painter: QPainter, palette, rect: QRectF) -> None:
        book = QRectF(rect.left() + 16, rect.top() + 8, 36, 42)
        painter.setPen(QPen(QColor(palette.border), 1.2))
        painter.setBrush(QColor(palette.bg_surface))
        painter.drawRoundedRect(book, 4.0, 4.0)
        painter.setPen(QPen(QColor(palette.accent), 2.0))
        painter.drawLine(QPointF(book.left() + 8, book.top() + 7), QPointF(book.left() + 8, book.bottom() - 7))
        painter.setPen(QPen(QColor(palette.text_muted), 1.0))
        for index in range(3):
            y = book.top() + 14 + index * 8
            painter.drawLine(QPointF(book.left() + 15, y), QPointF(book.right() - 7, y))


class ConstructionSignWidget(QWidget):
    """Paint an in-progress workflow placeholder as a planted wooden sign."""

    _MESSAGE = "Under construction.\nCome back soon."

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(420, 300)
        self.setAccessibleName(self._MESSAGE.replace("\n", " "))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = max(1, self.width())
        height = max(1, self.height())
        horizon = int(height * 0.72)
        painter.fillRect(self.rect(), QColor(palette.bg_primary))
        painter.fillRect(0, horizon, width, height - horizon, QColor(palette.bg_surface_alt))

        sign_width = int(min(560, max(260, width - 96)))
        sign_height = 144
        sign_x = int((width - sign_width) / 2)
        sign_y = int(max(36, min(horizon - sign_height - 58, height * 0.24)))
        post_top = sign_y + sign_height - 10
        post_bottom = min(height - 22, horizon + 104)

        post_width = 34
        post_x = int(width / 2 - post_width / 2)
        painter.setPen(QPen(QColor("#5C341E"), 1.2))
        painter.setBrush(QColor("#7A4728"))
        painter.drawRoundedRect(QRectF(post_x, post_top, post_width, post_bottom - post_top), 5, 5)
        painter.setPen(QPen(QColor("#A66A3A"), 1.0))
        painter.drawLine(post_x + 8, post_top + 8, post_x + 8, post_bottom - 8)

        plank_height = sign_height / 3
        for index, colour in enumerate(("#9B5F32", "#B3723D", "#8A512C")):
            plank_y = sign_y + index * plank_height
            painter.setPen(QPen(QColor("#5F371F"), 1.4))
            painter.setBrush(QColor(colour))
            painter.drawRoundedRect(QRectF(sign_x, plank_y, sign_width, plank_height + 2), 10, 10)
            painter.setPen(QPen(QColor("#C88B55"), 1.0))
            painter.drawLine(
                sign_x + 24,
                int(plank_y + plank_height * 0.46),
                sign_x + sign_width - 24,
                int(plank_y + plank_height * 0.46),
            )

        painter.setPen(QPen(QColor("#5F371F"), 1.2))
        painter.setBrush(QColor("#D2A169"))
        for nail_x in (sign_x + 26, sign_x + sign_width - 26):
            for nail_y in (sign_y + 22, sign_y + sign_height - 22):
                painter.drawEllipse(nail_x - 3, nail_y - 3, 6, 6)

        font = QFont(body_font_family())
        font.setPixelSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor("#FFF8EA"), 1.0))
        painter.drawText(
            QRectF(sign_x + 28, sign_y + 18, sign_width - 56, sign_height - 36),
            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
            self._MESSAGE,
        )
        painter.end()

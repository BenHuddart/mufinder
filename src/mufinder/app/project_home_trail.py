"""Project home page: the expedition chart, its field notes and the cell survey.

``WorkflowDefinition`` describes the workflow tree the project navigator is
built from. ``ProjectHomeChartWidget`` draws the workflow stages as waypoints
joined by their dependencies (solid when required, dashed when recommended,
dotted when optional), marks each stage's progress and the stage the user last
worked on, and emits the stage the user picks. ``FieldNotesPanel`` shows that
stage's facts, what it needs and a link to open it. ``ProjectHomeWidget`` lays
the three parts out with the cell survey from
:mod:`mufinder.app.cell_survey_widget`. ``ConstructionSignWidget`` stands in for
a workflow page that is not yet available.

The stages, dependencies and progress come from
:mod:`mufinder.services.project_summary`; this module only draws them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeyEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.cell_survey_widget import CellSurveyWidget
from mufinder.app.theme import (
    body_font_family,
    current_palette,
    heading_font_family,
    style_hint_text,
    style_section_title,
    style_warning_text,
)
from mufinder.services.project_summary import (
    STAGE_DEPENDENCIES,
    STAGE_GROUPS,
    DependencyKind,
    ProjectSummary,
    StageStatus,
    StageSummary,
    group_stage_keys,
    stage_info,
    summarize_project,
)
from mufinder.services.project_survey import discovered_sites_layer, survey_layers
from mufinder.services.project_document import active_host_structure

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    """Describe a workflow node and its completion rule."""

    key: str
    title: str
    route: str | None = None
    completion_goal: str | None = None
    completion_check: Callable[[ProjectDocument], bool] | None = None
    children: tuple["WorkflowDefinition", ...] = ()


_STATUS_TEXT = {
    StageStatus.COMPLETE: "Complete",
    StageStatus.IN_PROGRESS: "In progress",
    StageStatus.NOT_STARTED: "Not started",
}


class ProjectHomeChartWidget(QWidget):
    """Draw the workflow stages as an explorer's chart and let the user pick one.

    A click selects a stage (``stage_selected``); a double-click, Return or
    Enter opens it (``route_requested``). The arrow keys walk the stages in
    workflow order.
    """

    stage_selected = Signal(str)
    route_requested = Signal(str)

    #: Where each stage sits, as fractions of the chart's drawing area.
    NODE_POSITIONS: dict[str, tuple[float, float]] = {
        "host_structure": (0.05, 0.42),
        "convergence_cutoff": (0.22, 0.22),
        "convergence_kpoint": (0.22, 0.62),
        "pristine_relaxation": (0.37, 0.86),
        "uep": (0.40, 0.12),
        "muon_candidates": (0.52, 0.42),
        "muon_relaxations": (0.635, 0.42),
        "muon_analysis": (0.75, 0.42),
        "hyperfine_coupling": (0.865, 0.27),
        "magnetic_fields": (0.80, 0.12),
        "polarisation": (0.94, 0.12),
        "quantum_corrections": (0.94, 0.42),
        "energy_barriers": (0.93, 0.80),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._summary: ProjectSummary | None = None
        self._selected: str | None = None
        self._hovered: str | None = None
        self.setObjectName("projectHomeChart")
        self.setMinimumSize(560, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Project workflow chart")

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(960, 340)

    # --- state ------------------------------------------------------------------

    def set_summary(self, summary: ProjectSummary | None) -> None:
        self._summary = summary
        keys = {stage.key for stage in summary.stages} if summary is not None else set()
        if self._selected not in keys:
            self._selected = summary.current_stage if summary is not None else None
        self._update_accessible_name()
        self.update()

    def summary(self) -> ProjectSummary | None:
        return self._summary

    def selected_stage(self) -> str | None:
        return self._selected

    def select_stage(self, key: str) -> None:
        if self._summary is None or self._summary.stage(key) is None or key == self._selected:
            return
        self._selected = key
        self._update_accessible_name()
        self.update()
        self.stage_selected.emit(key)

    def stage_at(self, point: QPointF) -> str | None:
        radius = self._node_radius() + 6.0
        for key, center in self._node_centers().items():
            if math.hypot(point.x() - center.x(), point.y() - center.y()) <= radius:
                return key
            if self._label_rect(key, center).contains(point):
                return key
        return None

    def node_center(self, key: str) -> QPointF:
        return self._node_centers()[key]

    # --- events -----------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if event.button() == Qt.MouseButton.LeftButton:
            key = self.stage_at(event.position())
            if key is not None:
                self.select_stage(key)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        key = (
            self.stage_at(event.position()) if event.button() == Qt.MouseButton.LeftButton else None
        )
        if key is not None:
            self.select_stage(key)
            self.route_requested.emit(key)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        key = self.stage_at(event.position())
        if key != self._hovered:
            self._hovered = key
            self.setCursor(Qt.CursorShape.PointingHandCursor if key else Qt.CursorShape.ArrowCursor)
            self.update()
        stage = self._summary.stage(key) if self._summary is not None and key else None
        if stage is not None:
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"{stage.title}: {_STATUS_TEXT[stage.status]}",
                self,
            )
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._hovered is not None:
            self._hovered = None
            self.unsetCursor()
            self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API naming.
        if self._summary is None or self._selected is None:
            super().keyPressEvent(event)
            return
        keys = [stage.key for stage in self._summary.stages]
        index = keys.index(self._selected)
        if event.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.select_stage(keys[min(len(keys) - 1, index + 1)])
        elif event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.select_stage(keys[max(0, index - 1)])
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.route_requested.emit(self._selected)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    # --- geometry ---------------------------------------------------------------

    def _chart_rect(self) -> QRectF:
        margin_x = max(36.0, self.width() * 0.035)
        return QRectF(self.rect()).adjusted(margin_x, 30.0, -margin_x, -46.0)

    def _node_radius(self) -> float:
        return max(9.0, min(15.0, min(self.width(), self.height()) * 0.034))

    def _node_centers(self) -> dict[str, QPointF]:
        area = self._chart_rect()
        stages = self._summary.stages if self._summary is not None else ()
        return {
            stage.key: QPointF(
                area.left() + area.width() * self.NODE_POSITIONS[stage.key][0],
                area.top() + area.height() * self.NODE_POSITIONS[stage.key][1],
            )
            for stage in stages
            if stage.key in self.NODE_POSITIONS
        }

    def _label_font(self, *, bold: bool = False) -> QFont:
        font = QFont(body_font_family())
        font.setPixelSize(round(max(11.0, min(14.0, self.height() * 0.036))))
        font.setBold(bold)
        return font

    def _label_rect(self, key: str, center: QPointF) -> QRectF:
        title = (
            self._summary.stage(key).title
            if self._summary is not None and self._summary.stage(key)
            else key
        )
        metrics = QFontMetricsF(self._label_font(bold=True))
        width = metrics.horizontalAdvance(title) + 8.0
        top = center.y() + self._node_radius() + 4.0
        left = min(max(2.0, center.x() - width / 2.0), self.width() - width - 2.0)
        return QRectF(left, top, width, metrics.height() + 2.0)

    def _group_rects(self, centers: dict[str, QPointF]) -> dict[str, QRectF]:
        rects: dict[str, QRectF] = {}
        for group in STAGE_GROUPS:
            members = [key for key in group_stage_keys(group.key) if key in centers]
            if not members:
                continue
            rect = QRectF()
            for key in members:
                center = centers[key]
                radius = self._node_radius()
                node_rect = QRectF(center.x() - radius, center.y() - radius, 2 * radius, 2 * radius)
                rect = rect.united(node_rect.united(self._label_rect(key, center)))
            rect = rect.adjusted(-16.0, -26.0, 16.0, 10.0)
            title_width = QFontMetricsF(self._group_font()).horizontalAdvance(group.title) + 28.0
            if rect.width() < title_width:
                rect.adjust(
                    -(title_width - rect.width()) / 2.0,
                    0.0,
                    (title_width - rect.width()) / 2.0,
                    0.0,
                )
            rects[group.key] = rect
        return rects

    @staticmethod
    def _group_fill(palette) -> QColor:
        fill = QColor(palette.bg_surface)
        fill.setAlpha(150)
        return fill

    def _group_font(self) -> QFont:
        font = QFont(heading_font_family())
        font.setItalic(True)
        font.setPixelSize(round(max(11.0, min(15.0, self.height() * 0.04))))
        return font

    def _anchor(
        self, key: str, centers: dict[str, QPointF], groups: dict[str, QRectF], *, outgoing: bool
    ) -> QPointF | None:
        if key in centers:
            return centers[key]
        rect = groups.get(key)
        if rect is None:
            return None
        return QPointF(rect.right() if outgoing else rect.left(), rect.center().y())

    # --- painting ---------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(palette.bg_primary))
        if self._summary is None:
            painter.end()
            return
        centers = self._node_centers()
        groups = self._group_rects(centers)
        self._paint_groups(painter, palette, groups)
        self._paint_edges(painter, palette, centers, groups)
        for stage in self._summary.stages:
            if stage.key in centers:
                self._paint_node(painter, palette, stage, centers[stage.key])
        self._paint_legend(painter, palette)
        painter.end()

    def _paint_groups(self, painter: QPainter, palette, groups: dict[str, QRectF]) -> None:
        font = self._group_font()
        for group in STAGE_GROUPS:
            rect = groups.get(group.key)
            if rect is None:
                continue
            border = QColor(palette.border)
            painter.setPen(QPen(border, 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(self._group_fill(palette))
            painter.drawRoundedRect(rect, 14.0, 14.0)
            painter.setFont(font)
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(
                rect.adjusted(12.0, 4.0, -12.0, 0.0),
                int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft),
                group.title,
            )

    def _paint_edges(
        self, painter: QPainter, palette, centers: dict[str, QPointF], groups: dict[str, QRectF]
    ) -> None:
        assert self._summary is not None
        statuses = {stage.key: stage.status for stage in self._summary.stages}
        for dependency in STAGE_DEPENDENCIES:
            start = self._anchor(dependency.source, centers, groups, outgoing=True)
            end = self._anchor(dependency.target, centers, groups, outgoing=False)
            if start is None or end is None:
                continue
            travelled = (
                statuses.get(dependency.target, StageStatus.NOT_STARTED) != StageStatus.NOT_STARTED
            )
            colour = QColor(palette.text_muted if travelled else palette.border)
            width = 2.2 if dependency.kind == DependencyKind.REQUIRED else 1.6
            pen = QPen(colour, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            if dependency.kind == DependencyKind.RECOMMENDED:
                pen.setDashPattern([5.0, 3.5])
            elif dependency.kind == DependencyKind.OPTIONAL:
                pen.setDashPattern([0.4, 3.2])
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            mid_x = (start.x() + end.x()) / 2.0
            path = QPainterPath(start)
            path.cubicTo(QPointF(mid_x, start.y()), QPointF(mid_x, end.y()), end)
            painter.drawPath(path)

    def _paint_node(self, painter: QPainter, palette, stage: StageSummary, center: QPointF) -> None:
        radius = self._node_radius()
        if stage.key in (self._selected, self._hovered):
            halo = QColor(palette.accent)
            halo.setAlpha(80 if stage.key == self._selected else 40)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(halo)
            painter.drawEllipse(center, radius + 6.0, radius + 6.0)

        if stage.status == StageStatus.COMPLETE:
            painter.setPen(QPen(QColor(palette.border), 1.1))
            painter.setBrush(QColor(palette.seal_gold))
            painter.drawEllipse(center, radius, radius)
            tick = QPen(
                QColor(palette.bg_surface),
                max(2.0, radius * 0.18),
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
            )
            painter.setPen(tick)
            scale = radius / 14.0
            painter.drawLine(
                QPointF(center.x() - 6.0 * scale, center.y()),
                QPointF(center.x() - 2.0 * scale, center.y() + 5.0 * scale),
            )
            painter.drawLine(
                QPointF(center.x() - 2.0 * scale, center.y() + 5.0 * scale),
                QPointF(center.x() + 7.0 * scale, center.y() - 6.0 * scale),
            )
        elif stage.status == StageStatus.IN_PROGRESS:
            painter.setPen(QPen(QColor(palette.accent), 2.2))
            painter.setBrush(QColor(palette.bg_surface))
            painter.drawEllipse(center, radius, radius)
            half = QPainterPath(center)
            inner = QRectF(
                center.x() - radius * 0.62, center.y() - radius * 0.62, radius * 1.24, radius * 1.24
            )
            half.arcTo(inner, 90.0, -180.0)
            half.closeSubpath()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(palette.accent))
            painter.drawPath(half)
        else:
            ring = QColor(palette.text_muted)
            if stage.unmet_requirements:
                ring.setAlpha(140)
            pen = QPen(ring, 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            pen.setDashPattern([1.5, 2.5])
            painter.setPen(pen)
            painter.setBrush(QColor(palette.bg_primary))
            painter.drawEllipse(center, radius, radius)

        if stage.active_jobs:
            self._paint_job_badge(painter, palette, center, radius, stage.active_jobs)
        if self._summary is not None and stage.key == self._summary.current_stage:
            self._paint_flag(painter, palette, center, radius)

        selected = stage.key == self._selected
        label_rect = self._label_rect(stage.key, center)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(palette.bg_primary))
        painter.drawRoundedRect(label_rect, 4.0, 4.0)
        if stage_info(stage.key).group is not None:
            painter.setBrush(self._group_fill(palette))
            painter.drawRoundedRect(label_rect, 4.0, 4.0)
        painter.setFont(self._label_font(bold=selected))
        label_colour = QColor(
            palette.text_primary if stage.status != StageStatus.NOT_STARTED else palette.text_muted
        )
        painter.setPen(label_colour)
        painter.drawText(
            label_rect,
            int(
                Qt.AlignmentFlag.AlignHCenter
                | Qt.AlignmentFlag.AlignTop
                | Qt.TextFlag.TextSingleLine
            ),
            stage.title,
        )

    def _paint_job_badge(
        self, painter: QPainter, palette, center: QPointF, radius: float, count: int
    ) -> None:
        badge_center = QPointF(center.x() + radius * 0.85, center.y() - radius * 0.85)
        badge_radius = max(6.5, radius * 0.5)
        painter.setPen(QPen(QColor(palette.bg_primary), 1.5))
        painter.setBrush(QColor(palette.status_running))
        painter.drawEllipse(badge_center, badge_radius, badge_radius)
        font = QFont(body_font_family())
        font.setPixelSize(max(9, round(badge_radius * 1.2)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(
            QRectF(
                badge_center.x() - badge_radius,
                badge_center.y() - badge_radius,
                2 * badge_radius,
                2 * badge_radius,
            ),
            int(Qt.AlignmentFlag.AlignCenter),
            str(count) if count < 100 else "99+",
        )

    def _paint_flag(self, painter: QPainter, palette, center: QPointF, radius: float) -> None:
        base = QPointF(center.x() - radius * 0.7, center.y() - radius * 0.7)
        top = QPointF(base.x(), base.y() - radius * 1.5)
        painter.setPen(
            QPen(QColor(palette.text_primary), 1.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        painter.drawLine(base, top)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(palette.accent))
        painter.drawPolygon(
            QPolygonF(
                [
                    top,
                    QPointF(top.x() - radius * 0.95, top.y() + radius * 0.32),
                    QPointF(top.x(), top.y() + radius * 0.64),
                ]
            )
        )

    def _paint_legend(self, painter: QPainter, palette) -> None:
        font = QFont(body_font_family())
        font.setPixelSize(11)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        x = 14.0
        y = self.height() - 14.0
        for kind, text in (
            (DependencyKind.REQUIRED, "required"),
            (DependencyKind.RECOMMENDED, "recommended"),
            (DependencyKind.OPTIONAL, "optional route"),
        ):
            pen = QPen(QColor(palette.text_muted), 2.0 if kind == DependencyKind.REQUIRED else 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            if kind == DependencyKind.RECOMMENDED:
                pen.setDashPattern([5.0, 3.5])
            elif kind == DependencyKind.OPTIONAL:
                pen.setDashPattern([0.4, 3.2])
            painter.setPen(pen)
            painter.drawLine(QPointF(x, y - 4.0), QPointF(x + 24.0, y - 4.0))
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(QPointF(x + 30.0, y), text)
            x += 30.0 + metrics.horizontalAdvance(text) + 18.0

    def _update_accessible_name(self) -> None:
        stage = (
            self._summary.stage(self._selected)
            if self._summary is not None and self._selected
            else None
        )
        if stage is None:
            self.setAccessibleName("Project workflow chart")
            return
        self.setAccessibleName(
            f"Project workflow chart: {stage.title}, {_STATUS_TEXT[stage.status].lower()}"
        )


class FieldNotesPanel(QFrame):
    """Show one stage's progress, facts and needs, with a link that opens it."""

    route_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("overviewCard")
        self._stage: StageSummary | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        caption = QLabel("Field notes", self)
        style_hint_text(caption)
        layout.addWidget(caption)
        self._title = QLabel(self)
        style_section_title(self._title)
        layout.addWidget(self._title)
        self._status = QLabel(self)
        self._status.setObjectName("fieldNotesStatus")
        layout.addWidget(self._status)
        self._facts = QLabel(self)
        self._facts.setObjectName("cardBody")
        self._facts.setWordWrap(True)
        self._facts.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self._facts)
        self._needs = QLabel(self)
        style_warning_text(self._needs)
        self._needs.setWordWrap(True)
        layout.addWidget(self._needs)
        self._guidance = QLabel(self)
        style_hint_text(self._guidance)
        self._guidance.setWordWrap(True)
        layout.addWidget(self._guidance)
        layout.addStretch(1)
        self._open_button = QPushButton(self)
        self._open_button.setObjectName("fieldNotesOpenButton")
        self._open_button.clicked.connect(self._on_open)
        layout.addWidget(self._open_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.set_stage(None)

    def set_stage(self, stage: StageSummary | None) -> None:
        self._stage = stage
        visible = stage is not None
        for widget in (self._title, self._status, self._facts, self._guidance, self._open_button):
            widget.setVisible(visible)
        if stage is None:
            self._needs.setVisible(False)
            return
        self._title.setText(stage.title)
        status = _STATUS_TEXT[stage.status]
        if stage.active_jobs:
            status += f" · {stage.active_jobs} job{'s' if stage.active_jobs != 1 else ''} active"
        self._status.setText(status)
        self._facts.setText("\n".join(stage.facts))
        self._needs.setText(
            f"Needs: {', '.join(stage.unmet_requirements)}" if stage.unmet_requirements else ""
        )
        self._needs.setVisible(bool(stage.unmet_requirements))
        self._guidance.setText(stage.guidance)
        self._open_button.setText(f"Open {stage.title}")
        self._open_button.setToolTip(f"Go to the {stage.title} page.")

    def stage(self) -> StageSummary | None:
        return self._stage

    def _on_open(self) -> None:
        if self._stage is not None:
            self.route_requested.emit(self._stage.key)


class _FitToViewportLayout(QVBoxLayout):
    """Page layout that fills the scroll viewport and scrolls only below its minimum.

    Word-wrapped labels make a layout report height-for-width, and a resizable
    ``QScrollArea`` then sizes its content to the preferred height instead of
    the viewport, so the page would scroll even when it fits.
    """

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API naming.
        return False


class ProjectHomeWidget(QWidget):
    """The project home: progress line, workflow chart, field notes and cell survey.

    ``set_document`` records the project; the summary is recomputed when the
    page is visible, so navigating elsewhere does not pay for it.
    """

    route_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("projectHome")
        self._document: ProjectDocument | None = None
        self._stale = True

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(self._scroll)
        content = QWidget(self._scroll)
        content.setObjectName("projectHomeContent")
        self._scroll.setWidget(content)

        layout = _FitToViewportLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self._progress = QLabel(self)
        self._progress.setObjectName("projectHomeProgress")
        header.addWidget(self._progress, 1)
        self._journal_button = QPushButton("Open Journal", self)
        self._journal_button.setObjectName("projectHomeJournalButton")
        self._journal_button.setToolTip("Read the project's journal of actions and notes.")
        self._journal_button.clicked.connect(lambda: self.route_requested.emit("provenance_log"))
        header.addWidget(self._journal_button)
        layout.addLayout(header)

        self.chart = ProjectHomeChartWidget(self)
        self.chart.stage_selected.connect(self._show_stage)
        self.chart.route_requested.connect(self.route_requested)
        layout.addWidget(self.chart, 3)

        lower = QHBoxLayout()
        lower.setSpacing(12)
        self.field_notes = FieldNotesPanel(self)
        self.field_notes.route_requested.connect(self.route_requested)
        lower.addWidget(self.field_notes, 1)
        self.survey = CellSurveyWidget(self)
        lower.addWidget(self.survey, 1)
        layout.addLayout(lower, 2)

    def set_document(self, document: ProjectDocument | None) -> None:
        self._document = document
        self._stale = True
        if self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        self._stale = False
        summary = summarize_project(self._document) if self._document is not None else None
        self.chart.set_summary(summary)
        if summary is None:
            self._progress.setText("")
        else:
            text = f"{summary.completed_count} of {len(summary.stages)} stages complete"
            if summary.active_jobs:
                text += (
                    f" · {summary.active_jobs} job{'s' if summary.active_jobs != 1 else ''} active"
                )
            self._progress.setText(text)
        self._show_stage(self.chart.selected_stage())

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        if self._stale:
            self.refresh()

    def _show_stage(self, key: str | None) -> None:
        summary = self.chart.summary()
        stage = summary.stage(key) if summary is not None and key else None
        self.field_notes.set_stage(stage)
        document = self._document
        host = active_host_structure(document) if document is not None else None
        if document is None or host is None or stage is None:
            self.survey.set_survey(host, (), None)
            return
        self.survey.set_survey(
            host, survey_layers(document, stage.key), discovered_sites_layer(document)
        )


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

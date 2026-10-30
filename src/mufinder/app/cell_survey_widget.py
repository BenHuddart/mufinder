"""The project home's cell survey: a flat drawing of the host cell with muon positions.

``CellSurveyWidget`` draws the active host's unit cell projected down one
lattice vector (a, b or c, chosen with the toggle), its atoms shaded by depth,
and one result set of the selected stage on top: UEP minima, candidate sites,
relaxations (with a line from where each muon started), distinct sites, sites
with zero-point energies or fields, and diffusion paths. The arrows step
through a stage's result sets. A stage with no positions of its own shows the
sites found so far, faded. The positions come from
:mod:`mufinder.services.project_survey`; the projection from
:mod:`mufinder.science.cell_projection`.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import (
    arrow_icon,
    body_font_family,
    cell_survey_species_tones,
    current_palette,
    style_hint_text,
)
from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import element_symbol, is_muon_label
from mufinder.science.cell_projection import (
    PROJECTION_AXES,
    CellProjection,
    cell_projection,
    periodic_boundary_images,
)
from mufinder.science.lattice import LatticeError
from mufinder.services.project_survey import SurveyLayer, SurveyMarker, SurveyMarkerKind

FALLBACK_NOTE = "No positions at this stage: showing the sites found so far."


class _SurveyCanvas(QWidget):
    """Paint the projected cell, its atoms and one survey layer."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.host: Structure | None = None
        self.layer: SurveyLayer | None = None
        self.faded = False
        self.axis = "c"
        self._hits: list[tuple[QPointF, float, str]] = []
        self.setMinimumSize(220, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(320, 240)

    def hit_text_at(self, point: QPointF) -> str | None:
        best: tuple[float, str] | None = None
        for center, radius, text in self._hits:
            distance = math.hypot(point.x() - center.x(), point.y() - center.y())
            if distance <= radius + 3.0 and (best is None or distance < best[0]):
                best = (distance, text)
        return best[1] if best is not None else None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        text = self.hit_text_at(event.position())
        if text:
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._hits = []
        projection = self._projection()
        if projection is None:
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(
                QRectF(self.rect()).adjusted(12.0, 12.0, -12.0, -12.0),
                int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                "Load a host structure to survey its cell.",
            )
            painter.end()
            return
        to_screen = self._fit(projection)
        self._paint_cell(painter, palette, projection, to_screen)
        self._paint_atoms(painter, palette, projection, to_screen)
        if self.layer is not None:
            if self.faded:
                painter.setOpacity(0.45)
            self._paint_layer(painter, palette, projection, to_screen, self.layer)
            painter.setOpacity(1.0)
        self._paint_axes_caption(painter, palette, projection)
        painter.end()

    def _projection(self) -> CellProjection | None:
        if self.host is None:
            return None
        try:
            return cell_projection(self.host.lattice_vectors, self.axis)
        except LatticeError:
            return None

    def _fit(self, projection: CellProjection):
        min_x, min_y, max_x, max_y = projection.bounds()
        area = QRectF(self.rect()).adjusted(18.0, 24.0, -18.0, -12.0)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        scale = min(area.width() / span_x, area.height() / span_y)
        left = area.center().x() - span_x * scale / 2.0
        bottom = area.center().y() + span_y * scale / 2.0

        def to_screen(x: float, y: float) -> QPointF:
            return QPointF(left + (x - min_x) * scale, bottom - (y - min_y) * scale)

        return to_screen

    def _point(self, projection: CellProjection, to_screen, fractional: Sequence[float]) -> QPointF:
        projected = projection.project(fractional)
        return to_screen(projected.x, projected.y)

    def _paint_cell(
        self, painter: QPainter, palette, projection: CellProjection, to_screen
    ) -> None:
        painter.setPen(QPen(QColor(palette.border), 1.2))
        for start, end in projection.edges:
            painter.drawLine(to_screen(*start), to_screen(*end))

    def _paint_atoms(
        self, painter: QPainter, palette, projection: CellProjection, to_screen
    ) -> None:
        assert self.host is not None
        tones = cell_survey_species_tones()
        elements: list[str] = []
        atoms: list[tuple[float, QPointF, str]] = []
        for label, position in zip(self.host.species, self.host.fractional_positions):
            if is_muon_label(label):
                continue
            try:
                element = element_symbol(label)
            except ValueError:
                element = str(label)
            if element not in elements:
                elements.append(element)
            for image in periodic_boundary_images(position):
                projected = projection.project(image)
                atoms.append((projected.depth, to_screen(projected.x, projected.y), element))
        radius = max(3.5, min(9.0, min(self.width(), self.height()) * 0.028))
        self._paint_species_key(painter, palette, elements, tones)
        border = QColor(palette.bg_primary)
        for depth, center, element in sorted(atoms, key=lambda atom: atom[0]):
            colour = QColor(tones[elements.index(element) % len(tones)])
            colour.setAlphaF(0.4 + 0.6 * depth)
            painter.setPen(QPen(border, 0.8))
            painter.setBrush(colour)
            painter.drawEllipse(center, radius, radius)
            self._hits.append((center, radius, element))

    def _paint_species_key(
        self, painter: QPainter, palette, elements: list[str], tones: tuple[str, ...]
    ) -> None:
        font = QFont(body_font_family())
        font.setPixelSize(11)
        painter.setFont(font)
        y = 12.0
        for index, element in enumerate(elements):
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(tones[index % len(tones)]))
            painter.drawEllipse(QPointF(12.0, y), 4.5, 4.5)
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(QPointF(22.0, y + 4.0), element)
            y += 16.0

    def _paint_layer(
        self, painter: QPainter, palette, projection: CellProjection, to_screen, layer: SurveyLayer
    ) -> None:
        accent = QColor(palette.accent)
        font = QFont(body_font_family())
        font.setPixelSize(11)
        painter.setFont(font)
        for path in layer.paths:
            start = self._point(projection, to_screen, path.start)
            end = self._point(projection, to_screen, path.end)
            painter.setPen(QPen(accent, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(start, end)
            if path.label:
                middle = QPointF((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0)
                painter.setPen(QColor(palette.text_primary))
                painter.drawText(middle + QPointF(4.0, -4.0), path.label)
        for marker in layer.markers:
            self._paint_marker(painter, palette, projection, to_screen, marker, accent)

    def _paint_marker(
        self,
        painter: QPainter,
        palette,
        projection: CellProjection,
        to_screen,
        marker: SurveyMarker,
        accent: QColor,
    ) -> None:
        center = self._point(projection, to_screen, marker.position)
        if marker.kind == SurveyMarkerKind.UEP_MINIMUM:
            radius = 5.5
            pen = QPen(QColor(palette.text_primary), 1.3)
            pen.setDashPattern([2.0, 2.0])
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, radius, radius)
        elif marker.kind == SurveyMarkerKind.CANDIDATE:
            radius = 3.5
            colour = QColor(accent)
            colour.setAlpha(150)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            painter.drawEllipse(center, radius, radius)
        elif marker.kind == SurveyMarkerKind.RELAXED:
            radius = 4.5
            if marker.start is not None:
                trail = QColor(accent)
                trail.setAlpha(110)
                painter.setPen(QPen(trail, 1.4))
                painter.drawLine(self._point(projection, to_screen, marker.start), center)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(accent)
            painter.drawEllipse(center, radius, radius)
        else:
            radius = 4.0 + min(6.0, max(0, marker.weight - 1) * 0.6)
            painter.setPen(QPen(QColor(palette.bg_primary), 1.2))
            painter.setBrush(accent)
            painter.drawEllipse(center, radius, radius)
        if marker.label:
            painter.setPen(QColor(palette.text_primary))
            painter.drawText(center + QPointF(radius + 3.0, -radius), marker.label)
        if marker.detail:
            self._hits.append((center, radius, marker.detail))

    def _paint_axes_caption(self, painter: QPainter, palette, projection: CellProjection) -> None:
        across, up = projection.screen_axis_labels
        font = QFont(body_font_family())
        font.setPixelSize(11)
        painter.setFont(font)
        painter.setPen(QColor(palette.text_muted))
        painter.drawText(
            QRectF(8.0, 2.0, self.width() - 16.0, 16.0),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"{across} across, {up} up",
        )


class CellSurveyWidget(QFrame):
    """The cell survey with its projection toggle and result-set stepper."""

    layer_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("overviewCard")
        self._layers: tuple[SurveyLayer, ...] = ()
        self._fallback: SurveyLayer | None = None
        self._index = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        header = QHBoxLayout()
        caption = QLabel("Survey of the cell", self)
        style_hint_text(caption)
        header.addWidget(caption, 1)
        view_label = QLabel("View down", self)
        style_hint_text(view_label)
        header.addWidget(view_label)
        self._axis_group = QButtonGroup(self)
        self._axis_group.setExclusive(True)
        self.axis_buttons: dict[str, QToolButton] = {}
        for axis in PROJECTION_AXES:
            button = QToolButton(self)
            button.setText(axis)
            button.setCheckable(True)
            button.setChecked(axis == "c")
            button.setToolTip(f"Look down lattice vector {axis}.")
            button.toggled.connect(lambda checked, value=axis: checked and self.set_axis(value))
            self._axis_group.addButton(button)
            self.axis_buttons[axis] = button
            header.addWidget(button)
        layout.addLayout(header)

        self.canvas = _SurveyCanvas(self)
        layout.addWidget(self.canvas, 1)

        footer = QHBoxLayout()
        self.previous_button = QToolButton(self)
        self.previous_button.setIcon(arrow_icon("left"))
        self.previous_button.setToolTip("Show the previous result set.")
        self.previous_button.setAccessibleName("Previous result set")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        footer.addWidget(self.previous_button)
        self.layer_label = QLabel(self)
        self.layer_label.setObjectName("cardBody")
        self.layer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layer_label.setWordWrap(True)
        footer.addWidget(self.layer_label, 1)
        self.next_button = QToolButton(self)
        self.next_button.setIcon(arrow_icon("right"))
        self.next_button.setToolTip("Show the next result set.")
        self.next_button.setAccessibleName("Next result set")
        self.next_button.clicked.connect(lambda: self.step(1))
        footer.addWidget(self.next_button)
        layout.addLayout(footer)

        self.note_label = QLabel(self)
        style_hint_text(self.note_label)
        self.note_label.setWordWrap(True)
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.note_label)
        self._sync()

    def set_survey(
        self,
        host: Structure | None,
        layers: Sequence[SurveyLayer],
        fallback: SurveyLayer | None,
    ) -> None:
        """Show ``layers`` over ``host``, or ``fallback`` faded when there are none.

        The stepper keeps its place when the new layers belong to the same stage.
        """

        same_stage = (
            bool(layers) and bool(self._layers) and layers[0].stage_key == self._layers[0].stage_key
        )
        self._layers = tuple(layers)
        self._fallback = fallback
        self._index = min(self._index, len(self._layers) - 1) if same_stage else 0
        self.canvas.host = host
        self._sync()

    def layers(self) -> tuple[SurveyLayer, ...]:
        return self._layers

    def current_index(self) -> int:
        return self._index

    def current_layer(self) -> SurveyLayer | None:
        return self.canvas.layer

    def axis(self) -> str:
        return self.canvas.axis

    def set_axis(self, axis: str) -> None:
        if axis not in PROJECTION_AXES:
            return
        self.canvas.axis = axis
        button = self.axis_buttons[axis]
        if not button.isChecked():
            button.setChecked(True)
        self.canvas.update()

    def step(self, delta: int) -> None:
        if len(self._layers) < 2:
            return
        self._index = (self._index + delta) % len(self._layers)
        self._sync()
        self.layer_changed.emit(self._index)

    def _sync(self) -> None:
        many = len(self._layers) > 1
        self.previous_button.setVisible(many)
        self.next_button.setVisible(many)
        if self._layers:
            layer = self._layers[self._index]
            self.canvas.layer = layer
            self.canvas.faded = False
            count = f" ({self._index + 1} of {len(self._layers)})" if many else ""
            self.layer_label.setText(f"{layer.title}{count}")
            self.note_label.setText(layer.note)
        elif self._fallback is not None and self.canvas.host is not None:
            self.canvas.layer = self._fallback
            self.canvas.faded = True
            self.layer_label.setText(self._fallback.title)
            self.note_label.setText(FALLBACK_NOTE)
        else:
            self.canvas.layer = None
            self.canvas.faded = False
            self.layer_label.setText("")
            self.note_label.setText(
                "No muon positions found yet." if self.canvas.host is not None else ""
            )
        self.note_label.setVisible(bool(self.note_label.text()))
        self.canvas.update()

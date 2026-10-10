"""Energy Barriers workflow surface."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFontMetricsF,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.selection_snapshot import SelectionSnapshotPushButton, resolve_target_keys
from mufinder.app.theme import (
    arrow_icon,
    current_palette,
    status_brush,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
)
from mufinder.app.workflow_table_utils import (
    REMOVE_SELECTED_TEXT,
    SelectionFeedback,
    bind_delete_shortcuts,
    bind_reset_failed_context_menu,
    confirm_row_removal,
    create_workflow_table,
    standard_row_actions,
    set_workflow_table_item,
    style_workflow_configure_button,
    workflow_row_is_locked,
    workflow_row_status_text,
)
from mufinder.domain.models import Structure

# Below this widget width the inspector splitter stacks the plot controls above the
# structure viewer so the viewer keeps a usable size on small laptop screens.
_NARROW_INSPECTOR_WIDTH = 1100

# The completed-results table caps its height at this many visible rows so the
# path inspector keeps the remaining vertical space.
_MAX_VISIBLE_NEB_RESULT_ROWS = 6


@dataclass(frozen=True, slots=True)
class EnergyBarrierEndpointRow:
    """One completed relaxed structure available as an energy-barrier endpoint."""

    key: str
    label: str
    charge_state: str
    source_job: str
    relative_energy: str


@dataclass(frozen=True, slots=True)
class EnergyBarrierSourceRow:
    """One diffusion path eligible for NEB, shown in the Energy Barriers view."""

    key: str
    reactant: str
    product: str
    charge_state: str
    path_length: str
    status: str
    #: Configured CASTEP task-farm count, rendered as "Off" when farming is disabled.
    farms: str = "Off"
    configured: bool = False


@dataclass(frozen=True, slots=True)
class EnergyBarrierNebResultRow:
    """One completed NEB result row."""

    key: str
    label: str
    charge_state: str
    path_length: str
    barrier_forward: str
    barrier_reverse: str
    delta_energy: str
    images: str
    status: str
    transition_state_file: str


@dataclass(frozen=True, slots=True)
class EnergyBarrierPreviewImageRow:
    """One image entry in the NEB preview panel."""

    label: str
    reaction_coordinate: str
    energy: str
    reaction_coordinate_value: float
    energy_value: float


def _nice_ticks(low: float, high: float, target_count: int = 4) -> list[float]:
    """Return round-valued axis ticks covering ``[low, high]``."""

    span = high - low
    if span <= 0.0 or not math.isfinite(span):
        return [low]
    raw_step = span / max(target_count, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    step = 10.0 * magnitude
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        candidate = multiplier * magnitude
        if span / candidate <= target_count:
            step = candidate
            break
    first = math.ceil(low / step) * step
    ticks: list[float] = []
    value = first
    while value <= high + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks or [low]


def _format_tick(value: float) -> str:
    return f"{value:.4g}"


class _ReactionCoordinateProfile(QWidget):
    """Reaction-coordinate profile plot used by the NEB inspector.

    The plot is the single scrub surface for the inspector: clicking or
    dragging selects the nearest image, and Left/Right arrow keys step
    through images once the plot has focus.

    Energies render relative to the initial image (E − E₀), so the axis stays
    readable even though absolute CASTEP energies are ~10⁵ eV.
    """

    point_selected = Signal(int)

    _HOVER_DISTANCE = 18.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[EnergyBarrierPreviewImageRow] = []
        self._selected_index = 0
        self._hover_index: int | None = None
        self._dragging = False
        # A small hard minimum plus an Expanding policy lets the plot shrink
        # gracefully in short panes instead of overpainting the widgets below
        # (layouts squeeze cells below preferred size when space runs out, and
        # a large minimumHeight would make the widget overflow its cell).
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setAccessibleName("Reaction coordinate profile")

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(460, 240)

    def clear_profile(self) -> None:
        self._rows = []
        self._selected_index = 0
        self._hover_index = None
        self._dragging = False
        self.update()

    def set_profile(self, rows: list[EnergyBarrierPreviewImageRow], *, selected_index: int = 0) -> None:
        self._rows = list(rows)
        self._selected_index = max(0, min(selected_index, len(self._rows) - 1)) if self._rows else 0
        if self._hover_index is not None and self._hover_index >= len(self._rows):
            self._hover_index = None
        self.update()

    def point_count(self) -> int:
        return len(self._rows)

    def set_selected_index(self, index: int) -> None:
        if not self._rows:
            self._selected_index = 0
        else:
            self._selected_index = max(0, min(index, len(self._rows) - 1))
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API naming.
        if event.button() != Qt.MouseButton.LeftButton or not self._rows:
            super().mousePressEvent(event)
            return
        index = self._nearest_index(event.position())
        if index is not None:
            self._dragging = True
            if index != self._selected_index:
                self.point_selected.emit(index)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API naming.
        if not self._rows:
            super().mouseMoveEvent(event)
            return
        if self._dragging:
            index = self._nearest_index(event.position(), horizontal_only=True)
            if index is not None and index != self._selected_index:
                self.point_selected.emit(index)
        else:
            hover = self._nearest_index(event.position(), max_distance=self._HOVER_DISTANCE)
            if hover != self._hover_index:
                self._hover_index = hover
                self.update()
            self.setCursor(
                Qt.CursorShape.PointingHandCursor if hover is not None else Qt.CursorShape.ArrowCursor
            )
            if hover is not None:
                row = self._rows[hover]
                QToolTip.showText(
                    event.globalPosition().toPoint(),
                    f"{row.label}\nReaction coordinate: {row.reaction_coordinate}\nE − E₀: {row.energy}",
                    self,
                )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API naming.
        self._dragging = False
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._hover_index is not None:
            self._hover_index = None
            self.update()
        self.unsetCursor()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API naming.
        if self._rows and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            delta = -1 if event.key() == Qt.Key.Key_Left else 1
            target = max(0, min(self._selected_index + delta, len(self._rows) - 1))
            if target != self._selected_index:
                self.point_selected.emit(target)
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(palette.bg_primary))

        text_color = QColor(palette.text_muted)
        if not self._rows:
            painter.setPen(text_color)
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "No profile loaded")
            painter.end()
            return

        geometry = self._plot_geometry()
        if geometry is None:
            painter.end()
            return
        bounds, points, x_ticks, y_ticks = geometry
        metrics = QFontMetricsF(painter.font())
        axis_color = QColor(palette.border)
        grid_color = QColor(palette.bg_surface_alt)
        series_color = QColor(palette.accent)
        hover_color = QColor(palette.accent_hover)
        highlight_color = QColor(palette.success)
        annotation_color = QColor(palette.seal_gold)

        min_x, max_x, min_y, max_y = self._data_ranges()

        def x_pixel(value: float) -> float:
            return bounds.left() + (value - min_x) / (max_x - min_x) * bounds.width()

        def y_pixel(value: float) -> float:
            return bounds.bottom() - (value - min_y) / (max_y - min_y) * bounds.height()

        # Horizontal gridlines with y tick labels.
        for tick in y_ticks:
            y = y_pixel(tick)
            painter.setPen(QPen(grid_color, 1.0))
            painter.drawLine(QPointF(bounds.left(), y), QPointF(bounds.right(), y))
            painter.setPen(text_color)
            label_rect = QRectF(
                0.0,
                y - metrics.height() / 2.0,
                bounds.left() - 6.0,
                metrics.height(),
            )
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                _format_tick(tick),
            )

        # X ticks with labels.
        for tick in x_ticks:
            x = x_pixel(tick)
            painter.setPen(QPen(axis_color, 1.0))
            painter.drawLine(QPointF(x, bounds.bottom()), QPointF(x, bounds.bottom() + 4.0))
            painter.setPen(text_color)
            label_rect = QRectF(x - 48.0, bounds.bottom() + 6.0, 96.0, metrics.height())
            painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignHCenter), _format_tick(tick))

        # Axis lines.
        painter.setPen(QPen(axis_color, 1.0))
        painter.drawLine(bounds.bottomLeft(), bounds.bottomRight())
        painter.drawLine(bounds.bottomLeft(), bounds.topLeft())

        # Barrier annotation: dashed guides from the initial-image energy level
        # up to the peak image, labelled with the forward barrier height.
        energies = self._relative_energies()
        peak_index = max(range(len(energies)), key=lambda index: energies[index])
        barrier = energies[peak_index] - energies[0]
        if barrier > 1e-9:
            base_y = y_pixel(energies[0])
            peak_point = points[peak_index]
            dashed = QPen(annotation_color, 1.2, Qt.PenStyle.DashLine)
            painter.setPen(dashed)
            painter.drawLine(QPointF(points[0][0], base_y), QPointF(peak_point[0], base_y))
            painter.drawLine(QPointF(peak_point[0], base_y), QPointF(peak_point[0], peak_point[1]))
            painter.setPen(annotation_color)
            label = f"{barrier:.3g} eV"
            label_width = metrics.horizontalAdvance(label)
            label_x = peak_point[0] + 8.0
            if label_x + label_width > bounds.right():
                label_x = peak_point[0] - 8.0 - label_width
            label_y = (base_y + peak_point[1]) / 2.0 + metrics.ascent() / 2.0
            painter.drawText(QPointF(label_x, label_y), label)

        # Energy series.
        if len(points) >= 2:
            path = QPainterPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            painter.setPen(QPen(series_color, 2.0))
            painter.drawPath(path)

        # Image markers, with hover and selection emphasis.
        for index, point in enumerate(points):
            if index == self._selected_index:
                color, radius = highlight_color, 5.0
            elif index == self._hover_index:
                color, radius = hover_color, 5.0
            else:
                color, radius = series_color, 4.0
            painter.setBrush(color)
            painter.setPen(QPen(QColor(palette.bg_primary), 1.5))
            painter.drawEllipse(QPointF(point[0], point[1]), radius, radius)

        # Axis titles.
        painter.setPen(text_color)
        title_rect = QRectF(
            bounds.left(),
            self.rect().bottom() - metrics.height() - 2.0,
            bounds.width(),
            metrics.height(),
        )
        painter.drawText(title_rect, int(Qt.AlignmentFlag.AlignHCenter), "Reaction coordinate")
        y_title = "E − E₀ (eV)"
        painter.save()
        painter.translate(metrics.ascent() + 2.0, bounds.center().y())
        painter.rotate(-90)
        painter.drawText(QPointF(-metrics.horizontalAdvance(y_title) / 2.0, 0.0), y_title)
        painter.restore()
        painter.end()

    def _relative_energies(self) -> list[float]:
        offset = self._rows[0].energy_value if self._rows else 0.0
        return [row.energy_value - offset for row in self._rows]

    def _data_ranges(self) -> tuple[float, float, float, float]:
        coordinates = [row.reaction_coordinate_value for row in self._rows]
        energies = self._relative_energies()
        min_x, max_x = min(coordinates), max(coordinates)
        min_y, max_y = min(energies), max(energies)
        if abs(max_x - min_x) < 1e-12:
            min_x -= 0.5
            max_x += 0.5
        if abs(max_y - min_y) < 1e-12:
            min_y -= 0.5
            max_y += 0.5
        return min_x, max_x, min_y, max_y

    def _plot_geometry(
        self,
    ) -> tuple[QRectF, list[tuple[float, float]], list[float], list[float]] | None:
        if not self._rows:
            return None
        metrics = QFontMetricsF(self.font())
        min_x, max_x, min_y, max_y = self._data_ranges()
        x_ticks = _nice_ticks(min_x, max_x, 5)
        y_ticks = _nice_ticks(min_y, max_y, 4)
        y_tick_width = max((metrics.horizontalAdvance(_format_tick(tick)) for tick in y_ticks), default=0.0)
        left = metrics.height() + 8.0 + y_tick_width + 6.0
        bottom = 2.0 * metrics.height() + 12.0
        top = metrics.height() + 8.0
        right = 14.0
        bounds = QRectF(self.rect()).adjusted(left, top, -right, -bottom)
        if bounds.width() <= 1.0 or bounds.height() <= 1.0:
            return None
        points: list[tuple[float, float]] = []
        for row, energy in zip(self._rows, self._relative_energies()):
            x_fraction = (row.reaction_coordinate_value - min_x) / (max_x - min_x)
            y_fraction = (energy - min_y) / (max_y - min_y)
            points.append(
                (
                    bounds.left() + x_fraction * bounds.width(),
                    bounds.bottom() - y_fraction * bounds.height(),
                )
            )
        return bounds, points, x_ticks, y_ticks

    def _nearest_index(
        self,
        position: QPointF,
        *,
        max_distance: float | None = None,
        horizontal_only: bool = False,
    ) -> int | None:
        geometry = self._plot_geometry()
        if geometry is None:
            return None
        points = geometry[1]
        if not points:
            return None
        if horizontal_only:
            distances = [abs(point[0] - position.x()) for point in points]
        else:
            distances = [math.hypot(point[0] - position.x(), point[1] - position.y()) for point in points]
        nearest = min(range(len(points)), key=lambda index: distances[index])
        if max_distance is not None and distances[nearest] > max_distance:
            return None
        return nearest


def _build_image_stepper(
    parent: QWidget,
    on_step,
) -> tuple[QToolButton, QLabel, QToolButton]:
    """Create the shared previous / position / next image-stepper controls.

    Used by both the inspector controls and the pop-out window; callers arrange
    the returned widgets in their own layout. ``on_step`` receives -1 or +1.
    """

    prev_button = QToolButton(parent)
    prev_button.setIcon(arrow_icon("left"))
    prev_button.setText("")
    prev_button.setAutoRepeat(True)
    prev_button.setToolTip("Previous image along the reaction coordinate.")
    prev_button.setAccessibleName("Previous image")
    prev_button.clicked.connect(lambda: on_step(-1))
    position_label = QLabel("", parent)
    position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    next_button = QToolButton(parent)
    next_button.setIcon(arrow_icon("right"))
    next_button.setText("")
    next_button.setAutoRepeat(True)
    next_button.setToolTip("Next image along the reaction coordinate.")
    next_button.setAccessibleName("Next image")
    next_button.clicked.connect(lambda: on_step(1))
    return prev_button, position_label, next_button


class _ViewerPopoutWindow(QWidget):
    """Floating tool window that hosts the shared NEB structure viewer.

    The ``Qt.Tool`` flag keeps the window above the main window while the
    application is active, so scrubbing the reaction coordinate from the main
    window does not bury the popped-out viewer. A mirrored image stepper lets
    the user step through images from this window directly.
    """

    closing = Signal()
    step_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("NEB Structure Viewer")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        stepper_row = QHBoxLayout()
        stepper_row.setSpacing(8)
        self._prev_button, self._position_label, self._next_button = _build_image_stepper(
            self, lambda delta: self.step_requested.emit(delta)
        )
        stepper_row.addStretch(1)
        stepper_row.addWidget(self._prev_button)
        stepper_row.addWidget(self._position_label)
        stepper_row.addWidget(self._next_button)
        stepper_row.addStretch(1)
        layout.addLayout(stepper_row)

    def take_viewer(self, viewer: QWidget) -> None:
        layout = self.layout()
        layout.insertWidget(0, viewer, 1)

    def update_position(self, text: str, *, has_previous: bool, has_next: bool) -> None:
        self._position_label.setText(text)
        self._prev_button.setEnabled(has_previous)
        self._next_button.setEnabled(has_next)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self.closing.emit()
        super().closeEvent(event)


class EnergyBarriersWidget(QWidget):
    """Display eligible path sources and completed transition-state results."""

    endpoint_selection_changed = Signal(str, str)
    generate_paths_requested = Signal(str, str, str)
    remove_sources_requested = Signal(list)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    view_source_requested = Signal(str)
    configure_result_requested = Signal(object)
    submit_neb_local_requested = Signal(list)
    submit_neb_cluster_requested = Signal(list)
    neb_result_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Energy Barriers workflow")
        self._endpoint_rows: list[EnergyBarrierEndpointRow] = []
        self._source_rows: list[EnergyBarrierSourceRow] = []
        self._neb_result_rows: list[EnergyBarrierNebResultRow] = []
        self._preview_structures: list[Structure] = []
        self._preview_rows: list[EnergyBarrierPreviewImageRow] = []
        self._preview_index = 0
        self._preview_reset_pending = True
        self._preview_viewer: AtomicStructureViewer | None = None
        self._viewer_popout_window: _ViewerPopoutWindow | None = None
        self._viewer_popout_placeholder: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QVBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(3)

        title = QLabel("Energy Barriers")
        style_section_title(title)
        header.addWidget(title)

        subtitle = QLabel(
            "Build NEB jobs from two relaxed muon endpoints, then inspect the minimum-energy path inside MuFinder."
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        header.addWidget(subtitle)
        layout.addLayout(header)

        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("energyBarriersTabs")
        self._tabs.setDocumentMode(False)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.tabBar().setExpanding(False)
        self._tabs.setElideMode(Qt.TextElideMode.ElideRight)
        layout.addWidget(self._tabs, 1)

        paths_tab = QWidget(self._tabs)
        paths_layout = QVBoxLayout(paths_tab)
        paths_layout.setContentsMargins(10, 10, 10, 10)
        paths_layout.setSpacing(8)

        path_intro = QLabel(
            "Select two relaxed endpoints with the same charge state, generate symmetry-distinct paths, then configure or preview each NEB path."
        )
        path_intro.setWordWrap(True)
        style_hint_text(path_intro)
        paths_layout.addWidget(path_intro)

        endpoint_group = QGroupBox("Path Endpoints")
        endpoint_layout = QVBoxLayout(endpoint_group)
        endpoint_layout.setContentsMargins(10, 8, 10, 8)
        endpoint_layout.setSpacing(6)

        endpoint_hint = QLabel(
            "MuFinder only generates paths after you choose both endpoints explicitly. This keeps path generation pair-scoped and traceable."
        )
        endpoint_hint.setWordWrap(True)
        style_hint_text(endpoint_hint)
        endpoint_layout.addWidget(endpoint_hint)

        endpoint_form = QFormLayout()
        endpoint_form.setHorizontalSpacing(10)
        endpoint_form.setVerticalSpacing(8)
        self._reactant_combo = QComboBox(self)
        self._reactant_combo.currentIndexChanged.connect(self._emit_endpoint_selection_changed)
        endpoint_form.addRow("Initial", self._reactant_combo)
        self._product_combo = QComboBox(self)
        self._product_combo.currentIndexChanged.connect(self._emit_endpoint_selection_changed)
        endpoint_form.addRow("Final", self._product_combo)
        self._max_path_length_input = QLineEdit(self)
        self._max_path_length_input.setPlaceholderText("Auto")
        self._max_path_length_input.setClearButtonEnabled(True)
        self._max_path_length_input.setToolTip(
            "Optional maximum path length in Å. Leave blank to use MuFinder's automatic cutoff."
        )
        endpoint_form.addRow("Max path length (Å)", self._max_path_length_input)
        endpoint_layout.addLayout(endpoint_form)

        endpoint_actions = QHBoxLayout()
        endpoint_actions.setSpacing(8)
        self._generate_paths_button = QPushButton("Generate Paths")
        self._generate_paths_button.clicked.connect(self._emit_generate_paths)
        endpoint_actions.addWidget(self._generate_paths_button)
        endpoint_actions.addStretch(1)
        endpoint_layout.addLayout(endpoint_actions)
        paths_layout.addWidget(endpoint_group)

        self._sources_group = QGroupBox("Diffusion Paths")
        sources_layout = QVBoxLayout(self._sources_group)
        sources_layout.setContentsMargins(10, 8, 10, 8)
        sources_layout.setSpacing(6)

        source_hint = QLabel(
            "Each row is a candidate hop between two relaxed endpoints. Configure the climbing-image "
            "NEB controls (image count, spring constant, tolerances) per path, then build the NEB job "
            "locally or on the cluster. Use View to preview a path in the undistorted supercell. "
            "Right-click selected rows to run, cancel, inspect, reset, configure, or remove them."
        )
        source_hint.setWordWrap(True)
        style_hint_text(source_hint)
        sources_layout.addWidget(source_hint)
        self._sources_table = create_workflow_table(
            ("Reactant", "Product", "Charge", "Path Length", "Farms", "Status", "Parameters", "View"),
            resize_to_contents_columns=(6, 7),
        )
        self._sources_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._sources_table.itemSelectionChanged.connect(self._update_actions)
        # A right-click makes the rows under it the selection, so the menu's
        # run/configure/remove entries reuse the selection-based handlers.
        leading_actions, trailing_actions = standard_row_actions(
            status_for_row=self._source_row_status,
            run_local=lambda _rows: self._emit_submit_neb_local(),
            run_cluster=lambda _rows: self._emit_submit_neb_cluster(),
            cancel=lambda rows: self.cancel_requested.emit(self._source_keys_at_rows(rows)),
            inspect=lambda rows: self.inspect_failed_requested.emit(self._source_keys_at_rows(rows)),
            configure=lambda _rows: self._emit_batch_configure_results(),
            remove=lambda _rows: self._delete_selected_sources(),
            confirm_parent=self,
        )
        bind_reset_failed_context_menu(
            self._sources_table,
            key_for_row=self._source_key_at_row,
            status_for_row=self._source_row_status,
            emit_reset=self.reset_failed_requested.emit,
            leading_actions=leading_actions,
            additional_actions=trailing_actions,
        )
        sources_layout.addWidget(self._sources_table, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._batch_configure_results_button = QPushButton("Configure Selected Parameters...")
        self._remove_sources_button = QPushButton(REMOVE_SELECTED_TEXT)
        self._selection_count_label = QLabel("")
        style_hint_text(self._selection_count_label)
        # SelectionFeedback sets the run buttons' text ("Run All ..." / "Run
        # Selected ...") and enables Remove Selected from the selection.
        self._submit_neb_local_button = QPushButton()
        self._submit_neb_cluster_button = QPushButton()
        self._batch_configure_results_button.clicked.connect(self._emit_batch_configure_results)
        self._remove_sources_button.clicked.connect(self._delete_selected_sources)
        self._submit_neb_local_button.clicked.connect(self._emit_submit_neb_local)
        self._submit_neb_cluster_button.clicked.connect(self._emit_submit_neb_cluster)
        actions.addWidget(self._remove_sources_button)
        actions.addWidget(self._submit_neb_local_button)
        actions.addWidget(self._submit_neb_cluster_button)
        actions.addStretch(1)
        actions.addWidget(self._selection_count_label)
        actions.addWidget(self._batch_configure_results_button)
        sources_layout.addLayout(actions)
        self._sources_feedback = SelectionFeedback(
            self._sources_table,
            count_label=self._selection_count_label,
            run_local_button=self._submit_neb_local_button,
            run_cluster_button=self._submit_neb_cluster_button,
            selection_buttons=(self._remove_sources_button,),
        )
        paths_layout.addWidget(self._sources_group, 1)
        self._tabs.addTab(paths_tab, "Paths")

        neb_tab = QWidget(self._tabs)
        neb_layout = QVBoxLayout(neb_tab)
        neb_layout.setContentsMargins(10, 10, 10, 10)
        neb_layout.setSpacing(8)

        neb_intro = QLabel(
            "Inspect completed NEB profiles by stepping through images, the synchronized reaction-coordinate plot, and the shared structure viewer."
        )
        neb_intro.setWordWrap(True)
        style_hint_text(neb_intro)
        neb_layout.addWidget(neb_intro)

        self._neb_results_group = QGroupBox("Completed NEB Calculations")
        neb_results_layout = QVBoxLayout(self._neb_results_group)
        neb_results_layout.setContentsMargins(10, 8, 10, 8)
        neb_results_layout.setSpacing(6)

        neb_results_hint = QLabel(
            "Right-click a row to open the folder containing the CASTEP trajectory (.ts) file."
        )
        neb_results_hint.setWordWrap(True)
        style_hint_text(neb_results_hint)
        neb_results_layout.addWidget(neb_results_hint)
        self._neb_results_table = create_workflow_table(
            ("Path", "Charge", "Length", "Barrier (fwd)", "Barrier (rev)", "ΔE", "Images", "Status")
        )
        self._neb_results_table.itemSelectionChanged.connect(self._emit_neb_result_selected)
        self._neb_results_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._neb_results_table.customContextMenuRequested.connect(self._show_neb_results_context_menu)
        neb_results_layout.addWidget(self._neb_results_table)
        neb_layout.addWidget(self._neb_results_group)

        preview_group = QGroupBox("NEB Path Inspector")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 8, 10, 8)
        preview_layout.setSpacing(6)

        self._preview_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._preview_splitter.setChildrenCollapsible(False)
        controls = QWidget(self._preview_splitter)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        self._preview_title = QLabel("Select a completed NEB result to inspect the minimum-energy path.")
        self._preview_title.setWordWrap(True)
        controls_layout.addWidget(self._preview_title)

        self._preview_summary = QLabel("")
        self._preview_summary.setWordWrap(True)
        style_hint_text(self._preview_summary)
        controls_layout.addWidget(self._preview_summary)

        self._profile_plot = _ReactionCoordinateProfile(controls)
        self._profile_plot.point_selected.connect(self._apply_preview_index)
        controls_layout.addWidget(self._profile_plot, 1)

        stepper_row = QHBoxLayout()
        stepper_row.setSpacing(8)
        self._preview_prev_button, self._preview_position_label, self._preview_next_button = (
            _build_image_stepper(controls, self._step_preview)
        )
        stepper_row.addWidget(self._preview_prev_button)
        stepper_row.addWidget(self._preview_position_label)
        stepper_row.addWidget(self._preview_next_button)
        stepper_row.addStretch(1)
        self._popout_viewer_button = QPushButton("Pop Out Viewer", controls)
        self._popout_viewer_button.setToolTip(
            "Open the structure viewer in its own resizable window. It stays linked to the reaction coordinate."
        )
        self._popout_viewer_button.clicked.connect(self._toggle_viewer_popout)
        stepper_row.addWidget(self._popout_viewer_button)
        controls_layout.addLayout(stepper_row)

        self._preview_viewer_placeholder = QLabel(
            "Select a completed NEB result to load the 3D structure preview.",
            self._preview_splitter,
        )
        self._preview_viewer_placeholder.setWordWrap(True)
        self._preview_viewer_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_hint_text(self._preview_viewer_placeholder)
        self._preview_splitter.addWidget(controls)
        self._preview_splitter.addWidget(self._preview_viewer_placeholder)
        self._preview_splitter.setStretchFactor(0, 0)
        self._preview_splitter.setStretchFactor(1, 1)
        self._preview_splitter.setSizes([420, 820])
        preview_layout.addWidget(self._preview_splitter, 1)
        neb_layout.addWidget(preview_group, 1)
        self._tabs.addTab(neb_tab, "NEB Analysis")

        self.clear_neb_preview()
        self._apply_responsive_layout()
        self._delete_shortcut, self._backspace_shortcut = bind_delete_shortcuts(
            self, self._delete_selected_sources
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._apply_responsive_layout()

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "_preview_splitter"):
            return
        orientation = (
            Qt.Orientation.Vertical
            if self.width() < _NARROW_INSPECTOR_WIDTH
            else Qt.Orientation.Horizontal
        )
        if self._preview_splitter.orientation() == orientation:
            return
        self._preview_splitter.setOrientation(orientation)
        if orientation == Qt.Orientation.Vertical:
            self._preview_splitter.setSizes([320, 520])
        else:
            self._preview_splitter.setSizes([420, 820])

    def set_endpoint_options(
        self,
        rows: list[EnergyBarrierEndpointRow],
        *,
        reactant_key: str = "",
        product_key: str = "",
        max_path_length_text: str = "",
    ) -> None:
        self._endpoint_rows = list(rows)
        reactant_key = reactant_key.strip()
        product_key = product_key.strip()
        current_reactant = reactant_key or self.selected_endpoint_keys()[0]
        current_product = product_key or self.selected_endpoint_keys()[1]
        if self._max_path_length_input.text().strip() != max_path_length_text.strip():
            self._max_path_length_input.setText(max_path_length_text.strip())

        self._reactant_combo.blockSignals(True)
        self._product_combo.blockSignals(True)
        self._reactant_combo.clear()
        self._product_combo.clear()
        for row in rows:
            parts = (row.label, row.charge_state, row.source_job, row.relative_energy)
            label = "  ·  ".join(part for part in parts if part.strip())
            self._reactant_combo.addItem(label, row.key)
            self._product_combo.addItem(label, row.key)

        self._set_combo_key(self._reactant_combo, current_reactant, fallback_index=0)
        if len(rows) >= 2:
            fallback_index = 1 if self._reactant_combo.currentIndex() == 0 else 0
        else:
            fallback_index = 0
        self._set_combo_key(self._product_combo, current_product, fallback_index=fallback_index)
        self._reactant_combo.blockSignals(False)
        self._product_combo.blockSignals(False)
        self._update_actions()

    def set_sources(self, rows: list[EnergyBarrierSourceRow]) -> None:
        self._source_rows = list(rows)
        self._sources_group.setTitle(f"Diffusion Paths ({len(rows)})")
        _fill_source_table(
            self._sources_table,
            rows,
            configure_requested=self.configure_result_requested.emit,
            view_requested=self.view_source_requested.emit,
        )
        self._update_actions()
        # The rebuild replaced every row's cell widgets.
        self._sources_feedback.refresh()

    def set_neb_results(self, rows: list[EnergyBarrierNebResultRow]) -> None:
        previous_key = next(iter(self.selected_neb_result_keys()), "")
        self._neb_result_rows = list(rows)
        self._neb_results_group.setTitle(f"Completed NEB Calculations ({len(rows)})")
        table = self._neb_results_table
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.label,
                row.charge_state,
                row.path_length,
                row.barrier_forward,
                row.barrier_reverse,
                row.delta_energy,
                row.images,
                row.status,
            )
            tooltip = f"Trajectory file: {row.transition_state_file}" if row.transition_state_file else ""
            for column, value in enumerate(values):
                set_workflow_table_item(table, row_index, column, value)
                item = table.item(row_index, column)
                if item is not None:
                    item.setToolTip(tooltip)
            # Colour the Status cell with the project-standard per-status palette, matching the
            # Diffusion Paths sources table and the relaxation/convergence tables.
            status_item = table.item(row_index, _NEB_RESULT_STATUS_COLUMN)
            if status_item is not None:
                status_item.setForeground(status_brush(row.status))
        table.resizeRowsToContents()
        self._cap_neb_results_table_height()
        target_row = 0 if rows else -1
        if previous_key:
            for index, row in enumerate(rows):
                if row.key == previous_key:
                    target_row = index
                    break
        if target_row >= 0:
            table.selectRow(target_row)
        else:
            self.clear_neb_preview()

    def _cap_neb_results_table_height(self) -> None:
        table = self._neb_results_table
        header_height = table.horizontalHeader().sizeHint().height()
        row_height = table.rowHeight(0) if table.rowCount() else table.verticalHeader().defaultSectionSize()
        visible_rows = max(2, min(table.rowCount(), _MAX_VISIBLE_NEB_RESULT_ROWS))
        table.setMaximumHeight(header_height + visible_rows * row_height + 2 * table.frameWidth() + 6)

    def _show_neb_results_context_menu(self, position) -> None:
        row = self._neb_results_table.rowAt(position.y())
        if not (0 <= row < len(self._neb_result_rows)):
            return
        trajectory = self._neb_result_rows[row].transition_state_file.strip()
        if not trajectory:
            return
        menu = QMenu(self._neb_results_table)
        action = menu.addAction("Open Containing Folder")
        action.triggered.connect(lambda _checked=False, path=trajectory: _open_containing_folder(path))
        menu.exec(self._neb_results_table.viewport().mapToGlobal(position))

    def set_neb_preview(
        self,
        *,
        title: str,
        image_rows: list[EnergyBarrierPreviewImageRow],
        structures: list[Structure],
        selected_index: int = 0,
    ) -> None:
        self._preview_title.setText(title)
        self._preview_rows = list(image_rows)
        self._preview_structures = list(structures)
        self._preview_reset_pending = True
        if not image_rows or not structures:
            self.clear_neb_preview(title=title)
            return
        self._apply_preview_index(selected_index)

    def clear_neb_preview(self, *, title: str | None = None) -> None:
        self._preview_rows = []
        self._preview_structures = []
        self._preview_index = 0
        self._preview_reset_pending = True
        self._preview_title.setText(title or "Select a completed NEB result to inspect the minimum-energy path.")
        self._preview_summary.setText("")
        self._preview_position_label.setText("")
        self._preview_prev_button.setEnabled(False)
        self._preview_next_button.setEnabled(False)
        self._popout_viewer_button.setEnabled(self._viewer_popout_window is not None)
        self._sync_popout_stepper()
        self._profile_plot.clear_profile()
        if self._preview_viewer is not None:
            self._preview_viewer.set_structure(None)

    def selected_endpoint_keys(self) -> tuple[str, str]:
        reactant = self._reactant_combo.currentData()
        product = self._product_combo.currentData()
        return (
            str(reactant) if isinstance(reactant, str) else "",
            str(product) if isinstance(product, str) else "",
        )

    def selected_source_keys(self) -> list[str]:
        indexes = sorted({index.row() for index in self._sources_table.selectedIndexes()})
        keys: list[str] = []
        for row in indexes:
            if 0 <= row < len(self._source_rows):
                keys.append(self._source_rows[row].key)
        return keys

    def _source_key_at_row(self, row: int) -> str | None:
        if not (0 <= row < len(self._source_rows)):
            return None
        return self._source_rows[row].key or None

    def _source_row_status(self, row: int) -> str:
        return workflow_row_status_text(self._sources_table, row, _NEB_STATUS_COLUMN)

    def _emit_endpoint_selection_changed(self) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        self.endpoint_selection_changed.emit(reactant_key, product_key)
        self._update_actions()

    def _emit_generate_paths(self) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        self.generate_paths_requested.emit(reactant_key, product_key, self._max_path_length_input.text().strip())

    def _delete_selected_sources(self) -> None:
        keys = self.selected_source_keys()
        if not keys:
            return
        selected_rows = sorted({index.row() for index in self._sources_table.selectedIndexes()})
        submitted = sum(
            1 for row in selected_rows if workflow_row_is_locked(self._source_row_status(row))
        )
        if not confirm_row_removal(
            self, len(keys), noun=("path", "paths"), submitted_count=submitted
        ):
            return
        removed = set(keys)
        self.set_sources([row for row in self._source_rows if row.key not in removed])
        self.remove_sources_requested.emit(keys)

    def _selected_or_all_source_keys(self) -> list[str]:
        """Return the paths the run buttons act on: the selection, else every row."""

        return self.selected_source_keys() or [row.key for row in self._source_rows if row.key]

    def _source_keys_at_rows(self, rows: Sequence[int]) -> list[str]:
        return [key for row in rows if (key := self._source_key_at_row(row))]

    def _emit_submit_neb_local(self) -> None:
        self.submit_neb_local_requested.emit(self._selected_or_all_source_keys())

    def _emit_batch_configure_results(self) -> None:
        keys = self.selected_source_keys()
        if keys:
            self.configure_result_requested.emit(keys)

    def _emit_submit_neb_cluster(self) -> None:
        self.submit_neb_cluster_requested.emit(self._selected_or_all_source_keys())

    def selected_neb_result_keys(self) -> list[str]:
        indexes = sorted({index.row() for index in self._neb_results_table.selectedIndexes()})
        keys: list[str] = []
        for row in indexes:
            if 0 <= row < len(self._neb_result_rows):
                keys.append(self._neb_result_rows[row].key)
        return keys

    def _emit_neb_result_selected(self) -> None:
        self._update_actions()
        keys = self.selected_neb_result_keys()
        self.neb_result_selected.emit(keys[0] if keys else "")

    def _step_preview(self, delta: int) -> None:
        if self._preview_rows:
            self._apply_preview_index(self._preview_index + delta)

    def _sync_popout_stepper(self) -> None:
        window = self._viewer_popout_window
        if window is None:
            return
        window.update_position(
            self._preview_position_label.text(),
            has_previous=self._preview_prev_button.isEnabled(),
            has_next=self._preview_next_button.isEnabled(),
        )

    def _apply_preview_index(self, index: int) -> None:
        if not self._preview_rows or not self._preview_structures:
            self.clear_neb_preview(title=self._preview_title.text())
            return
        clamped = max(0, min(index, len(self._preview_rows) - 1, len(self._preview_structures) - 1))
        self._preview_index = clamped

        row = self._preview_rows[clamped]
        structure = self._preview_structures[clamped]
        self._preview_summary.setText(
            f"{row.label}  ·  Reaction coordinate {row.reaction_coordinate}  ·  E − E₀ {row.energy}"
        )
        self._profile_plot.set_profile(self._preview_rows, selected_index=clamped)
        self._preview_position_label.setText(f"Image {clamped + 1} / {len(self._preview_rows)}")
        self._preview_prev_button.setEnabled(clamped > 0)
        self._preview_next_button.setEnabled(clamped < len(self._preview_rows) - 1)
        self._popout_viewer_button.setEnabled(True)
        self._sync_popout_stepper()
        preview_viewer = self._ensure_preview_viewer()
        if self._preview_reset_pending:
            # A new NEB result: load fresh so the viewer fits the new cell.
            preview_viewer.set_structure(structure)
            preview_viewer.select_atoms(_muon_atom_indexes(structure))
            self._preview_reset_pending = False
        else:
            # Scrubbing along one band: move the atoms, keep camera and zoom.
            preview_viewer.update_structure_geometry_preserving_view(structure)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self._shutdown_preview_viewer()
        super().closeEvent(event)

    def _ensure_preview_viewer(self) -> AtomicStructureViewer:
        if self._preview_viewer is not None:
            return self._preview_viewer

        # Created unparented: parenting to the splitter first would register the
        # viewer as a sibling and make replaceWidget() refuse the swap.
        viewer = AtomicStructureViewer()
        self._preview_splitter.replaceWidget(1, viewer)
        self._preview_viewer_placeholder.deleteLater()
        self._preview_viewer = viewer
        self._preview_splitter.setStretchFactor(1, 1)
        return viewer

    def _toggle_viewer_popout(self) -> None:
        if self._viewer_popout_window is not None:
            self._viewer_popout_window.close()
            return
        viewer = self._ensure_preview_viewer()
        window = _ViewerPopoutWindow(self)
        placeholder = QLabel(
            "The structure viewer is open in its own window.\nClose that window or click Restore Viewer to dock it back here."
        )
        placeholder.setWordWrap(True)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_hint_text(placeholder)
        self._viewer_popout_placeholder = placeholder
        self._preview_splitter.replaceWidget(1, placeholder)
        window.take_viewer(viewer)
        viewer.show()
        window.closing.connect(self._restore_viewer_from_popout)
        window.step_requested.connect(self._step_preview)
        window.resize(960, 720)
        window.show()
        self._viewer_popout_window = window
        self._popout_viewer_button.setText("Restore Viewer")
        self._sync_popout_stepper()

    def _restore_viewer_from_popout(self) -> None:
        window = self._viewer_popout_window
        if window is None:
            return
        self._viewer_popout_window = None
        viewer = self._preview_viewer
        placeholder = self._viewer_popout_placeholder
        self._viewer_popout_placeholder = None
        if viewer is not None and placeholder is not None:
            self._preview_splitter.replaceWidget(1, viewer)
            placeholder.deleteLater()
            viewer.show()
        self._popout_viewer_button.setText("Pop Out Viewer")
        self._popout_viewer_button.setEnabled(bool(self._preview_rows))
        window.deleteLater()

    def _shutdown_preview_viewer(self) -> None:
        if self._viewer_popout_window is not None:
            self._viewer_popout_window.close()
        if self._preview_viewer is None:
            return
        try:
            self._preview_viewer.shutdown()
        except RuntimeError:
            pass
        self._preview_viewer = None

    def _update_actions(self) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        source_enabled = bool(self.selected_source_keys())
        can_generate = bool(reactant_key and product_key and len(self._endpoint_rows) >= 1)
        self._generate_paths_button.setEnabled(can_generate)
        # The run and remove buttons follow the selection through SelectionFeedback.
        self._batch_configure_results_button.setEnabled(source_enabled)

    @staticmethod
    def _set_combo_key(combo: QComboBox, key: str, *, fallback_index: int) -> None:
        if key:
            for index in range(combo.count()):
                data = combo.itemData(index)
                if isinstance(data, str) and data == key:
                    combo.setCurrentIndex(index)
                    return
        if combo.count():
            combo.setCurrentIndex(max(0, min(fallback_index, combo.count() - 1)))

# Column index of the "Status" header in the Diffusion Paths table (see create_workflow_table
# call); the row tuple writes reactant, product, charge, path length, then status.
_NEB_STATUS_COLUMN = 5

# Column index of the "Status" header in the Completed NEB Calculations table (see
# create_workflow_table call); the row tuple writes path, charge, length, barriers, ΔE,
# images, then status.
_NEB_RESULT_STATUS_COLUMN = 7


def _open_containing_folder(path_text: str) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path_text).parent)))


def _fill_source_table(
    table: QTableWidget,
    rows: list[EnergyBarrierSourceRow],
    *,
    configure_requested,
    view_requested,
) -> None:
    table.setRowCount(len(rows))
    key_by_row: dict[int, str] = {i: row.key for i, row in enumerate(rows)}
    for row_index, row in enumerate(rows):
        for column, value in enumerate(
            (row.reactant, row.product, row.charge_state, row.path_length, row.farms, row.status)
        ):
            set_workflow_table_item(table, row_index, column, value)
        # Colour the Status cell with the project-standard per-status palette (matching the
        # relaxation/convergence tables), so a path's state reads at a glance.
        status_item = table.item(row_index, _NEB_STATUS_COLUMN)
        if status_item is not None:
            status_item.setForeground(status_brush(row.status))
        configure_button = SelectionSnapshotPushButton(table)
        configure_button.selection_snapshot_key = row.key
        style_workflow_configure_button(
            configure_button,
            workflow_label="Energy Barriers",
            configured=row.configured,
            locked=workflow_row_is_locked(row.status),
        )
        def _on_configure_clicked(_checked: bool = False, key=row.key, btn=configure_button) -> None:
            keys = resolve_target_keys(
                btn,
                clicked_key=key,
                snapshot_keys=[key_by_row[r] for r in btn.selection_snapshot_rows if r in key_by_row],
                eligibility=lambda k: True,
                parent=table,
                action="Configure parameters for",
            )
            if keys:
                configure_requested(list(keys))

        configure_button.clicked.connect(_on_configure_clicked)
        table.setCellWidget(row_index, 6, configure_button)

        view_button = QPushButton("View...", table)
        view_button.setToolTip("Open the path viewer for this diffusion path.")
        view_button.clicked.connect(lambda _checked=False, key=row.key: view_requested(key))
        table.setCellWidget(row_index, 7, view_button)
    table.resizeRowsToContents()


def _muon_atom_indexes(structure: Structure) -> list[int]:
    return [
        index
        for index, species in enumerate(structure.species)
        if str(species).strip().casefold() == "h:mu" or str(species).strip().casefold().endswith(":mu")
    ]

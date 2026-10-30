"""Magnetic Structure Builder texture dialog and its preview canvas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from html import escape

import numpy as np
from matplotlib.transforms import Bbox
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.icons import icon
from mufinder.app.magnetic_component_dialogs import (
    AtomTargetCardsMixin,
    DecimalTriple,
    MagneticComponentTable,
    decimal_value,
    direction_group,
    direction_inputs,
    direction_value,
    selector_label,
    set_direction_inputs,
    unit_direction,
)
from mufinder.app.plot_canvas import PlotFrame, create_plot_canvas
from mufinder.app.theme import (
    TexturePreviewColors,
    current_palette,
    feature_dialog_stylesheet,
    set_option_card_state,
    style_help_button,
    style_hint_text,
    style_inline_group,
    style_option_card,
    style_plot_axes,
    style_section_subtitle,
    style_section_title,
    style_small_checked_tool_button,
    texture_preview_colors,
    use_square_top_table_corners,
)
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.science.lattice import reciprocal_to_cartesian
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticComponent,
    MagneticStructureError,
    MagneticTexture,
    MagneticTextureTerm,
    build_hexagonal_skyrmion_texture,
    build_square_skyrmion_texture,
    build_texture_component,
    evaluate_texture_grid,
    evaluate_texture_preview_grid,
    normalize_texture_magnitude,
    prune_texture_terms,
    skyrmion_q_basis_rlu,
    texture_diagnostics,
    topological_charge_magnetic_unit_cell,
    viable_atom_selectors,
)


class MagneticTexturePreview(QWidget):
    """Compact Qt-painted preview of a 2D texture slice."""

    _MIN_VIEW_FRACTION = 0.035

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._moments = np.empty((0, 0, 3), dtype=float)
        self._canvas_mode = "light"
        self._view_center = np.asarray([0.5, 0.5], dtype=float)
        self._view_fraction = 1.0
        self._initial_view_fraction = 1.0
        self._pan_start: QPointF | None = None
        self._pan_start_center = self._view_center.copy()
        self._box_start: QPointF | None = None
        self._box_end: QPointF | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(300, 260)

    def set_canvas_mode(self, mode: str) -> None:
        self._canvas_mode = "dark" if mode == "dark" else "light"
        self.update()

    def set_moments(
        self, moments: np.ndarray | None, *, initial_view_fraction: float | None = None
    ) -> None:
        if moments is None:
            self._moments = np.empty((0, 0, 3), dtype=float)
            self.reset_view()
        else:
            arr = np.asarray(moments, dtype=float)
            self._moments = arr[:, :, 0, :] if arr.ndim == 4 else arr
            if initial_view_fraction is not None:
                self.reset_view(initial_view_fraction=initial_view_fraction)
        self.update()

    def reset_view(self, *, initial_view_fraction: float = 1.0) -> None:
        self._initial_view_fraction = self._bounded_view_fraction(initial_view_fraction)
        self._view_center = np.asarray([0.5, 0.5], dtype=float)
        self._view_fraction = self._initial_view_fraction
        self._pan_start = None
        self._box_start = None
        self._box_end = None
        self.update()

    @staticmethod
    def _mix_color(start: QColor, end: QColor, t: float) -> QColor:
        t = max(0.0, min(1.0, float(t)))
        return QColor(
            round(start.red() + (end.red() - start.red()) * t),
            round(start.green() + (end.green() - start.green()) * t),
            round(start.blue() + (end.blue() - start.blue()) * t),
            round(start.alpha() + (end.alpha() - start.alpha()) * t),
        )

    def _display_moments(self) -> np.ndarray:
        if self._moments.size == 0:
            return self._moments
        i0, i1, j0, j1 = self._visible_index_bounds()
        return self._moments[i0:i1, j0:j1]

    def _plot_rect(self) -> QRectF:
        bounds = QRectF(self.rect()).adjusted(12.0, 12.0, -12.0, -12.0)
        side = max(10.0, min(bounds.width(), bounds.height()))
        return QRectF(
            bounds.left() + max(0.0, (bounds.width() - side) / 2.0),
            bounds.top() + max(0.0, (bounds.height() - side) / 2.0),
            side,
            side,
        )

    def _view_bounds(self) -> tuple[float, float, float, float]:
        half = 0.5 * self._view_fraction
        low_x = float(self._view_center[0] - half)
        high_x = float(self._view_center[0] + half)
        low_y = float(self._view_center[1] - half)
        high_y = float(self._view_center[1] + half)
        return low_x, high_x, low_y, high_y

    def _visible_index_bounds(self) -> tuple[int, int, int, int]:
        nx, ny = self._moments.shape[:2]
        low_x, high_x, low_y, high_y = self._view_bounds()
        i0 = max(0, min(nx - 1, int(np.floor(low_x * nx))))
        i1 = max(i0 + 1, min(nx, int(np.ceil(high_x * nx))))
        j0 = max(0, min(ny - 1, int(np.floor(low_y * ny))))
        j1 = max(j0 + 1, min(ny, int(np.ceil(high_y * ny))))
        return i0, i1, j0, j1

    def _point_to_view_fraction(
        self, point: QPointF, plot_rect: QRectF | None = None
    ) -> np.ndarray:
        rect = self._plot_rect() if plot_rect is None else plot_rect
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return self._view_center.copy()
        local_x = (float(point.x()) - rect.left()) / rect.width()
        local_y = 1.0 - (float(point.y()) - rect.top()) / rect.height()
        local = np.asarray([local_x, local_y], dtype=float)
        local = np.clip(local, 0.0, 1.0)
        low_x, _high_x, low_y, _high_y = self._view_bounds()
        return np.asarray([low_x, low_y], dtype=float) + local * self._view_fraction

    def _clamp_view(self) -> None:
        self._view_fraction = self._bounded_view_fraction(self._view_fraction)
        half = 0.5 * self._view_fraction
        self._view_center = np.clip(self._view_center, half, 1.0 - half)

    def _bounded_view_fraction(self, value: float) -> float:
        return max(self._MIN_VIEW_FRACTION, min(1.0, float(value)))

    def _zoom_at(self, point: QPointF, factor: float) -> None:
        if self._moments.size == 0:
            return
        old_fraction = self._view_fraction
        new_fraction = self._bounded_view_fraction(old_fraction * float(factor))
        if abs(new_fraction - old_fraction) <= 1e-12:
            return
        rect = self._plot_rect()
        anchor = self._point_to_view_fraction(point, rect)
        local_x = (float(point.x()) - rect.left()) / max(rect.width(), 1e-12)
        local_y = 1.0 - (float(point.y()) - rect.top()) / max(rect.height(), 1e-12)
        local = np.clip(np.asarray([local_x, local_y], dtype=float), 0.0, 1.0)
        self._view_fraction = new_fraction
        self._view_center = anchor - (local - 0.5) * new_fraction
        self._clamp_view()
        self.update()

    def _pan_by_pixels(self, delta: QPointF) -> None:
        rect = self._plot_rect()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        self._view_center = self._pan_start_center + np.asarray(
            [
                -float(delta.x()) / rect.width() * self._view_fraction,
                float(delta.y()) / rect.height() * self._view_fraction,
            ],
            dtype=float,
        )
        self._clamp_view()
        self.update()

    def _zoom_to_box(self, start: QPointF, end: QPointF) -> None:
        rect = self._plot_rect()
        if abs(start.x() - end.x()) < 6.0 or abs(start.y() - end.y()) < 6.0:
            return
        first = self._point_to_view_fraction(start, rect)
        second = self._point_to_view_fraction(end, rect)
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        span = max(float(high[0] - low[0]), float(high[1] - low[1]))
        if span <= 1e-12:
            return
        self._view_fraction = self._bounded_view_fraction(span)
        self._view_center = 0.5 * (low + high)
        self._clamp_view()
        self.update()

    def _background_values(self, moments: np.ndarray | None = None) -> tuple[np.ndarray, str]:
        values = self._moments if moments is None else moments
        if values.size == 0:
            return np.empty((0, 0), dtype=float), "none"
        z_values = values[..., 2]
        if float(np.max(np.abs(z_values))) > 1e-10:
            return z_values, "out_of_plane"
        return np.linalg.norm(values, axis=-1), "magnitude"

    def _moment_color(
        self,
        value: float,
        *,
        minimum: float,
        maximum: float,
        mode: str,
        colors: TexturePreviewColors,
    ) -> QColor:
        if mode == "out_of_plane":
            span = max(abs(minimum), abs(maximum), 1e-12)
            t = max(-1.0, min(1.0, value / span))
            if t < 0.0:
                return self._mix_color(
                    QColor(colors.negative_zero), QColor(colors.negative), abs(t)
                )
            return self._mix_color(QColor(colors.positive_zero), QColor(colors.positive), t)
        t = (value - minimum) / max(maximum - minimum, 1e-12)
        return self._mix_color(QColor(colors.magnitude_low), QColor(colors.magnitude_high), t)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colors = texture_preview_colors(self._canvas_mode)
        painter.fillRect(self.rect(), QColor(colors.surround))
        plot_rect = self._plot_rect()
        painter.fillRect(plot_rect, QColor(colors.plot_background))
        painter.setPen(QPen(QColor(colors.frame), 1.1))
        painter.drawRect(plot_rect)
        if self._moments.size == 0:
            painter.setPen(QColor(colors.placeholder_text))
            painter.drawText(plot_rect, Qt.AlignmentFlag.AlignCenter, "Texture preview")
            return

        moments = self._display_moments()
        nx, ny = moments.shape[:2]
        background_values, background_mode = self._background_values(moments)
        min_value = float(np.min(background_values))
        max_value = float(np.max(background_values))
        in_plane = np.linalg.norm(moments[..., :2], axis=-1)
        max_in_plane = float(np.max(in_plane))
        if max_in_plane <= 1e-12:
            return
        cell_w = plot_rect.width() / max(1, nx)
        cell_h = plot_rect.height() / max(1, ny)
        step = max(1, int(max(nx, ny) / 31))
        arrow_spacing = min(cell_w, cell_h) * step
        arrow_scale = 0.44 * arrow_spacing / max_in_plane
        head = max(1.5, min(3.4, arrow_spacing * 0.14))
        for i in range(0, nx, step):
            for j in range(0, ny, step):
                moment = moments[i, j]
                color = self._moment_color(
                    float(background_values[i, j]),
                    minimum=min_value,
                    maximum=max_value,
                    mode=background_mode,
                    colors=colors,
                )
                painter.setPen(QPen(color, max(0.9, min(1.45, arrow_spacing * 0.06))))
                painter.setBrush(color)
                start = QPointF(
                    plot_rect.left() + (i + 0.5) * cell_w,
                    plot_rect.top() + (ny - j - 0.5) * cell_h,
                )
                dx = float(moment[0]) * arrow_scale
                dy = -float(moment[1]) * arrow_scale
                end = QPointF(start.x() + dx, start.y() + dy)
                length = float(np.hypot(dx, dy))
                if length <= 1e-6:
                    painter.drawEllipse(start, 1.0, 1.0)
                    continue
                painter.drawLine(start, end)
                if length > 1e-6:
                    ux, uy = dx / length, dy / length
                    left = QPointF(
                        end.x() - head * (ux + 0.55 * uy), end.y() - head * (uy - 0.55 * ux)
                    )
                    right = QPointF(
                        end.x() - head * (ux - 0.55 * uy), end.y() - head * (uy + 0.55 * ux)
                    )
                    painter.drawPolygon(QPolygonF([end, left, right]))
        if self._box_start is not None and self._box_end is not None:
            painter.setPen(
                QPen(
                    QColor(colors.zoom_box),
                    1.3,
                    Qt.PenStyle.DashLine,
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRectF(self._box_start, self._box_end).normalized().intersected(plot_rect)
            )

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 0.82 if delta > 0 else 1.0 / 0.82
        self._zoom_at(event.position(), factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._moments.size == 0:
            return
        position = event.position()
        if not self._plot_rect().contains(position):
            return
        if event.button() == Qt.MouseButton.RightButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self._box_start = position
            self._box_end = position
            self.update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._pan_start = position
            self._pan_start_center = self._view_center.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        position = event.position()
        if self._box_start is not None:
            self._box_end = position
            self.update()
            event.accept()
            return
        if self._pan_start is not None:
            self._pan_by_pixels(position - self._pan_start)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        position = event.position()
        if self._box_start is not None:
            start = self._box_start
            self._box_start = None
            self._box_end = None
            self._zoom_to_box(start, position)
            event.accept()
            return
        if self._pan_start is not None:
            self._pan_start = None
            self.unsetCursor()
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view(initial_view_fraction=self._initial_view_fraction)
            event.accept()


def _beside_square_cell(cell_axis):
    """Return an axes locator that keeps a colour bar beside a fixed-aspect plot, as tall as it."""

    def locate(colorbar_axis, _renderer) -> Bbox:
        cell_axis.apply_aspect()
        cell = cell_axis.get_position()
        reserved = colorbar_axis.get_position(original=True)
        gap = reserved.x0 - cell_axis.get_position(original=True).x1
        return Bbox.from_bounds(cell.x1 + gap, cell.y0, reserved.width, cell.height)

    return locate


class CornerHelpGroupBox(QGroupBox):
    """Group box with a help button positioned in the title band."""

    def __init__(self, title: str, help_button: QToolButton, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._help_button = help_button
        self._help_button.setParent(self)
        self._help_button.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._position_help_button()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._position_help_button()

    def _position_help_button(self) -> None:
        size = self._help_button.sizeHint()
        self._help_button.resize(size)
        self._help_button.move(max(0, self.width() - size.width() - 12), 0)


class MagneticTextureDialog(AtomTargetCardsMixin, QDialog):
    """Dialog for creating a grouped multi-q spin texture."""

    _target_card_spacing = None
    _PREVIEW_RESOLUTION = 161
    _PREVIEW_RENDER_REPEATS = 8.0
    _PREVIEW_INITIAL_REPEATS = 3.0
    _CHARGE_RESOLUTION = 64
    _CHARGE_MAX_RESOLUTION = 256
    _ARROW_LEGEND = "Colour shows the out-of-plane moment; arrows show the in-plane moment."
    _MAGNITUDE_LEGEND = (
        "Colour shows the moment magnitude |m| over one magnetic cell, on the u-v plane "
        "of the normalization grid."
    )

    _MODES = (
        ("Hexagonal skyrmion", "hexagonal_skyrmion"),
        ("Square skyrmion", "square_skyrmion"),
        ("Custom Fourier texture", "custom"),
    )

    def __init__(
        self,
        structure: Structure,
        *,
        selected_indices: tuple[int, ...] = (),
        component: MagneticComponent | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Edit Magnetic Texture" if component is not None else "Add Magnetic Texture"
        )
        self.resize(1040, 680)
        self.setMinimumSize(860, 560)
        self._structure = structure
        if component is not None and component.selector.kind == "indices":
            selected_indices = (
                *selected_indices,
                *(int(value) for value in component.selector.values),
            )
        self._selected_indices = tuple(
            sorted(
                {
                    int(index)
                    for index in selected_indices
                    if 0 <= int(index) < len(structure.species)
                }
            )
        )
        self._editing_component_id = component.component_id if component is not None else None
        self._selector_rows: list[tuple[QRadioButton, QFrame, QLabel]] = []
        self._mode_rows: list[tuple[QRadioButton, QFrame, QLabel]] = []
        self._preview_error: str | None = None

        self._label_edit = QLineEdit("Magnetic texture")
        self._label_edit.setMinimumWidth(360)
        self._mode_group = QButtonGroup(self)
        self._mode_buttons: dict[str, QRadioButton] = {}
        for label, key in self._MODES:
            button = QRadioButton(label)
            self._mode_group.addButton(button)
            self._mode_buttons[key] = button

        self._amplitude = DecimalLineEdit(
            standard_notation=True, text="1.0", minimum=0.0, maximum=1000.0, placeholder="1.0"
        )
        self._q1 = direction_inputs((0.05, 0.0, 0.0), minimum=-10.0, maximum=10.0)
        self._q2 = direction_inputs((0.0, 0.05, 0.0), minimum=-10.0, maximum=10.0)
        self._q3 = direction_inputs((0.0, 0.0, 0.0), minimum=-10.0, maximum=10.0)
        self._core_direction = direction_inputs((0.0, 0.0, -1.0))
        self._cone_angle = DecimalLineEdit(
            standard_notation=True, text="90.0", minimum=0.0, maximum=180.0, placeholder="90"
        )
        self._background = DecimalLineEdit(
            standard_notation=True, text="0.0", minimum=-1000.0, maximum=1000.0, placeholder="0"
        )
        self._skyrmion_type = QComboBox()
        self._skyrmion_type.addItems(["Bloch", "Neel"])
        self._helicity = QComboBox()
        self._helicity.addItem("+1", 1)
        self._helicity.addItem("-1", -1)

        self._normalise_check = QCheckBox("Normalize moment magnitude")
        self._normalise_check.setToolTip(
            "Use Fourier modulation to reduce spatial variation in |m|."
        )
        self._grid_u = self._integer_input(32, 4, 128)
        self._grid_v = self._integer_input(32, 4, 128)
        self._grid_w = self._integer_input(1, 1, 64)
        self._normalise_threshold = DecimalLineEdit(
            standard_notation=True, text="0.001", minimum=1e-8, maximum=1.0, placeholder="0.001"
        )
        self._prune_mode = QComboBox()
        self._prune_mode.addItems(["No pruning", "Coefficient threshold", "Keep strongest N"])
        self._prune_threshold = DecimalLineEdit(
            standard_notation=True, text="0.001", minimum=0.0, maximum=1000.0, placeholder="0.001"
        )
        self._prune_top_n = self._integer_input(32, 1, 10000)
        self._magnitude_info_button = QToolButton()
        style_help_button(self._magnitude_info_button)
        self._magnitude_info_button.setIcon(icon("help"))
        self._magnitude_info_button.setIconSize(QSize(22, 22))
        self._magnitude_info_button.setToolTip("Explain magnitude normalization and pruning.")
        self._magnitude_info_button.clicked.connect(self._show_magnitude_pruning_info)

        self._custom_table = MagneticComponentTable(
            0, 9, delete_callback=self._remove_selected_texture_terms
        )
        self._custom_table.setHorizontalHeaderLabels(
            ["q1", "q2", "q3", "Re x", "Re y", "Re z", "Im x", "Im y", "Im z"]
        )
        self._custom_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._custom_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._custom_table.itemChanged.connect(lambda *_: self._refresh_preview())
        self._configure_custom_table_columns()
        use_square_top_table_corners(self._custom_table)
        self._add_term_button = QPushButton("Add Fourier Term")
        self._add_term_button.clicked.connect(self._add_custom_term_row)

        self._target_group = QButtonGroup(self)
        self._all_atoms_radio = QRadioButton("All atoms")
        self._species_radio = QRadioButton("Element")
        self._equivalent_radio = QRadioButton("Equivalent site")
        self._selected_radio = QRadioButton("Current selection")
        for button in (
            self._all_atoms_radio,
            self._species_radio,
            self._equivalent_radio,
            self._selected_radio,
        ):
            self._target_group.addButton(button)
        self._species_combo = QComboBox()
        for species in sorted(set(str(symbol) for symbol in structure.species)):
            self._species_combo.addItem(species)
        self._equivalent_combo = QComboBox()
        for selector in viable_atom_selectors(structure):
            if selector.kind == "symmetry_equivalent" and selector.values:
                self._equivalent_combo.addItem(selector_label(selector, structure), selector)

        self._preview = MagneticTexturePreview(self)
        self._preview_canvas_group = QButtonGroup(self)
        self._preview_light_button = self._preview_canvas_button("Light canvas", "appearance_light")
        self._preview_dark_button = self._preview_canvas_button("Dark canvas", "appearance_dark")
        self._preview_light_button.setChecked(True)
        self._preview_canvas_group.buttonClicked.connect(self._refresh_preview_canvas_mode)
        self._magnitude_map_check = QCheckBox("Show |m| map")
        self._magnitude_map_check.setToolTip(
            "Show the moment magnitude over one magnetic cell, sampled on the normalization grid."
        )
        self._magnitude_map_check.toggled.connect(self._refresh_preview_page)
        palette = current_palette()
        self._magnitude_figure, self._magnitude_canvas = create_plot_canvas(
            figsize=(4.0, 3.6), facecolor=palette.bg_surface, minimum_size=(300, 260)
        )
        self._magnitude_frame = PlotFrame(
            self._magnitude_canvas, title="Moment Magnitude", parent=self
        )
        self._magnitude_mesh = None
        self._charge_cache: tuple[object, str] | None = None
        self._preview_stack = QStackedWidget(self)
        self._preview_stack.addWidget(self._preview)
        self._preview_stack.addWidget(self._magnitude_frame)
        self._diagnostics = QLabel("Texture: -")
        self._diagnostics.setWordWrap(True)
        style_hint_text(self._diagnostics)

        self.setStyleSheet(feature_dialog_stylesheet(include_scroll_area=True))
        self._build_layout()
        self._connect_preview_updates()
        self._refresh_target_options()
        self._mode_buttons["hexagonal_skyrmion"].setChecked(True)
        self._add_custom_term_row(values=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        if component is not None:
            self._load_component(component)
        self._refresh_mode_controls()
        self._refresh_preview()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        title = QLabel("Add Magnetic Texture", self)
        style_section_title(title)
        subtitle = QLabel(
            "Define a spin texture in propagation-vector coefficient space, preview it, then apply it to atoms.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        controls = QWidget(self)
        self._controls_panel = controls
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        controls_layout.setSpacing(8)

        component_group = QGroupBox("Texture")
        component_form = QFormLayout(component_group)
        component_form.addRow("Label", self._label_edit)
        component_form.addRow("Amplitude", self._amplitude)
        controls_layout.addWidget(component_group)

        mode_group = QGroupBox("Texture Type")
        mode_layout = QGridLayout(mode_group)
        for column, (label, key) in enumerate(self._MODES):
            mode_layout.addWidget(
                self._build_mode_card(self._mode_buttons[key], self._mode_hint(key)), 0, column
            )
        controls_layout.addWidget(mode_group)

        parameter_row = QGridLayout()
        q_group = QGroupBox("Propagation Basis")
        q_layout = QVBoxLayout(q_group)
        self._q1_row = self._vector_row("q1 (r.l.u.)", self._q1)
        self._q2_row = self._vector_row("q2 (r.l.u.)", self._q2)
        self._q3_row = self._vector_row("q3 (r.l.u.)", self._q3)
        self._generated_q_hint = QLabel(
            "Skyrmion presets generate the remaining q vectors from q1, the core direction, and the selected lattice.",
            q_group,
        )
        self._generated_q_hint.setWordWrap(True)
        style_hint_text(self._generated_q_hint)
        q_layout.addWidget(self._q1_row)
        q_layout.addWidget(self._q2_row)
        q_layout.addWidget(self._q3_row)
        q_layout.addWidget(self._generated_q_hint)
        parameter_row.addWidget(q_group, 0, 0)

        self._skyrmion_group = QGroupBox("Skyrmion Parameters")
        skyrmion_layout = QVBoxLayout(self._skyrmion_group)
        self._core_direction_row = self._vector_row("Core direction", self._core_direction)
        self._rotation_row = self._field_row("Rotation", self._skyrmion_type)
        self._helicity_row = self._field_row("Helicity", self._helicity)
        self._cone_angle_row = self._field_row("Cone angle", self._cone_angle)
        self._square_background_row = self._field_row("Square background", self._background)
        skyrmion_layout.addWidget(self._core_direction_row)
        skyrmion_layout.addWidget(self._rotation_row)
        skyrmion_layout.addWidget(self._helicity_row)
        skyrmion_layout.addWidget(self._cone_angle_row)
        skyrmion_layout.addWidget(self._square_background_row)
        parameter_row.addWidget(self._skyrmion_group, 0, 1)
        parameter_row.setColumnStretch(0, 1)
        parameter_row.setColumnStretch(1, 1)
        controls_layout.addLayout(parameter_row)

        self._custom_group = QGroupBox("Custom Fourier Terms")
        custom_layout = QVBoxLayout(self._custom_group)
        custom_layout.addWidget(self._custom_table)
        custom_layout.addWidget(self._add_term_button)
        controls_layout.addWidget(self._custom_group)

        normalize_group = CornerHelpGroupBox("Magnitude and Pruning", self._magnitude_info_button)
        normalize_layout = QVBoxLayout(normalize_group)
        normalize_form = QFormLayout()
        normalize_form.addRow(self._normalise_check)
        grid_group = QWidget()
        style_inline_group(grid_group)
        grid_layout = QHBoxLayout(grid_group)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(QLabel("u"))
        grid_layout.addWidget(self._grid_u)
        grid_layout.addWidget(QLabel("v"))
        grid_layout.addWidget(self._grid_v)
        grid_layout.addWidget(QLabel("w"))
        grid_layout.addWidget(self._grid_w)
        normalize_form.addRow("Grid", grid_group)
        normalize_form.addRow("Threshold", self._normalise_threshold)
        normalize_form.addRow("Pruning", self._prune_mode)
        normalize_form.addRow("Prune threshold", self._prune_threshold)
        normalize_form.addRow("Keep N", self._prune_top_n)
        normalize_layout.addLayout(normalize_form)
        controls_layout.addWidget(normalize_group)

        target_group = QGroupBox("Apply Texture To")
        target_layout = QGridLayout(target_group)
        target_layout.addWidget(
            self._build_target_card(self._all_atoms_radio, "Apply the texture to every atom."), 0, 0
        )
        target_layout.addWidget(
            self._build_target_card(
                self._species_radio, "Apply to all atoms of one element.", self._species_combo
            ),
            0,
            1,
        )
        target_layout.addWidget(
            self._build_target_card(
                self._equivalent_radio,
                "Apply to one symmetry-equivalent atom site.",
                self._equivalent_combo,
            ),
            1,
            0,
        )
        target_layout.addWidget(
            self._build_target_card(
                self._selected_radio, "Apply to atoms selected in the builder preview."
            ),
            1,
            1,
        )
        controls_layout.addWidget(target_group)
        controls_layout.addStretch(1)

        self._controls_scroll = QScrollArea(self)
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._controls_scroll.setWidget(controls)

        preview_panel = QWidget(self)
        preview_layout = QVBoxLayout(preview_panel)
        preview_header = QHBoxLayout()
        preview_title = QLabel("Spin Texture Preview")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self._magnitude_map_check)
        preview_header.addWidget(self._preview_light_button)
        preview_header.addWidget(self._preview_dark_button)
        preview_layout.addLayout(preview_header)
        self._preview_legend = QLabel(self._ARROW_LEGEND)
        self._preview_legend.setWordWrap(True)
        style_hint_text(self._preview_legend)
        preview_layout.addWidget(self._preview_legend)
        preview_layout.addWidget(self._preview_stack, 1)
        preview_layout.addWidget(self._diagnostics)
        splitter.addWidget(self._controls_scroll)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _mode_hint(key: str) -> str:
        hints = {
            "hexagonal_skyrmion": "Three q vectors separated by 120 degrees in the core-normal plane.",
            "square_skyrmion": "Two perpendicular q vectors in the core-normal plane, with optional uniform background.",
            "custom": "Enter arbitrary complex Fourier coefficients and local q coefficients.",
        }
        return hints[key]

    def _build_mode_card(self, button: QRadioButton, hint_text: str) -> QFrame:
        card = QFrame(self)
        style_option_card(card)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        button.setParent(card)
        hint = QLabel(hint_text, card)
        hint.setWordWrap(True)
        hint.setStyleSheet("background: transparent;")
        style_hint_text(hint)
        card_layout.addWidget(button)
        card_layout.addWidget(hint)
        self._mode_rows.append((button, card, hint))
        button.toggled.connect(self._refresh_mode_controls)
        return card

    def _preview_canvas_button(self, tooltip: str, icon_name: str) -> QToolButton:
        button = QToolButton(self)
        style_small_checked_tool_button(button)
        button.setIcon(icon(icon_name))
        button.setToolTip(tooltip)
        button.setCheckable(True)
        button.setAutoRaise(False)
        self._preview_canvas_group.addButton(button)
        return button

    @staticmethod
    def _integer_input(value: int, minimum: int, maximum: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        spinbox.setMinimumWidth(86)
        return spinbox

    def _vector_row(self, label: str, inputs: DecimalTriple) -> QWidget:
        row = QWidget(self)
        style_inline_group(row)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label_widget = QLabel(label, row)
        label_widget.setMinimumWidth(88)
        layout.addWidget(label_widget)
        layout.addWidget(direction_group(inputs), 1)
        return row

    def _field_row(self, label: str, editor: QWidget) -> QWidget:
        row = QWidget(self)
        style_inline_group(row)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label_widget = QLabel(label, row)
        label_widget.setMinimumWidth(88)
        layout.addWidget(label_widget)
        layout.addWidget(editor, 1)
        return row

    def _connect_preview_updates(self) -> None:
        for edit in (
            self._label_edit,
            self._amplitude,
            self._cone_angle,
            self._background,
            self._normalise_threshold,
            self._prune_threshold,
            *self._q1,
            *self._q2,
            *self._q3,
            *self._core_direction,
        ):
            edit.textChanged.connect(lambda *_: self._refresh_preview())
        for widget in (self._skyrmion_type, self._helicity, self._prune_mode):
            widget.currentIndexChanged.connect(lambda *_: self._refresh_preview())
        for widget in (self._normalise_check,):
            widget.toggled.connect(lambda *_: self._refresh_preview())
        for widget in (self._grid_u, self._grid_v, self._grid_w, self._prune_top_n):
            widget.valueChanged.connect(lambda *_: self._refresh_preview())

    def _configure_custom_table_columns(self) -> None:
        header = self._custom_table.horizontalHeader()
        for column in range(self._custom_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    def _refresh_mode_controls(self, *_args) -> None:
        mode = self.mode()
        custom = mode == "custom"
        square = mode == "square_skyrmion"
        skyrmion = mode in {"hexagonal_skyrmion", "square_skyrmion"}
        for edit in (*self._q2, *self._q3):
            edit.setEnabled(custom)
        self._q2_row.setVisible(custom)
        self._q3_row.setVisible(custom)
        self._generated_q_hint.setVisible(not custom)
        self._custom_group.setVisible(custom)
        self._skyrmion_group.setVisible(skyrmion)
        self._square_background_row.setVisible(square)
        self._custom_table.setEnabled(custom)
        self._add_term_button.setEnabled(custom)
        self._background.setEnabled(square)
        for button, card, _hint in self._mode_rows:
            set_option_card_state(card, selected=button.isChecked())
        self._refresh_preview()

    def _refresh_preview_canvas_mode(self, *_args) -> None:
        self._preview.set_canvas_mode("dark" if self._preview_dark_button.isChecked() else "light")

    def _refresh_preview_page(self, *_args) -> None:
        show_map = self._magnitude_map_check.isChecked()
        if not show_map and self._magnitude_frame.is_popped_out():
            # A popped-out map would stop updating once hidden; bring it back first.
            self._magnitude_frame.show_here()
        self._preview_stack.setCurrentIndex(1 if show_map else 0)
        self._preview_legend.setText(self._MAGNITUDE_LEGEND if show_map else self._ARROW_LEGEND)
        for button in (self._preview_light_button, self._preview_dark_button):
            button.setVisible(not show_map)
        if show_map:
            self._refresh_preview()

    def _draw_magnitude_map(self, texture: MagneticTexture | None) -> None:
        palette = current_palette()
        self._magnitude_figure.clear()
        self._magnitude_mesh = None
        axis = self._magnitude_figure.add_subplot(111)
        style_plot_axes(axis)
        # Grid lines would draw over the colour map.
        axis.grid(False)
        axis.set_xlabel("Fraction of a₁")
        axis.set_ylabel("Fraction of a₂")
        if texture is not None and texture.terms:
            nu, nv, _nw = self._grid_resolution()
            magnitudes = np.linalg.norm(
                evaluate_texture_grid(texture, (nu, nv, 1))[:, :, 0, :], axis=-1
            )
            self._magnitude_mesh = axis.pcolormesh(
                np.linspace(0.0, 1.0, nu + 1),
                np.linspace(0.0, 1.0, nv + 1),
                magnitudes.T,
                shading="flat",
                cmap="viridis",
            )
            # The cell is drawn square. The layout reserves room for the colour bar
            # and its labels; at draw time the bar is resized to the cell's height.
            axis.set_aspect("equal")
            colorbar = self._magnitude_figure.colorbar(self._magnitude_mesh, ax=axis)
            colorbar.ax.set_box_aspect(None)
            colorbar.ax.set_axes_locator(_beside_square_cell(axis))
            colorbar.set_label("|m| (μB)", color=palette.text_primary)
            colorbar.ax.tick_params(colors=palette.text_primary)
            colorbar.outline.set_edgecolor(palette.border)
        self._magnitude_canvas.draw_idle()

    def _topological_charge_text(self, texture: MagneticTexture) -> str:
        q_basis = tuple(
            tuple(reciprocal_to_cartesian(row, self._structure.lattice_vectors))
            for row in self.q_basis_vectors()
        )
        # Edits that leave the texture unchanged (label, target, map toggle) reuse
        # the last result instead of resampling the cell.
        key = (texture, q_basis)
        if self._charge_cache is not None and self._charge_cache[0] == key:
            return self._charge_cache[1]
        try:
            charge = topological_charge_magnetic_unit_cell(
                texture,
                q_basis,
                (self._CHARGE_RESOLUTION, self._CHARGE_RESOLUTION),
                max_resolution=self._CHARGE_MAX_RESOLUTION,
            )
            text = f"Topological charge per magnetic cell: {charge}."
        except MagneticStructureError as exc:
            text = f"Topological charge per magnetic cell: undefined ({exc})."
        self._charge_cache = (key, text)
        return text

    def _show_magnitude_pruning_info(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Magnitude Normalization and Pruning")
        dialog.resize(760, 640)

        layout = QVBoxLayout(dialog)
        title = QLabel("Magnitude Normalization and Pruning", dialog)
        style_section_title(title)
        subtitle = QLabel(
            "Advanced Fourier-texture controls for reducing moment-length variation.", dialog
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._magnitude_pruning_help_html())
        layout.addWidget(browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _magnitude_pruning_help_html(self) -> str:
        grid = f"{int(self._grid_u.value())} × {int(self._grid_v.value())} × {int(self._grid_w.value())}"
        threshold = escape(self._normalise_threshold.text().strip() or "0.001")
        prune_threshold = escape(self._prune_threshold.text().strip() or "0.001")
        keep_n = int(self._prune_top_n.value())
        prune_mode = escape(self._prune_mode.currentText())
        return f"""
        <html>
        <body style="font-size: 13px;">
          <p>
            These controls rescale a multi-q texture by an inverse magnitude field, so the local
            moment length stays close to constant, and then prune the weakest Fourier terms.
            They are useful when a compact multi-q ansatz has the right winding pattern but the local
            moment length varies too strongly across the texture.
          </p>

          <h3>Why Normalize?</h3>
          <p>
            A finite Fourier texture is evaluated as
          </p>
          <p style="text-align: center; font-size: 15px;">
            <b>m</b>(<b>r</b>) =
            Re[Σ<sub>j</sub> <b>S</b><sub>j</sub> exp(-2πi <b>q</b><sub>j</sub>·<b>r</b>)].
          </p>
          <p>
            Linear superposition can make |<b>m</b>(<b>r</b>)| breathe from point to point.
            For a local-moment model, it is often more physical for the direction to vary while
            the moment magnitude remains approximately fixed.
          </p>

          <h3>What MuFinder Does</h3>
          <p>
            When <b>Normalize moment magnitude</b> is enabled, MuFinder samples the current texture
            on a regular reduced-coordinate grid ({grid} at the moment), estimates an inverse
            magnitude field, and Fourier-transforms that field:
          </p>
          <p style="text-align: center; font-size: 15px;">
            <b>m</b>'(<b>r</b>) = <b>m</b>(<b>r</b>) f(<b>r</b>),
            &nbsp;&nbsp; f(<b>r</b>) ≈ A / |<b>m</b>(<b>r</b>)|.
          </p>
          <p>
            The Fourier coefficients of f are convolved with the original texture coefficients,
            producing a richer Fourier texture whose sampled |<b>m</b>| is closer to the chosen
            amplitude A. The threshold currently shown here is <b>{threshold}</b>; smaller values
            keep more modulation harmonics and usually improve constancy, but generate more terms.
          </p>

          <h3>Pruning</h3>
          <p>
            Normalization can create many Fourier terms. Pruning is a deliberate approximation used
            to keep the model manageable for preview, storage, and later field calculations.
          </p>
          <ul>
            <li><b>Coefficient threshold</b>: remove terms with coefficient norm below {prune_threshold} μ<sub>B</sub>.</li>
            <li><b>Keep strongest N</b>: keep only the {keep_n} largest coefficient norms.</li>
          </ul>
          <p>
            Current pruning mode: <b>{prune_mode}</b>.
          </p>

          <h3>Practical Guidance</h3>
          <ul>
            <li>Leave normalization off while drafting simple helices or unmodulated skyrmion presets.</li>
            <li>Turn it on when the diagnostics show a large |m| range or the preview suggests moment collapse.</li>
            <li>Use moderate grids first; raising the grid resolution improves spectral fidelity but costs more.</li>
            <li>After normalization, inspect the diagnostics and preview before pruning aggressively.</li>
          </ul>

          <h3>Caveats</h3>
          <p>
            If the original texture has exact or near-zero |<b>m</b>| points, the ideal inverse
            magnitude is singular. MuFinder guards this numerically, but the result may need many
            high-order harmonics. A small uniform background component can improve conditioning.
          </p>
          <p>
            The result is an approximation, not an analytic proof of constant magnitude. Pruning
            may reintroduce some moment-length variation.
          </p>
        </body>
        </html>
        """

    def mode(self) -> str:
        for key, button in self._mode_buttons.items():
            if button.isChecked():
                return key
        return "hexagonal_skyrmion"

    def selector(self) -> AtomSelector:
        if self._species_radio.isChecked():
            return AtomSelector.species(self._species_combo.currentText())
        if self._equivalent_radio.isChecked():
            selector = self._equivalent_combo.currentData()
            if isinstance(selector, AtomSelector):
                return selector
        if self._selected_radio.isChecked() and self._selected_indices:
            return AtomSelector.indices(self._selected_indices)
        return AtomSelector.all()

    def q_basis_vectors(self) -> tuple[tuple[float, float, float], ...]:
        mode = self.mode()
        if mode == "custom":
            return (
                direction_value(self._q1, "q1"),
                direction_value(self._q2, "q2"),
                direction_value(self._q3, "q3"),
            )
        lattice = "square" if mode == "square_skyrmion" else "hexagonal"
        return skyrmion_q_basis_rlu(
            self._structure,
            q1_rlu=direction_value(self._q1, "q1"),
            core_direction=unit_direction(self._core_direction, "core direction"),
            lattice=lattice,
        )

    def texture(self) -> MagneticTexture:
        base = self._base_texture()
        if self._normalise_check.isChecked():
            base = normalize_texture_magnitude(
                base,
                grid_resolution=self._grid_resolution(),
                threshold=decimal_value(self._normalise_threshold, "normalization threshold"),
                target_magnitude=decimal_value(self._amplitude, "amplitude"),
            )
        mode = self._prune_mode.currentText()
        if mode == "Coefficient threshold":
            base = prune_texture_terms(
                base, threshold=decimal_value(self._prune_threshold, "prune threshold")
            )
        elif mode == "Keep strongest N":
            base = prune_texture_terms(base, top_n=int(self._prune_top_n.value()))
        return base

    def component(
        self,
        q_coeff_transform: Sequence[Sequence[float]],
        q_basis_rlu: Sequence[Sequence[float]],
    ) -> MagneticComponent:
        texture = self.texture()
        metadata = self._texture_metadata(texture, q_basis_rlu)
        return build_texture_component(
            label=self._label_edit.text().strip() or "Magnetic texture",
            selector=self.selector(),
            texture=texture,
            q_coeff_transform=q_coeff_transform,
            phase_turns=0.0,
            component_id=self._editing_component_id,
            metadata=metadata,
        )

    def _base_texture(self) -> MagneticTexture:
        mode = self.mode()
        if mode == "custom":
            return MagneticTexture(tuple(self._custom_terms())).combine_terms()
        q_basis = self.q_basis_vectors()
        q_dirs = self._q_basis_directions(q_basis)
        kwargs = {
            "magnitude": decimal_value(self._amplitude, "amplitude"),
            "core_direction": unit_direction(self._core_direction, "core direction"),
            "cone_angle_degrees": decimal_value(self._cone_angle, "cone angle"),
            "skyrmion_type": self._skyrmion_type.currentText().lower(),
            "helicity": int(self._helicity.currentData()),
        }
        if mode == "square_skyrmion":
            return build_square_skyrmion_texture(
                q_directions=q_dirs[:2],
                ferromagnetic_background=decimal_value(self._background, "square background"),
                **kwargs,
            )
        return build_hexagonal_skyrmion_texture(
            q_directions=(q_dirs[0], q_dirs[1], -(q_dirs[0] + q_dirs[1])), **kwargs
        )

    def _q_basis_directions(
        self, q_basis: Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reciprocal = np.linalg.inv(np.asarray(self._structure.lattice_vectors, dtype=float)).T
        dirs = []
        for row in q_basis:
            cart = np.asarray(row, dtype=float) @ reciprocal
            norm = float(np.linalg.norm(cart))
            dirs.append(cart / norm if norm > 1e-12 else np.zeros(3, dtype=float))
        return (dirs[0], dirs[1], dirs[2])

    def _texture_metadata(
        self, texture: MagneticTexture, q_basis_rlu: Sequence[Sequence[float]]
    ) -> dict[str, object]:
        before = texture_diagnostics(self._base_texture(), self._grid_resolution())
        after = texture_diagnostics(texture, self._grid_resolution())
        return {
            "type": "texture",
            "texture_mode": self.mode(),
            "q_basis_rlu": [[float(x) for x in row] for row in q_basis_rlu],
            "texture": texture.to_json(),
            "builder": {
                "amplitude": decimal_value(self._amplitude, "amplitude"),
                "q_basis_vectors": [[float(x) for x in row] for row in self.q_basis_vectors()],
                "core_direction": [
                    decimal_value(edit, "core direction component") for edit in self._core_direction
                ],
                "cone_angle_degrees": decimal_value(self._cone_angle, "cone angle"),
                "skyrmion_type": self._skyrmion_type.currentText().lower(),
                "helicity": int(self._helicity.currentData()),
                "square_background": decimal_value(self._background, "square background"),
            },
            "normalization": {
                "enabled": bool(self._normalise_check.isChecked()),
                "grid_resolution": list(self._grid_resolution()),
                "threshold": decimal_value(self._normalise_threshold, "normalization threshold"),
            },
            "pruning": {
                "mode": self._prune_mode.currentText(),
                "threshold": decimal_value(self._prune_threshold, "prune threshold"),
                "top_n": int(self._prune_top_n.value()),
            },
            "diagnostics": {"before": asdict(before), "after": asdict(after)},
        }

    def _custom_terms(self) -> tuple[MagneticTextureTerm, ...]:
        terms: list[MagneticTextureTerm] = []
        for row in range(self._custom_table.rowCount()):
            values = [self._table_decimal(row, column) for column in range(9)]
            coefficient = np.asarray(values[3:6], dtype=float) + 1j * np.asarray(
                values[6:9], dtype=float
            )
            if np.allclose(coefficient, 0.0):
                continue
            terms.append(MagneticTextureTerm.from_vector(values[:3], coefficient))
        if not terms:
            raise ValueError("custom texture requires at least one non-zero Fourier term")
        return tuple(terms)

    def _add_custom_term_row(self, *, values: tuple[float, ...] | None = None) -> None:
        values = values or (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        row = self._custom_table.rowCount()
        self._custom_table.insertRow(row)
        for column, value in enumerate(values):
            item = QTableWidgetItem(f"{float(value):.8g}")
            self._custom_table.setItem(row, column, item)
        self._refresh_preview()

    def _remove_selected_texture_terms(self) -> None:
        rows = sorted(
            {index.row() for index in self._custom_table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in rows:
            self._custom_table.removeRow(row)
        self._refresh_preview()

    def _load_component(self, component: MagneticComponent) -> None:
        self._label_edit.setText(component.label)
        self._set_selector(component.selector)
        metadata = dict(component.metadata)
        mode = str(metadata.get("texture_mode", "custom"))
        builder = metadata.get("builder")
        has_builder = isinstance(builder, Mapping)
        if not has_builder and mode in {"hexagonal_skyrmion", "square_skyrmion"}:
            mode = "custom"
        if mode in self._mode_buttons:
            self._mode_buttons[mode].setChecked(True)
        q_basis = builder.get("q_basis_vectors") if has_builder else metadata.get("q_basis_rlu")
        if isinstance(q_basis, Sequence) and len(q_basis) >= 3:
            for inputs, row in zip((self._q1, self._q2, self._q3), q_basis[:3], strict=True):
                if isinstance(row, Sequence):
                    set_direction_inputs(inputs, row)

        if has_builder:
            self._load_texture_builder_metadata(builder)

        texture = (
            MagneticTexture.from_json(metadata.get("texture", {}))
            if has_builder
            else MagneticTexture(
                tuple(
                    MagneticTextureTerm.from_vector(term.q_coeff, term.coefficient.to_array())
                    for term in component.terms
                )
            ).combine_terms()
        )
        if texture.terms:
            self._custom_table.setRowCount(0)
            for term in texture.terms:
                fc = term.coefficient
                self._add_custom_term_row(values=(*term.q_coeff, *fc.real, *fc.imaginary))
        normalization = metadata.get("normalization")
        if has_builder and isinstance(normalization, Mapping):
            self._normalise_check.setChecked(bool(normalization.get("enabled", False)))
            grid = normalization.get("grid_resolution")
            if isinstance(grid, Sequence) and len(grid) >= 3:
                self._grid_u.setValue(int(grid[0]))
                self._grid_v.setValue(int(grid[1]))
                self._grid_w.setValue(int(grid[2]))
            self._set_line_edit_from_metadata(
                self._normalise_threshold, normalization.get("threshold")
            )
        pruning = metadata.get("pruning")
        if has_builder and isinstance(pruning, Mapping):
            mode_text = str(pruning.get("mode", "No pruning"))
            index = self._prune_mode.findText(mode_text)
            if index >= 0:
                self._prune_mode.setCurrentIndex(index)
            self._set_line_edit_from_metadata(self._prune_threshold, pruning.get("threshold"))
            if pruning.get("top_n") is not None:
                self._prune_top_n.setValue(int(pruning.get("top_n")))

    def _load_texture_builder_metadata(self, builder: Mapping[str, object]) -> None:
        self._set_line_edit_from_metadata(self._amplitude, builder.get("amplitude"))
        self._set_direction_from_metadata(self._core_direction, builder.get("core_direction"))
        self._set_line_edit_from_metadata(self._cone_angle, builder.get("cone_angle_degrees"))
        self._set_line_edit_from_metadata(self._background, builder.get("square_background"))
        skyrmion_type = str(builder.get("skyrmion_type", "")).strip().lower()
        for index in range(self._skyrmion_type.count()):
            if self._skyrmion_type.itemText(index).lower() == skyrmion_type:
                self._skyrmion_type.setCurrentIndex(index)
                break
        helicity = builder.get("helicity")
        if helicity is not None:
            index = self._helicity.findData(int(helicity))
            if index >= 0:
                self._helicity.setCurrentIndex(index)

    def _set_direction_from_metadata(self, inputs: DecimalTriple, values: object) -> None:
        if (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and len(values) >= 3
        ):
            set_direction_inputs(inputs, values[:3])

    @staticmethod
    def _set_line_edit_from_metadata(edit: QLineEdit, value: object) -> None:
        if value is None:
            return
        edit.setText(f"{float(value):.8g}")

    def _refresh_preview(self) -> None:
        try:
            texture = self.texture()
            moments = evaluate_texture_preview_grid(
                texture,
                mode=self.mode(),
                core_direction=unit_direction(self._core_direction, "core direction"),
                resolution=self._PREVIEW_RESOLUTION,
                repeats=self._PREVIEW_RENDER_REPEATS,
            )
            self._preview.set_moments(
                moments,
                initial_view_fraction=self._PREVIEW_INITIAL_REPEATS / self._PREVIEW_RENDER_REPEATS,
            )
            before = texture_diagnostics(self._base_texture(), self._grid_resolution())
            after = texture_diagnostics(texture, self._grid_resolution())
            self._diagnostics.setText(
                "Texture: "
                f"{plural(before.term_count, 'term')} → {plural(after.term_count, 'term')}; "
                f"|m| mean {after.magnitude_mean:.4g} μB, std {after.magnitude_std:.4g}, "
                f"range {after.magnitude_min:.4g}-{after.magnitude_max:.4g}.\n"
                f"{self._topological_charge_text(texture)}"
            )
            if self._magnitude_map_check.isChecked():
                self._draw_magnitude_map(texture)
            self._preview_error = None
        except ValueError as exc:
            self._preview.set_moments(None)
            if self._magnitude_map_check.isChecked():
                self._draw_magnitude_map(None)
            self._diagnostics.setText(f"Texture preview unavailable: {exc}")
            self._preview_error = str(exc)

    def accept(self) -> None:  # noqa: D401 - Qt API override.
        """Accept only if the texture can be generated."""

        try:
            self.q_basis_vectors()
            self.texture()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid magnetic texture", str(exc))
            return
        super().accept()

    def _grid_resolution(self) -> tuple[int, int, int]:
        return (int(self._grid_u.value()), int(self._grid_v.value()), int(self._grid_w.value()))

    def _table_decimal(self, row: int, column: int) -> float:
        item = self._custom_table.item(row, column)
        if item is None or not item.text().strip():
            raise ValueError(f"custom term row {row + 1}, column {column + 1} is required")
        return float(item.text())

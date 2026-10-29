"""Embedded 3D atomic structure viewer for host structures."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from copy import deepcopy
from functools import wraps
import logging
import math

from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors
from ase.neighborlist import NeighborList
import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF, QRadialGradient
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLayoutItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import body_font_family, current_palette, style_hint_text
from mufinder.domain.models import Structure

_LOGGER = logging.getLogger(__name__)

_DEFAULT_VIEW = (20.0, 45.0)
_DEFAULT_VIEW_ROLL = 0.0
_DEFAULT_PYVISTA_LIGHT_DIRECTION = np.array([0.28, 0.36, 1.0], dtype=float) / math.sqrt(
    0.28**2 + 0.36**2 + 1.0**2
)
MUON_VIEWER_RADIUS_ANG = 0.2


@dataclass(slots=True)
class _CameraState:
    elev: float
    azim: float
    roll: float = 0.0


@dataclass(frozen=True, slots=True)
class _ViewerStyle:
    background: str
    surface: str
    surface_alt: str
    border: str
    text_primary: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_pressed: str
    seal_gold: str
    success: str
    info: str
    triad_axis_colors: tuple[str, str, str]
    triad_hub_inner: str
    triad_hub_mid: str
    triad_hub_outer: str
    unit_cell: str
    isosurface: str
    selection_fill: str
    selection_stroke: str
    triad_label_font_family: str


@dataclass(frozen=True, slots=True)
class _ViewerIsosurfaceSpec:
    vertices: np.ndarray
    faces: np.ndarray
    color: str
    alpha: float


@dataclass(frozen=True, slots=True)
class _ViewerSliceSpec:
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    z_coordinates: np.ndarray
    values: np.ndarray
    anchor: np.ndarray
    normal: np.ndarray
    color_limits: tuple[float, float]
    cmap: str
    alpha: float


@dataclass(frozen=True, slots=True)
class _ViewerPointOverlaySpec:
    positions: np.ndarray
    colors: np.ndarray
    radii: np.ndarray
    alpha: float


@dataclass(frozen=True, slots=True)
class _ViewerLineOverlaySpec:
    start: np.ndarray
    end: np.ndarray
    color: tuple[float, float, float]
    width: float
    alpha: float
    arrow: bool


class _AxesProxy:
    """Renderer-neutral camera proxy used by tests and viewer state."""

    def __init__(self, state: _CameraState | None = None) -> None:
        initial = _CameraState(*_DEFAULT_VIEW, _DEFAULT_VIEW_ROLL) if state is None else state
        self.elev = float(initial.elev)
        self.azim = float(initial.azim)
        self.roll = float(initial.roll)
        self._dirty = False

    def view_init(self, *, elev: float, azim: float, roll: float | None = None) -> None:
        self.elev = float(elev)
        self.azim = float(azim)
        if roll is not None:
            self.roll = float(roll)
        self._dirty = True

    def set_from_state(self, state: _CameraState) -> None:
        self.elev = float(state.elev)
        self.azim = float(state.azim)
        self.roll = float(state.roll)
        self._dirty = False

    def state(self) -> _CameraState:
        return _CameraState(self.elev, self.azim, self.roll)

    def consume_dirty(self) -> bool:
        dirty = self._dirty
        self._dirty = False
        return dirty


class _WrappingHintLabel(QLabel):
    """Word-wrapped label that reports a stable height-for-width during initial layout."""

    def _target_wrap_width(self) -> int:
        width = self.width()
        if width > 0:
            return width
        parent = self.parentWidget()
        if parent is not None and parent.width() > 0:
            return parent.width()
        return super().sizeHint().width()

    def _sync_minimum_height(self) -> None:
        if not self.wordWrap():
            return
        target_height = self.heightForWidth(self._target_wrap_width())
        if self.minimumHeight() != target_height:
            self.setMinimumHeight(target_height)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API naming.
        return self.wordWrap()

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API naming.
        if not self.wordWrap() or width <= 0:
            return super().sizeHint().height()

        margins = self.contentsMargins()
        content_width = max(1, width - margins.left() - margins.right())
        bounds = self.fontMetrics().boundingRect(
            QRect(0, 0, content_width, 10_000),
            int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
            self.text(),
        )
        return bounds.height() + margins.top() + margins.bottom() + 4

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        if not self.wordWrap():
            return hint

        target_width = self._target_wrap_width()
        return QSize(hint.width(), max(hint.height(), self.heightForWidth(target_width)))

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        if not self.wordWrap():
            return hint
        width = max(self._target_wrap_width(), hint.width())
        return QSize(0, self.heightForWidth(width))

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._sync_minimum_height()
        self.updateGeometry()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._sync_minimum_height()
        self.updateGeometry()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
        }:
            self._sync_minimum_height()
            self.updateGeometry()


class _ElidingLabel(QLabel):
    """Single-line label that elides with an ellipsis instead of widening its parent.

    A plain ``QLabel`` reports its whole text as its minimum width, so a long
    selection summary would stop every dialog holding the viewer from shrinking.
    ``text()`` still returns the full text; callers put it in the tooltip too.
    """

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), Qt.TextElideMode.ElideRight, rect.width())
        painter = QPainter(self)
        self.style().drawItemText(
            painter,
            rect,
            int(self.alignment()),
            self.palette(),
            self.isEnabled(),
            text,
            self.foregroundRole(),
        )


class _WrappingRowLayout(QLayout):
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


def _compact_text_button_width(button: QPushButton) -> int:
    """Return a width that fits a one- or two-character push button.

    Qt styles give every text push button a minimum of about 80 px, which made
    the a/b/c view buttons alone take 240 px of the viewer's toolbar.
    """

    return button.fontMetrics().horizontalAdvance(button.text()) + 28


def _set_axis_box_aspect_from_spans(axis, spans: np.ndarray) -> None:
    span_x, span_y, span_z = np.maximum(np.asarray(spans, dtype=float), 1e-6)
    axis.set_box_aspect((span_x, span_y, span_z))


def _unit_cell_geometry(cell: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            cell[0],
            cell[1],
            cell[2],
            cell[0] + cell[1],
            cell[0] + cell[2],
            cell[1] + cell[2],
            cell[0] + cell[1] + cell[2],
        ],
        dtype=float,
    )
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    return corners, edges


def _renders_once(method):
    """Run a viewer method as one batch, so its scene and triad draw at most once."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.batch_update():
            return method(self, *args, **kwargs)

    return wrapper


def _build_pyvista_unit_cell_mesh(pv, cell: np.ndarray):
    """Return the cell outline as one polyline mesh, so it is a single actor."""

    corners, edges = _unit_cell_geometry(cell)
    lines = np.array([[2, start, end] for start, end in edges], dtype=np.int64).reshape(-1)
    return pv.PolyData(corners, lines=lines)


def _get_atom_style(numbers: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colors = np.array(jmol_colors[numbers], dtype=float)
    muon_mask = symbols == "X"
    if np.any(muon_mask):
        colors[muon_mask] = np.array([1.0, 0.08, 0.58])
    radii = np.clip(0.9 * covalent_radii[numbers], 0.2, 1.15)
    if np.any(muon_mask):
        radii[muon_mask] = np.maximum(radii[muon_mask], MUON_VIEWER_RADIUS_ANG)
    return colors, radii


def _viewer_species_symbols(species: Iterable[object]) -> list[str]:
    symbols: list[str] = []
    for species_label in species:
        label = str(species_label).strip()
        lowered = label.casefold()
        if lowered == "h:mu" or lowered.endswith(":mu"):
            symbols.append("X")
        elif ":" in label:
            symbols.append(label.split(":", 1)[0] or "X")
        else:
            symbols.append(label or "X")
    return symbols


def viewer_atom_style_for_species(species: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    """Return the viewer color/radius arrays for structure species labels."""

    symbols = np.asarray(_viewer_species_symbols(species), dtype=object)
    numbers = np.zeros((len(symbols),), dtype=int)
    for index, symbol in enumerate(symbols):
        lookup_key = str(symbol)
        number = atomic_numbers.get(lookup_key)
        if number is None:
            number = atomic_numbers.get(lookup_key.capitalize())
        numbers[index] = 0 if number is None else int(number)
    return _get_atom_style(numbers, symbols)


def _atom_color_overrides(structure: Structure | None, atom_count: int) -> np.ndarray | None:
    if structure is None:
        return None
    raw_colors = structure.properties.get("atom_colors")
    if not isinstance(raw_colors, Iterable) or isinstance(raw_colors, (str, bytes)):
        return None
    entries = list(raw_colors)
    if len(entries) != int(atom_count):
        return None

    colors: list[tuple[float, float, float] | None] = []
    has_override = False
    for entry in entries:
        if entry is None:
            colors.append(None)
            continue
        if isinstance(entry, str) and not entry:
            colors.append(None)
            continue
        if isinstance(entry, str):
            qcolor = QColor(entry)
            if not qcolor.isValid():
                colors.append(None)
                continue
            colors.append(tuple(float(value) for value in qcolor.getRgbF()[:3]))
            has_override = True
            continue
        if isinstance(entry, Iterable):
            values = list(entry)
            if len(values) != 3:
                colors.append(None)
                continue
            try:
                rgb = tuple(float(value) for value in values)
            except (TypeError, ValueError):
                colors.append(None)
                continue
            if any(not np.isfinite(value) for value in rgb):
                colors.append(None)
                continue
            scale = 255.0 if max(rgb) > 1.0 else 1.0
            colors.append(tuple(float(np.clip(value / scale, 0.0, 1.0)) for value in rgb))
            has_override = True
            continue
        colors.append(None)
    if not has_override:
        return None

    overrides = np.full((int(atom_count), 3), np.nan, dtype=float)
    for index, color in enumerate(colors):
        if color is not None:
            overrides[index] = np.asarray(color, dtype=float)
    return overrides


def _compute_limits(
    positions: np.ndarray,
    *,
    atom_radii: np.ndarray | None = None,
    extra_points: np.ndarray | None = None,
    margin: float = 0.12,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points = np.asarray(positions, dtype=float)
    if atom_radii is not None and len(atom_radii) == len(points):
        radii = np.asarray(atom_radii, dtype=float).reshape(-1, 1)
        mins = np.min(points - radii, axis=0)
        maxs = np.max(points + radii, axis=0)
    else:
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)

    if extra_points is not None and len(extra_points):
        extras = np.asarray(extra_points, dtype=float)
        mins = np.minimum(mins, np.min(extras, axis=0))
        maxs = np.maximum(maxs, np.max(extras, axis=0))

    span = np.maximum(maxs - mins, 1e-6)
    padding = span * float(margin)
    return (
        (float(mins[0] - padding[0]), float(maxs[0] + padding[0])),
        (float(mins[1] - padding[1]), float(maxs[1] + padding[1])),
        (float(mins[2] - padding[2]), float(maxs[2] + padding[2])),
    )


def _set_equal_limits(
    axis,
    positions: np.ndarray,
    *,
    atom_radii: np.ndarray | None = None,
    extra_points: np.ndarray | None = None,
    margin: float = 0.12,
) -> None:
    limits = _compute_limits(
        positions, atom_radii=atom_radii, extra_points=extra_points, margin=margin
    )
    axis.set_xlim(*limits[0])
    axis.set_ylim(*limits[1])
    axis.set_zlim(*limits[2])
    spans = np.array(
        [limits[0][1] - limits[0][0], limits[1][1] - limits[1][0], limits[2][1] - limits[2][0]]
    )
    _set_axis_box_aspect_from_spans(axis, spans)


def _draw_atom_spheres(
    axis,
    positions: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    *,
    alpha: float = 0.95,
    resolution: int = 12,
) -> list[object]:
    artists: list[object] = []
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, resolution)
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    sin_v = np.sin(v)
    cos_v = np.cos(v)

    for position, radius, color in zip(positions, radii, colors):
        x = position[0] + radius * np.outer(cos_u, sin_v)
        y = position[1] + radius * np.outer(sin_u, sin_v)
        z = position[2] + radius * np.outer(np.ones_like(u), cos_v)
        surface = axis.plot_surface(
            x,
            y,
            z,
            color=color,
            linewidth=0.0,
            antialiased=True,
            shade=True,
            alpha=alpha,
        )
        artists.append(surface)
    return artists


def _normalized_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _blend_hex_colors(color_a: str, color_b: str, ratio: float) -> str:
    clamped = float(np.clip(ratio, 0.0, 1.0))
    qcolor_a = QColor(color_a)
    qcolor_b = QColor(color_b)
    red = round((1.0 - clamped) * qcolor_a.red() + clamped * qcolor_b.red())
    green = round((1.0 - clamped) * qcolor_a.green() + clamped * qcolor_b.green())
    blue = round((1.0 - clamped) * qcolor_a.blue() + clamped * qcolor_b.blue())
    return QColor(red, green, blue).name()


def _viewer_style() -> _ViewerStyle:
    palette = current_palette()
    triad_axis_colors = (
        palette.accent,
        palette.success,
        palette.status_submitting,
    )
    return _ViewerStyle(
        background=palette.bg_primary,
        surface=palette.bg_surface,
        surface_alt=palette.bg_surface_alt,
        border=palette.border,
        text_primary=palette.text_primary,
        text_muted=palette.text_muted,
        accent=palette.accent,
        accent_hover=palette.accent_hover,
        accent_pressed=palette.accent_pressed,
        seal_gold=palette.seal_gold,
        success=palette.success,
        info=palette.status_submitting,
        triad_axis_colors=triad_axis_colors,
        triad_hub_inner=_blend_hex_colors(palette.bg_surface, "#ffffff", 0.24),
        triad_hub_mid=_blend_hex_colors(palette.bg_surface_alt, palette.bg_surface, 0.5),
        triad_hub_outer=_blend_hex_colors(palette.border, palette.text_muted, 0.35),
        unit_cell=_blend_hex_colors(palette.text_muted, palette.border, 0.45),
        isosurface=palette.status_submitting,
        selection_fill=_blend_hex_colors(palette.accent, palette.bg_surface, 0.58),
        selection_stroke=palette.accent_pressed,
        triad_label_font_family=body_font_family(),
    )


def _configure_transparent_pyvista_background(plotter) -> None:
    """Match the VTK scene background to the app theme for reliable blending."""

    try:
        plotter.set_background(_viewer_background_color())
    except Exception:  # noqa: BLE001
        pass


def _viewer_background_color() -> str:
    """Return the app-level viewer background color for backends that cannot alpha composite."""

    return _viewer_style().background


def _viewer_style_signature() -> tuple[str, ...]:
    """Return the style values that are baked into rendered viewer scenes."""

    style = _viewer_style()
    return (
        style.background,
        style.surface,
        style.surface_alt,
        style.border,
        style.text_primary,
        style.text_muted,
        style.accent,
        style.accent_hover,
        style.accent_pressed,
        style.seal_gold,
        style.success,
        style.info,
        *style.triad_axis_colors,
        style.triad_hub_inner,
        style.triad_hub_mid,
        style.triad_hub_outer,
        style.unit_cell,
        style.isosurface,
        style.selection_fill,
        style.selection_stroke,
        style.triad_label_font_family,
    )


def _shutdown_pyvista_widget(widget: object | None) -> None:
    """Close a PyVista Qt interactor safely, stopping its internal timers first."""

    if widget is None:
        return
    shutdown = getattr(widget, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
            return
        except Exception:  # noqa: BLE001
            pass

    plotter = getattr(widget, "plotter", None)
    if plotter is None:
        plotter = getattr(widget, "_plotter", None)
    if plotter is not None:
        render_timer = getattr(plotter, "render_timer", None)
        if render_timer is not None:
            try:
                render_timer.stop()
            except Exception:  # noqa: BLE001
                pass
        close = getattr(plotter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


def _plot_cylinder_segment(
    axis,
    start: np.ndarray,
    end: np.ndarray,
    *,
    radius: float,
    color: np.ndarray,
    alpha: float,
    resolution: int,
) -> bool:
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-10:
        return False

    axis_dir = direction / length
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis_dir, reference))) > 0.92:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)

    basis_u = _normalized_vector(
        np.cross(axis_dir, reference), np.array([1.0, 0.0, 0.0], dtype=float)
    )
    basis_v = _normalized_vector(
        np.cross(axis_dir, basis_u), np.array([0.0, 1.0, 0.0], dtype=float)
    )

    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    axial = np.array([0.0, length], dtype=float)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x = (
        start[0]
        + np.outer(cos_t, np.full_like(axial, radius * basis_u[0]))
        + np.outer(sin_t, np.full_like(axial, radius * basis_v[0]))
        + np.outer(np.ones_like(theta), axial * axis_dir[0])
    )
    y = (
        start[1]
        + np.outer(cos_t, np.full_like(axial, radius * basis_u[1]))
        + np.outer(sin_t, np.full_like(axial, radius * basis_v[1]))
        + np.outer(np.ones_like(theta), axial * axis_dir[1])
    )
    z = (
        start[2]
        + np.outer(cos_t, np.full_like(axial, radius * basis_u[2]))
        + np.outer(sin_t, np.full_like(axial, radius * basis_v[2]))
        + np.outer(np.ones_like(theta), axial * axis_dir[2])
    )

    axis.plot_surface(
        x,
        y,
        z,
        color=color,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        alpha=alpha,
    )
    return True


def _draw_split_bond_cylinder(
    axis,
    start: np.ndarray,
    end: np.ndarray,
    start_color: np.ndarray,
    end_color: np.ndarray,
    *,
    radius: float,
    alpha: float,
    resolution: int = 14,
) -> int:
    midpoint = 0.5 * (np.asarray(start, dtype=float) + np.asarray(end, dtype=float))
    segments = 0
    if _plot_cylinder_segment(
        axis,
        np.asarray(start, dtype=float),
        midpoint,
        radius=radius,
        color=np.asarray(start_color, dtype=float),
        alpha=alpha,
        resolution=resolution,
    ):
        segments += 1
    if _plot_cylinder_segment(
        axis,
        midpoint,
        np.asarray(end, dtype=float),
        radius=radius,
        color=np.asarray(end_color, dtype=float),
        alpha=alpha,
        resolution=resolution,
    ):
        segments += 1
    return segments


def _draw_spin_arrows(
    axis,
    positions: np.ndarray,
    radii: np.ndarray,
    spins: np.ndarray,
    *,
    length_scale: float = 0.85,
    head_fraction: float = 0.38,
) -> np.ndarray:
    """Draw high-visibility 3D spin arrows."""

    if len(positions) == 0:
        return np.empty((0, 3), dtype=float)

    theta = np.linspace(0.0, 2.0 * np.pi, 18)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    extent_points: list[np.ndarray] = []
    for position, radius, spin in zip(positions, radii, spins):
        spin_array = np.asarray(spin, dtype=float)
        is_vector = spin_array.shape == (3,)
        spin_vector = (
            spin_array if is_vector else np.array([0.0, 0.0, float(spin_array)], dtype=float)
        )
        magnitude = float(np.linalg.norm(spin_vector))
        if magnitude <= 1e-10:
            continue

        direction_vector = spin_vector / magnitude
        direction = 1.0 if direction_vector[2] >= 0.0 else -1.0
        sphere_radius = float(radius)
        arrow_length = max(0.3, magnitude * length_scale) + 0.4 * sphere_radius
        half_length = 0.5 * arrow_length
        head_length = min(max(0.12, arrow_length * head_fraction), half_length * 0.9)
        shaft_tip_offset = max(half_length - head_length, 0.05)
        shaft_radius = max(0.055, 0.12 * sphere_radius)
        head_radius = max(0.12, 0.2 * sphere_radius)

        center_z = position[2]
        tail_z = center_z - direction * half_length
        shaft_end_z = center_z + direction * shaft_tip_offset
        tip_z = center_z + direction * half_length

        color = "#1f78b4" if direction > 0 else "#d7301f"

        if is_vector:
            tail = np.asarray(position, dtype=float) - direction_vector * half_length
            tip = np.asarray(position, dtype=float) + direction_vector * half_length
            axis.quiver(
                tail[0],
                tail[1],
                tail[2],
                direction_vector[0] * arrow_length,
                direction_vector[1] * arrow_length,
                direction_vector[2] * arrow_length,
                color=color,
                linewidth=max(1.0, shaft_radius * 12.0),
                arrow_length_ratio=head_fraction,
                normalize=False,
            )
            extent_points.append(tip)
            extent_points.append(tail)
            continue

        z_line = np.linspace(tail_z, shaft_end_z, 2)
        shaft_x = position[0] + shaft_radius * np.outer(cos_t, np.ones_like(z_line))
        shaft_y = position[1] + shaft_radius * np.outer(sin_t, np.ones_like(z_line))
        shaft_z = np.outer(np.ones_like(theta), z_line)
        axis.plot_surface(
            shaft_x,
            shaft_y,
            shaft_z,
            color=color,
            linewidth=0.0,
            antialiased=True,
            shade=True,
            alpha=0.88,
            zorder=18,
        )

        cone_z = np.linspace(shaft_end_z, tip_z, 12)
        if abs(tip_z - shaft_end_z) > 1e-12:
            taper = np.abs((tip_z - cone_z) / (tip_z - shaft_end_z))
        else:
            taper = np.zeros_like(cone_z)
        cone_r = head_radius * taper
        cone_x = position[0] + np.outer(cos_t, cone_r)
        cone_y = position[1] + np.outer(sin_t, cone_r)
        cone_z_grid = np.outer(np.ones_like(theta), cone_z)
        axis.plot_surface(
            cone_x,
            cone_y,
            cone_z_grid,
            color=color,
            linewidth=0.0,
            antialiased=True,
            shade=True,
            alpha=0.9,
            zorder=19,
        )
        extent_points.append(np.array([position[0], position[1], tip_z], dtype=float))
        extent_points.append(np.array([position[0], position[1], tail_z], dtype=float))

    if not extent_points:
        return np.empty((0, 3), dtype=float)
    return np.vstack(extent_points)


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    angle = math.radians(float(angle_deg))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]], dtype=float)


def _camera_rotation_matrix(state: _CameraState) -> np.ndarray:
    return (
        _rotation_matrix_z(state.roll)
        @ _rotation_matrix_x(state.elev)
        @ _rotation_matrix_z(state.azim)
    )


def _camera_state_from_vectors(
    position: np.ndarray | tuple[float, float, float],
    focal_point: np.ndarray | tuple[float, float, float],
    view_up: np.ndarray | tuple[float, float, float],
) -> _CameraState:
    camera = np.asarray(position, dtype=float)
    focal = np.asarray(focal_point, dtype=float)
    up_vector = _normalized_vector(
        np.asarray(view_up, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)
    )
    direction = _normalized_vector(camera - focal, np.array([1.0, 0.0, 0.0], dtype=float))
    elev = math.degrees(math.asin(float(np.clip(direction[2], -1.0, 1.0))))
    azim = math.degrees(math.atan2(float(direction[1]), float(direction[0])))

    reference_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(direction, reference_up))) > 0.96:
        reference_up = np.array([0.0, 1.0, 0.0], dtype=float)
    right = _normalized_vector(
        np.cross(direction, reference_up), np.array([1.0, 0.0, 0.0], dtype=float)
    )
    default_up = _normalized_vector(
        np.cross(right, direction), np.array([0.0, 0.0, 1.0], dtype=float)
    )
    sin_roll = float(np.dot(np.cross(default_up, up_vector), direction))
    cos_roll = float(np.clip(np.dot(default_up, up_vector), -1.0, 1.0))
    roll = math.degrees(math.atan2(sin_roll, cos_roll))
    return _CameraState(elev=float(elev), azim=float(azim), roll=float(roll))


def _camera_vectors_from_state(state: _CameraState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elev = math.radians(float(state.elev))
    azim = math.radians(float(state.azim))
    direction = np.array(
        [
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ],
        dtype=float,
    )
    position = 4.0 * direction
    focal_point = np.zeros(3, dtype=float)
    reference_up = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(direction, reference_up))) > 0.96:
        reference_up = np.array([0.0, 1.0, 0.0], dtype=float)
    right = _normalized_vector(
        np.cross(focal_point - position, reference_up), np.array([1.0, 0.0, 0.0], dtype=float)
    )
    up = _normalized_vector(
        np.cross(right, focal_point - position), np.array([0.0, 0.0, 1.0], dtype=float)
    )
    if abs(float(state.roll)) > 1e-12:
        angle = math.radians(float(state.roll))
        view_direction = _normalized_vector(
            focal_point - position, np.array([0.0, 0.0, -1.0], dtype=float)
        )
        up = (
            up * math.cos(angle)
            + np.cross(view_direction, up) * math.sin(angle)
            + view_direction * np.dot(view_direction, up) * (1.0 - math.cos(angle))
        )
    return position, focal_point, up


def _camera_screen_basis(
    camera_position: np.ndarray | tuple[float, float, float],
    focal_point: np.ndarray | tuple[float, float, float],
    view_up: np.ndarray | tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = np.asarray(camera_position, dtype=float)
    focal = np.asarray(focal_point, dtype=float)
    up_hint = _normalized_vector(
        np.asarray(view_up, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)
    )
    view = _normalized_vector(focal - position, np.array([0.0, 0.0, -1.0], dtype=float))
    right = _normalized_vector(np.cross(view, up_hint), np.array([1.0, 0.0, 0.0], dtype=float))
    up = _normalized_vector(np.cross(right, view), np.array([0.0, 0.0, 1.0], dtype=float))
    return view, right, up


class _CrystallographicTriadCanvas(QWidget):
    """Small renderer-neutral crystallographic axis triad."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cell = np.eye(3, dtype=float)
        self._camera_position = np.array([0.0, 0.0, 4.0], dtype=float)
        self._camera_focal_point = np.zeros(3, dtype=float)
        self._camera_up = np.array([0.0, 0.0, 1.0], dtype=float)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self.setFixedWidth(132)
        self.setFixedHeight(130)

    def set_scene(
        self,
        *,
        cell: np.ndarray | None,
        camera_position: np.ndarray | tuple[float, float, float],
        focal_point: np.ndarray | tuple[float, float, float],
        view_up: np.ndarray | tuple[float, float, float],
    ) -> None:
        if cell is not None and np.asarray(cell).shape == (3, 3):
            self._cell = np.asarray(cell, dtype=float)
        self._camera_position = np.asarray(camera_position, dtype=float)
        self._camera_focal_point = np.asarray(focal_point, dtype=float)
        self._camera_up = _normalized_vector(
            np.asarray(view_up, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)
        )
        self.update()

    def draw_idle(self) -> None:
        self.update()

    def refresh_theme(self, *, draw: bool = True) -> None:
        if draw:
            self.update()

    def _project_axis(self, direction: np.ndarray) -> tuple[float, float, float]:
        view = _normalized_vector(
            self._camera_focal_point - self._camera_position,
            np.array([0.0, 0.0, -1.0], dtype=float),
        )
        right = _normalized_vector(
            np.cross(view, self._camera_up), np.array([1.0, 0.0, 0.0], dtype=float)
        )
        up = _normalized_vector(np.cross(right, view), np.array([0.0, 0.0, 1.0], dtype=float))
        return (
            float(np.dot(direction, right)),
            float(np.dot(direction, up)),
            float(np.dot(direction, -view)),
        )

    @staticmethod
    def _darken(color: QColor, factor: float) -> QColor:
        return color.darker(int(max(100.0, factor * 100.0)))

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        style = _viewer_style()

        extent = np.linalg.norm(self._cell, axis=1)
        if float(np.max(extent)) <= 1e-12:
            painter.end()
            return

        rect = self.rect().adjusted(10, 8, -10, -10)
        origin = QPointF(rect.left() + 46.0, rect.bottom() - 28.0)
        scale = min(float(rect.width()), float(rect.height())) * 0.36
        directions = (
            _normalized_vector(self._cell[0], np.array([1.0, 0.0, 0.0], dtype=float)),
            _normalized_vector(self._cell[1], np.array([0.0, 1.0, 0.0], dtype=float)),
            _normalized_vector(self._cell[2], np.array([0.0, 0.0, 1.0], dtype=float)),
        )
        payloads: list[tuple[float, tuple[float, float, float], str, QColor]] = []
        for direction, label, color in zip(directions, ("a", "b", "c"), style.triad_axis_colors):
            projected = self._project_axis(direction)
            payloads.append((projected[2], projected, label, QColor(color)))

        shaft_radius = 4.8
        cone_length = 15.0
        cone_half_width = 7.8

        for _depth, projected, label, color in sorted(
            payloads, key=lambda item: item[0], reverse=True
        ):
            vector_2d = QPointF(float(projected[0] * scale), float(-projected[1] * scale))
            length = math.hypot(vector_2d.x(), vector_2d.y())
            if length <= 1e-6:
                continue
            direction_2d = QPointF(vector_2d.x() / length, vector_2d.y() / length)
            perpendicular = QPointF(-direction_2d.y(), direction_2d.x())
            shaft_end = QPointF(
                origin.x() + direction_2d.x() * max(length - cone_length, length * 0.68),
                origin.y() + direction_2d.y() * max(length - cone_length, length * 0.68),
            )
            tip = QPointF(origin.x() + vector_2d.x(), origin.y() + vector_2d.y())

            shaft_gradient = QLinearGradient(origin, shaft_end)
            shaft_gradient.setColorAt(0.0, self._darken(color, 170.0))
            shaft_gradient.setColorAt(0.5, color)
            shaft_gradient.setColorAt(1.0, color.lighter(125))
            pen = QPen()
            pen.setBrush(shaft_gradient)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setWidthF(shaft_radius)
            painter.setPen(pen)
            painter.drawLine(origin, shaft_end)

            base = shaft_end
            left = QPointF(
                base.x() + perpendicular.x() * cone_half_width,
                base.y() + perpendicular.y() * cone_half_width,
            )
            right = QPointF(
                base.x() - perpendicular.x() * cone_half_width,
                base.y() - perpendicular.y() * cone_half_width,
            )
            cone = QPolygonF([tip, left, right])
            cone_gradient = QLinearGradient(base, tip)
            cone_gradient.setColorAt(0.0, color)
            cone_gradient.setColorAt(1.0, color.lighter(145))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(cone_gradient)
            painter.drawPolygon(cone)

            label_pos = QPointF(
                tip.x() + direction_2d.x() * 12.0 + perpendicular.x() * 3.0,
                tip.y() + direction_2d.y() * 12.0 + perpendicular.y() * 3.0,
            )
            label_rect = QRectF(label_pos.x() - 12.0, label_pos.y() - 12.0, 24.0, 24.0)
            painter.setPen(QColor(style.text_primary))
            font = painter.font()
            font.setFamily(style.triad_label_font_family)
            font.setBold(True)
            font.setPixelSize(12)
            painter.setFont(font)
            painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignCenter), label)

        sphere_radius = 7.0
        sphere_gradient = QRadialGradient(origin.x() - 3.0, origin.y() - 3.0, sphere_radius)
        sphere_gradient.setColorAt(0.0, QColor(style.triad_hub_inner))
        sphere_gradient.setColorAt(0.45, QColor(style.triad_hub_mid))
        sphere_gradient.setColorAt(1.0, QColor(style.triad_hub_outer))
        painter.setPen(QPen(QColor(style.border), 0.8))
        painter.setBrush(sphere_gradient)
        painter.drawEllipse(origin, sphere_radius, sphere_radius)

        painter.end()


def _pyvista_modules():
    try:
        import pyvista as pv
        from pyvistaqt import QtInteractor
    except ImportError:
        return None, None
    return pv, QtInteractor


def _build_pyvista_sphere_mesh(
    pv,
    *,
    radius: float,
    center: np.ndarray | tuple[float, float, float],
    subdivisions: int = 4,
):
    resolution = max(32, 10 * int(subdivisions))
    sphere = pv.Sphere(
        radius=float(radius),
        center=tuple(float(value) for value in np.asarray(center, dtype=float)),
        theta_resolution=resolution,
        phi_resolution=resolution,
    )
    return sphere


def _build_pyvista_atom_glyph_mesh(
    pv,
    *,
    positions: np.ndarray,
    radii: np.ndarray,
    colors: np.ndarray,
    subdivisions: int,
):
    points = pv.PolyData(np.asarray(positions, dtype=float))
    points["glyph_radius"] = np.asarray(radii, dtype=float)
    points["atom_rgb"] = np.asarray(
        np.clip(np.asarray(colors, dtype=float), 0.0, 1.0) * 255.0, dtype=np.uint8
    )
    unit_sphere = _build_pyvista_sphere_mesh(
        pv,
        radius=1.0,
        center=(0.0, 0.0, 0.0),
        subdivisions=int(subdivisions),
    )
    return points.glyph(scale="glyph_radius", orient=False, geom=unit_sphere, factor=1.0)


def _build_pyvista_bond_tube_mesh(
    pv,
    *,
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]],
):
    if not segments:
        return None

    points: list[np.ndarray] = []
    lines: list[int] = []
    colors: list[np.ndarray] = []
    radii: list[float] = []
    for start, end, color, radius in segments:
        start_point = np.asarray(start, dtype=float)
        end_point = np.asarray(end, dtype=float)
        base_index = len(points)
        points.extend([start_point, end_point])
        lines.extend([2, base_index, base_index + 1])
        colors.append(np.asarray(np.clip(color, 0.0, 1.0) * 255.0, dtype=np.uint8))
        radii.append(float(radius))

    poly = pv.PolyData()
    poly.points = np.asarray(points, dtype=float)
    poly.lines = np.asarray(lines, dtype=np.int64)
    poly.cell_data["rgb"] = np.asarray(colors, dtype=np.uint8)
    tube_radius = max(float(np.median(np.asarray(radii, dtype=float))), 1e-4)
    return poly.tube(
        radius=tube_radius,
        n_sides=18,
    )


def _trim_bond_segment_to_surfaces(
    start: np.ndarray,
    end: np.ndarray,
    *,
    start_radius: float,
    end_radius: float,
    margin: float = 0.06,
) -> tuple[np.ndarray, np.ndarray] | None:
    start_point = np.asarray(start, dtype=float)
    end_point = np.asarray(end, dtype=float)
    direction = end_point - start_point
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return None

    axis_dir = direction / length
    trimmed_start = start_point + axis_dir * min(
        float(start_radius) * (1.0 - margin), 0.45 * length
    )
    trimmed_end = end_point - axis_dir * min(float(end_radius) * (1.0 - margin), 0.45 * length)
    if float(np.linalg.norm(trimmed_end - trimmed_start)) <= 1e-8:
        return None
    return trimmed_start, trimmed_end


def _build_pyvista_bond_line_mesh(
    pv,
    *,
    segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
):
    if not segments:
        return None

    points: list[np.ndarray] = []
    lines: list[int] = []
    colors: list[np.ndarray] = []
    for start, end, color in segments:
        start_point = np.asarray(start, dtype=float)
        end_point = np.asarray(end, dtype=float)
        base_index = len(points)
        points.extend([start_point, end_point])
        lines.extend([2, base_index, base_index + 1])
        colors.append(np.asarray(np.clip(color, 0.0, 1.0) * 255.0, dtype=np.uint8))

    poly = pv.PolyData()
    poly.points = np.asarray(points, dtype=float)
    poly.lines = np.asarray(lines, dtype=np.int64)
    poly.cell_data["rgb"] = np.asarray(colors, dtype=np.uint8)
    return poly


def _atom_pyvista_sphere_subdivisions(atom_count: int) -> int:
    if atom_count <= 24:
        return 6
    if atom_count <= 96:
        return 5
    return 4


def _should_use_fast_pyvista_bond_lines(atom_count: int, bond_count: int) -> bool:
    return atom_count >= 192 or bond_count >= 360


def _should_use_fast_pyvista_bond_cylinders(atom_count: int, bond_count: int) -> bool:
    return atom_count >= 96 or bond_count >= 180


def _require_pyvista_backend() -> str:
    pv, qt_interactor = _pyvista_modules()
    if pv is None or qt_interactor is None:
        raise RuntimeError("AtomicStructureViewer requires PyVista and pyvistaqt.")
    return "pyvista"


def _coerce_light_direction(
    direction: np.ndarray | tuple[float, float, float] | list[float] | None,
) -> np.ndarray:
    if direction is None:
        return _DEFAULT_PYVISTA_LIGHT_DIRECTION.copy()
    values = np.asarray(direction, dtype=float).reshape(-1)
    if values.size != 3:
        return _DEFAULT_PYVISTA_LIGHT_DIRECTION.copy()
    return _normalized_vector(values, _DEFAULT_PYVISTA_LIGHT_DIRECTION)


def _configure_pyvista_vesta_style(
    plotter,
    pv,
    *,
    light_direction: np.ndarray | tuple[float, float, float] | list[float] | None = None,
    focal_point: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    if pv is None:
        return

    try:
        plotter.enable_parallel_projection()
    except Exception:  # noqa: BLE001
        pass
    try:
        plotter.disable_shadows()
    except Exception:  # noqa: BLE001
        pass
    try:
        plotter.disable_eye_dome_lighting()
    except Exception:  # noqa: BLE001
        pass
    try:
        plotter.remove_environment_texture()
    except Exception:  # noqa: BLE001
        pass

    renderer = getattr(plotter, "renderer", None)
    if renderer is not None:
        for method_name in (
            "AutomaticLightCreationOff",
            "UseImageBasedLightingOff",
            "UseSphericalHarmonicsOff",
        ):
            method = getattr(renderer, method_name, None)
            if method is None:
                continue
            try:
                method()
            except Exception:  # noqa: BLE001
                pass
        try:
            renderer.SetEnvironmentTexture(None)
        except Exception:  # noqa: BLE001
            pass

    try:
        plotter.remove_all_lights()
    except Exception:  # noqa: BLE001
        pass

    direction = _coerce_light_direction(light_direction)
    light = pv.Light()
    light.set_camera_light()
    light.positional = False
    # Keep the light in camera coordinates so the atoms retain a stable,
    # simple 3D cue while rotating.
    light.position = tuple(float(value) for value in direction)
    light.focal_point = (0.0, 0.0, 0.0)
    light.intensity = 0.62
    light.ambient_color = (1.0, 1.0, 1.0)
    light.diffuse_color = (1.0, 1.0, 1.0)
    light.specular_color = (1.0, 1.0, 1.0)
    plotter.add_light(light)


class _PyVistaCanvas(QWidget):
    """Small Qt wrapper that gives ``QtInteractor`` a ``draw_idle`` API."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._closed = False
        self._refreshing_theme = False
        pv, qt_interactor = _pyvista_modules()
        if pv is None or qt_interactor is None:
            raise RuntimeError("PyVista backend requested but PyVista is not installed.")

        # auto_update=False disables pyvistaqt's internal render_timer, which
        # otherwise calls plotter.render() ~5x/sec forever regardless of
        # visibility. We drive rendering explicitly (_render/draw_idle), so the
        # timer is pure waste and pegged the CPU even on tabs without the view.
        self._plotter = qt_interactor(self, auto_update=False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        self._plotter.setPalette(self.palette())
        _configure_transparent_pyvista_background(self._plotter)
        self._plotter.enable_parallel_projection()
        try:
            self._plotter.enable_anti_aliasing("ssaa")
        except Exception:  # noqa: BLE001
            pass
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plotter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def plotter(self):
        return self._plotter

    def draw_idle(self) -> None:
        if self._closed or self._plotter is None:
            return
        self._plotter.render()

    def refresh_theme(self, *, draw: bool = True) -> None:
        """Apply the current palette; ``draw=False`` leaves the redraw to the caller."""

        plotter = getattr(self, "_plotter", None)
        if self._closed or self._refreshing_theme or plotter is None:
            return
        self._refreshing_theme = True
        try:
            plotter.setPalette(self.palette())
            _configure_transparent_pyvista_background(plotter)
            if draw:
                self.draw_idle()
        finally:
            self._refreshing_theme = False

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        plotter = self._plotter
        self._plotter = None
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        if plotter is None:
            return
        try:
            plotter.blockSignals(True)
        except Exception:  # noqa: BLE001
            pass
        try:
            plotter.setUpdatesEnabled(False)
            plotter.hide()
        except Exception:  # noqa: BLE001
            pass
        render_timer = getattr(plotter, "render_timer", None)
        if render_timer is not None:
            try:
                render_timer.stop()
            except Exception:  # noqa: BLE001
                pass
        render_window = getattr(plotter, "ren_win", None)
        if render_window is not None:
            try:
                render_window.SetAbortRender(1)
            except Exception:  # noqa: BLE001
                pass
            try:
                render_window.Finalize()
            except Exception:  # noqa: BLE001
                pass
        close = getattr(plotter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        try:
            plotter.setParent(None)
            plotter.deleteLater()
        except Exception:  # noqa: BLE001
            pass


class _PyVistaTriadCanvas(QWidget):
    """Mini PyVista scene that mirrors the main camera orientation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._closed = False
        self._refreshing_theme = False
        pv, qt_interactor = _pyvista_modules()
        if pv is None or qt_interactor is None:
            raise RuntimeError("PyVista triad requested but PyVista is not installed.")

        self._pv = pv
        # See _PyVistaCanvas: disable the always-on render_timer (auto_update).
        self._plotter = qt_interactor(self, auto_update=False)
        self._plotter.enable_parallel_projection()
        try:
            self._plotter.enable_anti_aliasing("ssaa")
        except Exception:  # noqa: BLE001
            pass

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._plotter.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFixedWidth(132)
        self.setFixedHeight(130)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plotter)

        self._cell = np.eye(3, dtype=float)
        self._camera_position = np.array([0.0, 0.0, 4.0], dtype=float)
        self._camera_focal_point = np.zeros(3, dtype=float)
        self._camera_up = np.array([0.0, 0.0, 1.0], dtype=float)
        self._light_direction = _DEFAULT_PYVISTA_LIGHT_DIRECTION.copy()
        self._style_signature: tuple[str, ...] | None = None

        self._configure_background()
        self._build_axes_scene()

    def _configure_background(self) -> None:
        self._plotter.setPalette(self.palette())
        _configure_transparent_pyvista_background(self._plotter)

    def _configure_lighting(self) -> None:
        _configure_pyvista_vesta_style(
            self._plotter,
            self._pv,
            light_direction=self._light_direction,
            focal_point=(0.0, 0.0, 0.0),
        )

    def _build_axes_scene(self) -> None:
        self._plotter.clear()
        self._configure_background()
        self._configure_lighting()
        style = _viewer_style()
        self._style_signature = _viewer_style_signature()

        if float(np.max(np.linalg.norm(self._cell, axis=1))) <= 1e-12:
            self._apply_camera()
            return

        directions = (
            _normalized_vector(self._cell[0], np.array([1.0, 0.0, 0.0], dtype=float)),
            _normalized_vector(self._cell[1], np.array([0.0, 1.0, 0.0], dtype=float)),
            _normalized_vector(self._cell[2], np.array([0.0, 0.0, 1.0], dtype=float)),
        )
        colors = style.triad_axis_colors
        labels = ("a", "b", "c")
        shaft_radius = 0.06
        shaft_length = 0.72
        cone_radius = 0.13
        cone_length = 0.24
        hub_radius = 0.11
        label_offset = 0.08
        label_points: list[np.ndarray] = []

        hub = _build_pyvista_sphere_mesh(
            self._pv, radius=hub_radius, center=(0.0, 0.0, 0.0), subdivisions=3
        )
        self._plotter.add_mesh(
            hub,
            color=style.triad_hub_mid,
            smooth_shading=True,
            ambient=0.52,
            diffuse=0.34,
            specular=0.0,
            specular_power=1.0,
            pbr=False,
            pickable=False,
            render=False,
        )

        for direction, color in zip(directions, colors):
            shaft_start = direction * (hub_radius * 0.55)
            shaft_end = direction * shaft_length
            shaft_center = 0.5 * (shaft_start + shaft_end)
            shaft_height = float(np.linalg.norm(shaft_end - shaft_start))
            shaft = self._pv.Cylinder(
                center=tuple(float(value) for value in shaft_center),
                direction=tuple(float(value) for value in direction),
                radius=shaft_radius,
                height=shaft_height,
                resolution=28,
            )
            cone_center = direction * (shaft_length + 0.5 * cone_length)
            cone = self._pv.Cone(
                center=tuple(float(value) for value in cone_center),
                direction=tuple(float(value) for value in direction),
                height=cone_length,
                radius=cone_radius,
                resolution=36,
            )
            for mesh in (shaft, cone):
                self._plotter.add_mesh(
                    mesh,
                    color=color,
                    smooth_shading=True,
                    ambient=0.46,
                    diffuse=0.36,
                    specular=0.0,
                    specular_power=1.0,
                    pbr=False,
                    pickable=False,
                    render=False,
                )
            label_points.append(direction * (shaft_length + cone_length + label_offset))

        self._plotter.add_point_labels(
            np.asarray(label_points, dtype=float),
            list(labels),
            font_size=15,
            text_color=style.text_primary,
            font_family="arial",
            shape=None,
            show_points=False,
            always_visible=True,
            pickable=False,
            render=False,
        )
        self._apply_camera()

    def _apply_camera(self) -> None:
        direction = _normalized_vector(
            self._camera_position - self._camera_focal_point,
            np.array([1.0, 0.0, 0.0], dtype=float),
        )
        position = 3.8 * direction
        self._plotter.camera.position = tuple(float(value) for value in position)
        self._plotter.camera.focal_point = (0.0, 0.0, 0.0)
        self._plotter.camera.up = tuple(float(value) for value in self._camera_up)
        self._plotter.camera.parallel_scale = 1.32
        self._plotter.reset_camera_clipping_range()

    def set_scene(
        self,
        *,
        cell: np.ndarray | None,
        camera_position: np.ndarray | tuple[float, float, float],
        focal_point: np.ndarray | tuple[float, float, float],
        view_up: np.ndarray | tuple[float, float, float],
    ) -> None:
        rebuild_scene = self._style_signature != _viewer_style_signature()
        if cell is not None and np.asarray(cell).shape == (3, 3):
            cell_array = np.asarray(cell, dtype=float)
            if not np.allclose(cell_array, self._cell):
                self._cell = cell_array
                rebuild_scene = True
        if rebuild_scene:
            self._build_axes_scene()
        self._camera_position = np.asarray(camera_position, dtype=float)
        self._camera_focal_point = np.asarray(focal_point, dtype=float)
        self._camera_up = _normalized_vector(
            np.asarray(view_up, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)
        )
        self._configure_background()
        self._apply_camera()

    def set_light_direction(
        self, direction: np.ndarray | tuple[float, float, float] | list[float]
    ) -> None:
        self._light_direction = _coerce_light_direction(direction)
        self._build_axes_scene()

    def refresh_theme(self, *, draw: bool = True) -> None:
        """Rebuild the axes in the current palette; ``draw=False`` leaves the redraw to the caller."""

        plotter = getattr(self, "_plotter", None)
        if self._closed or self._refreshing_theme or plotter is None:
            return
        self._refreshing_theme = True
        try:
            self._build_axes_scene()
            if draw:
                self.draw_idle()
        finally:
            self._refreshing_theme = False

    def draw_idle(self) -> None:
        if self._closed or self._plotter is None:
            return
        self._plotter.render()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        plotter = self._plotter
        self._plotter = None
        self.blockSignals(True)
        self.setUpdatesEnabled(False)
        if plotter is None:
            return
        try:
            plotter.blockSignals(True)
        except Exception:  # noqa: BLE001
            pass
        try:
            plotter.setUpdatesEnabled(False)
            plotter.hide()
        except Exception:  # noqa: BLE001
            pass
        render_timer = getattr(plotter, "render_timer", None)
        if render_timer is not None:
            try:
                render_timer.stop()
            except Exception:  # noqa: BLE001
                pass
        render_window = getattr(plotter, "ren_win", None)
        if render_window is not None:
            try:
                render_window.SetAbortRender(1)
            except Exception:  # noqa: BLE001
                pass
            try:
                render_window.Finalize()
            except Exception:  # noqa: BLE001
                pass
        close = getattr(plotter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
        try:
            plotter.setParent(None)
            plotter.deleteLater()
        except Exception:  # noqa: BLE001
            pass


class AtomicStructureViewer(QWidget):
    """Render a contained, ASE-style 3D view of a structure."""

    atom_selected = Signal(int)
    selected_atoms_changed = Signal(object)
    view_state_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._structure: Structure | None = None
        self._atoms: Atoms | None = None
        self._selected_index: int | None = None
        self._selected_indexes: list[int] = []
        self._spins: np.ndarray | None = None
        self._isosurface_vertices: np.ndarray | None = None
        self._isosurface_faces: np.ndarray | None = None
        self._isosurface_color = _viewer_style().isosurface
        self._isosurface_alpha = 0.24
        self._isosurface_specs: list[_ViewerIsosurfaceSpec] = []
        self._slice_specs: list[_ViewerSliceSpec] = []
        self._point_overlay_specs: list[_ViewerPointOverlaySpec] = []
        self._line_overlay_specs: list[_ViewerLineOverlaySpec] = []
        self._hide_atoms_in_front_of_planes = False
        self._camera_state = _CameraState(*_DEFAULT_VIEW, _DEFAULT_VIEW_ROLL)
        self._view = _DEFAULT_VIEW
        self._view_roll = _DEFAULT_VIEW_ROLL
        self._limits: (
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
        ) = None
        self._axis = _AxesProxy(self._camera_state)
        self._triad_axis: _AxesProxy | None = None
        self._latched_selection_modifier = False
        self._native_selection_modifier = False
        self._pyvista_left_press_position: tuple[int, int] | None = None
        self._pyvista_left_press_modifier = False
        self._pyvista_left_press_camera_signature: tuple[float, ...] | None = None
        self._pyvista_atom_positions = np.empty((0, 3), dtype=float)
        self._pyvista_atom_radii = np.empty((0,), dtype=float)
        self._pyvista_atom_colors = np.empty((0, 3), dtype=float)
        self._pyvista_all_atom_positions = np.empty((0, 3), dtype=float)
        self._pyvista_all_atom_radii = np.empty((0,), dtype=float)
        self._pyvista_all_atom_colors = np.empty((0, 3), dtype=float)
        self._pyvista_atom_lookup_indices = np.empty((0,), dtype=int)
        self._pyvista_atom_actor = None
        self._pyvista_isosurface_actor = None
        self._pyvista_isosurface_actors: list[object] = []
        self._pyvista_slice_actors: list[object] = []
        self._pyvista_point_overlay_actors: list[object] = []
        self._pyvista_line_overlay_actors: list[object] = []
        self._pyvista_selection_actor = None
        self._pyvista_pick_actor = None
        self._pyvista_bond_actors: list[object] = []
        self._pyvista_spin_actors: list[object] = []
        self._pyvista_last_camera_signature: tuple[float, ...] | None = None
        self._pyvista_camera_snapshot: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = (
            None
        )
        self._pyvista_light_direction = _DEFAULT_PYVISTA_LIGHT_DIRECTION.copy()
        self._shutting_down = False
        self._refreshing_theme = False
        self._theme_refresh_pending = False
        self._style_signature = _viewer_style_signature()
        # Drawing deferred while a batch_update() block is open or the viewer is not
        # visible: a full scene rebuild, a redraw of in-place actor changes, and a
        # redraw of the triad. A hidden viewer keeps them until showEvent.
        self._batch_depth = 0
        self._render_pending = False
        self._draw_pending = False
        self._triad_draw_pending = False

        self._backend_name = _require_pyvista_backend()
        self._canvas = _PyVistaCanvas(self)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.installEventFilter(self)
        try:
            self._triad_canvas = _PyVistaTriadCanvas(self)
            self._triad_canvas.set_light_direction(self._pyvista_light_direction)
        except RuntimeError:
            self._triad_canvas = _CrystallographicTriadCanvas(self)

        self._show_cell = QCheckBox("Cell")
        self._show_cell.setChecked(True)
        self._show_bonds = QCheckBox("Bonds")
        self._show_bonds.setChecked(False)
        self._show_spins = QCheckBox("Spins")
        self._show_spins.setChecked(False)
        self._show_bonds.setToolTip("Show or hide nearest-neighbor bonds.")
        self._show_spins.setToolTip("Show magnetic spin arrows.")
        self._view_x_button = QPushButton("a")
        self._view_y_button = QPushButton("b")
        self._view_z_button = QPushButton("c")
        self._view_x_button.setToolTip("Look along the a lattice vector.")
        self._view_y_button.setToolTip("Look along the b lattice vector.")
        self._view_z_button.setToolTip("Look along the c lattice vector.")
        for button in (self._view_x_button, self._view_y_button, self._view_z_button):
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.setFixedWidth(_compact_text_button_width(button))
        self._reset_view_button = QPushButton("Reset View")
        self._reset_view_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._reset_view_button.setMinimumWidth(self._reset_view_button.sizeHint().width() + 8)
        self._reset_view_button.setToolTip("Reset orientation and zoom to defaults.")
        self._interaction_hint = _WrappingHintLabel(
            "Left click atom: select | Shift+click atom: add/remove | Left click empty: clear | Right drag: rotate | "
            "Middle drag: pan | Wheel: zoom"
        )
        self._interaction_hint.setWordWrap(True)
        self._interaction_hint.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        style_hint_text(self._interaction_hint)
        self._selection_label = _ElidingLabel("Selected atom: -")
        self._selection_label.setWordWrap(False)
        self._selection_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        selection_height = self._selection_label.sizeHint().height()
        self._selection_label.setFixedHeight(selection_height)
        self._selection_label.setToolTip("")

        self._build_layout()

        self._show_cell.toggled.connect(self._redraw_preserving_view)
        self._show_bonds.toggled.connect(self._redraw_preserving_view)
        self._show_spins.toggled.connect(self._redraw_preserving_view)
        self._view_x_button.clicked.connect(lambda: self._set_named_view("a"))
        self._view_y_button.clicked.connect(lambda: self._set_named_view("b"))
        self._view_z_button.clicked.connect(lambda: self._set_named_view("c"))
        self._reset_view_button.clicked.connect(self._reset_view)

        self._configure_pyvista_interaction()
        self._canvas.plotter.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.plotter.installEventFilter(self)
        self._pyvista_camera_timer = QTimer(self)
        self._pyvista_camera_timer.setInterval(40)
        self._pyvista_camera_timer.timeout.connect(self._poll_pyvista_camera)
        # Do not start here: the timer is started lazily in showEvent and paused
        # in hideEvent, so a viewer that is never shown (e.g. its tab is never
        # opened) never polls. It only needs to tick while actually visible.

        self._render()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _refresh_interaction_hint_layout(self) -> None:
        self._interaction_hint._sync_minimum_height()
        self._interaction_hint.updateGeometry()
        layout = self.layout()
        if layout is not None:
            layout.activate()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        # Draw changes made while hidden first, so the camera poll below reads the
        # camera of the scene now on screen rather than the stale one.
        self._draw_stale_scene()
        # Camera polling only matters while the view is visible (it syncs the
        # axes/triad to user-driven camera moves). Resume it on show; it is
        # paused in hideEvent so it does not tick on tabs without the view.
        if not self._shutting_down and not self._pyvista_camera_timer.isActive():
            self._pyvista_camera_timer.start()
            # Sync the axes/triad to the current camera immediately rather than
            # waiting up to one tick: if the camera/structure changed while the
            # view was hidden (the poll was paused), this avoids a brief stale
            # gizmo on the first frame after the tab is shown.
            self._poll_pyvista_camera()
        # Bound to the viewer, so the call is dropped if it is deleted first.
        QTimer.singleShot(0, self, self._refresh_interaction_hint_layout)

    def _draw_stale_scene(self) -> None:
        """Draw what changed while the viewer was hidden: once, now that it shows."""

        if self._shutting_down or self._defers_drawing():
            return
        self._flush_batched_render()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().hideEvent(event)
        self._pyvista_camera_timer.stop()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._refresh_interaction_hint_layout()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.StyleChange,
            QEvent.Type.PaletteChange,
            QEvent.Type.ApplicationPaletteChange,
        }:
            self._schedule_theme_refresh()

    def _schedule_theme_refresh(self) -> None:
        if (
            getattr(self, "_shutting_down", False)
            or getattr(self, "_refreshing_theme", False)
            or getattr(self, "_theme_refresh_pending", False)
            or not hasattr(self, "_interaction_hint")
            or getattr(self, "_style_signature", None) == _viewer_style_signature()
        ):
            return
        self._theme_refresh_pending = True
        QTimer.singleShot(0, self._refresh_theme_if_needed)

    def _refresh_theme_if_needed(self) -> None:
        self._theme_refresh_pending = False
        if getattr(self, "_style_signature", None) == _viewer_style_signature():
            return
        self.refresh_theme()

    def refresh_theme(self) -> None:
        """Refresh palette-dependent VTK and Qt painting state."""

        if (
            getattr(self, "_shutting_down", False)
            or getattr(self, "_refreshing_theme", False)
            or not hasattr(self, "_interaction_hint")
        ):
            return
        if not self.isVisible():
            # The style signature stays old, so showEvent refreshes the theme then.
            return
        self._refreshing_theme = True
        try:
            # One batch: the canvases leave their redraw to the scene rebuild, and a
            # rebuild already pending (from changes made while hidden) keeps the
            # camera it will fit rather than the stale live one.
            with self.batch_update():
                self._style_signature = _viewer_style_signature()
                style_hint_text(self._interaction_hint)
                self._refresh_interaction_hint_layout()
                canvas = getattr(self, "_canvas", None)
                if hasattr(canvas, "refresh_theme"):
                    canvas.refresh_theme(draw=False)
                triad_canvas = getattr(self, "_triad_canvas", None)
                if hasattr(triad_canvas, "refresh_theme"):
                    triad_canvas.refresh_theme(draw=False)
                self._store_view()
                self._render()
        finally:
            self._refreshing_theme = False

    def _defers_drawing(self) -> bool:
        """Whether drawing waits: for the end of a batch, or for the viewer to show."""

        return bool(self._batch_depth) or not self.isVisible()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        canvas = getattr(self, "_canvas", None)
        plotter = getattr(canvas, "plotter", None)
        if watched is canvas or watched is plotter:
            event_type = event.type()
            if (
                event_type == QEvent.Type.MouseButtonPress
                and getattr(event, "button", lambda: Qt.MouseButton.NoButton)()
                == Qt.MouseButton.LeftButton
            ):
                modifiers = getattr(event, "modifiers", lambda: Qt.KeyboardModifier.NoModifier)()
                self._native_selection_modifier = bool(
                    modifiers & Qt.KeyboardModifier.ShiftModifier
                )
                self._latched_selection_modifier = self._native_selection_modifier
                focus_target = watched if isinstance(watched, QWidget) else self._canvas
                if isinstance(focus_target, QWidget):
                    focus_target.setFocus(Qt.FocusReason.MouseFocusReason)
            elif event_type in {QEvent.Type.KeyPress, QEvent.Type.KeyRelease}:
                modifiers = getattr(event, "modifiers", lambda: Qt.KeyboardModifier.NoModifier)()
                self._native_selection_modifier = bool(
                    modifiers & Qt.KeyboardModifier.ShiftModifier
                )
        return super().eventFilter(watched, event)

    def shutdown(self) -> None:
        """Release backend resources explicitly before parent dialogs tear down."""

        if self._shutting_down:
            return
        self._shutting_down = True
        self.blockSignals(True)
        self.setUpdatesEnabled(False)

        timer = getattr(self, "_pyvista_camera_timer", None)
        if timer is not None:
            timer.stop()
            timeout = getattr(timer, "timeout", None)
            try:
                if timeout is not None:
                    timeout.disconnect(self._poll_pyvista_camera)
            except (RuntimeError, TypeError):
                pass

        canvas = getattr(self, "_canvas", None)
        if canvas is not None:
            remove_filter = getattr(canvas, "removeEventFilter", None)
            try:
                if callable(remove_filter):
                    remove_filter(self)
            except RuntimeError:
                pass
            plotter = getattr(canvas, "plotter", None)
            if plotter is not None:
                remove_filter = getattr(plotter, "removeEventFilter", None)
                try:
                    if callable(remove_filter):
                        remove_filter(self)
                except RuntimeError:
                    pass

        _shutdown_pyvista_widget(self._triad_canvas)
        _shutdown_pyvista_widget(self._canvas)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(6)

        # Two groups that share a line when the viewer is wide and wrap onto two
        # lines when it is narrow, so the viewer can sit in a narrow splitter
        # pane without clipping its controls.
        display_group = QWidget(self)
        display_group.setObjectName("atomicViewerDisplayControls")
        display_layout = QHBoxLayout(display_group)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(6)
        display_layout.addWidget(self._show_cell)
        display_layout.addSpacing(4)
        display_layout.addWidget(self._show_bonds)
        display_layout.addSpacing(4)
        display_layout.addWidget(self._show_spins)

        orientation_group = QWidget(self)
        orientation_group.setObjectName("atomicViewerOrientationControls")
        orientation_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        orientation_layout = QHBoxLayout(orientation_group)
        orientation_layout.setContentsMargins(0, 0, 0, 0)
        orientation_layout.setSpacing(6)
        orientation_layout.addWidget(QLabel("Along:", orientation_group))
        orientation_layout.addWidget(self._view_x_button)
        orientation_layout.addWidget(self._view_y_button)
        orientation_layout.addWidget(self._view_z_button)
        orientation_layout.addStretch(1)
        orientation_layout.addWidget(self._reset_view_button)

        controls = _WrappingRowLayout(spacing=12)
        controls.addWidget(display_group)
        controls.addWidget(orientation_group)
        self._controls_layout = controls

        layout.addLayout(controls)
        layout.addWidget(self._interaction_hint)

        viewer_row = QHBoxLayout()
        viewer_row.setContentsMargins(0, 0, 0, 0)
        viewer_row.setSpacing(8)

        triad_column = QVBoxLayout()
        triad_column.setContentsMargins(0, 0, 0, 0)
        triad_column.setSpacing(0)
        triad_column.addStretch(1)
        triad_column.addWidget(
            self._triad_canvas,
            alignment=(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom),
        )

        viewer_row.addLayout(triad_column)
        viewer_row.addWidget(self._canvas, stretch=1)

        layout.addLayout(viewer_row, stretch=1)
        layout.addWidget(self._selection_label)

    def _configure_pyvista_interaction(self) -> None:
        plotter = self._canvas.plotter
        plotter.enable_custom_trackball_style(right="rotate", middle="pan")
        plotter.iren.picker = "point"
        plotter.iren.add_observer("LeftButtonPressEvent", self._on_pyvista_left_button_press)
        plotter.iren.add_observer("LeftButtonReleaseEvent", self._on_pyvista_left_button_release)

    @_renders_once
    def set_structure(self, structure: Structure | None) -> None:
        """Load a new structure into the viewer."""
        # Deferred to the end of this batch; marking it first skips in-place updates
        # (such as the spin toggle below) that the rebuild would discard.
        self._render()
        self._structure = structure
        self._atoms = self._to_atoms(structure) if structure is not None else None
        self._spins = self._extract_spins(structure)
        self._isosurface_vertices = None
        self._isosurface_faces = None
        self._isosurface_specs = []
        self._slice_specs = []
        self._point_overlay_specs = []
        self._line_overlay_specs = []
        self._hide_atoms_in_front_of_planes = False
        self._set_spin_visibility_default_from_data()
        self._selected_index = None
        self._selected_indexes = []
        self._latched_selection_modifier = False
        self._native_selection_modifier = False
        self._pyvista_left_press_position = None
        self._pyvista_left_press_modifier = False
        self._pyvista_left_press_camera_signature = None
        self._camera_state = _CameraState(*_DEFAULT_VIEW, _DEFAULT_VIEW_ROLL)
        self._axis.set_from_state(self._camera_state)
        self._triad_axis = None
        self._pyvista_atom_positions = np.empty((0, 3), dtype=float)
        self._pyvista_atom_radii = np.empty((0,), dtype=float)
        self._pyvista_atom_colors = np.empty((0, 3), dtype=float)
        self._pyvista_all_atom_positions = np.empty((0, 3), dtype=float)
        self._pyvista_all_atom_radii = np.empty((0,), dtype=float)
        self._pyvista_all_atom_colors = np.empty((0, 3), dtype=float)
        self._pyvista_atom_lookup_indices = np.empty((0,), dtype=int)
        self._pyvista_atom_actor = None
        self._pyvista_isosurface_actor = None
        self._pyvista_isosurface_actors = []
        self._pyvista_slice_actors = []
        self._pyvista_point_overlay_actors = []
        self._view = _DEFAULT_VIEW
        self._view_roll = _DEFAULT_VIEW_ROLL
        self._limits = None
        self._pyvista_camera_snapshot = None
        self._pyvista_selection_actor = None
        self._pyvista_pick_actor = None
        self._pyvista_bond_actors = []
        self._pyvista_spin_actors = []
        self._selection_label.setText("Selected atom: -")
        self._render()

    @_renders_once
    def set_structure_preserving_view(self, structure: Structure | None) -> None:
        """Load a new structure while preserving camera, zoom, and spin visibility."""

        if structure is None:
            self.set_structure(None)
            return

        previous_camera = deepcopy(getattr(self, "_camera_state", None))
        previous_limits = deepcopy(getattr(self, "_limits", None))
        previous_show_spins = bool(self._show_spins.isChecked())
        previous_selected = list(getattr(self, "_selected_indexes", []))
        live_pyvista_snapshot = self._current_pyvista_camera_snapshot()
        previous_pyvista_snapshot = deepcopy(
            live_pyvista_snapshot
            if live_pyvista_snapshot is not None
            else getattr(self, "_pyvista_camera_snapshot", None)
        )
        had_existing_structure = (
            self._structure is not None and self._atoms is not None and len(self._atoms) > 0
        )

        self.set_structure(structure)

        if self._show_spins.isChecked() != previous_show_spins:
            blocked = self._show_spins.blockSignals(True)
            self._show_spins.setChecked(previous_show_spins)
            self._show_spins.blockSignals(blocked)

        max_atom_index = len(structure.species)
        normalized_selection = [
            int(index) for index in previous_selected if 0 <= int(index) < max_atom_index
        ]
        self._selected_indexes = normalized_selection
        self._selected_index = None if not normalized_selection else int(normalized_selection[-1])
        self._update_selection_label()

        if not had_existing_structure or previous_camera is None:
            return
        self._camera_state = previous_camera
        self._limits = previous_limits
        self._axis.set_from_state(previous_camera)
        if self._triad_axis is not None:
            self._triad_axis.set_from_state(previous_camera)
        self._view = (previous_camera.elev, previous_camera.azim)
        self._view_roll = previous_camera.roll
        self._pyvista_camera_snapshot = previous_pyvista_snapshot
        self._render()

    @_renders_once
    def update_structure_geometry_preserving_view(self, structure: Structure | None) -> None:
        """Update atom positions for a structurally equivalent model without resetting the current view."""

        if structure is None:
            self.set_structure(None)
            return
        if not self._can_update_structure_geometry_in_place(structure):
            self.set_structure_preserving_view(structure)
            return

        self._structure = structure
        self._atoms = self._to_atoms(structure)
        self._spins = self._extract_spins(structure)
        max_atom_index = len(structure.species)
        self._selected_indexes = [
            index for index in self._selected_indexes if 0 <= int(index) < max_atom_index
        ]
        self._selected_index = (
            None if not self._selected_indexes else int(self._selected_indexes[-1])
        )
        self._update_selection_label()

        if self._render_deferred():
            return
        if self._try_update_pyvista_atom_geometry():
            return
        self.set_structure_preserving_view(structure)

    @_renders_once
    def set_spins(self, spins: list[float] | tuple[float, ...] | np.ndarray | None) -> None:
        """Replace current per-atom scalar spins or vector moments and redraw."""
        if self._atoms is None or spins is None:
            self._spins = None
        else:
            spin_array = np.asarray(spins, dtype=float)
            valid_shape = spin_array.shape == (len(self._atoms),) or spin_array.shape == (
                len(self._atoms),
                3,
            )
            self._spins = spin_array.copy() if valid_shape else None
        self._store_view()
        # Deferred: the rebuild also draws the spin toggle below.
        self._render()
        if self._spins is not None and np.any(np.abs(self._spins) > 1e-10):
            self._show_spins.setChecked(True)

    def set_moment_vectors(
        self,
        moments: list[list[float]] | tuple[tuple[float, float, float], ...] | np.ndarray | None,
    ) -> None:
        """Replace current per-atom magnetic moment vectors and redraw."""
        self.set_spins(moments)

    def set_isosurface(
        self,
        vertices: list[list[float]] | tuple[tuple[float, float, float], ...] | np.ndarray | None,
        faces: list[list[int]] | tuple[tuple[int, int, int], ...] | np.ndarray | None,
        *,
        color: str | None = None,
        alpha: float = 0.24,
    ) -> None:
        """Attach or clear an isosurface triangle mesh overlay."""

        if vertices is None or faces is None:
            self.set_isosurfaces([])
            return
        self.set_isosurfaces(
            [
                {
                    "vertices": vertices,
                    "faces": faces,
                    "color": _viewer_style().isosurface if color is None else str(color),
                    "alpha": float(alpha),
                }
            ]
        )

    @_renders_once
    def set_isosurfaces(self, surfaces: Iterable[Mapping[str, object]] | None) -> None:
        """Attach zero or more isosurface triangle mesh overlays."""

        specs: list[_ViewerIsosurfaceSpec] = []
        if surfaces is not None:
            for surface in surfaces:
                if not isinstance(surface, Mapping):
                    continue
                vertices = surface.get("vertices")
                faces = surface.get("faces")
                if vertices is None or faces is None:
                    continue
                vertices_array = np.asarray(vertices, dtype=float)
                faces_array = np.asarray(faces, dtype=int)
                if (
                    vertices_array.ndim != 2
                    or vertices_array.shape[1] != 3
                    or faces_array.ndim != 2
                    or faces_array.shape[1] != 3
                ):
                    continue
                color = surface.get("color")
                alpha = surface.get("alpha", 0.24)
                specs.append(
                    _ViewerIsosurfaceSpec(
                        vertices=vertices_array,
                        faces=faces_array,
                        color=_viewer_style().isosurface if color is None else str(color),
                        alpha=float(alpha),
                    )
                )

        self._isosurface_specs = specs
        if specs:
            primary = specs[0]
            self._isosurface_vertices = primary.vertices
            self._isosurface_faces = primary.faces
            self._isosurface_color = primary.color
            self._isosurface_alpha = float(primary.alpha)
        else:
            self._isosurface_vertices = None
            self._isosurface_faces = None
            self._isosurface_color = _viewer_style().isosurface
            self._isosurface_alpha = 0.24
        self._store_view()
        self._update_scene(self._try_update_pyvista_isosurface_overlay)

    @_renders_once
    def set_slice_planes(self, slices: Iterable[Mapping[str, object]] | None) -> None:
        """Attach zero or more slice overlays."""

        specs: list[_ViewerSliceSpec] = []
        if slices is not None:
            for entry in slices:
                if not isinstance(entry, Mapping):
                    continue
                x_coordinates = np.asarray(entry.get("x_coordinates"), dtype=float)
                y_coordinates = np.asarray(entry.get("y_coordinates"), dtype=float)
                z_coordinates = np.asarray(entry.get("z_coordinates"), dtype=float)
                values = np.asarray(entry.get("values"), dtype=float)
                anchor = np.asarray(entry.get("anchor"), dtype=float)
                normal = np.asarray(entry.get("normal"), dtype=float)
                if (
                    x_coordinates.ndim != 2
                    or y_coordinates.shape != x_coordinates.shape
                    or z_coordinates.shape != x_coordinates.shape
                    or values.shape != x_coordinates.shape
                    or anchor.shape != (3,)
                    or normal.shape != (3,)
                ):
                    continue
                try:
                    minimum = float(entry.get("minimum"))
                    maximum = float(entry.get("maximum"))
                except (TypeError, ValueError):
                    continue
                specs.append(
                    _ViewerSliceSpec(
                        x_coordinates=x_coordinates,
                        y_coordinates=y_coordinates,
                        z_coordinates=z_coordinates,
                        values=values,
                        anchor=anchor,
                        normal=_normalized_vector(normal, np.array([0.0, 0.0, 1.0], dtype=float)),
                        color_limits=(minimum, maximum),
                        cmap=str(entry.get("cmap", "viridis")),
                        alpha=float(entry.get("alpha", 1.0)),
                    )
                )
        self._slice_specs = specs
        self._store_view()
        self._update_scene(self._try_update_pyvista_slice_overlay)

    @_renders_once
    def set_point_overlays(self, overlays: Iterable[Mapping[str, object]] | None) -> None:
        """Attach zero or more free-position point overlays."""

        specs: list[_ViewerPointOverlaySpec] = []
        if overlays is not None:
            for entry in overlays:
                if not isinstance(entry, Mapping):
                    continue
                positions = np.asarray(entry.get("positions"), dtype=float)
                if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
                    continue
                alpha_raw = entry.get("alpha", 0.92)
                try:
                    alpha = float(alpha_raw)
                except (TypeError, ValueError):
                    alpha = 0.92

                colors_raw = entry.get("colors")
                colors: np.ndarray
                if isinstance(colors_raw, str):
                    qcolor = QColor(colors_raw)
                    if not qcolor.isValid():
                        continue
                    colors = np.tile(
                        np.asarray(qcolor.getRgbF()[:3], dtype=float), (len(positions), 1)
                    )
                elif (
                    isinstance(colors_raw, (list, tuple))
                    and colors_raw
                    and isinstance(colors_raw[0], str)
                ):
                    color_rows: list[np.ndarray] = []
                    if len(colors_raw) != len(positions):
                        continue
                    for raw_color in colors_raw:
                        qcolor = QColor(str(raw_color))
                        if not qcolor.isValid():
                            color_rows = []
                            break
                        color_rows.append(np.asarray(qcolor.getRgbF()[:3], dtype=float))
                    if len(color_rows) != len(positions):
                        continue
                    colors = np.asarray(color_rows, dtype=float)
                else:
                    colors = np.asarray(colors_raw, dtype=float)
                    if colors.ndim == 1 and colors.shape == (3,):
                        colors = np.tile(colors.reshape(1, 3), (len(positions), 1))
                    if colors.ndim != 2 or colors.shape != (len(positions), 3):
                        continue

                radii_raw = entry.get("radii", entry.get("sizes", 0.22))
                radii: np.ndarray
                if isinstance(radii_raw, (int, float)):
                    radii = np.full((len(positions),), float(radii_raw), dtype=float)
                else:
                    radii = np.asarray(radii_raw, dtype=float).reshape(-1)
                    if len(radii) != len(positions):
                        continue
                radii = np.clip(radii, 0.05, None)
                specs.append(
                    _ViewerPointOverlaySpec(
                        positions=np.asarray(positions, dtype=float),
                        colors=np.asarray(np.clip(colors, 0.0, 1.0), dtype=float),
                        radii=np.asarray(radii, dtype=float),
                        alpha=float(np.clip(alpha, 0.0, 1.0)),
                    )
                )

        self._point_overlay_specs = specs
        self._store_view()
        self._update_scene(self._try_update_pyvista_point_overlay)

    @_renders_once
    def set_line_overlays(self, overlays: Iterable[Mapping[str, object]] | None) -> None:
        """Attach zero or more line or arrow overlays."""

        specs: list[_ViewerLineOverlaySpec] = []
        if overlays is not None:
            for entry in overlays:
                if not isinstance(entry, Mapping):
                    continue
                if "positions" in entry:
                    positions = np.asarray(entry.get("positions"), dtype=float)
                    if positions.shape != (2, 3):
                        continue
                    start = positions[0]
                    end = positions[1]
                else:
                    start = np.asarray(entry.get("start"), dtype=float)
                    end = np.asarray(entry.get("end"), dtype=float)
                    if start.shape != (3,) or end.shape != (3,):
                        continue
                if not np.isfinite(start).all() or not np.isfinite(end).all():
                    continue

                color_raw = entry.get("color", entry.get("colors", "#15a3a3"))
                if isinstance(color_raw, str):
                    qcolor = QColor(color_raw)
                    if not qcolor.isValid():
                        continue
                    color = tuple(float(value) for value in qcolor.getRgbF()[:3])
                else:
                    color_array = np.asarray(color_raw, dtype=float).reshape(-1)
                    if color_array.shape != (3,) or not np.isfinite(color_array).all():
                        continue
                    scale = 255.0 if float(np.max(color_array)) > 1.0 else 1.0
                    color = tuple(float(np.clip(value / scale, 0.0, 1.0)) for value in color_array)

                try:
                    width = float(entry.get("width", 5.0))
                except (TypeError, ValueError):
                    width = 5.0
                try:
                    alpha = float(entry.get("alpha", 0.9))
                except (TypeError, ValueError):
                    alpha = 0.9

                specs.append(
                    _ViewerLineOverlaySpec(
                        start=np.asarray(start, dtype=float),
                        end=np.asarray(end, dtype=float),
                        color=color,
                        width=max(1.0, width),
                        alpha=float(np.clip(alpha, 0.0, 1.0)),
                        arrow=bool(entry.get("arrow", False)),
                    )
                )

        self._line_overlay_specs = specs
        self._store_view()
        self._update_scene(self._try_update_pyvista_line_overlay)

    @_renders_once
    def set_hide_atoms_in_front_of_planes(self, hide: bool) -> None:
        """Hide atoms on the camera-facing side of all active slice planes."""

        normalized = bool(hide)
        if normalized == self._hide_atoms_in_front_of_planes:
            return
        self._hide_atoms_in_front_of_planes = normalized
        self._store_view()
        self._render()

    @_renders_once
    def set_pyvista_light_direction(
        self,
        direction: np.ndarray | tuple[float, float, float] | list[float],
    ) -> None:
        self._pyvista_light_direction = _coerce_light_direction(direction)
        if hasattr(self._triad_canvas, "set_light_direction"):
            self._triad_canvas.set_light_direction(self._pyvista_light_direction)
        self._store_view()
        self._render()

    def _set_spin_visibility_default_from_data(self) -> None:
        if self._spins is not None and np.any(np.abs(self._spins) > 1e-10):
            self._show_spins.setChecked(True)
            return
        self._show_spins.setChecked(False)

    def select_atom(self, atom_index: int | None) -> None:
        """Highlight an atom without emitting a new selection signal."""
        self._set_selected_atoms([] if atom_index is None else [int(atom_index)], emit_signal=False)

    def select_atoms(self, atom_indexes: Iterable[int] | None) -> None:
        """Highlight zero or more atoms without emitting a new selection signal."""
        indexes = [] if atom_indexes is None else [int(value) for value in atom_indexes]
        self._set_selected_atoms(indexes, emit_signal=False)

    def selected_atom_indices(self) -> tuple[int, ...]:
        """Return currently selected atom indices."""
        return tuple(int(index) for index in self._selected_indexes)

    def _set_selected_atom(self, atom_index: int | None, *, emit_signal: bool) -> None:
        self._set_selected_atoms(
            [] if atom_index is None else [int(atom_index)], emit_signal=emit_signal
        )

    def _set_selected_atoms(self, atom_indexes: Iterable[int], *, emit_signal: bool) -> None:
        normalized: list[int] = []
        seen: set[int] = set()
        if self._atoms is not None:
            for raw_index in atom_indexes:
                index = int(raw_index)
                if index < 0 or index >= len(self._atoms) or index in seen:
                    continue
                seen.add(index)
                normalized.append(index)
        self._selected_indexes = normalized
        self._selected_index = None if not normalized else int(normalized[-1])
        self._update_selection_label()

        with self.batch_update():
            self._store_view()
            self._update_scene(self._try_update_pyvista_selection_overlay)

        if emit_signal:
            self.selected_atoms_changed.emit(list(self._selected_indexes))
            if self._selected_index is not None:
                self.atom_selected.emit(self._selected_index)

    def _update_selection_label(self) -> None:
        if self._atoms is None or not self._selected_indexes:
            self._selection_label.setText("Selected atom: -")
            self._selection_label.setToolTip("")
            return
        if len(self._selected_indexes) == 1:
            selection_text = self._format_selection_text(self._selected_indexes[0])
            self._selection_label.setText(selection_text)
            self._selection_label.setToolTip(selection_text)
            return
        entries = [
            f"#{index + 1} {self._atoms[index].symbol}" for index in self._selected_indexes[:4]
        ]
        suffix = (
            "" if len(self._selected_indexes) <= 4 else f" +{len(self._selected_indexes) - 4} more"
        )
        text = f"Selected atoms: {len(self._selected_indexes)} | " + ", ".join(entries) + suffix
        self._selection_label.setText(text)
        self._selection_label.setToolTip(text)

    def _format_selection_text(self, atom_index: int) -> str:
        assert self._atoms is not None
        frac = self._atoms.get_scaled_positions(wrap=True)[atom_index]
        cart = self._atoms.get_positions()[atom_index]
        symbol = self._atoms[atom_index].symbol
        return (
            f"Selected atom: #{atom_index + 1} {symbol} | "
            f"f=({frac[0]:.4f}, {frac[1]:.4f}, {frac[2]:.4f}) | "
            f"r=({cart[0]:.3f}, {cart[1]:.3f}, {cart[2]:.3f}) A"
        )

    def _to_atoms(self, structure: Structure) -> Atoms:
        lattice = np.asarray(structure.lattice_vectors, dtype=float)
        if lattice.shape != (3, 3):
            lattice = np.eye(3, dtype=float)
        pbc = structure.properties.get("pbc", [True, True, True])
        if not isinstance(pbc, Iterable):
            pbc = [True, True, True]
        symbols = _viewer_species_symbols(structure.species)
        return Atoms(
            symbols=symbols,
            scaled_positions=structure.fractional_positions,
            cell=lattice,
            pbc=[bool(value) for value in pbc],
        )

    def _extract_spins(self, structure: Structure | None) -> np.ndarray | None:
        if structure is None:
            return None

        count = len(structure.species)
        raw_spins = structure.properties.get("atomic_spins")
        if not isinstance(raw_spins, Iterable):
            return np.zeros(count, dtype=float)

        parsed: list[float] = []
        for value in raw_spins:
            try:
                parsed.append(float(value))
            except (TypeError, ValueError):
                parsed.append(0.0)
            if len(parsed) == count:
                break

        if len(parsed) < count:
            parsed.extend([0.0] * (count - len(parsed)))
        return np.asarray(parsed, dtype=float)

    def _can_update_structure_geometry_in_place(self, structure: Structure) -> bool:
        if self._structure is None or self._atoms is None:
            return False
        if len(self._structure.species) != len(structure.species):
            return False
        if list(self._structure.species) != list(structure.species):
            return False
        current_lattice = np.asarray(self._structure.lattice_vectors, dtype=float)
        updated_lattice = np.asarray(structure.lattice_vectors, dtype=float)
        return (
            current_lattice.shape == (3, 3)
            and updated_lattice.shape == (3, 3)
            and np.allclose(current_lattice, updated_lattice)
        )

    def _redraw_preserving_view(self) -> None:
        with self.batch_update():
            self._store_view()
            self._update_scene(self._try_update_pyvista_visibility_overlay)
        self.view_state_changed.emit()

    def _reset_view(self) -> None:
        self._camera_state = _CameraState(*_DEFAULT_VIEW, _DEFAULT_VIEW_ROLL)
        self._axis.set_from_state(self._camera_state)
        if self._triad_axis is not None:
            self._triad_axis.set_from_state(self._camera_state)
        self._view = _DEFAULT_VIEW
        self._view_roll = _DEFAULT_VIEW_ROLL
        self._limits = None
        self._pyvista_camera_snapshot = None
        self._render()
        self.view_state_changed.emit()

    def _set_named_view(self, axis_name: str) -> None:
        normalized_name = str(axis_name).strip().lower()
        if normalized_name == "x":
            normalized_name = "a"
        elif normalized_name == "y":
            normalized_name = "b"
        elif normalized_name == "z":
            normalized_name = "c"

        if self._atoms is None:
            if normalized_name == "a":
                self._camera_state = _CameraState(0.0, 0.0, 0.0)
            elif normalized_name == "b":
                self._camera_state = _CameraState(0.0, 90.0, 0.0)
            else:
                self._camera_state = _CameraState(90.0, -90.0, 0.0)
        else:
            cell = np.asarray(self._atoms.cell, dtype=float)
            if cell.shape != (3, 3):
                cell = np.eye(3, dtype=float)
            direction_map = {
                "a": cell[0],
                "b": cell[1],
                "c": cell[2],
            }
            fallback_up_map = {
                "a": (cell[2], cell[1], np.array([0.0, 0.0, 1.0], dtype=float)),
                "b": (cell[2], cell[0], np.array([0.0, 0.0, 1.0], dtype=float)),
                "c": (cell[0], cell[1], np.array([0.0, 1.0, 0.0], dtype=float)),
            }
            direction = np.asarray(direction_map.get(normalized_name, cell[2]), dtype=float)
            if float(np.linalg.norm(direction)) <= 1e-12:
                direction = np.array([0.0, 0.0, 1.0], dtype=float)
            up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
            for candidate in fallback_up_map.get(normalized_name, ()):
                candidate_array = _normalized_vector(np.asarray(candidate, dtype=float), up_hint)
                if (
                    abs(float(np.dot(_normalized_vector(direction, direction), candidate_array)))
                    < 0.96
                ):
                    up_hint = candidate_array
                    break
            camera_direction = _normalized_vector(direction, np.array([0.0, 0.0, 1.0], dtype=float))
            self._camera_state = _camera_state_from_vectors(
                position=4.0 * camera_direction,
                focal_point=(0.0, 0.0, 0.0),
                view_up=up_hint,
            )
        self._axis.set_from_state(self._camera_state)
        if self._triad_axis is not None:
            self._triad_axis.set_from_state(self._camera_state)
        self._view = (self._camera_state.elev, self._camera_state.azim)
        self._view_roll = self._camera_state.roll
        self._pyvista_camera_snapshot = None
        self._render()
        self.view_state_changed.emit()

    def current_view_direction(self) -> tuple[float, float, float]:
        """Return the current camera viewing direction in Cartesian coordinates."""

        camera_position, focal_point, _view_up = _camera_vectors_from_state(self._camera_state)
        direction = np.asarray(focal_point, dtype=float) - np.asarray(camera_position, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return (0.0, 0.0, 1.0)
        normalized = direction / norm
        return (float(normalized[0]), float(normalized[1]), float(normalized[2]))

    def _apply_camera_state(self) -> None:
        self._view = (self._camera_state.elev, self._camera_state.azim)
        self._view_roll = self._camera_state.roll
        self._axis.set_from_state(self._camera_state)
        if self._triad_axis is not None:
            self._triad_axis.set_from_state(self._camera_state)
        self._draw_crystallographic_axes()

    def _store_view(self, *_args) -> None:
        previous_state = self._camera_state
        previous_limits = self._limits
        if self._axis.consume_dirty():
            self._camera_state = self._axis.state()
            self._view = (self._camera_state.elev, self._camera_state.azim)
            self._view_roll = self._camera_state.roll
            self._pyvista_camera_snapshot = None
            self._apply_pyvista_camera()
        else:
            self._capture_pyvista_camera_snapshot()
        self._apply_camera_state()
        if self._camera_state != previous_state or self._limits != previous_limits:
            self.view_state_changed.emit()

    def _scene_geometry(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._atoms is not None
        positions = np.asarray(self._atoms.get_positions(), dtype=float)
        numbers = np.asarray(self._atoms.get_atomic_numbers(), dtype=int)
        symbols = np.asarray(self._atoms.get_chemical_symbols())
        colors, radii = _get_atom_style(numbers, symbols)
        color_overrides = _atom_color_overrides(self._structure, len(positions))
        if color_overrides is not None:
            override_mask = np.all(np.isfinite(color_overrides), axis=1)
            colors[override_mask] = color_overrides[override_mask]
        spin_tips = np.empty((0, 3), dtype=float)
        if (
            self._show_spins.isChecked()
            and self._spins is not None
            and len(self._spins) == len(positions)
        ):
            spin_tips = self._spin_extent_points(positions, radii, self._spins)
        extra_points = None
        if self._show_cell.isChecked():
            corners, _edges = _unit_cell_geometry(np.asarray(self._atoms.cell, dtype=float))
            extra_points = corners
        isosurface_vertices = [spec.vertices for spec in self._isosurface_specs if len(spec.faces)]
        if isosurface_vertices:
            isosurface_points = np.vstack(isosurface_vertices)
            extra_points = (
                isosurface_points
                if extra_points is None
                else np.vstack([extra_points, isosurface_points])
            )
        slice_points: list[np.ndarray] = []
        for spec in self._slice_specs:
            slice_points.append(
                np.column_stack(
                    [
                        spec.x_coordinates.reshape(-1),
                        spec.y_coordinates.reshape(-1),
                        spec.z_coordinates.reshape(-1),
                    ]
                )
            )
        if slice_points:
            overlay_points = np.vstack(slice_points)
            extra_points = (
                overlay_points
                if extra_points is None
                else np.vstack([extra_points, overlay_points])
            )
        if len(spin_tips):
            extra_points = (
                spin_tips if extra_points is None else np.vstack([extra_points, spin_tips])
            )
        if self._point_overlay_specs:
            overlay_points = np.vstack(
                [spec.positions for spec in self._point_overlay_specs if len(spec.positions)]
            )
            if len(overlay_points):
                extra_points = (
                    overlay_points
                    if extra_points is None
                    else np.vstack([extra_points, overlay_points])
                )
        if self._line_overlay_specs:
            overlay_points = np.vstack(
                [
                    np.vstack([spec.start.reshape(1, 3), spec.end.reshape(1, 3)])
                    for spec in self._line_overlay_specs
                ]
            )
            if len(overlay_points):
                extra_points = (
                    overlay_points
                    if extra_points is None
                    else np.vstack([extra_points, overlay_points])
                )
        if extra_points is None:
            extra_points = np.empty((0, 3), dtype=float)
        return positions, numbers, symbols, colors, radii, extra_points

    def _camera_view_direction(self) -> np.ndarray:
        elev = math.radians(float(self._camera_state.elev))
        azim = math.radians(float(self._camera_state.azim))
        direction = np.array(
            [
                math.cos(elev) * math.cos(azim),
                math.cos(elev) * math.sin(azim),
                math.sin(elev),
            ],
            dtype=float,
        )
        return _normalized_vector(direction, np.array([1.0, 1.0, 0.6], dtype=float))

    def _hidden_atom_indexes_for_slices(self, positions: np.ndarray) -> np.ndarray:
        if not self._hide_atoms_in_front_of_planes or not self._slice_specs or len(positions) == 0:
            return np.zeros((len(positions),), dtype=bool)
        view_direction = self._camera_view_direction()
        hidden_mask = np.ones((len(positions),), dtype=bool)
        for spec in self._slice_specs:
            camera_sign = float(np.dot(view_direction, spec.normal))
            if abs(camera_sign) <= 1e-12:
                hidden_mask &= False
                continue
            distances = np.asarray(
                (positions - spec.anchor.reshape(1, 3)) @ spec.normal.reshape(3, 1), dtype=float
            ).reshape(-1)
            hidden_mask &= (distances * camera_sign) > 1e-6
        return hidden_mask

    def _visible_atom_indices(self, positions: np.ndarray) -> np.ndarray:
        if len(positions) == 0:
            return np.empty((0,), dtype=int)
        return np.flatnonzero(~self._hidden_atom_indexes_for_slices(positions))

    def _visible_atom_mask(self, positions: np.ndarray) -> np.ndarray:
        if len(positions) == 0:
            return np.zeros((0,), dtype=bool)
        return ~self._hidden_atom_indexes_for_slices(positions)

    def _filtered_spin_values(
        self, spins: np.ndarray | None, visible_mask: np.ndarray
    ) -> np.ndarray | None:
        if spins is None:
            return None
        spin_values = np.asarray(spins, dtype=float)
        if len(spin_values) != len(visible_mask):
            return None
        return spin_values[visible_mask]

    def _selection_modifier_active(self, event=None) -> bool:
        key = ""
        if event is not None:
            key = str(getattr(event, "key", "") or "")
            mouse_event = getattr(event, "mouseevent", None)
            if mouse_event is not None:
                key = str(getattr(mouse_event, "key", "") or key)
        modifiers = QApplication.keyboardModifiers()
        return (
            ("shift" in key.lower())
            or self._native_selection_modifier
            or bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        )

    def _pyvista_selection_modifier_active(self) -> bool:
        if self._native_selection_modifier:
            return True
        if bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier):
            return True
        plotter = getattr(getattr(self, "_canvas", None), "plotter", None)
        iren = getattr(plotter, "iren", None)
        interactor = getattr(iren, "interactor", None)
        if interactor is None:
            return False
        try:
            return bool(interactor.GetShiftKey())
        except Exception:  # noqa: BLE001
            return False

    def _apply_pick_selection(self, atom_index: int, *, event=None) -> None:
        additive = self._selection_modifier_active(event) or self._latched_selection_modifier
        if additive and atom_index in self._selected_indexes:
            self._set_selected_atoms(
                [index for index in self._selected_indexes if index != atom_index], emit_signal=True
            )
            return
        if additive:
            self._set_selected_atoms(
                list(self._selected_indexes) + [int(atom_index)], emit_signal=True
            )
            return
        self._set_selected_atoms([int(atom_index)], emit_signal=True)

    def _spin_extent_points(
        self, positions: np.ndarray, radii: np.ndarray, spins: np.ndarray
    ) -> np.ndarray:
        tips: list[np.ndarray] = []
        for position, radius, spin in zip(positions, radii, spins):
            spin_array = np.asarray(spin, dtype=float)
            spin_vector = (
                spin_array
                if spin_array.shape == (3,)
                else np.array([0.0, 0.0, float(spin_array)], dtype=float)
            )
            magnitude = float(np.linalg.norm(spin_vector))
            if magnitude <= 1e-10:
                continue
            direction_vector = spin_vector / magnitude
            arrow_length = max(0.3, magnitude * 0.85) + 0.4 * float(radius)
            half_length = 0.5 * arrow_length
            tips.append(np.asarray(position, dtype=float) + direction_vector * half_length)
            tips.append(np.asarray(position, dtype=float) - direction_vector * half_length)
        if not tips:
            return np.empty((0, 3), dtype=float)
        return np.vstack(tips)

    @contextmanager
    def batch_update(self) -> Iterator[None]:
        """Group several viewer changes so the scene renders once, at the end.

        Inside the block every setter records its state but defers drawing; leaving
        the outermost block rebuilds the scene once if any change needed a rebuild,
        otherwise redraws once if any change updated actors in place. Blocks nest.
        """

        self._batch_depth += 1
        try:
            yield
        except BaseException:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                # Draw what was recorded, but never let a render failure on the
                # half-applied state hide the caller's exception.
                try:
                    self._flush_batched_render()
                except Exception:  # noqa: BLE001
                    _LOGGER.debug("Deferred viewer render failed", exc_info=True)
            raise
        else:
            self._batch_depth -= 1
            if self._batch_depth == 0:
                self._flush_batched_render()

    def _flush_batched_render(self) -> None:
        if (
            not self._shutting_down
            and not self._defers_drawing()
            and self._style_signature != _viewer_style_signature()
        ):
            # The theme changed while the viewer was hidden: refreshing it
            # rebuilds the scene, which draws every other pending change too.
            self.refresh_theme()
            return
        render_pending = self._render_pending
        draw_pending = self._draw_pending
        triad_draw_pending = self._triad_draw_pending
        self._render_pending = False
        self._draw_pending = False
        self._triad_draw_pending = False
        if self._shutting_down:
            return
        if render_pending:
            self._render()
            return
        if draw_pending:
            self._draw_canvas()
        if triad_draw_pending:
            # Re-reads the camera, which a hidden viewer left untouched.
            self._draw_crystallographic_axes()

    def _render(self) -> None:
        """Rebuild and draw the whole scene, or defer it to the end of a batch or to show.

        A hidden viewer only marks its scene stale; ``showEvent`` rebuilds it once.
        """

        if self._shutting_down:
            return
        if self._defers_drawing():
            self._render_pending = True
            return
        # The rebuild draws every change, including any deferred in-place one.
        self._render_pending = False
        self._draw_pending = False
        self._triad_draw_pending = False
        self._render_pyvista()
        self._draw_crystallographic_axes()

    def _update_scene(self, update_in_place: Callable[[], bool]) -> None:
        """Apply a change through ``update_in_place`` when it can, else rebuild the scene.

        A rebuild already pending in a batch draws the change too, so the in-place
        update is skipped rather than built and then thrown away.
        """

        if self._render_deferred():
            return
        if update_in_place():
            return
        self._render()

    def _render_deferred(self) -> bool:
        """Whether a scene rebuild is waiting for the current batch to end or for show."""

        return bool(self._render_pending and self._defers_drawing())

    def _draw_canvas(self) -> None:
        """Draw in-place actor changes, or defer the draw to the end of a batch or to show."""

        if self._defers_drawing():
            self._draw_pending = True
            return
        self._canvas.draw_idle()

    def _draw_triad(self) -> None:
        if self._defers_drawing():
            self._triad_draw_pending = True
            return
        self._triad_canvas.draw_idle()

    def _render_pyvista(self) -> None:
        if self._shutting_down:
            return
        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None or any(
            not callable(getattr(plotter, attribute, None))
            for attribute in ("clear", "enable_parallel_projection", "add_mesh")
        ):
            return
        pv, _qt_interactor = _pyvista_modules()
        assert pv is not None
        self._pyvista_atom_actor = None
        self._pyvista_isosurface_actor = None
        self._pyvista_isosurface_actors = []
        self._pyvista_slice_actors = []
        self._pyvista_point_overlay_actors = []
        self._pyvista_line_overlay_actors = []
        self._pyvista_selection_actor = None
        self._pyvista_pick_actor = None
        self._pyvista_bond_actors = []
        self._pyvista_spin_actors = []
        plotter.clear()
        _configure_transparent_pyvista_background(plotter)
        plotter.enable_parallel_projection()

        if self._atoms is None or len(self._atoms) == 0:
            self._configure_pyvista_lighting(plotter, pv)
            plotter.add_text(
                "Load a host structure to inspect it here.",
                position="upper_left",
                font_size=12,
                color=_viewer_style().text_primary,
                render=False,
            )
            self._pyvista_last_camera_signature = None
            self._canvas.draw_idle()
            return

        positions, _numbers, _symbols, colors, radii, extra_points = self._scene_geometry()
        scene_center = np.mean(positions, axis=0) if len(positions) else np.zeros(3, dtype=float)
        self._configure_pyvista_lighting(plotter, pv, focal_point=scene_center)
        self._pyvista_all_atom_positions = positions.copy()
        self._pyvista_all_atom_radii = radii.copy()
        self._pyvista_all_atom_colors = colors.copy()
        visible_mask = self._visible_atom_mask(positions)
        visible_indices = np.flatnonzero(visible_mask)
        visible_spins = self._filtered_spin_values(self._spins, visible_mask)
        self._pyvista_atom_lookup_indices = visible_indices.copy()
        draw_positions = positions[visible_indices]
        draw_radii = radii[visible_indices]
        draw_colors = colors[visible_indices]
        self._pyvista_atom_positions = draw_positions.copy()
        self._pyvista_atom_radii = draw_radii.copy()
        self._pyvista_atom_colors = draw_colors.copy()
        if len(draw_positions) == 0:
            if self._show_cell.isChecked():
                self._draw_pyvista_unit_cell(plotter, pv)
            if self._slice_specs:
                self._draw_pyvista_slices(plotter, pv)
            if self._limits is None:
                self._limits = _compute_limits(
                    positions, atom_radii=radii, extra_points=extra_points, margin=0.1
                )
            self._apply_pyvista_camera()
            self._canvas.draw_idle()
            return
        atom_alpha = 0.5 if self._show_spins.isChecked() and self._spins is not None else 1.0
        atom_subdivisions = _atom_pyvista_sphere_subdivisions(len(draw_positions))
        atom_glyphs = _build_pyvista_atom_glyph_mesh(
            pv,
            positions=draw_positions,
            radii=draw_radii,
            colors=draw_colors,
            subdivisions=atom_subdivisions,
        )
        self._pyvista_atom_actor = plotter.add_mesh(
            atom_glyphs,
            scalars="atom_rgb",
            rgb=True,
            opacity=float(atom_alpha),
            smooth_shading=True,
            ambient=0.74,
            diffuse=0.22,
            specular=0.0,
            specular_power=1.0,
            pbr=False,
            render=False,
        )

        pick_points = pv.PolyData(draw_positions)
        pick_points["pick_radius"] = np.maximum(1.8 * draw_radii, 0.45)
        self._pyvista_pick_actor = plotter.add_mesh(
            pick_points,
            color="#000000",
            opacity=0.01,
            point_size=float(max(16.0, np.max(np.maximum(32.0 * draw_radii, 18.0)))),
            render_points_as_spheres=True,
            pickable=True,
            name="atom-pick-points",
            render=False,
        )

        if self._show_bonds.isChecked():
            bond_alpha = float(np.clip(0.84 * atom_alpha, 0.2, 0.8))
            self._pyvista_bond_actors = self._draw_pyvista_bonds(
                plotter,
                pv,
                positions,
                colors,
                radii,
                visible_mask=visible_mask,
                alpha=bond_alpha,
            )

        if self._show_spins.isChecked() and visible_spins is not None:
            self._pyvista_spin_actors = self._draw_pyvista_spins(
                plotter, pv, draw_positions, draw_radii, visible_spins
            )

        if self._show_cell.isChecked():
            self._draw_pyvista_unit_cell(plotter, pv)
        if self._isosurface_specs:
            self._draw_pyvista_isosurfaces(plotter, pv)
        if self._slice_specs:
            self._draw_pyvista_slices(plotter, pv)
        if self._point_overlay_specs:
            self._draw_pyvista_point_overlays(plotter, pv)
        if self._line_overlay_specs:
            self._draw_pyvista_line_overlays(plotter, pv)
        if self._selected_indexes:
            self._draw_pyvista_selection(plotter, pv, positions, radii)

        if self._limits is None:
            self._limits = _compute_limits(
                positions, atom_radii=radii, extra_points=extra_points, margin=0.1
            )
        self._apply_pyvista_camera()
        self._canvas.draw_idle()

    def _current_pyvista_camera_snapshot(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        if self._backend_name != "pyvista" or self._shutting_down:
            return None
        if self._render_deferred():
            # The live camera still shows the scene before this batch; the deferred
            # rebuild shows this one (None until it fits a new structure).
            return self._pyvista_camera_snapshot
        camera = getattr(getattr(self._canvas, "plotter", None), "camera", None)
        if camera is None:
            return None
        return (
            np.asarray(camera.position, dtype=float),
            np.asarray(camera.focal_point, dtype=float),
            np.asarray(camera.up, dtype=float),
            float(camera.parallel_scale),
        )

    def _capture_pyvista_camera_snapshot(self) -> None:
        if self._backend_name != "pyvista" or self._shutting_down:
            return
        snapshot = self._current_pyvista_camera_snapshot()
        if snapshot is None:
            return
        self._pyvista_camera_snapshot = snapshot

    def _apply_pyvista_camera(self) -> None:
        if self._backend_name != "pyvista" or self._limits is None or self._shutting_down:
            return
        plotter = self._canvas.plotter
        if self._pyvista_camera_snapshot is not None:
            position, focal_point, up, parallel_scale = self._pyvista_camera_snapshot
            plotter.camera.position = tuple(float(value) for value in position)
            plotter.camera.focal_point = tuple(float(value) for value in focal_point)
            plotter.camera.up = tuple(float(value) for value in up)
            plotter.camera.parallel_scale = float(parallel_scale)
            plotter.reset_camera_clipping_range()
            self._configure_pyvista_lighting(
                plotter, _pyvista_modules()[0], focal_point=focal_point
            )
            self._pyvista_last_camera_signature = self._pyvista_camera_signature()
            self._axis.set_from_state(self._camera_state)
            return
        xlim, ylim, zlim = self._limits
        center = np.array(
            [0.5 * (xlim[0] + xlim[1]), 0.5 * (ylim[0] + ylim[1]), 0.5 * (zlim[0] + zlim[1])],
            dtype=float,
        )
        spans = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]], dtype=float)
        distance = float(max(np.max(spans) * 2.6, 1.0))
        elev = math.radians(float(self._camera_state.elev))
        azim = math.radians(float(self._camera_state.azim))
        view_direction = np.array(
            [
                math.cos(elev) * math.cos(azim),
                math.cos(elev) * math.sin(azim),
                math.sin(elev),
            ],
            dtype=float,
        )
        position = center + distance * view_direction
        reference_up = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(view_direction, reference_up))) > 0.96:
            reference_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = _normalized_vector(
            np.cross(view_direction, reference_up), np.array([1.0, 0.0, 0.0], dtype=float)
        )
        up = _normalized_vector(
            np.cross(right, view_direction), np.array([0.0, 0.0, 1.0], dtype=float)
        )
        if abs(float(self._camera_state.roll)) > 1e-12:
            angle = math.radians(float(self._camera_state.roll))
            up = (
                up * math.cos(angle)
                + np.cross(view_direction, up) * math.sin(angle)
                + view_direction * np.dot(view_direction, up) * (1.0 - math.cos(angle))
            )
        plotter.camera.position = tuple(float(value) for value in position)
        plotter.camera.focal_point = tuple(float(value) for value in center)
        plotter.camera.up = tuple(float(value) for value in up)
        plotter.camera.parallel_scale = float(0.58 * max(np.max(spans), 1.0))
        plotter.reset_camera_clipping_range()
        self._configure_pyvista_lighting(plotter, _pyvista_modules()[0], focal_point=center)
        self._capture_pyvista_camera_snapshot()
        self._pyvista_last_camera_signature = self._pyvista_camera_signature()
        self._axis.set_from_state(self._camera_state)

    def _configure_pyvista_lighting(
        self,
        plotter,
        pv,
        *,
        focal_point: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        _configure_pyvista_vesta_style(
            plotter,
            pv,
            light_direction=self._pyvista_light_direction,
            focal_point=focal_point,
        )

    def _pyvista_camera_signature(self) -> tuple[float, ...] | None:
        if self._backend_name != "pyvista":
            return None
        camera = getattr(getattr(self._canvas, "plotter", None), "camera", None)
        if camera is None:
            return None
        values = (
            *[float(value) for value in camera.position],
            *[float(value) for value in camera.focal_point],
            *[float(value) for value in camera.up],
            float(camera.parallel_scale),
        )
        return tuple(round(value, 6) for value in values)

    def _poll_pyvista_camera(self) -> None:
        if self._backend_name != "pyvista" or self._atoms is None or self._shutting_down:
            return
        if self._render_deferred():
            # The live camera belongs to the scene the pending rebuild replaces.
            return
        signature = self._pyvista_camera_signature()
        if signature is None or signature == self._pyvista_last_camera_signature:
            return
        camera = self._canvas.plotter.camera
        self._camera_state = _camera_state_from_vectors(
            camera.position, camera.focal_point, camera.up
        )
        self._capture_pyvista_camera_snapshot()
        self._pyvista_last_camera_signature = signature
        self._view = (self._camera_state.elev, self._camera_state.azim)
        self._view_roll = self._camera_state.roll
        self._axis.set_from_state(self._camera_state)
        self._draw_crystallographic_axes()
        self.view_state_changed.emit()

    def _pyvista_event_position(self) -> tuple[int, int] | None:
        plotter = getattr(getattr(self, "_canvas", None), "plotter", None)
        iren = getattr(plotter, "iren", None)
        if iren is None:
            return None
        get_event_position = getattr(iren, "get_event_position", None)
        if callable(get_event_position):
            try:
                position = get_event_position()
            except Exception:  # noqa: BLE001
                position = None
        else:
            interactor = getattr(iren, "interactor", None)
            if interactor is None:
                return None
            try:
                position = interactor.GetEventPosition()
            except Exception:  # noqa: BLE001
                position = None
        if not isinstance(position, tuple) or len(position) != 2:
            return None
        return (int(position[0]), int(position[1]))

    def _pyvista_pick_point_at_event_position(self) -> np.ndarray | None:
        plotter = getattr(getattr(self, "_canvas", None), "plotter", None)
        iren = getattr(plotter, "iren", None)
        position = self._pyvista_event_position()
        if plotter is None or iren is None or position is None:
            return None
        get_poked_renderer = getattr(iren, "get_poked_renderer", None)
        if not callable(get_poked_renderer):
            return None
        try:
            renderer = get_poked_renderer()
        except Exception:  # noqa: BLE001
            return None
        picker = getattr(iren, "picker", None)
        if renderer is None or picker is None:
            return None
        try:
            picked = picker.Pick(position[0], position[1], 0, renderer)
        except Exception:  # noqa: BLE001
            return None
        if int(picked) == 0:
            return None
        try:
            coordinates = np.asarray(picker.GetPickPosition(), dtype=float)
        except Exception:  # noqa: BLE001
            return None
        if coordinates.shape != (3,) or np.any(np.isnan(coordinates)):
            return None
        return coordinates

    def _on_pyvista_left_button_press(self, _obj=None, _event=None) -> None:
        if self._shutting_down:
            return
        self._pyvista_left_press_position = self._pyvista_event_position()
        self._pyvista_left_press_modifier = self._pyvista_selection_modifier_active()
        self._pyvista_left_press_camera_signature = self._pyvista_camera_signature()
        self._latched_selection_modifier = self._pyvista_left_press_modifier

    def _on_pyvista_left_button_release(self, _obj=None, _event=None) -> None:
        if self._shutting_down:
            return
        press_position = self._pyvista_left_press_position
        press_modifier = bool(self._pyvista_left_press_modifier)
        press_signature = self._pyvista_left_press_camera_signature
        release_position = self._pyvista_event_position()
        release_signature = self._pyvista_camera_signature()

        self._pyvista_left_press_position = None
        self._pyvista_left_press_modifier = False
        self._pyvista_left_press_camera_signature = None

        if press_position is None or release_position is None:
            return
        delta_x = release_position[0] - press_position[0]
        delta_y = release_position[1] - press_position[1]
        moved_too_far = (delta_x * delta_x + delta_y * delta_y) > 25
        camera_changed = (
            press_signature is not None
            and release_signature is not None
            and release_signature != press_signature
        )
        if moved_too_far or camera_changed:
            return

        self._latched_selection_modifier = press_modifier
        picked_point = self._pyvista_pick_point_at_event_position()
        if self._on_pyvista_point_picked(picked_point, additive=press_modifier):
            return
        if self._selected_indexes and not press_modifier:
            self._set_selected_atoms([], emit_signal=True)

    def _on_pyvista_point_picked(self, point, *, additive: bool | None = None) -> bool:
        if self._shutting_down:
            return False
        if point is None:
            return False
        coordinates = np.asarray(point, dtype=float)
        if (
            coordinates.shape != (3,)
            or self._pyvista_atom_positions.size == 0
            or np.any(np.isnan(coordinates))
        ):
            return False
        distances = np.linalg.norm(self._pyvista_atom_positions - coordinates.reshape(1, 3), axis=1)
        index = int(np.argmin(distances))
        tolerance = max(0.45, 1.35 * float(self._pyvista_atom_radii[index]))
        if float(distances[index]) <= tolerance:
            self._latched_selection_modifier = (
                self._pyvista_selection_modifier_active() if additive is None else bool(additive)
            )
            mapped_index = index
            if 0 <= index < len(self._pyvista_atom_lookup_indices):
                mapped_index = int(self._pyvista_atom_lookup_indices[index])
            self._apply_pick_selection(mapped_index)
            return True
        return False

    def _draw_pyvista_unit_cell(self, plotter, pv) -> None:
        assert self._atoms is not None
        outline = _build_pyvista_unit_cell_mesh(pv, np.asarray(self._atoms.cell, dtype=float))
        plotter.add_mesh(
            outline,
            color=_viewer_style().unit_cell,
            line_width=2.0,
            opacity=0.62,
            name="unit-cell-outline",
            render=False,
        )

    def _draw_pyvista_isosurfaces(self, plotter, pv) -> None:
        self._pyvista_isosurface_actors = []
        self._pyvista_isosurface_actor = None
        for index, spec in enumerate(self._isosurface_specs):
            if len(spec.faces) == 0:
                continue
            mesh = self._build_pyvista_isosurface_mesh(pv, vertices=spec.vertices, faces=spec.faces)
            actor = plotter.add_mesh(
                mesh,
                color=spec.color,
                opacity=float(spec.alpha),
                smooth_shading=False,
                name=f"isosurface-overlay-{index}",
                render=False,
            )
            self._pyvista_isosurface_actors.append(actor)
            if self._pyvista_isosurface_actor is None:
                self._pyvista_isosurface_actor = actor

    def _draw_pyvista_slices(self, plotter, pv) -> None:
        self._pyvista_slice_actors = []
        for index, spec in enumerate(self._slice_specs):
            mesh = pv.StructuredGrid(spec.x_coordinates, spec.y_coordinates, spec.z_coordinates)
            mesh["slice_value"] = np.asarray(spec.values, dtype=float).reshape(-1, order="F")
            actor = plotter.add_mesh(
                mesh,
                scalars="slice_value",
                cmap=spec.cmap,
                clim=spec.color_limits,
                opacity=float(spec.alpha),
                lighting=False,
                show_scalar_bar=False,
                name=f"slice-overlay-{index}",
                render=False,
            )
            self._pyvista_slice_actors.append(actor)

    def _draw_pyvista_point_overlays(self, plotter, pv) -> None:
        self._pyvista_point_overlay_actors = []
        for index, spec in enumerate(self._point_overlay_specs):
            if len(spec.positions) == 0:
                continue
            glyphs = _build_pyvista_atom_glyph_mesh(
                pv,
                positions=spec.positions,
                radii=spec.radii,
                colors=spec.colors,
                subdivisions=3,
            )
            actor = plotter.add_mesh(
                glyphs,
                scalars="atom_rgb",
                rgb=True,
                opacity=float(spec.alpha),
                smooth_shading=True,
                ambient=0.72,
                diffuse=0.2,
                specular=0.0,
                specular_power=1.0,
                pbr=False,
                render=False,
                name=f"point-overlay-{index}",
            )
            self._pyvista_point_overlay_actors.append(actor)

    def _draw_pyvista_line_overlays(self, plotter, pv) -> None:
        self._pyvista_line_overlay_actors = []
        for index, spec in enumerate(self._line_overlay_specs):
            direction = np.asarray(spec.end - spec.start, dtype=float)
            length = float(np.linalg.norm(direction))
            if length <= 1e-10:
                continue
            unit = direction / length
            color = tuple(float(value) for value in spec.color)

            if spec.arrow:
                head_length = min(max(length * 0.18, 0.18), length * 0.45)
                shaft_end = np.asarray(spec.end, dtype=float) - unit * (0.68 * head_length)
            else:
                head_length = 0.0
                shaft_end = np.asarray(spec.end, dtype=float)

            line = pv.Line(np.asarray(spec.start, dtype=float), shaft_end)
            line_actor = plotter.add_mesh(
                line,
                color=color,
                opacity=float(spec.alpha),
                line_width=float(spec.width),
                render_lines_as_tubes=True,
                smooth_shading=True,
                name=f"line-overlay-{index}",
                render=False,
            )
            self._pyvista_line_overlay_actors.append(line_actor)

            if not spec.arrow:
                continue

            cone = pv.Cone(
                center=np.asarray(spec.end, dtype=float) - unit * (0.5 * head_length),
                direction=unit,
                height=float(head_length),
                radius=max(0.06, 0.02 * float(spec.width)),
                resolution=20,
                capping=True,
            )
            cone_actor = plotter.add_mesh(
                cone,
                color=color,
                opacity=float(spec.alpha),
                smooth_shading=True,
                ambient=0.6,
                diffuse=0.28,
                specular=0.0,
                name=f"line-overlay-arrow-{index}",
                render=False,
            )
            self._pyvista_line_overlay_actors.append(cone_actor)

    def _build_pyvista_isosurface_mesh(self, pv, *, vertices: np.ndarray, faces: np.ndarray):
        faces = np.hstack(
            [
                np.full((len(faces), 1), 3, dtype=np.int64),
                np.asarray(faces, dtype=np.int64),
            ]
        ).ravel()
        return pv.PolyData(np.asarray(vertices, dtype=float), faces)

    def _remove_pyvista_isosurface_overlay(self) -> bool:
        if self._backend_name != "pyvista":
            return False
        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False

        removed = False
        actors = list(self._pyvista_isosurface_actors)
        if (
            self._pyvista_isosurface_actor is not None
            and self._pyvista_isosurface_actor not in actors
        ):
            actors.append(self._pyvista_isosurface_actor)
        for actor in actors:
            try:
                plotter.remove_actor(actor, render=False)
                removed = True
            except Exception:  # noqa: BLE001
                continue
        self._pyvista_isosurface_actors = []
        self._pyvista_isosurface_actor = None
        return removed

    def _try_update_pyvista_isosurface_overlay(self) -> bool:
        if (
            self._backend_name != "pyvista"
            or self._shutting_down
            or self._atoms is None
            or len(self._atoms) == 0
        ):
            return False

        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False

        pv, _qt_interactor = _pyvista_modules()
        if pv is None:
            return False

        has_overlay = any(len(spec.faces) > 0 for spec in self._isosurface_specs)
        if not has_overlay:
            if not self._remove_pyvista_isosurface_overlay():
                return False
            self._draw_canvas()
            return True

        self._remove_pyvista_isosurface_overlay()
        self._pyvista_isosurface_actors = []
        self._pyvista_isosurface_actor = None
        for index, spec in enumerate(self._isosurface_specs):
            if len(spec.faces) == 0:
                continue
            mesh = self._build_pyvista_isosurface_mesh(pv, vertices=spec.vertices, faces=spec.faces)
            actor = plotter.add_mesh(
                mesh,
                color=spec.color,
                opacity=float(spec.alpha),
                smooth_shading=False,
                name=f"isosurface-overlay-{index}",
                render=False,
            )
            self._pyvista_isosurface_actors.append(actor)
            if self._pyvista_isosurface_actor is None:
                self._pyvista_isosurface_actor = actor
        self._draw_canvas()
        return True

    def _remove_pyvista_slice_overlay(self) -> bool:
        removed = self._remove_pyvista_actor_group(self._pyvista_slice_actors)
        self._pyvista_slice_actors = []
        return removed

    def _try_update_pyvista_slice_overlay(self) -> bool:
        if (
            self._backend_name != "pyvista"
            or self._shutting_down
            or self._hide_atoms_in_front_of_planes
        ):
            return False

        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False

        has_overlay = bool(self._slice_specs)
        if not has_overlay:
            removed = self._remove_pyvista_slice_overlay()
            if removed or not self._slice_specs:
                self._draw_canvas()
                return True
            return False

        pv, _qt_interactor = _pyvista_modules()
        if pv is None:
            return False

        self._remove_pyvista_slice_overlay()
        self._draw_pyvista_slices(plotter, pv)
        self._draw_canvas()
        return True

    def _remove_pyvista_point_overlay(self) -> bool:
        removed = self._remove_pyvista_actor_group(self._pyvista_point_overlay_actors)
        self._pyvista_point_overlay_actors = []
        return removed

    def _remove_pyvista_line_overlay(self) -> bool:
        removed = self._remove_pyvista_actor_group(self._pyvista_line_overlay_actors)
        self._pyvista_line_overlay_actors = []
        return removed

    def _remove_pyvista_pick_actor(self) -> bool:
        if self._backend_name != "pyvista":
            return False
        plotter = getattr(self._canvas, "plotter", None)
        actor = getattr(self, "_pyvista_pick_actor", None)
        if plotter is None or actor is None:
            return False
        try:
            plotter.remove_actor(actor, render=False)
        except Exception:  # noqa: BLE001
            return False
        self._pyvista_pick_actor = None
        return True

    def _try_update_pyvista_point_overlay(self) -> bool:
        if (
            self._backend_name != "pyvista"
            or self._shutting_down
            or self._atoms is None
            or len(self._atoms) == 0
        ):
            return False

        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False

        if not self._point_overlay_specs:
            if not self._remove_pyvista_point_overlay():
                return False
            self._draw_canvas()
            return True

        pv, _qt_interactor = _pyvista_modules()
        if pv is None:
            return False

        self._remove_pyvista_point_overlay()
        self._draw_pyvista_point_overlays(plotter, pv)
        self._draw_canvas()
        return True

    def _try_update_pyvista_line_overlay(self) -> bool:
        if (
            self._backend_name != "pyvista"
            or self._shutting_down
            or self._atoms is None
            or len(self._atoms) == 0
        ):
            return False

        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False

        if not self._line_overlay_specs:
            if not self._remove_pyvista_line_overlay():
                return False
            self._draw_canvas()
            return True

        pv, _qt_interactor = _pyvista_modules()
        if pv is None:
            return False

        self._remove_pyvista_line_overlay()
        self._draw_pyvista_line_overlays(plotter, pv)
        self._draw_canvas()
        return True

    def _remove_pyvista_selection_overlay(self) -> bool:
        if self._backend_name != "pyvista":
            return False
        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False

        actor = self._pyvista_selection_actor
        if actor is None:
            return False
        removed = False
        try:
            plotter.remove_actor(actor, render=False)
            removed = True
        except Exception:  # noqa: BLE001
            removed = False
        self._pyvista_selection_actor = None
        return removed

    def _try_update_pyvista_selection_overlay(self) -> bool:
        if self._backend_name != "pyvista" or self._shutting_down:
            return False

        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None or not callable(getattr(plotter, "add_mesh", None)):
            return False

        has_selection = (
            bool(self._selected_indexes)
            and self._pyvista_all_atom_positions.size != 0
            and self._pyvista_all_atom_radii.size != 0
        )
        if not has_selection:
            if not self._remove_pyvista_selection_overlay():
                return False
            self._draw_canvas()
            return True

        pv, _qt_interactor = _pyvista_modules()
        if pv is None:
            return False

        self._remove_pyvista_selection_overlay()
        self._draw_pyvista_selection(
            plotter, pv, self._pyvista_all_atom_positions, self._pyvista_all_atom_radii
        )
        self._draw_canvas()
        return True

    def _try_update_pyvista_atom_geometry(self) -> bool:
        if (
            self._backend_name != "pyvista"
            or self._shutting_down
            or self._atoms is None
            or len(self._atoms) == 0
        ):
            return False
        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False
        pv, _qt_interactor = _pyvista_modules()
        if pv is None:
            return False

        positions, _numbers, _symbols, colors, radii, _extra_points = self._scene_geometry()
        scene_center = np.mean(positions, axis=0) if len(positions) else np.zeros(3, dtype=float)
        self._configure_pyvista_lighting(plotter, pv, focal_point=scene_center)
        self._pyvista_all_atom_positions = positions.copy()
        self._pyvista_all_atom_radii = radii.copy()
        self._pyvista_all_atom_colors = colors.copy()
        visible_mask = self._visible_atom_mask(positions)
        visible_indices = np.flatnonzero(visible_mask)
        visible_spins = self._filtered_spin_values(self._spins, visible_mask)
        self._pyvista_atom_lookup_indices = visible_indices.copy()
        draw_positions = positions[visible_indices]
        draw_radii = radii[visible_indices]
        draw_colors = colors[visible_indices]
        self._pyvista_atom_positions = draw_positions.copy()
        self._pyvista_atom_radii = draw_radii.copy()
        self._pyvista_atom_colors = draw_colors.copy()
        if len(draw_positions) == 0:
            return False

        self._remove_pyvista_selection_overlay()
        self._remove_pyvista_pick_actor()
        if self._pyvista_atom_actor is not None:
            try:
                plotter.remove_actor(self._pyvista_atom_actor, render=False)
            except Exception:  # noqa: BLE001
                pass
            self._pyvista_atom_actor = None

        atom_alpha = 0.5 if self._show_spins.isChecked() and self._spins is not None else 1.0
        atom_subdivisions = _atom_pyvista_sphere_subdivisions(len(draw_positions))
        atom_glyphs = _build_pyvista_atom_glyph_mesh(
            pv,
            positions=draw_positions,
            radii=draw_radii,
            colors=draw_colors,
            subdivisions=atom_subdivisions,
        )
        self._pyvista_atom_actor = plotter.add_mesh(
            atom_glyphs,
            scalars="atom_rgb",
            rgb=True,
            opacity=float(atom_alpha),
            smooth_shading=True,
            ambient=0.74,
            diffuse=0.22,
            specular=0.0,
            specular_power=1.0,
            pbr=False,
            render=False,
        )

        pick_points = pv.PolyData(draw_positions)
        pick_points["pick_radius"] = np.maximum(1.8 * draw_radii, 0.45)
        self._pyvista_pick_actor = plotter.add_mesh(
            pick_points,
            color="#000000",
            opacity=0.01,
            point_size=float(max(16.0, np.max(np.maximum(32.0 * draw_radii, 18.0)))),
            render_points_as_spheres=True,
            pickable=True,
            name="atom-pick-points",
            render=False,
        )

        self._try_update_pyvista_spin_overlay() if self._show_spins.isChecked() and visible_spins is not None else self._remove_pyvista_actor_group(
            self._pyvista_spin_actors
        )
        if not (self._show_spins.isChecked() and visible_spins is not None):
            self._pyvista_spin_actors = []
        self._try_update_pyvista_bond_overlay() if self._show_bonds.isChecked() else self._remove_pyvista_actor_group(
            self._pyvista_bond_actors
        )
        if not self._show_bonds.isChecked():
            self._pyvista_bond_actors = []
        if self._selected_indexes:
            self._try_update_pyvista_selection_overlay()
        self._pyvista_last_camera_signature = self._pyvista_camera_signature()
        self._draw_canvas()
        return True

    def _remove_pyvista_actor_group(self, actors: list[object]) -> bool:
        if self._backend_name != "pyvista":
            return False
        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None or not actors:
            return False

        removed = False
        for actor in actors:
            try:
                plotter.remove_actor(actor, render=False)
                removed = True
            except Exception:  # noqa: BLE001
                continue
        return removed

    @staticmethod
    def _set_pyvista_actor_opacity(actor: object | None, opacity: float) -> bool:
        if actor is None:
            return False
        property_getter = getattr(actor, "GetProperty", None)
        if not callable(property_getter):
            return False
        try:
            vtk_property = property_getter()
            vtk_property.SetOpacity(float(opacity))
        except Exception:  # noqa: BLE001
            return False
        return True

    def _try_update_pyvista_bond_overlay(self) -> bool:
        if self._backend_name != "pyvista" or self._shutting_down:
            return False
        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False

        removed = self._remove_pyvista_actor_group(self._pyvista_bond_actors)
        self._pyvista_bond_actors = []
        if not self._show_bonds.isChecked():
            if removed:
                self._draw_canvas()
            return removed

        if (
            self._atoms is None
            or (
                self._pyvista_all_atom_positions.size == 0
                and self._pyvista_atom_positions.size == 0
            )
            or (self._pyvista_all_atom_radii.size == 0 and self._pyvista_atom_radii.size == 0)
            or (self._pyvista_all_atom_colors.size == 0 and self._pyvista_atom_colors.size == 0)
        ):
            return False

        pv, _qt_interactor = _pyvista_modules()
        if pv is None:
            return False

        atom_alpha = 0.5 if self._show_spins.isChecked() and self._spins is not None else 1.0
        bond_alpha = float(np.clip(0.84 * atom_alpha, 0.2, 0.8))
        positions = (
            self._pyvista_all_atom_positions
            if self._pyvista_all_atom_positions.size != 0
            else self._pyvista_atom_positions
        )
        colors = (
            self._pyvista_all_atom_colors
            if self._pyvista_all_atom_colors.size != 0
            else self._pyvista_atom_colors
        )
        radii = (
            self._pyvista_all_atom_radii
            if self._pyvista_all_atom_radii.size != 0
            else self._pyvista_atom_radii
        )
        visible_mask = self._visible_atom_mask(positions)
        self._pyvista_bond_actors = self._draw_pyvista_bonds(
            plotter,
            pv,
            positions,
            colors,
            radii,
            visible_mask=visible_mask,
            alpha=bond_alpha,
        )
        self._draw_canvas()
        return True

    def _try_update_pyvista_spin_overlay(self) -> bool:
        if self._backend_name != "pyvista" or self._shutting_down:
            return False
        plotter = getattr(self._canvas, "plotter", None)
        if plotter is None:
            return False

        atom_alpha = 0.5 if self._show_spins.isChecked() and self._spins is not None else 1.0
        updated_atoms = self._set_pyvista_actor_opacity(self._pyvista_atom_actor, atom_alpha)

        removed = self._remove_pyvista_actor_group(self._pyvista_spin_actors)
        self._pyvista_spin_actors = []
        has_spins = (
            self._show_spins.isChecked()
            and self._spins is not None
            and (
                self._pyvista_all_atom_positions.size != 0 or self._pyvista_atom_positions.size != 0
            )
            and (self._pyvista_all_atom_radii.size != 0 or self._pyvista_atom_radii.size != 0)
        )
        if not has_spins:
            if self._show_bonds.isChecked():
                self._try_update_pyvista_bond_overlay()
            if removed or updated_atoms:
                self._draw_canvas()
            return removed or updated_atoms

        pv, _qt_interactor = _pyvista_modules()
        if pv is None:
            return False

        positions = (
            self._pyvista_all_atom_positions
            if self._pyvista_all_atom_positions.size != 0
            else self._pyvista_atom_positions
        )
        radii = (
            self._pyvista_all_atom_radii
            if self._pyvista_all_atom_radii.size != 0
            else self._pyvista_atom_radii
        )
        visible_mask = self._visible_atom_mask(positions)
        visible_spins = self._filtered_spin_values(self._spins, visible_mask)
        if visible_spins is None:
            return removed or updated_atoms
        draw_positions = positions[visible_mask]
        draw_radii = radii[visible_mask]
        self._pyvista_spin_actors = self._draw_pyvista_spins(
            plotter,
            pv,
            draw_positions,
            draw_radii,
            visible_spins,
        )
        if self._show_bonds.isChecked():
            self._try_update_pyvista_bond_overlay()
        self._draw_canvas()
        return True

    def _try_update_pyvista_visibility_overlay(self) -> bool:
        if self._backend_name != "pyvista" or self._shutting_down:
            return False
        source = self.sender()
        if source is self._show_bonds:
            return self._try_update_pyvista_bond_overlay()
        if source is self._show_spins:
            return self._try_update_pyvista_spin_overlay()
        return False

    def _draw_pyvista_selection(
        self, plotter, pv, positions: np.ndarray, radii: np.ndarray
    ) -> None:
        if not self._selected_indexes:
            return
        style = _viewer_style()
        highlight = None
        for atom_index in self._selected_indexes:
            if atom_index < 0 or atom_index >= len(positions):
                continue
            point = positions[atom_index]
            radius = float(radii[atom_index])
            sphere = _build_pyvista_sphere_mesh(
                pv,
                radius=1.18 * radius,
                center=point,
                subdivisions=4,
            )
            if highlight is None:
                highlight = sphere
                continue
            try:
                highlight = highlight.merge(sphere, inplace=False)
            except Exception:  # noqa: BLE001
                continue
        if highlight is None:
            return
        self._pyvista_selection_actor = plotter.add_mesh(
            highlight,
            color=style.selection_fill,
            opacity=0.34,
            smooth_shading=True,
            ambient=0.98,
            diffuse=0.04,
            specular=0.0,
            specular_power=1.0,
            pbr=False,
            render=False,
        )

    def _draw_pyvista_bonds(
        self,
        plotter,
        pv,
        positions: np.ndarray,
        colors: np.ndarray,
        radii: np.ndarray,
        *,
        visible_mask: np.ndarray | None = None,
        alpha: float,
    ) -> list[object]:
        assert self._atoms is not None
        actors: list[object] = []
        if visible_mask is None or len(visible_mask) != len(positions):
            visible_mask = np.ones((len(positions),), dtype=bool)
            if len(positions) == len(self._atoms):
                visible_mask = self._visible_atom_mask(positions)
        cutoffs = 1.15 * covalent_radii[self._atoms.get_atomic_numbers()]
        neighbour_list = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
        neighbour_list.update(self._atoms)
        cell = np.asarray(self._atoms.cell, dtype=float)
        bond_pairs: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]
        ] = []
        for atom_index in range(len(self._atoms)):
            if atom_index >= len(positions) or not visible_mask[atom_index]:
                continue
            neighbours, offsets = neighbour_list.get_neighbors(atom_index)
            start = positions[atom_index]
            for neighbour_index, offset in zip(neighbours, offsets):
                if neighbour_index <= atom_index or np.any(np.asarray(offset, dtype=int) != 0):
                    continue
                if neighbour_index >= len(positions) or not visible_mask[neighbour_index]:
                    continue
                end = positions[neighbour_index] + np.dot(offset, cell)
                bond_radius = float(
                    np.clip(0.18 * min(radii[atom_index], radii[neighbour_index]), 0.06, 0.16)
                )
                bond_pairs.append(
                    (
                        start,
                        end,
                        colors[atom_index],
                        colors[neighbour_index],
                        bond_radius,
                        float(radii[atom_index]),
                        float(radii[neighbour_index]),
                    )
                )

        if _should_use_fast_pyvista_bond_cylinders(len(self._atoms), len(bond_pairs)):
            tube_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
            for (
                start,
                end,
                start_color,
                end_color,
                bond_radius,
                start_radius,
                end_radius,
            ) in bond_pairs:
                visible_segment = _trim_bond_segment_to_surfaces(
                    start,
                    end,
                    start_radius=start_radius,
                    end_radius=end_radius,
                )
                if visible_segment is None:
                    continue
                bond_color = 0.5 * (
                    np.asarray(start_color, dtype=float) + np.asarray(end_color, dtype=float)
                )
                tube_segments.append(
                    (visible_segment[0], visible_segment[1], bond_color, bond_radius)
                )
            bond_tubes = _build_pyvista_bond_tube_mesh(pv, segments=tube_segments)
            if bond_tubes is not None:
                actors.append(
                    plotter.add_mesh(
                        bond_tubes,
                        scalars="rgb",
                        rgb=True,
                        opacity=float(alpha),
                        smooth_shading=True,
                        ambient=0.58,
                        diffuse=0.28,
                        specular=0.0,
                        specular_power=1.0,
                        pbr=False,
                        render=False,
                    )
                )
                return actors

        if _should_use_fast_pyvista_bond_lines(len(self._atoms), len(bond_pairs)):
            line_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for (
                start,
                end,
                start_color,
                end_color,
                _bond_radius,
                _start_radius,
                _end_radius,
            ) in bond_pairs:
                midpoint = 0.5 * (start + end)
                line_segments.append((start, midpoint, start_color))
                line_segments.append((midpoint, end, end_color))
            line_mesh = _build_pyvista_bond_line_mesh(pv, segments=line_segments)
            if line_mesh is not None:
                actors.append(
                    plotter.add_mesh(
                        line_mesh,
                        scalars="rgb",
                        rgb=True,
                        opacity=float(alpha),
                        line_width=2.2,
                        lighting=False,
                        pbr=False,
                        render=False,
                    )
                )
                return actors

        for (
            start,
            end,
            start_color,
            end_color,
            bond_radius,
            _start_radius,
            _end_radius,
        ) in bond_pairs:
            midpoint = 0.5 * (start + end)
            actor = self._add_pyvista_cylinder(
                plotter,
                pv,
                start,
                midpoint,
                radius=bond_radius,
                color=start_color,
                alpha=alpha,
            )
            if actor is not None:
                actors.append(actor)
            actor = self._add_pyvista_cylinder(
                plotter,
                pv,
                midpoint,
                end,
                radius=bond_radius,
                color=end_color,
                alpha=alpha,
            )
            if actor is not None:
                actors.append(actor)
        return actors

    def _add_pyvista_cylinder(
        self,
        plotter,
        pv,
        start: np.ndarray,
        end: np.ndarray,
        *,
        radius: float,
        color: np.ndarray,
        alpha: float,
    ):
        direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
        height = float(np.linalg.norm(direction))
        if height <= 1e-12:
            return None
        center = 0.5 * (np.asarray(start, dtype=float) + np.asarray(end, dtype=float))
        cylinder = pv.Cylinder(
            center=tuple(float(value) for value in center),
            direction=tuple(float(value) for value in direction / height),
            radius=float(radius),
            height=float(height),
            resolution=18,
        )
        return plotter.add_mesh(
            cylinder,
            color=tuple(float(component) for component in color),
            opacity=float(alpha),
            smooth_shading=True,
            ambient=0.58,
            diffuse=0.28,
            specular=0.0,
            specular_power=1.0,
            pbr=False,
            render=False,
        )

    def _draw_pyvista_spins(
        self, plotter, pv, positions: np.ndarray, radii: np.ndarray, spins: np.ndarray
    ) -> list[object]:
        actors: list[object] = []
        for position, radius, spin in zip(positions, radii, spins):
            spin_array = np.asarray(spin, dtype=float)
            spin_vector = (
                spin_array
                if spin_array.shape == (3,)
                else np.array([0.0, 0.0, float(spin_array)], dtype=float)
            )
            magnitude = float(np.linalg.norm(spin_vector))
            if magnitude <= 1e-10:
                continue
            direction_vector = spin_vector / magnitude
            direction = 1.0 if direction_vector[2] >= 0.0 else -1.0
            sphere_radius = float(radius)
            arrow_length = max(0.3, magnitude * 0.85) + 0.4 * sphere_radius
            half_length = 0.5 * arrow_length
            head_length = min(max(0.12, arrow_length * 0.38), half_length * 0.9)
            shaft_tip_offset = max(half_length - head_length, 0.05)
            shaft_radius = max(0.055, 0.12 * sphere_radius)
            head_radius = max(0.12, 0.2 * sphere_radius)
            tail = np.asarray(position, dtype=float) - direction_vector * half_length
            shaft_end = np.asarray(position, dtype=float) + direction_vector * shaft_tip_offset
            tip = np.asarray(position, dtype=float) + direction_vector * half_length
            color = "#1f78b4" if direction > 0 else "#d7301f"
            actor = self._add_pyvista_cylinder(
                plotter,
                pv,
                tail,
                shaft_end,
                radius=shaft_radius,
                color=np.array(QColor(color).getRgbF()[:3]),
                alpha=0.88,
            )
            if actor is not None:
                actors.append(actor)
            cone = pv.Cone(
                center=tuple(float(value) for value in 0.5 * (shaft_end + tip)),
                direction=tuple(float(value) for value in direction_vector),
                height=float(np.linalg.norm(tip - shaft_end)),
                radius=float(head_radius),
                resolution=18,
            )
            actors.append(
                plotter.add_mesh(
                    cone,
                    color=color,
                    opacity=0.9,
                    smooth_shading=True,
                    ambient=0.58,
                    diffuse=0.28,
                    specular=0.0,
                    specular_power=1.0,
                    pbr=False,
                    render=False,
                )
            )
        return actors

    def _draw_crystallographic_axes(self) -> None:
        if self._shutting_down:
            return
        if not self.isVisible() or self._render_deferred():
            # Reading a fresh plotter's camera makes PyVista reset it, which
            # renders. A hidden viewer syncs the triad when it is shown, and a
            # pending rebuild syncs it when it draws.
            self._triad_draw_pending = True
            return
        cell = None if self._atoms is None else np.asarray(self._atoms.cell, dtype=float)
        plotter = getattr(self._canvas, "plotter", None)
        camera = getattr(plotter, "camera", None)
        if camera is None:
            return
        camera_position_raw = getattr(camera, "position", None)
        focal_point_raw = getattr(camera, "focal_point", None)
        view_up_raw = getattr(camera, "up", None)
        if camera_position_raw is None or focal_point_raw is None or view_up_raw is None:
            return
        camera_position = np.asarray(camera_position_raw, dtype=float)
        focal_point = np.asarray(focal_point_raw, dtype=float)
        view_up = np.asarray(view_up_raw, dtype=float)
        if cell is None:
            self._triad_axis = None
            self._triad_canvas.set_scene(
                cell=np.zeros((3, 3), dtype=float),
                camera_position=camera_position,
                focal_point=focal_point,
                view_up=view_up,
            )
            self._draw_triad()
            return
        if self._triad_axis is None:
            self._triad_axis = _AxesProxy(self._camera_state)
        else:
            self._triad_axis.set_from_state(self._camera_state)
        self._triad_canvas.set_scene(
            cell=cell,
            camera_position=camera_position,
            focal_point=focal_point,
            view_up=view_up,
        )
        self._draw_triad()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self.shutdown()
        super().closeEvent(event)

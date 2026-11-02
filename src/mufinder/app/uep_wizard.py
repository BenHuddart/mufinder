"""Interactive UEP analysis dialog for one completed source calculation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from matplotlib import cm, colormaps, colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import QItemSelectionModel, QPointF, QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QSpinBox,
    QTabBar,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from mufinder.app import workflow_table_utils
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer, MUON_VIEWER_RADIUS_ANG
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.plot_canvas import create_plot_canvas
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import (
    cluster_color_map,
    adjust_font_size,
    body_font_family,
    card_margins,
    controls_scroll_area,
    current_palette,
    form_spacing,
    page_margins,
    set_progress_bar_state,
    style_hint_text,
    style_plot_axes,
    style_section_subtitle,
    style_section_title,
    style_warning_text,
    use_square_top_table_corners,
)
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.science.site_generation import generate_random_sites, lattice_frac_to_cart
from mufinder.science.uep import (
    DEFAULT_FORCE_TOLERANCE_EV_PER_ANG,
    DEFAULT_OPTIMIZER_MAX_STEPS,
    IONIC_GAUSSIAN_MIN_RESOLUTION,
    UepError,
    UepField,
    UepGridSliceSample,
    UepIsosurfaceMesh,
    UepLineSample,
    UepOptimizationResult,
    UepPlaneSample,
    UepSlicePlane,
    UepPotentialGrid,
    UepProbeResult,
    UepSourceData,
    cartesian_normal_to_lattice_plane_components,
    classification_from_dict,
    extract_isosurface,
    ionic_gaussian_resolution,
    load_uep_source,
    optimization_result_from_dict,
    optimize_muon_position,
    plane_from_best_fit,
    plane_from_fractional_anchor_and_miller,
    plane_from_three_points,
    potential_above_minimum,
    potential_grid,
    potential_level_ceiling,
    preview_potential_grid,
    probe_point,
    sample_line,
    sample_plane,
    sample_grid_slice,
    write_potential_grid_xsf,
)
from mufinder.science.uep_clustering import (
    DEFAULT_GROUPING_TOLERANCE_ANG,
    UepBatchSummary,
    UepClusterSummary,
    UepMinimaDistances,
    group_uep_minima,
    measure_uep_minima,
    position_in_source_cell,
    relaxed_site_number_map,
    uep_batch_summary_from_dict,
    saved_uep_search_log,
    uep_search_log,
    uep_search_outcome,
)

_ISOSURFACE_SLIDER_DEBOUNCE_MS = 120
#: How long after the last edit of the grouping tolerance its groups are saved and journalled.
_GROUPING_SETTLED_MS = 1500
_GROUPING_TOLERANCE_MINIMUM_ANG = 0.001
_RETIRED_DEFAULT_MUON_LABEL = re.compile(r"Site \d+")
_GROUPING_TOLERANCE_TOOLTIP = (
    "Two minima are one site when one lies within this distance of any symmetry image of the "
    "other and their classical energies differ by less than 1 meV."
)
_SLICE_DEFAULT_POINTS = 161
_SLICE_LIVE_PREVIEW_MAX_POINTS = 49
_SLICE_SURFACE_ALPHA = 0.72
_GW_FACTOR_MINIMUM = 1.0
_GW_FACTOR_MAXIMUM = 10.0
_GW_FACTOR_TOOLTIP = (
    "Sets the width of the ionic Gaussians: width = pseudopotential core radius / this factor. "
    "The batch search and the potential both use it. Press Enter to apply a new value."
)
_POTENTIAL_LEVEL_TOOLTIP = (
    "Levels are in eV above the lowest point of the potential grid. The potential's own zero "
    "is its cell average, so only differences between its values mean anything."
)
#: A window saved with the Isosurfaces and Slices tabs, or with the seven tabs before them.
_SAVED_TAB_TO_VISIBLE = {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0}
_POTENTIAL_CONTROLS_WIDTH = 372
_DEFAULT_MAX_START_ATTEMPTS = 5000
_BATCH_FORCE_TOLERANCE_TOOLTIP = (
    "Force threshold for each start, in eV/Å: the optimizer stops when the norm of the "
    "potential's gradient falls below this value."
)
_VIEWER_SPIN_DISPLAY_THRESHOLD = 0.1


def _coerced_spin_values(raw: object, count: int) -> list[float] | None:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or len(raw) != count:
        return None
    values: list[float] = []
    for value in raw:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def _viewer_spin_values(values: Sequence[float]) -> list[float]:
    return [
        float(value) if abs(float(value)) >= _VIEWER_SPIN_DISPLAY_THRESHOLD else 0.0
        for value in values
    ]


def _periodic_fractional_distance_squared(first: Sequence[float], second: Sequence[float]) -> float:
    total = 0.0
    for left, right in zip(first[:3], second[:3], strict=False):
        delta = float(left) - float(right)
        delta -= round(delta)
        total += delta * delta
    return total


def _align_spins_to_fractional_structure_order(
    *,
    target_species: Sequence[str],
    target_fractional_positions: Sequence[Sequence[float]],
    source_structure: Structure,
    source_spins: Sequence[float],
    tolerance: float = 1e-4,
) -> list[float] | None:
    if (
        len(source_structure.species) != len(target_species)
        or len(source_structure.fractional_positions) != len(target_fractional_positions)
        or len(source_spins) != len(target_species)
    ):
        return None

    tolerance_squared = tolerance * tolerance
    unused = set(range(len(source_structure.species)))
    aligned: list[float] = []
    for target_symbol, target_position in zip(
        target_species, target_fractional_positions, strict=True
    ):
        candidates = [
            (
                _periodic_fractional_distance_squared(
                    target_position, source_structure.fractional_positions[index]
                ),
                index,
            )
            for index in unused
            if str(source_structure.species[index]) == str(target_symbol)
        ]
        if not candidates:
            return None
        distance_squared, source_index = min(candidates)
        if distance_squared > tolerance_squared:
            return None
        unused.remove(source_index)
        aligned.append(float(source_spins[source_index]))
    return aligned


_BATCH_VIEWER_IDLE_TEXT = "Showing the host structure. Select one or more sites or starts to overlay their muon positions."
_BATCH_TABLE_INDENTATION = 10
_BATCH_TABLE_HEADERS = [
    "Site / Start",
    "Size",
    "fX",
    "fY",
    "fZ",
    "Classical energy\n(eV)",
    "ZPE\n(eV)",
    "Total energy\n(eV)",
]
_VIEWER_MUON_CANDIDATE_HEADERS = [
    "Site / Start",
    "fX",
    "fY",
    "fZ",
    "Classical energy\n(eV)",
    "ZPE\n(eV)",
    "Total energy\n(eV)",
]
_VIEWER_MUON_CURRENT_HEADERS = [
    "Muon",
    "Source",
    "fX",
    "fY",
    "fZ",
    "Classical energy\n(eV)",
    "ZPE\n(eV)",
    "Total energy\n(eV)",
    "Isosurface",
]


_THEORY_TEXT = (
    "UEP in MuFinder follows the method of Simone Sturniolo and Leandro Liborio, "
    '"Computational prediction of muon stopping sites: A novel take on the unperturbed '
    'electrostatic potential method," J. Chem. Phys. 153, 044111 (2020), '
    "DOI: 10.1063/5.0012381.\n\n"
    "The method assumes a neutral unperturbed host crystal and a positively charged "
    "muon rather than muonium. MuFinder combines the CASTEP electronic density with a "
    "Gaussian ionic model, evaluates the electrostatic potential in reciprocal space, "
    "and uses gradients and Hessians to identify stationary points.\n\n"
    "The paper also highlights that special Wyckoff positions without free coordinates "
    "are stationary points of symmetry-preserving scalar fields, so high-symmetry empty "
    "interstitial sites are useful UEP candidates. Symmetry alone is not enough to "
    "guarantee a minimum, which is why MuFinder also exposes Hessian classification, "
    "zero-point estimates, and grouping of the batch-search minima.\n\n"
    "For locally stable minima, MuFinder estimates the harmonic zero-point energy with "
    "E_zp = 1/2 sum_i hbar omega_i and omega_i = sqrt(lambda_i / m_mu), where lambda_i "
    "are the total-potential Hessian eigenvalues converted into SI units. Saddles and "
    "other unstable extrema are reported without a ZPE because one or more curvatures "
    "are negative.\n\n"
    "UEP tends to be more reliable in metallic or weakly perturbative hosts. It is much "
    "less reliable when the muon forms a strong local bond or reconstructs the lattice, "
    "as can happen in some insulating or semiconducting systems."
)


@dataclass
class _IsosurfaceLayerState:
    label: str
    key: str = ""
    #: In eV above the grid's minimum.
    level_ev: float | None = None
    color: str = "#dc2626"
    mesh: UepIsosurfaceMesh | None = None
    managed_muon_key: str | None = None


@dataclass
class _SlicePlaneState:
    label: str
    plane: UepSlicePlane | None = None
    sample: UepGridSliceSample | None = None


@dataclass(frozen=True, slots=True)
class _BatchMuonCandidate:
    key: str
    label: str
    cluster_label: str
    result_index: int
    cluster_index: int
    fractional_position: tuple[float, float, float]
    cartesian_position_ang: tuple[float, float, float]
    classical_energy_ev: float
    zero_point_energy_ev: float | None
    total_energy_ev: float | None
    display_fractional_position: tuple[float, float, float]


@dataclass(slots=True)
class _ManagedViewerMuonState:
    key: str
    label: str
    source_label: str
    source_kind: str
    fractional_position: tuple[float, float, float]
    cartesian_position_ang: tuple[float, float, float]
    result_index: int | None = None
    cluster_index: int | None = None
    classical_energy_ev: float | None = None
    zero_point_energy_ev: float | None = None
    total_energy_ev: float | None = None
    linked_isosurface_layer_key: str | None = None
    #: A search result's position as the tables read it, in the input structure's cell.
    #: ``fractional_position`` places the muon in the viewer, in the job's cell.
    display_fractional_position: tuple[float, float, float] | None = None


def _pyvista_camera_snapshots_match(
    first: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None,
    second: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None,
) -> bool:
    if first is second:
        return True
    if first is None or second is None:
        return False
    return (
        np.allclose(first[0], second[0], rtol=0.0, atol=1e-9)
        and np.allclose(first[1], second[1], rtol=0.0, atol=1e-9)
        and np.allclose(first[2], second[2], rtol=0.0, atol=1e-9)
        and np.isclose(float(first[3]), float(second[3]), rtol=0.0, atol=1e-9)
    )


class _TripleInput(QWidget):
    """Compact row of three decimal controls for fractional coordinates."""

    def __init__(
        self,
        *,
        label_x: str = "fX",
        label_y: str = "fY",
        label_z: str = "fZ",
        decimals: int = 6,
        minimum: float = -10.0,
        maximum: float = 10.0,
        label_width: int = 18,
        field_minimum_width: int = 124,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("slicePlaneTripleInput")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self._x = DecimalLineEdit(self)
        self._y = DecimalLineEdit(self)
        self._z = DecimalLineEdit(self)
        for control in (self._x, self._y, self._z):
            control.setRange(float(minimum), float(maximum))
            control.setDecimals(int(decimals))
            control.setMinimumWidth(int(field_minimum_width))
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        for label, control in ((label_x, self._x), (label_y, self._y), (label_z, self._z)):
            caption = QLabel(label, self)
            caption.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            caption.setFixedWidth(int(label_width))
            layout.addWidget(caption)
            layout.addWidget(control)
        layout.addStretch(1)

    def set_values(self, values: tuple[float, float, float]) -> None:
        self._x.setValue(float(values[0]))
        self._y.setValue(float(values[1]))
        self._z.setValue(float(values[2]))

    def values(self) -> tuple[float, float, float]:
        return (self._x.value(), self._y.value(), self._z.value())


class _RangeSlider(QWidget):
    """Simple dual-handle slider for selecting a bounded numeric interval."""

    valuesChanged = Signal(int, int)
    interactionFinished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 1000
        self._lower_value = 250
        self._upper_value = 750
        self._handle_radius = 8
        self._active_handle: str | None = None
        self.setMinimumHeight(34)

    def minimum(self) -> int:
        return self._minimum

    def maximum(self) -> int:
        return self._maximum

    def lowerValue(self) -> int:  # noqa: N802 - Qt API naming.
        return self._lower_value

    def upperValue(self) -> int:  # noqa: N802 - Qt API naming.
        return self._upper_value

    def isSliderDown(self) -> bool:  # noqa: N802 - Qt API naming.
        return self._active_handle is not None

    def setRange(self, minimum: int, maximum: int) -> None:  # noqa: N802 - Qt API naming.
        self._minimum = int(minimum)
        self._maximum = max(int(maximum), self._minimum)
        self.setValues(self._lower_value, self._upper_value)

    def setLowerValue(self, value: int) -> None:  # noqa: N802 - Qt API naming.
        self.setValues(int(value), self._upper_value)

    def setUpperValue(self, value: int) -> None:  # noqa: N802 - Qt API naming.
        self.setValues(self._lower_value, int(value))

    def setValues(self, lower: int, upper: int) -> None:
        clamped_lower = min(max(int(lower), self._minimum), self._maximum)
        clamped_upper = min(max(int(upper), self._minimum), self._maximum)
        if clamped_lower > clamped_upper:
            clamped_lower, clamped_upper = clamped_upper, clamped_lower
        changed = (clamped_lower != self._lower_value) or (clamped_upper != self._upper_value)
        self._lower_value = clamped_lower
        self._upper_value = clamped_upper
        if changed:
            self.valuesChanged.emit(self._lower_value, self._upper_value)
        self.update()

    def _groove_rect(self) -> tuple[int, int, int]:
        left = self._handle_radius + 2
        right = max(left + 1, self.width() - self._handle_radius - 2)
        return left, right, right - left

    def _handle_x(self, value: int) -> float:
        left, _right, width = self._groove_rect()
        if self._maximum == self._minimum:
            return float(left)
        fraction = (float(value) - float(self._minimum)) / float(self._maximum - self._minimum)
        return float(left) + (fraction * float(width))

    def _value_for_x(self, x_position: float) -> int:
        left, _right, width = self._groove_rect()
        if width <= 0 or self._maximum == self._minimum:
            return self._minimum
        fraction = (float(x_position) - float(left)) / float(width)
        fraction = min(max(fraction, 0.0), 1.0)
        return int(round(float(self._minimum) + fraction * float(self._maximum - self._minimum)))

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API naming.
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            palette = current_palette()
            left, right, _width = self._groove_rect()
            center_y = self.height() / 2.0
            groove_rect = QRectF(float(left), center_y - 2.5, float(right - left), 5.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(palette.border))
            painter.drawRoundedRect(groove_rect, 2.5, 2.5)

            lower_x = self._handle_x(self._lower_value)
            upper_x = self._handle_x(self._upper_value)
            selection_rect = QRectF(lower_x, center_y - 3.5, max(2.0, upper_x - lower_x), 7.0)
            painter.setBrush(QColor(palette.accent))
            painter.drawRoundedRect(selection_rect, 3.5, 3.5)

            for handle_x in (lower_x, upper_x):
                painter.setPen(QPen(QColor(palette.accent_pressed), 1.0))
                painter.setBrush(QColor(palette.bg_surface))
                painter.drawEllipse(
                    QPointF(handle_x, center_y),
                    float(self._handle_radius),
                    float(self._handle_radius),
                )
        finally:
            painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        lower_distance = abs(event.position().x() - self._handle_x(self._lower_value))
        upper_distance = abs(event.position().x() - self._handle_x(self._upper_value))
        self._active_handle = "lower" if lower_distance <= upper_distance else "upper"
        self.mouseMoveEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._active_handle is None:
            return
        value = self._value_for_x(event.position().x())
        if self._active_handle == "lower":
            self.setLowerValue(min(value, self._upper_value))
        else:
            self.setUpperValue(max(value, self._lower_value))

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802 - Qt API naming.
        self._active_handle = None
        self.interactionFinished.emit()


class _IsosurfaceLayerTabBar(QTabBar):
    """Compact colored tabs used to switch between isosurface layers."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer_colors: list[QColor] = []
        self.setDocumentMode(True)
        self.setDrawBase(False)
        self.setExpanding(False)
        self.setUsesScrollButtons(False)
        self.setElideMode(Qt.TextElideMode.ElideNone)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_layer_colors(self, colors: list[str]) -> None:
        self._layer_colors = []
        for value in colors:
            color = QColor(str(value))
            self._layer_colors.append(color if color.isValid() else QColor("#dc2626"))
        self.update()

    def tabSizeHint(self, index: int):  # noqa: N802 - Qt API naming.
        _ = index
        return QSize(30, 26)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        palette = current_palette()
        painter.fillRect(event.rect(), QColor(palette.bg_surface))
        base_fill = QColor(palette.bg_surface_alt)
        border = QColor(palette.border)
        active_border = QColor(palette.accent)

        for index in range(self.count()):
            rect = self.tabRect(index).adjusted(0, 2, -2, -1)
            if index == self.currentIndex():
                rect = rect.adjusted(0, -2, 0, 0)
            color = (
                self._layer_colors[index] if index < len(self._layer_colors) else QColor("#dc2626")
            )
            fill = QColor(color)
            if index != self.currentIndex():
                fill.setAlpha(190)
                fill = QColor(
                    int(0.55 * base_fill.red() + 0.45 * fill.red()),
                    int(0.55 * base_fill.green() + 0.45 * fill.green()),
                    int(0.55 * base_fill.blue() + 0.45 * fill.blue()),
                )
            painter.setPen(QPen(active_border if index == self.currentIndex() else border, 1.2))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, 7, 7)

            text_color = QColor("#ffffff")
            luminance = (0.2126 * fill.redF()) + (0.7152 * fill.greenF()) + (0.0722 * fill.blueF())
            if luminance > 0.62:
                text_color = QColor("#1f1a17")
            painter.setPen(text_color)
            font = painter.font()
            font.setBold(True)
            adjust_font_size(font, -1, minimum=9)
            painter.setFont(font)
            painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), self.tabText(index))


class _IsosurfaceLegendSwatch(QFrame):
    """Small rounded color swatch used in the isosurface legend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self._color = "#dc2626"
        self._apply_style()

    def set_color(self, color: str) -> None:
        qcolor = QColor(str(color))
        if qcolor.isValid():
            self._color = qcolor.name()
            self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"border-radius: 8px;border: 1px solid rgba(0, 0, 0, 0.14);background: {self._color};"
        )


class _AddSlicePlaneDialog(QDialog):
    """Selection-aware dialog for creating a new slice plane."""

    def __init__(
        self,
        *,
        selected_positions_ang: list[tuple[float, float, float]],
        selected_positions_fractional: list[tuple[float, float, float]] | None = None,
        default_normal_components: tuple[float, float, float] | None = None,
        lattice_vectors: tuple[
            tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
        ],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Slice Plane")
        self.resize(660, 470)
        self._selected_positions_ang = [
            tuple(float(value) for value in point) for point in selected_positions_ang
        ]
        if selected_positions_fractional is None:
            fractional_positions: list[tuple[float, float, float]] = []
            lattice = np.asarray(lattice_vectors, dtype=float)
            for point in self._selected_positions_ang:
                fractional = np.linalg.solve(lattice.T, np.asarray(point, dtype=float))
                fractional_positions.append(tuple(float(value) for value in fractional))
            self._selected_positions_fractional = fractional_positions
        else:
            self._selected_positions_fractional = [
                tuple(float(value) for value in point) for point in selected_positions_fractional
            ]
        self._lattice_vectors = lattice_vectors
        self._result_plane: UepSlicePlane | None = None
        self._default_anchor_fractional = (
            self._selected_positions_fractional[0]
            if self._selected_positions_fractional
            else (0.0, 0.0, 0.0)
        )
        self._default_normal_components = (
            tuple(float(value) for value in default_normal_components)
            if default_normal_components is not None
            else (0.0, 0.0, 1.0)
        )
        self._method_rows: list[tuple[QRadioButton, QFrame, QLabel]] = []
        self._apply_styles()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        title = QLabel("Define Slice Plane", self)
        style_section_title(title)
        subtitle = QLabel(
            "Choose a crystallographic plane-construction method. Point + normal is always available, while atom-derived methods unlock as you select atoms in the viewer.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        summary_card = QFrame(self)
        summary_card.setObjectName("uepIsosurfaceStatusCard")
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(12, 10, 12, 10)
        summary_layout.setSpacing(0)
        self._summary_label = QLabel(summary_card)
        self._summary_label.setWordWrap(True)
        style_hint_text(self._summary_label)
        summary_layout.addWidget(self._summary_label)
        layout.addWidget(summary_card)

        methods_group = QGroupBox("Plane Construction Method", self)
        methods_layout = QVBoxLayout(methods_group)
        methods_layout.setContentsMargins(14, 12, 14, 12)
        methods_layout.setSpacing(10)
        self._method_group = QButtonGroup(self)
        self._point_normal = QRadioButton("Point + normal", methods_group)
        self._three_atoms = QRadioButton("Three atoms", methods_group)
        self._best_fit = QRadioButton("Best fit", methods_group)
        self._point_normal_hint = QLabel(
            "Always available. Uses the selected atom's fractional position when one is selected; otherwise you can create the plane by manually specifying the anchor. The normal starts from the current viewer direction.",
            methods_group,
        )
        self._three_atoms_hint = QLabel(
            "Requires exactly 3 selected atoms. The three atoms must not be collinear.",
            methods_group,
        )
        self._best_fit_hint = QLabel(
            "Requires 4 or more selected atoms. Fits a least-squares plane through them.",
            methods_group,
        )
        for button, hint in (
            (self._point_normal, self._point_normal_hint),
            (self._three_atoms, self._three_atoms_hint),
            (self._best_fit, self._best_fit_hint),
        ):
            self._method_group.addButton(button)
            methods_layout.addWidget(self._build_method_card(button, hint, methods_group))
            button.toggled.connect(self._refresh_method_card_styles)
        layout.addWidget(methods_group)

        geometry_group = QGroupBox("Point + Normal", self)
        geometry_layout = QVBoxLayout(geometry_group)
        geometry_layout.setContentsMargins(14, 12, 14, 12)
        geometry_layout.setSpacing(12)
        self._anchor_input = _TripleInput(
            label_x="u",
            label_y="v",
            label_z="w",
            decimals=6,
            label_width=24,
            field_minimum_width=136,
            parent=geometry_group,
        )
        self._anchor_input.set_values(self._default_anchor_fractional)
        self._normal_input = _TripleInput(
            label_x="h",
            label_y="k",
            label_z="l",
            decimals=3,
            minimum=-12.0,
            maximum=12.0,
            label_width=24,
            field_minimum_width=136,
            parent=geometry_group,
        )
        self._normal_input.set_values(self._default_normal_components)
        geometry_layout.addWidget(
            self._build_geometry_block(
                title="Anchor",
                helper="Specify the plane anchor in fractional coordinates along a, b, and c.",
                input_widget=self._anchor_input,
                parent=geometry_group,
            )
        )
        geometry_layout.addWidget(
            self._build_geometry_block(
                title="Normal",
                helper="Specify the crystallographic plane normal using Miller indices (h, k, l).",
                input_widget=self._normal_input,
                parent=geometry_group,
            )
        )
        layout.addWidget(geometry_group)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        self._create_button = buttons.addButton(
            "Create Plane", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self._create_button.clicked.connect(self._accept_plane)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._point_normal.toggled.connect(self._refresh_point_normal_editor_state)
        self._refresh_enabled_methods()

    def _build_method_card(self, button: QRadioButton, hint: QLabel, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("slicePlaneMethodCard")
        card.setProperty("selected", False)
        card.setProperty("unavailable", False)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 14, 12, 14)
        card_layout.setSpacing(6)
        card.setMinimumHeight(96)
        button.setParent(card)
        hint.setParent(card)
        hint.setWordWrap(True)
        hint.setStyleSheet("background: transparent;")
        style_hint_text(hint)
        card_layout.addWidget(button)
        card_layout.addWidget(hint)
        self._method_rows.append((button, card, hint))
        return card

    def _build_geometry_block(
        self, *, title: str, helper: str, input_widget: QWidget, parent: QWidget
    ) -> QWidget:
        block = QWidget(parent)
        block.setObjectName("slicePlaneGeometryBlock")
        block.setStyleSheet("background: transparent;")
        block_layout = QVBoxLayout(block)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(6)
        heading = QLabel(title, block)
        heading.setObjectName("slicePlaneFieldHeading")
        heading.setStyleSheet("background: transparent;")
        helper_label = QLabel(helper, block)
        helper_label.setWordWrap(True)
        helper_label.setObjectName("slicePlaneFieldHint")
        style_hint_text(helper_label)
        helper_label.setStyleSheet("background: transparent;")
        block_layout.addWidget(heading)
        block_layout.addWidget(helper_label)
        block_layout.addWidget(input_widget)
        return block

    def _apply_styles(self) -> None:
        palette = current_palette()
        self.setStyleSheet(
            "QDialog {"
            f"background: {palette.bg_primary};"
            "}"
            "QGroupBox {"
            f"background: {palette.bg_surface};"
            f"border: 1px solid {palette.border};"
            "border-radius: 16px;"
            "margin-top: 14px;"
            "padding-top: 16px;"
            "font-weight: 700;"
            f"color: {palette.text_primary};"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "left: 14px;"
            "padding: 0 8px;"
            f"color: {palette.text_primary};"
            "}"
            "QFrame#slicePlaneMethodCard {"
            f"background: {palette.bg_surface_alt};"
            f"border: 1px solid {palette.border};"
            "border-radius: 14px;"
            "}"
            'QFrame#slicePlaneMethodCard[selected="true"] {'
            f"background: {palette.bg_surface};"
            f"border: 1px solid {palette.accent};"
            "}"
            'QFrame#slicePlaneMethodCard[unavailable="true"] {'
            f"background: {palette.bg_primary};"
            f"border: 1px solid {palette.border};"
            "}"
            "QRadioButton {"
            "font-size: 15px;"
            "font-weight: 600;"
            f"color: {palette.text_primary};"
            "spacing: 10px;"
            "padding-top: 4px;"
            "padding-bottom: 4px;"
            "min-height: 28px;"
            "}"
            "QRadioButton::indicator {"
            "width: 16px;"
            "height: 16px;"
            "border-radius: 8px;"
            f"border: 1px solid {palette.border};"
            f"background: {palette.bg_surface};"
            "}"
            "QRadioButton::indicator:checked {"
            f"background: {palette.accent};"
            f"border: 1px solid {palette.accent_pressed};"
            "}"
            "QRadioButton::indicator:disabled {"
            f"background: {palette.bg_surface_alt};"
            f"border: 1px solid {palette.border};"
            "}"
            "QLabel#slicePlaneFieldHeading {"
            f"color: {palette.text_primary};"
            "font-weight: 700;"
            "}"
            "QLabel#slicePlaneFieldHint {"
            "background: transparent;"
            "}"
            "QWidget#slicePlaneGeometryBlock {"
            "background: transparent;"
            "}"
            "QWidget#slicePlaneTripleInput {"
            "background: transparent;"
            "}"
            "QPushButton {"
            "padding: 8px 16px;"
            "border-radius: 12px;"
            f"border: 1px solid {palette.border};"
            f"background: {palette.bg_surface_alt};"
            f"color: {palette.text_primary};"
            "font-weight: 600;"
            "}"
            "QPushButton:hover {"
            f"background: {palette.bg_surface};"
            "}"
            "QPushButton:pressed {"
            f"background: {palette.bg_primary};"
            "}"
        )

    def _refresh_enabled_methods(self) -> None:
        selection_count = len(self._selected_positions_ang)
        if selection_count == 0:
            self._summary_label.setText(
                "No atoms are currently selected. You can still create a plane with Point + normal by manually specifying the anchor."
            )
        elif selection_count == 1:
            self._summary_label.setText(
                "One atom is selected. Point + normal will start from that atom's fractional position by default; atom-derived methods unlock as you select more atoms."
            )
        elif selection_count == 3:
            self._summary_label.setText(
                "Three atoms are selected. You can use Point + normal or define the plane directly through those three atoms."
            )
        elif selection_count >= 4:
            self._summary_label.setText(
                "Multiple atoms are selected. All plane-construction methods are available, including best-fit."
            )
        else:
            self._summary_label.setText(
                "Two atoms are selected. Point + normal remains available; select one more atom for the three-atom method, or four or more for best-fit."
            )
        self._point_normal.setEnabled(True)
        self._three_atoms.setEnabled(selection_count == 3)
        self._best_fit.setEnabled(selection_count >= 4)
        if not (
            self._point_normal.isChecked()
            or self._three_atoms.isChecked()
            or self._best_fit.isChecked()
        ):
            self._point_normal.setChecked(True)
        self._anchor_input.set_values(self._default_anchor_fractional)
        self._normal_input.set_values(self._default_normal_components)
        self._refresh_point_normal_editor_state()
        self._refresh_method_card_styles()
        self._create_button.setEnabled(True)

    def _refresh_point_normal_editor_state(self) -> None:
        enabled = self._point_normal.isChecked()
        self._anchor_input.setEnabled(enabled)
        self._normal_input.setEnabled(enabled)

    def _refresh_method_card_styles(self, *_args) -> None:
        for button, card, hint in self._method_rows:
            card.setProperty("selected", button.isChecked())
            card.setProperty("unavailable", not button.isEnabled())
            hint.setEnabled(button.isEnabled())
            button.setStyleSheet("background: transparent;")
            style = card.style()
            style.unpolish(card)
            style.polish(card)
            card.update()

    def _accept_plane(self) -> None:
        try:
            if self._point_normal.isChecked():
                self._result_plane = plane_from_fractional_anchor_and_miller(
                    anchor_fractional=self._anchor_input.values(),
                    miller_indices=self._normal_input.values(),
                    lattice_vectors=self._lattice_vectors,
                    points_u=_SLICE_DEFAULT_POINTS,
                    points_v=_SLICE_DEFAULT_POINTS,
                )
            elif self._three_atoms.isChecked():
                self._result_plane = plane_from_three_points(
                    point_a_ang=self._selected_positions_ang[0],
                    point_b_ang=self._selected_positions_ang[1],
                    point_c_ang=self._selected_positions_ang[2],
                    lattice_vectors=self._lattice_vectors,
                    points_u=_SLICE_DEFAULT_POINTS,
                    points_v=_SLICE_DEFAULT_POINTS,
                )
            else:
                self._result_plane = plane_from_best_fit(
                    points_ang=self._selected_positions_ang,
                    lattice_vectors=self._lattice_vectors,
                    points_u=_SLICE_DEFAULT_POINTS,
                    points_v=_SLICE_DEFAULT_POINTS,
                )
        except UepError as exc:
            QMessageBox.warning(self, "Slice Plane", str(exc))
            return
        self.accept()

    def plane(self) -> UepSlicePlane | None:
        return self._result_plane


def _confirm_remove_slice_plane(parent: QWidget, plane_number: int) -> bool:
    return workflow_table_utils.confirm_destructive(
        parent,
        "Remove slice plane",
        f"Remove plane {plane_number}?",
        "The plane and its potential slice leave the viewer.",
        "Remove plane",
    )


class _EditSlicePlaneDialog(QDialog):
    """Numeric editor for an existing slice plane."""

    plane_deleted = Signal()

    def __init__(
        self,
        *,
        plane_number: int,
        plane: UepSlicePlane,
        lattice_vectors: tuple[
            tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
        ],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Edit Slice Plane {plane_number}")
        self._result_plane = plane
        self._lattice_vectors = lattice_vectors
        lattice = np.asarray(lattice_vectors, dtype=float)
        anchor_fractional = np.linalg.solve(lattice.T, np.asarray(plane.anchor_ang, dtype=float))
        normal_components = cartesian_normal_to_lattice_plane_components(
            lattice_vectors, plane.normal_ang
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        header = QLabel(f"Plane {plane_number}", self)
        style_section_title(header)
        layout.addWidget(header)

        subtitle = QLabel(
            "Edit the slice plane using the same crystallographic coordinates used when it was created.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(subtitle)

        geometry_group = QGroupBox("Point + Normal", self)
        geometry_layout = QVBoxLayout(geometry_group)
        geometry_layout.setContentsMargins(14, 12, 14, 12)
        geometry_layout.setSpacing(12)
        self._anchor = _TripleInput(
            label_x="u",
            label_y="v",
            label_z="w",
            decimals=6,
            label_width=24,
            field_minimum_width=136,
            parent=geometry_group,
        )
        self._anchor.set_values(tuple(float(value) for value in anchor_fractional))
        self._normal = _TripleInput(
            label_x="h",
            label_y="k",
            label_z="l",
            decimals=3,
            minimum=-12.0,
            maximum=12.0,
            label_width=24,
            field_minimum_width=136,
            parent=geometry_group,
        )
        self._normal.set_values(normal_components)
        anchor_block = QWidget(geometry_group)
        anchor_block.setObjectName("slicePlaneGeometryBlock")
        anchor_block.setStyleSheet("background: transparent;")
        anchor_layout = QVBoxLayout(anchor_block)
        anchor_layout.setContentsMargins(0, 0, 0, 0)
        anchor_layout.setSpacing(6)
        anchor_heading = QLabel("Anchor", anchor_block)
        anchor_heading.setObjectName("slicePlaneFieldHeading")
        anchor_heading.setStyleSheet("background: transparent;")
        anchor_hint = QLabel(
            "Specify the plane anchor in fractional coordinates along a, b, and c.", anchor_block
        )
        anchor_hint.setWordWrap(True)
        style_hint_text(anchor_hint)
        anchor_hint.setStyleSheet("background: transparent;")
        anchor_layout.addWidget(anchor_heading)
        anchor_layout.addWidget(anchor_hint)
        anchor_layout.addWidget(self._anchor)
        geometry_layout.addWidget(anchor_block)

        normal_block = QWidget(geometry_group)
        normal_block.setObjectName("slicePlaneGeometryBlock")
        normal_block.setStyleSheet("background: transparent;")
        normal_layout = QVBoxLayout(normal_block)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.setSpacing(6)
        normal_heading = QLabel("Normal", normal_block)
        normal_heading.setObjectName("slicePlaneFieldHeading")
        normal_heading.setStyleSheet("background: transparent;")
        normal_hint = QLabel(
            "Specify the crystallographic plane normal using Miller indices (h, k, l).",
            normal_block,
        )
        normal_hint.setWordWrap(True)
        style_hint_text(normal_hint)
        normal_hint.setStyleSheet("background: transparent;")
        normal_layout.addWidget(normal_heading)
        normal_layout.addWidget(normal_hint)
        normal_layout.addWidget(self._normal)
        geometry_layout.addWidget(normal_block)
        layout.addWidget(geometry_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        delete_button = buttons.addButton(
            "Remove plane", QDialogButtonBox.ButtonRole.DestructiveRole
        )
        delete_button.clicked.connect(self._delete_plane)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._plane_number = plane_number

    def _delete_plane(self) -> None:
        if not _confirm_remove_slice_plane(self, self._plane_number):
            return
        self.plane_deleted.emit()
        self.done(2)

    def edited_plane(
        self,
    ) -> UepSlicePlane:
        return plane_from_fractional_anchor_and_miller(
            anchor_fractional=self._anchor.values(),
            miller_indices=self._normal.values(),
            lattice_vectors=self._lattice_vectors,
            points_u=_SLICE_DEFAULT_POINTS,
            points_v=_SLICE_DEFAULT_POINTS,
        )


class _ManageViewerMuonsDialog(QDialog):
    """Manage the viewer's muon overlays, from the batch-search groups or custom positions."""

    add_candidates_requested = Signal(object, bool)
    add_custom_requested = Signal(object)
    remove_requested = Signal(object)
    current_selection_changed = Signal(object)
    rename_requested = Signal(str, str)

    def __init__(
        self,
        *,
        title: str,
        subtitle: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1040, 620)
        self.setMinimumSize(920, 520)
        self._candidate_entries: dict[str, _BatchMuonCandidate] = {}
        self._current_entries: dict[str, _ManagedViewerMuonState] = {}
        self._include_isosurfaces_available = False
        self._current_selection_keys: list[str] = []
        self._updating_tables = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        heading = QLabel(title, self)
        style_section_title(heading)
        layout.addWidget(heading)

        subtitle_label = QLabel(subtitle, self)
        subtitle_label.setWordWrap(True)
        style_section_subtitle(subtitle_label)
        layout.addWidget(subtitle_label)

        custom_group = QGroupBox("Add Custom Muon", self)
        custom_layout = QHBoxLayout(custom_group)
        custom_layout.setContentsMargins(14, 12, 14, 12)
        custom_layout.setSpacing(10)
        custom_caption = QLabel("Fractional coordinates", custom_group)
        custom_caption.setMinimumWidth(136)
        self._custom_fractional_input = _TripleInput(parent=custom_group)
        self._custom_fractional_input.set_values((0.25, 0.25, 0.25))
        self._custom_add_button = QPushButton("Add Muon", custom_group)
        self._custom_add_button.clicked.connect(self._emit_add_custom)
        custom_layout.addWidget(custom_caption)
        custom_layout.addWidget(self._custom_fractional_input, 1)
        custom_layout.addWidget(self._custom_add_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(custom_group)

        tables_row = QHBoxLayout()
        tables_row.setContentsMargins(0, 0, 0, 0)
        tables_row.setSpacing(12)

        candidate_group = QGroupBox("Batch-search sites", self)
        candidate_layout = QVBoxLayout(candidate_group)
        candidate_layout.setContentsMargins(12, 12, 12, 12)
        candidate_layout.setSpacing(8)
        self._candidate_table = QTreeWidget(candidate_group)
        self._candidate_table.setColumnCount(len(_VIEWER_MUON_CANDIDATE_HEADERS))
        self._candidate_table.setHeaderLabels(_VIEWER_MUON_CANDIDATE_HEADERS)
        self._candidate_table.setRootIsDecorated(True)
        self._candidate_table.setIndentation(_BATCH_TABLE_INDENTATION)
        self._candidate_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._candidate_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._candidate_table.setUniformRowHeights(True)
        self._candidate_table.itemSelectionChanged.connect(self._refresh_actions)
        use_square_top_table_corners(self._candidate_table)
        candidate_layout.addWidget(self._candidate_table, 1)
        candidate_actions_row = QHBoxLayout()
        candidate_actions_row.setContentsMargins(0, 0, 0, 0)
        candidate_actions_row.setSpacing(8)
        self._include_isosurfaces = QCheckBox("Include isosurface at total energy", candidate_group)
        self._include_isosurfaces.setStyleSheet("QCheckBox { background: transparent; }")
        candidate_actions_row.addWidget(self._include_isosurfaces, 0, Qt.AlignmentFlag.AlignLeft)
        candidate_actions_row.addStretch(1)
        self._add_selected_button = QPushButton("Add Selected Muons", candidate_group)
        self._add_selected_button.clicked.connect(self._emit_add_candidates)
        candidate_actions_row.addWidget(self._add_selected_button, 0, Qt.AlignmentFlag.AlignRight)
        candidate_layout.addLayout(candidate_actions_row)
        tables_row.addWidget(candidate_group, 1)

        current_group = QGroupBox("Muons In Viewer", self)
        current_layout = QVBoxLayout(current_group)
        current_layout.setContentsMargins(12, 12, 12, 12)
        current_layout.setSpacing(8)
        self._cell_note = QLabel(
            "Start rows are in the input structure's cell, Custom rows in the job's cell.",
            current_group,
        )
        self._cell_note.setWordWrap(True)
        style_hint_text(self._cell_note)
        current_layout.addWidget(self._cell_note)
        self._current_table = QTreeWidget(current_group)
        self._current_table.setColumnCount(len(_VIEWER_MUON_CURRENT_HEADERS))
        self._current_table.setHeaderLabels(_VIEWER_MUON_CURRENT_HEADERS)
        self._current_table.setRootIsDecorated(False)
        self._current_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._current_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._current_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self._current_table.setUniformRowHeights(True)
        self._current_table.itemChanged.connect(self._handle_current_item_changed)
        self._current_table.itemSelectionChanged.connect(self._emit_current_selection_changed)
        self._current_table.itemSelectionChanged.connect(self._refresh_actions)
        use_square_top_table_corners(self._current_table)
        current_layout.addWidget(self._current_table, 1)
        self._remove_selected_button = QPushButton("Remove Selected Muons", current_group)
        self._remove_selected_button.clicked.connect(self._emit_remove_selected)
        self._current_table_delete_shortcuts = workflow_table_utils.bind_delete_shortcuts(
            self._current_table, self._emit_remove_selected
        )
        current_layout.addWidget(self._remove_selected_button, 0, Qt.AlignmentFlag.AlignRight)
        tables_row.addWidget(current_group, 1)

        layout.addLayout(tables_row, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        for table in (self._candidate_table, self._current_table):
            header = table.header()
            header.setMinimumSectionSize(24)
            header.setDefaultAlignment(
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            )
            header.setFixedHeight(
                max(header.sizeHint().height(), (self.fontMetrics().height() * 2) + 16)
            )
            for column in range(table.columnCount()):
                header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
            if table is self._candidate_table:
                header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
                header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)
        self._refresh_actions()

    def set_candidate_entries(self, entries: list[_BatchMuonCandidate]) -> None:
        self._updating_tables = True
        candidate_previous = self._candidate_table.blockSignals(True)
        self._candidate_entries = {entry.key: entry for entry in entries}
        self._candidate_table.clear()
        cluster_items: dict[int, QTreeWidgetItem] = {}
        for entry in entries:
            cluster_item = cluster_items.get(entry.cluster_index)
            if cluster_item is None:
                cluster_item = QTreeWidgetItem(
                    [
                        entry.cluster_label,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                cluster_item.setFlags(cluster_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                cluster_item.setExpanded(True)
                self._candidate_table.addTopLevelItem(cluster_item)
                cluster_items[entry.cluster_index] = cluster_item
            item = QTreeWidgetItem(
                [
                    entry.label,
                    f"{entry.display_fractional_position[0]:.6f}",
                    f"{entry.display_fractional_position[1]:.6f}",
                    f"{entry.display_fractional_position[2]:.6f}",
                    "" if entry.classical_energy_ev is None else f"{entry.classical_energy_ev:.6f}",
                    ""
                    if entry.zero_point_energy_ev is None
                    else f"{entry.zero_point_energy_ev:.6f}",
                    "" if entry.total_energy_ev is None else f"{entry.total_energy_ev:.6f}",
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, entry.key)
            cluster_item.addChild(item)
        self._candidate_table.blockSignals(candidate_previous)
        self._updating_tables = False
        self._refresh_actions()

    def set_current_entries(
        self, entries: list[_ManagedViewerMuonState], *, selected_keys: list[str] | None = None
    ) -> None:
        self._updating_tables = True
        current_previous = self._current_table.blockSignals(True)
        self._current_entries = {entry.key: entry for entry in entries}
        self._current_table.clear()
        for entry in entries:
            shown = entry.display_fractional_position or entry.fractional_position
            item = QTreeWidgetItem(
                [
                    entry.label,
                    entry.source_label,
                    f"{shown[0]:.6f}",
                    f"{shown[1]:.6f}",
                    f"{shown[2]:.6f}",
                    "" if entry.classical_energy_ev is None else f"{entry.classical_energy_ev:.6f}",
                    ""
                    if entry.zero_point_energy_ev is None
                    else f"{entry.zero_point_energy_ev:.6f}",
                    "" if entry.total_energy_ev is None else f"{entry.total_energy_ev:.6f}",
                    "Yes" if entry.linked_isosurface_layer_key else "",
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, entry.key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._current_table.addTopLevelItem(item)
        self._current_table.resizeColumnToContents(1)
        self.set_current_selection(selected_keys or self._current_selection_keys)
        self._current_table.blockSignals(current_previous)
        self._updating_tables = False
        self._reset_current_table_horizontal_scroll()
        self._refresh_actions()

    def set_current_selection(self, keys: list[str]) -> None:
        normalized = [key for key in keys if key in self._current_entries]
        self._current_selection_keys = list(normalized)
        previous = self._current_table.blockSignals(True)
        self._current_table.clearSelection()
        first_selected: QTreeWidgetItem | None = None
        for index in range(self._current_table.topLevelItemCount()):
            item = self._current_table.topLevelItem(index)
            if item is None:
                continue
            key = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(key, str) and key in normalized:
                item.setSelected(True)
                if first_selected is None:
                    first_selected = item
        if first_selected is not None:
            self._current_table.setCurrentItem(first_selected)
        self._current_table.blockSignals(previous)

    def set_include_isosurfaces_available(self, available: bool) -> None:
        self._include_isosurfaces_available = bool(available)
        self._refresh_actions()

    def _selected_candidate_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for item in self._candidate_table.selectedItems():
            key = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(key, str) and key and key not in seen:
                keys.append(key)
                seen.add(key)
        return keys

    def _selected_current_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for item in self._current_table.selectedItems():
            key = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(key, str) and key and key not in seen:
                keys.append(key)
                seen.add(key)
        return keys

    def _refresh_actions(self) -> None:
        if self._updating_tables:
            return
        selected_candidate_keys = self._selected_candidate_keys()
        self._add_selected_button.setEnabled(bool(selected_candidate_keys))
        self._remove_selected_button.setEnabled(bool(self._selected_current_keys()))
        selected = [
            self._candidate_entries[key]
            for key in selected_candidate_keys
            if key in self._candidate_entries
        ]
        eligible = bool(selected) and all(
            entry.zero_point_energy_ev is not None and entry.total_energy_ev is not None
            for entry in selected
        )
        enabled = self._include_isosurfaces_available and eligible
        if not enabled:
            previous = self._include_isosurfaces.blockSignals(True)
            self._include_isosurfaces.setChecked(False)
            self._include_isosurfaces.blockSignals(previous)
        self._include_isosurfaces.setEnabled(enabled)

    def _emit_add_candidates(self) -> None:
        self.add_candidates_requested.emit(
            self._selected_candidate_keys(),
            self._include_isosurfaces.isChecked(),
        )

    def _emit_add_custom(self) -> None:
        self.add_custom_requested.emit(
            tuple(float(value) for value in self._custom_fractional_input.values())
        )

    def _emit_remove_selected(self) -> None:
        """Remove the selected muons from the button or a Delete key, once confirmed."""

        keys = self._selected_current_keys()
        if not keys:
            return
        linked = sum(
            1 for key in keys if self._current_entries[key].linked_isosurface_layer_key is not None
        )
        detail = (
            f"This also removes {plural(linked, 'isosurface')} drawn at their energies."
            if linked
            else "The muon overlays leave the viewer."
        )
        if not workflow_table_utils.confirm_destructive(
            self,
            "Remove muons",
            f"Remove {plural(len(keys), 'muon')} from the viewer?",
            detail,
            "Remove muons",
        ):
            return
        self.remove_requested.emit(keys)

    def _emit_current_selection_changed(self) -> None:
        if self._updating_tables:
            return
        self._current_selection_keys = self._selected_current_keys()
        self.current_selection_changed.emit(list(self._current_selection_keys))

    def _handle_current_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._updating_tables or int(column) != 0:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(key, str) or key not in self._current_entries:
            return
        previous = self._current_entries[key].label
        updated = item.text(0).strip()
        if not updated:
            updated = previous
        if updated == previous:
            if updated != item.text(0):
                was_updating = self._updating_tables
                self._updating_tables = True
                item.setText(0, updated)
                self._updating_tables = was_updating
            return
        self.rename_requested.emit(key, updated)

    def _reset_current_table_horizontal_scroll(self) -> None:
        horizontal_scroll = self._current_table.horizontalScrollBar()
        if horizontal_scroll is not None:
            horizontal_scroll.setValue(horizontal_scroll.minimum())
        header = self._current_table.header()
        if header is not None:
            header.setOffset(0)
        QTimer.singleShot(0, self._reset_current_table_horizontal_scroll_deferred)

    def _reset_current_table_horizontal_scroll_deferred(self) -> None:
        horizontal_scroll = self._current_table.horizontalScrollBar()
        if horizontal_scroll is not None:
            horizontal_scroll.setValue(horizontal_scroll.minimum())
        header = self._current_table.header()
        if header is not None:
            header.setOffset(0)


class _IsosurfaceGridThread(QThread):
    """Background thread that computes the full-cell UEP potential grid."""

    grid_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, field: UepField) -> None:
        super().__init__()
        self._field = field

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            grid = potential_grid(self._field)
            if self.isInterruptionRequested():
                return
            self.grid_ready.emit(grid)
        except Exception as exc:  # noqa: BLE001
            if self.isInterruptionRequested():
                return
            self.failed.emit(str(exc))


class _IsosurfaceMeshThread(QThread):
    """Background thread that extracts one isosurface mesh from a cached grid."""

    mesh_ready = Signal(int, int, object)
    failed = Signal(int, int, str)

    def __init__(
        self, grid: UepPotentialGrid, *, level_ev: float, request_id: int, layer_index: int
    ) -> None:
        super().__init__()
        self._grid = grid
        self._level_ev = float(level_ev)
        self._request_id = int(request_id)
        self._layer_index = int(layer_index)

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            mesh = extract_isosurface(self._grid, level_ev=self._level_ev)
            if self.isInterruptionRequested():
                return
            self.mesh_ready.emit(self._request_id, self._layer_index, mesh)
        except Exception as exc:  # noqa: BLE001
            if self.isInterruptionRequested():
                return
            self.failed.emit(self._request_id, self._layer_index, str(exc))


class _BatchSearchThread(QThread):
    """Background thread that generates and optimizes one batch-search run."""

    prepared = Signal(int, str)
    progress_changed = Signal(int, int)
    batch_ready = Signal(object, object, str)
    failed = Signal(str)

    def __init__(
        self,
        field: UepField,
        *,
        lattice_vectors: tuple[
            tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
        ],
        host_positions_frac: tuple[tuple[float, float, float], ...],
        requested_starts: int,
        min_muon_muon: float,
        min_muon_atom: float,
        max_attempts: int,
        force_tolerance_ev_per_ang: float,
        max_steps: int,
        grouping_structure: object,
    ) -> None:
        super().__init__()
        self._field = field
        self._grouping_structure = grouping_structure
        self._lattice_vectors = lattice_vectors
        self._host_positions_frac = host_positions_frac
        self._requested_starts = int(requested_starts)
        self._min_muon_muon = float(min_muon_muon)
        self._min_muon_atom = float(min_muon_atom)
        self._max_attempts = int(max_attempts)
        self._force_tolerance_ev_per_ang = float(force_tolerance_ev_per_ang)
        self._max_steps = int(max_steps)

    def run(self) -> None:
        try:
            if self.isInterruptionRequested():
                return
            starts, attempts_used = generate_random_sites(
                lattice_vectors=self._lattice_vectors,
                host_positions_frac=self._host_positions_frac,
                count=self._requested_starts,
                min_muon_muon=self._min_muon_muon,
                min_muon_atom=self._min_muon_atom,
                max_attempts=self._max_attempts,
                allow_partial=True,
            )
            generation_lines = [
                f"Requested random starts: {self._requested_starts}",
                f"Initial starts generated: {len(starts)}",
                f"Gaussian width factor: {self._field.source.gw_factor:g}",
                f"Min muon-muon distance (Å): {self._min_muon_muon:.3f}",
                f"Min muon-atom distance (Å): {self._min_muon_atom:.3f}",
                f"Max attempts: {self._max_attempts}",
                f"Force tolerance (eV/Å): {self._force_tolerance_ev_per_ang:.4g}",
                f"Optimizer max steps: {self._max_steps}",
            ]
            if len(starts) < self._requested_starts:
                generation_lines.append(
                    "Initial-start count was reduced because the muon-muon distance constraint limited packing within the allowed attempts."
                )
            generation_lines.append(f"Generation attempts used: {attempts_used}")
            if self.isInterruptionRequested():
                return
            self.prepared.emit(len(starts), "\n".join(generation_lines + ["Optimizing starts…"]))

            results: list[UepOptimizationResult] = []
            total = len(starts)
            for index, point in enumerate(starts, start=1):
                if self.isInterruptionRequested():
                    return
                result = optimize_muon_position(
                    self._field,
                    start_position_ang=tuple(point.cartesian),
                    force_tolerance_ev_per_ang=self._force_tolerance_ev_per_ang,
                    max_steps=self._max_steps,
                )
                if self.isInterruptionRequested():
                    return
                results.append(result)
                self.progress_changed.emit(index, total)

            if self.isInterruptionRequested():
                return
            try:
                distances = measure_uep_minima(results, source_structure=self._grouping_structure)
            except Exception:  # noqa: BLE001 - the window reports it; the results are kept.
                distances = None
            if self.isInterruptionRequested():
                return
            self.batch_ready.emit(
                results, distances, "\n".join([*generation_lines, uep_search_outcome(results)])
            )
        except Exception as exc:  # noqa: BLE001
            if self.isInterruptionRequested():
                return
            self.failed.emit(str(exc))


class UepAnalysisDialog(QDialog):
    """Wizard-like dialog for UEP probing, plotting, batch search, and site grouping."""

    state_changed = Signal(dict)

    def __init__(
        self,
        *,
        source_job_label: str,
        source_structure_id,
        source_structure: Structure | None = None,
        den_fmt_path: str | Path,
        castep_path: str | Path,
        cell_path: str | Path | None = None,
        initial_state: Mapping[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"UEP Analysis - {source_job_label}")
        self.resize(clamped_size(self, 1180, 920))
        self.setMinimumSize(980, 640)

        self._source_job_label = source_job_label
        self._source_structure_id = source_structure_id
        self._source_structure = source_structure
        self._den_fmt_path = Path(den_fmt_path)
        self._castep_path = Path(castep_path)
        self._cell_path = Path(cell_path) if cell_path is not None else None
        initial_gw_factor = self._state_gw_factor(initial_state)
        self._source: UepSourceData = load_uep_source(
            den_fmt_path=self._den_fmt_path,
            castep_path=self._castep_path,
            cell_path=self._cell_path,
            gw_factor=initial_gw_factor,
        )
        self._field = UepField(self._source)
        self._latest_probe: UepProbeResult | None = None
        self._latest_line_sample = None
        self._latest_plane_sample = None
        self._latest_batch_results: list[UepOptimizationResult] = []
        self._latest_batch_summary: UepBatchSummary | None = None
        self._batch_item_by_key: dict[str, QTreeWidgetItem] = {}
        self._batch_minima_distances: UepMinimaDistances | None = None
        self._shutting_down = False
        #: The grid on its own scale, whose zero is the cell average.
        self._potential_grid_raw_cache: UepPotentialGrid | None = None
        #: The same grid above its minimum: what the levels and the saturation read.
        self._potential_grid_cache: UepPotentialGrid | None = None
        self._potential_grid_live_preview_cache: UepPotentialGrid | None = None
        self._potential_minimum_ev = 0.0
        self._potential_level_ceiling_ev = 0.0
        self._potential_level_maximum_ev = 0.0
        #: Levels and saturation restored on the potential's own scale, from a window saved
        #: without ``Set minimum potential to zero``. They move above the minimum with the grid.
        self._saved_levels_on_own_scale = False
        self._viewer_host_structure = self._build_viewer_host_structure()
        self._isosurface_layers: list[_IsosurfaceLayerState] = [
            _IsosurfaceLayerState(label="Isosurface 1", key=self._new_isosurface_layer_key())
        ]
        self._active_isosurface_index = 0
        self._slice_planes: list[_SlicePlaneState] = []
        self._viewer_muons: list[_ManagedViewerMuonState] = []
        self._selected_viewer_muon_keys: list[str] = []
        self._custom_muon_counter = 0
        self._viewer_structure = self._viewer_host_structure
        #: What the viewer's structure was built from; ``None`` when it needs building again.
        self._viewer_structure_signature: tuple[object, ...] | None = ()
        self._slice_saturation_bounds: tuple[float, float] | None = None
        self._gw_factor_commit_requested = False
        self._restoring_state = False
        self._isosurface_grid_thread: _IsosurfaceGridThread | None = None
        self._isosurface_mesh_thread: _IsosurfaceMeshThread | None = None
        self._batch_search_thread: _BatchSearchThread | None = None
        self._isosurface_request_id = 0
        self._pending_isosurface_level_ev: float | None = None
        self._pending_isosurface_layer_index: int | None = None
        self._pending_isosurface_rebuilds: list[tuple[str, float]] = []
        #: The layer the running mesh is for, which removing another layer renumbers.
        self._mesh_request_layer_key: str | None = None
        self._pending_slice_preview_bounds: tuple[float, float] | None = None
        self._viewer_muon_dialog: _ManageViewerMuonsDialog | None = None
        self._isosurface_preview_timer = QTimer(self)
        self._isosurface_preview_timer.setSingleShot(True)
        self._isosurface_preview_timer.setInterval(_ISOSURFACE_SLIDER_DEBOUNCE_MS)
        self._isosurface_preview_timer.timeout.connect(self._dispatch_pending_isosurface_request)
        self._slice_preview_timer = QTimer(self)
        self._slice_preview_timer.setSingleShot(True)
        self._slice_preview_timer.setInterval(_ISOSURFACE_SLIDER_DEBOUNCE_MS)
        self._slice_preview_timer.timeout.connect(self._dispatch_pending_slice_preview)
        self._grouping_settled_timer = QTimer(self)
        self._grouping_settled_timer.setSingleShot(True)
        self._grouping_settled_timer.setInterval(_GROUPING_SETTLED_MS)
        self._grouping_settled_timer.timeout.connect(self._emit_state_changed)

        palette = current_palette()
        self._line_figure = Figure(
            figsize=(6.0, 4.0), facecolor=palette.bg_surface, constrained_layout=True
        )
        self._line_canvas = FigureCanvasQTAgg(self._line_figure)
        self._plane_figure = Figure(
            figsize=(6.0, 4.0), facecolor=palette.bg_surface, constrained_layout=True
        )
        self._plane_canvas = FigureCanvasQTAgg(self._plane_figure)
        self._slice_colorbar_figure, self._slice_colorbar_canvas = create_plot_canvas(
            figsize=(3.0, 0.8), facecolor=palette.bg_surface, minimum_size=(200, 72)
        )
        self._slice_colorbar_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._slice_colorbar_canvas.setFixedHeight(84)

        gw_factor_row = self._build_gw_factor_row()
        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("uepAnalysisTabs")
        self._summary_tab = self._build_summary_tab()
        self._probe_tab = self._build_probe_tab()
        self._line_tab = self._build_line_tab()
        self._plane_tab = self._build_plane_tab()
        self._potential_tab = self._build_potential_tab()
        self._batch_tab = self._build_batch_tab()
        self._tabs.addTab(self._batch_tab, "Batch Search")
        self._tabs.addTab(self._potential_tab, "Potential")
        # The summary, probe, line and plane pages are no longer tabs; their
        # widgets only carry saved analysis state. Left visible they sat
        # outside any layout at the default 100x30 geometry, stacked above the
        # tab bar and covering the first tab.
        for retired_page in (self._summary_tab, self._probe_tab, self._line_tab, self._plane_tab):
            retired_page.hide()
        self._tabs.currentChanged.connect(self._on_analysis_tab_changed)
        self._apply_tab_branding()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(gw_factor_row)
        layout.addWidget(self._tabs, 1)

        self._populate_summary()
        self._restore_initial_state(initial_state)
        self._disable_default_dialog_buttons()
        # The width factor is first in the layout, but a window should not open editing it.
        self._tabs.setFocus()

    @staticmethod
    def _state_gw_factor(state: Mapping[str, object] | None) -> float:
        if isinstance(state, Mapping):
            raw_value = state.get("gw_factor")
            try:
                if raw_value is not None and float(raw_value) > 0.0:
                    return float(raw_value)
            except (TypeError, ValueError):
                pass
        return 5.0

    def _apply_tab_branding(self) -> None:
        palette = current_palette()
        self._tabs.setStyleSheet(
            f"""
            QTabWidget#uepAnalysisTabs::pane {{
                border: 1px solid {palette.border};
                border-radius: 14px;
                top: -1px;
                background: {palette.bg_surface};
                margin-top: 0px;
            }}
            QTabWidget#uepAnalysisTabs QTabBar::tab {{
                background: {palette.bg_surface_alt};
                color: {palette.text_muted};
                border: 1px solid {palette.border};
                border-bottom: none;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                padding: 10px 18px;
                margin-right: 6px;
                min-width: 148px;
                font-family: "{body_font_family()}";
                font-weight: 700;
            }}
            QTabWidget#uepAnalysisTabs QTabBar::tab:selected {{
                background: {palette.bg_surface};
                color: {palette.text_primary};
                border-color: {palette.accent};
            }}
            QTabWidget#uepAnalysisTabs QTabBar::tab:hover:!selected {{
                background: {palette.bg_surface};
                color: {palette.text_primary};
            }}
            """
        )

    def _disable_default_dialog_buttons(self) -> None:
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
            button.setDefault(False)

    def _apply_batch_progress_style(self, *, completed: bool) -> None:
        set_progress_bar_state(self._batch_progress, "complete" if completed else "active")

    @staticmethod
    def _normalized_visible_tab_index(raw_index: object) -> int:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return 0
        return _SAVED_TAB_TO_VISIBLE.get(index, 0)

    def _build_summary_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        source_group = QGroupBox("Source Validation", tab)
        source_layout = QVBoxLayout(source_group)
        self._summary_text = QTextEdit(source_group)
        self._summary_text.setReadOnly(True)
        source_layout.addWidget(self._summary_text)

        theory_group = QGroupBox("Theory and Caveats", tab)
        theory_layout = QVBoxLayout(theory_group)
        self._theory_text = QTextEdit(theory_group)
        self._theory_text.setReadOnly(True)
        self._theory_text.setPlainText(_THEORY_TEXT)
        theory_layout.addWidget(self._theory_text)

        layout.addWidget(source_group)
        layout.addWidget(theory_group)
        return tab

    def _build_probe_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Point Probe (fractional coordinates)", tab)
        form = QFormLayout(controls)
        self._probe_point = _TripleInput(parent=controls)
        self._probe_point.set_values((0.25, 0.25, 0.25))
        self._probe_run = QPushButton("Evaluate Point", controls)
        self._probe_run.clicked.connect(self._run_probe)
        actions = QHBoxLayout()
        actions.addWidget(self._probe_run)
        actions.addStretch(1)
        form.addRow("Fractional position", self._probe_point)
        form.addRow("", actions)

        self._probe_output = QTextEdit(tab)
        self._probe_output.setReadOnly(True)

        layout.addWidget(controls)
        layout.addWidget(self._probe_output, 1)
        return tab

    def _build_line_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Line Sampling (fractional coordinates)", tab)
        form = QFormLayout(controls)
        self._line_start = _TripleInput(parent=controls)
        self._line_end = _TripleInput(parent=controls)
        self._line_start.set_values((0.0, 0.0, 0.0))
        self._line_end.set_values((0.5, 0.5, 0.5))
        self._line_points = QSpinBox(controls)
        self._line_points.setRange(2, 500)
        self._line_points.setValue(101)
        self._line_run = QPushButton("Sample Line", controls)
        self._line_run.clicked.connect(self._run_line_sample)
        form.addRow("Start", self._line_start)
        form.addRow("End", self._line_end)
        form.addRow("Points", self._line_points)
        form.addRow("", self._line_run)

        self._line_data = QTextEdit(tab)
        self._line_data.setReadOnly(True)
        layout.addWidget(controls)
        layout.addWidget(self._line_canvas, 1)
        layout.addWidget(self._line_data, 1)
        return tab

    def _build_plane_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Plane Sampling (fractional coordinates)", tab)
        form = QFormLayout(controls)
        self._plane_origin = _TripleInput(parent=controls)
        self._plane_x = _TripleInput(parent=controls)
        self._plane_y = _TripleInput(parent=controls)
        self._plane_origin.set_values((0.0, 0.0, 0.0))
        self._plane_x.set_values((0.5, 0.0, 0.0))
        self._plane_y.set_values((0.0, 0.5, 0.0))
        self._plane_points_x = QSpinBox(controls)
        self._plane_points_x.setRange(2, 200)
        self._plane_points_x.setValue(61)
        self._plane_points_y = QSpinBox(controls)
        self._plane_points_y.setRange(2, 200)
        self._plane_points_y.setValue(61)
        self._plane_run = QPushButton("Sample Plane", controls)
        self._plane_run.clicked.connect(self._run_plane_sample)
        form.addRow("Origin", self._plane_origin)
        form.addRow("X corner", self._plane_x)
        form.addRow("Y corner", self._plane_y)
        form.addRow("Points along X", self._plane_points_x)
        form.addRow("Points along Y", self._plane_points_y)
        form.addRow("", self._plane_run)

        self._plane_data = QTextEdit(tab)
        self._plane_data.setReadOnly(True)
        layout.addWidget(controls)
        layout.addWidget(self._plane_canvas, 1)
        layout.addWidget(self._plane_data, 1)
        return tab

    def _build_gw_factor_row(self) -> QWidget:
        """The one Gaussian width factor, above the tabs: the search and the potential use it."""

        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(page_margins())
        layout.setSpacing(form_spacing())
        label = QLabel("Gaussian width factor", row)
        label.setToolTip(_GW_FACTOR_TOOLTIP)
        self._gw_factor_input = DecimalLineEdit(row, decimals=4)
        self._gw_factor_input.setMaximumWidth(124)
        self._gw_factor_input.setToolTip(_GW_FACTOR_TOOLTIP)
        self._gw_factor_input.setAccessibleName("Gaussian width factor")
        self._gw_factor_input.returnPressed.connect(self._on_gw_factor_return_pressed)
        self._gw_factor_input.editingFinished.connect(self._on_gw_factor_editing_finished)
        self._gw_factor_input.textChanged.connect(self._refresh_gw_factor_warning)
        label.setBuddy(self._gw_factor_input)
        self._gw_factor_warning = QLabel(row)
        style_warning_text(self._gw_factor_warning)
        self._gw_factor_warning.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(self._gw_factor_input)
        layout.addWidget(self._gw_factor_warning, 100)
        layout.addStretch(1)
        self._show_active_gw_factor()
        return row

    def _build_potential_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self._potential_viewer = AtomicStructureViewer(tab)
        self._potential_viewer.setMinimumSize(320, 320)
        self._potential_viewer.set_structure(self._viewer_host_structure)
        layout.addWidget(self._potential_viewer, 1)

        controls = QWidget(tab)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(form_spacing())
        controls_layout.addWidget(self._build_isosurface_card(controls))
        controls_layout.addWidget(self._build_slice_plane_card(controls))
        controls_layout.addWidget(self._build_slice_saturation_card(controls))

        self._potential_status = QLabel(controls)
        self._potential_status.setObjectName("uepIsosurfaceStatusLabel")
        self._potential_status.setWordWrap(True)
        self._potential_status.hide()
        controls_layout.addWidget(self._potential_status)

        self._isosurface_export_button = QPushButton("Export XSF…", controls)
        self._isosurface_export_button.setEnabled(False)
        self._isosurface_export_button.setToolTip(
            "Exports the potential on its own scale, whose zero is the cell average."
        )
        self._isosurface_export_button.clicked.connect(self._export_potential_grid_xsf)
        controls_layout.addWidget(self._isosurface_export_button, 0, Qt.AlignmentFlag.AlignLeft)

        self._slice_hide_atoms = QCheckBox("Hide atoms in front of planes", controls)
        self._slice_hide_atoms.setStyleSheet("QCheckBox { background: transparent; }")
        self._slice_hide_atoms.toggled.connect(self._on_slice_hide_atoms_toggled)
        controls_layout.addWidget(self._slice_hide_atoms)

        controls_layout.addWidget(self._build_viewer_muon_card(controls))
        controls_layout.addStretch(1)

        self._potential_controls_scroll = controls_scroll_area(tab)
        self._potential_controls_scroll.setWidget(controls)
        self._potential_controls_scroll.setFixedWidth(_POTENTIAL_CONTROLS_WIDTH)
        layout.addWidget(self._potential_controls_scroll)

        self._refresh_slice_plane_list()
        self._refresh_slice_colorbar()
        self._refresh_viewer_muon_legend()
        self._refresh_gw_factor_controls()
        return tab

    def _build_isosurface_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("analysisSliderPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(card_margins())
        layout.setSpacing(form_spacing())

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        title = QLabel("Isosurface level", card)
        title.setObjectName("uepIsosurfacePanelTitle")
        title.setToolTip(_POTENTIAL_LEVEL_TOOLTIP)
        header_row.addWidget(title)
        header_row.addStretch(1)

        self._isosurface_layer_tabs = _IsosurfaceLayerTabBar(card)
        self._isosurface_layer_tabs.setMinimumHeight(28)
        self._isosurface_layer_tabs.setAccessibleName("Isosurfaces")
        self._isosurface_layer_tabs.currentChanged.connect(self._on_isosurface_tab_changed)
        header_row.addWidget(self._isosurface_layer_tabs, 0, Qt.AlignmentFlag.AlignRight)

        self._isosurface_delete_button = QPushButton("−", card)
        self._isosurface_delete_button.setObjectName("uepIsosurfaceLayerButton")
        self._isosurface_delete_button.setAccessibleName("Remove isosurface")
        self._isosurface_delete_button.setToolTip("Remove this isosurface.")
        self._isosurface_delete_button.setFixedSize(26, 26)
        self._isosurface_delete_button.clicked.connect(self._delete_current_isosurface_layer)
        header_row.addWidget(self._isosurface_delete_button, 0, Qt.AlignmentFlag.AlignRight)

        self._isosurface_add_button = QPushButton("+", card)
        self._isosurface_add_button.setObjectName("uepIsosurfaceLayerButton")
        self._isosurface_add_button.setAccessibleName("Add isosurface")
        self._isosurface_add_button.setToolTip("Add an isosurface.")
        self._isosurface_add_button.setFixedSize(26, 26)
        self._isosurface_add_button.clicked.connect(self._add_isosurface_layer)
        header_row.addWidget(self._isosurface_add_button, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header_row)

        layer_button_style = (
            "QPushButton#uepIsosurfaceLayerButton {"
            "border: 1px solid rgba(0, 0, 0, 0.14);"
            f"background: {current_palette().bg_surface_alt};"
            f"color: {current_palette().text_primary};"
            "border-radius: 7px;"
            "font-size: 15px;"
            "font-weight: 700;"
            "padding: 0px;"
            "}"
            "QPushButton#uepIsosurfaceLayerButton:hover:!disabled {"
            f"border-color: {current_palette().accent};"
            "}"
            "QPushButton#uepIsosurfaceLayerButton:pressed:!disabled {"
            f"background: {current_palette().bg_surface};"
            "}"
            "QPushButton#uepIsosurfaceLayerButton:disabled {"
            f"color: {current_palette().text_muted};"
            f"background: {current_palette().bg_surface};"
            "}"
        )
        self._isosurface_add_button.setStyleSheet(layer_button_style)
        self._isosurface_delete_button.setStyleSheet(layer_button_style)

        self._isosurface_slider = QSlider(Qt.Orientation.Horizontal, card)
        self._isosurface_slider.setObjectName("uepIsosurfaceSlider")
        self._isosurface_slider.setAccessibleName("Isosurface level")
        self._isosurface_slider.setToolTip(_POTENTIAL_LEVEL_TOOLTIP)
        self._isosurface_slider.setRange(0, 10000)
        self._isosurface_slider.setValue(5000)
        self._isosurface_slider.setMinimumHeight(36)
        self._isosurface_slider.setSingleStep(10)
        self._isosurface_slider.setPageStep(250)
        self._isosurface_slider.setEnabled(False)
        self._isosurface_slider.valueChanged.connect(self._sync_isosurface_level_editor_from_slider)
        self._isosurface_slider.valueChanged.connect(self._schedule_isosurface_from_slider)
        self._isosurface_slider.sliderReleased.connect(self._request_isosurface_from_slider)
        layout.addWidget(self._isosurface_slider)

        self._isosurface_min_label = QLabel("", card)
        self._isosurface_min_label.setObjectName("uepIsosurfaceBound")
        self._isosurface_max_label = QLabel("", card)
        self._isosurface_max_label.setObjectName("uepIsosurfaceBound")
        bounds_row = QHBoxLayout()
        bounds_row.setContentsMargins(0, 0, 0, 0)
        bounds_row.addWidget(self._isosurface_min_label)
        bounds_row.addStretch(1)
        bounds_row.addWidget(self._isosurface_max_label)
        layout.addLayout(bounds_row)

        self._isosurface_level_input = DecimalLineEdit(card)
        self._isosurface_level_input.setRange(0.0, 1_000_000.0)
        self._isosurface_level_input.setDecimals(6)
        self._isosurface_level_input.setEnabled(False)
        self._isosurface_level_input.setToolTip(_POTENTIAL_LEVEL_TOOLTIP)
        self._isosurface_level_input.setAccessibleName("Isosurface level above the minimum in eV")
        self._isosurface_level_input.editingFinished.connect(self._request_isosurface_from_editor)
        self._isosurface_color_button = QPushButton(card)
        self._isosurface_color_button.clicked.connect(self._choose_isosurface_color)
        self._isosurface_color_button.setToolTip("Choose the isosurface color.")
        self._isosurface_color_button.setAccessibleName("Isosurface color")
        self._isosurface_color_button.setFixedSize(52, 32)
        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(8)
        level_label = QLabel("eV above minimum", card)
        level_label.setObjectName("uepIsosurfaceMinorLabel")
        level_label.setToolTip(_POTENTIAL_LEVEL_TOOLTIP)
        value_row.addWidget(self._isosurface_level_input, 1)
        value_row.addWidget(level_label)
        value_row.addWidget(self._isosurface_color_button)
        layout.addLayout(value_row)

        self._isosurface_legend_entries = QWidget(card)
        self._isosurface_legend_entries.setStyleSheet("background: transparent;")
        self._isosurface_legend_entries_layout = QVBoxLayout(self._isosurface_legend_entries)
        self._isosurface_legend_entries_layout.setContentsMargins(0, 0, 0, 0)
        self._isosurface_legend_entries_layout.setSpacing(5)
        layout.addWidget(self._isosurface_legend_entries)

        self._update_isosurface_color_button()
        self._refresh_isosurface_layer_tabs()
        return card

    def _build_slice_plane_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("analysisSliderPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(card_margins())
        layout.setSpacing(form_spacing())

        title = QLabel("Slice planes", card)
        title.setObjectName("uepIsosurfacePanelTitle")
        layout.addWidget(title)

        self._slice_plane_list = QListWidget(card)
        self._slice_plane_list.setAccessibleName("Slice planes")
        self._slice_plane_list.setMaximumHeight(self.fontMetrics().height() * 5)
        self._slice_plane_list.itemSelectionChanged.connect(self._refresh_slice_plane_actions)
        self._slice_plane_list.itemDoubleClicked.connect(
            lambda _item: self._edit_selected_slice_plane()
        )
        self._slice_plane_delete_shortcuts = workflow_table_utils.bind_delete_shortcuts(
            self._slice_plane_list, self._remove_selected_slice_plane
        )
        layout.addWidget(self._slice_plane_list)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        self._slice_add_plane_button = QPushButton("Add plane…", card)
        self._slice_add_plane_button.clicked.connect(self._open_add_slice_plane_dialog)
        self._slice_edit_plane_button = QPushButton("Edit…", card)
        self._slice_edit_plane_button.clicked.connect(self._edit_selected_slice_plane)
        self._slice_remove_plane_button = QPushButton("Remove", card)
        self._slice_remove_plane_button.clicked.connect(self._remove_selected_slice_plane)
        actions.addWidget(self._slice_add_plane_button)
        actions.addWidget(self._slice_edit_plane_button)
        actions.addWidget(self._slice_remove_plane_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        return card

    def _build_slice_saturation_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("analysisSliderPanel")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(card_margins())
        layout.setSpacing(form_spacing())

        title = QLabel("Slice saturation", card)
        title.setObjectName("uepIsosurfacePanelTitle")
        title.setToolTip(
            "The colour window of every slice plane, in eV above the minimum of the potential grid. "
            "Only the colours change."
        )
        layout.addWidget(title)

        self._slice_range_slider = _RangeSlider(card)
        self._slice_range_slider.setRange(0, 10000)
        self._slice_range_slider.setEnabled(False)
        self._slice_range_slider.setAccessibleName("Slice saturation")
        self._slice_range_slider.valuesChanged.connect(self._on_slice_slider_changed)
        self._slice_range_slider.interactionFinished.connect(self._on_slice_slider_released)
        layout.addWidget(self._slice_range_slider)

        value_row = QHBoxLayout()
        value_row.setContentsMargins(0, 0, 0, 0)
        value_row.setSpacing(8)
        self._slice_min_input = DecimalLineEdit(card)
        self._slice_max_input = DecimalLineEdit(card)
        self._slice_min_input.setAccessibleName("Slice saturation minimum above the minimum in eV")
        self._slice_max_input.setAccessibleName("Slice saturation maximum above the minimum in eV")
        for widget in (self._slice_min_input, self._slice_max_input):
            widget.setRange(0.0, 1_000_000.0)
            widget.setDecimals(6)
            widget.setEnabled(False)
            widget.setToolTip(_POTENTIAL_LEVEL_TOOLTIP)
            widget.editingFinished.connect(self._request_slice_range_from_editor)
        to_label = QLabel("to", card)
        to_label.setObjectName("uepIsosurfaceMinorLabel")
        unit_label = QLabel("eV", card)
        unit_label.setObjectName("uepIsosurfaceMinorLabel")
        value_row.addWidget(self._slice_min_input, 1)
        value_row.addWidget(to_label)
        value_row.addWidget(self._slice_max_input, 1)
        value_row.addWidget(unit_label)
        layout.addLayout(value_row)

        layout.addWidget(self._slice_colorbar_canvas)
        return card

    def _build_viewer_muon_card(self, parent: QWidget) -> QFrame:
        card = QFrame(parent)
        card.setObjectName("uepIsosurfaceInfoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(card_margins())
        layout.setSpacing(form_spacing())
        title = QLabel("Muons", card)
        title.setObjectName("uepIsosurfaceLegendTitle")
        layout.addWidget(title)
        self._viewer_muon_legend_entries = QWidget(card)
        self._viewer_muon_legend_entries.setStyleSheet("background: transparent;")
        self._viewer_muon_legend_entries_layout = QVBoxLayout(self._viewer_muon_legend_entries)
        self._viewer_muon_legend_entries_layout.setContentsMargins(0, 0, 0, 0)
        self._viewer_muon_legend_entries_layout.setSpacing(5)
        layout.addWidget(self._viewer_muon_legend_entries)
        self._manage_muons_button = QPushButton("Add / remove muon", card)
        self._manage_muons_button.clicked.connect(self._open_viewer_muon_dialog)
        layout.addWidget(self._manage_muons_button, 0, Qt.AlignmentFlag.AlignRight)
        return card

    def _build_batch_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 2, 6, 6)
        layout.setSpacing(10)

        controls = QWidget(tab)
        controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        title = QLabel("Random-Start Batch Search", controls)
        style_section_title(title)
        subtitle = QLabel(
            "Generate initial muon positions and relax them within the unperturbed electrostatic potential. The minima are grouped into sites when the search ends.",
            controls,
        )
        subtitle.setWordWrap(True)
        style_hint_text(subtitle)
        controls_layout.addWidget(title)
        controls_layout.addWidget(subtitle)

        cards_row = QHBoxLayout()
        cards_row.setContentsMargins(0, 0, 0, 0)
        cards_row.setSpacing(form_spacing())

        controls_card = QFrame(controls)
        self._batch_controls_card = controls_card
        controls_card.setObjectName("analysisSliderPanel")
        card_layout = QVBoxLayout(controls_card)
        card_layout.setContentsMargins(card_margins())
        card_layout.setSpacing(form_spacing())

        search_form = QFormLayout()
        search_form.setContentsMargins(0, 0, 0, 0)
        search_form.setSpacing(form_spacing())
        search_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        search_form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self._batch_form = search_form
        self._batch_count = QSpinBox(controls_card)
        self._batch_count.setRange(1, 200)
        self._batch_count.setValue(12)
        self._batch_min_muon_muon = DecimalLineEdit(controls_card)
        self._batch_min_muon_muon.setRange(0.0, 10.0)
        self._batch_min_muon_muon.setValue(0.8)
        self._batch_min_muon_atom = DecimalLineEdit(controls_card)
        self._batch_min_muon_atom.setRange(0.0, 10.0)
        self._batch_min_muon_atom.setValue(1.0)
        search_form.addRow("Random starts", self._batch_count)
        search_form.addRow("Min muon-muon distance (Å)", self._batch_min_muon_muon)
        search_form.addRow("Min muon-atom distance (Å)", self._batch_min_muon_atom)

        self._batch_advanced_toggle = QToolButton(controls_card)
        self._batch_advanced_toggle.setObjectName("sectionToggleButton")
        self._batch_advanced_toggle.setText("Advanced")
        self._batch_advanced_toggle.setAccessibleName("Advanced search settings")
        self._batch_advanced_toggle.setCheckable(True)
        self._batch_advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._batch_advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._batch_advanced_toggle.toggled.connect(self._on_batch_advanced_toggled)
        search_form.addRow(self._batch_advanced_toggle)

        self._batch_max_attempts = QSpinBox(controls_card)
        self._batch_max_attempts.setRange(10, 500000)
        self._batch_max_attempts.setValue(_DEFAULT_MAX_START_ATTEMPTS)
        self._batch_force_tolerance = DecimalLineEdit(controls_card)
        self._batch_force_tolerance.setRange(1e-8, 10.0)
        self._batch_force_tolerance.setDecimals(8)
        self._batch_force_tolerance.setValue(DEFAULT_FORCE_TOLERANCE_EV_PER_ANG)
        self._batch_force_tolerance.setToolTip(_BATCH_FORCE_TOLERANCE_TOOLTIP)
        self._batch_max_steps = QSpinBox(controls_card)
        self._batch_max_steps.setRange(1, 1000)
        self._batch_max_steps.setValue(DEFAULT_OPTIMIZER_MAX_STEPS)
        search_form.addRow("Max attempts", self._batch_max_attempts)
        search_form.addRow("Force tolerance (eV/Å)", self._batch_force_tolerance)
        search_form.addRow("Optimizer max steps", self._batch_max_steps)
        search_form.labelForField(self._batch_force_tolerance).setToolTip(
            _BATCH_FORCE_TOLERANCE_TOOLTIP
        )
        card_layout.addLayout(search_form)
        self._on_batch_advanced_toggled(False)

        self._batch_run = QPushButton("Run Batch Search", controls_card)
        self._batch_run.clicked.connect(self._run_batch_search)
        self._batch_progress = QProgressBar(controls_card)
        self._batch_progress.setObjectName("uepBatchProgress")
        self._batch_progress.setAccessibleName("Batch search progress")
        self._batch_progress.setRange(0, 1)
        self._batch_progress.setValue(0)
        self._batch_progress.setFormat("Ready")
        self._batch_progress.setTextVisible(True)
        self._apply_batch_progress_style(completed=False)
        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(form_spacing())
        run_row.addWidget(self._batch_run)
        run_row.addWidget(self._batch_progress, 1)
        card_layout.addLayout(run_row)
        card_layout.addStretch(1)

        log_card = QFrame(controls)
        log_card.setObjectName("uepIsosurfaceInfoCard")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(card_margins())
        log_layout.setSpacing(form_spacing())
        log_title = QLabel("Search Log", log_card)
        log_title.setObjectName("uepIsosurfaceLegendTitle")
        self._batch_summary = QTextEdit(log_card)
        self._batch_summary.setAccessibleName("Search log")
        self._batch_summary.setReadOnly(True)
        self._batch_summary.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._batch_summary.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._batch_summary.setMinimumHeight(96)
        # The log takes the height of the controls beside it and scrolls.
        self._batch_summary.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._batch_summary.setPlaceholderText("No batch search has been run yet.")
        log_layout.addWidget(log_title)
        log_layout.addWidget(self._batch_summary, 1)

        cards_row.addWidget(controls_card, 1)
        cards_row.addWidget(log_card, 1)
        controls_layout.addLayout(cards_row)

        results_panel = QWidget(tab)
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)
        self._batch_table = QTreeWidget(tab)
        self._batch_table.setColumnCount(8)
        self._batch_table.setHeaderLabels(_BATCH_TABLE_HEADERS)
        self._batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._batch_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._batch_table.setUniformRowHeights(True)
        self._batch_table.setItemsExpandable(True)
        self._batch_table.setRootIsDecorated(True)
        self._batch_table.setIndentation(_BATCH_TABLE_INDENTATION)
        self._batch_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        use_square_top_table_corners(self._batch_table)
        batch_header = self._batch_table.header()
        batch_header.setStretchLastSection(False)
        batch_header.setMinimumSectionSize(24)
        batch_header.setDefaultAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        batch_header.setFixedHeight(
            max(batch_header.sizeHint().height(), (self.fontMetrics().height() * 2) + 16)
        )
        for column in range(self._batch_table.columnCount()):
            batch_header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        self._batch_table.itemSelectionChanged.connect(self._sync_batch_structure_viewer)
        results_layout.addWidget(self._batch_table, 1)
        grouping_row = QHBoxLayout()
        grouping_row.setContentsMargins(0, 0, 0, 0)
        grouping_label = QLabel("Grouping tolerance (Å)", results_panel)
        self._batch_grouping_tolerance = DecimalLineEdit(
            results_panel,
            minimum=_GROUPING_TOLERANCE_MINIMUM_ANG,
            maximum=10.0,
            decimals=4,
            restore_value=self._grouping_tolerance,
        )
        self._batch_grouping_tolerance.setValue(DEFAULT_GROUPING_TOLERANCE_ANG)
        self._batch_grouping_tolerance.setMaximumWidth(124)
        self._batch_grouping_tolerance.setAccessibleName("Grouping tolerance in ångström")
        self._batch_grouping_tolerance.setToolTip(_GROUPING_TOLERANCE_TOOLTIP)
        # The text, not the value: the prefixes of the minimum already clamp to it.
        self._batch_grouping_tolerance.textChanged.connect(self._on_grouping_tolerance_edited)
        self._batch_grouping_tolerance.editingFinished.connect(self._on_grouping_tolerance_finished)
        grouping_label.setBuddy(self._batch_grouping_tolerance)
        grouping_row.addWidget(grouping_label)
        grouping_row.addWidget(self._batch_grouping_tolerance)
        grouping_row.addStretch(1)
        results_layout.addLayout(grouping_row)

        viewer_panel = QWidget(tab)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self._batch_viewer_summary = QLabel(_BATCH_VIEWER_IDLE_TEXT)
        self._batch_viewer_summary.setWordWrap(True)
        self._batch_viewer = AtomicStructureViewer(viewer_panel)
        self._batch_viewer.setMinimumWidth(480)
        self._batch_viewer.set_structure(self._viewer_host_structure)
        viewer_layout.addWidget(self._batch_viewer_summary)
        viewer_layout.addWidget(self._batch_viewer, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, tab)
        splitter.setChildrenCollapsible(False)
        self._batch_splitter = splitter
        splitter.addWidget(results_panel)
        splitter.addWidget(viewer_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([590, 590])
        splitter.splitterMoved.connect(lambda *_args: self._update_batch_table_column_layout())

        layout.addWidget(controls)
        layout.addWidget(splitter, 1)
        return tab

    def _on_batch_advanced_toggled(self, shown: bool) -> None:
        self._batch_advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow
        )
        for control in (
            self._batch_max_attempts,
            self._batch_force_tolerance,
            self._batch_max_steps,
        ):
            self._batch_form.setRowVisible(control, shown)

    def _open_viewer_muon_dialog(self) -> None:
        if self._viewer_muon_dialog is None:
            dialog = _ManageViewerMuonsDialog(
                title="Add / Remove Muons",
                subtitle=(
                    "Add muon sites from the batch-search groups or define a custom fractional position in the job's cell."
                ),
                parent=self,
            )
            dialog.add_candidates_requested.connect(self._add_candidate_muons_to_viewer)
            dialog.add_custom_requested.connect(self._add_custom_muon_to_viewer)
            dialog.remove_requested.connect(self._remove_viewer_muons)
            dialog.rename_requested.connect(self._rename_viewer_muon)
            dialog.current_selection_changed.connect(self._set_current_viewer_muon_selection)
            dialog.destroyed.connect(lambda *_args: setattr(self, "_viewer_muon_dialog", None))
            self._viewer_muon_dialog = dialog
        self._refresh_viewer_muon_dialog()
        self._viewer_muon_dialog.show()
        self._viewer_muon_dialog.raise_()
        self._viewer_muon_dialog.activateWindow()

    def _set_current_viewer_muon_selection(self, keys: list[str]) -> None:
        self._set_selected_viewer_muon_keys(list(keys))
        self._sync_viewer_muon_selection()
        self._refresh_viewer_muon_dialog()
        if not self._restoring_state:
            self._emit_state_changed()

    def _candidate_entry_by_key(self, key: str) -> _BatchMuonCandidate | None:
        for entry in self._clustered_muon_candidates():
            if entry.key == key:
                return entry
        return None

    def _add_candidate_muons_to_viewer(self, keys: list[str], include_isosurface: bool) -> None:
        selected_keys: list[str] = []
        for key in keys:
            candidate = self._candidate_entry_by_key(str(key))
            if candidate is None:
                continue
            existing = next(
                (
                    entry
                    for entry in self._viewer_muons
                    if entry.source_kind == "clustered"
                    and entry.result_index == int(candidate.result_index)
                ),
                None,
            )
            if existing is None:
                existing = _ManagedViewerMuonState(
                    key=self._new_viewer_muon_key(),
                    label=candidate.label,
                    source_label=candidate.cluster_label,
                    source_kind="clustered",
                    fractional_position=candidate.fractional_position,
                    cartesian_position_ang=candidate.cartesian_position_ang,
                    result_index=int(candidate.result_index),
                    cluster_index=int(candidate.cluster_index),
                    classical_energy_ev=float(candidate.classical_energy_ev),
                    zero_point_energy_ev=candidate.zero_point_energy_ev,
                    total_energy_ev=candidate.total_energy_ev,
                    display_fractional_position=candidate.display_fractional_position,
                )
                self._viewer_muons.append(existing)
            if include_isosurface:
                self._ensure_isosurface_layer_for_muon(existing)
            selected_keys.append(existing.key)
        self._set_selected_viewer_muon_keys(selected_keys)
        self._sync_potential_viewer()
        self._refresh_viewer_muon_dialog()
        if not self._restoring_state:
            self._emit_state_changed()

    def _add_custom_muon_to_viewer(self, fractional: tuple[float, float, float]) -> None:
        cartesian = tuple(
            float(value)
            for value in lattice_frac_to_cart(fractional, self._source.structure.lattice_vectors)
        )
        entry = _ManagedViewerMuonState(
            key=self._new_viewer_muon_key(),
            label=self._current_custom_muon_label(),
            source_label="Manual",
            source_kind="custom",
            fractional_position=tuple(float(value) for value in fractional),
            cartesian_position_ang=cartesian,
        )
        self._viewer_muons.append(entry)
        self._set_selected_viewer_muon_keys([entry.key])
        self._sync_potential_viewer()
        self._refresh_viewer_muon_dialog()
        if not self._restoring_state:
            self._emit_state_changed()

    def _rename_viewer_muon(self, key: str, label: str) -> None:
        normalized_key = str(key)
        normalized_label = str(label).strip()
        if not normalized_label:
            return
        for entry in self._viewer_muons:
            if entry.key != normalized_key:
                continue
            if entry.label == normalized_label:
                return
            entry.label = normalized_label
            break
        else:
            return
        self._refresh_viewer_muon_legend()
        self._refresh_viewer_muon_dialog()
        if not self._restoring_state:
            self._emit_state_changed()

    def _remove_viewer_muons(self, keys: list[str], *, emit_state: bool = True) -> None:
        if not keys:
            return
        key_set = {str(key) for key in keys}
        retained: list[_ManagedViewerMuonState] = []
        for entry in self._viewer_muons:
            if entry.key in key_set:
                self._remove_isosurface_layer_by_key(entry.linked_isosurface_layer_key)
            else:
                retained.append(entry)
        self._viewer_muons = retained
        self._set_selected_viewer_muon_keys(self._selected_viewer_muon_keys)
        self._sync_potential_viewer()
        self._refresh_viewer_muon_dialog()
        if emit_state and not self._restoring_state:
            self._emit_state_changed()

    def _build_viewer_host_structure(self) -> Structure:
        source_structure = self._source.structure
        properties: dict[str, Any] = {"pbc": [True, True, True]}
        linked_source = self._source_structure
        if source_structure.atomic_spins is not None and len(source_structure.atomic_spins) == len(
            source_structure.species
        ):
            properties["atomic_spins"] = _viewer_spin_values(source_structure.atomic_spins)
        elif linked_source is not None:
            linked_spins = _coerced_spin_values(
                linked_source.properties.get("atomic_spins"),
                len(linked_source.species),
            )
            if linked_spins is not None:
                aligned_spins = _align_spins_to_fractional_structure_order(
                    target_species=source_structure.species,
                    target_fractional_positions=source_structure.fractional_positions,
                    source_structure=linked_source,
                    source_spins=linked_spins,
                )
                if aligned_spins is not None:
                    properties["atomic_spins"] = _viewer_spin_values(aligned_spins)
        if linked_source is not None and len(linked_source.species) == len(
            source_structure.species
        ):
            for key in ("magnetic_structure", "propagation_vector"):
                value = linked_source.properties.get(key)
                if isinstance(value, (list, dict)):
                    properties[key] = deepcopy(value)
        return Structure(
            label=f"{self._source_job_label} host",
            origin="uep_source",
            lattice_vectors=[list(vector) for vector in source_structure.lattice_vectors],
            species=list(source_structure.species),
            fractional_positions=[
                list(position) for position in source_structure.fractional_positions
            ],
            properties=properties,
        )

    @staticmethod
    def _new_viewer_muon_key() -> str:
        return f"viewer-muon:{uuid4().hex}"

    @staticmethod
    def _new_isosurface_layer_key() -> str:
        return f"isosurface-layer:{uuid4().hex}"

    @staticmethod
    def _serialize_managed_viewer_muon(entry: _ManagedViewerMuonState) -> dict[str, object]:
        return {
            "key": entry.key,
            "label": entry.label,
            "source_label": entry.source_label,
            "source_kind": entry.source_kind,
            "fractional_position": [float(value) for value in entry.fractional_position],
            "cartesian_position_ang": [float(value) for value in entry.cartesian_position_ang],
            "result_index": entry.result_index,
            "cluster_index": entry.cluster_index,
            "classical_energy_ev": entry.classical_energy_ev,
            "zero_point_energy_ev": entry.zero_point_energy_ev,
            "total_energy_ev": entry.total_energy_ev,
            "linked_isosurface_layer_key": entry.linked_isosurface_layer_key,
        }

    @classmethod
    def _deserialize_managed_viewer_muon(cls, payload: object) -> _ManagedViewerMuonState | None:
        if not isinstance(payload, Mapping):
            return None
        fractional = cls._tuple3(payload.get("fractional_position"))
        cartesian = cls._tuple3(payload.get("cartesian_position_ang"))
        if fractional is None or cartesian is None:
            return None
        key_raw = payload.get("key")
        key = str(key_raw).strip() if key_raw is not None else ""
        if not key:
            key = cls._new_viewer_muon_key()
        result_index_raw = payload.get("result_index")
        cluster_index_raw = payload.get("cluster_index")
        try:
            result_index = None if result_index_raw is None else int(result_index_raw)
        except (TypeError, ValueError):
            result_index = None
        try:
            cluster_index = None if cluster_index_raw is None else int(cluster_index_raw)
        except (TypeError, ValueError):
            cluster_index = None
        return _ManagedViewerMuonState(
            key=key,
            label=str(payload.get("label", "Muon")),
            source_label=str(payload.get("source_label", "")),
            source_kind=str(payload.get("source_kind", "custom")),
            fractional_position=fractional,
            cartesian_position_ang=cartesian,
            result_index=result_index,
            cluster_index=cluster_index,
            classical_energy_ev=cls._optional_float(payload.get("classical_energy_ev")),
            zero_point_energy_ev=cls._optional_float(payload.get("zero_point_energy_ev")),
            total_energy_ev=cls._optional_float(payload.get("total_energy_ev")),
            linked_isosurface_layer_key=(
                None
                if payload.get("linked_isosurface_layer_key") in (None, "")
                else str(payload.get("linked_isosurface_layer_key"))
            ),
        )

    @staticmethod
    def _optional_float(value: object) -> float | None:
        try:
            if value is None:
                return None
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) else None

    def _clustered_muon_candidates(self) -> list[_BatchMuonCandidate]:
        summary = self._latest_batch_summary
        if summary is None or not self._latest_batch_results:
            return []
        site_numbers = relaxed_site_number_map(
            self._latest_batch_results,
            included=(index for cluster in summary.clusters for index in cluster.member_indexes),
        )
        candidates: list[_BatchMuonCandidate] = []
        for cluster in summary.clusters:
            cluster_label = f"Site {cluster.cluster_index}"
            for result_index in self._batch_sorted_member_indexes(cluster):
                result = self._latest_batch_results[result_index]
                candidates.append(
                    _BatchMuonCandidate(
                        key=f"result:{int(result_index)}",
                        label=f"Start {site_numbers[result_index]}",
                        cluster_label=cluster_label,
                        result_index=int(result_index),
                        cluster_index=int(cluster.cluster_index),
                        fractional_position=tuple(
                            float(value) for value in result.optimized_fractional_position
                        ),
                        cartesian_position_ang=tuple(
                            float(value) for value in result.optimized_position_ang
                        ),
                        classical_energy_ev=float(result.classical_energy_ev),
                        zero_point_energy_ev=self._optional_float(result.zero_point_energy_ev),
                        display_fractional_position=self._batch_table_position(result),
                        total_energy_ev=self._optional_float(result.total_energy_ev),
                    )
                )
        return candidates

    def _set_selected_viewer_muon_keys(self, keys: list[str]) -> None:
        self._selected_viewer_muon_keys = [
            key for key in keys if any(entry.key == key for entry in self._viewer_muons)
        ]

    def _viewer_muon_atom_indexes(self) -> list[int]:
        selected_keys = set(self._selected_viewer_muon_keys)
        host_count = len(self._viewer_host_structure.species)
        return [
            host_count + index
            for index, entry in enumerate(self._viewer_muons)
            if entry.key in selected_keys
        ]

    def _viewer_muon_cluster_color_map(self) -> dict[int, str]:
        labels: list[int] = []
        if self._latest_batch_summary is not None:
            labels.extend(
                int(cluster.cluster_index) for cluster in self._latest_batch_summary.clusters
            )
        for entry in self._viewer_muons:
            if entry.cluster_index is not None:
                labels.append(int(entry.cluster_index))
        return cluster_color_map(labels)

    def _viewer_muon_color(self, entry: _ManagedViewerMuonState) -> str:
        if entry.cluster_index is not None:
            color = self._viewer_muon_cluster_color_map().get(int(entry.cluster_index))
            if color is not None:
                return color
        try:
            index = self._viewer_muons.index(entry)
        except ValueError:
            index = 0
        cmap = colormaps["tab20"]
        return mcolors.to_hex(cmap((index + 10) % cmap.N))

    def _build_structure_with_viewer_muons(self) -> Structure:
        muons = self._viewer_muons
        if not muons:
            return self._viewer_host_structure
        host = self._viewer_host_structure
        properties = deepcopy(host.properties)
        for key in ("atomic_spins", "initial_magnetic_moments"):
            values = properties.get(key)
            if isinstance(values, list) and len(values) == len(host.species):
                properties[key] = list(values) + [0.0] * len(muons)
        host_atom_colors = properties.get("atom_colors")
        if isinstance(host_atom_colors, list) and len(host_atom_colors) == len(host.species):
            atom_colors = list(host_atom_colors)
        else:
            atom_colors = [None] * len(host.species)
        atom_colors.extend(self._viewer_muon_color(entry) for entry in muons)
        properties["atom_colors"] = atom_colors
        return Structure(
            label=host.label,
            origin=host.origin,
            lattice_vectors=[list(vector) for vector in host.lattice_vectors],
            species=list(host.species) + ["X"] * len(muons),
            fractional_positions=[list(position) for position in host.fractional_positions]
            + [[float(value) for value in entry.fractional_position] for entry in muons],
            properties=properties,
        )

    def _restore_viewer_camera(
        self,
        viewer: AtomicStructureViewer,
        previous_camera,
        previous_limits,
        previous_pyvista_snapshot,
    ) -> None:
        if previous_camera is None:
            return
        viewer._camera_state = previous_camera
        viewer._limits = previous_limits
        viewer._axis.set_from_state(previous_camera)
        if getattr(viewer, "_triad_axis", None) is not None:
            viewer._triad_axis.set_from_state(previous_camera)
        viewer._view = (previous_camera.elev, previous_camera.azim)
        viewer._view_roll = previous_camera.roll
        viewer._pyvista_camera_snapshot = previous_pyvista_snapshot
        viewer._render()

    def _set_viewer_structure_preserving_view(
        self, viewer: AtomicStructureViewer, structure: Structure
    ) -> None:
        if getattr(viewer, "_structure", None) is structure:
            return
        previous_camera = deepcopy(getattr(viewer, "_camera_state", None))
        previous_limits = deepcopy(getattr(viewer, "_limits", None))
        previous_show_spins = bool(viewer._show_spins.isChecked())
        live_pyvista_snapshot = getattr(viewer, "_current_pyvista_camera_snapshot", lambda: None)()
        previous_pyvista_snapshot = deepcopy(
            live_pyvista_snapshot
            if live_pyvista_snapshot is not None
            else getattr(viewer, "_pyvista_camera_snapshot", None)
        )
        with viewer.batch_update():
            viewer.set_structure(structure)
            if viewer._show_spins.isChecked() != previous_show_spins:
                blocked = viewer._show_spins.blockSignals(True)
                viewer._show_spins.setChecked(previous_show_spins)
                viewer._show_spins.blockSignals(blocked)
            self._restore_viewer_camera(
                viewer, previous_camera, previous_limits, previous_pyvista_snapshot
            )

    def _sync_viewer_muon_selection(self) -> None:
        indexes = self._viewer_muon_atom_indexes()
        if list(getattr(self._potential_viewer, "_selected_indexes", [])) == indexes:
            return
        self._potential_viewer.select_atoms(indexes)

    @staticmethod
    def _clear_widget_layout(layout: QVBoxLayout | QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    @staticmethod
    def _keep_dialog_in_front(dialog: QDialog | None) -> None:
        if dialog is None or not dialog.isVisible():
            return
        QTimer.singleShot(0, lambda: UepAnalysisDialog._raise_dialog_and_parent(dialog))

    @staticmethod
    def _raise_dialog_and_parent(dialog: QDialog | None) -> None:
        if dialog is None or not dialog.isVisible():
            return
        parent = dialog.parentWidget()
        if parent is not None and parent.isVisible():
            parent.raise_()
            parent.activateWindow()
        dialog.raise_()
        dialog.activateWindow()

    def _refresh_viewer_muon_dialog(self) -> None:
        dialog = self._viewer_muon_dialog
        if dialog is None:
            return
        dialog.set_candidate_entries(self._clustered_muon_candidates())
        dialog.set_current_entries(
            self._viewer_muons, selected_keys=list(self._selected_viewer_muon_keys)
        )
        dialog.set_include_isosurfaces_available(self._potential_grid_cache is not None)
        self._keep_dialog_in_front(dialog)

    def _current_custom_muon_label(self) -> str:
        self._custom_muon_counter += 1
        return f"Custom {self._custom_muon_counter}"

    def _sync_custom_muon_counter(self) -> None:
        maximum = 0
        for entry in self._viewer_muons:
            if entry.source_kind != "custom":
                continue
            prefix = "Custom "
            if entry.label.startswith(prefix):
                suffix = entry.label.removeprefix(prefix)
                if suffix.isdigit():
                    maximum = max(maximum, int(suffix))
        self._custom_muon_counter = maximum

    def _find_isosurface_layer_index_by_key(self, layer_key: str | None) -> int | None:
        if not layer_key:
            return None
        for index, layer in enumerate(self._isosurface_layers):
            if layer.key == layer_key:
                return index
        return None

    def _remove_isosurface_layer_by_key(self, layer_key: str | None) -> None:
        layer_index = self._find_isosurface_layer_index_by_key(layer_key)
        if layer_index is None:
            return
        layer = self._isosurface_layers[layer_index]
        self._unlink_isosurface_layer_from_its_muon(layer)
        del self._isosurface_layers[layer_index]
        if not self._isosurface_layers:
            self._isosurface_layers = [
                _IsosurfaceLayerState(label="Isosurface 1", key=self._new_isosurface_layer_key())
            ]
        self._active_isosurface_index = min(
            self._active_isosurface_index, len(self._isosurface_layers) - 1
        )
        self._update_active_isosurface_label()
        self._apply_active_isosurface_layer_to_controls()

    def _unlink_isosurface_layer_from_its_muon(self, layer: _IsosurfaceLayerState) -> None:
        if layer.managed_muon_key is None:
            return
        for entry in self._viewer_muons:
            if entry.key == layer.managed_muon_key:
                entry.linked_isosurface_layer_key = None
                return

    def _ensure_isosurface_layer_for_muon(self, entry: _ManagedViewerMuonState) -> None:
        if (
            entry.linked_isosurface_layer_key is not None
            or entry.total_energy_ev is None
            or self._potential_grid_cache is None
        ):
            return
        level = float(entry.total_energy_ev) - self._potential_minimum_ev
        layer_key = self._new_isosurface_layer_key()
        layer = _IsosurfaceLayerState(
            label=entry.label,
            key=layer_key,
            level_ev=level,
            color=self._viewer_muon_color(entry),
            managed_muon_key=entry.key,
        )
        self._isosurface_layers.append(layer)
        entry.linked_isosurface_layer_key = layer_key
        try:
            layer.mesh = extract_isosurface(self._potential_grid_cache, level_ev=level)
        except UepError as exc:
            self._set_potential_status(
                f"The isosurface of muon {entry.label} could not be extracted: {exc}", kind="error"
            )
        self._update_active_isosurface_label()

    def _clear_clustered_viewer_muons(self) -> None:
        self._remove_viewer_muons(
            [entry.key for entry in self._viewer_muons if entry.source_kind == "clustered"],
            emit_state=False,
        )

    def _batch_result_sort_key(self, result_index: int) -> tuple[float, int]:
        result = self._latest_batch_results[int(result_index)]
        energy = float(result.classical_energy_ev)
        if not np.isfinite(energy):
            energy = float("inf")
        return (energy, int(result_index))

    def _batch_sorted_member_indexes(self, cluster: UepClusterSummary) -> tuple[int, ...]:
        return tuple(
            sorted(
                (int(index) for index in cluster.member_indexes), key=self._batch_result_sort_key
            )
        )

    def _payload_member_indexes(self, payload: Mapping[str, object]) -> tuple[int, ...]:
        cluster = payload.get("cluster")
        members = cluster.get("member_indexes") if isinstance(cluster, Mapping) else None
        if not isinstance(members, (list, tuple)):
            return ()
        return tuple(sorted((int(index) for index in members), key=self._batch_result_sort_key))

    @staticmethod
    def _batch_cluster_key(cluster_index: int) -> str:
        return f"cluster:{int(cluster_index)}"

    @staticmethod
    def _batch_site_key(cluster_index: int, result_index: int) -> str:
        return f"cluster:{int(cluster_index)}:site:{int(result_index)}"

    @staticmethod
    def _format_batch_energy(value: float | None) -> str:
        return "" if value is None or not np.isfinite(float(value)) else f"{float(value):.6f}"

    @staticmethod
    def _format_batch_fractional(value: float) -> str:
        return f"{float(value):.6f}"

    def _iter_batch_tree_items(self) -> list[tuple[QTreeWidgetItem, int]]:
        items: list[tuple[QTreeWidgetItem, int]] = []
        for top_level_index in range(self._batch_table.topLevelItemCount()):
            top_level_item = self._batch_table.topLevelItem(top_level_index)
            if top_level_item is None:
                continue
            items.append((top_level_item, 0))
            for child_index in range(top_level_item.childCount()):
                child_item = top_level_item.child(child_index)
                if child_item is not None:
                    items.append((child_item, 1))
        return items

    def _batch_table_content_widths(self) -> list[int]:
        font_metrics = self._batch_table.fontMetrics()
        header_metrics = self._batch_table.header().fontMetrics()
        widths: list[int] = []
        for column, header in enumerate(_BATCH_TABLE_HEADERS):
            header_width = (
                max(header_metrics.horizontalAdvance(line) for line in header.splitlines()) + 16
            )
            content_width = header_width
            for item, depth in self._iter_batch_tree_items():
                text = item.text(column)
                if not text:
                    continue
                padding = 14
                if column == 0:
                    padding += (depth * self._batch_table.indentation()) + 14
                content_width = max(content_width, font_metrics.horizontalAdvance(text) + padding)
            widths.append(content_width)
        return widths

    def _update_batch_table_column_layout(self) -> None:
        if not hasattr(self, "_batch_table"):
            return
        widths = self._batch_table_content_widths()
        if not widths:
            return
        viewport_width = max(int(self._batch_table.viewport().width()), 0)
        total_width = sum(widths)
        if viewport_width > total_width:
            extra = viewport_width - total_width
            weighted_columns = ((0, 0.48), (5, 0.18), (6, 0.14), (7, 0.20))
            distributed = 0
            for column, fraction in weighted_columns:
                delta = int(extra * fraction)
                widths[column] += delta
                distributed += delta
            widths[0] += max(extra - distributed, 0)
        for column, width in enumerate(widths):
            self._batch_table.setColumnWidth(
                column, max(int(width), self._batch_table.header().minimumSectionSize())
            )

    def _expanded_batch_keys(self) -> list[str]:
        items = (
            self._batch_table.topLevelItem(row)
            for row in range(self._batch_table.topLevelItemCount())
        )
        return [
            str(item.data(0, Qt.ItemDataRole.UserRole + 1)) for item in items if item.isExpanded()
        ]

    def _selected_batch_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for item in self._batch_table.selectedItems():
            key = item.data(0, Qt.ItemDataRole.UserRole + 1)
            if isinstance(key, str) and key and key not in seen:
                keys.append(key)
                seen.add(key)
        return keys

    def _selected_batch_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in self._batch_table.selectedItems():
            key = item.data(0, Qt.ItemDataRole.UserRole + 1)
            payload = item.data(0, Qt.ItemDataRole.UserRole)
            if not isinstance(key, str) or key in seen or not isinstance(payload, dict):
                continue
            payloads.append(payload)
            seen.add(key)
        return payloads

    def _batch_selection_result_indexes(self) -> tuple[int, ...]:
        result_indexes: list[int] = []
        seen: set[int] = set()
        for payload in self._selected_batch_payloads():
            kind = str(payload.get("kind", ""))
            indexes: list[int] = []
            if kind == "cluster":
                indexes.extend(self._payload_member_indexes(payload))
            elif kind == "site":
                raw_index = payload.get("site_result_index")
                try:
                    indexes.append(int(raw_index))
                except (TypeError, ValueError):
                    continue
            for result_index in indexes:
                if result_index in seen or not (
                    0 <= result_index < len(self._latest_batch_results)
                ):
                    continue
                result_indexes.append(result_index)
                seen.add(result_index)
        return tuple(result_indexes)

    def _restore_batch_selection(
        self,
        *,
        keys: list[str] | tuple[str, ...] | None = None,
        legacy_row: int | None = None,
    ) -> None:
        self._batch_table.clearSelection()
        if not self._batch_item_by_key:
            return
        normalized_keys = [key for key in (keys or []) if isinstance(key, str)]
        if (
            not normalized_keys
            and legacy_row is not None
            and 0 <= legacy_row < self._batch_table.topLevelItemCount()
        ):
            item = self._batch_table.topLevelItem(legacy_row)
            if item is not None:
                normalized_keys = [self._batch_cluster_key(legacy_row + 1)]
        first_selected: QTreeWidgetItem | None = None
        for key in normalized_keys:
            item = self._batch_item_by_key.get(key)
            if item is None:
                continue
            parent = item.parent()
            if parent is not None:
                parent.setExpanded(True)
            item.setSelected(True)
            if first_selected is None:
                first_selected = item
        if first_selected is not None:
            # Without NoUpdate this would clear every other row just selected.
            self._batch_table.setCurrentItem(
                first_selected, 0, QItemSelectionModel.SelectionFlag.NoUpdate
            )

    def _batch_point_overlay_for_selection(self) -> list[dict[str, object]]:
        positions = [
            self._latest_batch_results[result_index].optimized_position_ang
            for result_index in self._batch_selection_result_indexes()
        ]
        if not positions:
            return []
        return [
            {
                "positions": positions,
                "colors": current_palette().accent,
                "radii": [MUON_VIEWER_RADIUS_ANG] * len(positions),
                "alpha": 0.96,
            }
        ]

    def _sync_batch_structure_viewer(self) -> None:
        payloads = self._selected_batch_payloads()
        result_indexes = self._batch_selection_result_indexes()
        with self._batch_viewer.batch_update():
            self._batch_viewer.set_structure(self._viewer_host_structure)
            self._batch_viewer.set_point_overlays(self._batch_point_overlay_for_selection())
        if not payloads or not result_indexes:
            self._batch_viewer_summary.setText(_BATCH_VIEWER_IDLE_TEXT)
            return

        cluster_indexes: set[int] = set()
        for payload in payloads:
            cluster_payload = payload.get("cluster")
            if isinstance(cluster_payload, Mapping):
                cluster_indexes.add(int(cluster_payload["cluster_index"]))
        selection_count = len(payloads)
        muon_count = len(result_indexes)
        if (
            selection_count == 1
            and str(payloads[0].get("kind", "")) == "cluster"
            and len(cluster_indexes) == 1
        ):
            cluster_index = next(iter(cluster_indexes))
            self._batch_viewer_summary.setText(
                f"Showing all {muon_count} optimized muon positions from site {cluster_index} overlaid on the host structure."
            )
            return
        self._batch_viewer_summary.setText(
            f"Showing {muon_count} selected optimized muon positions from {selection_count} selected table entr"
            f"{'y' if selection_count == 1 else 'ies'} across {len(cluster_indexes)} site"
            f"{'' if len(cluster_indexes) == 1 else 's'}."
        )

    @classmethod
    def _deserialize_slice_plane(cls, payload: object) -> UepSlicePlane | None:
        if not isinstance(payload, Mapping):
            return None
        anchor = cls._tuple3(payload.get("anchor_ang"))
        normal = cls._tuple3(payload.get("normal_ang"))
        basis_u = cls._tuple3(payload.get("basis_u_ang"))
        basis_v = cls._tuple3(payload.get("basis_v_ang"))
        extent_u_raw = payload.get("extent_u_ang")
        extent_v_raw = payload.get("extent_v_ang")
        if (
            None in (anchor, normal, basis_u, basis_v)
            or not isinstance(extent_u_raw, (list, tuple))
            or not isinstance(extent_v_raw, (list, tuple))
        ):
            return None
        if len(extent_u_raw) != 2 or len(extent_v_raw) != 2:
            return None
        try:
            return UepSlicePlane(
                anchor_ang=anchor,
                normal_ang=normal,
                basis_u_ang=basis_u,
                basis_v_ang=basis_v,
                extent_u_ang=(float(extent_u_raw[0]), float(extent_u_raw[1])),
                extent_v_ang=(float(extent_v_raw[0]), float(extent_v_raw[1])),
                points_u=int(payload.get("points_u", _SLICE_DEFAULT_POINTS)),
                points_v=int(payload.get("points_v", _SLICE_DEFAULT_POINTS)),
            )
        except (TypeError, ValueError):
            return None

    def _active_isosurface_layer(self) -> _IsosurfaceLayerState:
        if not self._isosurface_layers:
            self._isosurface_layers = [_IsosurfaceLayerState(label="Isosurface 1")]
            self._active_isosurface_index = 0
        self._active_isosurface_index = min(
            max(int(self._active_isosurface_index), 0), len(self._isosurface_layers) - 1
        )
        return self._isosurface_layers[self._active_isosurface_index]

    def _restored_viewer_muons(self, state: Mapping[str, object]) -> list[_ManagedViewerMuonState]:
        """The saved muons; a window saved with two viewers shows both lists' muons in one."""

        muons: list[_ManagedViewerMuonState] = []
        seen_results: set[int] = set()
        seen_positions: set[tuple[float, ...]] = set()
        for name in ("viewer_muons", "isosurface_viewer_muons", "slice_viewer_muons"):
            raw = state.get(name)
            if not isinstance(raw, list):
                continue
            for payload in raw:
                entry = self._deserialize_managed_viewer_muon(payload)
                if entry is None:
                    continue
                position = tuple(round(value, 8) for value in entry.fractional_position)
                if entry.source_kind == "clustered" and entry.result_index is not None:
                    if entry.result_index in seen_results:
                        continue
                    seen_results.add(entry.result_index)
                elif position in seen_positions:
                    continue
                seen_positions.add(position)
                muons.append(entry)
        return muons

    def _default_isosurface_level(self) -> float | None:
        if self._potential_grid_cache is None:
            return None
        return 0.5 * self._potential_level_ceiling_ev

    def _isosurface_level_bounds(self) -> tuple[float, float] | None:
        """What a level can be: the whole grid, above the slider's clipped top too."""

        if self._potential_grid_cache is None:
            return None
        return (0.0, self._potential_level_maximum_ev)

    def _slice_level_bounds(self) -> tuple[float, float] | None:
        return self._isosurface_level_bounds()

    def _level_to_slider(self, level: float, slider_maximum: int) -> int:
        ceiling = self._potential_level_ceiling_ev
        fraction = 0.0 if ceiling <= 0.0 else min(max(float(level) / ceiling, 0.0), 1.0)
        return int(round(fraction * float(slider_maximum)))

    def _default_slice_bounds(self) -> tuple[float, float] | None:
        if self._potential_grid_cache is None:
            return None
        if self._slice_saturation_bounds is None:
            self._slice_saturation_bounds = (0.0, self._potential_level_ceiling_ev)
        return self._slice_saturation_bounds

    def _selected_slice_atom_positions(self) -> list[tuple[float, float, float]]:
        lattice_vectors = self._viewer_structure.lattice_vectors
        result: list[tuple[float, float, float]] = []
        for point in self._selected_slice_atom_fractional_positions():
            cartesian = lattice_frac_to_cart(point, lattice_vectors)
            result.append((float(cartesian[0]), float(cartesian[1]), float(cartesian[2])))
        return result

    def _selected_slice_atom_fractional_positions(self) -> list[tuple[float, float, float]]:
        structure = getattr(self._potential_viewer, "_structure", None) or self._viewer_structure
        selected = getattr(self._potential_viewer, "_selected_indexes", [])
        fractional_positions = np.asarray(structure.fractional_positions, dtype=float)
        result: list[tuple[float, float, float]] = []
        for index in selected:
            if 0 <= int(index) < len(fractional_positions):
                point = fractional_positions[int(index)]
                result.append((float(point[0]), float(point[1]), float(point[2])))
        return result

    def _slice_default_normal_components(self) -> tuple[float, float, float]:
        view_direction = self._potential_viewer.current_view_direction()
        return cartesian_normal_to_lattice_plane_components(
            self._source.structure.lattice_vectors, view_direction
        )

    def _set_potential_status(self, text: str = "", *, kind: str = "info") -> None:
        """Say what the tab is busy with or what failed; nothing when all is well."""

        label = self._potential_status
        label.setText(str(text))
        label.setVisible(bool(text))
        label.setProperty("statusKind", str(kind))
        style = label.style()
        style.unpolish(label)
        style.polish(label)
        label.update()

    def _clear_earlier_potential_message(self) -> None:
        """Clear what the last action reported, as the next one starts."""

        self._set_potential_status()

    def _normalized_slice_bounds(
        self, minimum: float, maximum: float
    ) -> tuple[float, float] | None:
        bounds = self._slice_level_bounds()
        if bounds is None:
            return None
        lower = min(max(float(minimum), bounds[0]), bounds[1])
        upper = min(max(float(maximum), bounds[0]), bounds[1])
        if lower > upper:
            lower, upper = upper, lower
        return (lower, upper)

    def _apply_slice_bound_controls(self, lower: float, upper: float, *, sync_slider: bool) -> None:
        bounds = self._slice_level_bounds()
        if bounds is None:
            return
        if sync_slider:
            slider_maximum = self._slice_range_slider.maximum()
            previous = self._slice_range_slider.blockSignals(True)
            self._slice_range_slider.setValues(
                self._level_to_slider(lower, slider_maximum),
                self._level_to_slider(upper, slider_maximum),
            )
            self._slice_range_slider.blockSignals(previous)
        for widget, value in ((self._slice_min_input, lower), (self._slice_max_input, upper)):
            blocked = widget.blockSignals(True)
            widget.setRange(bounds[0], bounds[1])
            widget.setValue(value)
            widget.blockSignals(blocked)

    def _set_slice_bounds(
        self,
        minimum: float,
        maximum: float,
        *,
        emit_state: bool,
        refresh_legend: bool = True,
        preview: bool = False,
    ) -> None:
        normalized = self._normalized_slice_bounds(minimum, maximum)
        if normalized is None:
            return
        lower, upper = normalized
        self._slice_saturation_bounds = (lower, upper)
        self._apply_slice_bound_controls(lower, upper, sync_slider=True)
        self._sync_slice_viewer(preview=preview)
        if refresh_legend:
            self._refresh_slice_colorbar()
        if emit_state and not self._restoring_state:
            self._emit_state_changed()

    def _queue_slice_preview(self, minimum: float, maximum: float, *, immediate: bool) -> None:
        normalized = self._normalized_slice_bounds(minimum, maximum)
        if normalized is None:
            return
        self._pending_slice_preview_bounds = normalized
        if immediate:
            self._slice_preview_timer.stop()
            self._dispatch_pending_slice_preview()
            return
        self._slice_preview_timer.start()

    def _dispatch_pending_slice_preview(self) -> None:
        pending = self._pending_slice_preview_bounds
        if pending is None:
            return
        self._pending_slice_preview_bounds = None
        self._set_slice_bounds(
            pending[0], pending[1], emit_state=False, refresh_legend=False, preview=True
        )

    def _on_slice_slider_changed(self, lower: int, upper: int) -> None:
        if self._potential_grid_cache is None:
            return
        slider_maximum = max(1, self._slice_range_slider.maximum())
        ceiling = self._potential_level_ceiling_ev
        normalized = self._normalized_slice_bounds(
            float(lower) / float(slider_maximum) * ceiling,
            float(upper) / float(slider_maximum) * ceiling,
        )
        if normalized is None:
            return
        self._slice_saturation_bounds = normalized
        self._apply_slice_bound_controls(normalized[0], normalized[1], sync_slider=False)
        self._queue_slice_preview(normalized[0], normalized[1], immediate=False)

    def _on_slice_slider_released(self) -> None:
        if self._slice_saturation_bounds is None:
            self._refresh_slice_colorbar()
            return
        self._pending_slice_preview_bounds = None
        self._slice_preview_timer.stop()
        self._set_slice_bounds(
            self._slice_saturation_bounds[0],
            self._slice_saturation_bounds[1],
            emit_state=True,
            refresh_legend=True,
            preview=False,
        )

    def _request_slice_range_from_editor(self) -> None:
        self._set_slice_bounds(
            self._slice_min_input.value(), self._slice_max_input.value(), emit_state=True
        )

    def _refresh_slice_controls(self) -> None:
        enabled = self._potential_grid_cache is not None and not self._is_isosurface_busy()
        self._slice_range_slider.setEnabled(enabled)
        self._slice_min_input.setEnabled(enabled)
        self._slice_max_input.setEnabled(enabled)
        self._slice_add_plane_button.setEnabled(not self._is_isosurface_busy())
        self._refresh_slice_plane_actions()

    def _slice_plane_numbering(self) -> None:
        for index, plane in enumerate(self._slice_planes):
            plane.label = f"Plane {index + 1}"

    def _refresh_slice_plane_list(self) -> None:
        selected = self._selected_slice_plane_index()
        previous = self._slice_plane_list.blockSignals(True)
        self._slice_plane_list.clear()
        for state in self._slice_planes:
            item = QListWidgetItem(state.label, self._slice_plane_list)
            item.setToolTip(f"Double-click to edit {state.label.lower()}.")
        if selected is not None and self._slice_planes:
            self._slice_plane_list.setCurrentRow(min(selected, len(self._slice_planes) - 1))
        self._slice_plane_list.blockSignals(previous)
        self._refresh_slice_plane_actions()

    def _selected_slice_plane_index(self) -> int | None:
        row = self._slice_plane_list.currentRow()
        selected = row >= 0 and self._slice_plane_list.item(row).isSelected()
        return row if selected else None

    def _refresh_slice_plane_actions(self) -> None:
        has_selection = self._selected_slice_plane_index() is not None
        idle = not self._is_isosurface_busy()
        self._slice_edit_plane_button.setEnabled(has_selection and idle)
        self._slice_remove_plane_button.setEnabled(has_selection and idle)

    def _resample_slice_plane(self, state: _SlicePlaneState) -> None:
        if state.plane is None or self._potential_grid_cache is None:
            state.sample = None
            return
        state.sample = sample_grid_slice(self._potential_grid_cache, state.plane)

    def _resample_all_slice_planes(self) -> None:
        for state in self._slice_planes:
            try:
                self._resample_slice_plane(state)
            except UepError as exc:
                state.sample = None
                self._set_potential_status(
                    f"{state.label} could not be sampled: {exc}", kind="error"
                )
        self._sync_slice_viewer()
        self._refresh_slice_colorbar()
        self._refresh_slice_plane_list()

    @staticmethod
    def _downsample_slice_matrix(values: np.ndarray, *, max_points: int) -> np.ndarray:
        matrix = np.asarray(values)
        if matrix.ndim != 2:
            return matrix
        target = max(2, int(max_points))
        stride_u = max(1, int(np.ceil(matrix.shape[0] / float(target))))
        stride_v = max(1, int(np.ceil(matrix.shape[1] / float(target))))
        reduced = matrix[::stride_u, ::stride_v]
        if reduced.shape[0] < 2 and matrix.shape[0] >= 2:
            reduced = matrix[[0, -1], ::stride_v]
        if reduced.shape[1] < 2 and matrix.shape[1] >= 2:
            reduced = reduced[:, [0, -1]]
        return reduced

    def _viewer_slice_surfaces(self, *, preview: bool = False) -> list[dict[str, object]]:
        bounds = self._default_slice_bounds()
        if bounds is None:
            return []
        surfaces: list[dict[str, object]] = []
        for state in self._slice_planes:
            sample = state.sample
            plane = state.plane
            if sample is None or plane is None:
                continue
            x_coordinates = np.asarray(sample.x_coordinates_ang, dtype=float)
            y_coordinates = np.asarray(sample.y_coordinates_ang, dtype=float)
            z_coordinates = np.asarray(sample.z_coordinates_ang, dtype=float)
            values = np.asarray(sample.potential_total_ev, dtype=float)
            if preview:
                x_coordinates = self._downsample_slice_matrix(
                    x_coordinates, max_points=_SLICE_LIVE_PREVIEW_MAX_POINTS
                )
                y_coordinates = self._downsample_slice_matrix(
                    y_coordinates, max_points=_SLICE_LIVE_PREVIEW_MAX_POINTS
                )
                z_coordinates = self._downsample_slice_matrix(
                    z_coordinates, max_points=_SLICE_LIVE_PREVIEW_MAX_POINTS
                )
                values = self._downsample_slice_matrix(
                    values, max_points=_SLICE_LIVE_PREVIEW_MAX_POINTS
                )
            surfaces.append(
                {
                    "x_coordinates": x_coordinates,
                    "y_coordinates": y_coordinates,
                    "z_coordinates": z_coordinates,
                    "values": values,
                    "anchor": np.asarray(plane.anchor_ang, dtype=float),
                    "normal": np.asarray(plane.normal_ang, dtype=float),
                    "minimum": float(bounds[0]),
                    "maximum": float(bounds[1]),
                    "alpha": _SLICE_SURFACE_ALPHA,
                    "cmap": "coolwarm",
                }
            )
        return surfaces

    def _sync_potential_viewer(self) -> None:
        """Show the muons, the isosurfaces and the planes; drawn once.

        Loading a structure clears a viewer's overlays, so it is followed by both kinds.
        """

        signature = tuple(
            (entry.key, entry.fractional_position, self._viewer_muon_color(entry))
            for entry in self._viewer_muons
        )
        with self._potential_viewer.batch_update():
            if signature != self._viewer_structure_signature:
                self._viewer_structure_signature = signature
                self._viewer_structure = self._build_structure_with_viewer_muons()
                self._set_viewer_structure_preserving_view(
                    self._potential_viewer, self._viewer_structure
                )
            self._potential_viewer.set_isosurfaces(self._viewer_isosurface_surfaces())
            self._show_slice_viewer_state(preview=False)
            self._sync_viewer_muon_selection()
        self._refresh_isosurface_legend()
        self._refresh_viewer_muon_legend()

    def _sync_slice_viewer(self, *, preview: bool = False) -> None:
        with self._potential_viewer.batch_update():
            self._show_slice_viewer_state(preview=preview)

    def _show_slice_viewer_state(self, *, preview: bool) -> None:
        viewer = self._potential_viewer
        previous_camera = deepcopy(getattr(viewer, "_camera_state", None))
        previous_limits = deepcopy(getattr(viewer, "_limits", None))
        live_pyvista_snapshot = getattr(viewer, "_current_pyvista_camera_snapshot", lambda: None)()
        previous_pyvista_snapshot = deepcopy(
            live_pyvista_snapshot
            if live_pyvista_snapshot is not None
            else getattr(viewer, "_pyvista_camera_snapshot", None)
        )
        viewer.set_slice_planes(self._viewer_slice_surfaces(preview=preview))
        viewer.set_hide_atoms_in_front_of_planes(self._slice_hide_atoms.isChecked())
        current_pyvista_snapshot = getattr(
            viewer, "_current_pyvista_camera_snapshot", lambda: None
        )()
        snapshot_changed = (
            previous_pyvista_snapshot is not None
            and not _pyvista_camera_snapshots_match(
                current_pyvista_snapshot,
                previous_pyvista_snapshot,
            )
        )
        if previous_camera is not None and (
            getattr(viewer, "_camera_state", None) != previous_camera
            or getattr(viewer, "_limits", None) != previous_limits
            or snapshot_changed
        ):
            self._restore_viewer_camera(
                viewer, previous_camera, previous_limits, previous_pyvista_snapshot
            )

    def _add_muon_legend_entries(self, layout: QVBoxLayout, parent: QWidget) -> None:
        if not self._viewer_muons:
            empty = QLabel("No muons overlaid.", parent)
            empty.setObjectName("uepIsosurfaceBound")
            empty.setWordWrap(True)
            empty.setStyleSheet("background: transparent;")
            layout.addWidget(empty)
            return

        for entry in self._viewer_muons:
            row = QWidget(parent)
            row.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            swatch = _IsosurfaceLegendSwatch(row)
            swatch.set_color(self._viewer_muon_color(entry))
            row_layout.addWidget(swatch, 0, Qt.AlignmentFlag.AlignVCenter)

            label_parts = [entry.label]
            if entry.cluster_index is not None:
                label_parts.append(f"Site {int(entry.cluster_index)}")
            elif entry.source_label:
                label_parts.append(entry.source_label)
            label = QLabel(" | ".join(label_parts), row)
            label.setObjectName("uepIsosurfaceBound")
            label.setWordWrap(True)
            label.setStyleSheet("background: transparent;")
            row_layout.addWidget(label, 1, Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(row)

    def _refresh_viewer_muon_legend(self) -> None:
        self._clear_widget_layout(self._viewer_muon_legend_entries_layout)
        self._add_muon_legend_entries(
            self._viewer_muon_legend_entries_layout, self._viewer_muon_legend_entries
        )
        self._viewer_muon_legend_entries.adjustSize()

    def _refresh_slice_colorbar(self) -> None:
        figure = self._slice_colorbar_figure
        figure.clear()
        bounds = self._default_slice_bounds()
        if bounds is None or bounds[1] <= bounds[0]:
            self._slice_colorbar_canvas.draw_idle()
            return
        palette = current_palette()
        norm = mcolors.Normalize(vmin=float(bounds[0]), vmax=float(bounds[1]))
        axis = figure.add_subplot(1, 1, 1)
        colorbar = figure.colorbar(
            cm.ScalarMappable(norm=norm, cmap=colormaps["coolwarm"]),
            cax=axis,
            orientation="horizontal",
        )
        style_plot_axes(axis, grid=False)
        colorbar.outline.set_edgecolor(palette.border)
        colorbar.outline.set_linewidth(1.0)
        true_max = float(bounds[1])
        true_min = float(bounds[0])
        for state in self._slice_planes:
            if state.sample is None:
                continue
            true_min = min(true_min, float(state.sample.minimum_potential_ev))
            true_max = max(true_max, float(state.sample.maximum_potential_ev))
        lower_mark = "≤ " if float(bounds[0]) > true_min + 1e-9 else ""
        upper_mark = "≥ " if float(bounds[1]) < true_max - 1e-9 else ""
        colorbar.set_ticks(
            [float(bounds[0]), float(bounds[1])],
            labels=[f"{lower_mark}{bounds[0]:.3f}", f"{upper_mark}{bounds[1]:.3f}"],
        )
        tick_labels = axis.xaxis.get_ticklabels()
        tick_labels[0].set_horizontalalignment("left")
        tick_labels[1].set_horizontalalignment("right")
        axis.set_xlabel("Potential above minimum (eV)")
        self._slice_colorbar_canvas.draw_idle()

    def _open_add_slice_plane_dialog(self) -> None:
        dialog = _AddSlicePlaneDialog(
            selected_positions_ang=self._selected_slice_atom_positions(),
            selected_positions_fractional=self._selected_slice_atom_fractional_positions(),
            default_normal_components=self._slice_default_normal_components(),
            lattice_vectors=self._source.structure.lattice_vectors,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        plane = dialog.plane()
        if plane is None:
            return
        state = _SlicePlaneState(label=f"Plane {len(self._slice_planes) + 1}", plane=plane)
        self._clear_earlier_potential_message()
        self._slice_planes.append(state)
        self._slice_plane_numbering()
        self._resample_all_slice_planes()
        self._slice_plane_list.setCurrentRow(len(self._slice_planes) - 1)
        self._emit_state_changed()

    def _remove_slice_plane(self, index: int) -> None:
        self._clear_earlier_potential_message()
        del self._slice_planes[int(index)]
        self._slice_plane_numbering()
        self._resample_all_slice_planes()
        self._emit_state_changed()

    def _remove_selected_slice_plane(self) -> None:
        index = self._selected_slice_plane_index()
        if index is None or self._is_isosurface_busy():
            return
        if _confirm_remove_slice_plane(self, index + 1):
            self._remove_slice_plane(index)

    def _edit_selected_slice_plane(self) -> None:
        index = self._selected_slice_plane_index()
        if index is not None:
            self._edit_slice_plane(index)

    def _edit_slice_plane(self, index: int) -> None:
        if not 0 <= int(index) < len(self._slice_planes):
            return
        state = self._slice_planes[int(index)]
        if state.plane is None:
            return
        dialog = _EditSlicePlaneDialog(
            plane_number=int(index) + 1,
            plane=state.plane,
            lattice_vectors=self._source.structure.lattice_vectors,
            parent=self,
        )
        deleted = {"value": False}
        dialog.plane_deleted.connect(lambda: deleted.__setitem__("value", True))
        result = dialog.exec()
        if deleted["value"] or result == 2:
            self._remove_slice_plane(int(index))
            return
        if result != QDialog.DialogCode.Accepted:
            return
        try:
            state.plane = dialog.edited_plane()
        except UepError as exc:
            QMessageBox.warning(self, "Slice Plane", str(exc))
            return
        self._clear_earlier_potential_message()
        self._resample_all_slice_planes()
        self._emit_state_changed()

    def _on_slice_hide_atoms_toggled(self, checked: bool) -> None:
        self._potential_viewer.set_hide_atoms_in_front_of_planes(bool(checked))
        if not self._restoring_state:
            self._emit_state_changed()

    def _viewer_isosurface_surfaces(
        self,
        *,
        preview_layer_index: int | None = None,
        preview_mesh: UepIsosurfaceMesh | None = None,
    ) -> list[dict[str, object]]:
        surfaces: list[dict[str, object]] = []
        for index, layer in enumerate(self._isosurface_layers):
            mesh = layer.mesh
            if (
                preview_layer_index is not None
                and index == preview_layer_index
                and preview_mesh is not None
            ):
                mesh = preview_mesh
            if mesh is None:
                continue
            surfaces.append(
                {
                    "vertices": mesh.vertices_ang,
                    "faces": mesh.faces,
                    "color": layer.color,
                    "alpha": 0.3,
                }
            )
        return surfaces

    def _sync_isosurface_viewer_layers(self) -> None:
        self._potential_viewer.set_isosurfaces(self._viewer_isosurface_surfaces())
        self._refresh_isosurface_legend()

    def _refresh_isosurface_legend(self) -> None:
        self._clear_widget_layout(self._isosurface_legend_entries_layout)
        for layer in self._isosurface_layers:
            row = QWidget(self._isosurface_legend_entries)
            row.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            swatch = _IsosurfaceLegendSwatch(row)
            swatch.set_color(layer.color)
            row_layout.addWidget(swatch, 0, Qt.AlignmentFlag.AlignVCenter)

            level_text = "--" if layer.mesh is None else f"{float(layer.mesh.level_ev):.6f} eV"
            level_label = QLabel(level_text, row)
            level_label.setObjectName("uepIsosurfaceBound")
            level_label.setWordWrap(False)
            level_label.setStyleSheet("background: transparent;")
            row_layout.addWidget(level_label, 1, Qt.AlignmentFlag.AlignVCenter)
            self._isosurface_legend_entries_layout.addWidget(row)
        self._isosurface_legend_entries.adjustSize()

    def _refresh_isosurface_layer_tabs(self) -> None:
        if not self._isosurface_layers:
            self._isosurface_layers = [
                _IsosurfaceLayerState(label="Isosurface 1", key=self._new_isosurface_layer_key())
            ]
        self._active_isosurface_index = min(
            max(int(self._active_isosurface_index), 0), len(self._isosurface_layers) - 1
        )
        previous = self._isosurface_layer_tabs.blockSignals(True)
        try:
            if self._isosurface_layer_tabs.count() != len(self._isosurface_layers):
                while self._isosurface_layer_tabs.count():
                    self._isosurface_layer_tabs.removeTab(0)
                for index, _layer in enumerate(self._isosurface_layers):
                    self._isosurface_layer_tabs.addTab(str(index + 1))
            else:
                for index, _layer in enumerate(self._isosurface_layers):
                    expected = str(index + 1)
                    if self._isosurface_layer_tabs.tabText(index) != expected:
                        self._isosurface_layer_tabs.setTabText(index, expected)
            self._isosurface_layer_tabs.set_layer_colors(
                [layer.color for layer in self._isosurface_layers]
            )
            self._isosurface_layer_tabs.setCurrentIndex(self._active_isosurface_index)
        finally:
            self._isosurface_layer_tabs.blockSignals(previous)
        idle = not self._is_isosurface_busy()
        self._isosurface_delete_button.setEnabled(len(self._isosurface_layers) > 1 and idle)
        self._isosurface_add_button.setEnabled(idle)
        self._isosurface_layer_tabs.setEnabled(idle)
        self._refresh_isosurface_legend()

    def _apply_active_isosurface_layer_to_controls(self) -> None:
        layer = self._active_isosurface_layer()
        level = layer.level_ev
        if level is None:
            level = self._default_isosurface_level()
            layer.level_ev = level
        if level is not None:
            self._set_isosurface_level(level)
        self._set_isosurface_color(layer.color, emit_state=False)
        self._refresh_isosurface_layer_tabs()
        self._sync_isosurface_viewer_layers()

    def _update_active_isosurface_label(self) -> None:
        for index, layer in enumerate(self._isosurface_layers):
            layer.label = f"Isosurface {index + 1}"
        self._refresh_isosurface_layer_tabs()

    def _clear_pending_isosurface_request(self) -> None:
        self._isosurface_preview_timer.stop()
        self._pending_isosurface_level_ev = None
        self._pending_isosurface_layer_index = None
        self._pending_isosurface_rebuilds = []

    def _add_isosurface_layer(self) -> None:
        if self._is_isosurface_busy():
            return
        current_level = self._current_isosurface_level()
        if current_level is not None:
            self._active_isosurface_layer().level_ev = current_level
        self._clear_pending_isosurface_request()
        default_level = self._current_isosurface_level()
        if default_level is None:
            default_level = self._default_isosurface_level()
        self._isosurface_layers.append(
            _IsosurfaceLayerState(
                label=f"Isosurface {len(self._isosurface_layers) + 1}",
                key=self._new_isosurface_layer_key(),
                level_ev=default_level,
                color="#dc2626",
            )
        )
        self._active_isosurface_index = len(self._isosurface_layers) - 1
        self._update_active_isosurface_label()
        self._apply_active_isosurface_layer_to_controls()
        self._emit_state_changed()

    def _delete_current_isosurface_layer(self) -> None:
        if len(self._isosurface_layers) <= 1 or self._is_isosurface_busy():
            return
        removed = self._isosurface_layers[self._active_isosurface_index]
        if not workflow_table_utils.confirm_destructive(
            self,
            "Remove isosurface",
            f"Remove isosurface {self._active_isosurface_index + 1}?",
            "The surface leaves the viewer. Any muon drawn with it stays.",
            "Remove isosurface",
        ):
            return
        self._clear_pending_isosurface_request()
        self._unlink_isosurface_layer_from_its_muon(removed)
        del self._isosurface_layers[self._active_isosurface_index]
        self._active_isosurface_index = min(
            self._active_isosurface_index, len(self._isosurface_layers) - 1
        )
        self._update_active_isosurface_label()
        self._apply_active_isosurface_layer_to_controls()
        self._refresh_viewer_muon_dialog()
        self._emit_state_changed()

    def _on_isosurface_tab_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._isosurface_layers):
            return
        if self._is_isosurface_busy():
            self._refresh_isosurface_layer_tabs()
            return
        current_level = self._current_isosurface_level()
        if current_level is not None and self._isosurface_layers:
            self._active_isosurface_layer().level_ev = current_level
        self._clear_pending_isosurface_request()
        self._active_isosurface_index = int(index)
        self._apply_active_isosurface_layer_to_controls()
        if not self._restoring_state:
            self._emit_state_changed()

    def _on_analysis_tab_changed(self, _index: int) -> None:
        self._ensure_potential_grid()
        self._emit_state_changed()

    def _ensure_potential_grid(self) -> None:
        """Compute the grid the first time the Potential tab is on screen for this factor."""

        if (
            self._potential_grid_raw_cache is None
            and not self._shutting_down
            and self.isVisible()
            and self._tabs.currentWidget() is self._potential_tab
        ):
            self._start_isosurface_grid_calculation()

    def _apply_potential_grid_cache(self, grid: UepPotentialGrid) -> None:
        self._potential_grid_raw_cache = grid
        self._potential_minimum_ev = float(np.min(grid.potential_total_ev))
        self._potential_grid_cache = potential_above_minimum(grid)
        self._potential_level_maximum_ev = float(
            np.max(self._potential_grid_cache.potential_total_ev)
        )
        self._potential_grid_live_preview_cache = preview_potential_grid(self._potential_grid_cache)
        self._potential_level_ceiling_ev = potential_level_ceiling(self._potential_grid_cache)
        self._move_saved_levels_above_minimum()

        bounds = self._isosurface_level_bounds()
        self._isosurface_level_input.setRange(bounds[0], bounds[1], keep_text=True)
        self._update_isosurface_bounds()
        saturation = self._default_slice_bounds()
        self._set_slice_bounds(saturation[0], saturation[1], emit_state=False)
        self._resample_all_slice_planes()
        self._refresh_isosurface_controls()

    def _move_saved_levels_above_minimum(self) -> None:
        if not self._saved_levels_on_own_scale:
            return
        self._saved_levels_on_own_scale = False
        minimum = self._potential_minimum_ev
        for layer in self._isosurface_layers:
            if layer.level_ev is not None:
                layer.level_ev = float(layer.level_ev) - minimum
        if self._slice_saturation_bounds is None:
            return
        lower = self._slice_saturation_bounds[0] - minimum
        upper = self._slice_saturation_bounds[1] - minimum
        top = self._isosurface_level_bounds()[1]
        # The whole range was the only default such a window had, not a choice.
        whole_range = lower <= 1e-9 and upper >= top - 1e-9
        self._slice_saturation_bounds = None if whole_range else (lower, upper)

    def _invalidate_potential_grid_cache(self) -> None:
        self._clear_pending_isosurface_request()
        self._potential_grid_raw_cache = None
        self._potential_grid_cache = None
        self._potential_grid_live_preview_cache = None
        self._slice_saturation_bounds = None
        for layer in self._isosurface_layers:
            layer.mesh = None
        for state in self._slice_planes:
            state.sample = None
        self._sync_potential_viewer()
        self._update_isosurface_bounds()
        self._refresh_slice_colorbar()
        self._refresh_isosurface_controls()

    def _refresh_isosurface_controls(self) -> None:
        grid_ready = self._potential_grid_cache is not None
        grid_busy = self._isosurface_grid_thread is not None
        level_controls_enabled = grid_ready and not grid_busy
        self._isosurface_export_button.setEnabled(grid_ready and not self._is_isosurface_busy())
        self._isosurface_slider.setEnabled(level_controls_enabled)
        self._isosurface_level_input.setEnabled(level_controls_enabled)
        self._refresh_isosurface_layer_tabs()
        self._refresh_slice_controls()
        self._refresh_gw_factor_controls()
        self._refresh_viewer_muon_dialog()

    def _is_isosurface_busy(self) -> bool:
        return (
            self._isosurface_grid_thread is not None
            or self._isosurface_mesh_thread is not None
            or bool(self._pending_isosurface_rebuilds)
        )

    def _current_isosurface_level(self) -> float | None:
        bounds = self._isosurface_level_bounds()
        layer = self._active_isosurface_layer()
        if bounds is None:
            return None if layer.level_ev is None else float(layer.level_ev)
        editor_level = float(self._isosurface_level_input.value())
        return min(max(editor_level, bounds[0]), bounds[1])

    def _update_isosurface_bounds(self) -> None:
        if self._potential_grid_cache is None:
            self._isosurface_min_label.setText("")
            self._isosurface_max_label.setText("")
            return
        self._isosurface_min_label.setText("0 eV")
        self._isosurface_max_label.setText(f"{self._potential_level_ceiling_ev:.2f} eV")

    def _update_isosurface_color_button(self) -> None:
        self._isosurface_color_button.setText("")
        self._isosurface_color_button.setStyleSheet(
            "min-width: 64px; "
            f"border: 1px solid rgba(0, 0, 0, 0.14); background: {self._active_isosurface_layer().color}; "
            "color: #ffffff;"
        )

    def _set_isosurface_color(self, color: str, *, emit_state: bool) -> None:
        qcolor = QColor(str(color))
        if not qcolor.isValid():
            return
        normalized = qcolor.name()
        layer = self._active_isosurface_layer()
        changed = normalized != layer.color
        if changed:
            layer.color = normalized
        self._update_isosurface_color_button()
        if changed:
            self._refresh_isosurface_layer_tabs()
            self._sync_isosurface_viewer_layers()
        if changed and emit_state and not self._restoring_state:
            self._emit_state_changed()

    def _choose_isosurface_color(self) -> None:
        selected = QColorDialog.getColor(
            QColor(self._active_isosurface_layer().color), self, "Choose Isosurface Color"
        )
        if not selected.isValid():
            return
        self._set_isosurface_color(selected.name(), emit_state=True)

    def _set_isosurface_level(self, level: float) -> None:
        bounds = self._isosurface_level_bounds()
        numeric = float(level)
        if bounds is not None:
            numeric = min(max(numeric, bounds[0]), bounds[1])
            previous = self._isosurface_slider.blockSignals(True)
            self._isosurface_slider.setValue(
                self._level_to_slider(numeric, self._isosurface_slider.maximum())
            )
            self._isosurface_slider.blockSignals(previous)
            self._isosurface_level_input.setRange(bounds[0], bounds[1], keep_text=True)
        previous = self._isosurface_level_input.blockSignals(True)
        self._isosurface_level_input.setValue(numeric)
        self._isosurface_level_input.blockSignals(previous)
        self._active_isosurface_layer().level_ev = numeric

    def _sync_isosurface_level_editor_from_slider(self) -> None:
        if self._potential_grid_cache is None:
            return
        fraction = float(self._isosurface_slider.value()) / float(
            max(1, self._isosurface_slider.maximum())
        )
        level = fraction * self._potential_level_ceiling_ev
        previous = self._isosurface_level_input.blockSignals(True)
        self._isosurface_level_input.setValue(level)
        self._isosurface_level_input.blockSignals(previous)
        self._active_isosurface_layer().level_ev = float(level)

    def _request_isosurface_from_editor(self) -> None:
        if self._potential_grid_cache is None:
            return
        self._set_isosurface_level(self._isosurface_level_input.value())
        self._queue_isosurface_request(self._isosurface_level_input.value(), immediate=True)
        if not self._restoring_state:
            self._emit_state_changed()

    def _start_isosurface_grid_calculation(self) -> None:
        if self._is_isosurface_busy():
            return
        self._clear_pending_isosurface_request()
        self._potential_grid_live_preview_cache = None

        thread = _IsosurfaceGridThread(self._field)
        thread.setParent(self)
        thread.grid_ready.connect(self._on_isosurface_grid_ready)
        thread.failed.connect(self._on_isosurface_grid_failed)
        thread.finished.connect(self._on_isosurface_grid_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._isosurface_grid_thread = thread
        self._set_potential_status("Calculating the potential…", kind="busy")
        self._refresh_isosurface_controls()
        thread.start()

    def _on_isosurface_grid_thread_finished(self) -> None:
        self._isosurface_grid_thread = None
        if not self._shutting_down:
            self._refresh_isosurface_controls()

    def _on_isosurface_mesh_thread_finished(self) -> None:
        self._isosurface_mesh_thread = None
        if self._shutting_down:
            return
        if self._pending_isosurface_rebuilds and self._potential_grid_cache is not None:
            self._start_next_isosurface_rebuild()
            return
        self._refresh_isosurface_controls()
        self._dispatch_pending_isosurface_request()

    def _on_isosurface_grid_ready(self, grid: object) -> None:
        if self._shutting_down:
            return
        if not isinstance(grid, UepPotentialGrid):
            self._on_isosurface_grid_failed("Unexpected isosurface grid payload.")
            return
        self._set_potential_status()
        self._apply_potential_grid_cache(grid)
        top = self._isosurface_level_bounds()[1]
        rebuilds: list[tuple[str, float]] = []
        for layer in self._isosurface_layers:
            level = layer.level_ev
            if level is None:
                level = self._default_isosurface_level()
            # A new width factor can take the grid's top below a level kept from the last.
            layer.level_ev = min(max(float(level), 0.0), top)
            rebuilds.append((layer.key, layer.level_ev))
        self._pending_isosurface_rebuilds = rebuilds
        active_level = self._isosurface_layers[self._active_isosurface_index].level_ev
        self._set_isosurface_level(float(active_level))
        self._start_next_isosurface_rebuild()

    def _on_isosurface_grid_failed(self, message: str) -> None:
        self._pending_isosurface_rebuilds = []
        self._set_potential_status(
            f"The potential could not be calculated: {message}", kind="error"
        )
        self._refresh_isosurface_controls()

    def _start_next_isosurface_rebuild(self) -> None:
        if self._isosurface_mesh_thread is not None:
            return
        while self._pending_isosurface_rebuilds:
            layer_key, level = self._pending_isosurface_rebuilds.pop(0)
            layer_index = self._find_isosurface_layer_index_by_key(layer_key)
            if layer_index is not None:
                self._start_isosurface_mesh_calculation(float(level), layer_index=layer_index)
                return
        self._refresh_isosurface_controls()

    def _start_isosurface_mesh_calculation(
        self, level_ev: float, *, layer_index: int | None = None
    ) -> None:
        if self._potential_grid_cache is None or self._isosurface_mesh_thread is not None:
            return
        layer_index = int(self._active_isosurface_index if layer_index is None else layer_index)
        self._isosurface_request_id += 1
        self._mesh_request_layer_key = self._isosurface_layers[layer_index].key
        if not self._pending_isosurface_rebuilds:
            self._clear_earlier_potential_message()
        thread = _IsosurfaceMeshThread(
            self._potential_grid_cache,
            level_ev=level_ev,
            request_id=self._isosurface_request_id,
            layer_index=layer_index,
        )
        thread.setParent(self)
        thread.mesh_ready.connect(self._on_isosurface_mesh_ready)
        thread.failed.connect(self._on_isosurface_mesh_failed)
        thread.finished.connect(self._on_isosurface_mesh_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._isosurface_mesh_thread = thread
        self._isosurface_layers[layer_index].level_ev = float(level_ev)
        self._refresh_isosurface_controls()
        thread.start()

    def _queue_isosurface_request(self, level_ev: float, *, immediate: bool) -> None:
        self._pending_isosurface_level_ev = float(level_ev)
        self._pending_isosurface_layer_index = int(self._active_isosurface_index)
        if self._is_isosurface_busy():
            return
        if immediate:
            self._isosurface_preview_timer.stop()
            self._dispatch_pending_isosurface_request()
            return
        self._isosurface_preview_timer.start()

    def _schedule_isosurface_from_slider(self) -> None:
        if self._restoring_state or self._potential_grid_cache is None:
            return
        level = self._current_isosurface_level()
        if level is None:
            return
        self._render_live_isosurface_preview(level)
        self._queue_isosurface_request(level, immediate=False)

    def _render_live_isosurface_preview(self, level_ev: float) -> None:
        live_grid = self._potential_grid_live_preview_cache
        if live_grid is None:
            return
        try:
            mesh = extract_isosurface(live_grid, level_ev=float(level_ev))
        except UepError:
            return
        self._potential_viewer.set_isosurfaces(
            self._viewer_isosurface_surfaces(
                preview_layer_index=self._active_isosurface_index,
                preview_mesh=mesh,
            )
        )

    def _dispatch_pending_isosurface_request(self) -> None:
        if self._potential_grid_cache is None or self._is_isosurface_busy():
            return
        level = self._pending_isosurface_level_ev
        layer_index = self._pending_isosurface_layer_index
        if level is None:
            return
        if layer_index is None or layer_index != self._active_isosurface_index:
            self._clear_pending_isosurface_request()
            return
        if self._isosurface_slider.isSliderDown():
            self._isosurface_preview_timer.start()
            return
        self._clear_pending_isosurface_request()
        self._set_isosurface_level(level)
        self._start_isosurface_mesh_calculation(level)

    def _request_isosurface_from_slider(self) -> None:
        if self._potential_grid_cache is None:
            return
        level = self._current_isosurface_level()
        if level is None:
            return
        self._queue_isosurface_request(level, immediate=True)

    def _export_potential_grid_xsf(self) -> None:
        """Write the potential on its own scale, whatever the levels on screen read."""

        grid = self._potential_grid_raw_cache
        if grid is None:
            return
        default_name = f"{self._source.den_fmt_path.stem}_potential.xsf"
        default_path = self._source.den_fmt_path.with_name(default_name)
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export UEP Potential Grid",
            str(default_path),
            "XSF Files (*.xsf)",
        )
        if not selected_path:
            return
        try:
            exported_path = write_potential_grid_xsf(
                selected_path,
                structure=self._source.structure,
                potential=grid,
            )
        except (OSError, UepError) as exc:
            QMessageBox.critical(self, "Export XSF", f"Failed to export XSF: {exc}")
            return
        self._set_potential_status(f"Exported {exported_path.name}", kind="exported")
        self._emit_state_changed()

    def _render_isosurface_mesh_for_layer(
        self, mesh: UepIsosurfaceMesh, *, layer_index: int
    ) -> None:
        if not 0 <= int(layer_index) < len(self._isosurface_layers):
            return
        layer = self._isosurface_layers[int(layer_index)]
        layer.mesh = mesh
        layer.level_ev = float(mesh.level_ev)
        asked_for_another = self._pending_isosurface_level_ev is not None
        if int(layer_index) == self._active_isosurface_index and not asked_for_another:
            self._set_isosurface_level(mesh.level_ev)
        self._sync_isosurface_viewer_layers()

    def _on_isosurface_mesh_ready(self, request_id: int, layer_index: int, mesh: object) -> None:
        if int(request_id) != self._isosurface_request_id or not isinstance(
            mesh, UepIsosurfaceMesh
        ):
            return
        current_index = self._find_isosurface_layer_index_by_key(self._mesh_request_layer_key)
        if current_index is None:
            return
        self._render_isosurface_mesh_for_layer(mesh, layer_index=current_index)
        self._emit_state_changed()

    def _on_isosurface_mesh_failed(self, request_id: int, layer_index: int, message: str) -> None:
        if int(request_id) != self._isosurface_request_id:
            return
        # No surface at this level, as at the very bottom of the range: the last one is gone too.
        failed_index = self._find_isosurface_layer_index_by_key(self._mesh_request_layer_key)
        if failed_index is not None:
            self._isosurface_layers[failed_index].mesh = None
            self._sync_isosurface_viewer_layers()
        self._set_potential_status(
            f"The isosurface could not be extracted: {message}", kind="error"
        )
        self._emit_state_changed()

    def _show_active_gw_factor(self) -> None:
        """Show the factor in use, even one saved outside the range that can be typed.

        A saved factor keeps its value: the window never changes the potential a project's
        search ran on without being asked.
        """

        active = float(self._source.gw_factor)
        field = self._gw_factor_input
        previous = field.blockSignals(True)
        field.setRange(min(_GW_FACTOR_MINIMUM, active), max(_GW_FACTOR_MAXIMUM, active))
        field.setValue(active)
        field.blockSignals(previous)
        self._refresh_gw_factor_warning()
        self._refresh_gw_factor_controls()

    def _refresh_gw_factor_warning(self) -> None:
        active = float(self._source.gw_factor)
        messages: list[str] = []
        if not self._gw_factor_input.hasAcceptableInput():
            messages.append(
                f"Enter a width factor from {_GW_FACTOR_MINIMUM:g} to {_GW_FACTOR_MAXIMUM:g}."
            )
        if not _GW_FACTOR_MINIMUM <= active <= _GW_FACTOR_MAXIMUM:
            messages.append(
                f"This job was saved with a width factor of {active:g}, outside "
                f"{_GW_FACTOR_MINIMUM:g} to {_GW_FACTOR_MAXIMUM:g}. It stays in use until you change it."
            )
        resolution = ionic_gaussian_resolution(self._field)
        if resolution < IONIC_GAUSSIAN_MIN_RESOLUTION:
            messages.append(
                f"The density grid does not resolve the ionic Gaussians at this factor "
                f"(σ·Gmax = {resolution:.2f}, below {IONIC_GAUSSIAN_MIN_RESOLUTION:g}), so the "
                "potential rings around the ions. Use a smaller factor or a finer grid."
            )
        self._gw_factor_warning.setText(" ".join(messages))
        self._gw_factor_warning.setVisible(bool(messages))

    def _on_gw_factor_return_pressed(self) -> None:
        self._gw_factor_commit_requested = True
        self._apply_gw_factor_change(requested_value=float(self._gw_factor_input.value()))

    def _on_gw_factor_editing_finished(self) -> None:
        if self._gw_factor_commit_requested:
            self._gw_factor_commit_requested = False
            return
        self._show_active_gw_factor()

    def _refresh_gw_factor_controls(self) -> None:
        busy = self._isosurface_grid_thread is not None or (
            self._isosurface_mesh_thread is not None
            or bool(self._pending_isosurface_rebuilds)
            or self._batch_search_thread is not None
        )
        self._gw_factor_input.setEnabled(not busy)

    def _clear_probe_line_plane_outputs(self) -> None:
        self._latest_probe = None
        self._probe_output.clear()

        self._latest_line_sample = None
        self._line_data.clear()
        self._line_figure.clear()
        self._line_canvas.draw_idle()

        self._latest_plane_sample = None
        self._plane_data.clear()
        self._plane_figure.clear()
        self._plane_canvas.draw_idle()

    def _clear_batch_results(self) -> None:
        self._clear_clustered_viewer_muons()
        self._latest_batch_results = []
        self._latest_batch_summary = None
        self._batch_minima_distances = None
        self._batch_item_by_key = {}
        self._batch_table.clear()
        self._batch_table.setHeaderLabels(_BATCH_TABLE_HEADERS)
        self._update_batch_table_column_layout()
        self._batch_summary.clear()
        self._batch_viewer.set_structure(self._viewer_host_structure)
        self._batch_viewer_summary.setText(_BATCH_VIEWER_IDLE_TEXT)
        self._set_batch_progress_idle()
        self._refresh_viewer_muon_dialog()

    def _set_batch_log(self, text: str) -> None:
        self._batch_summary.setPlainText(text)

    def _append_batch_log(self, text: str) -> None:
        message = text.strip()
        if not message:
            return
        existing = self._batch_summary.toPlainText().strip()
        self._batch_summary.setPlainText(f"{existing}\n{message}" if existing else message)

    def _batch_table_position(self, result: UepOptimizationResult) -> tuple[float, float, float]:
        return position_in_source_cell(result, self._grouping_structure())

    def _grouping_structure(self) -> Any:
        """The structure a search is grouped on: the source job's input, whatever its supercell."""

        return (
            self._source_structure if self._source_structure is not None else self._source.structure
        )

    def _group_batch_results(self, results: list[UepOptimizationResult]) -> UepBatchSummary | None:
        """Group ``results`` at the current tolerance; ``None``, with a log line, if that fails."""

        structure = self._grouping_structure()
        try:
            if self._batch_minima_distances is None:
                self._batch_minima_distances = measure_uep_minima(
                    results, source_structure=structure
                )
            return group_uep_minima(
                results,
                source_structure=structure,
                tolerance_ang=self._grouping_tolerance(),
                distances=self._batch_minima_distances,
            )
        except Exception as exc:  # noqa: BLE001 - the results stay; the log says why.
            self._append_batch_log(f"Grouping failed: {exc}")
            return None

    def _grouping_tolerance(self) -> float:
        """The tolerance shown, or the one in use while the field holds no valid number."""

        if self._batch_grouping_tolerance.hasAcceptableInput():
            return self._batch_grouping_tolerance.value()
        if self._latest_batch_summary is not None:
            return self._latest_batch_summary.threshold_ang
        return DEFAULT_GROUPING_TOLERANCE_ANG

    def _selection_after_regrouping(self, keys: list[str], summary: UepBatchSummary) -> list[str]:
        """Carry selected rows to the new groups, which may be numbered differently.

        A selected site stays selected in whichever group holds it now, and a selected
        group becomes the group that holds its representative.
        """

        group_of_result = {
            int(member): cluster.cluster_index
            for cluster in summary.clusters
            for member in cluster.member_indexes
        }
        old_groups = (
            {}
            if self._latest_batch_summary is None
            else {
                self._batch_cluster_key(cluster.cluster_index): cluster.representative_index
                for cluster in self._latest_batch_summary.clusters
            }
        )
        carried: list[str] = []
        for key in keys:
            _, _, site = key.partition(":site:")
            result_index = int(site) if site else old_groups.get(key)
            group = group_of_result.get(result_index)
            if group is None:
                continue
            carried.append(
                self._batch_site_key(group, result_index)
                if site
                else self._batch_cluster_key(group)
            )
        return list(dict.fromkeys(carried))

    def _set_grouping_tolerance(self, value: float) -> None:
        blocked = self._batch_grouping_tolerance.blockSignals(True)
        self._batch_grouping_tolerance.setValue(float(value))
        self._batch_grouping_tolerance.blockSignals(blocked)

    def _on_grouping_tolerance_edited(self, _text: str = "") -> None:
        """Regroup at the tolerance being typed; the state follows when the edit settles."""

        if not self._latest_batch_results or self._batch_search_thread is not None:
            return
        # A number on its way to a valid one, such as the ``0.0`` of ``0.05``.
        if not self._batch_grouping_tolerance.hasAcceptableInput():
            return
        summary = self._group_batch_results(self._latest_batch_results)
        if summary is None or summary == self._latest_batch_summary:
            return
        selected_keys = self._selection_after_regrouping(self._selected_batch_keys(), summary)
        expanded_keys = self._selection_after_regrouping(self._expanded_batch_keys(), summary)
        self._populate_batch_results(
            self._latest_batch_results,
            summary,
            summary_text=uep_search_log(self._batch_summary.toPlainText(), summary),
            selected_keys=selected_keys,
            expanded_keys=expanded_keys,
        )
        self._grouping_settled_timer.start()

    def _on_grouping_tolerance_finished(self) -> None:
        self._on_grouping_tolerance_edited()
        self.flush_pending_state()

    def flush_pending_state(self) -> None:
        """Emit a regrouped state now, for a save, rather than when the edit settles."""

        if self._grouping_settled_timer.isActive():
            self._grouping_settled_timer.stop()
            self._emit_state_changed()

    def _apply_gw_factor_change(self, requested_value: float | None = None) -> None:
        gw_factor = float(
            self._gw_factor_input.value() if requested_value is None else requested_value
        )
        if np.isclose(gw_factor, float(self._source.gw_factor)):
            self._show_active_gw_factor()
            return
        if not self._confirm_discard_batch_results(
            "Change the Gaussian width factor",
            f"Change the Gaussian width factor to {gw_factor:g} and discard the search results?",
            "Change width factor",
        ):
            self._show_active_gw_factor()
            return

        try:
            new_source = load_uep_source(
                den_fmt_path=self._den_fmt_path,
                castep_path=self._castep_path,
                cell_path=self._cell_path,
                gw_factor=gw_factor,
            )
            new_field = UepField(new_source)
        except (OSError, UepError, ValueError) as exc:
            QMessageBox.warning(self, "UEP analysis unavailable", str(exc))
            self._show_active_gw_factor()
            return

        self._source = new_source
        self._field = new_field
        self._show_active_gw_factor()
        self._viewer_host_structure = self._build_viewer_host_structure()
        self._viewer_structure_signature = None
        self._populate_summary()
        self._clear_probe_line_plane_outputs()
        self._clear_batch_results()
        self._invalidate_potential_grid_cache()
        self._ensure_potential_grid()
        self._emit_state_changed()

    def _populate_summary(self) -> None:
        pseudo_species = sorted({entry.species for entry in self._source.pseudopotentials})
        lines = [
            f"Source job: {self._source_job_label}",
            f"Density file: {self._source.den_fmt_path}",
            f"CASTEP file: {self._source.castep_path}",
            f"CELL fallback: {self._source.cell_path if self._source.cell_path is not None else 'not used'}",
            f"Atoms in source structure: {len(self._source.structure.species)}",
            f"Pseudopotential species recovered: {', '.join(pseudo_species)}",
            f"Gaussian width factor (gw_factor): {self._source.gw_factor:.3f}",
            (
                "Mulliken spin moments recovered from CASTEP output: yes"
                if self._source.structure.atomic_spins is not None
                else "Mulliken spin moments recovered from CASTEP output: no"
            ),
        ]
        if self._source.neutrality_warning:
            lines.append(f"Neutrality note: {self._source.neutrality_warning}")
        lines.append("Required UEP source files are present: `.den_fmt` and `.castep`.")
        self._summary_text.setPlainText("\n".join(lines))

    def _fractional_to_cartesian(
        self, fractional: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        cart = lattice_frac_to_cart(fractional, self._source.structure.lattice_vectors)
        return (float(cart[0]), float(cart[1]), float(cart[2]))

    @staticmethod
    def _tuple3(values: object) -> tuple[float, float, float] | None:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            return None
        try:
            return (float(values[0]), float(values[1]), float(values[2]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _matrix3(
        values: object,
    ) -> (
        tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
        | None
    ):
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            return None
        rows: list[tuple[float, float, float]] = []
        for row in values:
            parsed = UepAnalysisDialog._tuple3(row)
            if parsed is None:
                return None
            rows.append(parsed)
        return (rows[0], rows[1], rows[2])

    @classmethod
    def _deserialize_probe(cls, payload: object) -> UepProbeResult | None:
        if not isinstance(payload, Mapping):
            return None
        position = cls._tuple3(payload.get("position_ang"))
        fractional = cls._tuple3(payload.get("fractional_position"))
        gradient_total = cls._tuple3(payload.get("gradient_total_ev_per_ang"))
        gradient_electronic = cls._tuple3(payload.get("gradient_electronic_ev_per_ang"))
        gradient_ionic = cls._tuple3(payload.get("gradient_ionic_ev_per_ang"))
        hessian_total = cls._matrix3(payload.get("hessian_total_ev_per_ang2"))
        hessian_electronic = cls._matrix3(payload.get("hessian_electronic_ev_per_ang2"))
        hessian_ionic = cls._matrix3(payload.get("hessian_ionic_ev_per_ang2"))
        classification = classification_from_dict(payload.get("classification"))
        if None in (
            position,
            fractional,
            gradient_total,
            gradient_electronic,
            gradient_ionic,
            hessian_total,
            hessian_electronic,
            hessian_ionic,
            classification,
        ):
            return None
        try:
            return UepProbeResult(
                position_ang=position,
                fractional_position=fractional,
                potential_total_ev=float(payload.get("potential_total_ev", 0.0)),
                potential_electronic_ev=float(payload.get("potential_electronic_ev", 0.0)),
                potential_ionic_ev=float(payload.get("potential_ionic_ev", 0.0)),
                gradient_total_ev_per_ang=gradient_total,
                gradient_electronic_ev_per_ang=gradient_electronic,
                gradient_ionic_ev_per_ang=gradient_ionic,
                hessian_total_ev_per_ang2=hessian_total,
                hessian_electronic_ev_per_ang2=hessian_electronic,
                hessian_ionic_ev_per_ang2=hessian_ionic,
                classification=classification,
                zero_point_energy_ev=(
                    None
                    if payload.get("zero_point_energy_ev") is None
                    else float(payload.get("zero_point_energy_ev"))
                ),
                total_quantum_energy_ev=(
                    None
                    if payload.get("total_quantum_energy_ev") is None
                    else float(payload.get("total_quantum_energy_ev"))
                ),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tuple_of_floats(values: object) -> tuple[float, ...] | None:
        if not isinstance(values, (list, tuple)):
            return None
        try:
            return tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _matrix_of_floats(cls, values: object) -> tuple[tuple[float, ...], ...] | None:
        if not isinstance(values, (list, tuple)):
            return None
        rows: list[tuple[float, ...]] = []
        for row in values:
            parsed = cls._tuple_of_floats(row)
            if parsed is None:
                return None
            rows.append(parsed)
        return tuple(rows)

    @classmethod
    def _deserialize_line_sample(cls, payload: object):
        if not isinstance(payload, Mapping):
            return None
        fields = (
            cls._tuple_of_floats(payload.get("distance_ang")),
            cls._tuple_of_floats(payload.get("potential_total_ev")),
            cls._tuple_of_floats(payload.get("potential_electronic_ev")),
            cls._tuple_of_floats(payload.get("potential_ionic_ev")),
            cls._tuple_of_floats(payload.get("charge_total_e_per_ang3")),
            cls._tuple_of_floats(payload.get("charge_electronic_e_per_ang3")),
            cls._tuple_of_floats(payload.get("charge_ionic_e_per_ang3")),
        )
        if any(field is None for field in fields):
            return None
        return UepLineSample(
            distance_ang=fields[0],
            potential_total_ev=fields[1],
            potential_electronic_ev=fields[2],
            potential_ionic_ev=fields[3],
            charge_total_e_per_ang3=fields[4],
            charge_electronic_e_per_ang3=fields[5],
            charge_ionic_e_per_ang3=fields[6],
        )

    @classmethod
    def _deserialize_plane_sample(cls, payload: object):
        if not isinstance(payload, Mapping):
            return None
        fields = (
            cls._matrix_of_floats(payload.get("x_coordinates_ang")),
            cls._matrix_of_floats(payload.get("y_coordinates_ang")),
            cls._matrix_of_floats(payload.get("potential_total_ev")),
            cls._matrix_of_floats(payload.get("potential_electronic_ev")),
            cls._matrix_of_floats(payload.get("potential_ionic_ev")),
            cls._matrix_of_floats(payload.get("charge_total_e_per_ang3")),
            cls._matrix_of_floats(payload.get("charge_electronic_e_per_ang3")),
            cls._matrix_of_floats(payload.get("charge_ionic_e_per_ang3")),
        )
        if any(field is None for field in fields):
            return None
        return UepPlaneSample(
            x_coordinates_ang=fields[0],
            y_coordinates_ang=fields[1],
            potential_total_ev=fields[2],
            potential_electronic_ev=fields[3],
            potential_ionic_ev=fields[4],
            charge_total_e_per_ang3=fields[5],
            charge_electronic_e_per_ang3=fields[6],
            charge_ionic_e_per_ang3=fields[7],
        )

    def _render_probe_result(self, probe: UepProbeResult) -> None:
        self._latest_probe = probe
        self._probe_output.setPlainText(
            "\n".join(
                [
                    f"Cartesian position (A): {probe.position_ang}",
                    f"Fractional position: {probe.fractional_position}",
                    f"Total potential (eV): {probe.potential_total_ev:.8f}",
                    f"Electronic contribution (eV): {probe.potential_electronic_ev:.8f}",
                    f"Ionic contribution (eV): {probe.potential_ionic_ev:.8f}",
                    f"Gradient (eV/A): {probe.gradient_total_ev_per_ang}",
                    f"Hessian (eV/A^2): {probe.hessian_total_ev_per_ang2}",
                    f"Classification: {probe.classification.label}",
                    f"Hessian eigenvalues: {probe.classification.eigenvalues}",
                    "Zero-point energy (eV): "
                    + (
                        "n/a"
                        if probe.zero_point_energy_ev is None
                        else f"{probe.zero_point_energy_ev:.8f}"
                    ),
                    "Quantum-corrected total (eV): "
                    + (
                        "n/a"
                        if probe.total_quantum_energy_ev is None
                        else f"{probe.total_quantum_energy_ev:.8f}"
                    ),
                ]
            )
        )

    def _render_line_sample(self, sample) -> None:
        self._latest_line_sample = sample
        self._line_figure.clear()
        axis = self._line_figure.add_subplot(1, 1, 1)
        axis.plot(sample.distance_ang, sample.potential_total_ev, label="Total", color="#1d4ed8")
        axis.plot(
            sample.distance_ang, sample.potential_electronic_ev, label="Electronic", color="#0f766e"
        )
        axis.plot(sample.distance_ang, sample.potential_ionic_ev, label="Ionic", color="#b91c1c")
        axis.set_xlabel("Distance along line (A)")
        axis.set_ylabel("Potential (eV)")
        axis.legend()
        axis.grid(True, alpha=0.3)
        self._line_canvas.draw_idle()

        rows = ["distance_A,total_eV,electronic_eV,ionic_eV"]
        for index in range(min(25, len(sample.distance_ang))):
            rows.append(
                f"{sample.distance_ang[index]:.6f},{sample.potential_total_ev[index]:.8f},"
                f"{sample.potential_electronic_ev[index]:.8f},{sample.potential_ionic_ev[index]:.8f}"
            )
        self._line_data.setPlainText("\n".join(rows))

    def _render_plane_sample(self, sample: UepPlaneSample) -> None:
        self._latest_plane_sample = sample
        self._draw_plane(sample)
        grid = np.array(sample.potential_total_ev, dtype=float)
        self._plane_data.setPlainText(
            "\n".join(
                [
                    f"Grid shape: {grid.shape[0]} x {grid.shape[1]}",
                    f"Minimum potential (eV): {float(np.min(grid)):.8f}",
                    f"Maximum potential (eV): {float(np.max(grid)):.8f}",
                    f"Mean potential (eV): {float(np.mean(grid)):.8f}",
                ]
            )
        )

    def _populate_batch_results(
        self,
        results: list[UepOptimizationResult],
        summary: UepBatchSummary,
        *,
        summary_text: str | None = None,
        selected_keys: list[str] | tuple[str, ...] | None = None,
        legacy_selected_row: int | None = None,
        expanded_keys: list[str] | tuple[str, ...] = (),
    ) -> None:
        self._latest_batch_results = list(results)
        self._latest_batch_summary = summary
        site_number_map = relaxed_site_number_map(
            results,
            included=(index for cluster in summary.clusters for index in cluster.member_indexes),
        )
        self._batch_table.clear()
        self._batch_table.setHeaderLabels(_BATCH_TABLE_HEADERS)
        self._batch_item_by_key = {}
        for cluster in summary.clusters:
            sorted_member_indexes = self._batch_sorted_member_indexes(cluster)
            representative = results[cluster.representative_index]
            cluster_payload = {
                "kind": "cluster",
                "cluster": asdict(cluster),
                "representative": asdict(representative),
                "display_member_indexes": list(sorted_member_indexes),
            }
            cluster_item = QTreeWidgetItem(
                [
                    f"Site {cluster.cluster_index}",
                    str(cluster.size),
                    self._format_batch_fractional(cluster.representative_fractional_position[0]),
                    self._format_batch_fractional(cluster.representative_fractional_position[1]),
                    self._format_batch_fractional(cluster.representative_fractional_position[2]),
                    self._format_batch_energy(representative.classical_energy_ev),
                    self._format_batch_energy(representative.zero_point_energy_ev),
                    self._format_batch_energy(representative.total_energy_ev),
                ]
            )
            cluster_key = self._batch_cluster_key(cluster.cluster_index)
            cluster_item.setData(0, Qt.ItemDataRole.UserRole, cluster_payload)
            cluster_item.setData(0, Qt.ItemDataRole.UserRole + 1, cluster_key)
            self._batch_item_by_key[cluster_key] = cluster_item
            for site_number, result_index in enumerate(sorted_member_indexes, start=1):
                site = results[result_index]
                global_site_number = site_number_map.get(int(result_index), site_number)
                site_payload = {
                    "kind": "site",
                    "cluster": asdict(cluster),
                    "site": asdict(site),
                    "site_number": global_site_number,
                    "site_result_index": int(result_index),
                }
                position = self._batch_table_position(site)
                site_item = QTreeWidgetItem(
                    [
                        f"Start {global_site_number}",
                        "1",
                        self._format_batch_fractional(position[0]),
                        self._format_batch_fractional(position[1]),
                        self._format_batch_fractional(position[2]),
                        self._format_batch_energy(site.classical_energy_ev),
                        self._format_batch_energy(site.zero_point_energy_ev),
                        self._format_batch_energy(site.total_energy_ev),
                    ]
                )
                site_key = self._batch_site_key(cluster.cluster_index, result_index)
                site_item.setData(0, Qt.ItemDataRole.UserRole, site_payload)
                site_item.setData(0, Qt.ItemDataRole.UserRole + 1, site_key)
                self._batch_item_by_key[site_key] = site_item
                cluster_item.addChild(site_item)
            self._batch_table.addTopLevelItem(cluster_item)
            cluster_item.setExpanded(cluster_key in expanded_keys)
        if summary_text is not None:
            self._batch_summary.setPlainText(summary_text)
            # The end of the log says how the search went.
            self._batch_summary.moveCursor(QTextCursor.MoveOperation.End)
        self._batch_progress.setRange(0, max(len(results), 1))
        self._batch_progress.setValue(max(len(results), 1))
        set_progress_bar_state(self._batch_progress, "complete" if summary.clusters else "warning")
        self._batch_progress.setFormat(
            "Batch search complete" if summary.clusters else "No minima found"
        )
        self._update_batch_table_column_layout()
        if selected_keys or legacy_selected_row is not None:
            self._restore_batch_selection(keys=selected_keys, legacy_row=legacy_selected_row)
        else:
            self._batch_viewer.set_structure(self._viewer_host_structure)
            self._batch_viewer_summary.setText(_BATCH_VIEWER_IDLE_TEXT)
        self._relabel_viewer_muons()
        self._refresh_viewer_muon_dialog()

    def _relabel_viewer_muons(self) -> None:
        """Bring muons already in a viewer up to date with the groups their sites are in."""

        candidates = {entry.result_index: entry for entry in self._clustered_muon_candidates()}
        for entry in self._viewer_muons:
            candidate = candidates.get(entry.result_index)
            if entry.source_kind == "clustered" and candidate is not None:
                # A default label from before starts had their own name; a typed one stays.
                if _RETIRED_DEFAULT_MUON_LABEL.fullmatch(entry.label):
                    entry.label = candidate.label
                entry.source_label = candidate.cluster_label
                entry.display_fractional_position = candidate.display_fractional_position
                entry.cluster_index = int(candidate.cluster_index)
        if self._viewer_muons:
            self._sync_potential_viewer()

    def state_snapshot(self) -> dict[str, object]:
        selected_keys = self._selected_batch_keys()
        return {
            "gw_factor": float(self._source.gw_factor),
            "current_tab": int(self._tabs.currentIndex()),
            "probe_point": list(self._probe_point.values()),
            "latest_probe": None if self._latest_probe is None else asdict(self._latest_probe),
            "probe_output": self._probe_output.toPlainText(),
            "line_start": list(self._line_start.values()),
            "line_end": list(self._line_end.values()),
            "line_points": int(self._line_points.value()),
            "line_sample": (
                None if self._latest_line_sample is None else asdict(self._latest_line_sample)
            ),
            "line_data": self._line_data.toPlainText(),
            "plane_origin": list(self._plane_origin.values()),
            "plane_x": list(self._plane_x.values()),
            "plane_y": list(self._plane_y.values()),
            "plane_points_x": int(self._plane_points_x.value()),
            "plane_points_y": int(self._plane_points_y.value()),
            "plane_sample": (
                None if self._latest_plane_sample is None else asdict(self._latest_plane_sample)
            ),
            "plane_data": self._plane_data.toPlainText(),
            "isosurface_active_index": int(self._active_isosurface_index),
            "isosurface_layers": [
                {
                    "label": layer.label,
                    "key": layer.key,
                    "level_ev": None if layer.level_ev is None else float(layer.level_ev),
                    "color": layer.color,
                    "managed_muon_key": layer.managed_muon_key,
                }
                for layer in self._isosurface_layers
            ],
            # False only while levels restored on the potential's own scale wait for the grid.
            "isosurface_zero_minimum": not self._saved_levels_on_own_scale,
            "viewer_muons": [
                self._serialize_managed_viewer_muon(entry) for entry in self._viewer_muons
            ],
            "selected_viewer_muon_keys": list(self._selected_viewer_muon_keys),
            "slice_saturation_bounds": (
                None
                if self._slice_saturation_bounds is None
                else [
                    float(self._slice_saturation_bounds[0]),
                    float(self._slice_saturation_bounds[1]),
                ]
            ),
            "slice_hide_atoms_in_front": bool(self._slice_hide_atoms.isChecked()),
            "slice_planes": [
                {
                    "label": state.label,
                    "plane": (
                        None
                        if state.plane is None
                        else {
                            "anchor_ang": list(state.plane.anchor_ang),
                            "normal_ang": list(state.plane.normal_ang),
                            "basis_u_ang": list(state.plane.basis_u_ang),
                            "basis_v_ang": list(state.plane.basis_v_ang),
                            "extent_u_ang": list(state.plane.extent_u_ang),
                            "extent_v_ang": list(state.plane.extent_v_ang),
                            "points_u": int(state.plane.points_u),
                            "points_v": int(state.plane.points_v),
                        }
                    ),
                }
                for state in self._slice_planes
            ],
            "batch_count": int(self._batch_count.value()),
            "batch_min_muon_muon": float(self._batch_min_muon_muon.value()),
            "batch_min_muon_atom": float(self._batch_min_muon_atom.value()),
            "batch_max_attempts": int(self._batch_max_attempts.value()),
            "batch_optimization_tolerance_ev": float(self._batch_force_tolerance.value()),
            "batch_max_steps": int(self._batch_max_steps.value()),
            "batch_results": [asdict(result) for result in self._latest_batch_results],
            "batch_summary_data": (
                None if self._latest_batch_summary is None else asdict(self._latest_batch_summary)
            ),
            "batch_summary_text": self._batch_summary.toPlainText(),
            "batch_selected_keys": selected_keys,
        }

    def _emit_state_changed(self) -> None:
        if self._shutting_down:
            return
        self.state_changed.emit(self.state_snapshot())
        self._keep_visible_child_dialogs_in_front()

    def _keep_visible_child_dialogs_in_front(self) -> None:
        self._keep_dialog_in_front(self._viewer_muon_dialog)

    def _restore_batch_search(self, state: Mapping[str, object]) -> None:
        """Show a saved search and its groups, grouping it when it has none in this form."""

        raw_results = state.get("batch_results")
        results = (
            [
                result
                for entry in raw_results
                if (result := optimization_result_from_dict(entry)) is not None
            ]
            if isinstance(raw_results, list)
            else []
        )
        if not results:
            return
        text = state.get("batch_summary_text")
        if isinstance(text, str):
            self._batch_summary.setPlainText(text)
        summary = uep_batch_summary_from_dict(state.get("batch_summary_data"))
        if summary is not None and not self._summary_fits_results(summary, results):
            summary = None
        if summary is None:
            summary = self._group_batch_results(results)
        else:
            self._set_grouping_tolerance(summary.threshold_ang)
        if summary is None:
            return
        raw_selected_keys = state.get("batch_selected_keys")
        selected_row = state.get("batch_selected_row")
        log = saved_uep_search_log(self._batch_summary.toPlainText(), results, summary)
        self._populate_batch_results(
            results,
            summary,
            summary_text=log,
            selected_keys=(
                [str(value) for value in raw_selected_keys]
                if isinstance(raw_selected_keys, list)
                else None
            ),
            legacy_selected_row=selected_row if isinstance(selected_row, int) else None,
        )

    @staticmethod
    def _summary_fits_results(
        summary: UepBatchSummary, results: list[UepOptimizationResult]
    ) -> bool:
        return all(
            0 <= int(index) < len(results)
            for cluster in summary.clusters
            for index in cluster.member_indexes
        )

    def _restore_initial_state(self, state: Mapping[str, object] | None) -> None:
        if not isinstance(state, Mapping):
            return

        self._restoring_state = True
        try:
            for control, key in (
                (self._probe_point, "probe_point"),
                (self._line_start, "line_start"),
                (self._line_end, "line_end"),
                (self._plane_origin, "plane_origin"),
                (self._plane_x, "plane_x"),
                (self._plane_y, "plane_y"),
            ):
                values = self._tuple3(state.get(key))
                if values is not None:
                    control.set_values(values)

            for control, key in (
                (self._line_points, "line_points"),
                (self._plane_points_x, "plane_points_x"),
                (self._plane_points_y, "plane_points_y"),
                (self._batch_count, "batch_count"),
                (self._batch_max_attempts, "batch_max_attempts"),
                (self._batch_max_steps, "batch_max_steps"),
            ):
                raw = state.get(key)
                if raw is not None:
                    try:
                        control.setValue(int(raw))
                    except (TypeError, ValueError):
                        pass

            for control, key in (
                (self._batch_min_muon_muon, "batch_min_muon_muon"),
                (self._batch_min_muon_atom, "batch_min_muon_atom"),
                (self._batch_force_tolerance, "batch_optimization_tolerance_ev"),
            ):
                raw = state.get(key)
                if raw is not None:
                    try:
                        control.setValue(float(raw))
                    except (TypeError, ValueError):
                        pass

            # A window saved without ``Set minimum potential to zero`` holds its levels and
            # saturation on the potential's own scale; the grid moves them above its minimum.
            zero_minimum_raw = state.get("isosurface_zero_minimum")
            self._saved_levels_on_own_scale = not (
                bool(zero_minimum_raw) if isinstance(zero_minimum_raw, bool | int) else False
            )

            restored_layers: list[_IsosurfaceLayerState] = []
            raw_layers = state.get("isosurface_layers")
            if isinstance(raw_layers, list):
                for index, entry in enumerate(raw_layers):
                    if not isinstance(entry, Mapping):
                        continue
                    color_raw = entry.get("color", "#dc2626")
                    qcolor = QColor(str(color_raw))
                    restored_layers.append(
                        _IsosurfaceLayerState(
                            label=str(entry.get("label", f"Isosurface {index + 1}")),
                            key=str(entry.get("key", self._new_isosurface_layer_key())),
                            level_ev=self._optional_float(entry.get("level_ev")),
                            color=qcolor.name() if qcolor.isValid() else "#dc2626",
                            managed_muon_key=(
                                None
                                if entry.get("managed_muon_key") in (None, "")
                                else str(entry.get("managed_muon_key"))
                            ),
                        )
                    )
            if restored_layers:
                self._isosurface_layers = restored_layers
                raw_active_index = state.get("isosurface_active_index", 0)
                try:
                    self._active_isosurface_index = int(raw_active_index)
                except (TypeError, ValueError):
                    self._active_isosurface_index = 0
                self._active_isosurface_index = min(
                    max(self._active_isosurface_index, 0),
                    len(self._isosurface_layers) - 1,
                )
            else:
                self._isosurface_layers = [
                    _IsosurfaceLayerState(
                        label="Isosurface 1", key=self._new_isosurface_layer_key()
                    )
                ]
                self._active_isosurface_index = 0
                layer = self._active_isosurface_layer()
                color = state.get("isosurface_color")
                if isinstance(color, str):
                    qcolor = QColor(color)
                    if qcolor.isValid():
                        layer.color = qcolor.name()
                mesh = state.get("isosurface_mesh")
                layer.level_ev = self._optional_float(
                    mesh.get("level_ev") if isinstance(mesh, Mapping) else None
                )
                if layer.level_ev is None:
                    layer.level_ev = self._optional_float(state.get("isosurface_level_value"))
            self._update_active_isosurface_label()

            self._viewer_muons = self._restored_viewer_muons(state)
            self._set_selected_viewer_muon_keys(
                [
                    str(key)
                    for name in (
                        "selected_viewer_muon_keys",
                        "isosurface_selected_muon_keys",
                        "slice_selected_muon_keys",
                    )
                    if isinstance(state.get(name), list)
                    for key in state[name]
                ]
            )

            slice_bounds_raw = state.get("slice_saturation_bounds")
            if isinstance(slice_bounds_raw, (list, tuple)) and len(slice_bounds_raw) == 2:
                try:
                    self._slice_saturation_bounds = (
                        float(slice_bounds_raw[0]),
                        float(slice_bounds_raw[1]),
                    )
                except (TypeError, ValueError):
                    self._slice_saturation_bounds = None
            else:
                self._slice_saturation_bounds = None

            slice_planes_raw = state.get("slice_planes")
            restored_slice_planes: list[_SlicePlaneState] = []
            if isinstance(slice_planes_raw, list):
                for index, entry in enumerate(slice_planes_raw):
                    if not isinstance(entry, Mapping):
                        continue
                    restored_slice_planes.append(
                        _SlicePlaneState(
                            label=str(entry.get("label", f"Plane {index + 1}")),
                            plane=self._deserialize_slice_plane(entry.get("plane")),
                        )
                    )
            self._slice_planes = restored_slice_planes
            self._slice_plane_numbering()
            self._sync_custom_muon_counter()

            hide_atoms_raw = state.get("slice_hide_atoms_in_front")
            previous_hide = self._slice_hide_atoms.blockSignals(True)
            self._slice_hide_atoms.setChecked(bool(hide_atoms_raw))
            self._slice_hide_atoms.blockSignals(previous_hide)

            probe = self._deserialize_probe(state.get("latest_probe"))
            if probe is not None:
                self._render_probe_result(probe)
            else:
                text = state.get("probe_output")
                if isinstance(text, str):
                    self._probe_output.setPlainText(text)

            line_sample = self._deserialize_line_sample(state.get("line_sample"))
            if line_sample is not None:
                self._render_line_sample(line_sample)
            else:
                text = state.get("line_data")
                if isinstance(text, str):
                    self._line_data.setPlainText(text)

            plane_sample = self._deserialize_plane_sample(state.get("plane_sample"))
            if plane_sample is not None:
                self._render_plane_sample(plane_sample)
            else:
                text = state.get("plane_data")
                if isinstance(text, str):
                    self._plane_data.setPlainText(text)

            self._apply_active_isosurface_layer_to_controls()
            self._refresh_isosurface_controls()
            self._refresh_slice_plane_list()

            self._restore_batch_search(state)
            self._sync_potential_viewer()
            self._batch_advanced_toggle.setChecked(
                self._batch_max_attempts.value() != _DEFAULT_MAX_START_ATTEMPTS
                or self._batch_max_steps.value() != DEFAULT_OPTIMIZER_MAX_STEPS
                or not np.isclose(
                    self._batch_force_tolerance.value(), DEFAULT_FORCE_TOLERANCE_EV_PER_ANG
                )
            )

            current_tab = state.get("current_tab")
            self._tabs.setCurrentIndex(
                min(self._normalized_visible_tab_index(current_tab), self._tabs.count() - 1)
            )
            self._refresh_gw_factor_controls()
        finally:
            self._restoring_state = False

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._update_batch_table_column_layout()
        self._ensure_potential_grid()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._update_batch_table_column_layout()

    def _run_probe(self) -> None:
        try:
            fractional = self._probe_point.values()
            probe = probe_point(self._field, self._fractional_to_cartesian(fractional))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "UEP probe failed", str(exc))
            return

        self._render_probe_result(probe)
        self._emit_state_changed()

    def _run_line_sample(self) -> None:
        try:
            sample = sample_line(
                self._field,
                start_ang=self._fractional_to_cartesian(self._line_start.values()),
                end_ang=self._fractional_to_cartesian(self._line_end.values()),
                num_points=self._line_points.value(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "UEP line sampling failed", str(exc))
            return

        self._render_line_sample(sample)
        self._emit_state_changed()

    def _draw_plane(self, sample: UepPlaneSample) -> None:
        self._plane_figure.clear()
        axis = self._plane_figure.add_subplot(1, 1, 1)
        image = axis.imshow(
            np.array(sample.potential_total_ev, dtype=float).T,
            origin="lower",
            aspect="auto",
            extent=(
                float(np.min(sample.x_coordinates_ang)),
                float(np.max(sample.x_coordinates_ang)),
                float(np.min(sample.y_coordinates_ang)),
                float(np.max(sample.y_coordinates_ang)),
            ),
            cmap="viridis",
        )
        axis.set_xlabel("Plane X (A)")
        axis.set_ylabel("Plane Y (A)")
        axis.set_title("Total UEP (eV)")
        self._plane_figure.colorbar(image, ax=axis, label="Potential (eV)")
        self._plane_canvas.draw_idle()

    def _run_plane_sample(self) -> None:
        try:
            sample = sample_plane(
                self._field,
                origin_ang=self._fractional_to_cartesian(self._plane_origin.values()),
                corner_x_ang=self._fractional_to_cartesian(self._plane_x.values()),
                corner_y_ang=self._fractional_to_cartesian(self._plane_y.values()),
                points_x=self._plane_points_x.value(),
                points_y=self._plane_points_y.value(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "UEP plane sampling failed", str(exc))
            return

        self._render_plane_sample(sample)
        self._emit_state_changed()

    def _set_batch_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self._batch_count,
            self._batch_min_muon_muon,
            self._batch_min_muon_atom,
            self._batch_max_attempts,
            self._batch_force_tolerance,
            self._batch_max_steps,
            self._batch_run,
            self._batch_grouping_tolerance,
        ):
            widget.setEnabled(enabled)
        self._refresh_gw_factor_controls()

    def _set_batch_progress_idle(self) -> None:
        self._apply_batch_progress_style(completed=False)
        self._batch_progress.setRange(0, 1)
        self._batch_progress.setValue(0)
        self._batch_progress.setFormat("Ready")

    def _on_batch_search_prepared(self, total: int, summary_text: str) -> None:
        self._set_batch_log(summary_text)
        self._apply_batch_progress_style(completed=False)
        self._batch_progress.setRange(0, max(int(total), 1))
        self._batch_progress.setValue(0)
        self._batch_progress.setFormat("Optimizing %v/%m starts")

    def _on_batch_progress_changed(self, completed: int, total: int) -> None:
        bounded_total = max(int(total), 1)
        if self._batch_progress.maximum() != bounded_total or self._batch_progress.minimum() != 0:
            self._batch_progress.setRange(0, bounded_total)
            self._batch_progress.setFormat("Optimizing %v/%m starts")
        self._batch_progress.setValue(min(max(int(completed), 0), bounded_total))

    def _on_batch_search_ready(self, results: object, distances: object, summary_text: str) -> None:
        if not isinstance(results, list):
            QMessageBox.warning(
                self, "UEP batch search failed", "Batch search returned an unexpected result."
            )
            self._set_batch_progress_failed()
            return
        self._apply_batch_progress_style(completed=True)
        self._batch_progress.setRange(0, max(len(results), 1))
        self._batch_progress.setValue(max(len(results), 1))
        self._batch_progress.setFormat("Batch search complete")
        self._batch_minima_distances = (
            distances if isinstance(distances, UepMinimaDistances) else None
        )
        self._latest_batch_results = list(results)
        self._batch_summary.setPlainText(summary_text)
        summary = self._group_batch_results(self._latest_batch_results)
        if summary is not None:
            self._populate_batch_results(
                self._latest_batch_results,
                summary,
                summary_text=uep_search_log(self._batch_summary.toPlainText(), summary),
            )
        self._emit_state_changed()

    def _set_batch_progress_failed(self) -> None:
        self._apply_batch_progress_style(completed=False)
        self._batch_progress.setRange(0, 1)
        self._batch_progress.setValue(0)
        self._batch_progress.setFormat("Batch search failed")

    def _on_batch_search_failed(self, message: str) -> None:
        self._set_batch_progress_failed()
        self._set_batch_log(f"Batch search failed: {message}")
        QMessageBox.warning(self, "UEP batch search failed", message)

    def _on_batch_search_thread_finished(self) -> None:
        sender = self.sender()
        if sender is None or sender is self._batch_search_thread:
            self._batch_search_thread = None
        self._set_batch_controls_enabled(True)

    def _confirm_discard_batch_results(self, title: str, text: str, accept_label: str) -> bool:
        """Ask before an action discards sites the table lists; true when it lists none."""

        summary = self._latest_batch_summary
        if summary is None or not summary.clusters:
            return True
        detail = (
            f"This discards {plural(len(summary.clusters), 'site')} "
            f"from {plural(len(self._latest_batch_results), 'start')}"
        )
        muons = sum(entry.source_kind == "clustered" for entry in self._viewer_muons)
        if muons:
            detail += f" and removes {plural(muons, 'muon')} from the viewer"
        return workflow_table_utils.confirm_destructive(
            self, title, text, f"{detail}.", accept_label
        )

    def _run_batch_search(self) -> None:
        if self._batch_search_thread is not None:
            return
        if not self._confirm_discard_batch_results(
            "Run a new search",
            "Run a new batch search and replace the current results?",
            "Run search",
        ):
            return

        had_results = bool(self._latest_batch_results)
        self._grouping_settled_timer.stop()
        self._clear_batch_results()
        if had_results:
            self._emit_state_changed()

        thread = _BatchSearchThread(
            self._field,
            lattice_vectors=tuple(
                tuple(float(value) for value in vector)
                for vector in self._source.structure.lattice_vectors
            ),
            host_positions_frac=tuple(
                tuple(float(value) for value in position)
                for position in self._source.structure.fractional_positions
            ),
            requested_starts=self._batch_count.value(),
            min_muon_muon=self._batch_min_muon_muon.value(),
            min_muon_atom=self._batch_min_muon_atom.value(),
            max_attempts=self._batch_max_attempts.value(),
            force_tolerance_ev_per_ang=self._batch_force_tolerance.value(),
            max_steps=self._batch_max_steps.value(),
            grouping_structure=self._grouping_structure(),
        )
        self._batch_search_thread = thread
        thread.prepared.connect(self._on_batch_search_prepared)
        thread.progress_changed.connect(self._on_batch_progress_changed)
        thread.batch_ready.connect(self._on_batch_search_ready)
        thread.failed.connect(self._on_batch_search_failed)
        thread.finished.connect(self._on_batch_search_thread_finished)
        self._set_batch_controls_enabled(False)
        self._apply_batch_progress_style(completed=False)
        self._batch_progress.setRange(0, 0)
        self._batch_progress.setFormat("Packing starts…")
        self._set_batch_log("Packing admissible starts and launching the optimization batch…")
        thread.start()

    def closeEvent(self, event) -> None:
        if self._shutting_down:
            super().closeEvent(event)
            return
        try:
            self.state_changed.emit(self.state_snapshot())
        except RuntimeError:
            pass
        self._shutting_down = True
        self._grouping_settled_timer.stop()
        self._isosurface_preview_timer.stop()
        self._slice_preview_timer.stop()
        self._pending_isosurface_rebuilds = []
        for thread in (
            getattr(self, "_isosurface_grid_thread", None),
            getattr(self, "_isosurface_mesh_thread", None),
            getattr(self, "_batch_search_thread", None),
        ):
            if thread is None:
                continue
            thread.requestInterruption()
            thread.quit()
            thread.wait()
        dialog = getattr(self, "_viewer_muon_dialog", None)
        if dialog is not None:
            dialog.close()
            self._viewer_muon_dialog = None
        for viewer in (
            getattr(self, "_potential_viewer", None),
            getattr(self, "_batch_viewer", None),
        ):
            if viewer is not None:
                viewer.shutdown()
        super().closeEvent(event)

    def _selected_batch_payload(self) -> dict[str, Any] | None:
        item = self._batch_table.currentItem()
        if item is None or not item.isSelected():
            return None
        payload = item.data(0, Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, dict) else None

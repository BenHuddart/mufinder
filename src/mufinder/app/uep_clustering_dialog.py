"""Interactive clustering dialog for relaxed UEP batch-search sites."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict

from matplotlib import colormaps, colors as mcolors
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer, MUON_VIEWER_RADIUS_ANG
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.plot_canvas import create_plot_canvas
from mufinder.app.screen_utils import clamp_to_screen, clamped_size
from mufinder.app.theme import current_palette, style_hint_text, style_section_subtitle, style_section_title
from mufinder.domain.models import Structure
from mufinder.science.uep import (
    UepClusteringConfig,
    UepClusteringResult,
    UepError,
    UepOptimizationResult,
    cluster_prepared_uep_dataset,
    prepare_uep_clustering_dataset,
    relaxed_site_number_map,
)

_HIER_CUT_LINE_GRAB_PIXELS = 8.0
_HIER_CUT_LINE_DRAG_PIXELS = 4.0
_COMPACT_LAYOUT_MAX_WIDTH = 1180
_COMPACT_LAYOUT_MAX_HEIGHT = 760
_LEFT_CANVAS_MINIMUM_SIZE = (240, 200)
_BUBBLE_CANVAS_MINIMUM_SIZE = (220, 150)


def cluster_color_map(cluster_labels: Iterable[int]) -> dict[int, str]:
    """Return the shared cluster-colour palette used by UEP clustering views."""

    cmap = colormaps["tab20"]
    unique_labels = sorted({int(label) for label in cluster_labels})
    colors: dict[int, str] = {}
    for index, label in enumerate(unique_labels):
        colors[int(label)] = mcolors.to_hex(cmap(index % cmap.N))
    return colors


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


class _DendrogramNavigationToolbar(NavigationToolbar2QT):
    """Compact toolbar exposing only the dendrogram navigation tools we use."""

    toolitems = (
        ("Reset", "Reset the dendrogram view", "home", "home"),
        ("Pan", "Pan the dendrogram", "move", "pan"),
        ("Zoom", "Zoom to a rectangle", "zoom_to_rect", "zoom"),
    )

    def __init__(self, canvas, parent=None) -> None:
        super().__init__(canvas, parent, coordinates=False)
        self._button_widgets: dict[str, QPushButton] = {}
        for action in tuple(self.actions()):
            self.removeAction(action)

        palette = current_palette()
        self.setStyleSheet(
            f"""
            QToolBar#uepClusteringDendrogramToolbar {{
                border: none;
                background: transparent;
                spacing: 8px;
                padding: 0px;
            }}
            QPushButton#uepClusteringToolbarButton {{
                border: 1px solid {palette.border};
                border-radius: 10px;
                padding: 6px 14px;
                background: {palette.bg_surface};
                color: {palette.text_primary};
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton#uepClusteringToolbarButton:hover {{
                border-color: {palette.accent};
                background: {palette.bg_surface_alt};
            }}
            QPushButton#uepClusteringToolbarButton:pressed {{
                border-color: {palette.accent_hover};
                background: {palette.bg_surface_alt};
            }}
            QPushButton#uepClusteringToolbarButton:checked,
            QPushButton#uepClusteringToolbarButton[active="true"] {{
                border-color: {palette.accent_pressed};
                background: {palette.accent};
                color: #FFFFFF;
            }}
            QPushButton#uepClusteringToolbarButton:checked:hover,
            QPushButton#uepClusteringToolbarButton[active="true"]:hover {{
                background: {palette.accent_hover};
            }}
            """
        )
        self._home_button = self._add_button("home", "Reset", checkable=False)
        self._pan_button = self._add_button("pan", "Pan", checkable=True)
        self._zoom_button = self._add_button("zoom", "Zoom", checkable=True)
        self._sync_button_states()

    def _add_button(self, callback_name: str, label: str, *, checkable: bool) -> QPushButton:
        button = QPushButton(label, self)
        button.setObjectName("uepClusteringToolbarButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setCheckable(checkable)
        button.setProperty("active", False)
        button.clicked.connect(getattr(self, callback_name))
        self.addWidget(button)
        self._button_widgets[callback_name] = button
        return button

    def _sync_button_states(self) -> None:
        active_mode = getattr(self.mode, "name", "")
        for key, active in (("pan", active_mode == "PAN"), ("zoom", active_mode == "ZOOM")):
            button = self._button_widgets.get(key)
            if button is None:
                continue
            blocked = button.blockSignals(True)
            button.setChecked(active)
            button.blockSignals(blocked)
            button.setProperty("active", active)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _update_buttons_checked(self) -> None:
        super()._update_buttons_checked()
        self._sync_button_states()


class _EqualInitialSplitter(QSplitter):
    """Splitter that waits for real geometry before applying a 50:50 default."""

    def __init__(self, orientation: Qt.Orientation, parent: QWidget | None = None) -> None:
        super().__init__(orientation, parent)
        self._equal_initial_sizes_pending = True
        self._equal_initial_sizes_scheduled = False
        self._last_equal_extent = 0
        self._stable_equal_extent_passes = 0
        self._programmatic_sizing = False

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._schedule_equal_initial_sizes()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._schedule_equal_initial_sizes()

    def setOrientation(self, orientation: Qt.Orientation) -> None:  # noqa: N802 - Qt API naming.
        if orientation != self.orientation():
            self.reset_default_balance()
        super().setOrientation(orientation)

    def reset_default_balance(self) -> None:
        self._equal_initial_sizes_pending = True
        self._last_equal_extent = 0
        self._stable_equal_extent_passes = 0
        self._schedule_equal_initial_sizes()

    def mark_user_sized(self) -> None:
        self._equal_initial_sizes_pending = False

    def is_programmatic_sizing(self) -> bool:
        return self._programmatic_sizing

    def setSizes(self, sizes: list[int]) -> None:  # noqa: N802 - Qt API naming.
        self._programmatic_sizing = True
        try:
            super().setSizes(sizes)
        finally:
            QTimer.singleShot(0, self._clear_programmatic_sizing)

    def moveSplitter(self, pos: int, index: int) -> None:  # noqa: N802 - Qt API naming.
        self._programmatic_sizing = True
        try:
            super().moveSplitter(pos, index)
        finally:
            QTimer.singleShot(0, self._clear_programmatic_sizing)

    def _clear_programmatic_sizing(self) -> None:
        self._programmatic_sizing = False

    def _schedule_equal_initial_sizes(self) -> None:
        if self._equal_initial_sizes_pending and self.isVisible() and not self._equal_initial_sizes_scheduled:
            self._equal_initial_sizes_scheduled = True
            QTimer.singleShot(0, self._apply_equal_initial_sizes)

    def _apply_equal_initial_sizes(self) -> None:
        self._equal_initial_sizes_scheduled = False
        if not self._equal_initial_sizes_pending or self.count() < 2:
            return
        extent = self.width() if self.orientation() == Qt.Orientation.Horizontal else self.height()
        if extent <= 2:
            self._schedule_equal_initial_sizes()
            return
        first = max(extent // 2, 1)
        self.setSizes([first, max(extent - first, 1)])
        if extent == self._last_equal_extent:
            self._stable_equal_extent_passes += 1
        else:
            self._last_equal_extent = extent
            self._stable_equal_extent_passes = 0
        self._equal_initial_sizes_pending = self._stable_equal_extent_passes < 2
        self._schedule_equal_initial_sizes()


class UepClusteringDialog(QDialog):
    """Separate clustering window for relaxed UEP batch-search sites."""

    clustering_ready = Signal(object)
    state_changed = Signal(dict)

    def __init__(
        self,
        *,
        source_job_label: str,
        source_structure,
        viewer_host_structure: Structure,
        optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
        initial_state: Mapping[str, object] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowTitle(f"UEP Clustering - {source_job_label}")
        self.resize(clamped_size(self, 1280, 860))
        self.setMinimumSize(860, 620)

        self._source_job_label = source_job_label
        self._viewer_host_structure = viewer_host_structure
        self._optimizations = tuple(optimizations)
        self._site_number_map = relaxed_site_number_map(self._optimizations)
        self._restoring_state = False
        self._shutting_down = False
        self._prepared = prepare_uep_clustering_dataset(
            self._optimizations,
            source_structure=source_structure,
        )
        self._current_result: UepClusteringResult | None = None
        self._hierarchical_distance_max = self._prepared_max_hier_distance()
        self._pca_points, self._pca_components, self._pca_origin = self._compute_pca_projection()
        self._hier_axis = None
        self._hier_cut_line_artist = None
        self._hier_line_press_event: tuple[float, float] | None = None
        self._hier_line_dragging = False

        # Constrained layout at every size: the dendrogram and energy-spread
        # axes need fixed-pixel room for their tick and axis labels, which a
        # fractional margin cannot guarantee once the dialog is small.
        self._left_figure, self._left_canvas = create_plot_canvas(
            figsize=(6.4, 5.0),
            facecolor=current_palette().bg_surface,
            minimum_size=_LEFT_CANVAS_MINIMUM_SIZE,
        )
        self._left_canvas.mpl_connect("button_press_event", self._on_left_canvas_button_press)
        self._left_canvas.mpl_connect("motion_notify_event", self._on_left_canvas_motion)
        self._left_canvas.mpl_connect("button_release_event", self._on_left_canvas_button_release)
        self._bubble_figure, self._bubble_canvas = create_plot_canvas(
            figsize=(5.6, 3.2),
            facecolor=current_palette().bg_surface,
            minimum_size=_BUBBLE_CANVAS_MINIMUM_SIZE,
        )
        self._geometry_restored = False
        self._responsive_layout_initialized = False
        self._has_restored_splitter_state = False
        self._pending_main_splitter_sizes: list[int] | None = None
        self._pending_right_splitter_sizes: list[int] | None = None

        self._build_ui()
        self._restore_initial_state(initial_state)
        self._apply_responsive_layout()
        self.refresh_clustering(emit_signals=False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Clustering", self)
        style_section_title(title)
        subtitle = QLabel(
            "Cluster the optimized muon positions using either hierarchical or k-means clustering.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls_card = QFrame(self)
        controls_card.setObjectName("analysisSliderPanel")
        controls_layout = QGridLayout(controls_card)
        self._controls_layout = controls_layout
        controls_layout.setContentsMargins(16, 14, 16, 14)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(8)

        method_label = QLabel("Clustering method", controls_card)
        method_label.setObjectName("uepIsosurfaceMinorLabel")
        self._method_combo = QComboBox(controls_card)
        self._method_combo.addItem("Hierarchical", "hier")
        self._method_combo.addItem("K-means", "kmeans")
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        controls_layout.addWidget(method_label, 0, 0)
        controls_layout.addWidget(self._method_combo, 0, 1)

        self._hier_controls = QWidget(controls_card)
        self._hier_controls.setStyleSheet("background: transparent;")
        self._hier_controls.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        hier_layout = QHBoxLayout(self._hier_controls)
        hier_layout.setContentsMargins(0, 0, 0, 0)
        hier_layout.setSpacing(8)
        hier_label = QLabel("Tolerance t", self._hier_controls)
        hier_label.setObjectName("uepIsosurfaceMinorLabel")
        hier_label.setStyleSheet("background: transparent;")
        self._hier_t_input = DecimalLineEdit(self._hier_controls)
        self._hier_t_input.setRange(0.0, max(self._hierarchical_distance_max, 1.0))
        self._hier_t_input.setDecimals(6)
        self._hier_t_input.setMinimumWidth(96)
        self._hier_t_input.setMaximumWidth(126)
        self._hier_t_input.setStyleSheet("QLineEdit { background: transparent; }")
        self._hier_t_input.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._hier_t_input.editingFinished.connect(self._on_hier_input_changed)
        hier_layout.addWidget(hier_label)
        hier_layout.addWidget(self._hier_t_input, 0)

        self._kmeans_controls = QWidget(controls_card)
        self._kmeans_controls.setStyleSheet("background: transparent;")
        self._kmeans_controls.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        kmeans_layout = QHBoxLayout(self._kmeans_controls)
        kmeans_layout.setContentsMargins(0, 0, 0, 0)
        kmeans_layout.setSpacing(8)
        kmeans_label = QLabel("Clusters k", self._kmeans_controls)
        kmeans_label.setObjectName("uepIsosurfaceMinorLabel")
        kmeans_label.setStyleSheet("background: transparent;")
        self._kmeans_k_spin = QSpinBox(self._kmeans_controls)
        self._kmeans_k_spin.setRange(1, max(len(self._optimizations), 1))
        self._kmeans_k_spin.setStyleSheet(
            """
            QSpinBox {
                background: transparent;
            }
            QSpinBox::up-button,
            QSpinBox::down-button {
                background: transparent;
            }
            """
        )
        self._kmeans_k_spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._kmeans_k_spin.valueChanged.connect(self._on_kmeans_spin_changed)
        kmeans_layout.addWidget(kmeans_label)
        kmeans_layout.addWidget(self._kmeans_k_spin, 0)

        self._controls_summary_label = QLabel("Relaxed sites: -  Clusters: -", controls_card)
        self._controls_summary_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._controls_summary_label.setWordWrap(True)
        style_hint_text(self._controls_summary_label)

        controls_layout.addWidget(self._hier_controls, 0, 2, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        controls_layout.addWidget(self._kmeans_controls, 0, 2, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        controls_layout.addWidget(self._controls_summary_label, 0, 3)
        controls_layout.setColumnStretch(2, 1)
        controls_layout.setColumnStretch(3, 1)
        layout.addWidget(controls_card)

        splitter = _EqualInitialSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._main_splitter = splitter

        left_panel = QWidget(splitter)
        left_panel.setMinimumWidth(0)
        left_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        self._left_panel_title = QLabel("Hierarchical clustering", left_panel)
        self._left_panel_title.setObjectName("uepIsosurfacePanelTitle")
        left_layout.addWidget(self._left_panel_title)
        self._hier_toolbar = _DendrogramNavigationToolbar(self._left_canvas, left_panel)
        self._hier_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._hier_toolbar.setMovable(False)
        self._hier_toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self._hier_toolbar.setObjectName("uepClusteringDendrogramToolbar")
        for action in tuple(self._hier_toolbar.actions()):
            if action.isSeparator():
                self._hier_toolbar.removeAction(action)
        left_layout.addWidget(self._hier_toolbar)
        left_layout.addWidget(self._left_canvas, 1)

        right_panel = QWidget(splitter)
        right_panel.setMinimumWidth(0)
        right_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        viewer_panel = QWidget(right_panel)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(8)

        self._structure_viewer = AtomicStructureViewer(viewer_panel)
        self._structure_viewer.setMinimumWidth(0)
        self._structure_viewer.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self._structure_viewer.set_structure(self._viewer_host_structure)
        self._structure_viewer.setMinimumHeight(260)
        viewer_layout.addWidget(self._structure_viewer, 1)
        self._viewer_panel = viewer_panel

        bubble_card = QFrame(right_panel)
        bubble_card.setObjectName("uepIsosurfaceInfoCard")
        bubble_layout = QVBoxLayout(bubble_card)
        bubble_layout.setContentsMargins(12, 10, 12, 10)
        bubble_layout.setSpacing(6)
        bubble_title = QLabel("Cluster Energy Spread", bubble_card)
        bubble_title.setObjectName("uepIsosurfaceLegendTitle")
        bubble_layout.addWidget(bubble_title)
        bubble_layout.addWidget(self._bubble_canvas, 1)
        self._bubble_note_label = QLabel("", bubble_card)
        self._bubble_note_label.setWordWrap(True)
        style_hint_text(self._bubble_note_label)
        bubble_layout.addWidget(self._bubble_note_label)
        self._bubble_card = bubble_card

        right_splitter = _EqualInitialSplitter(Qt.Orientation.Vertical, right_panel)
        right_splitter.setChildrenCollapsible(False)
        right_splitter.addWidget(viewer_panel)
        right_splitter.addWidget(bubble_card)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([1, 1])
        self._right_splitter = right_splitter
        self._right_splitter.splitterMoved.connect(
            lambda *_args: None
            if self._right_splitter.is_programmatic_sizing()
            else self._right_splitter.mark_user_sized()
        )
        self._right_splitter.splitterMoved.connect(lambda *_args: self._emit_state_changed())
        right_layout.addWidget(right_splitter, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([1, 1])
        self._main_splitter.splitterMoved.connect(
            lambda *_args: None
            if self._main_splitter.is_programmatic_sizing()
            else self._main_splitter.mark_user_sized()
        )
        self._main_splitter.splitterMoved.connect(lambda *_args: self._emit_state_changed())
        layout.addWidget(splitter, 1)

        self._structure_viewer.view_state_changed.connect(self._emit_state_changed)
        self._sync_method_controls()

    def _prepared_max_hier_distance(self) -> float:
        if not self._prepared.linkage_matrix:
            return 1.0
        linkage_matrix = np.asarray(self._prepared.linkage_matrix, dtype=float)
        if linkage_matrix.size == 0:
            return 1.0
        return max(float(np.max(linkage_matrix[:, 2])), 1e-6)

    def _compute_pca_projection(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vectors = np.asarray(self._prepared.normalized_genome_vectors, dtype=float)
        if vectors.size == 0:
            return np.empty((0, 2), dtype=float), np.empty((0, 0), dtype=float), np.empty((0,), dtype=float)
        origin = np.mean(vectors, axis=0)
        centred = vectors - origin
        if len(vectors) == 1:
            return np.zeros((1, 2), dtype=float), np.zeros((2, vectors.shape[1]), dtype=float), origin
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
        components = np.asarray(vh[:2], dtype=float)
        if components.shape[0] == 1:
            components = np.vstack([components, np.zeros((1, components.shape[1]), dtype=float)])
        projected = centred @ components.T
        return np.asarray(projected[:, :2], dtype=float), components[:2], origin

    def _method_key(self) -> str:
        data = self._method_combo.currentData()
        return str(data) if isinstance(data, str) else "hier"

    def _current_hier_t(self) -> float:
        return max(self._hier_t_input.value(), 1e-12)

    def _clamp_hier_t(self, value: float) -> float:
        upper = max(float(self._hierarchical_distance_max), 1e-12)
        return min(max(float(value), 0.0), upper)

    def _color_map_for_result(self, result: UepClusteringResult) -> dict[int, str]:
        return cluster_color_map(result.cluster_labels)

    def _current_config(self) -> UepClusteringConfig:
        method = self._method_key()
        return UepClusteringConfig(
            clustering_method=method,
            clustering_hier_t=self._current_hier_t(),
            clustering_kmeans_k=int(self._kmeans_k_spin.value()),
        )

    def _sync_method_controls(self) -> None:
        hierarchical = self._method_key() == "hier"
        self._hier_controls.setVisible(hierarchical)
        self._kmeans_controls.setVisible(not hierarchical)
        self._hier_toolbar.setVisible(hierarchical)
        self._left_panel_title.setText("Hierarchical clustering" if hierarchical else "K-means clustering")
        self._apply_responsive_layout()

    def _set_default_splitter_balance(self) -> None:
        main_total = self._main_splitter.width()
        if main_total <= 0:
            main_total = self.width()
        main_first = max(int(main_total // 2), 1)
        self._pending_main_splitter_sizes = [main_first, max(int(main_total - main_first), 1)]
        self._main_splitter.setSizes(self._pending_main_splitter_sizes)

        right_total = self._right_splitter.height()
        if right_total <= 0:
            right_total = max(int(self.height() // 2), 2)
        right_first = max(int(right_total // 2), 1)
        self._pending_right_splitter_sizes = [right_first, max(int(right_total - right_first), 1)]
        self._right_splitter.setSizes(self._pending_right_splitter_sizes)

    def _reapply_pending_splitter_sizes(self) -> None:
        if self._pending_main_splitter_sizes is not None:
            self._main_splitter.moveSplitter(int(self._pending_main_splitter_sizes[0]), 1)
        if self._pending_right_splitter_sizes is not None:
            self._right_splitter.moveSplitter(int(self._pending_right_splitter_sizes[0]), 1)

    def _viewer_state_snapshot(self) -> dict[str, object]:
        camera = getattr(self._structure_viewer, "_camera_state", None)
        limits = getattr(self._structure_viewer, "_limits", None)
        return {
            "show_cell": bool(self._structure_viewer._show_cell.isChecked()),
            "show_bonds": bool(self._structure_viewer._show_bonds.isChecked()),
            "show_spins": bool(self._structure_viewer._show_spins.isChecked()),
            "camera_state": (
                None
                if camera is None
                else {
                    "elev": float(camera.elev),
                    "azim": float(camera.azim),
                    "roll": float(camera.roll),
                }
            ),
            "limits": (
                None
                if limits is None
                else [[float(lower), float(upper)] for lower, upper in limits]
            ),
        }

    def _restore_viewer_state(self, state: Mapping[str, object] | None) -> None:
        if not isinstance(state, Mapping):
            return

        for widget, key in (
            (self._structure_viewer._show_cell, "show_cell"),
            (self._structure_viewer._show_bonds, "show_bonds"),
            (self._structure_viewer._show_spins, "show_spins"),
        ):
            raw = state.get(key)
            if isinstance(raw, bool | int):
                blocked = widget.blockSignals(True)
                widget.setChecked(bool(raw))
                widget.blockSignals(blocked)

        raw_camera = state.get("camera_state")
        if isinstance(raw_camera, Mapping):
            try:
                camera_state = type(self._structure_viewer._camera_state)(
                    float(raw_camera.get("elev", 20.0)),
                    float(raw_camera.get("azim", 45.0)),
                    float(raw_camera.get("roll", 0.0)),
                )
            except (TypeError, ValueError):
                camera_state = None
            if camera_state is not None:
                self._structure_viewer._camera_state = camera_state
                self._structure_viewer._axis.set_from_state(camera_state)
                if getattr(self._structure_viewer, "_triad_axis", None) is not None:
                    self._structure_viewer._triad_axis.set_from_state(camera_state)
                self._structure_viewer._view = (camera_state.elev, camera_state.azim)
                self._structure_viewer._view_roll = camera_state.roll

        raw_limits = state.get("limits")
        if isinstance(raw_limits, list | tuple) and len(raw_limits) == 3:
            parsed_limits: list[tuple[float, float]] = []
            for entry in raw_limits:
                if not isinstance(entry, list | tuple) or len(entry) != 2:
                    parsed_limits = []
                    break
                try:
                    parsed_limits.append((float(entry[0]), float(entry[1])))
                except (TypeError, ValueError):
                    parsed_limits = []
                    break
            self._structure_viewer._limits = tuple(parsed_limits) if len(parsed_limits) == 3 else None

        self._structure_viewer._render()

    def _emit_state_changed(self) -> None:
        if self._restoring_state or self._shutting_down:
            return
        self.state_changed.emit(self.state_snapshot())

    def _apply_responsive_layout(self) -> None:
        # Compact mode only reflows the controls card. The plot panels stay
        # side by side: stacking them vertically left the dendrogram a few
        # pixels tall once the viewer and energy-spread card took their minimum
        # heights.
        compact = (self.width() <= _COMPACT_LAYOUT_MAX_WIDTH) or (self.height() <= _COMPACT_LAYOUT_MAX_HEIGHT)

        self._controls_layout.removeWidget(self._hier_controls)
        self._controls_layout.removeWidget(self._kmeans_controls)
        if compact:
            self._controls_layout.addWidget(
                self._hier_controls,
                1,
                0,
                1,
                3,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self._controls_layout.addWidget(
                self._kmeans_controls,
                1,
                0,
                1,
                3,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self._controls_layout.addWidget(self._controls_summary_label, 2, 0, 1, 3)
            self._controls_layout.setColumnStretch(0, 0)
            self._controls_layout.setColumnStretch(1, 1)
            self._controls_layout.setColumnStretch(2, 1)
            self._controls_layout.setColumnStretch(3, 0)
            self._controls_layout.setContentsMargins(12, 10, 12, 10)
            self._controls_layout.setHorizontalSpacing(10)
            self._controls_layout.setVerticalSpacing(6)
        else:
            self._controls_layout.addWidget(
                self._hier_controls,
                0,
                2,
                1,
                1,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self._controls_layout.addWidget(
                self._kmeans_controls,
                0,
                2,
                1,
                1,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self._controls_layout.addWidget(self._controls_summary_label, 0, 3)
            self._controls_layout.setColumnStretch(0, 0)
            self._controls_layout.setColumnStretch(1, 0)
            self._controls_layout.setColumnStretch(2, 1)
            self._controls_layout.setColumnStretch(3, 1)
            self._controls_layout.setContentsMargins(16, 14, 16, 14)
            self._controls_layout.setHorizontalSpacing(12)
            self._controls_layout.setVerticalSpacing(8)
        if not self._responsive_layout_initialized and not self._has_restored_splitter_state:
            self._set_default_splitter_balance()
        self._responsive_layout_initialized = True

    def _on_method_changed(self, _index: int) -> None:
        self._sync_method_controls()
        self.refresh_clustering()

    def _on_hier_input_changed(self) -> None:
        self._set_hierarchical_t_visual(self._hier_t_input.value())
        self.refresh_clustering()

    def _on_kmeans_spin_changed(self, value: int) -> None:
        self.refresh_clustering()

    def _set_hierarchical_t_visual(self, value: float) -> None:
        numeric = self._clamp_hier_t(value)
        blocked_input = self._hier_t_input.blockSignals(True)
        self._hier_t_input.setValue(numeric)
        self._hier_t_input.blockSignals(blocked_input)
        self._update_hier_cut_line(numeric)

    def _update_hier_cut_line(self, value: float) -> None:
        line = self._hier_cut_line_artist
        if line is None:
            return
        numeric = self._clamp_hier_t(value)
        try:
            line.set_ydata([numeric, numeric])
        except Exception:  # noqa: BLE001
            return
        self._left_canvas.draw_idle()

    def _hier_toolbar_is_active(self) -> bool:
        return bool(str(getattr(self._hier_toolbar, "mode", "")).strip())

    def _event_near_hier_cut_line(self, event) -> bool:
        axis = self._hier_axis
        if self._method_key() != "hier" or axis is None or self._hier_cut_line_artist is None:
            return False
        if self._hier_toolbar_is_active():
            return False
        if getattr(event, "inaxes", None) is not axis:
            return False
        if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return False
        transformed = axis.transData.transform((0.0, self._current_hier_t()))
        return abs(float(event.y) - float(transformed[1])) <= _HIER_CUT_LINE_GRAB_PIXELS

    def _on_left_canvas_button_press(self, event) -> None:
        self._hier_line_press_event = None
        self._hier_line_dragging = False
        if getattr(event, "button", None) != 1:
            return
        if not self._event_near_hier_cut_line(event):
            return
        self._hier_line_press_event = (float(event.x), float(event.y))

    def _on_left_canvas_motion(self, event) -> None:
        if self._hier_line_press_event is None:
            return
        if getattr(event, "inaxes", None) is not self._hier_axis or getattr(event, "ydata", None) is None:
            return
        if not self._hier_line_dragging:
            if abs(float(event.y) - self._hier_line_press_event[1]) < _HIER_CUT_LINE_DRAG_PIXELS:
                return
            self._hier_line_dragging = True
        self._set_hierarchical_t_visual(float(event.ydata))

    def _on_left_canvas_button_release(self, event) -> None:
        if self._hier_line_press_event is None:
            return
        dragging = self._hier_line_dragging
        self._hier_line_press_event = None
        self._hier_line_dragging = False
        if dragging:
            if getattr(event, "ydata", None) is not None:
                self._set_hierarchical_t_visual(float(event.ydata))
            self.refresh_clustering()

    def _hierarchical_node_cluster_map(self, result: UepClusteringResult) -> dict[int, int | None]:
        labels = np.asarray(result.cluster_labels, dtype=int)
        linkage_matrix = np.asarray(result.linkage_matrix, dtype=float)
        count = len(labels)
        node_map: dict[int, int | None] = {index: int(labels[index]) for index in range(count)}
        for row_index, row in enumerate(linkage_matrix):
            node_id = count + row_index
            left = node_map.get(int(row[0]))
            right = node_map.get(int(row[1]))
            node_map[node_id] = left if left is not None and right is not None and left == right else None
        return node_map

    def _render_hierarchical_plot(self, result: UepClusteringResult, color_map: Mapping[int, str]) -> None:
        from scipy.cluster import hierarchy

        self._left_figure.clear()
        axis = self._left_figure.add_subplot(111)
        self._hier_axis = axis
        axis.set_facecolor(current_palette().bg_surface)
        linkage_matrix = np.asarray(result.linkage_matrix, dtype=float)
        if linkage_matrix.size == 0:
            self._hier_cut_line_artist = None
            axis.axis("off")
            axis.text(0.5, 0.5, "Only one relaxed site is available.", ha="center", va="center")
            self._left_canvas.draw_idle()
            return

        muted = current_palette().text_muted
        node_clusters = self._hierarchical_node_cluster_map(result)
        hierarchy.dendrogram(
            linkage_matrix,
            ax=axis,
            labels=[str(self._site_number_map.get(index, index + 1)) for index in range(len(result.cluster_labels))],
            color_threshold=0.0,
            above_threshold_color=muted,
            link_color_func=lambda node_id: color_map.get(int(node_clusters.get(int(node_id), -1) or -1), muted),
        )
        self._hier_cut_line_artist = axis.axhline(
            result.config.clustering_hier_t,
            color=current_palette().accent,
            linestyle="--",
            linewidth=1.4,
            picker=False,
        )
        axis.set_ylabel("Distance")
        axis.set_xlabel("Relaxed site")
        axis.tick_params(axis="x", labelrotation=0)
        axis.tick_params(colors=current_palette().text_primary)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(current_palette().border)
        axis.spines["bottom"].set_color(current_palette().border)
        self._left_canvas.draw_idle()

    def _project_pca_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0 or self._pca_components.size == 0:
            return np.zeros((len(points), 2), dtype=float)
        return (np.asarray(points, dtype=float) - self._pca_origin) @ self._pca_components.T

    def _render_kmeans_plot(self, result: UepClusteringResult, color_map: Mapping[int, str]) -> None:
        self._left_figure.clear()
        self._hier_axis = None
        self._hier_cut_line_artist = None
        axis = self._left_figure.add_subplot(111)
        axis.set_facecolor(current_palette().bg_surface)
        if len(self._pca_points) == 0:
            axis.axis("off")
            axis.text(0.5, 0.5, "No relaxed sites are available.", ha="center", va="center")
            self._left_canvas.draw_idle()
            return

        labels = np.asarray(result.cluster_labels, dtype=int)
        colors = [color_map[int(label)] for label in labels]
        axis.scatter(
            self._pca_points[:, 0],
            self._pca_points[:, 1],
            c=colors,
            s=88.0,
            alpha=0.95,
            edgecolors="none",
        )
        centroids = np.asarray(result.kmeans_centroids, dtype=float)
        projected_centroids = self._project_pca_points(centroids)
        if len(projected_centroids):
            centroid_colors = [color_map[index + 1] for index in range(len(projected_centroids))]
            axis.scatter(
                projected_centroids[:, 0],
                projected_centroids[:, 1],
                c=centroid_colors,
                s=220.0,
                marker="X",
                linewidths=1.2,
                edgecolors=current_palette().bg_surface,
            )
            for index, point in enumerate(projected_centroids, start=1):
                axis.text(point[0], point[1], f" C{index}", color=current_palette().text_primary, va="center")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(current_palette().border)
        axis.spines["bottom"].set_color(current_palette().border)
        axis.tick_params(colors=current_palette().text_primary)
        self._left_canvas.draw_idle()

    def _render_structure_view(self, result: UepClusteringResult, color_map: Mapping[int, str]) -> None:
        overlays = [
            {
                "positions": [optimization.optimized_position_ang for optimization in self._optimizations],
                "colors": [color_map[int(label)] for label in result.cluster_labels],
                "radii": [MUON_VIEWER_RADIUS_ANG] * len(self._optimizations),
                "alpha": 0.96,
            }
        ]
        previous_camera = deepcopy(getattr(self._structure_viewer, "_camera_state", None))
        previous_limits = deepcopy(getattr(self._structure_viewer, "_limits", None))
        live_pyvista_snapshot = getattr(self._structure_viewer, "_current_pyvista_camera_snapshot", lambda: None)()
        previous_pyvista_snapshot = deepcopy(
            live_pyvista_snapshot
            if live_pyvista_snapshot is not None
            else getattr(self._structure_viewer, "_pyvista_camera_snapshot", None)
        )
        if getattr(self._structure_viewer, "_structure", None) is not self._viewer_host_structure:
            self._structure_viewer.set_structure(self._viewer_host_structure)
        self._structure_viewer.set_point_overlays(overlays)
        current_pyvista_snapshot = getattr(self._structure_viewer, "_current_pyvista_camera_snapshot", lambda: None)()
        snapshot_changed = previous_pyvista_snapshot is not None and not _pyvista_camera_snapshots_match(
            current_pyvista_snapshot,
            previous_pyvista_snapshot,
        )
        if previous_camera is not None and (
            getattr(self._structure_viewer, "_camera_state", None) != previous_camera
            or getattr(self._structure_viewer, "_limits", None) != previous_limits
            or snapshot_changed
        ):
            self._structure_viewer._camera_state = previous_camera
            self._structure_viewer._limits = previous_limits
            self._structure_viewer._axis.set_from_state(previous_camera)
            if getattr(self._structure_viewer, "_triad_axis", None) is not None:
                self._structure_viewer._triad_axis.set_from_state(previous_camera)
            self._structure_viewer._view = (previous_camera.elev, previous_camera.azim)
            self._structure_viewer._view_roll = previous_camera.roll
            self._structure_viewer._pyvista_camera_snapshot = previous_pyvista_snapshot
            self._structure_viewer._render()

    def _render_cluster_energy_summary(self, result: UepClusteringResult, color_map: Mapping[int, str]) -> None:
        self._bubble_figure.clear()
        axis = self._bubble_figure.add_subplot(111)
        axis.set_facecolor(current_palette().bg_surface)
        finite_totals = [
            float(entry.total_energy_ev)
            for entry in self._optimizations
            if entry.total_energy_ev is not None and np.isfinite(entry.total_energy_ev)
        ]
        global_minimum = min(finite_totals) if finite_totals else 0.0
        omitted = 0
        for cluster in result.summary.clusters:
            if cluster.average_energy_ev is None or cluster.standard_deviation_ev is None:
                omitted += 1
                continue
            x_value = float(cluster.average_energy_ev) - float(global_minimum)
            y_value = float(cluster.standard_deviation_ev)
            size = 80.0 + (36.0 * float(cluster.size))
            color = color_map.get(int(cluster.cluster_index), current_palette().accent)
            axis.scatter(
                [x_value],
                [y_value],
                s=[size],
                c=[color],
                alpha=0.88,
                edgecolors=current_palette().bg_surface,
                linewidths=1.0,
            )
            axis.scatter(
                [x_value],
                [y_value],
                s=[18.0],
                c=[current_palette().text_primary],
                alpha=0.95,
                edgecolors=current_palette().bg_surface,
                linewidths=0.6,
                zorder=4,
            )
            axis.text(x_value, y_value, f" {cluster.cluster_index}", color=current_palette().text_primary, va="center")
        # Short labels: the card title already says "Cluster Energy Spread", and
        # on a small screen the plot is only about 150 px tall and 390 px wide.
        axis.set_xlabel("Mean energy above minimum (eV)")
        axis.set_ylabel("Std. dev. (eV)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(current_palette().border)
        axis.spines["bottom"].set_color(current_palette().border)
        axis.tick_params(colors=current_palette().text_primary)
        self._bubble_note_label.setText(
            ""
            if omitted == 0
            else f"Omitted {omitted} cluster{'s' if omitted != 1 else ''} without finite total-energy statistics."
        )
        self._bubble_canvas.draw_idle()

    def _update_summary_labels(self, result: UepClusteringResult) -> None:
        cluster_count = len(result.summary.clusters)
        if self._method_key() == "hier":
            self._left_panel_title.setText("Hierarchical clustering")
        else:
            self._left_panel_title.setText("K-means clustering")
        self._controls_summary_label.setText(
            f"Relaxed sites: {len(self._optimizations)}   Clusters: {cluster_count}"
        )
        summary_text = (
            f"Showing {len(self._optimizations)} relaxed site{'s' if len(self._optimizations) != 1 else ''} "
            "colored by the active cluster assignment."
        )
        self._structure_viewer.setToolTip(summary_text)

    def refresh_clustering(self, *, emit_signals: bool = True) -> None:
        try:
            result = cluster_prepared_uep_dataset(self._prepared, config=self._current_config())
        except UepError as exc:
            self._left_figure.clear()
            axis = self._left_figure.add_subplot(111)
            axis.axis("off")
            axis.text(0.5, 0.5, str(exc), ha="center", va="center")
            self._left_canvas.draw_idle()
            return

        self._current_result = result
        color_map = self._color_map_for_result(result)
        if self._method_key() == "hier":
            self._render_hierarchical_plot(result, color_map)
        else:
            self._render_kmeans_plot(result, color_map)
        self._render_structure_view(result, color_map)
        self._render_cluster_energy_summary(result, color_map)
        self._update_summary_labels(result)
        if emit_signals and not self._restoring_state:
            self.clustering_ready.emit(result)
            self._emit_state_changed()

    def current_result(self) -> UepClusteringResult | None:
        return self._current_result

    def state_snapshot(self) -> dict[str, object]:
        geometry = self.geometry()
        return {
            "method": self._method_key(),
            "hierarchical_t": float(self._current_hier_t()),
            "kmeans_k": int(self._kmeans_k_spin.value()),
            "is_open": bool(self.isVisible()),
            "window_geometry": [int(geometry.x()), int(geometry.y()), int(geometry.width()), int(geometry.height())],
            "splitter_sizes": list(self._main_splitter.sizes()),
            "right_splitter_sizes": list(self._right_splitter.sizes()),
            "viewer_state": self._viewer_state_snapshot(),
            "current_result_summary": None if self._current_result is None else asdict(self._current_result.summary),
        }

    def _restore_initial_state(self, state: Mapping[str, object] | None) -> None:
        if not isinstance(state, Mapping):
            self._hier_t_input.setValue(0.3)
            self._kmeans_k_spin.setValue(min(4, max(len(self._optimizations), 1)))
            return

        self._restoring_state = True
        try:
            method = state.get("method")
            if method == "kmeans":
                self._method_combo.setCurrentIndex(1)
            else:
                self._method_combo.setCurrentIndex(0)

            hier_t = state.get("hierarchical_t")
            try:
                if hier_t is not None:
                    self._hier_t_input.setValue(float(hier_t))
            except (TypeError, ValueError):
                self._hier_t_input.setValue(0.3)

            kmeans_k = state.get("kmeans_k")
            try:
                value = int(kmeans_k) if kmeans_k is not None else min(4, max(len(self._optimizations), 1))
            except (TypeError, ValueError):
                value = min(4, max(len(self._optimizations), 1))
            value = min(max(value, 1), max(len(self._optimizations), 1))
            self._kmeans_k_spin.setValue(value)

            splitter_sizes = state.get("splitter_sizes")
            if isinstance(splitter_sizes, list) and len(splitter_sizes) == 2:
                try:
                    self._main_splitter.setSizes([int(splitter_sizes[0]), int(splitter_sizes[1])])
                    self._main_splitter.mark_user_sized()
                    self._has_restored_splitter_state = True
                    self._pending_main_splitter_sizes = [int(splitter_sizes[0]), int(splitter_sizes[1])]
                except (TypeError, ValueError):
                    pass
            right_splitter_sizes = state.get("right_splitter_sizes")
            if isinstance(right_splitter_sizes, list) and len(right_splitter_sizes) == 2:
                try:
                    self._right_splitter.setSizes([int(right_splitter_sizes[0]), int(right_splitter_sizes[1])])
                    self._right_splitter.mark_user_sized()
                    self._has_restored_splitter_state = True
                    self._pending_right_splitter_sizes = [int(right_splitter_sizes[0]), int(right_splitter_sizes[1])]
                except (TypeError, ValueError):
                    pass
            self._restore_viewer_state(state.get("viewer_state"))
            raw_geometry = state.get("window_geometry")
            if isinstance(raw_geometry, list) and len(raw_geometry) == 4:
                try:
                    self.setGeometry(
                        int(raw_geometry[0]),
                        int(raw_geometry[1]),
                        max(int(raw_geometry[2]), self.minimumWidth()),
                        max(int(raw_geometry[3]), self.minimumHeight()),
                    )
                    clamp_to_screen(self)
                    self._geometry_restored = True
                except (TypeError, ValueError):
                    self._geometry_restored = False
        finally:
            self._restoring_state = False
        self._sync_method_controls()
        splitter_sizes = state.get("splitter_sizes") if isinstance(state, Mapping) else None
        if isinstance(splitter_sizes, list) and len(splitter_sizes) == 2:
            try:
                self._pending_main_splitter_sizes = [int(splitter_sizes[0]), int(splitter_sizes[1])]
                self._main_splitter.setSizes(self._pending_main_splitter_sizes)
                self._main_splitter.mark_user_sized()
            except (TypeError, ValueError):
                pass
        right_splitter_sizes = state.get("right_splitter_sizes") if isinstance(state, Mapping) else None
        if isinstance(right_splitter_sizes, list) and len(right_splitter_sizes) == 2:
            try:
                self._pending_right_splitter_sizes = [int(right_splitter_sizes[0]), int(right_splitter_sizes[1])]
                self._right_splitter.setSizes(self._pending_right_splitter_sizes)
                self._right_splitter.mark_user_sized()
            except (TypeError, ValueError):
                pass

    def shutdown(self) -> None:
        """Release child viewer resources before Qt destroys the dialog tree."""

        if self._shutting_down:
            return
        self._shutting_down = True
        self._structure_viewer.shutdown()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._apply_responsive_layout()
        self._emit_state_changed()

    def moveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().moveEvent(event)
        self._emit_state_changed()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._apply_responsive_layout()
        QTimer.singleShot(0, self._reapply_pending_splitter_sizes)
        self._emit_state_changed()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if hasattr(self, "state_changed") and hasattr(self, "_method_combo"):
            snapshot = self.state_snapshot()
            snapshot["is_open"] = False
            self.state_changed.emit(snapshot)
        if hasattr(self, "_structure_viewer"):
            self.shutdown()
        super().closeEvent(event)

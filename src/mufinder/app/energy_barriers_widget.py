"""Energy Barriers workflow surface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.backend_bases import DrawEvent, KeyEvent, MouseButton, MouseEvent
from matplotlib.lines import Line2D
import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.pinned_sites_scope import PinnedSitesScopeBar
from mufinder.app.plot_canvas import PlotFrame, create_plot_canvas
from mufinder.app.theme import (
    STRUCTURE_OVERLAY_COLOURS,
    arrow_icon,
    card_margins,
    current_palette,
    form_spacing,
    page_margins,
    plot_series_colours,
    style_hint_text,
    style_plot_axes,
    style_section_subtitle,
    style_section_title,
)
from mufinder.app.workflow_row_widget import SourceTableWidget
from mufinder.app.workflow_table_utils import (
    NO_VALUE_TEXT,
    REMOVE_SELECTED_TEXT,
    SelectionFeedback,
    bind_delete_shortcuts,
    confirm_row_removal,
    create_workflow_table,
    format_energy_mev,
    workflow_row_is_locked,
)
from mufinder.domain.constants import MEV_PER_EV
from mufinder.domain.models import Structure

# The path table shows up to this many rows when the page has room, and always holds
# the smaller number, so a short path list never scrolls. The default window fits the
# smaller number with the inspector at its minimum.
_MAX_VISIBLE_PATH_ROWS = 5
_MIN_HELD_PATH_ROWS = 3
# The endpoint selectors keep this many characters of a label on narrow pages; the
# open list shows every label in full.
_ENDPOINT_COMBO_MIN_CHARACTERS = 14

_NO_PATH_SELECTED_TITLE = "No path selected"
_SELECT_A_PATH_HINT = "Select a path to see its hop, or its band once it has run."


@dataclass(frozen=True, slots=True)
class EnergyBarrierEndpointRow:
    """One completed relaxed structure available as an energy-barrier endpoint."""

    key: str
    label: str


@dataclass(frozen=True, slots=True)
class EnergyBarrierSourceRow:
    """One hop in the path table, with its barriers once its NEB band has finished."""

    key: str
    hop: str
    path_length: str
    status: str
    #: Configured CASTEP task-farm count, rendered as "Off" when farming is disabled.
    farms: str = "Off"
    barrier_forward: str = NO_VALUE_TEXT
    barrier_reverse: str = NO_VALUE_TEXT
    delta_energy: str = NO_VALUE_TEXT
    #: Id of the finished NEB result; the inspector shows its band instead of the hop.
    band_key: str = ""
    transition_state_file: str = ""


@dataclass(frozen=True, slots=True)
class EnergyBarrierPreviewImageRow:
    """One image entry in the NEB preview panel."""

    label: str
    #: Cumulative distance along the band from the first image.
    distance: str
    #: Energy relative to the first image.
    energy: str
    distance_value: float  # Å
    energy_value: float  # eV, absolute


class _BandProfilePlot(QWidget):
    """Reaction profile of one band: E − E₀ in meV against distance along the band in Å.

    The plot is the single scrub surface for the inspector: clicking or dragging
    selects the nearest image, and Left/Right arrow keys step through images once
    the plot has focus. It marks the transition state, the highest image, with the
    forward barrier beside it.

    Stepping happens at the pace of the keyboard, with the structure viewer
    following, so it repaints the selected-image marker alone over a saved copy of
    the rest of the plot.
    """

    point_selected = Signal(int)

    _HOVER_DISTANCE = 18.0
    # Below this a band has no barrier to mark, only rounding.
    _MIN_BARRIER_EV = 1e-9
    # Room above the highest image for its label, as a share of the energy range.
    _LABEL_HEADROOM = 0.25
    # A label ending at the plot's right side stops short of the pop-out button.
    _POP_OUT_BUTTON_CLEARANCE_PT = 24
    _SERIES_MARKER_SIZE = 4.0
    _HIGHLIGHT_MARKER_SIZE = 8.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[EnergyBarrierPreviewImageRow] = []
        self._energies_mev = np.empty(0)
        self._selected_index = 0
        self._dragging = False
        self._drawn_palette = current_palette()
        self._axes: Axes | None = None
        self._selected_marker: Line2D | None = None
        self._background: Any = None
        self._figure, self._canvas = create_plot_canvas(
            figsize=(4.6, 2.4), facecolor=self._drawn_palette.bg_primary
        )
        self._canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._canvas.setAccessibleName("Reaction profile")
        self._frame = PlotFrame(self._canvas, title="Reaction profile", parent=self)
        # Moving the canvas between windows can lose the release of a drag.
        self._frame.popped_out_changed.connect(self._end_drag)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._frame)
        self._canvas.mpl_connect("draw_event", self._on_draw)
        self._canvas.mpl_connect("button_press_event", self._on_press)
        self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._canvas.mpl_connect("button_release_event", self._on_release)
        self._canvas.mpl_connect("figure_leave_event", self._on_leave)
        self._canvas.mpl_connect("key_press_event", self._on_key_press)

    def clear_profile(self) -> None:
        self._frame.show_here()
        if not self._rows:
            return
        self._rows = []
        self._selected_index = 0
        self._dragging = False
        self._draw_profile()

    def set_profile(
        self, rows: list[EnergyBarrierPreviewImageRow], *, selected_index: int = 0
    ) -> None:
        """Draw ``rows``; the same rows again only move the selected-image marker."""

        rows = list(rows)
        if rows == self._rows:
            self.set_selected_index(selected_index)
            return
        self._rows = rows
        self._selected_index = self._clamped(selected_index)
        self._draw_profile()

    def point_count(self) -> int:
        return len(self._rows)

    def selected_index(self) -> int:
        return self._selected_index

    def set_selected_index(self, index: int) -> None:
        index = self._clamped(index)
        if index == self._selected_index:
            return
        self._selected_index = index
        self._move_selected_marker()

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802 - Qt API naming.
        super().changeEvent(event)
        if (
            event.type()
            in {
                QEvent.Type.PaletteChange,
                QEvent.Type.ApplicationPaletteChange,
                QEvent.Type.StyleChange,
            }
            and current_palette() != self._drawn_palette
        ):
            self._draw_profile()

    def _clamped(self, index: int) -> int:
        return max(0, min(index, len(self._rows) - 1)) if self._rows else 0

    def _draw_profile(self) -> None:
        """Lay out the whole plot for the current rows and palette."""

        palette = current_palette()
        self._drawn_palette = palette
        self._figure.set_facecolor(palette.bg_primary)
        self._background = None
        self._figure.clear()
        self._axes = None
        self._selected_marker = None
        if not self._rows:
            self._canvas.draw_idle()
            return

        axes = self._figure.add_subplot()
        self._axes = axes
        style_plot_axes(axes, facecolor=palette.bg_primary)
        distances = [row.distance_value for row in self._rows]
        absolute = np.array([row.energy_value for row in self._rows], dtype=float)
        energies = (absolute - absolute[0]) * MEV_PER_EV
        self._energies_mev = energies
        axes.plot(
            distances,
            energies,
            color=plot_series_colours().energy,
            marker="o",
            markersize=self._SERIES_MARKER_SIZE,
        )
        self._mark_transition_state(axes, distances, energies)
        self._selected_marker = axes.plot(
            [distances[self._selected_index]],
            [energies[self._selected_index]],
            linestyle="none",
            marker="o",
            markersize=self._HIGHLIGHT_MARKER_SIZE,
            markerfacecolor=palette.success,
            markeredgecolor=palette.bg_primary,
            # Drawn by _on_draw and _move_selected_marker, not with the rest of the figure.
            animated=True,
        )[0]
        axes.set_xlabel("Distance along band (Å)")
        axes.set_ylabel("E − E₀ (meV)")
        self._canvas.draw_idle()

    def _mark_transition_state(
        self, axes: Axes, distances: Sequence[float], energies: np.ndarray
    ) -> None:
        """Mark the highest image and label the rise to it from the initial image."""

        if not np.any(np.isfinite(energies)):
            return
        peak = int(np.nanargmax(energies))
        barrier_mev = float(energies[peak])
        if barrier_mev / MEV_PER_EV <= self._MIN_BARRIER_EV:
            return
        palette = current_palette()
        peak_distance = distances[peak]
        axes.plot(
            [distances[0], peak_distance, peak_distance],
            [0.0, 0.0, barrier_mev],
            color=palette.seal_gold,
            linestyle="--",
            linewidth=1.2,
        )
        axes.plot(
            [peak_distance],
            [barrier_mev],
            linestyle="none",
            marker="D",
            markersize=self._HIGHLIGHT_MARKER_SIZE,
            markerfacecolor=palette.seal_gold,
            markeredgecolor=palette.bg_primary,
        )
        # Above the highest image no curve can pass, whatever the band's shape.
        low, high = axes.get_ylim()
        axes.set_ylim(low, high + self._LABEL_HEADROOM * (high - low))
        along_band = (peak_distance - distances[0]) / max(distances[-1] - distances[0], 1e-12)
        if along_band < 0.25:
            alignment, offset = "left", 0
        elif along_band > 0.75:
            alignment, offset = "right", -self._POP_OUT_BUTTON_CLEARANCE_PT
        else:
            alignment, offset = "center", 0
        axes.annotate(
            f"Forward {format_energy_mev(barrier_mev / MEV_PER_EV)}",
            xy=(peak_distance, barrier_mev),
            xytext=(offset, 6),
            textcoords="offset points",
            horizontalalignment=alignment,
            verticalalignment="bottom",
            color=palette.text_primary,
            clip_on=True,
            in_layout=False,
        )

    def _on_draw(self, _event: DrawEvent) -> None:
        """Keep the plot without the selected-image marker, then paint the marker on it."""

        if self._axes is None or self._selected_marker is None or self._canvas.is_saving():
            return
        self._background = self._canvas.copy_from_bbox(self._axes.bbox)
        self._axes.draw_artist(self._selected_marker)

    def _move_selected_marker(self) -> None:
        if self._axes is None or self._selected_marker is None:
            return
        self._selected_marker.set_data(
            [self._rows[self._selected_index].distance_value],
            [self._energies_mev[self._selected_index]],
        )
        if self._background is None:
            self._canvas.draw_idle()
            return
        self._canvas.restore_region(self._background)
        self._axes.draw_artist(self._selected_marker)
        self._canvas.blit(self._axes.bbox)

    def _nearest_index(
        self,
        event: MouseEvent,
        *,
        max_distance: float | None = None,
        horizontal_only: bool = False,
    ) -> int | None:
        """Return the image nearest ``event``; matplotlib's events count device pixels."""

        if self._axes is None:
            return None
        points = self._axes.transData.transform(
            np.column_stack(([row.distance_value for row in self._rows], self._energies_mev))
        )
        distances = np.abs(points[:, 0] - event.x)
        if not horizontal_only:
            distances = np.hypot(distances, points[:, 1] - event.y)
        distances = np.nan_to_num(distances, nan=np.inf)
        nearest = int(np.argmin(distances))
        if not np.isfinite(distances[nearest]):
            return None
        if (
            max_distance is not None
            and distances[nearest] > max_distance * self._canvas.device_pixel_ratio
        ):
            return None
        return nearest

    def _select(self, index: int | None) -> None:
        if index is not None and index != self._selected_index:
            self.point_selected.emit(index)

    def _on_press(self, event: MouseEvent) -> None:
        # A popped-out plot's zoom and pan tools hold this lock and own the mouse.
        if event.button is not MouseButton.LEFT or self._canvas.widgetlock.locked():
            return
        self._dragging = True
        self._select(self._nearest_index(event))

    def _on_motion(self, event: MouseEvent) -> None:
        if self._dragging:
            self._select(self._nearest_index(event, horizontal_only=True))
            return
        # The cursor and the pointer belong to a zoom or pan tool while it holds the lock.
        if self._canvas.widgetlock.locked():
            return
        hover = self._nearest_index(event, max_distance=self._HOVER_DISTANCE)
        if hover is None:
            self._canvas.unsetCursor()
            QToolTip.hideText()
            return
        self._canvas.setCursor(Qt.CursorShape.PointingHandCursor)
        row = self._rows[hover]
        QToolTip.showText(
            QCursor.pos(),
            f"{row.label}\nDistance along band: {row.distance}\nE − E₀: {row.energy}",
            self._canvas,
        )

    def _on_release(self, _event: MouseEvent) -> None:
        self._dragging = False

    def _end_drag(self) -> None:
        self._dragging = False

    def _on_leave(self, _event: object) -> None:
        self._canvas.unsetCursor()

    def _on_key_press(self, event: KeyEvent) -> None:
        if self._rows and event.key in ("left", "right"):
            self._select(self._clamped(self._selected_index + (-1 if event.key == "left" else 1)))


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
    prev_button.setToolTip("Previous image along the band.")
    prev_button.setAccessibleName("Previous image")
    prev_button.clicked.connect(lambda: on_step(-1))
    position_label = QLabel("", parent)
    position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    next_button = QToolButton(parent)
    next_button.setIcon(arrow_icon("right"))
    next_button.setText("")
    next_button.setAutoRepeat(True)
    next_button.setToolTip("Next image along the band.")
    next_button.setAccessibleName("Next image")
    next_button.clicked.connect(lambda: on_step(1))
    return prev_button, position_label, next_button


class _ViewerPopoutWindow(QWidget):
    """Floating tool window that hosts the inspector's structure viewer.

    The ``Qt.Tool`` flag keeps the window above the main window while the
    application is active, so scrubbing the band from the main
    window does not bury the popped-out viewer. A mirrored image stepper lets
    the user step through images from this window directly.
    """

    closing = Signal()
    step_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle("Energy Barriers Structure Viewer")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(page_margins())
        layout.setSpacing(form_spacing())

        stepper_row = QHBoxLayout()
        stepper_row.setSpacing(form_spacing())
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
        """Show the image position; without one (no band) the stepper is hidden."""

        self._position_label.setText(text)
        self._prev_button.setEnabled(has_previous)
        self._next_button.setEnabled(has_next)
        self._prev_button.setVisible(bool(text))
        self._position_label.setVisible(bool(text))
        self._next_button.setVisible(bool(text))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self.closing.emit()
        super().closeEvent(event)


class _InspectorSplitter(QSplitter):
    """Splitter that asks only for its minimum height, so the path table is served first.

    The structure viewer prefers about 500 px. Asked for by size hint, that would leave the
    table at its minimum however tall the window; the splitter still takes what is left.
    """

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(super().sizeHint().width(), self.minimumSizeHint().height())


class EnergyBarriersWidget(SourceTableWidget):
    """One page: choose endpoints, list the hops and their barriers, inspect the selected hop.

    The path table and its targeting come from :class:`SourceTableWidget`; this
    widget adds the endpoint pair that generates paths, path removal and the
    inspector, which follows the table selection: the hop in the pristine supercell
    before a band has finished, the band's profile and images after.
    """

    endpoint_selection_changed = Signal(str, str)
    show_every_site_toggled = Signal(bool)
    open_muon_sites_requested = Signal()
    generate_paths_requested = Signal(str, str, str)
    remove_sources_requested = Signal(list)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    configure_result_requested = Signal(object)
    submit_neb_local_requested = Signal(list)
    submit_neb_cluster_requested = Signal(list)
    #: The path the inspector should show; empty when no row is selected.
    path_selected = Signal(str)

    # Path table columns: hop, length, farms, status, forward, reverse, delta E.
    _SOURCE_STATUS_COLUMN = 3
    _SOURCE_CONFIGURE_COLUMN = None
    _SOURCE_WORKFLOW_LABEL = "Energy Barriers"
    _CONFIGURE_SIGNAL = "configure_result_requested"
    _SUBMIT_LOCAL_SIGNAL = "submit_neb_local_requested"
    _SUBMIT_CLUSTER_SIGNAL = "submit_neb_cluster_requested"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Energy Barriers workflow")
        self._endpoint_rows: list[EnergyBarrierEndpointRow] = []
        self._source_rows: list[EnergyBarrierSourceRow]
        # Flush-path guard: set_sources skips an unchanged rebuild.
        self._source_rows_valid = False
        # What the inspector was last asked to show: (path key, band key, hop name).
        self._inspected: tuple[str, str, str] = ("", "", "")
        self._preview_structures: list[Structure] = []
        self._preview_rows: list[EnergyBarrierPreviewImageRow] = []
        self._preview_index = 0
        self._preview_reset_pending = True
        self._preview_viewer: AtomicStructureViewer | None = None
        self._viewer_popout_window: _ViewerPopoutWindow | None = None
        self._viewer_popout_placeholder: QWidget | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(page_margins())
        layout.setSpacing(form_spacing())

        title = QLabel("Energy Barriers")
        style_section_title(title)
        layout.addWidget(title)

        subtitle = QLabel("Run a nudged elastic band between two relaxed muon sites.")
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(subtitle)

        layout.addLayout(self._build_endpoint_row())

        scope_row = QHBoxLayout()
        scope_row.setSpacing(form_spacing())
        self._site_scope = PinnedSitesScopeBar(self)
        self._site_scope.show_every_site_toggled.connect(self.show_every_site_toggled)
        self._site_scope.open_muon_sites_requested.connect(self.open_muon_sites_requested)
        scope_row.addWidget(self._site_scope, 1)
        self._path_symmetry_note = QLabel("", self)
        self._path_symmetry_note.setWordWrap(True)
        style_hint_text(self._path_symmetry_note)
        self._path_symmetry_note.setVisible(False)
        scope_row.addWidget(self._path_symmetry_note, 1)
        layout.addLayout(scope_row)

        layout.addWidget(self._build_paths_group())
        layout.addWidget(self._build_inspector_group(), 1)

        self.clear_neb_preview()
        self._delete_shortcut, self._backspace_shortcut = bind_delete_shortcuts(
            self, self._delete_selected_sources
        )

    def _build_endpoint_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(form_spacing())
        row.addWidget(QLabel("Endpoints", self))
        self._reactant_combo = self._endpoint_combo("Initial endpoint")
        row.addWidget(self._reactant_combo, 1)
        row.addWidget(QLabel("→", self))
        self._product_combo = self._endpoint_combo("Final endpoint")
        row.addWidget(self._product_combo, 1)
        row.addSpacing(8)
        row.addWidget(QLabel("Max length", self))
        self._max_path_length_input = QLineEdit(self)
        self._max_path_length_input.setPlaceholderText("Auto")
        self._max_path_length_input.setClearButtonEnabled(True)
        self._max_path_length_input.setMaximumWidth(self.fontMetrics().horizontalAdvance("0" * 12))
        self._max_path_length_input.setAccessibleName("Maximum path length in Å")
        self._max_path_length_input.setToolTip(
            "Optional maximum path length in Å. Leave blank to use MuFinder's automatic cutoff."
        )
        row.addWidget(self._max_path_length_input)
        row.addWidget(QLabel("Å", self))
        self._generate_paths_button = QPushButton("Generate Paths", self)
        self._generate_paths_button.clicked.connect(self._emit_generate_paths)
        row.addWidget(self._generate_paths_button)
        return row

    def _endpoint_combo(self, accessible_name: str) -> QComboBox:
        combo = QComboBox(self)
        combo.setAccessibleName(accessible_name)
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(_ENDPOINT_COMBO_MIN_CHARACTERS)
        combo.currentIndexChanged.connect(self._emit_endpoint_selection_changed)
        return combo

    def _build_paths_group(self) -> QGroupBox:
        self._sources_group = QGroupBox("Paths", self)
        sources_layout = QVBoxLayout(self._sources_group)
        sources_layout.setContentsMargins(card_margins())
        sources_layout.setSpacing(form_spacing())
        self._sources_table = create_workflow_table(
            ("Hop", "Length", "Farms", "Status", "Forward", "Reverse", "ΔE"),
            resize_to_contents_columns=(1, 2, 3, 4, 5, 6),
            stretch_last_section=False,
            accessible_name="Paths",
            prefer_maximum_height=True,
        )
        self._sources_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._sources_table.itemSelectionChanged.connect(self._on_path_selection_changed)
        self._bind_source_row_menu(
            remove=lambda _rows: self._delete_selected_sources(),
            emit_reset=self.reset_failed_requested.emit,
        )
        sources_layout.addWidget(self._sources_table)

        actions = QHBoxLayout()
        actions.setSpacing(form_spacing())
        self._batch_configure_button = QPushButton("Configure…")
        self._remove_sources_button = QPushButton(REMOVE_SELECTED_TEXT)
        self._selection_count_label = QLabel("")
        style_hint_text(self._selection_count_label)
        # SelectionFeedback sets the run buttons' text ("Run All ..." / "Run
        # Selected ...") and enables Remove Selected from the selection.
        self._submit_neb_local_button = QPushButton()
        self._submit_neb_cluster_button = QPushButton()
        self._batch_configure_button.clicked.connect(self._emit_batch_configure)
        self._remove_sources_button.clicked.connect(self._delete_selected_sources)
        self._submit_neb_local_button.clicked.connect(self._emit_submit_local)
        self._submit_neb_cluster_button.clicked.connect(self._emit_submit_cluster)
        actions.addWidget(self._batch_configure_button)
        actions.addWidget(self._submit_neb_local_button)
        actions.addWidget(self._submit_neb_cluster_button)
        actions.addWidget(self._remove_sources_button)
        actions.addStretch(1)
        actions.addWidget(self._selection_count_label)
        sources_layout.addLayout(actions)
        self._sources_feedback = SelectionFeedback(
            self._sources_table,
            count_label=self._selection_count_label,
            run_local_button=self._submit_neb_local_button,
            run_cluster_button=self._submit_neb_cluster_button,
            selection_buttons=(self._remove_sources_button,),
        )
        self._cap_path_table_height()
        return self._sources_group

    def _build_inspector_group(self) -> QGroupBox:
        group = QGroupBox("Selected path", self)
        inspector_layout = QVBoxLayout(group)
        inspector_layout.setContentsMargins(card_margins())
        inspector_layout.setSpacing(form_spacing())

        header = QHBoxLayout()
        header.setSpacing(form_spacing())
        self._preview_title = QLabel(_NO_PATH_SELECTED_TITLE, group)
        self._preview_title.setWordWrap(True)
        header.addWidget(self._preview_title, 1)
        self._preview_summary = QLabel("", group)
        self._preview_summary.setWordWrap(True)
        style_hint_text(self._preview_summary)
        header.addWidget(self._preview_summary, 1)
        self._popout_viewer_button = QPushButton("Pop Out Viewer", group)
        self._popout_viewer_button.setToolTip(
            "Open the structure viewer in its own resizable window. It stays linked to this page."
        )
        self._popout_viewer_button.clicked.connect(self._toggle_viewer_popout)
        header.addWidget(self._popout_viewer_button)
        inspector_layout.addLayout(header)

        self._preview_splitter = _InspectorSplitter(Qt.Orientation.Horizontal, group)
        self._preview_splitter.setChildrenCollapsible(False)
        # The band's profile and image stepper; hidden while no band is shown.
        self._band_controls = QWidget(self._preview_splitter)
        controls_layout = QVBoxLayout(self._band_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(form_spacing())

        self._profile_plot = _BandProfilePlot(self._band_controls)
        self._profile_plot.point_selected.connect(self._apply_preview_index)
        controls_layout.addWidget(self._profile_plot, 1)

        stepper_row = QHBoxLayout()
        stepper_row.setSpacing(form_spacing())
        self._preview_prev_button, self._preview_position_label, self._preview_next_button = (
            _build_image_stepper(self._band_controls, self._step_preview)
        )
        stepper_row.addStretch(1)
        stepper_row.addWidget(self._preview_prev_button)
        stepper_row.addWidget(self._preview_position_label)
        stepper_row.addWidget(self._preview_next_button)
        stepper_row.addStretch(1)
        controls_layout.addLayout(stepper_row)

        self._preview_viewer_placeholder = QLabel(
            _SELECT_A_PATH_HINT,
            self._preview_splitter,
        )
        self._preview_viewer_placeholder.setWordWrap(True)
        self._preview_viewer_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_hint_text(self._preview_viewer_placeholder)
        self._preview_splitter.addWidget(self._band_controls)
        self._preview_splitter.addWidget(self._preview_viewer_placeholder)
        self._preview_splitter.setStretchFactor(0, 0)
        self._preview_splitter.setStretchFactor(1, 1)
        self._preview_splitter.setSizes([420, 620])
        inspector_layout.addWidget(self._preview_splitter, 1)
        return group

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
            self._reactant_combo.addItem(row.label, row.key)
            self._product_combo.addItem(row.label, row.key)

        self._set_combo_key(self._reactant_combo, current_reactant, fallback_index=0)
        if len(rows) >= 2:
            fallback_index = 1 if self._reactant_combo.currentIndex() == 0 else 0
        else:
            fallback_index = 0
        self._set_combo_key(self._product_combo, current_product, fallback_index=fallback_index)
        self._reactant_combo.blockSignals(False)
        self._product_combo.blockSignals(False)
        self._update_actions()

    def set_path_symmetry_note(self, text: str) -> None:
        """Show which symmetry groups the generated paths; hidden when ``text`` is empty."""

        self._path_symmetry_note.setText(text)
        self._path_symmetry_note.setVisible(bool(text))

    def set_site_scope(self, *, show_every_site: bool, notice: str = "") -> None:
        """Show whether the endpoints list every relaxed site, and the pinned-sites notice."""

        self._site_scope.set_state(show_every_site=show_every_site, notice=notice)

    def set_sources(self, rows: list[EnergyBarrierSourceRow]) -> None:
        """Show ``rows``, keeping the selected paths; unchanged rows skip the rebuild."""

        rows = list(rows)
        if self._source_rows_valid and rows == self._source_rows:
            return
        self._source_rows_valid = True
        selected = set(self.selected_source_keys())
        self._sources_group.setTitle(f"Paths ({len(rows)})")
        self._set_source_rows(rows)
        self._select_source_keys(selected)
        self._cap_path_table_height()
        self._update_inspected_path()

    def source_row_values(self, row: EnergyBarrierSourceRow) -> tuple[str, ...]:
        return (
            row.hop,
            row.path_length,
            row.farms,
            row.status,
            row.barrier_forward,
            row.barrier_reverse,
            row.delta_energy,
        )

    def create_source_row_cells(self, row_index: int, row: EnergyBarrierSourceRow) -> None:
        if not row.transition_state_file:
            return
        tooltip = f"Trajectory file: {row.transition_state_file}"
        for column in range(self._sources_table.columnCount()):
            item = self._sources_table.item(row_index, column)
            if item is not None:
                item.setToolTip(tooltip)

    def refresh_source_actions(self, selected_rows: Sequence[EnergyBarrierSourceRow]) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        can_generate = bool(reactant_key and product_key and len(self._endpoint_rows) >= 1)
        self._generate_paths_button.setEnabled(can_generate)

    def _select_source_keys(self, keys: set[str]) -> None:
        """Select the rows ``keys`` name without announcing it, then refresh the actions."""

        table = self._sources_table
        previous = table.blockSignals(True)
        try:
            table.clearSelection()
            model = table.selectionModel()
            for row_index, row in enumerate(self._source_rows):
                if row.key in keys:
                    model.select(
                        table.model().index(row_index, 0),
                        model.SelectionFlag.Select | model.SelectionFlag.Rows,
                    )
        finally:
            table.blockSignals(previous)
        self._update_actions()
        self._sources_feedback.refresh()

    def _cap_path_table_height(self) -> None:
        table = self._sources_table
        header_height = table.horizontalHeader().sizeHint().height()
        row_height = (
            table.rowHeight(0) if table.rowCount() else table.verticalHeader().defaultSectionSize()
        )
        chrome = header_height + 2 * table.frameWidth() + 6
        rows = max(1, table.rowCount())
        table.setMinimumHeight(chrome + min(rows, _MIN_HELD_PATH_ROWS) * row_height)
        table.setMaximumHeight(chrome + min(rows, _MAX_VISIBLE_PATH_ROWS) * row_height)

    def _on_path_selection_changed(self) -> None:
        self._update_actions()
        self._update_inspected_path()

    def _update_inspected_path(self) -> None:
        """Ask for the first selected path when it, its band or its name changed."""

        selected = self._selected_source_rows()
        inspected = (
            (selected[0].key, selected[0].band_key, selected[0].hop) if selected else ("", "", "")
        )
        if inspected != self._inspected:
            self._inspected = inspected
            self.path_selected.emit(inspected[0])

    def set_path_preview(
        self,
        *,
        title: str,
        summary: str,
        structure: Structure,
        start_cartesian: Sequence[float],
        end_cartesian: Sequence[float],
    ) -> None:
        """Show a hop that has no band: the pristine supercell with the hop arrow."""

        self._clear_band()
        self._preview_title.setText(title)
        self._preview_summary.setText(summary)
        self._popout_viewer_button.setEnabled(True)
        start = tuple(float(value) for value in start_cartesian)
        end = tuple(float(value) for value in end_cartesian)
        viewer = self._ensure_preview_viewer()
        with viewer.batch_update():
            viewer.set_structure(structure)
            viewer.select_atoms([])
            viewer.set_point_overlays(
                [
                    {
                        "positions": [start],
                        "colors": [STRUCTURE_OVERLAY_COLOURS.path_start],
                        "radii": [0.28],
                    },
                    {
                        "positions": [end],
                        "colors": [STRUCTURE_OVERLAY_COLOURS.path_end],
                        "radii": [0.28],
                    },
                ]
            )
            viewer.set_line_overlays(
                [
                    {
                        "start": start,
                        "end": end,
                        "color": STRUCTURE_OVERLAY_COLOURS.path_arrow,
                        "width": 7.0,
                        "alpha": 0.92,
                        "arrow": True,
                    }
                ]
            )

    def set_neb_preview(
        self,
        *,
        title: str,
        image_rows: list[EnergyBarrierPreviewImageRow],
        structures: list[Structure],
        selected_index: int = 0,
    ) -> None:
        """Show a finished band: its profile, image stepper and images in the viewer."""

        if not image_rows or not structures:
            self.clear_neb_preview(title=title)
            return
        self._preview_title.setText(title)
        self._preview_rows = list(image_rows)
        self._preview_structures = list(structures)
        self._preview_reset_pending = True
        self._band_controls.setVisible(True)
        self._profile_plot.set_profile(self._preview_rows, selected_index=selected_index)
        self._apply_preview_index(selected_index)

    def clear_neb_preview(self, *, title: str | None = None) -> None:
        """Empty the inspector; ``title`` says why when a path is selected."""

        self._clear_band()
        self._preview_title.setText(title or _NO_PATH_SELECTED_TITLE)
        self._preview_summary.setText("")
        self._popout_viewer_button.setEnabled(self._viewer_popout_window is not None)
        # A title says why a selected path shows nothing, so the empty area adds no text.
        empty_message = "" if title else _SELECT_A_PATH_HINT
        if self._preview_viewer is None:
            self._preview_viewer_placeholder.setText(empty_message)
        else:
            self._preview_viewer.set_empty_message(empty_message)
            with self._preview_viewer.batch_update():
                self._preview_viewer.set_point_overlays(None)
                self._preview_viewer.set_line_overlays(None)
                self._preview_viewer.set_structure(None)

    def _clear_band(self) -> None:
        self._preview_rows = []
        self._preview_structures = []
        self._preview_index = 0
        self._preview_reset_pending = True
        self._preview_position_label.setText("")
        self._preview_prev_button.setEnabled(False)
        self._preview_next_button.setEnabled(False)
        self._profile_plot.clear_profile()
        self._band_controls.setVisible(False)
        self._sync_popout_stepper()

    def selected_endpoint_keys(self) -> tuple[str, str]:
        reactant = self._reactant_combo.currentData()
        product = self._product_combo.currentData()
        return (
            str(reactant) if isinstance(reactant, str) else "",
            str(product) if isinstance(product, str) else "",
        )

    def _emit_endpoint_selection_changed(self) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        self.endpoint_selection_changed.emit(reactant_key, product_key)
        self._update_actions()

    def _emit_generate_paths(self) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        self.generate_paths_requested.emit(
            reactant_key, product_key, self._max_path_length_input.text().strip()
        )

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
        self._preview_summary.setText(f"{row.distance} along the band  ·  E − E₀ {row.energy}")
        self._profile_plot.set_selected_index(clamped)
        self._preview_position_label.setText(f"Image {clamped + 1} / {len(self._preview_rows)}")
        self._preview_prev_button.setEnabled(clamped > 0)
        self._preview_next_button.setEnabled(clamped < len(self._preview_rows) - 1)
        self._popout_viewer_button.setEnabled(True)
        self._sync_popout_stepper()
        preview_viewer = self._ensure_preview_viewer()
        if self._preview_reset_pending:
            # A new NEB result: load fresh so the viewer fits the new cell.
            with preview_viewer.batch_update():
                preview_viewer.set_point_overlays(None)
                preview_viewer.set_line_overlays(None)
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
            # Taken out of the window first: PySide ties Python objects to the parent it
            # last saw, so deleting the window would kill the viewer's layout items.
            viewer.setParent(None)
            self._preview_splitter.replaceWidget(1, viewer)
            placeholder.deleteLater()
            viewer.show()
        self._popout_viewer_button.setText("Pop Out Viewer")
        self._popout_viewer_button.setEnabled(viewer is not None and viewer.has_structure())
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


def _muon_atom_indexes(structure: Structure) -> list[int]:
    return [
        index
        for index, species in enumerate(structure.species)
        if str(species).strip().casefold() == "h:mu"
        or str(species).strip().casefold().endswith(":mu")
    ]

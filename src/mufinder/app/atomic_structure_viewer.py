"""Embedded 3D atomic structure viewer for host structures."""

from __future__ import annotations

from collections.abc import Iterable

from ase import Atoms
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.neighborlist import NeighborList
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.projections import register_projection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from mufinder.app.theme import style_hint_text
from mufinder.domain.models import Structure

_DEFAULT_VIEW = (20.0, 45.0)


def _view_init_with_roll(axis, *, elev: float, azim: float, roll: float) -> None:
    """Apply camera orientation while remaining compatible with older Matplotlib versions."""

    try:
        axis.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        axis.view_init(elev=elev, azim=azim)


class _WideViewportAxes3D(Axes3D):
    """Axes3D variant that preserves the requested rectangular viewport."""

    name = "mufinder3d"

    def apply_aspect(self, position=None):
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, which="active")

    def get_proj(self):
        projection = super().get_proj()
        fig_w, fig_h = self.figure.get_size_inches()
        position = self.get_position()
        viewport_w = float(fig_w * position.width)
        viewport_h = float(fig_h * position.height)
        if viewport_w <= 1e-12 or viewport_h <= 1e-12:
            return projection

        if viewport_w > viewport_h:
            scale = viewport_h / viewport_w
            correction = np.array(
                [[scale, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=float,
            )
        else:
            scale = viewport_w / viewport_h
            correction = np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, scale, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=float,
            )
        return correction @ projection


register_projection(_WideViewportAxes3D)


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


def _get_atom_style(numbers: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colors = np.array(jmol_colors[numbers], dtype=float)
    muon_mask = symbols == "X"
    if np.any(muon_mask):
        colors[muon_mask] = np.array([1.0, 0.08, 0.58])
    radii = np.clip(0.9 * covalent_radii[numbers], 0.2, 1.15)
    if np.any(muon_mask):
        radii[muon_mask] = np.maximum(radii[muon_mask], 0.26)
    return colors, radii


def _set_equal_limits(
    axis,
    positions: np.ndarray,
    *,
    atom_radii: np.ndarray | None = None,
    extra_points: np.ndarray | None = None,
    margin: float = 0.12,
) -> None:
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

    axis.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
    axis.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])
    axis.set_zlim(mins[2] - padding[2], maxs[2] + padding[2])
    axis_spans = np.maximum((maxs + padding) - (mins - padding), 1e-6)
    _set_axis_box_aspect_from_spans(axis, axis_spans)


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

    basis_u = _normalized_vector(np.cross(axis_dir, reference), np.array([1.0, 0.0, 0.0], dtype=float))
    basis_v = _normalized_vector(np.cross(axis_dir, basis_u), np.array([0.0, 1.0, 0.0], dtype=float))

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
    """Draw high-visibility 3D spin arrows aligned with the z-axis."""

    if len(positions) == 0:
        return np.empty((0, 3), dtype=float)

    theta = np.linspace(0.0, 2.0 * np.pi, 18)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    extent_points: list[np.ndarray] = []
    for position, radius, spin in zip(positions, radii, spins):
        magnitude = float(abs(spin))
        if magnitude <= 1e-10:
            continue

        direction = 1.0 if spin >= 0.0 else -1.0
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

        # Shaft
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

        # Head (cone)
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


def _normalized_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=float)
    return np.asarray(vector, dtype=float) / norm


class AtomicStructureViewer(QWidget):
    """Render a contained, ASE-style 3D view of a structure."""

    atom_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._structure: Structure | None = None
        self._atoms: Atoms | None = None
        self._axis = None
        self._triad_axis = None
        self._pick_artist = None
        self._sphere_artists: list[object] = []
        self._selected_index: int | None = None
        self._spins: np.ndarray | None = None
        self._view = _DEFAULT_VIEW
        self._view_roll = 0.0
        self._limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None

        self._figure = Figure(figsize=(6.0, 5.0), facecolor="white")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._triad_figure = Figure(figsize=(2.2, 1.35), facecolor="white")
        self._triad_canvas = FigureCanvasQTAgg(self._triad_figure)
        self._figure.patch.set_alpha(0.0)
        self._triad_figure.patch.set_alpha(0.0)
        self._canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._triad_canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._canvas.setStyleSheet("background: transparent;")
        self._triad_canvas.setStyleSheet("background: transparent;")
        self._triad_canvas.setFixedWidth(132)
        self._triad_canvas.setFixedHeight(130)

        self._show_cell = QCheckBox("Cell")
        self._show_cell.setChecked(True)
        self._show_bonds = QCheckBox("Bonds")
        self._show_bonds.setChecked(False)
        self._show_spins = QCheckBox("Spins")
        self._show_spins.setChecked(False)
        self._show_bonds.setToolTip("Show or hide nearest-neighbor bonds.")
        self._show_spins.setToolTip("Show magnetic spin arrows.")
        self._view_x_button = QPushButton("X")
        self._view_y_button = QPushButton("Y")
        self._view_z_button = QPushButton("Z")
        self._reset_view_button = QPushButton("Reset View")
        self._reset_view_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._reset_view_button.setMinimumWidth(self._reset_view_button.sizeHint().width() + 8)
        self._reset_view_button.setToolTip("Reset orientation and zoom to defaults.")
        self._interaction_hint = QLabel(
            "Left click atom: select | Left click empty: clear | Right drag: rotate | "
            "Middle drag: pan | Scroll: zoom"
        )
        self._interaction_hint.setWordWrap(True)
        style_hint_text(self._interaction_hint)
        self._selection_label = QLabel("Selected atom: -")
        self._selection_label.setWordWrap(True)

        self._build_layout()

        self._show_cell.toggled.connect(self._redraw_preserving_view)
        self._show_bonds.toggled.connect(self._redraw_preserving_view)
        self._show_spins.toggled.connect(self._redraw_preserving_view)
        self._view_x_button.clicked.connect(lambda: self._set_named_view("x"))
        self._view_y_button.clicked.connect(lambda: self._set_named_view("y"))
        self._view_z_button.clicked.connect(lambda: self._set_named_view("z"))
        self._reset_view_button.clicked.connect(self._reset_view)
        self._canvas.mpl_connect("pick_event", self._on_pick)
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("button_release_event", self._store_view)
        self._canvas.mpl_connect("resize_event", self._on_canvas_resize)

        self._render()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(self._show_cell)
        controls.addSpacing(4)
        controls.addWidget(self._show_bonds)
        controls.addSpacing(4)
        controls.addWidget(self._show_spins)
        controls.addSpacing(8)
        controls.addWidget(QLabel("View:"))
        controls.addWidget(self._view_x_button)
        controls.addWidget(self._view_y_button)
        controls.addWidget(self._view_z_button)
        controls.addStretch(1)
        controls.addWidget(self._reset_view_button)

        layout.addLayout(controls)
        layout.addWidget(self._interaction_hint)

        viewer_row = QHBoxLayout()
        viewer_row.setContentsMargins(0, 0, 0, 0)
        viewer_row.setSpacing(0)

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

    def set_structure(self, structure: Structure | None) -> None:
        """Load a new structure into the viewer."""
        self._structure = structure
        self._atoms = self._to_atoms(structure) if structure is not None else None
        self._spins = self._extract_spins(structure)
        self._set_spin_visibility_default_from_data()
        self._selected_index = None
        self._view = _DEFAULT_VIEW
        self._view_roll = 0.0
        self._limits = None
        self._selection_label.setText("Selected atom: -")
        self._render()

    def set_spins(self, spins: list[float] | tuple[float, ...] | np.ndarray | None) -> None:
        """Replace current per-atom spin values and redraw."""
        if self._atoms is None or spins is None:
            self._spins = None
        else:
            spin_array = np.asarray(spins, dtype=float)
            if len(spin_array) != len(self._atoms):
                self._spins = None
            else:
                self._spins = spin_array.copy()
        if self._spins is not None and np.any(np.abs(self._spins) > 1e-10):
            self._show_spins.setChecked(True)
        self._store_view()
        self._render()

    def _set_spin_visibility_default_from_data(self) -> None:
        if self._spins is not None and np.any(np.abs(self._spins) > 1e-10):
            self._show_spins.setChecked(True)
            return
        self._show_spins.setChecked(False)

    def select_atom(self, atom_index: int | None) -> None:
        """Highlight an atom without emitting a new selection signal."""
        self._set_selected_atom(atom_index, emit_signal=False)

    def _set_selected_atom(self, atom_index: int | None, *, emit_signal: bool) -> None:
        if self._atoms is None or atom_index is None or atom_index < 0 or atom_index >= len(self._atoms):
            self._selected_index = None
            self._selection_label.setText("Selected atom: -")
        else:
            self._selected_index = int(atom_index)
            self._selection_label.setText(self._format_selection_text(self._selected_index))

        self._store_view()
        self._render()

        if emit_signal and self._selected_index is not None:
            self.atom_selected.emit(self._selected_index)

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
        return Atoms(
            symbols=structure.species,
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

    def _redraw_preserving_view(self) -> None:
        self._store_view()
        self._render()

    def _reset_view(self) -> None:
        self._view = _DEFAULT_VIEW
        self._view_roll = 0.0
        self._limits = None
        self._render()

    def _set_named_view(self, axis_name: str) -> None:
        if axis_name == "x":
            self._view = (0.0, 0.0)
        elif axis_name == "y":
            self._view = (0.0, 90.0)
        else:
            self._view = (90.0, -90.0)
        self._view_roll = 0.0
        self._render()

    def _store_view(self, *_args) -> None:
        if self._axis is not None:
            self._view = (float(self._axis.elev), float(self._axis.azim))
            self._view_roll = float(getattr(self._axis, "roll", 0.0))
            self._limits = (
                tuple(float(value) for value in self._axis.get_xlim()),
                tuple(float(value) for value in self._axis.get_ylim()),
                tuple(float(value) for value in self._axis.get_zlim()),
            )
            if self._triad_axis is not None:
                _view_init_with_roll(
                    self._triad_axis,
                    elev=self._view[0],
                    azim=self._view[1],
                    roll=self._view_roll,
                )
                self._triad_canvas.draw_idle()

    def _sync_triad_to_main_view(self) -> None:
        if self._axis is None or self._triad_axis is None:
            return
        _view_init_with_roll(
            self._triad_axis,
            elev=float(self._axis.elev),
            azim=float(self._axis.azim),
            roll=float(getattr(self._axis, "roll", 0.0)),
        )

    def _on_pick(self, event) -> None:
        if self._pick_artist is None or event.artist is not self._pick_artist:
            return
        indices = getattr(event, "ind", None)
        if indices is None or len(indices) == 0:
            return
        self._set_selected_atom(int(indices[0]), emit_signal=True)

    def _on_mouse_press(self, event) -> None:
        if self._axis is None or event.inaxes != self._axis:
            return
        if event.button == 1 and self._selected_index is not None:
            # Blank-space clicks should clear selection; atom picks are restored via pick_event.
            self._set_selected_atom(None, emit_signal=False)
        # Matplotlib mouse_init handles rotate/pan; this callback remains for extensibility.
        return

    def _on_mouse_release(self, _event) -> None:
        self._sync_triad_to_main_view()
        self._store_view()

    def _on_mouse_move(self, event) -> None:
        if self._axis is None or event.inaxes != self._axis:
            return
        self._sync_triad_to_main_view()
        self._canvas.draw_idle()
        self._triad_canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if self._axis is None or event.inaxes != self._axis:
            return
        zoom = 0.9 if event.button == "up" else 1.1
        xlim = self._axis.get_xlim()
        ylim = self._axis.get_ylim()
        zlim = self._axis.get_zlim()

        x_center = 0.5 * (xlim[0] + xlim[1])
        y_center = 0.5 * (ylim[0] + ylim[1])
        z_center = 0.5 * (zlim[0] + zlim[1])
        x_half = 0.5 * (xlim[1] - xlim[0]) * zoom
        y_half = 0.5 * (ylim[1] - ylim[0]) * zoom
        z_half = 0.5 * (zlim[1] - zlim[0]) * zoom

        self._axis.set_xlim(x_center - x_half, x_center + x_half)
        self._axis.set_ylim(y_center - y_half, y_center + y_half)
        self._axis.set_zlim(z_center - z_half, z_center + z_half)
        self._store_view()
        self._canvas.draw_idle()

    def _on_canvas_resize(self, _event) -> None:
        if self._axis is None:
            return
        x_span = max(self._axis.get_xlim()[1] - self._axis.get_xlim()[0], 1e-6)
        y_span = max(self._axis.get_ylim()[1] - self._axis.get_ylim()[0], 1e-6)
        z_span = max(self._axis.get_zlim()[1] - self._axis.get_zlim()[0], 1e-6)
        _set_axis_box_aspect_from_spans(self._axis, np.array([x_span, y_span, z_span], dtype=float))
        self._canvas.draw_idle()

    def _render(self) -> None:
        self._figure.clear()
        axis = self._figure.add_axes([0.0, 0.02, 1.0, 0.96], projection="mufinder3d")
        self._axis = axis
        self._pick_artist = None
        self._sphere_artists = []
        axis.set_facecolor("none")
        axis.patch.set_alpha(0.0)
        axis.set_proj_type("ortho")
        # ASE-like interaction mapping: right drag rotates, middle drag pans.
        axis.mouse_init(rotate_btn=3, pan_btn=2)

        if self._atoms is None or len(self._atoms) == 0:
            self._triad_figure.clear()
            self._triad_axis = None
            self._triad_canvas.draw_idle()
            axis.set_axis_off()
            axis.text2D(0.5, 0.5, "Load a host structure to inspect it here.", ha="center", va="center")
            self._canvas.draw_idle()
            return

        positions = np.asarray(self._atoms.get_positions(), dtype=float)
        numbers = np.asarray(self._atoms.get_atomic_numbers(), dtype=int)
        symbols = np.asarray(self._atoms.get_chemical_symbols())
        colors, radii = _get_atom_style(numbers, symbols)
        pick_sizes = np.clip((17.0 * radii) ** 2, 80.0, 2600.0)

        sphere_resolution = 12 if len(positions) <= 80 else 9
        atom_alpha = 0.95
        if self._show_spins.isChecked() and self._spins is not None:
            atom_alpha = 0.5
        if self._show_bonds.isChecked():
            bond_alpha = float(np.clip(0.84 * atom_alpha, 0.2, 0.8))
            self._draw_bonds(axis, positions, colors, radii, alpha=bond_alpha)
        self._sphere_artists = _draw_atom_spheres(
            axis,
            positions,
            radii,
            colors,
            alpha=atom_alpha,
            resolution=sphere_resolution,
        )

        spin_tips = np.empty((0, 3), dtype=float)
        if self._show_spins.isChecked() and self._spins is not None and len(self._spins) == len(positions):
            spin_tips = _draw_spin_arrows(axis, positions, radii, self._spins)

        self._pick_artist = axis.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=pick_sizes,
            c="black",
            alpha=0.01,
            edgecolors="none",
            picker=True,
        )

        extra_points = None
        if self._show_cell.isChecked():
            extra_points = self._draw_unit_cell(axis)
        if len(spin_tips):
            extra_points = spin_tips if extra_points is None else np.vstack([extra_points, spin_tips])

        if self._selected_index is not None:
            self._highlight_selected_atom(axis, positions, radii)

        if self._limits is None:
            _set_equal_limits(axis, positions, atom_radii=radii, extra_points=extra_points, margin=0.1)
        else:
            axis.set_xlim(*self._limits[0])
            axis.set_ylim(*self._limits[1])
            axis.set_zlim(*self._limits[2])
            x_span = max(self._limits[0][1] - self._limits[0][0], 1e-6)
            y_span = max(self._limits[1][1] - self._limits[1][0], 1e-6)
            z_span = max(self._limits[2][1] - self._limits[2][0], 1e-6)
            _set_axis_box_aspect_from_spans(axis, np.array([x_span, y_span, z_span], dtype=float))

        _view_init_with_roll(axis, elev=self._view[0], azim=self._view[1], roll=self._view_roll)
        self._draw_crystallographic_axes()
        axis.set_axis_off()
        axis.grid(False)

        for coord_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
            coord_axis.pane.set_alpha(0.0)

        self._canvas.draw_idle()

    def _draw_unit_cell(self, axis) -> np.ndarray:
        assert self._atoms is not None
        corners, edges = _unit_cell_geometry(np.asarray(self._atoms.cell, dtype=float))
        for start, end in edges:
            p0 = corners[start]
            p1 = corners[end]
            axis.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color="0.25",
                linewidth=1.1,
                alpha=0.62,
            )
        return corners

    def _draw_bonds(
        self,
        axis,
        positions: np.ndarray,
        colors: np.ndarray,
        radii: np.ndarray,
        *,
        alpha: float,
    ) -> None:
        assert self._atoms is not None
        cutoffs = 1.15 * covalent_radii[self._atoms.get_atomic_numbers()]
        neighbour_list = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
        neighbour_list.update(self._atoms)

        cell = np.asarray(self._atoms.cell, dtype=float)
        cylinder_resolution = 14 if len(positions) <= 120 else 10
        for atom_index in range(len(self._atoms)):
            neighbours, offsets = neighbour_list.get_neighbors(atom_index)
            start = positions[atom_index]
            for neighbour_index, offset in zip(neighbours, offsets):
                if neighbour_index <= atom_index:
                    continue
                if np.any(np.asarray(offset, dtype=int) != 0):
                    continue
                end = positions[neighbour_index] + np.dot(offset, cell)
                bond_radius = float(np.clip(0.18 * min(radii[atom_index], radii[neighbour_index]), 0.06, 0.16))
                _draw_split_bond_cylinder(
                    axis,
                    start,
                    end,
                    colors[atom_index],
                    colors[neighbour_index],
                    radius=bond_radius,
                    alpha=alpha,
                    resolution=cylinder_resolution,
                )

    def _draw_crystallographic_axes(self) -> None:
        assert self._atoms is not None
        if self._triad_axis is None:
            self._triad_figure.clear()
            self._triad_axis = self._triad_figure.add_axes([0.02, 0.06, 0.96, 0.9], projection="3d")

        triad_axis = self._triad_axis
        triad_axis.cla()
        triad_axis.set_facecolor((1.0, 1.0, 1.0, 0.0))
        triad_axis.patch.set_alpha(0.0)
        triad_axis.set_proj_type("ortho")
        triad_axis.mouse_init(rotate_btn=[], pan_btn=[])
        triad_axis.set_axis_off()
        triad_axis.grid(False)

        extent = np.linalg.norm(np.asarray(self._atoms.cell, dtype=float), axis=1)
        if np.max(extent) <= 1e-12:
            return
        axis_length = 0.95

        cell = np.asarray(self._atoms.cell, dtype=float)
        directions = (
            _normalized_vector(cell[0], np.array([1.0, 0.0, 0.0])),
            _normalized_vector(cell[1], np.array([0.0, 1.0, 0.0])),
            _normalized_vector(cell[2], np.array([0.0, 0.0, 1.0])),
        )
        labels = ("a", "b", "c")
        colors = ("#d73027", "#1a9850", "#4575b4")

        for direction, label, color in zip(directions, labels, colors):
            triad_axis.quiver(
                0.0,
                0.0,
                0.0,
                direction[0],
                direction[1],
                direction[2],
                length=axis_length,
                normalize=False,
                color=color,
                linewidth=2.0,
                arrow_length_ratio=0.22,
            )
            tip = direction * axis_length
            triad_axis.text(
                tip[0] + direction[0] * 0.14,
                tip[1] + direction[1] * 0.14,
                tip[2] + direction[2] * 0.14,
                label,
                color=color,
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
            )

        triad_axis.set_xlim(-1.05, 1.05)
        triad_axis.set_ylim(-1.05, 1.05)
        triad_axis.set_zlim(-1.05, 1.05)
        triad_axis.set_box_aspect((1.0, 1.0, 1.0))
        _view_init_with_roll(
            triad_axis,
            elev=self._view[0],
            azim=self._view[1],
            roll=self._view_roll,
        )

        for coord_axis in (triad_axis.xaxis, triad_axis.yaxis, triad_axis.zaxis):
            coord_axis.pane.set_alpha(0.0)

        self._triad_canvas.draw_idle()

    def _highlight_selected_atom(self, axis, positions: np.ndarray, radii: np.ndarray) -> None:
        assert self._selected_index is not None
        point = positions[self._selected_index]

        radius = float(radii[self._selected_index])
        halo_radius = 1.16 * radius
        shell_radius = 1.24 * radius

        u = np.linspace(0.0, 2.0 * np.pi, 16)
        v = np.linspace(0.0, np.pi, 16)
        cos_u = np.cos(u)
        sin_u = np.sin(u)
        sin_v = np.sin(v)
        cos_v = np.cos(v)

        halo_x = point[0] + halo_radius * np.outer(cos_u, sin_v)
        halo_y = point[1] + halo_radius * np.outer(sin_u, sin_v)
        halo_z = point[2] + halo_radius * np.outer(np.ones_like(u), cos_v)
        axis.plot_surface(
            halo_x,
            halo_y,
            halo_z,
            color="#ffcc33",
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=0.22,
            zorder=20,
        )

        shell_x = point[0] + shell_radius * np.outer(cos_u, sin_v)
        shell_y = point[1] + shell_radius * np.outer(sin_u, sin_v)
        shell_z = point[2] + shell_radius * np.outer(np.ones_like(u), cos_v)
        axis.plot_wireframe(
            shell_x,
            shell_y,
            shell_z,
            rstride=2,
            cstride=2,
            color="#f2a900",
            linewidth=0.6,
            alpha=0.9,
            zorder=21,
        )

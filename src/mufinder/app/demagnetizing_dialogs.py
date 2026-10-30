"""Magnetic Fields demagnetizing-field sample dialog and its 3D preview."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import (
    _configure_transparent_pyvista_background,
    _pyvista_modules,
    _shutdown_pyvista_widget,
)
from mufinder.app.magnetic_component_dialogs import DecimalTriple, direction_group, direction_inputs
from mufinder.app.sample_photo_dialog import (
    PREVIEW_DEBOUNCE_MS,
    SamplePhotoDialog,
    demag_length_input,
)
from mufinder.app.theme import (
    current_palette,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
)
from mufinder.science.demagnetization import (
    DemagnetizationError,
    DemagnetizingSample,
    SampleOrientation,
    fourier_demagnetizing_result,
    principal_demagnetizing_tensor,
    sample_orientation,
)

ADVANCED_DEMAG_SHAPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("ellipsoid", "Ellipsoid"),
    ("rectangular_prism", "Rectangular prism"),
    ("image_plate", "Plate from photo"),
)


# Offered only for a sample given as voxels through the Python API.
DEMAG_VOXEL_GRID_OPTION = ("voxel_grid", "Voxel grid")


ADVANCED_DEMAG_SHAPES = frozenset(
    {key for key, _label in ADVANCED_DEMAG_SHAPE_OPTIONS} | {DEMAG_VOXEL_GRID_OPTION[0]}
)


# The preview draws at most this many voxels, skipping evenly through larger grids.
_PREVIEW_MAX_VOXELS = 400_000


DEMAG_ORIENTATION_OPTIONS: tuple[tuple[str, str], ...] = (
    ("aligned", "Aligned with crystal axes"),
    ("directions", "Crystal directions"),
)


# Offered only for a sample whose rotation was saved without crystal directions.
DEMAG_SAVED_ROTATION_OPTION = ("saved", "Saved rotation")


def demag_sample_summary(sample: DemagnetizingSample) -> str:
    """Return a one-line description of an advanced sample's shape, without its orientation."""

    shape = str(sample.shape).strip().lower()
    if shape == "ellipsoid":
        a, b, c = sample.semi_axes
        return f"Ellipsoid a={a:g}, b={b:g}, c={c:g}"
    if shape == "rectangular_prism":
        a, b, c = sample.prism_lengths
        return f"Rectangular prism {a:g} x {b:g} x {c:g}"
    if shape == "image_plate":
        if sample.plate_mask is None:
            return "No outline yet. Click 'From photo…' to outline the sample in a photo."
        width, height = sample.plate_size
        text = f"Plate from photo {width:g} x {height:g} x {sample.plate_thickness:g}"
        if sample.photo is not None and sample.photo.path:
            text += f" ({Path(sample.photo.path).name})"
        return text
    if shape == DEMAG_VOXEL_GRID_OPTION[0]:
        counts = "unset" if sample.voxels is None else " x ".join(map(str, sample.voxels.shape))
        return f"Voxel grid {counts} voxels"
    return "No advanced shape selected."


class _DemagnetizingPyVistaCanvas(QWidget):
    """PyVista preview canvas for demagnetizing sample shapes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pv, qt_interactor = _pyvista_modules()
        if self._pv is None or qt_interactor is None:
            raise RuntimeError(
                "PyVista and pyvistaqt are required for demagnetizing-field visualisation."
            )
        self._sample: DemagnetizingSample | None = None
        self._lattice_vectors: np.ndarray | None = None
        # The last voxel surface and the shape it was built for: editing the
        # orientation only rotates it.
        self._voxel_mesh_cache: tuple[DemagnetizingSample, object] | None = None
        self._refreshing_theme = False
        self._style_signature = self._style_signature_for_current_palette()
        # auto_update=False disables pyvistaqt's always-on render_timer; we
        # render explicitly (draw_idle/plotter.render), so the ~5Hz timer is
        # pure CPU waste that runs even when this widget is not visible.
        self._plotter = qt_interactor(self, auto_update=False)
        _configure_transparent_pyvista_background(self._plotter)
        self._plotter.enable_parallel_projection()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plotter)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _style_signature_for_current_palette() -> tuple[str, ...]:
        palette = current_palette()
        return (
            palette.bg_primary,
            palette.status_running,
            palette.accent,
            palette.success,
            palette.status_submitting,
            palette.text_primary,
        )

    def set_sample(
        self, sample: DemagnetizingSample, lattice_vectors: Sequence[Sequence[float]] | None = None
    ) -> None:
        """Draw the sample rotated into the crystal frame, with arrows along the crystal axes.

        Without ``lattice_vectors`` the axis arrows follow Cartesian x, y and z.
        """

        self._sample = sample
        self._lattice_vectors = (
            None if lattice_vectors is None else np.asarray(lattice_vectors, dtype=float)
        )
        rotation = sample.rotation_matrix() if sample.rotation is not None else None
        plotter = self._plotter
        pv = self._pv
        plotter.clear()
        _configure_transparent_pyvista_background(plotter)
        shape = str(sample.shape).strip().lower()
        voxel_mesh = self._voxel_mesh(sample) if sample.uses_fourier_method else None
        if sample.uses_fourier_method:
            mesh, extent = voxel_mesh if voxel_mesh is not None else (None, 1.0)
            if mesh is not None and rotation is not None:
                transform = np.eye(4, dtype=float)
                transform[:3, :3] = rotation
                mesh = mesh.transform(transform, inplace=False)
        elif shape == "rectangular_prism":
            lengths = np.asarray(sample.prism_lengths, dtype=float)
            mesh = pv.Cube(
                bounds=(
                    -0.5 * lengths[0],
                    0.5 * lengths[0],
                    -0.5 * lengths[1],
                    0.5 * lengths[1],
                    -0.5 * lengths[2],
                    0.5 * lengths[2],
                )
            )
            extent = float(np.max(lengths))
            if rotation is not None:
                transform = np.eye(4, dtype=float)
                transform[:3, :3] = rotation
                mesh = mesh.transform(transform, inplace=False)
        else:
            axes = np.asarray(sample.semi_axes, dtype=float)
            mesh = self._ellipsoid_mesh(axes, rotation)
            extent = float(np.max(axes))

        palette = current_palette()
        if mesh is not None:
            plotter.add_mesh(
                mesh,
                color=palette.status_running,
                opacity=0.42,
                smooth_shading=not sample.uses_fourier_method,
            )
        axis_colors = (palette.accent, palette.success, palette.status_submitting)
        directions = np.eye(3, dtype=float)
        if self._lattice_vectors is not None and self._lattice_vectors.shape == (3, 3):
            lengths = np.linalg.norm(self._lattice_vectors, axis=1)
            if np.all(lengths > 0.0):
                directions = self._lattice_vectors / lengths[:, None]
        for direction, color, label in zip(directions, axis_colors, ("a", "b", "c"), strict=True):
            self._add_arrow((0.0, 0.0, 0.0), 1.25 * extent * direction, color, label)
        self._add_arrow(
            (0.0, 0.0, -0.18 * extent), (0.0, 0.0, 1.65 * extent), palette.text_primary, "M"
        )
        plotter.reset_camera()
        plotter.render()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        plotter = getattr(self, "_plotter", None)
        signature = self._style_signature_for_current_palette()
        if (
            self._refreshing_theme
            or plotter is None
            or signature == getattr(self, "_style_signature", None)
            or event.type()
            not in {
                QEvent.Type.StyleChange,
                QEvent.Type.PaletteChange,
                QEvent.Type.ApplicationPaletteChange,
            }
        ):
            return
        self._refreshing_theme = True
        try:
            self._style_signature = signature
            _configure_transparent_pyvista_background(plotter)
            if self._sample is not None:
                self.set_sample(self._sample, self._lattice_vectors)
            else:
                plotter.render()
        finally:
            self._refreshing_theme = False

    def shutdown(self) -> None:
        _shutdown_pyvista_widget(self._plotter)

    def _voxel_mesh(self, sample: DemagnetizingSample):
        """Return the surface of the sample's voxels, centred on the origin, and its half-extent.

        A plate is drawn one voxel thick, since its layers are identical. A
        sample without voxels yet (a plate with no photo) draws nothing.
        """

        shape_only = replace(sample, rotation=None, normal_hkl=None, in_plane_uvw=None, photo=None)
        cached = self._voxel_mesh_cache
        if cached is not None and cached[0] == shape_only:
            return cached[1]
        mesh = self._build_voxel_mesh(shape_only)
        self._voxel_mesh_cache = (shape_only, mesh)
        return mesh

    def _build_voxel_mesh(self, sample: DemagnetizingSample):
        try:
            if sample.shape == "image_plate":
                sample = replace(sample, thickness_voxels=1)
            model = sample.voxel_model()
        except DemagnetizationError:
            return None
        mask, spacing = model.mask, np.asarray(model.spacing, dtype=float)
        step = max(1, int(np.ceil((mask.size / _PREVIEW_MAX_VOXELS) ** (1.0 / 3.0))))
        if step > 1:
            mask = mask[::step, ::step, ::step]
            spacing = spacing * step
        grid = self._pv.ImageData(
            dimensions=tuple(int(n) + 1 for n in mask.shape),
            spacing=tuple(float(s) for s in spacing),
            origin=tuple(-0.5 * float(n) * float(s) for n, s in zip(mask.shape, spacing)),
        )
        grid.cell_data["inside"] = mask.ravel(order="F").astype(np.uint8)
        surface = grid.threshold(0.5, scalars="inside").extract_surface(algorithm="dataset_surface")
        return surface, 0.5 * float(np.max(np.asarray(mask.shape) * spacing))

    def _ellipsoid_mesh(self, axes: np.ndarray, rotation: np.ndarray | None = None):
        theta = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
        phi = np.linspace(0.0, np.pi, 36)
        tt, pp = np.meshgrid(theta, phi)
        points = np.column_stack(
            [
                (axes[0] * np.sin(pp) * np.cos(tt)).ravel(),
                (axes[1] * np.sin(pp) * np.sin(tt)).ravel(),
                (axes[2] * np.cos(pp)).ravel(),
            ]
        )
        if rotation is not None:
            points = points @ rotation.T
        faces: list[int] = []
        n_phi, n_theta = tt.shape
        for i in range(n_phi - 1):
            for j in range(n_theta):
                a = i * n_theta + j
                b = i * n_theta + ((j + 1) % n_theta)
                c = (i + 1) * n_theta + ((j + 1) % n_theta)
                d = (i + 1) * n_theta + j
                faces.extend([4, a, b, c, d])
        return self._pv.PolyData(points, np.asarray(faces, dtype=int))

    def _add_arrow(
        self, start: Sequence[float], vector: Sequence[float], color: str, label: str
    ) -> None:
        pv = self._pv
        start_arr = np.asarray(start, dtype=float)
        vector_arr = np.asarray(vector, dtype=float)
        length = float(np.linalg.norm(vector_arr))
        if length <= 1e-12:
            return
        arrow = pv.Arrow(
            start=tuple(start_arr),
            direction=tuple(vector_arr / length),
            # pv.Arrow builds a unit arrow and scales all of it by ``scale``.
            scale=length,
            shaft_radius=0.018,
            tip_radius=0.055,
            tip_length=0.18,
        )
        self._plotter.add_mesh(arrow, color=color)
        self._plotter.add_point_labels(
            np.asarray([start_arr + vector_arr], dtype=float),
            [label],
            text_color=color,
            font_size=14,
            point_size=0,
            shape_opacity=0.0,
        )


class DemagnetizingFieldAdvancedDialog(QDialog):
    """Advanced demagnetizing-field shape and tensor settings."""

    def __init__(
        self,
        sample: DemagnetizingSample | None = None,
        parent: QWidget | None = None,
        *,
        lattice_vectors: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Demagnetizing Field")
        self.resize(860, 560)
        self._preview: _DemagnetizingPyVistaCanvas | QLabel
        # The tensor follows every edit at once; the 3D preview redraws once edits pause.
        self._pending_preview: DemagnetizingSample | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._draw_pending_preview)
        self._lattice_vectors = (
            None if lattice_vectors is None else [list(row) for row in lattice_vectors]
        )

        sample = sample or DemagnetizingSample(shape="ellipsoid")
        if str(sample.shape).strip().lower() not in ADVANCED_DEMAG_SHAPES:
            sample = DemagnetizingSample(shape="ellipsoid")

        self._shape = QComboBox(self)
        for key, label in ADVANCED_DEMAG_SHAPE_OPTIONS:
            self._shape.addItem(label, key)
        if sample.shape == DEMAG_VOXEL_GRID_OPTION[0]:
            self._shape.addItem(DEMAG_VOXEL_GRID_OPTION[1], DEMAG_VOXEL_GRID_OPTION[0])
        index = self._shape.findData(sample.shape)
        self._shape.setCurrentIndex(index if index >= 0 else 0)

        self._ellipsoid_a = demag_length_input(sample.semi_axes[0])
        self._ellipsoid_b = demag_length_input(sample.semi_axes[1])
        self._ellipsoid_c = demag_length_input(sample.semi_axes[2])
        self._prism_a = demag_length_input(sample.prism_lengths[0])
        self._prism_b = demag_length_input(sample.prism_lengths[1])
        self._prism_c = demag_length_input(sample.prism_lengths[2])
        # The plate and voxel grid are edited as a whole: through the photo
        # dialog, or not at all.
        self._plate_sample = (
            sample if sample.shape == "image_plate" else DemagnetizingSample(shape="image_plate")
        )
        self._voxel_sample = sample if sample.shape == DEMAG_VOXEL_GRID_OPTION[0] else None
        self._plate_summary = QLabel("", self)
        self._plate_summary.setWordWrap(True)
        style_hint_text(self._plate_summary)
        self._photo_button = QPushButton("From photo…", self)
        self._photo_button.clicked.connect(self._open_photo_dialog)
        self._voxel_summary = QLabel("", self)
        self._voxel_summary.setWordWrap(True)
        style_hint_text(self._voxel_summary)

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._ellipsoid_page())
        self._stack.addWidget(self._prism_page())
        self._stack.addWidget(self._plate_page())
        self._stack.addWidget(self._voxel_page())

        # The seed's orientation is kept whenever the dialog cannot edit it: a
        # rotation saved without crystal directions, or directions with no host.
        self._seed = sample
        self._principal: tuple[DemagnetizingSample, np.ndarray] | None = None
        self._orientation_mode = QComboBox(self)
        for key, label in DEMAG_ORIENTATION_OPTIONS:
            self._orientation_mode.addItem(label, key)
        if sample.rotation is not None and sample.normal_hkl is None:
            self._orientation_mode.addItem(
                DEMAG_SAVED_ROTATION_OPTION[1], DEMAG_SAVED_ROTATION_OPTION[0]
            )
            initial_mode = DEMAG_SAVED_ROTATION_OPTION[0]
        else:
            initial_mode = "directions" if sample.normal_hkl is not None else "aligned"
        self._orientation_mode.setCurrentIndex(self._orientation_mode.findData(initial_mode))
        self._orientation_mode.setEnabled(self._lattice_vectors is not None)
        self._normal_inputs = direction_inputs(
            sample.normal_hkl or (0.0, 0.0, 1.0), placeholders="hkl"
        )
        self._in_plane_inputs = direction_inputs(
            sample.in_plane_uvw or (1.0, 0.0, 0.0), placeholders="uvw"
        )
        self._orientation_hint = QLabel("", self)
        self._orientation_hint.setWordWrap(True)
        style_hint_text(self._orientation_hint)

        self._tensor_label = QLabel("", self)
        self._tensor_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        style_hint_text(self._tensor_label)

        try:
            self._preview = _DemagnetizingPyVistaCanvas(self)
        except RuntimeError as exc:
            self._preview = QLabel(str(exc), self)
            self._preview.setWordWrap(True)
            self._preview.setMinimumSize(360, 280)
            style_hint_text(self._preview)

        title = QLabel("Advanced Demagnetizing Field", self)
        style_section_title(title)
        subtitle = QLabel(
            "Choose a sample shape, inspect the demagnetizing tensor, and preview the geometry used for the field correction.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)

        form = QFormLayout()
        form.addRow("Shape", self._shape)
        form.addRow("Parameters", self._stack)

        orientation_group = QGroupBox("Orientation", self)
        orientation_form = QFormLayout(orientation_group)
        orientation_form.addRow("Sample axes", self._orientation_mode)
        self._index_rows: list[QWidget] = []
        for label, inputs in (
            ("Normal (h k l)", self._normal_inputs),
            ("In-plane axis [u v w]", self._in_plane_inputs),
        ):
            label_widget = QLabel(label, orientation_group)
            group = direction_group(inputs)
            orientation_form.addRow(label_widget, group)
            self._index_rows.extend((label_widget, group))
        orientation_form.addRow(self._orientation_hint)

        left = QVBoxLayout()
        left.addWidget(title)
        left.addWidget(subtitle)
        left.addLayout(form)
        left.addWidget(orientation_group)
        left.addWidget(self._tensor_label)
        left.addStretch(1)

        body = QHBoxLayout()
        body.addLayout(left, 0)
        body.addWidget(self._preview, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._use_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._use_button.setText("Use Shape")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(body)
        layout.addWidget(buttons)

        self._shape.currentIndexChanged.connect(self._sync)
        for widget in (
            self._ellipsoid_a,
            self._ellipsoid_b,
            self._ellipsoid_c,
            self._prism_a,
            self._prism_b,
            self._prism_c,
        ):
            widget.valueChanged.connect(self._sync)
        self._orientation_mode.currentIndexChanged.connect(self._sync)
        for edit in (*self._normal_inputs, *self._in_plane_inputs):
            edit.textChanged.connect(self._sync)
        self._sync()
        # The dialog opens with its preview drawn.
        self._draw_pending_preview()

    def selected_sample(self) -> DemagnetizingSample:
        """Return the configured sample, oriented when ``Crystal directions`` is chosen.

        Raises :class:`DemagnetizationError` for unusable crystal directions.
        """

        return self._resolve()[0]

    def _resolve(self) -> tuple[DemagnetizingSample, SampleOrientation | None]:
        """Return the sample and, for directions set here, their resolved orientation."""

        sample = self._shape_sample()
        mode = self._orientation_key()
        if mode == "aligned":
            return sample, None
        seed = self._seed
        if mode == DEMAG_SAVED_ROTATION_OPTION[0] or self._lattice_vectors is None:
            return (
                replace(
                    sample,
                    rotation=seed.rotation,
                    normal_hkl=seed.normal_hkl,
                    in_plane_uvw=seed.in_plane_uvw,
                ),
                None,
            )
        normal = self._index_values(self._normal_inputs, "normal (h k l)")
        in_plane = self._index_values(self._in_plane_inputs, "in-plane axis [u v w]")
        orientation = sample_orientation(self._lattice_vectors, normal, in_plane)
        return (
            replace(
                sample, rotation=orientation.rotation, normal_hkl=normal, in_plane_uvw=in_plane
            ),
            orientation,
        )

    def _shape_sample(self) -> DemagnetizingSample:
        shape = str(self._shape.currentData() or "ellipsoid")
        if shape == "image_plate":
            return replace(self._plate_sample, rotation=None, normal_hkl=None, in_plane_uvw=None)
        if shape == DEMAG_VOXEL_GRID_OPTION[0] and self._voxel_sample is not None:
            return replace(self._voxel_sample, rotation=None, normal_hkl=None, in_plane_uvw=None)
        return DemagnetizingSample(
            shape=shape,
            semi_axes=(
                float(self._ellipsoid_a.value()),
                float(self._ellipsoid_b.value()),
                float(self._ellipsoid_c.value()),
            ),
            prism_lengths=(
                float(self._prism_a.value()),
                float(self._prism_b.value()),
                float(self._prism_c.value()),
            ),
        )

    def _open_photo_dialog(self) -> None:
        dialog = SamplePhotoDialog(self._plate_sample, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._plate_sample = dialog.selected_sample()
        self._sync()

    def done(self, result: int) -> None:
        """Close the dialog; a preview still waiting for edits to pause is not drawn."""

        self._preview_timer.stop()
        self._pending_preview = None
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._preview_timer.stop()
        self._pending_preview = None
        if isinstance(self._preview, _DemagnetizingPyVistaCanvas):
            self._preview.shutdown()
        super().closeEvent(event)

    def _ellipsoid_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        form.addRow("Semi-axis a", self._ellipsoid_a)
        form.addRow("Semi-axis b", self._ellipsoid_b)
        form.addRow("Semi-axis c", self._ellipsoid_c)
        return page

    def _prism_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        form.addRow("Length a", self._prism_a)
        form.addRow("Length b", self._prism_b)
        form.addRow("Length c", self._prism_c)
        return page

    def _plate_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plate_summary)
        row = QHBoxLayout()
        row.addWidget(self._photo_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _voxel_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._voxel_summary)
        layout.addStretch(1)
        return page

    def _sync_voxel_summaries(self) -> None:
        self._plate_summary.setText(demag_sample_summary(self._plate_sample))
        self._photo_button.setText(
            "From photo…" if self._plate_sample.plate_mask is None else "Edit outline…"
        )
        if self._voxel_sample is not None:
            self._voxel_summary.setText(
                demag_sample_summary(self._voxel_sample)
                + ". A voxel grid is set through the Python API and cannot be edited here."
            )

    def _orientation_key(self) -> str:
        return str(self._orientation_mode.currentData() or "aligned")

    @staticmethod
    def _index_values(inputs: DecimalTriple, name: str) -> tuple[float, float, float]:
        try:
            h, k, l_ = (float(edit.text().strip()) for edit in inputs)
        except ValueError as exc:
            raise DemagnetizationError(f"enter three numbers for the {name}") from exc
        return (h, k, l_)

    def _orientation_text(self, orientation: SampleOrientation | None) -> str:
        mode = self._orientation_key()
        if mode == "aligned":
            text = "The sample axes a, b and c lie along Cartesian x, y and z."
            if self._lattice_vectors is None:
                text += " Orienting the sample by crystal directions needs a host structure."
            return text
        if mode == DEMAG_SAVED_ROTATION_OPTION[0]:
            return (
                "The sample axes follow a rotation saved without crystal directions. Choose "
                "Crystal directions to set them by (h k l) and [u v w]."
            )
        if self._lattice_vectors is None:
            return "The saved crystal directions are kept; editing them needs a host structure."
        text = "The sample c axis is normal to (h k l) and its a axis lies along [u v w]."
        if orientation is not None and not orientation.orthogonal:
            text += (
                f" [u v w] is {orientation.angle_deg:.2f}° from the normal, so the a axis "
                "follows its component in the plane."
            )
        return text

    def _principal_tensor(self, sample: DemagnetizingSample) -> np.ndarray:
        """Return the shape's principal tensor, reused while only the orientation changes."""

        shape = replace(sample, rotation=None, normal_hkl=None, in_plane_uvw=None)
        if self._principal is None or self._principal[0] != shape:
            self._principal = (shape, principal_demagnetizing_tensor(shape))
        return self._principal[1]

    def _sync(self, *_args: object) -> None:
        shape = str(self._shape.currentData() or "ellipsoid")
        pages = ("ellipsoid", "rectangular_prism", "image_plate", DEMAG_VOXEL_GRID_OPTION[0])
        self._stack.setCurrentIndex(pages.index(shape) if shape in pages else 0)
        self._sync_voxel_summaries()
        editable = self._orientation_key() == "directions" and self._lattice_vectors is not None
        for row in self._index_rows:
            row.setEnabled(editable)
        sample = self._shape_sample()
        try:
            sample, orientation = self._resolve()
            self._orientation_hint.setText(self._orientation_text(orientation))
        except DemagnetizationError as exc:
            self._orientation_hint.setText(f"Orientation unavailable: {exc}")
            self._tensor_label.setText("Tensor unavailable until the orientation is valid.")
            self._use_button.setEnabled(False)
            self._schedule_preview(sample)
            return
        try:
            tensor = sample.rotate(self._principal_tensor(sample))
        except DemagnetizationError as exc:
            self._tensor_label.setText(f"Tensor unavailable: {exc}")
            self._use_button.setEnabled(False)
        else:
            # Elements that round to zero print as 0.00000, not -0.00000.
            tensor = np.where(np.abs(tensor) < 5e-6, 0.0, tensor)
            text = (
                "N = "
                f"[[{tensor[0, 0]:.5f}, {tensor[0, 1]:.5f}, {tensor[0, 2]:.5f}], "
                f"[{tensor[1, 0]:.5f}, {tensor[1, 1]:.5f}, {tensor[1, 2]:.5f}], "
                f"[{tensor[2, 0]:.5f}, {tensor[2, 1]:.5f}, {tensor[2, 2]:.5f}]]"
            )
            if sample.uses_fourier_method:
                result = fourier_demagnetizing_result(sample)
                box = " x ".join(str(n) for n in result.box_shape)
                text += (
                    f"\nTr N = {result.trace:.5f} from {result.voxel_count} voxels in a "
                    f"{box} FFT box."
                )
            self._tensor_label.setText(text)
            self._use_button.setEnabled(True)
        self._schedule_preview(sample)

    def _schedule_preview(self, sample: DemagnetizingSample) -> None:
        """Show ``sample`` in the 3D preview once no edit has come for ``PREVIEW_DEBOUNCE_MS``."""

        self._pending_preview = sample
        self._preview_timer.start()

    def _draw_pending_preview(self) -> None:
        self._preview_timer.stop()
        sample, self._pending_preview = self._pending_preview, None
        if sample is not None and isinstance(self._preview, _DemagnetizingPyVistaCanvas):
            self._preview.set_sample(sample, self._lattice_vectors)

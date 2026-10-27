"""Outline a plate-like sample in a photograph for its demagnetizing tensor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage

from mufinder.app.theme import (
    current_palette,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
)
from mufinder.science.demagnetization import (
    DemagnetizationError,
    DemagnetizingSample,
    PlatePhoto,
)
from mufinder.science.sample_image import (
    PlateSegmentation,
    SampleImageError,
    in_plane_to_image,
    otsu_threshold,
    segment_plate_image,
)

# Photos are segmented at this size at most; the saved outline is smaller still.
PHOTO_WORKING_EDGE = 1024
PHOTO_FILE_FILTER = "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)"
_PREVIEW_SIZE = (440, 330)


def load_greyscale_image(path: str | Path, *, max_edge: int = PHOTO_WORKING_EDGE) -> np.ndarray:
    """Return an image file as a 2-D ``uint8`` grey-level array, rows top to bottom.

    Images larger than ``max_edge`` on their longer side are scaled down
    smoothly first. Raises :class:`ValueError` for a file Qt cannot read.
    """

    image = QImage(str(path))
    if image.isNull():
        raise ValueError(f"could not read an image from {path}")
    if max(image.width(), image.height()) > max_edge:
        image = image.scaled(
            max_edge,
            max_edge,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    grey = image.convertToFormat(QImage.Format.Format_Grayscale8)
    width, height, stride = grey.width(), grey.height(), grey.bytesPerLine()
    buffer = np.frombuffer(grey.constBits(), dtype=np.uint8, count=stride * height)
    return buffer.reshape(height, stride)[:, :width].copy()


def demag_length_input(value: float, *, step: float = 0.25) -> QDoubleSpinBox:
    """Return a spin box for a sample length; only ratios of lengths matter, so the unit is arbitrary."""

    spinbox = QDoubleSpinBox()
    spinbox.setRange(0.0001, 1_000_000.0)
    spinbox.setDecimals(4)
    spinbox.setSingleStep(step)
    spinbox.setValue(float(value))
    spinbox.setSuffix(" arb.")
    return spinbox


def _grey_qimage(grey: np.ndarray) -> QImage:
    data = np.ascontiguousarray(grey, dtype=np.uint8)
    height, width = data.shape
    return QImage(data.data, width, height, width, QImage.Format.Format_Grayscale8).copy()


def _overlay_qimage(mask: np.ndarray, color: QColor) -> QImage:
    """Return the outline as an RGBA image: a translucent fill with an opaque edge."""

    edge = mask & ~ndimage.binary_erosion(mask, border_value=0)
    edge = ndimage.binary_dilation(edge, iterations=max(1, max(mask.shape) // 300))
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color.red(), color.green(), color.blue()
    rgba[..., 3] = np.where(edge, 255, np.where(mask, 70, 0)).astype(np.uint8)
    height, width = mask.shape
    return QImage(rgba.data, width, height, 4 * width, QImage.Format.Format_RGBA8888).copy()


class SamplePhotoDialog(QDialog):
    """Pick a photograph, segment the sample's outline and set the plate's size and grid.

    The outline is drawn over the photo. The threshold starts at the Otsu
    level; ``Invert`` swaps sample and background and ``Fill holes`` fills
    background enclosed by the sample. A seed whose photo is no longer on disk
    keeps its saved outline, which is shown instead and cannot be re-segmented.
    """

    def __init__(
        self, sample: DemagnetizingSample | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plate from Photo")
        seed = sample if sample is not None and sample.shape == "image_plate" else None
        photo = seed.photo if seed is not None and seed.photo is not None else PlatePhoto()
        self._grey: np.ndarray | None = None
        self._segmentation: PlateSegmentation | None = None
        self._saved_outline = (
            None if seed is None or seed.plate_mask is None else seed.plate_mask.array()
        )
        self._threshold: float | None = photo.threshold
        self._updating = False

        self._path = QLineEdit(self)
        self._path.setReadOnly(True)
        self._path.setPlaceholderText("No photo chosen")
        self._choose_button = QPushButton("Choose...", self)
        self._choose_button.clicked.connect(self._choose_photo)

        self._image = QLabel(self)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image.setMinimumSize(*_PREVIEW_SIZE)
        self._image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image.setText("Choose a photo of the sample on a plain background.")
        style_hint_text(self._image)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._threshold_slider.setRange(0, 255)
        self._threshold_value = QLabel("", self)
        self._threshold_value.setMinimumWidth(80)
        self._auto_button = QPushButton("Auto", self)
        self._auto_button.setToolTip("Reset the threshold to the Otsu level of the photo.")
        self._auto_button.clicked.connect(self._use_auto_threshold)
        self._invert = QCheckBox("Invert", self)
        self._invert.setToolTip("Swap sample and background.")
        self._invert.setChecked(photo.invert)
        self._fill_holes = QCheckBox("Fill holes", self)
        self._fill_holes.setToolTip(
            "Count background enclosed by the sample, such as glare, as sample."
        )
        self._fill_holes.setChecked(photo.fill_holes)

        defaults = seed or DemagnetizingSample(shape="image_plate")
        self._width = demag_length_input(defaults.plate_size[0], step=0.1)
        self._height = demag_length_input(defaults.plate_size[1], step=0.1)
        self._match_button = QPushButton("Match photo", self)
        self._match_button.setToolTip(
            "Set the height from the width and the outline's aspect ratio in the photo."
        )
        self._match_button.clicked.connect(self._match_photo_aspect)
        self._thickness = demag_length_input(defaults.plate_thickness, step=0.1)
        self._grid = QSpinBox(self)
        self._grid.setRange(16, 256)
        self._grid.setSingleStep(16)
        self._grid.setValue(int(defaults.fourier_grid))
        self._grid.setToolTip("Voxels along the longer in-plane edge.")
        self._layers = QSpinBox(self)
        self._layers.setRange(1, 64)
        self._layers.setValue(int(defaults.thickness_voxels))
        self._layers.setToolTip("Voxels through the thickness.")
        self._padding = QDoubleSpinBox(self)
        self._padding.setRange(2.0, 8.0)
        self._padding.setDecimals(2)
        self._padding.setSingleStep(0.25)
        self._padding.setValue(max(2.0, float(defaults.padding)))
        self._padding.setToolTip(
            "Edge of the FFT box along every axis, in units of the longest sample edge."
        )
        self._smoothing = QDoubleSpinBox(self)
        self._smoothing.setRange(0.0, 4.0)
        self._smoothing.setDecimals(2)
        self._smoothing.setSingleStep(0.25)
        self._smoothing.setValue(float(defaults.smoothing))
        self._smoothing.setSuffix(" voxels")
        self._smoothing.setToolTip(
            "Width of a Gaussian filter that damps ringing; 0 turns it off. It lowers Tr N."
        )

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        style_hint_text(self._status)

        title = QLabel("Plate from Photo", self)
        style_section_title(title)
        subtitle = QLabel(
            "Outline the sample in a photo taken along its normal, then give the size of the "
            "outline's bounding box and the plate's thickness.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)

        path_row = QHBoxLayout()
        path_row.addWidget(self._path, 1)
        path_row.addWidget(self._choose_button)
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self._threshold_slider, 1)
        threshold_row.addWidget(self._threshold_value)
        threshold_row.addWidget(self._auto_button)
        switches = QHBoxLayout()
        switches.addWidget(self._invert)
        switches.addWidget(self._fill_holes)
        switches.addStretch(1)
        height_row = QHBoxLayout()
        height_row.addWidget(self._height, 1)
        height_row.addWidget(self._match_button)

        segment_form = QFormLayout()
        segment_form.addRow("Photo", path_row)
        segment_form.addRow("Threshold", threshold_row)
        segment_form.addRow("", switches)
        size_form = QFormLayout()
        size_form.addRow("Width (along a)", self._width)
        size_form.addRow("Height (along b)", height_row)
        size_form.addRow("Thickness (along c)", self._thickness)
        size_form.addRow("Fourier grid", self._grid)
        size_form.addRow("Voxels through thickness", self._layers)
        size_form.addRow("Padding", self._padding)
        size_form.addRow("Smoothing", self._smoothing)

        left = QVBoxLayout()
        left.addWidget(title)
        left.addWidget(subtitle)
        left.addLayout(segment_form)
        left.addLayout(size_form)
        left.addWidget(self._status)
        left.addStretch(1)
        body = QHBoxLayout()
        body.addLayout(left, 0)
        body.addWidget(self._image, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._use_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._use_button.setText("Use Outline")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(body)
        layout.addWidget(buttons)

        self._threshold_slider.valueChanged.connect(self._on_threshold_moved)
        self._invert.toggled.connect(self._resegment)
        self._fill_holes.toggled.connect(self._resegment)

        if photo.path and Path(photo.path).is_file():
            self.load_photo(photo.path, keep_settings=True)
        else:
            self._show_saved_outline(photo.path)

    def selected_sample(self) -> DemagnetizingSample:
        """Return the plate as an ``image_plate`` sample along its own axes.

        Raises :class:`DemagnetizationError` when there is no outline.
        """

        outline = self._outline()
        if outline is None:
            raise DemagnetizationError("choose a photo with the sample in it")
        return DemagnetizingSample.image_plate(
            outline,
            (float(self._width.value()), float(self._height.value())),
            float(self._thickness.value()),
            fourier_grid=int(self._grid.value()),
            thickness_voxels=int(self._layers.value()),
            padding=float(self._padding.value()),
            smoothing=float(self._smoothing.value()),
            photo=PlatePhoto(
                path=self._path.text(),
                threshold=self._threshold,
                invert=self._invert.isChecked(),
                fill_holes=self._fill_holes.isChecked(),
            ),
        )

    def load_photo(self, path: str | Path, *, keep_settings: bool = False) -> bool:
        """Load and segment a photo; report a file Qt cannot read in the status line.

        Without ``keep_settings`` the threshold returns to the Otsu level and
        the height is matched to the outline.
        """

        try:
            grey = load_greyscale_image(path)
        except ValueError as exc:
            self._status.setText(str(exc).capitalize() + ".")
            return False
        self._grey = grey
        self._saved_outline = None
        self._path.setText(str(path))
        if not keep_settings:
            self._threshold = None
            self._invert.setChecked(False)
        self._resegment()
        if not keep_settings and self._segmentation is not None:
            self._match_photo_aspect()
        return True

    def _choose_photo(self) -> None:
        start = str(Path(self._path.text()).parent) if self._path.text() else ""
        path, _selected = QFileDialog.getOpenFileName(
            self, "Choose Sample Photo", start, PHOTO_FILE_FILTER
        )
        if path:
            self.load_photo(path)

    def _use_auto_threshold(self) -> None:
        self._threshold = None
        self._resegment()

    def _on_threshold_moved(self, value: int) -> None:
        if self._updating:
            return
        self._threshold = float(value)
        self._resegment()

    def _match_photo_aspect(self) -> None:
        if self._segmentation is None:
            return
        self._height.setValue(float(self._width.value()) * self._segmentation.aspect_ratio)

    def _resegment(self, *_args: object) -> None:
        if self._grey is None:
            return
        self._segmentation = None
        try:
            level = otsu_threshold(self._grey) if self._threshold is None else self._threshold
            self._set_slider(level)
            self._segmentation = segment_plate_image(
                self._grey,
                threshold=level,
                invert=self._invert.isChecked(),
                fill_holes=self._fill_holes.isChecked(),
            )
        except SampleImageError as exc:
            self._status.setText(f"No outline: {exc}.")
            self._render(self._grey, None)
            self._use_button.setEnabled(False)
            return
        r0, r1, c0, c1 = self._segmentation.bounds
        source = "Otsu threshold" if self._threshold is None else "threshold"
        polarity = "bright" if self._segmentation.sample_is_bright else "dark"
        self._status.setText(
            f"The outline is {c1 - c0} x {r1 - r0} pixels, a {polarity} sample at the {source}."
        )
        self._render(self._grey, self._segmentation.mask)
        self._use_button.setEnabled(True)

    def _show_saved_outline(self, missing_path: str) -> None:
        segmenting = (
            self._threshold_slider,
            self._auto_button,
            self._invert,
            self._fill_holes,
            self._match_button,
        )
        for widget in segmenting:
            widget.setEnabled(False)
        if self._saved_outline is None:
            self._use_button.setEnabled(False)
            return
        self._path.setText(missing_path)
        image_mask = in_plane_to_image(self._saved_outline)
        self._render(np.where(image_mask, 90, 225).astype(np.uint8), image_mask)
        where = f"The photo {missing_path} is not available" if missing_path else "No photo"
        self._status.setText(
            f"{where}; the saved outline is shown and kept. Choose a photo to outline it again."
        )
        self._use_button.setEnabled(True)

    def _outline(self) -> np.ndarray | None:
        if self._segmentation is not None:
            return self._segmentation.in_plane_mask()
        if self._grey is None:
            return self._saved_outline
        return None

    def _set_slider(self, level: float) -> None:
        self._updating = True
        try:
            self._threshold_slider.setValue(int(round(level)))
        finally:
            self._updating = False
        self._threshold_value.setText(
            f"{level:.1f}" + (" (auto)" if self._threshold is None else "")
        )

    def _render(self, grey: np.ndarray, mask: np.ndarray | None) -> None:
        pixmap = QPixmap.fromImage(_grey_qimage(grey).convertToFormat(QImage.Format.Format_RGB32))
        if mask is not None:
            painter = QPainter(pixmap)
            painter.drawImage(0, 0, _overlay_qimage(mask, QColor(current_palette().accent)))
            painter.end()
        self._image.setPixmap(
            pixmap.scaled(
                *_PREVIEW_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

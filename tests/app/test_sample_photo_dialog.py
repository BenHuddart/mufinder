"""Tests for outlining a photographed plate for its demagnetizing tensor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QImage

import mufinder.app.sample_photo_dialog as photo_module
from mufinder.app.sample_photo_dialog import SamplePhotoDialog, load_greyscale_image
from mufinder.science.demagnetization import (
    DemagnetizationError,
    DemagnetizingSample,
    PlatePhoto,
    demagnetizing_tensor,
)


def _write_png(path: Path, grey: np.ndarray) -> Path:
    data = np.ascontiguousarray(grey, dtype=np.uint8)
    height, width = data.shape
    image = QImage(data.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    assert image.save(str(path))
    return path


def _plate_photo(tmp_path: Path, *, name: str = "plate.png", hole: bool = False) -> Path:
    """A dark 60 x 30 pixel rectangle centred on a 100 x 80 light background."""

    grey = np.full((80, 100), 220, dtype=np.uint8)
    grey[25:55, 20:80] = 40
    if hole:
        grey[35:45, 45:55] = 220
    return _write_png(tmp_path / name, grey)


def test_load_greyscale_image_reads_rows_top_to_bottom(qapp, tmp_path) -> None:
    grey = np.zeros((4, 6), dtype=np.uint8)
    grey[0, 5] = 255
    path = _write_png(tmp_path / "corner.png", grey)

    np.testing.assert_array_equal(load_greyscale_image(path), grey)


def test_load_greyscale_image_scales_large_photos_down(qapp, tmp_path) -> None:
    path = _write_png(tmp_path / "big.png", np.full((300, 600), 128, dtype=np.uint8))

    assert load_greyscale_image(path, max_edge=200).shape == (100, 200)


def test_load_greyscale_image_rejects_unreadable_files(qapp, tmp_path) -> None:
    path = tmp_path / "notes.png"
    path.write_text("not an image")

    with pytest.raises(ValueError, match="could not read an image"):
        load_greyscale_image(path)


def test_photo_dialog_segments_a_dark_plate_and_matches_its_aspect(qapp, tmp_path) -> None:
    dialog = SamplePhotoDialog()
    assert not dialog._use_button.isEnabled()

    assert dialog.load_photo(_plate_photo(tmp_path))
    dialog._width.setValue(6.0)
    dialog._match_photo_aspect()
    dialog._thickness.setValue(0.1)

    assert dialog._use_button.isEnabled()
    assert "60 x 30 pixels, a dark sample at the Otsu threshold" in dialog._status.text()
    assert "(auto)" in dialog._threshold_value.text()
    assert dialog._height.value() == pytest.approx(3.0)
    assert dialog._image.pixmap() is not None
    sample = dialog.selected_sample()
    assert sample.shape == "image_plate"
    assert sample.plate_mask is not None and sample.plate_mask.shape == (60, 30)
    assert sample.plate_mask.array().all()
    assert sample.plate_size == pytest.approx((6.0, 3.0))
    assert sample.photo == PlatePhoto(path=str(tmp_path / "plate.png"))
    assert demagnetizing_tensor(sample)[2, 2] > 0.9
    dialog.close()


def test_photo_dialog_threshold_invert_and_holes_change_the_outline(qapp, tmp_path) -> None:
    dialog = SamplePhotoDialog()
    dialog.load_photo(_plate_photo(tmp_path, hole=True))
    filled = dialog.selected_sample().plate_mask

    dialog._fill_holes.setChecked(False)
    holed = dialog.selected_sample().plate_mask
    assert holed is not None and filled is not None
    assert holed.count == filled.count - 100

    dialog._threshold_slider.setValue(10)
    # Nothing is darker than 10, so the centre is light and the whole frame is taken.
    assert "No outline" in dialog._status.text()
    assert not dialog._use_button.isEnabled()
    with pytest.raises(DemagnetizationError, match="choose a photo"):
        dialog.selected_sample()

    dialog._use_auto_threshold()
    dialog._invert.setChecked(True)
    inverted = dialog.selected_sample()
    assert inverted.photo is not None and inverted.photo.invert
    assert inverted.photo.threshold is None
    # The background frame is the outline now, spanning the whole photo.
    assert "100 x 80 pixels, a bright sample" in dialog._status.text()
    dialog.close()


def test_photo_dialog_records_a_moved_threshold(qapp, tmp_path) -> None:
    dialog = SamplePhotoDialog()
    dialog.load_photo(_plate_photo(tmp_path))

    dialog._threshold_slider.setValue(150)

    photo = dialog.selected_sample().photo
    assert photo is not None and photo.threshold == 150.0
    assert "(auto)" not in dialog._threshold_value.text()
    dialog.close()


def test_photo_dialog_reopens_a_saved_plate_with_its_settings(qapp, tmp_path) -> None:
    path = _plate_photo(tmp_path)
    seed = DemagnetizingSample.image_plate(
        np.ones((60, 30), dtype=bool),
        (6.0, 3.0),
        0.2,
        fourier_grid=64,
        thickness_voxels=3,
        padding=3.0,
        smoothing=0.5,
        photo=PlatePhoto(path=str(path), threshold=150.0, invert=False, fill_holes=False),
    )

    dialog = SamplePhotoDialog(seed)

    assert dialog._threshold_slider.value() == 150
    assert not dialog._fill_holes.isChecked()
    assert (dialog._width.value(), dialog._height.value()) == pytest.approx((6.0, 3.0))
    assert dialog._grid.value() == 64
    assert dialog._layers.value() == 3
    assert dialog._padding.value() == pytest.approx(3.0)
    assert dialog._smoothing.value() == pytest.approx(0.5)
    assert dialog.selected_sample() == seed
    dialog.close()


def test_photo_dialog_keeps_the_saved_outline_when_the_photo_is_gone(qapp, tmp_path) -> None:
    outline = np.zeros((20, 10), dtype=bool)
    outline[2:18, 1:9] = True
    seed = DemagnetizingSample.image_plate(
        outline, (4.0, 2.0), 0.1, photo=PlatePhoto(path=str(tmp_path / "moved.png"))
    )

    dialog = SamplePhotoDialog(seed)

    assert "is not available; the saved outline is shown and kept" in dialog._status.text()
    assert not dialog._threshold_slider.isEnabled()
    assert not dialog._invert.isEnabled()
    assert dialog._use_button.isEnabled()
    assert dialog.selected_sample().plate_mask == seed.plate_mask
    dialog.close()


def test_photo_dialog_reports_an_unreadable_photo_and_keeps_the_outline(qapp, tmp_path) -> None:
    dialog = SamplePhotoDialog()
    dialog.load_photo(_plate_photo(tmp_path))
    before = dialog.selected_sample()
    broken = tmp_path / "broken.png"
    broken.write_text("not an image")

    assert not dialog.load_photo(broken)

    assert "Could not read an image" in dialog._status.text()
    assert dialog.selected_sample() == before
    dialog.close()


def test_photo_dialog_choose_loads_the_picked_file(qapp, tmp_path, monkeypatch) -> None:
    path = _plate_photo(tmp_path, name="picked.png")
    monkeypatch.setattr(
        photo_module.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: (str(path), "")),
    )
    dialog = SamplePhotoDialog()

    dialog._choose_photo()

    assert dialog._path.text() == str(path)
    assert dialog._use_button.isEnabled()
    dialog.close()


def test_photo_dialog_choose_cancelled_changes_nothing(qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        photo_module.QFileDialog,
        "getOpenFileName",
        staticmethod(lambda *args, **kwargs: ("", "")),
    )
    dialog = SamplePhotoDialog()

    dialog._choose_photo()

    assert dialog._path.text() == ""
    assert not dialog._use_button.isEnabled()
    dialog.close()


def test_photo_dialog_reports_the_fft_box_and_refuses_one_too_large(qapp, tmp_path) -> None:
    dialog = SamplePhotoDialog()
    dialog.load_photo(_plate_photo(tmp_path))
    dialog._width.setValue(6.0)
    dialog._match_photo_aspect()
    dialog._thickness.setValue(0.1)
    dialog._grid.setValue(64)

    assert dialog._box_status.text().startswith("The FFT box is 160 x 160 x 150,")
    assert dialog._use_button.isEnabled()

    dialog._thickness.setValue(0.0001)
    dialog._layers.setValue(64)

    assert "Cannot calculate: the FFT box would hold" in dialog._box_status.text()
    assert "lower the Fourier grid" in dialog._box_status.text()
    assert not dialog._use_button.isEnabled()

    dialog._layers.setValue(1)
    dialog._thickness.setValue(0.5)
    assert dialog._box_status.text().startswith("The FFT box is")
    assert dialog._use_button.isEnabled()
    dialog.close()


def test_photo_dialog_without_an_outline_reports_no_box(qapp) -> None:
    dialog = SamplePhotoDialog()

    assert dialog._box_status.text() == ""
    assert not dialog._use_button.isEnabled()
    dialog.close()


def test_choosing_a_photo_after_a_missing_one_reenables_segmentation(qapp, tmp_path) -> None:
    seed = DemagnetizingSample.image_plate(
        np.ones((20, 10), dtype=bool),
        (4.0, 2.0),
        0.1,
        photo=PlatePhoto(path=str(tmp_path / "moved.png")),
    )
    dialog = SamplePhotoDialog(seed)
    assert not dialog._threshold_slider.isEnabled()

    dialog.load_photo(_plate_photo(tmp_path))

    for widget in (
        dialog._threshold_slider,
        dialog._auto_button,
        dialog._invert,
        dialog._fill_holes,
        dialog._match_button,
    ):
        assert widget.isEnabled()
    dialog.close()

"""Tests for segmenting a photographed plate into an in-plane outline."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.science.sample_image import (
    SampleImageError,
    extrude_plate,
    fit_mask,
    image_to_in_plane,
    in_plane_to_image,
    otsu_threshold,
    resample_mask,
    segment_plate_image,
)


def _photo(*, bright_sample: bool, hole: bool = False, noise: float = 0.0) -> np.ndarray:
    """A 60 x 80 photo of a rectangle in rows 10-40 and columns 20-70, on a plain background."""

    image = np.full((60, 80), 40.0 if bright_sample else 200.0)
    image[10:40, 20:70] = 200.0 if bright_sample else 40.0
    if hole:
        image[20:25, 40:45] = 40.0 if bright_sample else 200.0
    if noise:
        image += np.random.default_rng(7).normal(0.0, noise, image.shape)
    return image


def test_otsu_threshold_splits_a_two_level_image_in_the_middle() -> None:
    assert otsu_threshold(_photo(bright_sample=True)) == pytest.approx(120.0, abs=1.0)


def test_otsu_threshold_separates_noisy_classes() -> None:
    level = otsu_threshold(_photo(bright_sample=True, noise=15.0))

    assert 90.0 < level < 150.0


@pytest.mark.parametrize("bright", [True, False], ids=["bright-sample", "dark-sample"])
def test_segmentation_picks_the_class_at_the_centre(bright: bool) -> None:
    segmentation = segment_plate_image(_photo(bright_sample=bright, noise=5.0))

    assert segmentation.sample_is_bright is bright
    assert segmentation.bounds == (10, 40, 20, 70)
    assert segmentation.cropped.all()
    assert segmentation.aspect_ratio == pytest.approx(30.0 / 50.0)


def test_invert_selects_the_background() -> None:
    # Without hole filling: the sample is a hole in the background.
    segmentation = segment_plate_image(_photo(bright_sample=True), invert=True, fill_holes=False)

    assert segmentation.sample_is_bright is False
    assert not segmentation.mask[15, 30]
    assert segmentation.mask[0, 0]


def test_holes_are_filled_unless_asked_not_to() -> None:
    image = _photo(bright_sample=True, hole=True)

    assert segment_plate_image(image).mask[22, 42]
    assert not segment_plate_image(image, fill_holes=False).mask[22, 42]


def test_an_explicit_threshold_replaces_otsu() -> None:
    image = _photo(bright_sample=True)
    image[10:40, 20:30] = 120.0

    low = segment_plate_image(image, threshold=100.0)
    high = segment_plate_image(image, threshold=150.0)

    assert low.threshold == 100.0
    assert low.bounds == (10, 40, 20, 70)
    assert high.bounds == (10, 40, 30, 70)


def test_a_boolean_image_is_taken_as_the_outline() -> None:
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:6, 3:9] = True
    segmentation = segment_plate_image(mask)

    assert segmentation.threshold is None
    assert segmentation.bounds == (2, 6, 3, 9)


def test_segmentation_reports_an_outline_that_fills_the_image() -> None:
    with pytest.raises(SampleImageError, match="fills the whole image"):
        segment_plate_image(_photo(bright_sample=True), threshold=250.0)


def test_segmentation_reports_an_empty_outline() -> None:
    with pytest.raises(SampleImageError, match="no sample was found"):
        segment_plate_image(np.zeros((4, 4), dtype=bool))


@pytest.mark.parametrize(
    ("image", "match"),
    [
        (np.full((5, 5), 3.0), "single grey level"),
        (np.ones(5), "2-D"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), "non-finite"),
        ([["a", "b"], ["c", "d"]], "grey levels"),
    ],
    ids=["flat", "one-dimensional", "nan", "text"],
)
def test_segmentation_rejects_unusable_images(image: object, match: str) -> None:
    with pytest.raises(SampleImageError, match=match):
        segment_plate_image(image)  # type: ignore[arg-type]


def test_in_plane_mask_puts_x_right_and_y_up_the_image() -> None:
    image_mask = np.zeros((3, 4), dtype=bool)
    image_mask[0, 3] = True  # top-right pixel
    in_plane = image_to_in_plane(image_mask)

    assert in_plane.shape == (4, 3)
    assert in_plane[3, 2]
    np.testing.assert_array_equal(in_plane_to_image(in_plane), image_mask)


def test_resample_keeps_cells_at_least_half_covered() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, :4] = True
    mask[:, 4] = True

    resampled = resample_mask(mask, (2, 2))

    np.testing.assert_array_equal(resampled, [[True, False], [True, False]])
    assert resample_mask(np.ones((3, 3), dtype=bool), (7, 5)).all()


def test_fit_mask_limits_the_longer_side() -> None:
    assert fit_mask(np.ones((600, 300), dtype=bool), 256).shape == (256, 128)
    assert fit_mask(np.ones((20, 10), dtype=bool), 256).shape == (20, 10)


def test_extrude_repeats_the_outline_through_the_layers() -> None:
    mask = np.array([[True, False], [True, True]])
    solid = extrude_plate(mask, 3)

    assert solid.shape == (2, 2, 3)
    for layer in range(3):
        np.testing.assert_array_equal(solid[:, :, layer], mask)
    with pytest.raises(SampleImageError, match="at least one voxel"):
        extrude_plate(mask, 0)


def test_resample_rejects_unusable_shapes() -> None:
    with pytest.raises(SampleImageError, match="two positive"):
        resample_mask(np.ones((2, 2), dtype=bool), (0, 2))
    with pytest.raises(SampleImageError, match="two-dimensional"):
        resample_mask(np.ones((2, 2, 2), dtype=bool), (2, 2))

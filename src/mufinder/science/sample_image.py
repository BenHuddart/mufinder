"""Segment a photographed plate-like sample into an in-plane outline.

The functions take a 2-D greyscale (or boolean) array, with row 0 at the top
of the image as an image file stores it, and return boolean masks. Loading the
image file is the caller's job, so this module needs no imaging dependency.
An in-plane mask is indexed ``[x, y]``: ``x`` runs to the right of the image
and ``y`` up it, the sample ``a`` and ``b`` axes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_HISTOGRAM_BINS = 256
_SUPERSAMPLE = 4


class SampleImageError(ValueError):
    """Raised when a sample photograph cannot be segmented."""


@dataclass(frozen=True, slots=True)
class PlateSegmentation:
    """The outline of a plate found in a photograph.

    ``mask`` covers the whole image, rows top to bottom. ``threshold`` is the
    grey level that separated sample from background (``None`` for a boolean
    input), ``sample_is_bright`` the polarity used, and ``bounds`` the
    ``(row_start, row_stop, column_start, column_stop)`` of the outline.
    """

    mask: np.ndarray
    threshold: float | None
    sample_is_bright: bool
    bounds: tuple[int, int, int, int]

    @property
    def cropped(self) -> np.ndarray:
        """Return the mask cropped to the outline's bounding box, rows top to bottom."""

        r0, r1, c0, c1 = self.bounds
        return self.mask[r0:r1, c0:c1]

    def in_plane_mask(self) -> np.ndarray:
        """Return the cropped outline indexed ``[x, y]`` with ``y`` pointing up the image."""

        return image_to_in_plane(self.cropped)

    @property
    def aspect_ratio(self) -> float:
        """Return the outline's bounding-box height over width, in pixels."""

        r0, r1, c0, c1 = self.bounds
        return float(r1 - r0) / float(c1 - c0)


def greyscale_levels(image: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a 2-D image as a finite float array, rejecting other shapes."""

    try:
        grey = np.asarray(image, dtype=float)
    except (TypeError, ValueError) as exc:
        raise SampleImageError("the image must be a 2-D array of grey levels") from exc
    if grey.ndim != 2 or min(grey.shape) < 2:
        raise SampleImageError("the image must be a 2-D array at least 2 pixels on each side")
    if not np.all(np.isfinite(grey)):
        raise SampleImageError("the image contains non-finite grey levels")
    return grey


def otsu_threshold(image: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Return the grey level that best separates two classes (N. Otsu, IEEE Trans. Syst. Man Cybern. 9, 62 (1979)).

    The level maximises the between-class variance of a 256-bin histogram.
    When several neighbouring splits tie, as across the empty bins of a
    two-level image, it is the middle of them.
    """

    grey = greyscale_levels(image)
    low, high = float(grey.min()), float(grey.max())
    if high <= low:
        raise SampleImageError("the image has a single grey level, so no outline can be found")
    counts, edges = np.histogram(grey, bins=_HISTOGRAM_BINS, range=(low, high))
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight = np.cumsum(counts).astype(float)
    weight_above = weight[-1] - weight
    moment = np.cumsum(counts * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_below = moment / weight
        mean_above = (moment[-1] - moment) / weight_above
        between = weight * weight_above * (mean_below - mean_above) ** 2
    between = np.where(np.isfinite(between), between, -1.0)[:-1]
    start = int(np.argmax(between))
    # Splits across a run of empty bins tie; the threshold is the middle of the run.
    stop = start
    while stop + 1 < len(between) and between[stop + 1] >= between[start] * (1.0 - 1e-12):
        stop += 1
    return float(0.5 * (edges[start + 1] + edges[stop + 1]))


def segment_plate_image(
    image: Sequence[Sequence[float]] | np.ndarray,
    *,
    threshold: float | None = None,
    invert: bool = False,
    fill_holes: bool = True,
) -> PlateSegmentation:
    """Return the outline of the sample in a photograph.

    A boolean ``image`` is taken as the outline itself. A greyscale image is
    split at ``threshold``, or at the Otsu threshold when it is ``None``. The
    sample is the class that covers most of the central ninth of the image, so
    a dark crystal on a light background and a light one on a dark background
    both segment; ``invert`` swaps that choice. An outline that is empty or
    fills the whole image is an error. ``fill_holes`` fills regions of
    background enclosed by the sample, such as reflections.
    """

    array = np.asarray(image)
    if array.dtype == bool:
        if array.ndim != 2 or min(array.shape) < 2:
            raise SampleImageError("the image must be a 2-D array at least 2 pixels on each side")
        mask = ~array if invert else array.copy()
        level: float | None = None
        bright = not invert
    else:
        grey = greyscale_levels(array)
        level = otsu_threshold(grey) if threshold is None else float(threshold)
        if not np.isfinite(level):
            raise SampleImageError("the threshold must be a finite grey level")
        above = grey > level
        rows, columns = grey.shape
        row_slice = slice(rows // 3, max(rows // 3 + 1, 2 * rows // 3))
        column_slice = slice(columns // 3, max(columns // 3 + 1, 2 * columns // 3))
        centre = above[row_slice, column_slice]
        bright = bool(np.mean(centre) >= 0.5)
        if invert:
            bright = not bright
        mask = above if bright else ~above
    if fill_holes:
        mask = ndimage.binary_fill_holes(mask)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise SampleImageError(
            "no sample was found at this threshold; move the threshold or switch Invert"
        )
    if np.all(mask):
        raise SampleImageError(
            "the outline fills the whole image at this threshold; move the threshold or "
            "switch Invert"
        )
    row_hits = np.flatnonzero(np.any(mask, axis=1))
    column_hits = np.flatnonzero(np.any(mask, axis=0))
    bounds = (
        int(row_hits[0]),
        int(row_hits[-1]) + 1,
        int(column_hits[0]),
        int(column_hits[-1]) + 1,
    )
    return PlateSegmentation(mask=mask, threshold=level, sample_is_bright=bright, bounds=bounds)


def image_to_in_plane(mask: np.ndarray) -> np.ndarray:
    """Return an image-ordered mask (rows top to bottom) indexed ``[x, y]`` with ``y`` up."""

    return np.ascontiguousarray(np.asarray(mask, dtype=bool)[::-1, :].T)


def in_plane_to_image(mask: np.ndarray) -> np.ndarray:
    """Return an ``[x, y]`` mask in image order, rows top to bottom; the inverse of :func:`image_to_in_plane`."""

    return np.ascontiguousarray(np.asarray(mask, dtype=bool).T[::-1, :])


def resample_mask(mask: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Return a boolean mask resampled to ``shape`` covering the same extent.

    A target cell is set when at least half of it is covered, estimated from a
    4 x 4 grid of samples per cell taken from the nearest source pixel.
    """

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise SampleImageError("the mask must be two-dimensional")
    target = tuple(int(value) for value in shape)
    if len(target) != 2 or min(target) < 1:
        raise SampleImageError("the resampled mask needs two positive dimensions")
    indices = []
    for size, count in zip(source.shape, target, strict=True):
        positions = (np.arange(count * _SUPERSAMPLE) + 0.5) / (count * _SUPERSAMPLE) * size
        indices.append(np.clip(positions.astype(int), 0, size - 1))
    samples = source[np.ix_(indices[0], indices[1])].astype(float)
    coverage = samples.reshape(target[0], _SUPERSAMPLE, target[1], _SUPERSAMPLE).mean(axis=(1, 3))
    return coverage >= 0.5


def fit_mask(mask: np.ndarray, longest: int) -> np.ndarray:
    """Return ``mask`` resampled so its longer side has at most ``longest`` cells."""

    source = np.asarray(mask, dtype=bool)
    scale = min(1.0, float(longest) / float(max(source.shape)))
    if scale >= 1.0:
        return source.copy()
    shape = tuple(max(1, int(round(size * scale))) for size in source.shape)
    return resample_mask(source, shape)


def extrude_plate(mask: np.ndarray, layers: int) -> np.ndarray:
    """Return a 3-D ``[x, y, z]`` voxel mask: the in-plane ``mask`` repeated through ``layers``."""

    count = int(layers)
    if count < 1:
        raise SampleImageError("the plate needs at least one voxel through its thickness")
    source = np.asarray(mask, dtype=bool)
    return np.repeat(source[:, :, None], count, axis=2)

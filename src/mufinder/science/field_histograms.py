"""Step curves for plotting and exporting muon field-magnitude distributions.

A distribution series is ``(centres, probabilities, magnitudes, label)``: the
histogram bin centres in tesla, each bin's probability scaled so the tallest
bin is 1, the sampled field magnitudes and the site label. These helpers turn
one or more series into step curves that start and end on a zero baseline
across a shared field window, and sum overlapping series on common intervals.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mufinder.science.magnetic_fields import field_magnitude_histogram

#: One distribution: bin centres (T), scaled probabilities, sampled magnitudes (T), label.
DistributionSeries = tuple[np.ndarray, np.ndarray, np.ndarray, str]


def field_magnitudes(result: object) -> np.ndarray:
    """Return the magnitudes of a distribution result's sampled ``fields_T``, or none when malformed."""

    fields = np.asarray(getattr(result, "fields_T", []), dtype=float)
    if fields.ndim != 2 or fields.shape[1] != 3:
        return np.asarray([], dtype=float)
    return np.linalg.norm(fields, axis=1)


def normalized_probabilities(counts: np.ndarray) -> np.ndarray:
    """Scale ``counts`` so the largest is 1, or return zeros when none is positive."""

    counts = np.asarray(counts, dtype=float)
    if counts.size == 0 or not np.any(counts > 0.0):
        return np.zeros_like(counts, dtype=float)
    return counts / float(np.max(counts))


def distribution_series(results: Sequence[object]) -> list[DistributionSeries]:
    """Return one series per distribution result that has a histogram or samples.

    A result's stored ``histogram_centres_T`` and ``histogram_counts`` are used
    when they match in length; otherwise its sampled fields are binned with
    :func:`~mufinder.science.magnetic_fields.field_magnitude_histogram` into
    10 to 80 bins (twice the square root of the sample count), so a single
    field value gives a narrow spike. A result with neither is left out. Raises
    :class:`~mufinder.science.magnetic_fields.MagneticFieldError` when the
    samples it bins include a non-finite magnitude.
    """

    series: list[DistributionSeries] = []
    for index, result in enumerate(results):
        centres = np.asarray(getattr(result, "histogram_centres_T", []), dtype=float)
        counts = np.asarray(getattr(result, "histogram_counts", []), dtype=float)
        magnitudes = field_magnitudes(result)
        if centres.size == 0 or counts.size != centres.size:
            if magnitudes.size == 0:
                continue
            bins = min(80, max(10, int(np.sqrt(magnitudes.size)) * 2))
            centres, counts = field_magnitude_histogram(magnitudes, bins=bins)
        if not centres.size or not counts.size:
            continue
        probabilities = normalized_probabilities(counts)
        label = str(getattr(result, "site_label", f"site {index + 1}"))
        series.append((centres, probabilities, magnitudes, label))
    return series


def common_binned_series(
    series: Sequence[DistributionSeries],
) -> tuple[list[DistributionSeries], np.ndarray] | None:
    """Re-bin every series on one set of edges spanning all their samples.

    Uses at least 10 bins, or the most bins any series has. Returns the
    re-binned series and the shared edges, or ``None`` when a series has no
    samples to re-bin.
    """

    finite_magnitudes = [
        magnitudes[np.isfinite(magnitudes)]
        for _centres, _probabilities, magnitudes, _label in series
        if magnitudes.size
    ]
    if len(finite_magnitudes) != len(series):
        return None
    all_magnitudes = np.concatenate(finite_magnitudes)
    if all_magnitudes.size == 0:
        return None
    low, high = field_axis_limits(np.asarray([], dtype=float), all_magnitudes)
    bins = max(10, max(len(centres) for centres, _probabilities, _magnitudes, _label in series))
    common_edges = np.linspace(low, high, int(bins) + 1, dtype=float)
    common_centres = common_edges[:-1] + 0.5 * np.diff(common_edges)
    out: list[DistributionSeries] = []
    for _centres, _probabilities, magnitudes, label in series:
        counts, _edges = np.histogram(magnitudes[np.isfinite(magnitudes)], bins=common_edges)
        out.append((common_centres, normalized_probabilities(counts), magnitudes, label))
    return out, common_edges


def histogram_edges_from_centres(centres: np.ndarray, low: float, high: float) -> np.ndarray:
    """Return bin edges halfway between ``centres``, the outer two mirrored about the end centres.

    A single centre gets a narrow bin scaled to the centre and to the
    ``low``-``high`` window.
    """

    centres = np.asarray(centres, dtype=float)
    if centres.size == 1:
        half_width = 0.5 * max(abs(float(centres[0])) * 1e-6, (high - low) * 1e-4, 1e-9)
        return np.asarray([centres[0] - half_width, centres[0] + half_width], dtype=float)
    midpoints = 0.5 * (centres[:-1] + centres[1:])
    first_edge = centres[0] - (midpoints[0] - centres[0])
    last_edge = centres[-1] + (centres[-1] - midpoints[-1])
    return np.concatenate(([first_edge], midpoints, [last_edge])).astype(float)


def interval_steps_with_zero_baseline(
    edges: np.ndarray,
    values: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the step curve of ``values`` on the bins ``edges``, clipped to ``low``-``high``.

    The curve is zero outside the bins, so it rises from and falls back to the
    baseline inside the window.
    """

    x_values: list[float] = []
    y_values: list[float] = []
    if low < edges[0]:
        x_values.extend([float(low), float(min(edges[0], high))])
        y_values.extend([0.0, 0.0])
    for index, probability in enumerate(values):
        left = max(float(edges[index]), float(low))
        right = min(float(edges[index + 1]), float(high))
        if right <= left:
            continue
        if x_values and not np.isclose(x_values[-1], left):
            x_values.append(left)
            y_values.append(y_values[-1])
        x_values.extend([left, right])
        y_values.extend([float(probability), float(probability)])
    if not x_values:
        return np.asarray([low, high], dtype=float), np.asarray([0.0, 0.0], dtype=float)
    if high > edges[-1]:
        edge = max(float(edges[-1]), float(low))
        if not np.isclose(x_values[-1], edge):
            x_values.append(edge)
            y_values.append(y_values[-1])
        x_values.extend([edge, float(high)])
        y_values.extend([0.0, 0.0])
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def probability_steps_with_zero_baseline(
    centres: np.ndarray,
    probabilities: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the step curve of one series on the bins around its ``centres``.

    See :func:`interval_steps_with_zero_baseline`. Mismatched or empty input,
    or an empty window, is returned unchanged.
    """

    centres = np.asarray(centres, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if centres.size == 0 or probabilities.size != centres.size or high <= low:
        return centres, probabilities
    edges = histogram_edges_from_centres(centres, low, high)
    return interval_steps_with_zero_baseline(edges, probabilities, low, high)


def summed_probability_steps(
    series: Sequence[DistributionSeries],
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the sum of every series' probability at each bin centre and the window ends.

    Returns the field values, the summed curve and the summed values (the same
    array twice); the sum is zero at ``low`` and ``high``.
    """

    x_values = np.unique(
        np.concatenate(
            (
                [low, high],
                *(
                    np.asarray(centres, dtype=float)
                    for centres, _probabilities, _magnitudes, _label in series
                ),
            )
        )
    )
    x_values = x_values[np.isfinite(x_values)]
    x_values.sort()
    if x_values.size < 2:
        x_values = np.asarray([low, high], dtype=float)
    sum_values = np.zeros_like(x_values, dtype=float)
    for centres, probabilities, _magnitudes, _label in series:
        edges = histogram_edges_from_centres(np.asarray(centres, dtype=float), low, high)
        for index, probability in enumerate(probabilities):
            left = edges[index]
            right = edges[index + 1]
            mask = (x_values >= left) & (x_values < right)
            if index == len(probabilities) - 1:
                mask = mask | np.isclose(x_values, right)
            sum_values[mask] += float(probability)
    sum_values[np.isclose(x_values, low) | np.isclose(x_values, high)] = 0.0
    return x_values, sum_values, sum_values


def summed_probability_intervals(
    series: Sequence[DistributionSeries],
    low: float,
    high: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Split ``low``-``high`` at every series' bin edges and read each series on the pieces.

    Returns the interval edges, each label's probability on every interval and
    their sum.
    """

    source_edges = [
        histogram_edges_from_centres(centres, low, high)
        for centres, _probabilities, _magnitudes, _label in series
    ]
    raw_edges = np.concatenate(
        ([low, high], *(np.clip(edges, low, high) for edges in source_edges))
    )
    edges = np.unique(raw_edges[np.isfinite(raw_edges)])
    edges.sort()
    if edges.size < 2:
        edges = np.asarray([low, high], dtype=float)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    values_by_label: dict[str, np.ndarray] = {}
    for (centres, probabilities, _magnitudes, label), edges_for_series in zip(
        series, source_edges, strict=True
    ):
        values = np.zeros_like(midpoints, dtype=float)
        for index, probability in enumerate(probabilities):
            left = edges_for_series[index]
            right = edges_for_series[index + 1]
            mask = (midpoints >= left) & (midpoints < right)
            if index == len(probabilities) - 1:
                mask = mask | np.isclose(midpoints, right)
            values[mask] = float(probability)
        values_by_label[label] = values
    sum_values = (
        np.sum(tuple(values_by_label.values()), axis=0)
        if values_by_label
        else np.zeros_like(midpoints)
    )
    return edges, values_by_label, sum_values


def field_axis_limits(centres: np.ndarray, magnitudes: np.ndarray) -> tuple[float, float]:
    """Return the field window to plot: the samples (else the centres) with 8 % padding.

    A narrow distribution gets a window at least 4 % of its centre wide, and a
    window never starts below zero when every field is non-negative. With no
    finite value the window is 0 to 1 T.
    """

    source = magnitudes if magnitudes.size else centres
    finite = np.asarray(source[np.isfinite(source)], dtype=float)
    if finite.size == 0:
        return (0.0, 1.0)
    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    centre = 0.5 * (data_min + data_max)
    span = data_max - data_min
    scale = max(abs(centre), 1.0)
    min_window = max(abs(centre) * 0.04, scale * 1e-4)
    if span < min_window:
        half_window = 0.5 * min_window
        low = centre - half_window
        high = centre + half_window
    else:
        padding = max(0.08 * span, scale * 1e-5)
        low = data_min - padding
        high = data_max + padding
    if data_min >= 0.0:
        low = max(0.0, low)
    if high <= low:
        high = low + max(scale * 1e-4, 1e-6)
    return (low, high)

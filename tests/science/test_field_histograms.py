"""Tests for the field-distribution step curves."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mufinder.science.field_histograms import (
    common_binned_series,
    distribution_series,
    field_axis_limits,
    field_magnitudes,
    histogram_edges_from_centres,
    interval_steps_with_zero_baseline,
    magnitude_histogram,
    normalized_probabilities,
    probability_steps_with_zero_baseline,
    summed_probability_intervals,
    summed_probability_steps,
)


def _distribution(label, fields, centres=(), counts=()):
    return SimpleNamespace(
        site_label=label,
        fields_T=np.asarray(fields, dtype=float),
        histogram_centres_T=np.asarray(centres, dtype=float),
        histogram_counts=np.asarray(counts, dtype=float),
    )


def test_zero_padding_uses_bin_edges() -> None:
    x_values, y_values = probability_steps_with_zero_baseline(
        np.asarray([1.0, 2.0]), np.asarray([1.0, 0.5]), 0.0, 3.0
    )

    assert x_values == pytest.approx([0.0, 0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.0])
    assert y_values == pytest.approx([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0])


def test_steps_of_mismatched_input_are_returned_unchanged() -> None:
    centres = np.asarray([1.0, 2.0])
    probabilities = np.asarray([1.0])

    x_values, y_values = probability_steps_with_zero_baseline(centres, probabilities, 0.0, 3.0)

    assert x_values is not None and list(x_values) == [1.0, 2.0]
    assert list(y_values) == [1.0]


def test_interval_steps_clip_to_the_window() -> None:
    x_values, y_values = interval_steps_with_zero_baseline(
        np.asarray([0.0, 1.0, 2.0]), np.asarray([0.5, 1.0]), 0.5, 1.5
    )

    assert x_values == pytest.approx([0.5, 1.0, 1.0, 1.5])
    assert y_values == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_interval_steps_outside_the_window_are_a_flat_baseline() -> None:
    x_values, y_values = interval_steps_with_zero_baseline(
        np.asarray([5.0, 6.0]), np.asarray([1.0]), 0.0, 1.0
    )

    assert x_values == pytest.approx([0.0, 1.0])
    assert y_values == pytest.approx([0.0, 0.0])


def test_edges_from_one_centre_make_a_narrow_bin() -> None:
    edges = histogram_edges_from_centres(np.asarray([2.0]), 0.0, 4.0)

    assert edges[0] < 2.0 < edges[1]
    assert edges[1] - edges[0] == pytest.approx(4e-4)


def test_edges_from_centres_sit_halfway_and_mirror_at_the_ends() -> None:
    edges = histogram_edges_from_centres(np.asarray([1.0, 2.0, 4.0]), 0.0, 5.0)

    assert edges == pytest.approx([0.5, 1.5, 3.0, 5.0])


def test_series_uses_the_stored_histogram() -> None:
    result = _distribution("site 1", [[0, 0, 1], [0, 0, 3]], centres=[1.0, 3.0], counts=[4, 2])

    ((centres, probabilities, magnitudes, label),) = distribution_series([result])

    assert label == "site 1"
    assert centres == pytest.approx([1.0, 3.0])
    assert probabilities == pytest.approx([1.0, 0.5])
    assert magnitudes == pytest.approx([1.0, 3.0])


def test_series_bins_the_samples_without_a_stored_histogram() -> None:
    fields = [[0.0, 0.0, value] for value in np.linspace(1.0, 2.0, 100)]
    result = _distribution("site 1", fields, centres=[1.0], counts=[])

    ((centres, probabilities, _magnitudes, _label),) = distribution_series([result])

    # Twice the square root of 100 samples.
    assert centres.size == 20
    assert probabilities.max() == pytest.approx(1.0)


def test_series_leaves_out_a_result_with_neither_histogram_nor_samples() -> None:
    malformed = _distribution("malformed", [[1.0, 2.0]])
    unlabelled = SimpleNamespace(
        fields_T=[[0.0, 0.0, 1.0]], histogram_centres_T=[1.0], histogram_counts=[1.0]
    )

    series = distribution_series([malformed, unlabelled])

    assert [label for *_values, label in series] == ["site 2"]


def test_magnitudes_and_histogram_of_no_samples_are_empty() -> None:
    assert field_magnitudes(SimpleNamespace()).size == 0
    centres, counts = magnitude_histogram(np.asarray([], dtype=float))
    assert centres.size == 0 and counts.size == 0


def test_probabilities_scale_to_the_tallest_bin() -> None:
    assert normalized_probabilities(np.asarray([2.0, 4.0])) == pytest.approx([0.5, 1.0])
    assert normalized_probabilities(np.asarray([0.0, 0.0])) == pytest.approx([0.0, 0.0])


def test_common_bins_span_every_series() -> None:
    first = _distribution("a", [[0, 0, 1.0], [0, 0, 1.2]], centres=[1.0, 1.2], counts=[1, 1])
    second = _distribution("b", [[0, 0, 2.0], [0, 0, 2.2]], centres=[2.0, 2.2], counts=[1, 1])
    series = distribution_series([first, second])

    common = common_binned_series(series)

    assert common is not None
    rebinned, edges = common
    assert len(edges) == 11
    low, high = field_axis_limits(np.asarray([]), np.asarray([1.0, 1.2, 2.0, 2.2]))
    assert edges[0] == pytest.approx(low) and edges[-1] == pytest.approx(high)
    assert [label for *_values, label in rebinned] == ["a", "b"]
    assert all(np.array_equal(centres, rebinned[0][0]) for centres, *_rest in rebinned)


def test_common_bins_need_samples_in_every_series() -> None:
    sampled = _distribution("a", [[0, 0, 1.0]], centres=[1.0], counts=[1])
    stored_only = (np.asarray([1.0]), np.asarray([1.0]), np.asarray([], dtype=float), "b")

    assert common_binned_series([*distribution_series([sampled]), stored_only]) is None


def test_summed_intervals_add_overlapping_series() -> None:
    series = [
        (np.asarray([1.0, 2.0]), np.asarray([1.0, 0.5]), np.asarray([]), "a"),
        (np.asarray([2.0, 3.0]), np.asarray([0.5, 1.0]), np.asarray([]), "b"),
    ]

    edges, values, total = summed_probability_intervals(series, 0.0, 4.0)

    assert edges == pytest.approx([0.0, 0.5, 1.5, 2.5, 3.5, 4.0])
    assert values["a"] == pytest.approx([0.0, 1.0, 0.5, 0.0, 0.0])
    assert values["b"] == pytest.approx([0.0, 0.0, 0.5, 1.0, 0.0])
    assert total == pytest.approx([0.0, 1.0, 1.0, 1.0, 0.0])


def test_summed_steps_are_zero_at_the_window_ends() -> None:
    series = [
        (np.asarray([1.0, 2.0]), np.asarray([1.0, 0.5]), np.asarray([]), "a"),
        (np.asarray([2.0, 3.0]), np.asarray([0.5, 1.0]), np.asarray([]), "b"),
    ]

    x_values, curve, total = summed_probability_steps(series, 0.0, 4.0)

    assert x_values == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert curve == pytest.approx([0.0, 1.0, 1.0, 1.0, 0.0])
    assert total is curve


@pytest.mark.parametrize(
    ("centres", "magnitudes", "expected"),
    [
        ([], [], (0.0, 1.0)),
        ([], [1.0, 2.0], (0.92, 2.08)),
        ([], [2.0, 2.0], (1.96, 2.04)),
        ([1.0, 3.0], [], (0.84, 3.16)),
        ([], [-1.0, 1.0], (-1.16, 1.16)),
        ([], [0.0, 0.5], (0.0, 0.54)),
    ],
)
def test_field_axis_limits(centres, magnitudes, expected) -> None:
    limits = field_axis_limits(
        np.asarray(centres, dtype=float), np.asarray(magnitudes, dtype=float)
    )

    assert limits == pytest.approx(expected)

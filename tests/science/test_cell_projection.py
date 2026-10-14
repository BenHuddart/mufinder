"""Tests for the flat cell projections in :mod:`mufinder.science.cell_projection`."""

from __future__ import annotations

import math

import pytest

from mufinder.science.cell_projection import cell_projection, periodic_boundary_images
from mufinder.science.lattice import LatticeError

CUBIC = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
HEXAGONAL = [[3.0, 0.0, 0.0], [-1.5, 1.5 * math.sqrt(3.0), 0.0], [0.0, 0.0, 5.0]]


def test_cubic_cell_viewed_down_c_is_a_square_with_a_right_and_b_up() -> None:
    projection = cell_projection(CUBIC, "c")

    assert projection.bounds() == pytest.approx((0.0, 0.0, 4.0, 4.0))
    assert projection.screen_axis_labels == ("a", "b")
    point = projection.project([0.25, 0.5, 0.75])
    assert (point.x, point.y, point.depth) == pytest.approx((1.0, 2.0, 0.75))


@pytest.mark.parametrize(("axis", "labels"), [("a", ("b", "c")), ("b", ("c", "a"))])
def test_other_axes_put_the_next_two_vectors_on_screen(axis: str, labels: tuple[str, str]) -> None:
    projection = cell_projection(CUBIC, axis)

    assert projection.screen_axis_labels == labels
    point = projection.project([0.1, 0.2, 0.3])
    expected = {"a": (0.8, 1.2, 0.1), "b": (1.2, 0.4, 0.2)}[axis]
    assert (point.x, point.y, point.depth) == pytest.approx(expected)


def test_projection_has_twelve_edges_and_keeps_oblique_angles() -> None:
    projection = cell_projection(HEXAGONAL, "c")

    assert len(projection.edges) == 12
    tip_of_b = projection.project([0.0, 1.0, 0.0])
    angle = math.degrees(math.atan2(tip_of_b.y, tip_of_b.x))
    assert angle == pytest.approx(120.0)


def test_depth_is_clamped_to_the_cell() -> None:
    projection = cell_projection(CUBIC, "c")

    assert projection.project([0.0, 0.0, 1.4]).depth == 1.0
    assert projection.project([0.0, 0.0, -0.2]).depth == 0.0


def test_invalid_axis_and_degenerate_lattice_raise() -> None:
    with pytest.raises(LatticeError):
        cell_projection(CUBIC, "d")
    with pytest.raises(LatticeError):
        cell_projection([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], "a")


def test_boundary_images_add_far_faces_for_corner_and_face_atoms() -> None:
    assert len(periodic_boundary_images([0.0, 0.0, 0.0])) == 8
    assert sorted(periodic_boundary_images([0.5, 0.5, 0.0])) == [[0.5, 0.5, 0.0], [0.5, 0.5, 1.0]]
    assert periodic_boundary_images([0.25, 0.5, 0.75]) == [[0.25, 0.5, 0.75]]


def test_boundary_images_wrap_positions_into_the_cell_first() -> None:
    assert sorted(periodic_boundary_images([1.25, -0.5, 0.9999999])) == [[0.25, 0.5, 0.0], [0.25, 0.5, 1.0]]

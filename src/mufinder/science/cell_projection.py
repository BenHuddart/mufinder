"""Flat projections of a unit cell for 2D drawings.

A :class:`CellProjection` looks down one lattice vector and maps fractional
coordinates onto a screen plane in Angstrom, so a widget can draw the cell, its
atoms and muon positions without a 3D renderer. Looking down ``c``, screen ``x``
follows ``a`` and screen ``y`` follows the part of ``b`` perpendicular to it;
looking down ``a`` the pair is ``b`` and ``c``, and looking down ``b`` it is
``c`` and ``a``. Screen ``y`` points up, as in a plot, so a widget flips it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import math

from mufinder.science.lattice import LatticeError, lattice_frac_to_cart, lattice_rows, wrap_fractional

#: The lattice vectors a projection can look down, in order.
PROJECTION_AXES = ("a", "b", "c")

_Vector = tuple[float, float, float]
_Point2D = tuple[float, float]


@dataclass(frozen=True, slots=True)
class ProjectedPoint:
    """A fractional position mapped onto the screen plane.

    ``depth`` is the position's fractional coordinate along the viewing axis,
    clamped to ``[0, 1]``; ``1`` is the face of the cell nearest the viewer.
    """

    x: float
    y: float
    depth: float


@dataclass(frozen=True, slots=True)
class CellProjection:
    """Map fractional positions in one cell onto a screen plane."""

    axis: str
    lattice: tuple[_Vector, _Vector, _Vector]
    screen_x: _Vector
    screen_y: _Vector
    edges: tuple[tuple[_Point2D, _Point2D], ...]

    @property
    def axis_index(self) -> int:
        return PROJECTION_AXES.index(self.axis)

    @property
    def screen_axis_labels(self) -> tuple[str, str]:
        """The lattice vectors that run along screen ``x`` and ``y``."""

        index = self.axis_index
        return PROJECTION_AXES[(index + 1) % 3], PROJECTION_AXES[(index + 2) % 3]

    def project(self, fractional: Sequence[float]) -> ProjectedPoint:
        cartesian = lattice_frac_to_cart(fractional, self.lattice)
        return ProjectedPoint(
            x=_dot(cartesian, self.screen_x),
            y=_dot(cartesian, self.screen_y),
            depth=min(1.0, max(0.0, float(fractional[self.axis_index]))),
        )

    def bounds(self) -> tuple[float, float, float, float]:
        """Return ``(min_x, min_y, max_x, max_y)`` of the projected cell."""

        xs = [x for edge in self.edges for x, _ in edge]
        ys = [y for edge in self.edges for _, y in edge]
        return min(xs), min(ys), max(xs), max(ys)


def cell_projection(lattice_vectors: Sequence[Sequence[float]], axis: str = "c") -> CellProjection:
    """Return the projection of a cell viewed down lattice vector ``axis``.

    Raises :class:`~mufinder.science.lattice.LatticeError` for an unknown axis
    or a lattice that is not three independent vectors.
    """

    if axis not in PROJECTION_AXES:
        raise LatticeError(f"Projection axis must be one of {', '.join(PROJECTION_AXES)}.")
    rows = lattice_rows(lattice_vectors)
    lattice = tuple(tuple(row) for row in rows)
    index = PROJECTION_AXES.index(axis)
    view = _unit(lattice[index])
    along = lattice[(index + 1) % 3]
    screen_x = _unit(_subtract(along, _scale(view, _dot(along, view))))
    screen_y = _cross(view, screen_x)

    def corner(fractional: Sequence[float]) -> _Point2D:
        cartesian = lattice_frac_to_cart(fractional, lattice)
        return _dot(cartesian, screen_x), _dot(cartesian, screen_y)

    edges: list[tuple[_Point2D, _Point2D]] = []
    for start in itertools.product((0.0, 1.0), repeat=3):
        for component in range(3):
            if start[component] == 0.0:
                end = list(start)
                end[component] = 1.0
                edges.append((corner(start), corner(end)))
    return CellProjection(
        axis=axis,
        lattice=lattice,  # type: ignore[arg-type]
        screen_x=screen_x,
        screen_y=screen_y,
        edges=tuple(edges),
    )


def periodic_boundary_images(fractional: Sequence[float], *, tolerance: float = 1e-3) -> list[list[float]]:
    """Return a position wrapped into the cell plus its images on the far faces.

    An atom at ``x = 0`` also sits at ``x = 1``; a drawing of one cell shows
    both, so a corner atom yields eight positions and a face atom two.
    """

    wrapped = [0.0 if value > 1.0 - tolerance else value for value in wrap_fractional(fractional)]
    choices = [(value, 1.0) if value < tolerance else (value,) for value in wrapped]
    return [list(image) for image in itertools.product(*choices)]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _scale(vector: Sequence[float], factor: float) -> _Vector:
    return vector[0] * factor, vector[1] * factor, vector[2] * factor


def _subtract(a: Sequence[float], b: Sequence[float]) -> _Vector:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> _Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(vector: Sequence[float]) -> _Vector:
    length = math.sqrt(_dot(vector, vector))
    if length < 1e-12:
        raise LatticeError("Lattice vectors must be three independent, non-zero vectors.")
    return vector[0] / length, vector[1] / length, vector[2] / length

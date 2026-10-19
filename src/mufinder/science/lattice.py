"""Lattice algebra shared by the science modules.

This is the one home for fractional/Cartesian conversion, lattice inversion,
minimum-image displacements and distances, and greedy deduplication of
periodic positions. A lattice is three row vectors in Angstrom, so a
fractional row vector ``f`` maps to ``f @ lattice``.

Two families of helpers agree with each other to the last bit:

* the scalar helpers (:func:`lattice_frac_to_cart`, :func:`minimum_image_delta`,
  :func:`minimum_image_distance`, ...) take one point, use plain float
  arithmetic and are cheap enough for tight per-pair loops;
* the broadcast helpers (:func:`fractional_to_cartesian`,
  :func:`minimum_image_distance_matrix`, ...) take arrays of points.

Each Cartesian component is formed in the same term order by both, so only a
broadcast norm can differ from the scalar one, by an ulp or so. Decisions that
must match the scalar helper exactly re-measure near ties with it; see
:func:`near_minimum_indexes` and :func:`deduplicate_periodic_positions`.

Every validating helper takes an ``error`` exception class so a caller keeps
its own exception type.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

#: Relative slack within which a broadcast distance is re-measured with the scalar helper.
NEAR_TIE_RTOL = 1e-9

_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class LatticeError(ValueError):
    """Raised when a lattice or point has the wrong shape, or a lattice is singular."""


def lattice_rows(
    lattice_vectors: Sequence[Sequence[float]],
    *,
    error: type[Exception] = LatticeError,
) -> list[list[float]]:
    """Return the lattice as three rows of three floats, raising ``error`` for any other shape."""

    if len(lattice_vectors) != 3 or any(len(vector) != 3 for vector in lattice_vectors):
        raise error("Lattice vectors must be a 3x3 matrix.")
    return [[float(component) for component in vector] for vector in lattice_vectors]


def lattice_matrix(
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    error: type[Exception] = LatticeError,
) -> np.ndarray:
    """Return the lattice as a ``(3, 3)`` float array, raising ``error`` for any other shape."""

    try:
        matrix = np.asarray(lattice_vectors, dtype=float)
    except (TypeError, ValueError) as exc:
        raise error("Lattice vectors must be a 3x3 matrix.") from exc
    if matrix.shape != (3, 3):
        raise error("Lattice vectors must be a 3x3 matrix.")
    return matrix


def determinant3(matrix: Sequence[Sequence[float]]) -> float:
    """Return the determinant of a 3x3 matrix by cofactor expansion along the first row."""

    m = matrix
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def cell_volume(lattice_vectors: Sequence[Sequence[float]]) -> float:
    """Return the cell volume (cubic Angstrom) spanned by the lattice rows."""

    return abs(determinant3(lattice_vectors))


def inverse_lattice(
    lattice_vectors: Sequence[Sequence[float]],
    *,
    error: type[Exception] = LatticeError,
) -> list[list[float]]:
    """Return the inverse of a 3x3 lattice matrix (adjugate over determinant).

    Raises ``error`` when the determinant is within ``1e-12`` of zero.
    """

    m = lattice_rows(lattice_vectors, error=error)
    det = determinant3(m)
    if math.isclose(det, 0.0, abs_tol=1e-12):
        raise error("Lattice matrix is singular and cannot be inverted.")

    inv_det = 1.0 / det
    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]


# Scalar helpers: one point, plain float arithmetic.


def lattice_frac_to_cart(
    fractional: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
    *,
    error: type[Exception] = LatticeError,
) -> list[float]:
    """Convert one fractional coordinate to Cartesian (Angstrom)."""

    lattice = lattice_rows(lattice_vectors, error=error)
    return _frac_to_cart(fractional, lattice)


def lattice_cart_to_frac(
    cartesian: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
    *,
    error: type[Exception] = LatticeError,
) -> list[float]:
    """Convert one Cartesian coordinate (Angstrom) to fractional.

    The Cartesian row vector is multiplied by the inverse lattice, so this undoes
    :func:`lattice_frac_to_cart` for any lattice, not only a symmetric one.
    """

    return _frac_to_cart(cartesian, inverse_lattice(lattice_vectors, error=error))


def reciprocal_to_cartesian(
    reciprocal: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
    *,
    error: type[Exception] = LatticeError,
) -> list[float]:
    """Convert a wavevector in reciprocal lattice units to Cartesian (inverse Angstrom, without ``2π``).

    The reciprocal basis vectors are the columns of the inverse lattice, so
    ``(h, k, l)`` maps to ``h a* + k b* + l c*`` with ``a* · a = 1``.
    """

    inverse = inverse_lattice(lattice_vectors, error=error)
    q0, q1, q2 = reciprocal[0], reciprocal[1], reciprocal[2]
    return [inverse[row][0] * q0 + inverse[row][1] * q1 + inverse[row][2] * q2 for row in range(3)]


def _frac_to_cart(fractional: Sequence[float], lattice: Sequence[Sequence[float]]) -> list[float]:
    f0, f1, f2 = fractional[0], fractional[1], fractional[2]
    return [
        f0 * lattice[0][0] + f1 * lattice[1][0] + f2 * lattice[2][0],
        f0 * lattice[0][1] + f1 * lattice[1][1] + f2 * lattice[2][1],
        f0 * lattice[0][2] + f1 * lattice[1][2] + f2 * lattice[2][2],
    ]


def wrap_fractional(position: Sequence[float]) -> list[float]:
    """Wrap each fractional component into ``[0, 1)`` with Python's ``%``."""

    return [float(value) % 1.0 for value in position]


def minimum_image_delta(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Return the fractional displacement ``a - b`` folded into ``[-0.5, 0.5]`` per component.

    Each component has its nearest integer removed (round half to even), which
    is the minimum image for an orthogonal cell.
    """

    delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    return [d - round(d) for d in delta]


def minimum_image_cartesian_delta(
    a: Sequence[float],
    b: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]] | None = None,
) -> list[float]:
    """Return :func:`minimum_image_delta` of ``a - b`` in Cartesian coordinates.

    With ``lattice_vectors=None`` the displacement stays fractional.
    """

    delta = minimum_image_delta(a, b)
    if lattice_vectors is None:
        return [float(value) for value in delta]
    return _frac_to_cart(delta, lattice_vectors)


def minimum_image_distance(
    a: Sequence[float],
    b: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]] | None = None,
) -> float:
    """Return the length of :func:`minimum_image_cartesian_delta` of ``a - b``.

    The result is in Angstrom, or in fractional units when ``lattice_vectors`` is
    ``None``. The fold is per component (:func:`minimum_image_delta`), so for a
    strongly skewed cell it can exceed the true nearest-image distance; use
    :func:`nearest_image_distance` when that matters.
    """

    cart = minimum_image_cartesian_delta(a, b, lattice_vectors)
    return math.sqrt(cart[0] ** 2 + cart[1] ** 2 + cart[2] ** 2)


def nearest_image_distance(
    a: Sequence[float],
    b: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
) -> float:
    """Return the shortest Cartesian distance between ``a`` and any periodic image of ``b``.

    The per-component minimum-image displacement is searched over its 27
    neighbouring cells, which finds the nearest image in skewed cells too.
    """

    lattice = lattice_matrix(lattice_vectors)
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta -= np.round(delta)
    offsets = np.array([(x, y, z) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)])
    cartesian = (delta[None, :] + offsets) @ lattice
    return float(np.min(np.linalg.norm(cartesian, axis=1)))


def vector_length(vector: Sequence[float]) -> float:
    """Return the Euclidean length of ``vector``."""

    return math.sqrt(sum(float(value) * float(value) for value in vector))


# Broadcast helpers: arrays of points.


def fractional_to_cartesian(
    fractional: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    error: type[Exception] = LatticeError,
) -> np.ndarray:
    """Convert fractional points of shape ``(..., 3)`` to Cartesian, in the scalar term order."""

    lattice = lattice_matrix(lattice_vectors, error=error)
    points = np.asarray(fractional, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise error("Fractional positions must be 3-vectors.")
    return points[..., 0, None] * lattice[0] + points[..., 1, None] * lattice[1] + points[..., 2, None] * lattice[2]


def cartesian_to_fractional(
    cartesian: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    error: type[Exception] = LatticeError,
) -> np.ndarray:
    """Convert Cartesian points of shape ``(..., 3)`` to fractional, in the scalar term order."""

    inverse = np.asarray(inverse_lattice(lattice_matrix(lattice_vectors, error=error).tolist(), error=error))
    points = np.asarray(cartesian, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise error("Cartesian positions must be 3-vectors.")
    return points[..., 0, None] * inverse[0] + points[..., 1, None] * inverse[1] + points[..., 2, None] * inverse[2]


def minimum_image_distance_matrix(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> np.ndarray:
    """Return ``minimum_image_distance(right[j], left[i], lattice)`` for every pair, by broadcasting.

    The Cartesian components are bit-for-bit those of the scalar helper; only the
    final norm may differ from it in the last bit or so (see :func:`near_minimum_indexes`).
    """

    left_array = np.asarray(left, dtype=float).reshape(-1, 3)
    right_array = np.asarray(right, dtype=float).reshape(-1, 3)
    vectors = np.asarray(_IDENTITY if lattice_vectors is None else lattice_vectors, dtype=float)
    delta = right_array[None, :, :] - left_array[:, None, :]
    delta = delta - np.round(delta)
    if lattice_vectors is None:
        return np.linalg.norm(delta, axis=-1)
    cartesian = (
        delta[..., 0, None] * vectors[0]
        + delta[..., 1, None] * vectors[1]
        + delta[..., 2, None] * vectors[2]
    )
    return np.linalg.norm(cartesian, axis=-1)


def near_minimum_indexes(distances: np.ndarray) -> np.ndarray:
    """Indexes of broadcast ``distances`` that could hold the scalar helper's minimum."""

    minimum = float(distances.min())
    return np.flatnonzero(distances <= minimum + minimum * NEAR_TIE_RTOL)


def greedy_unique_indexes(within_tolerance: np.ndarray) -> list[int]:
    """Indexes kept by a first-wins scan that drops anything within tolerance of a kept entry.

    ``within_tolerance[i, j]`` says whether entry ``j`` lies within tolerance of entry ``i``.
    """

    covered = np.zeros(within_tolerance.shape[0], dtype=bool)
    kept: list[int] = []
    for index in range(within_tolerance.shape[0]):
        if covered[index]:
            continue
        kept.append(index)
        covered |= within_tolerance[index]
    return kept


def deduplicate_periodic_positions(
    positions: Sequence[Sequence[float]] | np.ndarray,
    lattice_vectors: Sequence[Sequence[float]] | None,
    *,
    tolerance: float,
) -> list[list[float]]:
    """Wrap ``positions`` and keep each one not within ``tolerance`` of an earlier kept one.

    Distances are :func:`minimum_image_distance` values: Angstrom for a lattice,
    fractional units for ``lattice_vectors=None``. The comparison is inclusive and
    decided exactly as the scalar helper would decide it, and the first occurrence
    of each group is the one kept.
    """

    tolerance = float(tolerance)
    wrapped = [wrap_fractional(position) for position in positions]
    if tolerance >= 0.0:
        # Dropping exact repeats first cannot change the result: a repeat is always within
        # tolerance of whatever kept entry covered (or was) its first occurrence.
        wrapped = [list(position) for position in dict.fromkeys(tuple(position) for position in wrapped)]
    if not wrapped:
        return []
    distances = minimum_image_distance_matrix(wrapped, wrapped, lattice_vectors)
    within = distances <= tolerance
    for left, right in np.argwhere(np.abs(distances - tolerance) <= abs(tolerance) * NEAR_TIE_RTOL):
        # The scan compares each new position against the kept ones: ``distance(new, kept)``.
        within[left, right] = minimum_image_distance(wrapped[right], wrapped[left], lattice_vectors) <= tolerance
    return [wrapped[index] for index in greedy_unique_indexes(within)]


__all__ = [
    "LatticeError",
    "NEAR_TIE_RTOL",
    "cartesian_to_fractional",
    "cell_volume",
    "deduplicate_periodic_positions",
    "determinant3",
    "fractional_to_cartesian",
    "greedy_unique_indexes",
    "inverse_lattice",
    "lattice_cart_to_frac",
    "lattice_frac_to_cart",
    "lattice_matrix",
    "lattice_rows",
    "minimum_image_cartesian_delta",
    "minimum_image_delta",
    "minimum_image_distance",
    "minimum_image_distance_matrix",
    "near_minimum_indexes",
    "nearest_image_distance",
    "vector_length",
    "wrap_fractional",
]

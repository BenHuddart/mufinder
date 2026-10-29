"""Lattice algebra shared by the science modules.

This is the one home for fractional/Cartesian conversion, lattice inversion,
periodic displacements and distances, and greedy deduplication of periodic
positions. A lattice is three row vectors in Angstrom, so a fractional row
vector ``f`` maps to ``f @ lattice``.

Two periodic distances live here. The *nearest-image* helpers
(:func:`nearest_image_distance`, :func:`nearest_image_distance_matrix`, ...)
fold in a reduced basis of the lattice (:func:`reduced_lattice`) and search
the 27 cells around the fold with the lattice metric, so they find the
shortest distance in hexagonal, primitive-centred, monoclinic and sheared
cells; every distance that decides something (a minimum spacing, a grouping or
matching tolerance) uses them. The *minimum-image* helpers
(:func:`minimum_image_distance`, ...) only fold each fractional component into
``[-0.5, 0.5]``. That is exact in an orthogonal cell and for any distance below
about 0.4 of the shortest lattice vector, so they serve the deduplication of
symmetry images, whose tolerances are far smaller than that.

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
from functools import lru_cache
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


def cartesian_to_reciprocal(
    wavevector: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
    *,
    error: type[Exception] = LatticeError,
) -> list[float]:
    """Convert a Cartesian wavevector (inverse Angstrom, without ``2π``) to reciprocal lattice units.

    The inverse of :func:`reciprocal_to_cartesian`: component ``i`` is ``k · a_i``.
    """

    rows = lattice_rows(lattice_vectors, error=error)
    if len(wavevector) != 3:
        raise error("Wavevector must be a 3-vector.")
    k0, k1, k2 = (float(value) for value in wavevector)
    return [row[0] * k0 + row[1] * k1 + row[2] * k2 for row in rows]


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


_NEIGHBOUR_CELL_OFFSETS = np.array(
    [(x, y, z) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0) for z in (-1.0, 0.0, 1.0)]
)
#: Pairs measured at once by :func:`nearest_image_distance_matrix` (27 candidates each).
_NEAREST_IMAGE_BLOCK = 8192
#: Relative shortening a basis vector must gain for :func:`reduced_lattice` to take it.
_REDUCTION_RTOL = 1e-12
_REDUCTION_MAX_PASSES = 1000
_PAIR_COMBINATIONS = tuple((s, t) for s in (-1, 0, 1) for t in (-1, 0, 1) if (s, t) != (0, 0))


@lru_cache(maxsize=64)
def _reduced_basis(
    lattice: tuple[tuple[float, ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.asarray(lattice, dtype=float)
    transform = np.eye(3, dtype=int)
    for _ in range(_REDUCTION_MAX_PASSES):
        changed = False
        for i in range(3):
            j, k = (index for index in range(3) if index != i)
            # Lagrange steps against each other vector, then the +-1 combinations of both.
            trials: list[np.ndarray] = []
            for other in (j, k):
                norm = float(basis[other] @ basis[other])
                coefficient = round(float(basis[i] @ basis[other]) / norm) if norm else 0
                if coefficient:
                    step = np.zeros(3, dtype=int)
                    step[other] = -coefficient
                    trials.append(step)
            for s, t in _PAIR_COMBINATIONS:
                step = np.zeros(3, dtype=int)
                step[j], step[k] = s, t
                trials.append(step)
            length = float(basis[i] @ basis[i])
            for step in trials:
                candidate = basis[i] + step @ basis
                candidate_length = float(candidate @ candidate)
                if candidate_length < length * (1.0 - _REDUCTION_RTOL):
                    basis[i] = candidate
                    transform[i] = transform[i] + step @ transform
                    length = candidate_length
                    changed = True
        if not changed:
            break
    inverse = np.rint(np.linalg.inv(transform)).astype(float)
    for array in (basis, inverse):
        array.setflags(write=False)
    return basis, transform.astype(float), inverse


def reduced_lattice(
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    error: type[Exception] = LatticeError,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a reduced basis of the same lattice and the integer matrix that makes it.

    ``reduced = transform @ lattice`` with ``transform`` integer and unimodular,
    and no reduced vector can be shortened by an integer multiple of another or by
    adding or subtracting the other two. The nearest-image helpers search the
    27 cells around a fold in this basis, which finds the nearest image even for
    a valid but strongly sheared input cell.
    """

    lattice = lattice_matrix(lattice_vectors, error=error)
    if math.isclose(determinant3(lattice.tolist()), 0.0, abs_tol=1e-12):
        raise error("Lattice matrix is singular and cannot be reduced.")
    basis, transform, _inverse = _reduced_basis(
        tuple(tuple(float(v) for v in row) for row in lattice)
    )
    return basis.copy(), transform.astype(int)


def _reduction(lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduced basis, ``transform`` and its integer inverse for a validated lattice."""

    return _reduced_basis(tuple(tuple(float(value) for value in row) for row in lattice))


def nearest_image_delta(
    a: Sequence[float],
    b: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
) -> list[float]:
    """Return the fractional displacement from ``b``'s nearest periodic image to ``a``.

    The displacement is folded in a reduced basis of the lattice
    (:func:`reduced_lattice`) and searched over its 27 neighbouring cells with
    the lattice metric, so the nearest image is found in hexagonal,
    primitive-centred, monoclinic and sheared cells alike. ``b + delta`` is that
    image: ``delta`` differs from ``a - b`` by whole lattice vectors.
    """

    lattice = lattice_matrix(lattice_vectors)
    reduced, transform, inverse = _reduction(lattice)
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    folded = difference @ inverse
    folded -= np.round(folded)
    candidates = folded[None, :] + _NEIGHBOUR_CELL_OFFSETS
    lengths = np.linalg.norm(candidates @ reduced, axis=1)
    best = candidates[int(np.argmin(lengths))] @ transform
    delta = difference + np.rint(best - difference)
    return [float(value) for value in delta]


def nearest_image_cartesian_delta(
    a: Sequence[float],
    b: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
) -> list[float]:
    """Return :func:`nearest_image_delta` of ``a - b`` in Cartesian coordinates."""

    return _frac_to_cart(nearest_image_delta(a, b, lattice_vectors), lattice_vectors)


def nearest_image_distance(
    a: Sequence[float],
    b: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
) -> float:
    """Return the shortest Cartesian distance between ``a`` and any periodic image of ``b``.

    The length of :func:`nearest_image_cartesian_delta` in Angstrom.
    """

    return vector_length(nearest_image_cartesian_delta(a, b, lattice_vectors))


def nearest_image_distance_matrix(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return the nearest-image distance from ``left[i]`` to ``right[j]`` for every pair (Angstrom).

    The broadcast counterpart of :func:`nearest_image_distance`, with the same
    search in a reduced basis; it may differ from the scalar helper in the last
    bit or so. The result has shape ``(len(left), len(right))``; rows are
    measured in blocks so the 27-cell search stays within a few megabytes
    however large the inputs.
    """

    lattice = lattice_matrix(lattice_vectors)
    reduced, _transform, inverse = _reduction(lattice)
    left_array = np.asarray(left, dtype=float).reshape(-1, 3) @ inverse
    right_array = np.asarray(right, dtype=float).reshape(-1, 3) @ inverse
    shifts = (_NEIGHBOUR_CELL_OFFSETS @ reduced)[None, None]
    distances = np.empty((len(left_array), len(right_array)), dtype=float)
    rows = max(1, _NEAREST_IMAGE_BLOCK // max(1, len(right_array)))
    for start in range(0, len(left_array), rows):
        block = left_array[start : start + rows]
        delta = right_array[None, :, :] - block[:, None, :]
        delta -= np.round(delta)
        candidates = (delta @ reduced)[:, :, None, :] + shifts
        distances[start : start + rows] = np.sqrt(
            np.min(np.einsum("ijkl,ijkl->ijk", candidates, candidates), axis=-1)
        )
    return distances


def nearest_image_delta_matrix(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return the fractional displacement from ``left[i]`` to the nearest image of ``right[j]``.

    The broadcast counterpart of :func:`nearest_image_delta` (with the arguments
    in path order: ``left[i] + delta[i, j]`` is the image of ``right[j]`` nearest
    ``left[i]``), with the same search in a reduced basis. The result has shape
    ``(len(left), len(right), 3)`` and is measured in blocks, as
    :func:`nearest_image_distance_matrix` is.
    """

    lattice = lattice_matrix(lattice_vectors)
    reduction = _reduction(lattice)
    left_array = np.asarray(left, dtype=float).reshape(-1, 3)
    right_array = np.asarray(right, dtype=float).reshape(-1, 3)
    deltas = np.empty((len(left_array), len(right_array), 3), dtype=float)
    rows = max(1, _NEAREST_IMAGE_BLOCK // max(1, len(right_array)))
    for start in range(0, len(left_array), rows):
        difference = right_array[None, :, :] - left_array[start : start + rows, None, :]
        deltas[start : start + rows] = _nearest_image_fold(
            difference.reshape(-1, 3), reduction
        ).reshape(difference.shape)
    return deltas


def nearest_image_deltas(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return the fractional displacement from ``left[i]`` to the nearest image of ``right[i]``.

    The row-by-row counterpart of :func:`nearest_image_delta_matrix`: ``left``
    and ``right`` have the same length and the result has shape ``(n, 3)``.
    """

    left_array = np.asarray(left, dtype=float).reshape(-1, 3)
    right_array = np.asarray(right, dtype=float).reshape(-1, 3)
    if left_array.shape != right_array.shape:
        raise LatticeError("nearest_image_deltas needs as many right positions as left ones.")
    reduction = _reduction(lattice_matrix(lattice_vectors))
    deltas = np.empty_like(left_array)
    for start in range(0, len(left_array), _NEAREST_IMAGE_BLOCK):
        stop = start + _NEAREST_IMAGE_BLOCK
        deltas[start:stop] = _nearest_image_fold(
            right_array[start:stop] - left_array[start:stop], reduction
        )
    return deltas


def _nearest_image_fold(
    difference: np.ndarray, reduction: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """Shift each fractional ``difference`` row by whole lattice vectors to its shortest image."""

    reduced, transform, inverse = reduction
    folded = difference @ inverse
    folded -= np.round(folded)
    candidates = folded[:, None, :] + _NEIGHBOUR_CELL_OFFSETS
    cartesian = candidates @ reduced
    lengths = np.einsum("ikl,ikl->ik", cartesian, cartesian)
    best = candidates[np.arange(len(candidates)), np.argmin(lengths, axis=-1)]
    return difference + np.rint(best @ transform - difference)


def vector_length(vector: Sequence[float]) -> float:
    """Return the Euclidean length of ``vector``."""

    return math.sqrt(sum(float(value) * float(value) for value in vector))


def crystal_axis_to_cartesian(
    components: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
    *,
    error: type[Exception] = LatticeError,
) -> list[float]:
    """Return ``sum_i c_i a_i / |a_i|``: components along the unit lattice directions, in Cartesian.

    This is the magnetic CIF ``crystalaxis`` convention for moments, so in a cell
    whose axes are not orthogonal the components are not Cartesian ones.
    """

    rows = lattice_rows(lattice_vectors, error=error)
    if len(components) != 3:
        raise error("Crystal-axis components must be a 3-vector.")
    lengths = [vector_length(row) for row in rows]
    if any(length == 0.0 for length in lengths):
        raise error("Lattice vectors must be non-zero.")
    return _frac_to_cart(
        [float(value) / length for value, length in zip(components, lengths, strict=True)], rows
    )


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
    return (
        points[..., 0, None] * lattice[0]
        + points[..., 1, None] * lattice[1]
        + points[..., 2, None] * lattice[2]
    )


def cartesian_to_fractional(
    cartesian: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    *,
    error: type[Exception] = LatticeError,
) -> np.ndarray:
    """Convert Cartesian points of shape ``(..., 3)`` to fractional, in the scalar term order."""

    inverse = np.asarray(
        inverse_lattice(lattice_matrix(lattice_vectors, error=error).tolist(), error=error)
    )
    points = np.asarray(cartesian, dtype=float)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise error("Cartesian positions must be 3-vectors.")
    return (
        points[..., 0, None] * inverse[0]
        + points[..., 1, None] * inverse[1]
        + points[..., 2, None] * inverse[2]
    )


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
        wrapped = [
            list(position) for position in dict.fromkeys(tuple(position) for position in wrapped)
        ]
    if not wrapped:
        return []
    distances = minimum_image_distance_matrix(wrapped, wrapped, lattice_vectors)
    within = distances <= tolerance
    for left, right in np.argwhere(np.abs(distances - tolerance) <= abs(tolerance) * NEAR_TIE_RTOL):
        # The scan compares each new position against the kept ones: ``distance(new, kept)``.
        within[left, right] = (
            minimum_image_distance(wrapped[right], wrapped[left], lattice_vectors) <= tolerance
        )
    return [wrapped[index] for index in greedy_unique_indexes(within)]


__all__ = [
    "LatticeError",
    "NEAR_TIE_RTOL",
    "cartesian_to_fractional",
    "cell_volume",
    "crystal_axis_to_cartesian",
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
    "nearest_image_cartesian_delta",
    "nearest_image_delta",
    "nearest_image_delta_matrix",
    "nearest_image_deltas",
    "nearest_image_distance",
    "nearest_image_distance_matrix",
    "reduced_lattice",
    "vector_length",
    "wrap_fractional",
]

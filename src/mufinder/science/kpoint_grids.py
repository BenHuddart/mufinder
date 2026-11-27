"""Helpers for Monkhorst-Pack grid generation from reciprocal-space spacing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return (float(a[0]) * float(b[0])) + (float(a[1]) * float(b[1])) + (float(a[2]) * float(b[2]))


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        (float(a[1]) * float(b[2])) - (float(a[2]) * float(b[1])),
        (float(a[2]) * float(b[0])) - (float(a[0]) * float(b[2])),
        (float(a[0]) * float(b[1])) - (float(a[1]) * float(b[0])),
    )


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(_dot(v, v))


def reciprocal_lattice_vectors(
    lattice_vectors: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return reciprocal vectors in 1/Ang using b_i = 2*pi * (a_j x a_k) / V."""

    if len(lattice_vectors) != 3:
        raise ValueError("lattice_vectors must contain three vectors.")

    a1 = lattice_vectors[0]
    a2 = lattice_vectors[1]
    a3 = lattice_vectors[2]
    if len(a1) != 3 or len(a2) != 3 or len(a3) != 3:
        raise ValueError("Each lattice vector must have three components.")

    v = _dot(a1, _cross(a2, a3))
    if not math.isfinite(v) or abs(v) <= 1e-12:
        raise ValueError("Lattice vectors must define a non-zero cell volume.")

    scale = (2.0 * math.pi) / v
    b1_raw = _cross(a2, a3)
    b2_raw = _cross(a3, a1)
    b3_raw = _cross(a1, a2)

    b1 = tuple(scale * float(component) for component in b1_raw)
    b2 = tuple(scale * float(component) for component in b2_raw)
    b3 = tuple(scale * float(component) for component in b3_raw)
    return b1, b2, b3


def grid_from_spacing(
    lattice_vectors: Sequence[Sequence[float]],
    spacing: float,
) -> tuple[int, int, int]:
    """Return CASTEP-equivalent KPOINT_MP_GRID for a spacing in 1/Ang.

    CASTEP uses reciprocal-vector magnitudes with:
    ``n_i = floor(|b_i| / (2*pi*spacing)) + 1``
    """

    spacing_value = float(spacing)
    if not math.isfinite(spacing_value) or spacing_value <= 0.0:
        raise ValueError("spacing must be a positive finite value in 1/Ang.")

    reciprocal = reciprocal_lattice_vectors(lattice_vectors)
    denominator = 2.0 * math.pi * spacing_value

    grid: list[int] = []
    for vector in reciprocal:
        magnitude = _norm(vector)
        # 1e-9 corrects ULP-level rounding errors when the ratio is the
        # result of spacing_from_grid's exact boundary formula (|b|/(2π·(N-1))),
        # where two-step floating-point division can yield N-1 - ε instead of
        # exactly N-1, causing floor to snap down by one.  The epsilon is well
        # below 1 so it never advances a genuinely non-integer ratio.
        count = int(math.floor(magnitude / denominator + 1e-9)) + 1
        grid.append(max(1, count))

    return (grid[0], grid[1], grid[2])


def kpoint_count_for_spacing(
    lattice_vectors: Sequence[Sequence[float]],
    spacing: float,
) -> float:
    """Continuous interpolation of CASTEP's spacing -> n_kpoints relation.

    Drops the ``floor`` from :func:`grid_from_spacing` so the mapping is
    smooth, letting us position spacing-labelled tick marks on a convergence
    plot whose primary x-axis is the integer number of k-points. At integer
    grids the continuous value agrees with the discrete CASTEP grid.
    """

    spacing_value = float(spacing)
    if not math.isfinite(spacing_value) or spacing_value <= 0.0:
        raise ValueError("spacing must be a positive finite value in 1/Ang.")

    reciprocal = reciprocal_lattice_vectors(lattice_vectors)
    denominator = 2.0 * math.pi * spacing_value

    product = 1.0
    for vector in reciprocal:
        magnitude = _norm(vector)
        product *= (magnitude / denominator) + 1.0
    return product


def generate_kpoint_grids(
    lattice_vectors: Sequence[Sequence[float]],
    spacing: float,
) -> tuple[tuple[int, int, int], ...]:
    """Enumerate the distinct CASTEP grids from ``(1,1,1)`` up to the spacing-derived cap.

    Walks CASTEP's spacing trajectory through ``(n_1, n_2, n_3)`` space so that
    every emitted grid is reproducible by some ``KPOINT_MP_SPACING`` via
    :func:`grid_from_spacing`. CASTEP's spacing-to-grid mapping is piecewise
    constant with breakpoints at ``|b_i|/(2*pi*n)`` for axis ``i`` and integer
    ``n``; probing at each breakpoint enumerates every reachable grid between
    ``(1, 1, 1)`` and ``grid_from_spacing(lattice_vectors, spacing)``.
    """

    max_grid = grid_from_spacing(lattice_vectors, spacing)
    reciprocal = reciprocal_lattice_vectors(lattice_vectors)
    magnitudes = tuple(_norm(vector) for vector in reciprocal)
    two_pi = 2.0 * math.pi

    cap = max(max_grid)
    thresholds: set[float] = set()
    for magnitude in magnitudes:
        if magnitude <= 0.0:
            continue
        for index in range(1, cap + 1):
            thresholds.add(magnitude / (two_pi * index))

    grids: list[tuple[int, int, int]] = [(1, 1, 1)]
    seen: set[tuple[int, int, int]] = {(1, 1, 1)}
    for threshold in sorted(thresholds, reverse=True):
        candidate = grid_from_spacing(lattice_vectors, threshold)
        clamped = (
            min(candidate[0], max_grid[0]),
            min(candidate[1], max_grid[1]),
            min(candidate[2], max_grid[2]),
        )
        if clamped not in seen:
            seen.add(clamped)
            grids.append(clamped)
        if clamped == max_grid:
            break

    return tuple(grids)


def kpoint_count(grid: tuple[int, int, int]) -> int:
    return int(grid[0]) * int(grid[1]) * int(grid[2])


def format_kpoint_grid(grid: tuple[int, int, int]) -> str:
    return f"{int(grid[0])} x {int(grid[1])} x {int(grid[2])}"


def most_uniform_grid(
    lattice_vectors: Sequence[Sequence[float]],
    grids: Sequence[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """Return the grid whose per-axis k-point spacings are closest to equal.

    Each axis is sampled at spacing ``|b_i| / (2*pi*n_i)``; the most uniform grid
    has the smallest ratio of its largest to its smallest axis spacing, which is
    the shape ``generate_kpoint_grids`` produces for the cell. Used to choose
    between grids with the same k-point count, which cannot share a point on a
    count-based convergence axis. Ties keep the earlier grid.
    """

    if not grids:
        raise ValueError("At least one grid is required.")
    magnitudes = [_norm(vector) for vector in reciprocal_lattice_vectors(lattice_vectors)]

    def anisotropy(grid: tuple[int, int, int]) -> float:
        spacings = [
            magnitude / (2.0 * math.pi * max(1, int(n)))
            for magnitude, n in zip(magnitudes, grid, strict=True)
        ]
        return max(spacings) / min(spacings)

    best = grids[0]
    best_score = anisotropy(best)
    for grid in grids[1:]:
        score = anisotropy(grid)
        if score < best_score - 1e-12:
            best, best_score = grid, score
    return best


def lattice_matches(
    lhs: Sequence[Sequence[float]],
    rhs: Sequence[Sequence[float]],
    *,
    tol: float = 1e-6,
) -> bool:
    if len(lhs) != 3 or len(rhs) != 3:
        return False

    for lhs_row, rhs_row in zip(lhs, rhs, strict=False):
        if len(lhs_row) != 3 or len(rhs_row) != 3:
            return False
        for lhs_value, rhs_value in zip(lhs_row, rhs_row, strict=False):
            if abs(float(lhs_value) - float(rhs_value)) > tol:
                return False
    return True


def lattice_parameters(
    lattice_vectors: Sequence[Sequence[float]],
) -> tuple[float, float, float, float, float, float]:
    """Return ``(a, b, c, alpha, beta, gamma)`` for a 3x3 lattice.

    Lengths are in Angstrom and angles are in degrees.
    """

    if len(lattice_vectors) != 3:
        raise ValueError("lattice_vectors must contain three vectors.")

    a1 = lattice_vectors[0]
    a2 = lattice_vectors[1]
    a3 = lattice_vectors[2]
    if len(a1) != 3 or len(a2) != 3 or len(a3) != 3:
        raise ValueError("Each lattice vector must have three components.")

    a = _norm(a1)
    b = _norm(a2)
    c = _norm(a3)
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        raise ValueError("Lattice vector lengths must be greater than zero.")

    alpha = _angle_degrees(a2, a3)
    beta = _angle_degrees(a1, a3)
    gamma = _angle_degrees(a1, a2)
    return (a, b, c, alpha, beta, gamma)


def spacing_from_grid(
    lattice_vectors: Sequence[Sequence[float]],
    grid: tuple[int, int, int],
) -> float:
    """Return a ``KPOINT_MP_SPACING`` that CASTEP rounds back to this grid.

    Returns ``min_i( |b_i| / (2*pi * (N_i - 1)) )`` over axes with ``N_i >= 2``
    — the inclusive upper boundary of CASTEP's valid input range for ``grid``.
    Any value at or just below this boundary reproduces ``grid`` exactly via
    CASTEP's ``n_i = floor(|b_i|/(2*pi*s)) + 1`` formula.

    Raises ``ValueError`` for ``(1, 1, 1)`` — Gamma-only sampling is reached
    by any spacing above the coarsest reciprocal-space transition, so no
    single value characterises the grid.
    """

    n1 = int(grid[0])
    n2 = int(grid[1])
    n3 = int(grid[2])
    if n1 <= 0 or n2 <= 0 or n3 <= 0:
        raise ValueError("grid values must be positive integers.")

    reciprocal = reciprocal_lattice_vectors(lattice_vectors)
    magnitudes = (_norm(reciprocal[0]), _norm(reciprocal[1]), _norm(reciprocal[2]))
    counts = (n1, n2, n3)

    candidates = [
        magnitude / (2.0 * math.pi * float(count - 1))
        for magnitude, count in zip(magnitudes, counts, strict=True)
        if count >= 2
    ]
    if not candidates:
        raise ValueError("No spacing represents the (1, 1, 1) Gamma-only grid.")
    return min(candidates)


def distinct_grid_spacings(
    lattice_vectors: Sequence[Sequence[float]],
    grids: Mapping[float, tuple[int, int, int]],
) -> dict[float, float]:
    """Return the spacing of each grid on ``lattice_vectors``, keyed like ``grids``.

    The keys are visited in ascending order and the series is strictly
    decreasing: the Gamma-only grid, grids with no spacing, and repeats of the
    previous spacing are left out.
    """

    spacings: dict[float, float] = {}
    last_spacing: float | None = None
    for x_value in sorted(grids):
        grid = grids[x_value]
        # (1,1,1) is Gamma-only sampling; no single spacing characterises
        # it (every spacing above the cell's coarsest reciprocal transition
        # produces it) so spacing_from_grid raises. Skip explicitly so the
        # annotation machinery never exposes it to the UI.
        if grid == (1, 1, 1):
            continue
        try:
            spacing = spacing_from_grid(lattice_vectors, grid)
        except ValueError:
            continue
        # Skip duplicate spacings: when one axis dominates the round-trip
        # bound, successive grids can report the same value even as other
        # axes grow. Once a plateau starts, every subsequent repeat is
        # suppressed so the annotation series stays strictly decreasing
        # and does not mislead the user about convergence status.
        if last_spacing is not None and math.isclose(spacing, last_spacing, rel_tol=1e-9):
            continue
        spacings[float(x_value)] = spacing
        last_spacing = spacing
    return spacings


def supercell_lattice_vectors(
    lattice_vectors: Sequence[Sequence[float]],
    supercell_matrix: Sequence[Sequence[int]],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return the Cartesian lattice vectors for a transformed supercell."""

    if len(supercell_matrix) != 3:
        raise ValueError("supercell_matrix must contain three rows.")
    for row in supercell_matrix:
        if len(row) != 3:
            raise ValueError("Each supercell matrix row must contain three values.")

    if len(lattice_vectors) != 3:
        raise ValueError("lattice_vectors must contain three vectors.")
    for vector in lattice_vectors:
        if len(vector) != 3:
            raise ValueError("Each lattice vector must have three components.")

    result: list[tuple[float, float, float]] = []
    for row in supercell_matrix:
        components: list[float] = []
        for axis in range(3):
            value = 0.0
            for column in range(3):
                value += float(row[column]) * float(lattice_vectors[column][axis])
            components.append(value)
        result.append((components[0], components[1], components[2]))

    return (result[0], result[1], result[2])


def equivalent_grid_for_supercell(
    reference_lattice: Sequence[Sequence[float]],
    reference_grid: tuple[int, int, int],
    supercell_lattice: Sequence[Sequence[float]],
) -> tuple[int, int, int]:
    """Return a supercell grid with approximately equivalent k-point density."""

    reference_reciprocal = reciprocal_lattice_vectors(reference_lattice)
    supercell_reciprocal = reciprocal_lattice_vectors(supercell_lattice)

    grid: list[int] = []
    for reference_count, reference_vector, supercell_vector in zip(
        reference_grid,
        reference_reciprocal,
        supercell_reciprocal,
        strict=True,
    ):
        count = int(reference_count)
        if count <= 0:
            raise ValueError("reference_grid values must be positive integers.")
        reference_magnitude = _norm(reference_vector)
        if reference_magnitude <= 0.0:
            raise ValueError("reference_lattice must define non-zero reciprocal vectors.")

        ratio = _norm(supercell_vector) / reference_magnitude
        grid.append(max(1, int(math.ceil((float(count) * ratio) - 1e-12))))

    return (grid[0], grid[1], grid[2])


def _angle_degrees(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    denominator = _norm(lhs) * _norm(rhs)
    if denominator <= 0.0:
        raise ValueError("Cannot compute angle from zero-length lattice vector.")
    cosine = _dot(lhs, rhs) / denominator
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))

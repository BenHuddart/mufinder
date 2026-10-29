"""Muon candidate-site generation with periodic boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import warnings
from collections.abc import Callable
from typing import Sequence

import numpy as np

from mufinder.science.lattice import (
    cell_volume,
    lattice_rows,
    minimum_image_distance_matrix,
    nearest_image_cartesian_delta,
    nearest_image_distance_matrix,
    vector_length,
    wrap_fractional,
)
from mufinder.science.lattice import lattice_cart_to_frac as _lattice_cart_to_frac
from mufinder.science.lattice import lattice_frac_to_cart as _lattice_frac_to_cart
from mufinder.science.spglib_compat import (
    SYMMETRY_TOLERANCE_ANG,
    SymmetryOperations,
    get_crystal_symmetry,
    spglib_cell,
    symmetry_operations,
    symmetry_orbit,
)


class SiteGenerationError(ValueError):
    """Raised when candidate generation cannot satisfy constraints."""


_RANDOM_FILLING_FRACTION = 0.350
_FREE_VOLUME_GRID_RESOLUTION = 18
ProgressCallback = Callable[[int, int], None]

# Spacing search for a target count: bisect between a lower bound and an upper bound
# grown from the longest lattice vector until it predicts no more than the target.
_SPACING_SEARCH_LOWER_ANG = 1e-3
_SPACING_SEARCH_GROWTH = 1.5
_SPACING_SEARCH_MAX_SCALE = 8.0
_SPACING_SEARCH_MIN_UPPER_ANG = 10.0
_SPACING_SEARCH_BISECTIONS = 36
_SPACING_SEARCH_MIN_RESULT_ANG = 0.1

#: Neighbours within this distance (Angstrom) of an atom set the directions an attached start avoids.
ATTACHED_NEIGHBOUR_CUTOFF_ANG = 2.6
#: Two trial directions for an attached start whose cosine exceeds this are one direction.
_ATTACHED_DIRECTION_COSINE_LIMIT = 0.999
#: Neighbours closer than this (Angstrom) coincide with the atom and give no direction.
_COINCIDENT_ATOM_ANG = 1e-8
#: Slack (Angstrom) below the minimum muon-atom distance an attached start may still have.
_ATTACHED_CLEARANCE_SLACK_ANG = 1e-8
#: Two attached starts on one atom closer than this (Angstrom) are one start.
_ATTACHED_DUPLICATE_START_ANG = 1e-5
#: A direction vector shorter than this has no direction.
_ZERO_VECTOR_LENGTH = 1e-12


@dataclass(slots=True)
class GeneratedPoint:
    """Generated position in fractional and Cartesian coordinates."""

    fractional: list[float]
    cartesian: list[float]


def _validate_lattice(lattice_vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    return lattice_rows(lattice_vectors, error=SiteGenerationError)


def _identity_lattice() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def lattice_frac_to_cart(
    fractional: Sequence[float], lattice_vectors: Sequence[Sequence[float]]
) -> list[float]:
    """Convert fractional coordinate to Cartesian coordinate.

    See :func:`mufinder.science.lattice.lattice_frac_to_cart`; this name raises
    :class:`SiteGenerationError` for a malformed lattice.
    """
    return _lattice_frac_to_cart(fractional, lattice_vectors, error=SiteGenerationError)


def lattice_cart_to_frac(
    cartesian: Sequence[float], lattice_vectors: Sequence[Sequence[float]]
) -> list[float]:
    """Convert Cartesian coordinate to fractional coordinate.

    See :func:`mufinder.science.lattice.lattice_cart_to_frac`; this name raises
    :class:`SiteGenerationError` for a malformed or singular lattice.
    """
    return _lattice_cart_to_frac(cartesian, lattice_vectors, error=SiteGenerationError)


def _sphere_volume(radius: float) -> float:
    return 4.0 * math.pi * float(radius) ** 3 / 3.0


#: Deduplication tolerance (Angstrom) for the images of one candidate's symmetry orbit.
_ORBIT_TOLERANCE_ANG = 1e-5


def _symmetry_operations(
    *,
    lattice_vectors: Sequence[Sequence[float]],
    host_species: Sequence[str],
    host_positions_frac: Sequence[Sequence[float]],
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> SymmetryOperations | None:
    lattice = _validate_lattice(lattice_vectors)
    cell = spglib_cell(
        lattice_vectors=lattice, species=host_species, fractional_positions=host_positions_frac
    )
    if cell is None:
        return None
    return symmetry_operations(cell, symprec=float(symprec))


def symmetry_equivalent_positions(
    position_frac: Sequence[float],
    *,
    lattice_vectors: Sequence[Sequence[float]],
    host_species: Sequence[str],
    host_positions_frac: Sequence[Sequence[float]],
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> list[list[float]]:
    """Return symmetry-equivalent fractional positions for one candidate.

    If symmetry detection is unavailable for the supplied structure, the input
    position is returned as a single wrapped representative.
    """

    lattice = _validate_lattice(lattice_vectors)
    operations = _symmetry_operations(
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions_frac,
        symprec=symprec,
    )
    return symmetry_orbit(
        position_frac, operations, lattice_vectors=lattice, tolerance=_ORBIT_TOLERANCE_ANG
    )


def closest_symmetry_equivalent_position(
    position_frac: Sequence[float],
    *,
    anchor_frac: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
    host_species: Sequence[str],
    host_positions_frac: Sequence[Sequence[float]],
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> list[float]:
    """Return the symmetry-equivalent representative closest to an anchor.

    Closeness is the nearest-image distance, so the choice is right in skewed cells
    too; of equally close representatives the first in orbit order wins.
    """

    lattice = _validate_lattice(lattice_vectors)
    anchor = wrap_fractional(anchor_frac)
    equivalents = symmetry_equivalent_positions(
        position_frac,
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions_frac,
        symprec=symprec,
    )
    distances = nearest_image_distance_matrix([anchor], equivalents, lattice)[0]
    return equivalents[int(np.argmin(distances))]


def _symmetry_multiplicity(
    *,
    lattice_vectors: Sequence[Sequence[float]],
    host_species: Sequence[str] | None,
    host_positions_frac: Sequence[Sequence[float]],
    include_symmetry_equivalents: bool,
    symprec: float,
) -> int:
    if not include_symmetry_equivalents or not host_species:
        return 1
    operations = _symmetry_operations(
        lattice_vectors=lattice_vectors,
        host_species=host_species,
        host_positions_frac=host_positions_frac,
        symprec=symprec,
    )
    if operations is None:
        return 1
    _rotations, translations = operations
    return max(1, len(translations))


def estimate_free_volume(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_positions_frac: Sequence[Sequence[float]],
    exclusion_radius: float,
    grid_resolution: int = _FREE_VOLUME_GRID_RESOLUTION,
) -> float:
    """Estimate cell volume outside host-atom exclusion spheres.

    This deterministic grid estimate is intentionally conservative and cheap
    enough for UI use.  It replaces the old GUI-side ray-tracing estimate while
    preserving the same volume model used to infer random-site spacings.
    """

    lattice = _validate_lattice(lattice_vectors) if lattice_vectors else _identity_lattice()
    volume = cell_volume(lattice)
    if exclusion_radius <= 0.0 or not host_positions_frac:
        return volume
    if volume <= 0.0:
        raise SiteGenerationError("Cell volume must be positive.")

    resolution = max(4, int(grid_resolution))
    axis = (np.arange(resolution, dtype=float) + 0.5) / float(resolution)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    hosts = np.asarray([wrap_fractional(position) for position in host_positions_frac], dtype=float)
    lattice_array = np.asarray(lattice, dtype=float)
    delta_frac = grid[:, None, :] - hosts[None, :, :]
    delta_frac -= np.rint(delta_frac)
    delta_cart = delta_frac @ lattice_array
    distances = np.linalg.norm(delta_cart, axis=2)
    free_fraction = float(
        np.count_nonzero(np.min(distances, axis=1) >= float(exclusion_radius))
    ) / float(len(grid))
    return volume * free_fraction


def estimate_random_site_count_from_separation(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_positions_frac: Sequence[Sequence[float]],
    min_muon_muon: float,
    min_muon_atom: float,
    host_species: Sequence[str] | None = None,
    include_symmetry_equivalents: bool = True,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    filling_fraction: float = _RANDOM_FILLING_FRACTION,
    grid_resolution: int = _FREE_VOLUME_GRID_RESOLUTION,
) -> float:
    """Estimate random candidate count for one muon-muon separation."""

    if min_muon_muon <= 0.0:
        raise SiteGenerationError("Muon-muon spacing must be greater than zero.")
    if min_muon_atom < 0.0:
        raise SiteGenerationError("Distance constraints must be non-negative.")

    lattice = _validate_lattice(lattice_vectors) if lattice_vectors else _identity_lattice()
    muon_radius = float(min_muon_muon) / 2.0
    single_site_volume = _sphere_volume(muon_radius) * _symmetry_multiplicity(
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions_frac,
        include_symmetry_equivalents=include_symmetry_equivalents,
        symprec=symprec,
    )
    if single_site_volume <= 0.0:
        raise SiteGenerationError("Muon site volume must be positive.")

    atom_exclusion_radius = max(float(min_muon_atom) - muon_radius, 0.0)
    free_volume = estimate_free_volume(
        lattice_vectors=lattice,
        host_positions_frac=host_positions_frac,
        exclusion_radius=atom_exclusion_radius,
        grid_resolution=grid_resolution,
    )
    return max(0.0, float(filling_fraction) * free_volume / single_site_volume)


def estimate_muon_muon_separation_for_count(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_positions_frac: Sequence[Sequence[float]],
    target_count: int,
    min_muon_atom: float,
    host_species: Sequence[str] | None = None,
    include_symmetry_equivalents: bool = True,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    filling_fraction: float = _RANDOM_FILLING_FRACTION,
    grid_resolution: int = _FREE_VOLUME_GRID_RESOLUTION,
) -> float:
    """Estimate the muon-muon spacing that should produce ``target_count`` sites."""

    if target_count <= 0:
        raise SiteGenerationError("Target count must be greater than zero.")
    if min_muon_atom < 0.0:
        raise SiteGenerationError("Distance constraints must be non-negative.")

    lattice = _validate_lattice(lattice_vectors) if lattice_vectors else _identity_lattice()
    vector_lengths = [
        math.sqrt(sum(component * component for component in vector)) for vector in lattice
    ]
    high = max(max(vector_lengths, default=1.0), 1.0)
    low = _SPACING_SEARCH_LOWER_ANG

    def expected_count(spacing: float) -> float:
        return estimate_random_site_count_from_separation(
            lattice_vectors=lattice,
            host_positions_frac=host_positions_frac,
            min_muon_muon=spacing,
            min_muon_atom=min_muon_atom,
            host_species=host_species,
            include_symmetry_equivalents=include_symmetry_equivalents,
            symprec=symprec,
            filling_fraction=filling_fraction,
            grid_resolution=grid_resolution,
        )

    max_high = max(high * _SPACING_SEARCH_MAX_SCALE, _SPACING_SEARCH_MIN_UPPER_ANG)
    while expected_count(high) > target_count and high < max_high:
        high *= _SPACING_SEARCH_GROWTH

    for _ in range(_SPACING_SEARCH_BISECTIONS):
        midpoint = 0.5 * (low + high)
        if expected_count(midpoint) > target_count:
            low = midpoint
        else:
            high = midpoint

    return max(_SPACING_SEARCH_MIN_RESULT_ANG, high)


def _violates_spacing(
    point: np.ndarray,
    checks: Sequence[tuple[np.ndarray, float]],
    lattice_vectors: Sequence[Sequence[float]],
) -> bool:
    """Whether ``point`` lies closer than a limit to the nearest image of any of its references.

    ``checks`` pairs each reference array with its limit. The per-component fold is
    never shorter than the nearest image, so any fold inside its limit rejects the
    point at once; the 27-cell nearest-image search runs only when no fold does,
    which in a filling cell is mostly the trial about to be accepted.
    """

    position = np.asarray(point, dtype=float).reshape(1, 3)
    active = [(references, limit) for references, limit in checks if len(references)]
    for references, limit in active:
        if np.any(minimum_image_distance_matrix(position, references, lattice_vectors) < limit):
            return True
    return any(
        bool(np.any(nearest_image_distance_matrix(position, references, lattice_vectors) < limit))
        for references, limit in active
    )


def _host_position_array(host_positions_frac: Sequence[Sequence[float]]) -> np.ndarray:
    hosts = np.asarray(host_positions_frac, dtype=float)
    if hosts.size == 0:
        return hosts.reshape(0, 3)
    if hosts.ndim != 2 or hosts.shape[1] != 3:
        raise SiteGenerationError("Host positions must be fractional (x, y, z) triples.")
    return hosts


def _normalized(vector: Sequence[float]) -> list[float] | None:
    norm = vector_length(vector)
    if norm <= _ZERO_VECTOR_LENGTH:
        return None
    return [float(value) / norm for value in vector]


def generate_random_sites(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_positions_frac: Sequence[Sequence[float]],
    count: int | None = None,
    min_muon_muon: float,
    min_muon_atom: float,
    max_attempts: int = 20000,
    seed: int | None = None,
    allow_partial: bool = False,
    include_symmetry_equivalents: bool = False,
    host_species: Sequence[str] | None = None,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    expected_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GeneratedPoint], int]:
    """Generate candidate sites by random filling with distance constraints.

    With ``count`` set, trials stop once that many sites are accepted, and fewer
    raise :class:`SiteGenerationError` unless ``allow_partial``. With ``count=None``
    the cell is filled until ``max_attempts`` trials have run, and
    ``expected_count`` (when given) is the progress total until it is passed.

    A trial is accepted when its nearest-image distance is at least
    ``min_muon_atom`` from every host atom and at least ``min_muon_muon`` from
    every accepted image. With ``include_symmetry_equivalents`` the accepted images
    are each accepted site's whole symmetry orbit. The host and those orbits are
    symmetry-invariant and so are distances, so checking the trial alone checks
    its whole orbit.

    Returns a tuple: (sites, attempts_used).
    """
    fill = count is None
    if fill:
        if min_muon_muon <= 0.0:
            raise SiteGenerationError("Muon-muon spacing must be greater than zero.")
        if min_muon_atom < 0.0:
            raise SiteGenerationError("Distance constraints must be non-negative.")
    else:
        if count <= 0:
            raise SiteGenerationError("Count must be greater than zero.")
        if min_muon_muon < 0.0 or min_muon_atom < 0.0:
            raise SiteGenerationError("Distance constraints must be non-negative.")

    lattice = _validate_lattice(lattice_vectors) if lattice_vectors else _identity_lattice()
    rng = random.Random(seed)
    operations = (
        _symmetry_operations(
            lattice_vectors=lattice,
            host_species=host_species or [],
            host_positions_frac=host_positions_frac,
            symprec=symprec,
        )
        if include_symmetry_equivalents and host_species
        else None
    )

    host_positions = _host_position_array(host_positions_frac)
    accepted_positions = np.empty((0, 3), dtype=float)
    accepted_points: list[GeneratedPoint] = []
    progress_total = max(1, int(expected_count or 1)) if fill else int(count)
    if progress_callback is not None:
        progress_callback(0, progress_total)

    attempts = 0
    while (fill or len(accepted_points) < count) and attempts < max_attempts:
        attempts += 1
        candidate_frac = [rng.random(), rng.random(), rng.random()]
        if _violates_spacing(
            np.asarray(candidate_frac, dtype=float),
            ((host_positions, min_muon_atom), (accepted_positions, min_muon_muon)),
            lattice,
        ):
            continue

        orbit = symmetry_orbit(
            candidate_frac, operations, lattice_vectors=lattice, tolerance=_ORBIT_TOLERANCE_ANG
        )
        accepted_positions = np.concatenate([accepted_positions, np.asarray(orbit, dtype=float)])
        accepted_points.append(
            GeneratedPoint(
                fractional=wrap_fractional(candidate_frac),
                cartesian=lattice_frac_to_cart(candidate_frac, lattice),
            )
        )
        if progress_callback is not None:
            progress_callback(len(accepted_points), max(progress_total, len(accepted_points)))

    if not accepted_points:
        raise SiteGenerationError(
            "Could not generate any candidate sites with current constraints. "
            "Relax distance limits or increase supercell size."
        )
    if not fill and len(accepted_points) < count and not allow_partial:
        raise SiteGenerationError(
            f"Could only generate {len(accepted_points)} of {count} candidate sites after "
            f"{attempts} attempts. "
            "Increase max_attempts, relax distance limits, or increase supercell size."
        )

    return accepted_points, attempts


def generate_random_sites_by_distance(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_positions_frac: Sequence[Sequence[float]],
    min_muon_muon: float,
    min_muon_atom: float,
    max_attempts: int = 20000,
    seed: int | None = None,
    include_symmetry_equivalents: bool = False,
    host_species: Sequence[str] | None = None,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    expected_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GeneratedPoint], int]:
    """Deprecated: call :func:`generate_random_sites` with ``count=None``.

    Kept for external callers (see the compatibility ledger); raises a
    ``DeprecationWarning``.
    """

    warnings.warn(
        "generate_random_sites_by_distance is deprecated; "
        "call generate_random_sites with count=None instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return generate_random_sites(
        lattice_vectors=lattice_vectors,
        host_positions_frac=host_positions_frac,
        count=None,
        min_muon_muon=min_muon_muon,
        min_muon_atom=min_muon_atom,
        max_attempts=max_attempts,
        seed=seed,
        include_symmetry_equivalents=include_symmetry_equivalents,
        host_species=host_species,
        symprec=symprec,
        expected_count=expected_count,
        progress_callback=progress_callback,
    )


def _fibonacci_sphere_directions(count: int) -> list[list[float]]:
    if count <= 0:
        return []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    directions: list[list[float]] = []
    for index in range(count):
        z = 1.0 - (2.0 * (index + 0.5) / count)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = index * golden_angle
        directions.append([math.cos(theta) * radius, math.sin(theta) * radius, z])
    return directions


def symmetry_distinct_atom_indexes(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_species: Sequence[str],
    host_positions_frac: Sequence[Sequence[float]],
    target_element: str,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> list[int]:
    """Return the index of the first atom of ``target_element`` in each symmetry-equivalent set.

    The sets are spglib's ``equivalent_atoms`` for the host. When the host's symmetry
    cannot be detected every matching atom is its own set
    (:func:`mufinder.science.spglib_compat.symmetry_detection_warning` says why).
    """

    target = target_element.strip().lower()
    matching = [
        index for index, species in enumerate(host_species) if species.strip().lower() == target
    ]
    if not matching:
        return []
    lattice = _validate_lattice(lattice_vectors) if lattice_vectors else _identity_lattice()
    cell = spglib_cell(
        lattice_vectors=lattice, species=host_species, fractional_positions=host_positions_frac
    )
    symmetry = None if cell is None else get_crystal_symmetry(cell, symprec=float(symprec))
    if symmetry is None:
        return matching
    equivalent_atoms = symmetry["equivalent_atoms"]
    seen: set[int] = set()
    distinct: list[int] = []
    for index in matching:
        orbit = int(equivalent_atoms[index])
        if orbit not in seen:
            seen.add(orbit)
            distinct.append(index)
    return distinct


def generate_attached_element_sites(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_species: Sequence[str],
    host_positions_frac: Sequence[Sequence[float]],
    target_element: str,
    muons_per_atom: int = 1,
    bond_distance: float = 1.0,
    min_muon_atom: float = 0.8,
    neighbor_cutoff: float = ATTACHED_NEIGHBOUR_CUTOFF_ANG,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GeneratedPoint], int]:
    """Generate deterministic element-attached starts using local bonding geometry.

    One atom of ``target_element`` is taken from each symmetry-equivalent set
    (:func:`symmetry_distinct_atom_indexes`): starts on equivalent atoms are
    symmetry images of each other. For each such atom, candidate directions point
    away from nearby neighbours and a deterministic spherical fallback set. The
    best ``muons_per_atom`` candidates for each atom are selected by clearance from
    the surrounding host structure. Every distance is a nearest-image distance.
    """

    if not target_element.strip():
        raise SiteGenerationError("Target element cannot be empty.")
    if muons_per_atom <= 0:
        raise SiteGenerationError("Muons per atom must be greater than zero.")
    if bond_distance <= 0.0:
        raise SiteGenerationError("Bond distance must be greater than zero.")
    if min_muon_atom < 0.0:
        raise SiteGenerationError("Muon-atom distance must be non-negative.")

    lattice = _validate_lattice(lattice_vectors) if lattice_vectors else _identity_lattice()
    target_indexes = symmetry_distinct_atom_indexes(
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions_frac,
        target_element=target_element,
        symprec=symprec,
    )
    if not target_indexes:
        raise SiteGenerationError(f"No host atoms found for element '{target_element}'.")
    expected_count = len(target_indexes) * muons_per_atom
    if progress_callback is not None:
        progress_callback(0, expected_count)

    axis_directions = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]

    host_positions = _host_position_array(host_positions_frac)
    accepted_points: list[GeneratedPoint] = []
    attempts = 0
    direction_pool_size = max(24, muons_per_atom * 8)
    for target_index in target_indexes:
        anchor_frac = host_positions_frac[target_index]
        anchor_cart = lattice_frac_to_cart(anchor_frac, lattice)
        neighbor_dirs: list[list[float]] = []
        for other_index, other_frac in enumerate(host_positions_frac):
            if other_index == target_index:
                continue
            delta = nearest_image_cartesian_delta(other_frac, anchor_frac, lattice)
            distance = vector_length(delta)
            if _COINCIDENT_ATOM_ANG < distance <= neighbor_cutoff:
                direction = _normalized(delta)
                if direction is not None:
                    neighbor_dirs.append(direction)

        candidate_dirs: list[list[float]] = []
        for direction in neighbor_dirs:
            candidate_dirs.append([-value for value in direction])
        if neighbor_dirs:
            summed = [sum(direction[axis] for direction in neighbor_dirs) for axis in range(3)]
            away = _normalized([-value for value in summed])
            if away is not None:
                candidate_dirs.append(away)
        candidate_dirs.extend(axis_directions)
        candidate_dirs.extend(_fibonacci_sphere_directions(direction_pool_size))

        unique_dirs: list[list[float]] = []
        for direction in candidate_dirs:
            normalized = _normalized(direction)
            if normalized is None:
                continue
            if any(
                sum(normalized[i] * existing[i] for i in range(3))
                > _ATTACHED_DIRECTION_COSINE_LIMIT
                for existing in unique_dirs
            ):
                continue
            unique_dirs.append(normalized)

        candidates = [
            wrap_fractional(
                lattice_cart_to_frac(
                    [anchor_cart[axis] + direction[axis] * bond_distance for axis in range(3)],
                    lattice,
                )
            )
            for direction in unique_dirs
        ]
        attempts += len(candidates)
        # Distance from each candidate to every host atom, and to every atom but the anchor.
        clearances = nearest_image_distance_matrix(candidates, host_positions, lattice)
        others = np.delete(clearances, target_index, axis=1)
        per_atom: list[tuple[float, list[float]]] = []
        for row, candidate_frac in enumerate(candidates):
            if clearances.shape[1] and (
                float(clearances[row].min()) < min_muon_atom - _ATTACHED_CLEARANCE_SLACK_ANG
            ):
                continue
            clearance = float(others[row].min()) if others.shape[1] else math.inf
            per_atom.append((clearance, candidate_frac))
        per_atom.sort(key=lambda item: item[0], reverse=True)
        selected_for_atom: list[list[float]] = []
        for _clearance, candidate_frac in per_atom:
            if selected_for_atom and bool(
                np.any(
                    nearest_image_distance_matrix([candidate_frac], selected_for_atom, lattice)
                    < _ATTACHED_DUPLICATE_START_ANG
                )
            ):
                continue
            selected_for_atom.append(candidate_frac)
            accepted_points.append(
                GeneratedPoint(
                    fractional=candidate_frac,
                    cartesian=lattice_frac_to_cart(candidate_frac, lattice),
                )
            )
            if progress_callback is not None:
                progress_callback(len(accepted_points), expected_count)
            if len(selected_for_atom) >= muons_per_atom:
                break

    if len(accepted_points) < expected_count:
        generated = len(accepted_points)
        if generated == 0:
            raise SiteGenerationError(
                "Could not place any attached-element candidate sites with current constraints."
            )
        raise SiteGenerationError(
            f"Could only generate {generated} of {expected_count} attached-element candidate sites. "
            "Relax the muon-atom distance, reduce muons per atom, or choose another element."
        )

    return accepted_points, attempts

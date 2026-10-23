"""Muon candidate-site generation with periodic boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections.abc import Callable
from typing import Sequence

import numpy as np

from mufinder.science.lattice import (
    NEAR_TIE_RTOL,
    cell_volume,
    lattice_rows,
    minimum_image_cartesian_delta,
    minimum_image_distance,
    minimum_image_distance_matrix,
    near_minimum_indexes,
    vector_length,
    wrap_fractional,
)
from mufinder.science.lattice import lattice_cart_to_frac as _lattice_cart_to_frac
from mufinder.science.lattice import lattice_frac_to_cart as _lattice_frac_to_cart
from mufinder.science.spglib_compat import (
    SymmetryOperations,
    spglib_cell,
    symmetry_operations,
    symmetry_orbit,
)


class SiteGenerationError(ValueError):
    """Raised when candidate generation cannot satisfy constraints."""


_RANDOM_FILLING_FRACTION = 0.350
_FREE_VOLUME_GRID_RESOLUTION = 18
ProgressCallback = Callable[[int, int], None]


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
    symprec: float = 1e-5,
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
    symprec: float = 1e-5,
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
    symprec: float = 1e-5,
) -> list[float]:
    """Return the symmetry-equivalent representative closest to an anchor."""

    lattice = _validate_lattice(lattice_vectors)
    anchor = wrap_fractional(anchor_frac)
    equivalents = symmetry_equivalent_positions(
        position_frac,
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions_frac,
        symprec=symprec,
    )
    candidates = near_minimum_indexes(
        minimum_image_distance_matrix([anchor], equivalents, lattice)[0]
    )
    # Ascending candidates keep ``min``'s first-wins tie-break over the full list.
    best = min(
        candidates.tolist(),
        key=lambda index: minimum_image_distance(equivalents[index], anchor, lattice),
    )
    return equivalents[best]


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
    symprec: float = 1e-5,
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
    symprec: float = 1e-5,
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
    low = 1e-3

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

    max_high = max(high * 8.0, 10.0)
    while expected_count(high) > target_count and high < max_high:
        high *= 1.5

    for _ in range(36):
        midpoint = 0.5 * (low + high)
        if expected_count(midpoint) > target_count:
            low = midpoint
        else:
            high = midpoint

    return max(0.1, high)


# Bounds the broadcast matrices (references x points) so large orbits against many
# accepted positions stay within a few megabytes.
_DISTANCE_CHECK_BLOCK = 65536


def _any_closer_than(
    points: np.ndarray,
    references: np.ndarray,
    lattice_vectors: Sequence[Sequence[float]],
    limit: float,
) -> bool:
    """Whether ``minimum_image_distance(point, reference) < limit`` for any pair, decided exactly as that helper would."""

    if len(points) == 0 or len(references) == 0:
        return False
    margin = abs(limit) * NEAR_TIE_RTOL
    block = max(1, _DISTANCE_CHECK_BLOCK // len(references))
    for start in range(0, len(points), block):
        chunk = points[start : start + block]
        distances = minimum_image_distance_matrix(references, chunk, lattice_vectors)
        if np.any(distances < limit - margin):
            return True
        for reference_index, point_index in np.argwhere(np.abs(distances - limit) <= margin):
            # ``tolist`` hands the scalar helper Python floats, as the generators always did.
            point = chunk[point_index].tolist()
            if (
                minimum_image_distance(point, references[reference_index].tolist(), lattice_vectors)
                < limit
            ):
                return True
    return False


def _identity_leads(operations: tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None) -> bool:
    """Whether the first operation is the identity, so every orbit starts with the position itself."""

    if operations is None:
        return False
    rotations, translations = operations
    if len(rotations) == 0 or len(translations) == 0:
        return False
    return bool(np.array_equal(rotations[0], np.eye(3)) and not np.any(translations[0]))


def _valid_candidate_orbit(
    candidate_frac: list[float],
    *,
    lattice_vectors: Sequence[Sequence[float]],
    operations: tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None,
    identity_leads: bool,
    host_positions: np.ndarray,
    accepted_positions: np.ndarray,
    min_muon_atom: float,
    min_muon_muon: float,
) -> list[list[float]] | None:
    """Return the candidate's orbit when every image satisfies both spacings, otherwise ``None``.

    ``candidate_frac`` must already lie in ``[0, 1)``.  "Muon-muon" spacing is between
    alternative one-muon starting positions, used to reduce duplicate relaxations to the
    same final site.
    """

    def violates(points: np.ndarray) -> bool:
        return _any_closer_than(
            points, host_positions, lattice_vectors, min_muon_atom
        ) or _any_closer_than(points, accepted_positions, lattice_vectors, min_muon_muon)

    # With the identity first, the orbit's first (always kept) image is exactly ``candidate_frac``,
    # so a candidate that fails on its own fails as an orbit: reject before building the orbit.
    if identity_leads and violates(np.asarray([candidate_frac], dtype=float)):
        return None
    orbit = symmetry_orbit(
        candidate_frac, operations, lattice_vectors=lattice_vectors, tolerance=_ORBIT_TOLERANCE_ANG
    )
    if violates(np.asarray(orbit, dtype=float).reshape(-1, 3)):
        return None
    return orbit


def _host_position_array(host_positions_frac: Sequence[Sequence[float]]) -> np.ndarray:
    hosts = np.asarray(host_positions_frac, dtype=float)
    if hosts.size == 0:
        return hosts.reshape(0, 3)
    if hosts.ndim != 2 or hosts.shape[1] != 3:
        raise SiteGenerationError("Host positions must be fractional (x, y, z) triples.")
    return hosts


def _normalized(vector: Sequence[float]) -> list[float] | None:
    norm = vector_length(vector)
    if norm <= 1e-12:
        return None
    return [float(value) / norm for value in vector]


def _host_clearance(
    candidate_frac: Sequence[float],
    host_positions_frac: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
) -> float:
    if not host_positions_frac:
        return math.inf
    return min(
        minimum_image_distance(candidate_frac, host, lattice_vectors)
        for host in host_positions_frac
    )


def generate_random_sites(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_positions_frac: Sequence[Sequence[float]],
    count: int,
    min_muon_muon: float,
    min_muon_atom: float,
    max_attempts: int = 20000,
    seed: int | None = None,
    allow_partial: bool = False,
    include_symmetry_equivalents: bool = False,
    host_species: Sequence[str] | None = None,
    symprec: float = 1e-5,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GeneratedPoint], int]:
    """Generate candidate sites by random filling with distance constraints.

    Returns a tuple: (sites, attempts_used).
    """
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
    identity_leads = _identity_leads(operations)
    accepted_positions = np.empty((0, 3), dtype=float)
    accepted_points: list[GeneratedPoint] = []
    if progress_callback is not None:
        progress_callback(0, count)

    attempts = 0
    while len(accepted_points) < count and attempts < max_attempts:
        attempts += 1
        candidate_frac = [rng.random(), rng.random(), rng.random()]
        candidate_orbit = _valid_candidate_orbit(
            candidate_frac,
            lattice_vectors=lattice,
            operations=operations,
            identity_leads=identity_leads,
            host_positions=host_positions,
            accepted_positions=accepted_positions,
            min_muon_atom=min_muon_atom,
            min_muon_muon=min_muon_muon,
        )
        if candidate_orbit is None:
            continue

        accepted_positions = np.concatenate(
            [accepted_positions, np.asarray(candidate_orbit, dtype=float)]
        )
        accepted_points.append(
            GeneratedPoint(
                fractional=wrap_fractional(candidate_frac),
                cartesian=lattice_frac_to_cart(candidate_frac, lattice),
            )
        )
        if progress_callback is not None:
            progress_callback(len(accepted_points), count)

    if len(accepted_points) < count:
        generated = len(accepted_points)
        if generated == 0:
            raise SiteGenerationError(
                "Could not generate any candidate sites with current constraints. "
                "Relax distance limits or increase supercell size."
            )
        if allow_partial:
            return accepted_points, attempts
        raise SiteGenerationError(
            f"Could only generate {generated} of {count} candidate sites after {attempts} attempts. "
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
    symprec: float = 1e-5,
    expected_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GeneratedPoint], int]:
    """Generate as many randomly placed candidate sites as constraints allow.

    Returns a tuple: (sites, attempts_used).
    """
    if min_muon_muon <= 0.0:
        raise SiteGenerationError("Muon-muon spacing must be greater than zero.")
    if min_muon_atom < 0.0:
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
    identity_leads = _identity_leads(operations)
    accepted_positions = np.empty((0, 3), dtype=float)
    accepted_points: list[GeneratedPoint] = []
    progress_total = max(1, int(expected_count or 1))
    if progress_callback is not None:
        progress_callback(0, progress_total)

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        candidate_frac = [rng.random(), rng.random(), rng.random()]
        candidate_orbit = _valid_candidate_orbit(
            candidate_frac,
            lattice_vectors=lattice,
            operations=operations,
            identity_leads=identity_leads,
            host_positions=host_positions,
            accepted_positions=accepted_positions,
            min_muon_atom=min_muon_atom,
            min_muon_muon=min_muon_muon,
        )
        if candidate_orbit is None:
            continue

        accepted_positions = np.concatenate(
            [accepted_positions, np.asarray(candidate_orbit, dtype=float)]
        )
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

    return accepted_points, attempts


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


def _host_clearance_excluding_index(
    candidate_frac: Sequence[float],
    host_positions_frac: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
    excluded_index: int,
) -> float:
    clearances = [
        minimum_image_distance(candidate_frac, host, lattice_vectors)
        for index, host in enumerate(host_positions_frac)
        if index != excluded_index
    ]
    return min(clearances) if clearances else math.inf


def generate_attached_element_sites(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_species: Sequence[str],
    host_positions_frac: Sequence[Sequence[float]],
    target_element: str,
    muons_per_atom: int = 1,
    bond_distance: float = 1.0,
    min_muon_atom: float = 0.8,
    neighbor_cutoff: float = 2.6,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[GeneratedPoint], int]:
    """Generate deterministic element-attached starts using local bonding geometry.

    For every atom matching ``target_element``, candidate directions point away
    from nearby neighbours and a deterministic spherical fallback set. The best
    ``muons_per_atom`` candidates for each atom are selected by clearance from
    the surrounding host structure.
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
    target = target_element.strip().lower()
    target_indexes = [
        idx for idx, species in enumerate(host_species) if species.strip().lower() == target
    ]
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
            delta = minimum_image_cartesian_delta(other_frac, anchor_frac, lattice)
            distance = vector_length(delta)
            if 1e-8 < distance <= neighbor_cutoff:
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
                sum(normalized[i] * existing[i] for i in range(3)) > 0.999
                for existing in unique_dirs
            ):
                continue
            unique_dirs.append(normalized)

        per_atom: list[tuple[float, list[float]]] = []
        for direction in unique_dirs:
            candidate_cart = [
                anchor_cart[axis] + direction[axis] * bond_distance for axis in range(3)
            ]
            candidate_frac = wrap_fractional(lattice_cart_to_frac(candidate_cart, lattice))
            attempts += 1
            if _host_clearance(candidate_frac, host_positions_frac, lattice) < min_muon_atom - 1e-8:
                continue
            clearance = _host_clearance_excluding_index(
                candidate_frac,
                host_positions_frac,
                lattice,
                target_index,
            )
            per_atom.append((clearance, candidate_frac))
        per_atom.sort(key=lambda item: item[0], reverse=True)
        selected_for_atom: list[list[float]] = []
        for _clearance, candidate_frac in per_atom:
            if any(
                minimum_image_distance(candidate_frac, accepted, lattice) < 1e-5
                for accepted in selected_for_atom
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

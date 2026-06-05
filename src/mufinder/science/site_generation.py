"""Muon candidate-site generation with periodic boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections.abc import Callable
from typing import Sequence

from ase.data import atomic_numbers
import numpy as np

from mufinder.science.spglib_compat import get_crystal_symmetry


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
    if len(lattice_vectors) != 3 or any(len(v) != 3 for v in lattice_vectors):
        raise SiteGenerationError("Lattice vectors must be a 3x3 matrix.")
    return [[float(c) for c in vec] for vec in lattice_vectors]


def _identity_lattice() -> list[list[float]]:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _wrap_fractional(position: Sequence[float]) -> list[float]:
    return [float(v) % 1.0 for v in position]


def _species_to_numbers(host_species: Sequence[str]) -> list[int] | None:
    numbers: list[int] = []
    for symbol in host_species:
        number = atomic_numbers.get(str(symbol).strip().capitalize())
        if number is None:
            return None
        numbers.append(int(number))
    return numbers


def lattice_frac_to_cart(fractional: Sequence[float], lattice_vectors: Sequence[Sequence[float]]) -> list[float]:
    """Convert fractional coordinate to Cartesian coordinate."""
    lattice = _validate_lattice(lattice_vectors)
    x = fractional[0] * lattice[0][0] + fractional[1] * lattice[1][0] + fractional[2] * lattice[2][0]
    y = fractional[0] * lattice[0][1] + fractional[1] * lattice[1][1] + fractional[2] * lattice[2][1]
    z = fractional[0] * lattice[0][2] + fractional[1] * lattice[1][2] + fractional[2] * lattice[2][2]
    return [x, y, z]


def _determinant3(m: Sequence[Sequence[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _cell_volume(lattice_vectors: Sequence[Sequence[float]]) -> float:
    return abs(_determinant3(lattice_vectors))


def _sphere_volume(radius: float) -> float:
    return 4.0 * math.pi * float(radius) ** 3 / 3.0


def _inverse3(m: Sequence[Sequence[float]]) -> list[list[float]]:
    det = _determinant3(m)
    if math.isclose(det, 0.0, abs_tol=1e-12):
        raise SiteGenerationError("Lattice matrix is singular and cannot be inverted.")

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


def lattice_cart_to_frac(cartesian: Sequence[float], lattice_vectors: Sequence[Sequence[float]]) -> list[float]:
    """Convert Cartesian coordinate to fractional coordinate."""
    lattice = _validate_lattice(lattice_vectors)
    inv = _inverse3(lattice)
    x = inv[0][0] * cartesian[0] + inv[0][1] * cartesian[1] + inv[0][2] * cartesian[2]
    y = inv[1][0] * cartesian[0] + inv[1][1] * cartesian[1] + inv[1][2] * cartesian[2]
    z = inv[2][0] * cartesian[0] + inv[2][1] * cartesian[1] + inv[2][2] * cartesian[2]
    return [x, y, z]


def _minimum_image_delta(a: Sequence[float], b: Sequence[float]) -> list[float]:
    delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    return [d - round(d) for d in delta]


def _pbc_distance(
    a_frac: Sequence[float],
    b_frac: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
) -> float:
    delta_frac = _minimum_image_delta(a_frac, b_frac)
    delta_cart = lattice_frac_to_cart(delta_frac, lattice_vectors)
    return math.sqrt(delta_cart[0] ** 2 + delta_cart[1] ** 2 + delta_cart[2] ** 2)


def _deduplicate_fractional_positions(
    positions: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
    *,
    tolerance_ang: float = 1e-5,
) -> list[list[float]]:
    unique: list[list[float]] = []
    for position in positions:
        wrapped = _wrap_fractional(position)
        if any(_pbc_distance(wrapped, existing, lattice_vectors) <= tolerance_ang for existing in unique):
            continue
        unique.append(wrapped)
    return unique


def _symmetry_operations(
    *,
    lattice_vectors: Sequence[Sequence[float]],
    host_species: Sequence[str],
    host_positions_frac: Sequence[Sequence[float]],
    symprec: float = 1e-5,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    lattice = _validate_lattice(lattice_vectors)
    numbers = _species_to_numbers(host_species)
    if numbers is None or len(numbers) != len(host_positions_frac):
        return None
    symmetry = get_crystal_symmetry(
        (
            np.asarray(lattice, dtype=float),
            np.asarray(host_positions_frac, dtype=float),
            np.asarray(numbers, dtype=int),
        ),
        symprec=float(symprec),
    )
    if symmetry is None:
        return None
    return (
        [np.asarray(rotation, dtype=float) for rotation in symmetry.get("rotations", [])],
        [np.asarray(translation, dtype=float) for translation in symmetry.get("translations", [])],
    )


def _equivalent_positions_from_operations(
    position_frac: Sequence[float],
    *,
    lattice_vectors: Sequence[Sequence[float]],
    operations: tuple[list[np.ndarray], list[np.ndarray]] | None,
) -> list[list[float]]:
    if operations is None:
        return [_wrap_fractional(position_frac)]
    rotations, translations = operations
    position = np.asarray(position_frac, dtype=float)
    equivalents = [
        _wrap_fractional(np.dot(rotation, position) + translation)
        for rotation, translation in zip(rotations, translations)
    ]
    return _deduplicate_fractional_positions(equivalents, lattice_vectors)


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
    return _equivalent_positions_from_operations(position_frac, lattice_vectors=lattice, operations=operations)


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
    anchor = _wrap_fractional(anchor_frac)
    equivalents = symmetry_equivalent_positions(
        position_frac,
        lattice_vectors=lattice,
        host_species=host_species,
        host_positions_frac=host_positions_frac,
        symprec=symprec,
    )
    return min(equivalents, key=lambda equivalent: _pbc_distance(equivalent, anchor, lattice))


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
    volume = _cell_volume(lattice)
    if exclusion_radius <= 0.0 or not host_positions_frac:
        return volume
    if volume <= 0.0:
        raise SiteGenerationError("Cell volume must be positive.")

    resolution = max(4, int(grid_resolution))
    axis = (np.arange(resolution, dtype=float) + 0.5) / float(resolution)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    hosts = np.asarray([_wrap_fractional(position) for position in host_positions_frac], dtype=float)
    lattice_array = np.asarray(lattice, dtype=float)
    delta_frac = grid[:, None, :] - hosts[None, :, :]
    delta_frac -= np.rint(delta_frac)
    delta_cart = delta_frac @ lattice_array
    distances = np.linalg.norm(delta_cart, axis=2)
    free_fraction = float(np.count_nonzero(np.min(distances, axis=1) >= float(exclusion_radius))) / float(len(grid))
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
    vector_lengths = [math.sqrt(sum(component * component for component in vector)) for vector in lattice]
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


def _is_valid_candidate(
    candidate_frac: Sequence[float],
    host_positions_frac: Sequence[Sequence[float]],
    existing_muon_positions_frac: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
    min_muon_atom: float,
    min_muon_muon: float,
) -> bool:
    # "Muon-muon" spacing here is between alternative one-muon starting
    # positions, used to reduce duplicate relaxations to the same final site.
    for host_frac in host_positions_frac:
        if _pbc_distance(candidate_frac, host_frac, lattice_vectors) < min_muon_atom:
            return False

    for muon_frac in existing_muon_positions_frac:
        if _pbc_distance(candidate_frac, muon_frac, lattice_vectors) < min_muon_muon:
            return False

    return True


def _is_valid_candidate_orbit(
    candidate_orbit_frac: Sequence[Sequence[float]],
    host_positions_frac: Sequence[Sequence[float]],
    existing_muon_positions_frac: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
    min_muon_atom: float,
    min_muon_muon: float,
) -> bool:
    for candidate_frac in candidate_orbit_frac:
        if not _is_valid_candidate(
            candidate_frac,
            host_positions_frac,
            existing_muon_positions_frac,
            lattice_vectors,
            min_muon_atom=min_muon_atom,
            min_muon_muon=min_muon_muon,
        ):
            return False
    return True


def _sample_unit_sphere_direction(rng: random.Random) -> list[float]:
    while True:
        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)
        z = rng.uniform(-1.0, 1.0)
        norm2 = x * x + y * y + z * z
        if 1e-12 < norm2 <= 1.0:
            inv = 1.0 / math.sqrt(norm2)
            return [x * inv, y * inv, z * inv]


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def _normalized(vector: Sequence[float]) -> list[float] | None:
    norm = _norm(vector)
    if norm <= 1e-12:
        return None
    return [float(value) / norm for value in vector]


def _minimum_image_cartesian_delta(
    a_frac: Sequence[float],
    b_frac: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
) -> list[float]:
    return lattice_frac_to_cart(_minimum_image_delta(a_frac, b_frac), lattice_vectors)


def _host_clearance(
    candidate_frac: Sequence[float],
    host_positions_frac: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
) -> float:
    if not host_positions_frac:
        return math.inf
    return min(_pbc_distance(candidate_frac, host, lattice_vectors) for host in host_positions_frac)


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

    accepted_frac: list[list[float]] = []
    accepted_points: list[GeneratedPoint] = []
    if progress_callback is not None:
        progress_callback(0, count)

    attempts = 0
    while len(accepted_points) < count and attempts < max_attempts:
        attempts += 1
        candidate_frac = [rng.random(), rng.random(), rng.random()]
        candidate_orbit = (
            _equivalent_positions_from_operations(
                candidate_frac,
                lattice_vectors=lattice,
                operations=operations,
            )
            if operations is not None
            else [_wrap_fractional(candidate_frac)]
        )
        if not _is_valid_candidate_orbit(
            candidate_orbit,
            host_positions_frac,
            accepted_frac,
            lattice,
            min_muon_atom=min_muon_atom,
            min_muon_muon=min_muon_muon,
        ):
            continue

        accepted_frac.extend(candidate_orbit)
        accepted_points.append(
            GeneratedPoint(
                fractional=_wrap_fractional(candidate_frac),
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

    accepted_frac: list[list[float]] = []
    accepted_points: list[GeneratedPoint] = []
    progress_total = max(1, int(expected_count or 1))
    if progress_callback is not None:
        progress_callback(0, progress_total)

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        candidate_frac = [rng.random(), rng.random(), rng.random()]
        candidate_orbit = (
            _equivalent_positions_from_operations(
                candidate_frac,
                lattice_vectors=lattice,
                operations=operations,
            )
            if operations is not None
            else [_wrap_fractional(candidate_frac)]
        )
        if not _is_valid_candidate_orbit(
            candidate_orbit,
            host_positions_frac,
            accepted_frac,
            lattice,
            min_muon_atom=min_muon_atom,
            min_muon_muon=min_muon_muon,
        ):
            continue

        accepted_frac.extend(candidate_orbit)
        accepted_points.append(
            GeneratedPoint(
                fractional=_wrap_fractional(candidate_frac),
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
        _pbc_distance(candidate_frac, host, lattice_vectors)
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
        idx
        for idx, species in enumerate(host_species)
        if species.strip().lower() == target
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
            delta = _minimum_image_cartesian_delta(other_frac, anchor_frac, lattice)
            distance = _norm(delta)
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
            if any(sum(normalized[i] * existing[i] for i in range(3)) > 0.999 for existing in unique_dirs):
                continue
            unique_dirs.append(normalized)

        per_atom: list[tuple[float, list[float]]] = []
        for direction in unique_dirs:
            candidate_cart = [anchor_cart[axis] + direction[axis] * bond_distance for axis in range(3)]
            candidate_frac = _wrap_fractional(lattice_cart_to_frac(candidate_cart, lattice))
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
            if any(_pbc_distance(candidate_frac, accepted, lattice) < 1e-5 for accepted in selected_for_atom):
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

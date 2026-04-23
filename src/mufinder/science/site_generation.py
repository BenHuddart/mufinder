"""Muon candidate-site generation with periodic boundary conditions."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence


class SiteGenerationError(ValueError):
    """Raised when candidate generation cannot satisfy constraints."""


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


def _is_valid_candidate(
    candidate_frac: Sequence[float],
    host_positions_frac: Sequence[Sequence[float]],
    existing_muon_positions_frac: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
    min_muon_atom: float,
    min_muon_muon: float,
) -> bool:
    for host_frac in host_positions_frac:
        if _pbc_distance(candidate_frac, host_frac, lattice_vectors) < min_muon_atom:
            return False

    for muon_frac in existing_muon_positions_frac:
        if _pbc_distance(candidate_frac, muon_frac, lattice_vectors) < min_muon_muon:
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

    accepted_frac: list[list[float]] = []
    accepted_points: list[GeneratedPoint] = []

    attempts = 0
    while len(accepted_points) < count and attempts < max_attempts:
        attempts += 1
        candidate_frac = [rng.random(), rng.random(), rng.random()]
        if not _is_valid_candidate(
            candidate_frac,
            host_positions_frac,
            accepted_frac,
            lattice,
            min_muon_atom=min_muon_atom,
            min_muon_muon=min_muon_muon,
        ):
            continue

        accepted_frac.append(candidate_frac)
        accepted_points.append(
            GeneratedPoint(
                fractional=_wrap_fractional(candidate_frac),
                cartesian=lattice_frac_to_cart(candidate_frac, lattice),
            )
        )

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

    accepted_frac: list[list[float]] = []
    accepted_points: list[GeneratedPoint] = []

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        candidate_frac = [rng.random(), rng.random(), rng.random()]
        if not _is_valid_candidate(
            candidate_frac,
            host_positions_frac,
            accepted_frac,
            lattice,
            min_muon_atom=min_muon_atom,
            min_muon_muon=min_muon_muon,
        ):
            continue

        accepted_frac.append(candidate_frac)
        accepted_points.append(
            GeneratedPoint(
                fractional=_wrap_fractional(candidate_frac),
                cartesian=lattice_frac_to_cart(candidate_frac, lattice),
            )
        )

    if not accepted_points:
        raise SiteGenerationError(
            "Could not generate any candidate sites with current constraints. "
            "Relax distance limits or increase supercell size."
        )

    return accepted_points, attempts


def generate_near_element_sites(
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    host_species: Sequence[str],
    host_positions_frac: Sequence[Sequence[float]],
    target_element: str,
    count: int,
    shell_min: float,
    shell_max: float,
    min_muon_muon: float,
    min_muon_atom: float,
    max_attempts: int = 30000,
    seed: int | None = None,
) -> tuple[list[GeneratedPoint], int]:
    """Generate candidate sites near selected host atoms under constraints.

    shell_min and shell_max are radial offsets in Angstrom around target atoms.
    Returns a tuple: (sites, attempts_used).
    """
    if not target_element.strip():
        raise SiteGenerationError("Target element cannot be empty.")
    if count <= 0:
        raise SiteGenerationError("Count must be greater than zero.")
    if shell_min < 0.0 or shell_max <= 0.0 or shell_max <= shell_min:
        raise SiteGenerationError("Require 0 <= shell_min < shell_max for near-element generation.")

    lattice = _validate_lattice(lattice_vectors) if lattice_vectors else _identity_lattice()
    rng = random.Random(seed)
    target = target_element.strip().lower()

    target_positions = [
        host_positions_frac[idx]
        for idx, species in enumerate(host_species)
        if species.strip().lower() == target
    ]
    if not target_positions:
        raise SiteGenerationError(f"No host atoms found for element '{target_element}'.")

    accepted_frac: list[list[float]] = []
    accepted_points: list[GeneratedPoint] = []

    attempts = 0
    while len(accepted_points) < count and attempts < max_attempts:
        attempts += 1

        anchor_frac = target_positions[rng.randrange(len(target_positions))]
        anchor_cart = lattice_frac_to_cart(anchor_frac, lattice)
        direction = _sample_unit_sphere_direction(rng)
        radius = rng.uniform(shell_min, shell_max)
        candidate_cart = [
            anchor_cart[0] + direction[0] * radius,
            anchor_cart[1] + direction[1] * radius,
            anchor_cart[2] + direction[2] * radius,
        ]

        candidate_frac = _wrap_fractional(lattice_cart_to_frac(candidate_cart, lattice))
        if not _is_valid_candidate(
            candidate_frac,
            host_positions_frac,
            accepted_frac,
            lattice,
            min_muon_atom=min_muon_atom,
            min_muon_muon=min_muon_muon,
        ):
            continue

        accepted_frac.append(candidate_frac)
        accepted_points.append(
            GeneratedPoint(
                fractional=candidate_frac,
                cartesian=lattice_frac_to_cart(candidate_frac, lattice),
            )
        )

    if len(accepted_points) < count:
        generated = len(accepted_points)
        if generated == 0:
            raise SiteGenerationError(
                "Could not place any near-element candidate sites with current constraints."
            )
        raise SiteGenerationError(
            f"Could only generate {generated} of {count} near-element candidate sites after {attempts} attempts. "
            "Increase max_attempts, relax distance limits, or increase supercell size."
        )

    return accepted_points, attempts

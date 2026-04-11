"""Bond, angle, and coordination analysis for crystal structures."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import AtomMatching


@dataclass(frozen=True, slots=True)
class BondLengthComparison:
    """Summary of nearest-neighbour bond-length differences."""

    mean_difference: float | None
    max_difference: float | None


@dataclass(frozen=True, slots=True)
class BondAngleComparison:
    """Summary of nearest-neighbour bond-angle differences."""

    mean_difference: float | None


@dataclass(frozen=True, slots=True)
class CoordinationComparison:
    """Summary of coordination-number changes across matched sites."""

    changed_sites: int
    preserved_fraction: float | None


def _minimum_image_vector(
    lhs_fractional: list[float],
    rhs_fractional: list[float],
    lattice_vectors: list[list[float]],
) -> np.ndarray:
    delta_fractional = np.asarray(rhs_fractional, dtype=float) - np.asarray(lhs_fractional, dtype=float)
    delta_fractional -= np.round(delta_fractional)
    lattice = np.asarray(lattice_vectors, dtype=float)
    return delta_fractional @ lattice


def _distance(
    lhs_fractional: list[float],
    rhs_fractional: list[float],
    lattice_vectors: list[list[float]],
) -> float:
    return float(np.linalg.norm(_minimum_image_vector(lhs_fractional, rhs_fractional, lattice_vectors)))


def _radius_for_species(species: str) -> float:
    number = atomic_numbers.get(str(species).capitalize())
    if number is None:
        return 1.0
    radius = covalent_radii[number]
    return float(radius if radius > 0.0 else 1.0)


def nearest_neighbour_distances(structure: Structure) -> list[float | None]:
    """Return nearest-neighbour distance for each atom in a periodic structure."""

    distances: list[float | None] = [None] * len(structure.species)
    for lhs_index, lhs_position in enumerate(structure.fractional_positions):
        best: float | None = None
        for rhs_index, rhs_position in enumerate(structure.fractional_positions):
            if lhs_index == rhs_index:
                continue
            distance = _distance(lhs_position, rhs_position, structure.lattice_vectors)
            if distance <= 1e-12:
                continue
            if best is None or distance < best:
                best = distance
        distances[lhs_index] = best
    return distances


def coordination_numbers(structure: Structure, *, cutoff_factor: float = 1.3) -> list[int]:
    """Estimate coordination number using radius-based neighbour cutoffs."""

    counts = [0] * len(structure.species)
    for lhs_index, lhs_position in enumerate(structure.fractional_positions):
        lhs_radius = _radius_for_species(structure.species[lhs_index])
        for rhs_index, rhs_position in enumerate(structure.fractional_positions):
            if lhs_index == rhs_index:
                continue
            rhs_radius = _radius_for_species(structure.species[rhs_index])
            threshold = cutoff_factor * (lhs_radius + rhs_radius)
            if _distance(lhs_position, rhs_position, structure.lattice_vectors) <= threshold:
                counts[lhs_index] += 1
    return counts


def mean_bond_angles(structure: Structure, *, cutoff_factor: float = 1.3) -> list[float | None]:
    """Compute mean neighbour-neighbour angle around each site."""

    means: list[float | None] = [None] * len(structure.species)
    for center_index, center in enumerate(structure.fractional_positions):
        center_radius = _radius_for_species(structure.species[center_index])
        neighbour_vectors: list[np.ndarray] = []
        for neighbour_index, neighbour in enumerate(structure.fractional_positions):
            if center_index == neighbour_index:
                continue
            neighbour_radius = _radius_for_species(structure.species[neighbour_index])
            threshold = cutoff_factor * (center_radius + neighbour_radius)
            vector = _minimum_image_vector(center, neighbour, structure.lattice_vectors)
            distance = float(np.linalg.norm(vector))
            if 1e-12 < distance <= threshold:
                neighbour_vectors.append(vector)

        if len(neighbour_vectors) < 2:
            means[center_index] = None
            continue

        angles: list[float] = []
        for left_index in range(len(neighbour_vectors)):
            for right_index in range(left_index + 1, len(neighbour_vectors)):
                left = neighbour_vectors[left_index]
                right = neighbour_vectors[right_index]
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                if denominator <= 1e-12:
                    continue
                cosine = float(np.dot(left, right) / denominator)
                cosine = max(-1.0, min(1.0, cosine))
                angles.append(degrees(acos(cosine)))
        means[center_index] = float(np.mean(angles)) if angles else None
    return means


def compare_nearest_neighbour_bond_lengths(
    experimental: Structure,
    dft: Structure,
    matching: AtomMatching,
) -> BondLengthComparison:
    """Compare nearest-neighbour bond lengths on matched atoms."""

    experimental_nn = nearest_neighbour_distances(experimental)
    dft_nn = nearest_neighbour_distances(dft)

    deltas: list[float] = []
    for pair in matching.pairs:
        lhs = experimental_nn[pair.experimental_index]
        rhs = dft_nn[pair.dft_index]
        if lhs is None or rhs is None:
            continue
        deltas.append(abs(float(rhs) - float(lhs)))

    if not deltas:
        return BondLengthComparison(mean_difference=None, max_difference=None)
    return BondLengthComparison(
        mean_difference=float(np.mean(deltas)),
        max_difference=float(np.max(deltas)),
    )


def compare_bond_angles(
    experimental: Structure,
    dft: Structure,
    matching: AtomMatching,
) -> BondAngleComparison:
    """Compare mean local bond-angle environments on matched atoms."""

    experimental_angles = mean_bond_angles(experimental)
    dft_angles = mean_bond_angles(dft)

    deltas: list[float] = []
    for pair in matching.pairs:
        lhs = experimental_angles[pair.experimental_index]
        rhs = dft_angles[pair.dft_index]
        if lhs is None or rhs is None:
            continue
        deltas.append(abs(float(rhs) - float(lhs)))

    if not deltas:
        return BondAngleComparison(mean_difference=None)
    return BondAngleComparison(mean_difference=float(np.mean(deltas)))


def compare_coordination_numbers(
    experimental: Structure,
    dft: Structure,
    matching: AtomMatching,
) -> CoordinationComparison:
    """Compare coordination numbers on matched atoms."""

    experimental_coordination = coordination_numbers(experimental)
    dft_coordination = coordination_numbers(dft)

    if not matching.pairs:
        return CoordinationComparison(changed_sites=0, preserved_fraction=None)

    changed_sites = 0
    for pair in matching.pairs:
        if experimental_coordination[pair.experimental_index] != dft_coordination[pair.dft_index]:
            changed_sites += 1

    preserved_fraction = float((len(matching.pairs) - changed_sites) / len(matching.pairs))
    return CoordinationComparison(changed_sites=changed_sites, preserved_fraction=preserved_fraction)


def connectivity_edges(structure: Structure, *, cutoff_factor: float = 1.3) -> set[tuple[int, int]]:
    """Build an undirected bonding graph from radius-based neighbour cutoffs."""

    edges: set[tuple[int, int]] = set()
    for lhs_index, lhs_position in enumerate(structure.fractional_positions):
        lhs_radius = _radius_for_species(structure.species[lhs_index])
        for rhs_index in range(lhs_index + 1, len(structure.fractional_positions)):
            rhs_position = structure.fractional_positions[rhs_index]
            rhs_radius = _radius_for_species(structure.species[rhs_index])
            threshold = cutoff_factor * (lhs_radius + rhs_radius)
            if _distance(lhs_position, rhs_position, structure.lattice_vectors) <= threshold:
                edges.add((lhs_index, rhs_index))
    return edges

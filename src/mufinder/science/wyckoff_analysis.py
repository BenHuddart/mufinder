"""Wyckoff-site assignment helpers for relaxed muon clusters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from ase.data import atomic_numbers
import numpy as np
import spglib

from mufinder.domain.models import Structure
from mufinder.science.spglib_compat import get_crystal_symmetry, get_crystal_symmetry_dataset
from mufinder.science.relaxation_analysis import (
    RelaxationClusterSummary,
    RelaxedMuonSite,
    _fractional_delta_minimum_image,
    _pbc_distance,
    _wrap_fractional,
    anchored_sites,
    normalize_site_energies,
)
from mufinder.science.site_generation import lattice_frac_to_cart


@dataclass(frozen=True, slots=True)
class WyckoffSiteMatch:
    """Best Wyckoff-site match for one relaxed-muon cluster."""

    matched: bool
    label: str
    fractional_position: tuple[float, float, float] | None
    cartesian_position_ang: tuple[float, float, float] | None
    multiplicity: int | None
    letter: str | None
    space_group_number: int | None
    space_group_symbol: str | None
    site_symmetry_symbol: str | None
    max_distance_ang: float | None
    mean_distance_ang: float | None
    occupied: bool
    special: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClusterWyckoffAssignment:
    """Wyckoff assignment for one structural-relaxation cluster."""

    cluster_index: int
    match: WyckoffSiteMatch


@dataclass(frozen=True, slots=True)
class _CandidateLabel:
    label: str
    multiplicity: int
    letter: str
    site_symmetry_symbol: str | None
    space_group_number: int
    space_group_symbol: str


def _species_to_numbers(species: Sequence[str]) -> list[int] | None:
    numbers: list[int] = []
    for symbol in species:
        key = str(symbol).strip().capitalize()
        number = atomic_numbers.get(key)
        if number is None:
            return None
        numbers.append(int(number))
    return numbers


def _host_cell(structure: Structure) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    numbers = _species_to_numbers(structure.species)
    if numbers is None:
        return None
    if len(numbers) != len(structure.fractional_positions):
        return None
    return (
        np.asarray(structure.lattice_vectors, dtype=float),
        np.asarray(structure.fractional_positions, dtype=float),
        np.asarray(numbers, dtype=int),
    )


def _operation_leaves_position_fixed(
    rotation: np.ndarray,
    translation: np.ndarray,
    position: Sequence[float],
    lattice: Sequence[Sequence[float]],
    *,
    tolerance_ang: float,
) -> bool:
    transformed = _wrap_fractional(np.dot(rotation, np.asarray(position, dtype=float)) + translation)
    return _pbc_distance(transformed, position, lattice) <= float(tolerance_ang)


def _deduplicate_positions(
    positions: Sequence[Sequence[float]],
    lattice: Sequence[Sequence[float]],
    *,
    tolerance_ang: float,
) -> tuple[tuple[float, float, float], ...]:
    unique: list[tuple[float, float, float]] = []
    for position in positions:
        wrapped = _wrap_fractional(position)
        if any(_pbc_distance(wrapped, existing, lattice) <= tolerance_ang for existing in unique):
            continue
        unique.append(wrapped)
    return tuple(unique)


def _symmetry_orbit(
    position: Sequence[float],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    lattice: Sequence[Sequence[float]],
    *,
    tolerance_ang: float,
) -> tuple[tuple[float, float, float], ...]:
    point = np.asarray(position, dtype=float)
    positions = [
        _wrap_fractional(np.dot(rotation, point) + translation)
        for rotation, translation in zip(rotations, translations, strict=False)
    ]
    return _deduplicate_positions(positions, lattice, tolerance_ang=tolerance_ang)


def _dummy_atomic_number(host_numbers: Sequence[int]) -> int:
    used = {int(number) for number in host_numbers}
    for number in range(118, 0, -1):
        if number not in used:
            return number
    return 0


def _label_candidate(
    host_cell: tuple[np.ndarray, np.ndarray, np.ndarray],
    orbit: Sequence[Sequence[float]],
    *,
    symprec: float,
) -> _CandidateLabel | None:
    lattice, host_positions, host_numbers = host_cell
    dummy_number = _dummy_atomic_number(host_numbers)
    positions = np.vstack([host_positions, np.asarray(orbit, dtype=float)])
    numbers = np.asarray([*host_numbers.tolist(), *([dummy_number] * len(orbit))], dtype=int)
    dataset = get_crystal_symmetry_dataset(
        (lattice, positions, numbers),
        symprec=float(symprec),
    )
    if dataset is None:
        return None
    offset = len(host_positions)
    letters = [str(value) for value in dataset.wyckoffs[offset:]]
    if not letters:
        return None
    letter = letters[0]
    site_symbols = [str(value) for value in dataset.site_symmetry_symbols[offset:]]
    site_symbol = site_symbols[0] if site_symbols else None
    return _CandidateLabel(
        label=f"{len(orbit)}{letter}",
        multiplicity=len(orbit),
        letter=letter,
        site_symmetry_symbol=site_symbol,
        space_group_number=int(dataset.number),
        space_group_symbol=str(dataset.international),
    )


def _orbit_is_occupied(
    orbit: Sequence[Sequence[float]],
    host_structure: Structure,
    *,
    tolerance_ang: float,
) -> bool:
    for candidate in orbit:
        if any(
            _pbc_distance(candidate, host_position, host_structure.lattice_vectors) <= float(tolerance_ang)
            for host_position in host_structure.fractional_positions
        ):
            return True
    return False


def _project_to_constraints(
    reference: Sequence[float],
    constraints: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float, float] | None:
    point = np.asarray(reference, dtype=float)
    rows: list[np.ndarray] = []
    values: list[float] = []
    for rotation, translation in constraints:
        n = np.round(np.dot(rotation, point) + translation - point)
        matrix = rotation - np.eye(3)
        rhs = n - translation
        for row, value in zip(matrix, rhs, strict=True):
            if np.linalg.norm(row) <= 1e-12:
                continue
            rows.append(np.asarray(row, dtype=float))
            values.append(float(value))
    if not rows:
        return None
    a = np.vstack(rows)
    b = np.asarray(values, dtype=float)
    correction = a.T @ np.linalg.pinv(a @ a.T) @ (a @ point - b)
    projected = point - correction
    residual = np.linalg.norm(a @ projected - b)
    if not np.isfinite(residual) or residual > 1e-5:
        return None
    return _wrap_fractional(projected)


def _project_nearby_special_candidates(
    reference: Sequence[float],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    lattice: Sequence[Sequence[float]],
    *,
    tolerance_ang: float,
) -> tuple[tuple[float, float, float], ...]:
    operations: list[tuple[np.ndarray, np.ndarray]] = []
    identity = np.eye(3)
    for rotation, translation in zip(rotations, translations, strict=False):
        is_identity = np.allclose(rotation, identity) and np.allclose(translation - np.round(translation), 0.0)
        if not is_identity:
            operations.append((rotation, translation))

    candidates: list[tuple[float, float, float]] = []
    for operation in operations:
        projected = _project_to_constraints(reference, (operation,))
        if projected is not None:
            candidates.append(projected)

    for left, right in combinations(operations, 2):
        projected = _project_to_constraints(reference, (left, right))
        if projected is not None:
            candidates.append(projected)

    nearby = [
        candidate
        for candidate in candidates
        if _pbc_distance(candidate, reference, lattice) <= max(float(tolerance_ang) * 2.0, float(tolerance_ang) + 0.1)
    ]
    return _deduplicate_positions(nearby, lattice, tolerance_ang=1e-4)


def _cluster_reference(
    positions: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    if not positions:
        return (0.0, 0.0, 0.0)
    anchor = np.asarray(_wrap_fractional(positions[0]), dtype=float)
    deltas = [
        _fractional_delta_minimum_image(anchor, _wrap_fractional(position))
        for position in positions
    ]
    return _wrap_fractional(anchor + np.mean(np.asarray(deltas, dtype=float), axis=0))


def _cartesian_position(
    fractional: Sequence[float],
    lattice: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    return tuple(float(value) for value in lattice_frac_to_cart(fractional, lattice))


def _unavailable_match(message: str) -> WyckoffSiteMatch:
    return WyckoffSiteMatch(
        matched=False,
        label="",
        fractional_position=None,
        cartesian_position_ang=None,
        multiplicity=None,
        letter=None,
        space_group_number=None,
        space_group_symbol=None,
        site_symmetry_symbol=None,
        max_distance_ang=None,
        mean_distance_ang=None,
        occupied=False,
        special=False,
        warnings=(message,),
    )


def _match_distances_to_orbit(
    member_positions: Sequence[Sequence[float]],
    orbit: Sequence[Sequence[float]],
    lattice: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    distances: list[float] = []
    for member in member_positions:
        distances.append(min(_pbc_distance(member, candidate, lattice) for candidate in orbit))
    return tuple(distances)


def assign_cluster_wyckoff_sites(
    sites: Sequence[RelaxedMuonSite],
    clusters: Sequence[RelaxationClusterSummary],
    host_structure: Structure,
    *,
    match_tolerance_ang: float = 0.25,
    occupancy_tolerance_ang: float = 0.05,
    symprec: float = 1e-5,
) -> tuple[ClusterWyckoffAssignment, ...]:
    """Assign the highest-symmetry unoccupied Wyckoff site consistent with each cluster."""

    if match_tolerance_ang < 0.0:
        raise ValueError("Wyckoff match tolerance must be non-negative.")
    if occupancy_tolerance_ang < 0.0:
        raise ValueError("Wyckoff occupancy tolerance must be non-negative.")

    ordered_sites = normalize_site_energies(sites)
    labels = [0] * len(ordered_sites)
    for cluster in clusters:
        for member_index in cluster.member_indexes:
            if 0 <= int(member_index) < len(labels):
                labels[int(member_index)] = int(cluster.cluster_index)

    try:
        anchored = anchored_sites(ordered_sites, host_structure, labels)
    except Exception:  # noqa: BLE001 - fall back to unshifted display positions.
        anchored = tuple()

    host_cell = _host_cell(host_structure)
    if host_cell is None:
        return tuple(
            ClusterWyckoffAssignment(
                cluster_index=int(cluster.cluster_index),
                match=_unavailable_match("Host symmetry is unavailable because one or more species are not recognised."),
            )
            for cluster in clusters
        )

    try:
        symmetry = get_crystal_symmetry(host_cell, symprec=float(symprec))
        dataset = get_crystal_symmetry_dataset(host_cell, symprec=float(symprec))
    except Exception:  # noqa: BLE001 - spglib can fail on malformed cells.
        symmetry = None
        dataset = None
    if symmetry is None or dataset is None:
        return tuple(
            ClusterWyckoffAssignment(
                cluster_index=int(cluster.cluster_index),
                match=_unavailable_match("Host symmetry could not be identified by spglib."),
            )
            for cluster in clusters
        )

    rotations = [np.asarray(rotation, dtype=float) for rotation in symmetry.get("rotations", [])]
    translations = [np.asarray(translation, dtype=float) for translation in symmetry.get("translations", [])]
    symmetry_count = max(1, len(rotations))
    assignments: list[ClusterWyckoffAssignment] = []
    for cluster in clusters:
        member_positions: list[tuple[float, float, float]] = []
        for member_index in cluster.member_indexes:
            index = int(member_index)
            if index < 0 or index >= len(ordered_sites):
                continue
            if index < len(anchored):
                member_positions.append(anchored[index].shifted_fractional_position)
            else:
                member_positions.append(ordered_sites[index].folded_fractional_position)
        if not member_positions:
            assignments.append(
                ClusterWyckoffAssignment(
                    cluster_index=int(cluster.cluster_index),
                    match=_unavailable_match("Cluster contains no available relaxed sites."),
                )
            )
            continue

        reference = _cluster_reference(member_positions)
        raw_candidates = _project_nearby_special_candidates(
            reference,
            rotations,
            translations,
            host_structure.lattice_vectors,
            tolerance_ang=match_tolerance_ang,
        )
        candidates = [*raw_candidates, reference]
        ranked: list[tuple[tuple[float, float, float, float], WyckoffSiteMatch]] = []
        for candidate in candidates:
            orbit = _symmetry_orbit(
                candidate,
                rotations,
                translations,
                host_structure.lattice_vectors,
                tolerance_ang=max(float(symprec), 1e-5),
            )
            if not orbit:
                continue
            distances = _match_distances_to_orbit(member_positions, orbit, host_structure.lattice_vectors)
            max_distance = max(distances) if distances else None
            mean_distance = float(np.mean(distances)) if distances else None
            if max_distance is None or max_distance > float(match_tolerance_ang):
                continue
            occupied = _orbit_is_occupied(
                orbit,
                host_structure,
                tolerance_ang=float(occupancy_tolerance_ang),
            )
            if occupied:
                continue
            label = _label_candidate(host_cell, orbit, symprec=symprec)
            if label is None:
                continue
            stabilizer = sum(
                1
                for rotation, translation in zip(rotations, translations, strict=False)
                if _operation_leaves_position_fixed(
                    rotation,
                    translation,
                    candidate,
                    host_structure.lattice_vectors,
                    tolerance_ang=max(float(symprec) * 10.0, 1e-4),
                )
            )
            special = stabilizer > 1 or len(orbit) < symmetry_count
            is_reference = _pbc_distance(candidate, reference, host_structure.lattice_vectors) <= 1e-7
            if is_reference and not special and raw_candidates:
                continue
            match = WyckoffSiteMatch(
                matched=True,
                label=label.label,
                fractional_position=_wrap_fractional(candidate),
                cartesian_position_ang=_cartesian_position(candidate, host_structure.lattice_vectors),
                multiplicity=label.multiplicity,
                letter=label.letter,
                space_group_number=label.space_group_number,
                space_group_symbol=label.space_group_symbol,
                site_symmetry_symbol=label.site_symmetry_symbol,
                max_distance_ang=float(max_distance),
                mean_distance_ang=float(mean_distance) if mean_distance is not None else None,
                occupied=False,
                special=bool(special),
                warnings=(),
            )
            rank = (
                float(label.multiplicity),
                -float(stabilizer),
                float(max_distance),
                float(mean_distance) if mean_distance is not None else 0.0,
            )
            ranked.append((rank, match))

        if ranked:
            ranked.sort(key=lambda item: item[0])
            selected = ranked[0][1]
        else:
            selected = _unavailable_match("No unoccupied Wyckoff position was consistent with every cluster member.")
        assignments.append(ClusterWyckoffAssignment(cluster_index=int(cluster.cluster_index), match=selected))

    return tuple(assignments)

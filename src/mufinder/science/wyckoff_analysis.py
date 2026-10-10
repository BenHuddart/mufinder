"""Wyckoff-site assignment helpers for relaxed muon clusters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from ase.data import atomic_numbers
import numpy as np

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
from mufinder.science.site_generation import (
    _greedy_unique_indexes,
    _pbc_distance_matrix,
    lattice_frac_to_cart,
)


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


def _operation_images(
    position: Sequence[float],
    rotations: Sequence[np.ndarray] | np.ndarray,
    translations: Sequence[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Apply every symmetry operation to ``position`` at once (unwrapped images, one per row)."""

    count = min(len(rotations), len(translations))
    if count == 0:
        return np.empty((0, 3), dtype=float)
    rotation_stack = np.asarray(rotations[:count], dtype=float).reshape(count, 3, 3)
    translation_stack = np.asarray(translations[:count], dtype=float).reshape(count, 3)
    return rotation_stack @ np.asarray(position, dtype=float) + translation_stack


def _deduplicate_positions(
    positions: Sequence[Sequence[float]] | np.ndarray,
    lattice: Sequence[Sequence[float]],
    *,
    tolerance_ang: float,
) -> tuple[tuple[float, float, float], ...]:
    # Dropping exact repeats first cannot change the result: a repeat is always within
    # tolerance of whatever kept entry covered (or was) its first occurrence.
    wrapped = list(dict.fromkeys(_wrap_fractional(position) for position in positions))
    if not wrapped:
        return ()
    distances = _pbc_distance_matrix(wrapped, wrapped, lattice)
    return tuple(wrapped[index] for index in _greedy_unique_indexes(distances <= tolerance_ang))


def _orbit_from_images(
    images: np.ndarray,
    lattice: Sequence[Sequence[float]],
    *,
    tolerance_ang: float,
) -> tuple[tuple[float, float, float], ...]:
    # ``np.mod`` matches Python's ``%``; each image is wrapped once here and once more on deduplication.
    return _deduplicate_positions(np.mod(images, 1.0), lattice, tolerance_ang=tolerance_ang)


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
    host_positions: Sequence[Sequence[float]],
    lattice: Sequence[Sequence[float]],
    *,
    tolerance_ang: float,
) -> bool:
    if not orbit or not host_positions:
        return False
    distances = _pbc_distance_matrix(orbit, host_positions, lattice)
    return bool(np.any(distances <= float(tolerance_ang)))


_PROJECTION_RESIDUAL_TOLERANCE = 1e-5
# The batched pair screen only decides which pairs are worth projecting exactly, so it
# keeps anything within this slack of the exact acceptance thresholds.
_PAIR_SCREEN_SLACK = 1e-6


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
    if not np.isfinite(residual) or residual > _PROJECTION_RESIDUAL_TOLERANCE:
        return None
    return _wrap_fractional(projected)


def _non_identity_operations(
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    operations: list[tuple[np.ndarray, np.ndarray]] = []
    identity = np.eye(3)
    for rotation, translation in zip(rotations, translations, strict=False):
        is_identity = np.allclose(rotation, identity) and np.allclose(translation - np.round(translation), 0.0)
        if not is_identity:
            operations.append((rotation, translation))
    return operations


def _batched_pseudo_inverse(matrices: np.ndarray) -> np.ndarray:
    u, singular_values, vt = np.linalg.svd(matrices)
    cutoff = 1e-10 * np.max(singular_values, axis=-1, keepdims=True)
    inverse = np.divide(1.0, singular_values, out=np.zeros_like(singular_values), where=singular_values > cutoff)
    return np.swapaxes(vt, -1, -2) @ (inverse[..., :, None] * np.swapaxes(u, -1, -2))


def _screen_operation_pairs(
    point: np.ndarray,
    operations: Sequence[tuple[np.ndarray, np.ndarray]],
    lattice: Sequence[Sequence[float]],
    *,
    cutoff_ang: float,
) -> list[tuple[int, int]]:
    """Return the operation pairs (in ``combinations`` order) whose joint projection may be accepted.

    Every pair is projected at once in NumPy. A pair is kept unless its projection is
    clearly inconsistent or clearly farther than ``cutoff_ang`` from ``point``; callers
    re-project the kept pairs with :func:`_project_to_constraints` so the accepted
    positions are bit-for-bit those of the exact per-pair projection.
    """

    count = len(operations)
    if count < 2:
        return []
    rows = np.empty((count, 3, 3), dtype=float)
    rhs = np.empty((count, 3), dtype=float)
    for index, (rotation, translation) in enumerate(operations):
        # The same integer shift ``_project_to_constraints`` picks for this operation.
        n = np.round(np.dot(rotation, point) + translation - point)
        rows[index] = rotation - np.eye(3)
        rhs[index] = n - translation
    # ``_project_to_constraints`` drops rows of ``R - I`` that are zero, with their right-hand side.
    active = np.linalg.norm(rows, axis=-1) > 1e-12
    rows = np.where(active[..., None], rows, 0.0)
    rhs = np.where(active, rhs, 0.0)

    left, right = np.triu_indices(count, k=1)
    a = np.concatenate([rows[left], rows[right]], axis=1)
    b = np.concatenate([rhs[left], rhs[right]], axis=1)
    a_t = np.swapaxes(a, -1, -2)
    misfit = a @ point - b
    correction = (_batched_pseudo_inverse(a_t @ a) @ (a_t @ misfit[..., None]))[..., 0]
    projected = point - correction
    residual = np.linalg.norm((a @ projected[..., None])[..., 0] - b, axis=-1)
    delta = projected - point
    delta = delta - np.round(delta)
    distance = np.linalg.norm(delta @ np.asarray(lattice, dtype=float), axis=-1)
    keep = (
        np.isfinite(residual)
        & (residual <= _PROJECTION_RESIDUAL_TOLERANCE + _PAIR_SCREEN_SLACK)
        & (distance <= cutoff_ang + _PAIR_SCREEN_SLACK)
    )
    return [(int(i), int(j)) for i, j in zip(left[keep], right[keep], strict=True)]


def _project_nearby_special_candidates(
    reference: Sequence[float],
    rotations: Sequence[np.ndarray],
    translations: Sequence[np.ndarray],
    lattice: Sequence[Sequence[float]],
    *,
    tolerance_ang: float,
) -> tuple[tuple[float, float, float], ...]:
    """Project ``reference`` onto the fixed-point loci of single operations and operation pairs.

    Gives the same result as projecting onto every operation and every pair of operations
    with :func:`_project_to_constraints`, without doing most of that work:

    * a pair that includes an operation whose own projection failed is skipped. Either that
      operation has no fixed point (no solution, so neither does the pair), or it is a pure
      translation with no constraints (the pair reproduces the other operation's single
      projection exactly, which deduplication would drop);
    * the remaining pairs are screened in one batched NumPy pass, and only those that can
      pass the residual and distance checks are projected exactly.
    """

    point = np.asarray(reference, dtype=float)
    operations = _non_identity_operations(rotations, translations)
    cutoff = max(float(tolerance_ang) * 2.0, float(tolerance_ang) + 0.1)

    candidates: list[tuple[float, float, float]] = []
    pairable: list[tuple[np.ndarray, np.ndarray]] = []
    for operation in operations:
        projected = _project_to_constraints(reference, (operation,))
        if projected is not None:
            candidates.append(projected)
            pairable.append(operation)

    for left, right in _screen_operation_pairs(point, pairable, lattice, cutoff_ang=cutoff):
        projected = _project_to_constraints(reference, (pairable[left], pairable[right]))
        if projected is not None:
            candidates.append(projected)

    if not candidates:
        return ()
    distances = _pbc_distance_matrix([reference], candidates, lattice)[0]
    nearby = [candidate for candidate, distance in zip(candidates, distances, strict=True) if distance <= cutoff]
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
    distances = _pbc_distance_matrix(member_positions, orbit, lattice)
    return tuple(float(value) for value in distances.min(axis=1))


@lru_cache(maxsize=32)
def _host_symmetry_operations(
    lattice: tuple[tuple[float, ...], ...],
    positions: tuple[tuple[float, ...], ...],
    numbers: tuple[int, ...],
    symprec: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the host's stacked rotations and translations, cached so repeated assignments skip spglib.

    The returned arrays are read-only because they are shared between callers.
    """

    host_cell = (
        np.asarray(lattice, dtype=float),
        np.asarray(positions, dtype=float),
        np.asarray(numbers, dtype=int),
    )
    try:
        symmetry = get_crystal_symmetry(host_cell, symprec=symprec)
    except Exception:  # noqa: BLE001 - spglib can fail on malformed cells.
        return None
    if symmetry is None:
        return None
    rotations = np.asarray(symmetry.get("rotations", []), dtype=float).reshape(-1, 3, 3)
    translations = np.asarray(symmetry.get("translations", []), dtype=float).reshape(-1, 3)
    rotations.setflags(write=False)
    translations.setflags(write=False)
    return rotations, translations


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

    lattice_array, host_positions, host_numbers = host_cell
    host_key = (
        tuple(tuple(float(value) for value in row) for row in lattice_array),
        tuple(tuple(float(value) for value in row) for row in host_positions),
        tuple(int(value) for value in host_numbers),
    )
    if _host_symmetry_operations(*host_key, float(symprec)) is None:
        return tuple(
            ClusterWyckoffAssignment(
                cluster_index=int(cluster.cluster_index),
                match=_unavailable_match("Host symmetry could not be identified by spglib."),
            )
            for cluster in clusters
        )

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
        match = _match_cluster(
            host_key,
            tuple(tuple(float(value) for value in position) for position in member_positions),
            match_tolerance_ang=float(match_tolerance_ang),
            occupancy_tolerance_ang=float(occupancy_tolerance_ang),
            symprec=float(symprec),
        )
        assignments.append(ClusterWyckoffAssignment(cluster_index=int(cluster.cluster_index), match=match))

    return tuple(assignments)


@lru_cache(maxsize=512)
def _match_cluster(
    host_key: tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], tuple[int, ...]],
    member_positions: tuple[tuple[float, ...], ...],
    *,
    match_tolerance_ang: float,
    occupancy_tolerance_ang: float,
    symprec: float,
) -> WyckoffSiteMatch:
    """Select the Wyckoff match for one cluster's anchored member positions.

    This is a pure function of its hashable arguments, so results are cached: the
    clustering dialog re-runs the assignment on every method change, usually with many
    of the same clusters.
    """

    lattice, host_positions, host_numbers = host_key
    operations = _host_symmetry_operations(lattice, host_positions, host_numbers, symprec)
    if operations is None:  # pragma: no cover - the caller checks this first.
        return _unavailable_match("Host symmetry could not be identified by spglib.")
    rotations, translations = operations
    host_cell = (
        np.asarray(lattice, dtype=float),
        np.asarray(host_positions, dtype=float),
        np.asarray(host_numbers, dtype=int),
    )
    symmetry_count = max(1, len(rotations))

    reference = _cluster_reference(member_positions)
    raw_candidates = _project_nearby_special_candidates(
        reference,
        rotations,
        translations,
        lattice,
        tolerance_ang=match_tolerance_ang,
    )
    candidates = [*raw_candidates, reference]
    # Rank every geometrically acceptable candidate first; the rank does not depend on
    # the spglib label, so only candidates in rank order need labelling, stopping at the
    # first one spglib can label. Python's sort is stable, so ties keep candidate order.
    ranked: list[tuple[tuple[float, float, float, float], tuple[tuple[float, float, float], ...], tuple[float, ...], float, float | None, bool]] = []
    for candidate in candidates:
        images = _operation_images(candidate, rotations, translations)
        orbit = _orbit_from_images(images, lattice, tolerance_ang=max(symprec, 1e-5))
        if not orbit:
            continue
        distances = _match_distances_to_orbit(member_positions, orbit, lattice)
        max_distance = max(distances) if distances else None
        mean_distance = float(np.mean(distances)) if distances else None
        if max_distance is None or max_distance > match_tolerance_ang:
            continue
        if _orbit_is_occupied(orbit, host_positions, lattice, tolerance_ang=occupancy_tolerance_ang):
            continue
        # Operations mapping the candidate onto itself (images wrapped once).
        stabilizer = int(
            np.count_nonzero(
                _pbc_distance_matrix(np.mod(images, 1.0), [candidate], lattice) <= max(symprec * 10.0, 1e-4)
            )
        )
        special = stabilizer > 1 or len(orbit) < symmetry_count
        is_reference = _pbc_distance(candidate, reference, lattice) <= 1e-7
        if is_reference and not special and raw_candidates:
            continue
        rank = (
            float(len(orbit)),
            -float(stabilizer),
            float(max_distance),
            float(mean_distance) if mean_distance is not None else 0.0,
        )
        ranked.append((rank, orbit, candidate, max_distance, mean_distance, special))

    ranked.sort(key=lambda item: item[0])
    for _, orbit, candidate, max_distance, mean_distance, special in ranked:
        label = _label_candidate(host_cell, orbit, symprec=symprec)
        if label is None:
            continue
        return WyckoffSiteMatch(
            matched=True,
            label=label.label,
            fractional_position=_wrap_fractional(candidate),
            cartesian_position_ang=_cartesian_position(candidate, lattice),
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
    return _unavailable_match("No unoccupied Wyckoff position was consistent with every cluster member.")

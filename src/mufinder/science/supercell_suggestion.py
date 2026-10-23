"""Suggest a diagonal supercell scaling for muon stopping-site calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


MIN_IMAGE_DISTANCE_ANGSTROM: float = 10.0  # muon image-separation rule of thumb (Å)
MAX_SUPERCELL_ATOMS: int = 200  # affordability ceiling — atom count must be strictly less
DISPLAY_DISTANCE_ANGSTROM: float = 8.0  # minimum image distance shown in suggestion table

_SCALING_CAP: int = 64  # defensive per-axis cap; practical cells never reach this


class SupercellSuggestionError(ValueError):
    """Raised for invalid lattice or atom-count inputs in supercell suggestion."""


@dataclass(frozen=True, slots=True)
class SupercellCandidate:
    """A diagonal supercell candidate with computed geometry."""

    scaling: tuple[int, int, int]
    atom_count: int  # = primitive_atom_count * n1*n2*n3 (exact int product)
    min_image_distance: float  # Å, shortest translation in {-1,0,1}^3 \ {0}
    satisfies_criterion: bool  # min_image_distance >= distance_target


@dataclass(frozen=True, slots=True)
class SupercellSuggestion:
    """Result of suggest_supercell, carrying all data for the presentation layer."""

    candidates: tuple[SupercellCandidate, ...]  # Pareto front, sorted by atom_count asc
    recommended_index: int  # index of the recommended candidate in candidates
    primitive_atom_count: int
    distance_target: float  # the min_image_distance threshold used (10 Å by default)
    display_threshold: float  # minimum distance included in candidates (8 Å by default)
    max_atoms: int  # the atom-count affordability ceiling used
    primitive_exceeds_budget: bool  # primitive cell already >= max_atoms


def _min_image_distance(super_vectors: np.ndarray) -> float:
    """Return min |n1*A1 + n2*A2 + n3*A3| over (n1,n2,n3) in {-1,0,1}^3 \\ {(0,0,0)}."""
    best = math.inf
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            for n3 in (-1, 0, 1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                t = n1 * super_vectors[0] + n2 * super_vectors[1] + n3 * super_vectors[2]
                d = float(np.linalg.norm(t))
                if d < best:
                    best = d
    return best


def _scaled_vectors(lattice: np.ndarray, n: tuple[int, int, int]) -> np.ndarray:
    """Row vectors of the diagonal supercell n_i * a_i."""
    return lattice * np.array(n, dtype=float)[:, None]


def _make_candidate(
    lattice: np.ndarray,
    n: tuple[int, int, int],
    primitive_atom_count: int,
    distance_target: float,
) -> SupercellCandidate:
    d = _min_image_distance(_scaled_vectors(lattice, n))
    return SupercellCandidate(
        scaling=n,
        atom_count=n[0] * n[1] * n[2] * primitive_atom_count,
        min_image_distance=d,
        satisfies_criterion=d >= distance_target,
    )


def _find_criterion_candidate(
    lattice: np.ndarray,
    primitive_atom_count: int,
    distance_target: float,
) -> SupercellCandidate:
    """Find the smallest diagonal scaling meeting distance_target (used to bound enumeration)."""
    a_norms = [float(np.linalg.norm(lattice[i])) for i in range(3)]
    n = [max(1, math.ceil(distance_target / a_norms[i])) for i in range(3)]

    while True:
        sv = _scaled_vectors(lattice, (n[0], n[1], n[2]))
        if _min_image_distance(sv) >= distance_target:
            break
        # Grow the axis with the shortest supercell length (most likely bottleneck).
        lengths = [n[i] * a_norms[i] for i in range(3)]
        axis = lengths.index(min(lengths))
        if n[axis] >= _SCALING_CAP:
            break
        n[axis] += 1

    return _make_candidate(lattice, (n[0], n[1], n[2]), primitive_atom_count, distance_target)


def _pareto_sort_key(
    candidate: SupercellCandidate,
) -> tuple[int, float, float, tuple[int, ...]]:
    n = candidate.scaling
    anisotropy = max(n) / min(n) if min(n) > 0 else float("inf")
    return (candidate.atom_count, -candidate.min_image_distance, anisotropy, tuple(sorted(n)))


def _find_pareto_candidates(
    lattice: np.ndarray,
    primitive_atom_count: int,
    display_threshold: float,
    distance_target: float,
    max_atoms: int,
    criterion: SupercellCandidate,
) -> list[SupercellCandidate]:
    """Return the Pareto front of candidates with min_image_distance >= display_threshold.

    Enumerates all (n1,n2,n3) up to max(max_atoms, criterion.atom_count) atoms.
    Applies a running-maximum sweep (sorted by atom_count asc, distance desc) so
    each candidate in the result strictly improves on all cheaper candidates'
    distances.  The criterion cell is force-inserted when it would otherwise be
    absent (rare edge case for non-orthogonal lattices).
    """
    upper_atoms = max(max_atoms, criterion.atom_count)
    max_product = upper_atoms // primitive_atom_count

    raw: list[SupercellCandidate] = []
    for n1 in range(1, max_product + 1):
        for n2 in range(1, max_product // n1 + 1):
            max_n3 = max_product // (n1 * n2)
            for n3 in range(1, max_n3 + 1):
                c = _make_candidate(lattice, (n1, n2, n3), primitive_atom_count, distance_target)
                if c.min_image_distance >= display_threshold:
                    raw.append(c)

    raw.sort(key=_pareto_sort_key)

    # Running-maximum sweep: include only cells that strictly improve the distance.
    pareto: list[SupercellCandidate] = []
    max_dist = -math.inf
    for c in raw:
        if c.min_image_distance > max_dist:
            pareto.append(c)
            max_dist = c.min_image_distance

    # Guarantee the criterion is present (it is the target the user cares about).
    if not any(c.scaling == criterion.scaling for c in pareto):
        idx = next(
            (i for i, c in enumerate(pareto) if c.atom_count > criterion.atom_count),
            len(pareto),
        )
        pareto.insert(idx, criterion)

    return pareto


def suggest_supercell(
    lattice_vectors: Sequence[Sequence[float]],
    *,
    primitive_atom_count: int,
    min_image_distance: float = MIN_IMAGE_DISTANCE_ANGSTROM,
    max_atoms: int = MAX_SUPERCELL_ATOMS,
    display_threshold: float = DISPLAY_DISTANCE_ANGSTROM,
) -> SupercellSuggestion:
    """Return a set of Pareto-optimal diagonal supercell candidates for display.

    Enumerates all diagonal scalings (n1, n2, n3) whose minimum periodic
    muon-image distance — the shortest non-zero lattice translation over all
    (m1, m2, m3) in {-1,0,1}^3 — is at least *display_threshold* Angstroms,
    up to the atom count of the smallest cell meeting *min_image_distance*.
    Retains the Pareto front (sorted by atom_count ascending) so each candidate
    strictly improves on all cheaper options' distances.

    The recommended candidate is the cheapest cell meeting *min_image_distance*
    that has fewer than *max_atoms* atoms.  When no such cell exists the
    recommendation falls back to the best affordable cell (< *max_atoms* atoms),
    or to the criterion cell when no affordable cell is available.

    Raises SupercellSuggestionError for degenerate or malformed inputs.
    """
    if len(lattice_vectors) != 3:
        raise SupercellSuggestionError("lattice_vectors must contain three vectors.")
    for row in lattice_vectors:
        if len(row) != 3:
            raise SupercellSuggestionError("Each lattice vector must have three components.")
    if primitive_atom_count < 1:
        raise SupercellSuggestionError(
            f"primitive_atom_count must be a positive integer, got {primitive_atom_count!r}."
        )

    a1, a2, a3 = lattice_vectors[0], lattice_vectors[1], lattice_vectors[2]
    volume = (
        float(a1[0]) * (float(a2[1]) * float(a3[2]) - float(a2[2]) * float(a3[1]))
        - float(a1[1]) * (float(a2[0]) * float(a3[2]) - float(a2[2]) * float(a3[0]))
        + float(a1[2]) * (float(a2[0]) * float(a3[1]) - float(a2[1]) * float(a3[0]))
    )
    if not math.isfinite(volume) or abs(volume) <= 1e-12:
        raise SupercellSuggestionError("Lattice vectors must define a non-zero cell volume.")

    lattice = np.asarray(lattice_vectors, dtype=float)
    primitive_exceeds_budget = primitive_atom_count >= max_atoms

    if primitive_exceeds_budget:
        solo = _make_candidate(lattice, (1, 1, 1), primitive_atom_count, min_image_distance)
        return SupercellSuggestion(
            candidates=(solo,),
            recommended_index=0,
            primitive_atom_count=primitive_atom_count,
            distance_target=min_image_distance,
            display_threshold=display_threshold,
            max_atoms=max_atoms,
            primitive_exceeds_budget=True,
        )

    # Use the greedy criterion search to establish the upper bound for enumeration.
    criterion = _find_criterion_candidate(lattice, primitive_atom_count, min_image_distance)

    pareto = _find_pareto_candidates(
        lattice, primitive_atom_count, display_threshold, min_image_distance, max_atoms, criterion
    )

    # Recommended: cheapest cell meeting the distance criterion and under the atom budget.
    recommended_index: int | None = next(
        (i for i, c in enumerate(pareto) if c.satisfies_criterion and c.atom_count < max_atoms),
        None,
    )
    if recommended_index is None:
        # No affordable criterion-meeting cell; recommend best affordable (highest distance).
        affordable = [i for i, c in enumerate(pareto) if c.atom_count < max_atoms]
        if affordable:
            recommended_index = affordable[-1]
        else:
            # No affordable cell at all; recommend the cheapest criterion-meeting cell.
            meeting = [i for i, c in enumerate(pareto) if c.satisfies_criterion]
            recommended_index = meeting[0] if meeting else 0

    return SupercellSuggestion(
        candidates=tuple(pareto),
        recommended_index=recommended_index,
        primitive_atom_count=primitive_atom_count,
        distance_target=min_image_distance,
        display_threshold=display_threshold,
        max_atoms=max_atoms,
        primitive_exceeds_budget=False,
    )

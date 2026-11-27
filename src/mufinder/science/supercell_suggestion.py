"""Suggest a diagonal supercell scaling for muon stopping-site calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mufinder.science.lattice import shortest_translation_length

MIN_IMAGE_DISTANCE_ANGSTROM: float = 10.0  # muon image-separation rule of thumb (Å)
MAX_SUPERCELL_ATOMS: int = 200  # affordability ceiling — atom count must be strictly less
DISPLAY_DISTANCE_ANGSTROM: float = 8.0  # minimum image distance shown in suggestion table

_SCALING_CAP: int = 64  # per-axis cap, the Supercell Editor's largest scaling
_SEARCH_BUDGET_FACTOR: int = 5  # the search for a cell reaching the target stops at 5x max_atoms
_TIE_RTOL_ABOVE: float = 1.0 + 1e-9  # a distance must exceed another by this factor to improve


class SupercellSuggestionError(ValueError):
    """Raised for invalid lattice or atom-count inputs in supercell suggestion."""


@dataclass(frozen=True, slots=True)
class SupercellCandidate:
    """A diagonal supercell candidate with computed geometry."""

    scaling: tuple[int, int, int]
    atom_count: int  # = primitive_atom_count * n1*n2*n3 (exact int product)
    min_image_distance: float  # Å, the supercell's shortest lattice translation
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


def _scaled_vectors(lattice: np.ndarray, n: tuple[int, int, int]) -> np.ndarray:
    """Row vectors of the diagonal supercell n_i * a_i."""
    return lattice * np.array(n, dtype=float)[:, None]


def _make_candidate(
    lattice: np.ndarray,
    n: tuple[int, int, int],
    primitive_atom_count: int,
    distance_target: float,
) -> SupercellCandidate:
    d = shortest_translation_length(_scaled_vectors(lattice, n))
    return SupercellCandidate(
        scaling=n,
        atom_count=n[0] * n[1] * n[2] * primitive_atom_count,
        min_image_distance=d,
        satisfies_criterion=d >= distance_target,
    )


def _candidates_between(
    lattice: np.ndarray,
    primitive_atom_count: int,
    distance_target: float,
    low_product: int,
    high_product: int,
) -> list[SupercellCandidate]:
    """Every scaling with ``low_product < n1*n2*n3 <= high_product`` and each ``n_i <= 64``."""
    found: list[SupercellCandidate] = []
    for n1 in range(1, min(high_product, _SCALING_CAP) + 1):
        for n2 in range(1, min(high_product // n1, _SCALING_CAP) + 1):
            first_n3 = low_product // (n1 * n2) + 1
            last_n3 = min(high_product // (n1 * n2), _SCALING_CAP)
            for n3 in range(first_n3, last_n3 + 1):
                found.append(
                    _make_candidate(lattice, (n1, n2, n3), primitive_atom_count, distance_target)
                )
    return found


def _search_candidates(
    lattice: np.ndarray,
    primitive_atom_count: int,
    distance_target: float,
    max_atoms: int,
) -> list[SupercellCandidate]:
    """Every diagonal cell up to the atom budget, or up to the cheapest that reaches the target.

    The search covers the cells of at most *max_atoms* atoms. When none of them
    reaches *distance_target* it goes on, doubling its bound, to the cheapest cell
    that does, and keeps no cell larger than that one; it stops at
    ``_SEARCH_BUDGET_FACTOR`` times the budget.
    """
    budget = max_atoms // primitive_atom_count
    ceiling = max(budget, _SEARCH_BUDGET_FACTOR * max_atoms // primitive_atom_count)
    candidates = _candidates_between(lattice, primitive_atom_count, distance_target, 0, budget)
    searched = budget
    while searched < ceiling and not any(c.satisfies_criterion for c in candidates):
        bound = min(2 * searched, ceiling)
        candidates += _candidates_between(
            lattice, primitive_atom_count, distance_target, searched, bound
        )
        searched = bound
    reaching = [c.atom_count for c in candidates if c.satisfies_criterion]
    if reaching and min(reaching) > budget * primitive_atom_count:
        candidates = [c for c in candidates if c.atom_count <= min(reaching)]
    return candidates


def _pareto_sort_key(
    candidate: SupercellCandidate,
) -> tuple[int, float, float, tuple[int, ...]]:
    n = candidate.scaling
    anisotropy = max(n) / min(n) if min(n) > 0 else float("inf")
    return (candidate.atom_count, -candidate.min_image_distance, anisotropy, tuple(sorted(n)))


def _farthest_apart(ordered: list[SupercellCandidate]) -> SupercellCandidate | None:
    """The cell whose images are farthest apart, the cheapest of any (near) tie."""
    best: SupercellCandidate | None = None
    for c in ordered:
        if best is None or c.min_image_distance > best.min_image_distance * _TIE_RTOL_ABOVE:
            best = c
    return best


def _pareto_front(
    candidates: list[SupercellCandidate],
    display_threshold: float,
    max_atoms: int,
) -> list[SupercellCandidate]:
    """Return the cells to list, sorted by atom count, each farther apart than all cheaper ones.

    A cell is listed when its distance is at least *display_threshold*, and the
    anchors always are: the cheapest cell that reaches the target or, when none
    does, the one whose images are farthest apart and the best cell under
    *max_atoms*, so the recommendation can stay affordable. Every cheaper cell is
    closer than an anchor, so the running-maximum sweep keeps it. Distances
    within a relative 1e-9 of each other are a tie, which the cheaper cell wins.
    """
    ordered = sorted(candidates, key=_pareto_sort_key)
    reaching = next((c for c in ordered if c.satisfies_criterion), None)
    if reaching is not None:
        anchors = [reaching]
    else:
        affordable = [c for c in ordered if c.atom_count < max_atoms]
        anchors = [_farthest_apart(ordered), _farthest_apart(affordable)]

    pareto: list[SupercellCandidate] = []
    max_dist = -math.inf
    for c in ordered:
        listed = c.min_image_distance >= display_threshold or any(c is a for a in anchors)
        if listed and c.min_image_distance > max_dist * _TIE_RTOL_ABOVE:
            pareto.append(c)
            max_dist = c.min_image_distance
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

    Enumerates the diagonal scalings (n1, n2, n3), each at most 64, whose minimum
    periodic muon-image distance — the supercell's shortest non-zero lattice
    translation (:func:`~mufinder.science.lattice.shortest_translation_length`) —
    is at least *display_threshold* Angstroms, up to *max_atoms* atoms or, when no
    cell that small meets *min_image_distance*, up to the cheapest that does
    (searched no further than five times *max_atoms*). Retains the Pareto front
    (sorted by atom_count ascending) so each candidate strictly improves on all
    cheaper options' distances; the cheapest cell meeting *min_image_distance* is
    always on it.

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

    candidates = _search_candidates(lattice, primitive_atom_count, min_image_distance, max_atoms)
    pareto = _pareto_front(candidates, display_threshold, max_atoms)

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

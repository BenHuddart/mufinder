"""Tests for the supercell suggestion algorithm."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mufinder.science.supercell_suggestion import (
    DISPLAY_DISTANCE_ANGSTROM,
    MAX_SUPERCELL_ATOMS,
    MIN_IMAGE_DISTANCE_ANGSTROM,
    SupercellSuggestionError,
    suggest_supercell,
)
from mufinder.science import supercell_suggestion as suggestion_module
from tests.support.lattices import (
    brute_force_shortest_translation,
    monoclinic_lattice,
    triclinic_lattice,
)
from tests.support.structures import hexagonal_magnesium_oxide_host, si_pair_host
from tests.support.symmetry_hosts import (
    bcc_host,
    cof2_host,
    diamond_si_host,
    fcc_primitive_host,
    gdru2si2_host,
    hcp_host,
    rutile_host,
    si_primitive_host,
    triclinic_host,
    zno_host,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cubic(a: float) -> list[list[float]]:
    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


def _axis_lengths(lattice: list[list[float]], scaling: tuple[int, int, int]) -> list[float]:
    return [scaling[i] * math.sqrt(sum(c**2 for c in lattice[i])) for i in range(3)]


def _recommended(result) -> object:
    return result.candidates[result.recommended_index]


# ---------------------------------------------------------------------------
# Pareto front structure
# ---------------------------------------------------------------------------


def test_pareto_sorted_by_atom_count_ascending() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    counts = [c.atom_count for c in result.candidates]
    assert counts == sorted(counts)


def test_pareto_each_candidate_strictly_improves_distance() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    distances = [c.min_image_distance for c in result.candidates]
    for i in range(1, len(distances)):
        assert distances[i] > distances[i - 1]


def test_pareto_all_candidates_meet_display_threshold() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    for c in result.candidates:
        assert c.min_image_distance >= DISPLAY_DISTANCE_ANGSTROM


def test_pareto_contains_candidate_meeting_criterion() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    criterion_cells = [c for c in result.candidates if c.satisfies_criterion]
    assert criterion_cells, "Expected at least one candidate meeting the 10 Å criterion"


# ---------------------------------------------------------------------------
# Recommended: affordable criterion cell
# ---------------------------------------------------------------------------


def test_cubic_3a_recommended_is_criterion_meeting_and_affordable() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    rec = _recommended(result)
    assert rec.scaling == (4, 4, 4)
    assert rec.atom_count == 64
    assert rec.min_image_distance == pytest.approx(12.0)
    assert rec.satisfies_criterion is True
    assert rec.atom_count < MAX_SUPERCELL_ATOMS


def test_recommended_index_points_to_cheapest_criterion_cell() -> None:
    # Pareto is sorted by atom_count asc; recommended = first with satisfies_criterion.
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    idx = result.recommended_index
    assert result.candidates[idx].satisfies_criterion is True
    # All cheaper candidates should NOT satisfy the criterion.
    for c in result.candidates[:idx]:
        assert c.satisfies_criterion is False


def test_recommended_satisfies_criterion_and_is_affordable() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    rec = _recommended(result)
    assert rec.satisfies_criterion is True
    assert rec.atom_count < MAX_SUPERCELL_ATOMS


# ---------------------------------------------------------------------------
# Candidates below the criterion (8–10 Å range)
# ---------------------------------------------------------------------------


def test_cubic_3a_has_8_angstrom_candidate() -> None:
    # (3,3,3) gives 9 Å for the cubic 3 Å cell; should appear in candidates.
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    scalings = [c.scaling for c in result.candidates]
    assert (3, 3, 3) in scalings


def test_sub_criterion_candidates_do_not_satisfy_criterion() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    idx = result.recommended_index
    for c in result.candidates[:idx]:
        assert not c.satisfies_criterion
        assert c.min_image_distance >= DISPLAY_DISTANCE_ANGSTROM


# ---------------------------------------------------------------------------
# Recommended: best affordable when criterion exceeds budget
# ---------------------------------------------------------------------------


def test_tiny_cubic_recommended_is_best_affordable_when_criterion_expensive() -> None:
    # 1 Å cubic, 1 atom → criterion (10,10,10)=1000 atoms ≥ 200.
    # All Pareto cells with ≥ 8 Å have min(n) ≥ 8, so product ≥ 512 > 200.
    # No affordable cell meets ≥ 8 Å → recommend cheapest criterion cell.
    result = suggest_supercell(_cubic(1.0), primitive_atom_count=1)
    rec = _recommended(result)
    # The cheapest criterion-meeting cell for 1 Å cubic is (10,10,10)
    assert rec.satisfies_criterion is True
    assert rec.scaling == (10, 10, 10)


def test_tiny_cubic_pareto_shows_cells_from_8_angstrom() -> None:
    result = suggest_supercell(_cubic(1.0), primitive_atom_count=1)
    # (8,8,8)=512, (9,9,9)=729, (10,10,10)=1000 should all appear.
    scalings = [c.scaling for c in result.candidates]
    assert (8, 8, 8) in scalings
    assert (9, 9, 9) in scalings
    assert (10, 10, 10) in scalings


def test_recommended_is_affordable_alternative_when_criterion_over_budget() -> None:
    # 3 Å cubic, 4 atoms → criterion (4,4,4)=256 atoms ≥ 200.
    # (3,3,3)=108 atoms is affordable and meets the 8 Å display threshold (9 Å).
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=4)
    rec = _recommended(result)
    assert rec.atom_count < MAX_SUPERCELL_ATOMS
    assert not rec.satisfies_criterion  # best affordable doesn't reach 10 Å


def test_recommended_affordable_is_tagged_not_criterion() -> None:
    # When criterion exceeds budget, recommended should be the last affordable Pareto cell.
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=4)
    idx = result.recommended_index
    # All candidates after recommended have atom_count >= max_atoms.
    for c in result.candidates[idx + 1 :]:
        assert c.atom_count >= MAX_SUPERCELL_ATOMS


# ---------------------------------------------------------------------------
# Non-orthogonal cell: 27-point scan and refinement
# ---------------------------------------------------------------------------


def test_non_orthogonal_refinement_uses_diagonal_scan() -> None:
    # Oblique: a1=[5,0,0], a2=[4,3,0], a3=[0,0,5]. All |a_i|=5.
    # The Pareto front starts with (2,3,2): supercell rows [10,0,0],[12,9,0],[0,0,10];
    # translation (1,-1,0)=[-2,-9,0] has length sqrt(85)≈9.22, which is shorter than
    # every axis (10, 15, 10). This proves the 27-point scan is used — a naive axis-only
    # check would report 10 Å and incorrectly accept (2,3,2).
    lattice = [[5.0, 0.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]]
    result = suggest_supercell(lattice, primitive_atom_count=1)

    first = result.candidates[0]
    axis_lengths = _axis_lengths(lattice, first.scaling)
    assert first.min_image_distance >= DISPLAY_DISTANCE_ANGSTROM
    assert first.min_image_distance < min(axis_lengths)

    rec = _recommended(result)
    assert rec.satisfies_criterion is True
    assert rec.min_image_distance >= 10.0


# ---------------------------------------------------------------------------
# Primitive exceeds budget
# ---------------------------------------------------------------------------


def test_primitive_exceeds_budget_returns_single_candidate() -> None:
    result = suggest_supercell(_cubic(4.0), primitive_atom_count=200)
    assert result.primitive_exceeds_budget is True
    assert len(result.candidates) == 1
    assert result.candidates[0].scaling == (1, 1, 1)
    assert result.recommended_index == 0


def test_primitive_just_under_budget_is_not_flagged() -> None:
    result = suggest_supercell(_cubic(4.0), primitive_atom_count=199)
    assert result.primitive_exceeds_budget is False


# ---------------------------------------------------------------------------
# Display threshold and distance target stored in result
# ---------------------------------------------------------------------------


def test_result_stores_distance_target_and_display_threshold() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    assert result.distance_target == pytest.approx(MIN_IMAGE_DISTANCE_ANGSTROM)
    assert result.display_threshold == pytest.approx(DISPLAY_DISTANCE_ANGSTROM)
    assert result.max_atoms == MAX_SUPERCELL_ATOMS


def test_custom_display_threshold_excludes_cells_below_it() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1, display_threshold=11.0)
    for c in result.candidates:
        assert c.min_image_distance >= 11.0


def test_custom_min_image_distance_sets_criterion_target() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1, min_image_distance=6.0)
    rec = _recommended(result)
    assert rec.min_image_distance >= 6.0
    assert result.distance_target == pytest.approx(6.0)


def test_custom_max_atoms_shifts_recommendation() -> None:
    # With max_atoms=50, the 64-atom criterion cell is over budget.
    # Recommendation falls back to best affordable (< 50 atoms).
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=1, max_atoms=50)
    rec = _recommended(result)
    assert rec.atom_count < 50
    assert not rec.satisfies_criterion


# ---------------------------------------------------------------------------
# Atom count exactness
# ---------------------------------------------------------------------------


def test_atom_count_is_exact_integer_product() -> None:
    result = suggest_supercell(_cubic(3.0), primitive_atom_count=3)
    for c in result.candidates:
        n1, n2, n3 = c.scaling
        assert c.atom_count == n1 * n2 * n3 * 3


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_repeated_calls() -> None:
    r1 = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    r2 = suggest_supercell(_cubic(3.0), primitive_atom_count=1)
    assert r1.recommended_index == r2.recommended_index
    assert [c.scaling for c in r1.candidates] == [c.scaling for c in r2.candidates]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_rejects_non_3x3_lattice() -> None:
    with pytest.raises(SupercellSuggestionError, match="three vectors"):
        suggest_supercell([[1.0, 0, 0], [0, 1.0, 0]], primitive_atom_count=1)


def test_rejects_wrong_vector_length() -> None:
    with pytest.raises(SupercellSuggestionError, match="three components"):
        suggest_supercell([[1.0, 0], [0, 1.0, 0], [0, 0, 1.0]], primitive_atom_count=1)


def test_rejects_coplanar_lattice() -> None:
    with pytest.raises(SupercellSuggestionError, match="non-zero cell volume"):
        suggest_supercell([[1.0, 0, 0], [2.0, 0, 0], [0, 0, 1.0]], primitive_atom_count=1)


def test_rejects_zero_atom_count() -> None:
    with pytest.raises(SupercellSuggestionError, match="positive integer"):
        suggest_supercell(_cubic(3.0), primitive_atom_count=0)


def test_rejects_negative_atom_count() -> None:
    with pytest.raises(SupercellSuggestionError, match="positive integer"):
        suggest_supercell(_cubic(3.0), primitive_atom_count=-1)


# ---------------------------------------------------------------------------
# Image distance: the shortest lattice translation, not the adjacent cells alone
# ---------------------------------------------------------------------------


def _adjacent_cell_search(super_vectors) -> float:
    """The image distance as Suggest measured it before: coefficients of 0 or ±1 only.

    Kept here as the reference for the old result; a strongly sheared basis hides a
    shorter translation from it.
    """
    super_vectors = np.asarray(super_vectors, dtype=float)
    best = math.inf
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            for n3 in (-1, 0, 1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                t = n1 * super_vectors[0] + n2 * super_vectors[1] + n3 * super_vectors[2]
                best = min(best, float(np.linalg.norm(t)))
    return best


def _adjacent_cell_suggestion(lattice, primitive_atom_count: int):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(suggestion_module, "shortest_translation_length", _adjacent_cell_search)
        return suggest_supercell(lattice, primitive_atom_count=primitive_atom_count)


def _scaled(lattice, scaling: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(lattice, dtype=float) * np.asarray(scaling, dtype=float)[:, None]


# Host vectors (20, 0, 0), (42, 4, 0), (0, 0, 20) Å: -2 A1 + A2 = (2, 4, 0) is 4.47 Å.
_SHEARED_HOST = [[20.0, 0.0, 0.0], [42.0, 4.0, 0.0], [0.0, 0.0, 20.0]]
# A milder shear, (4, 0, 0), (9, 3, 0), (0, 0, 5) Å, whose recommendation was already larger.
_SHEARED_SMALL_HOST = [[4.0, 0.0, 0.0], [9.0, 3.0, 0.0], [0.0, 0.0, 5.0]]

# Strongly sheared hosts whose cheapest cell reaching 10 Å has an axis far longer than the others.
_SHEARED_LONG_HOST = [[4.0, 0.0, 0.0], [60.0, 0.5, 0.0], [0.0, 0.0, 4.0]]
_SHEARED_FLAT_HOST = [[20.0, 0.0, 0.0], [40.0, 0.15, 0.0], [0.0, 0.0, 20.0]]
_SHEARED_SLOW_HOST = [[10.5, 12.0, 0.0], [43.5, 50.0, 0.0], [0.0, 0.0, 3.0]]

# Each ordinary cell with the scalings Suggest lists for it and the one it recommends,
# as they were before the image distance was measured in a reduced basis (less only the
# rows that tied a cheaper one but for roundoff).
_ORDINARY_CELLS = {
    "cubic": (_cubic(3.0), 1, [(3, 3, 3), (4, 4, 4), (5, 5, 5)], (4, 4, 4)),
    "tetragonal CoF2": (
        cof2_host().lattice_vectors,
        6,
        [(2, 2, 3), (3, 3, 3), (3, 3, 4)],
        (3, 3, 3),
    ),
    "hexagonal": (
        hcp_host().lattice_vectors,
        2,
        # 3x4x2 and 4x5x3 are as far apart as 3x3x2 and 4x4x3 but for roundoff.
        [(3, 3, 2), (4, 4, 2), (4, 4, 3), (5, 5, 3), (5, 5, 4)],
        (4, 4, 2),
    ),
    "monoclinic 110": (
        monoclinic_lattice(5.0, 6.0, 7.0, 110.0),
        4,
        [(2, 2, 2), (3, 2, 2), (3, 3, 2), (3, 3, 3), (4, 3, 3), (4, 4, 3)],
        (2, 2, 2),
    ),
    "triclinic": (
        triclinic_lattice(5.1, 6.2, 7.3, 80.0, 85.0, 95.0),
        3,
        [(2, 2, 2), (3, 2, 2), (3, 3, 2), (3, 3, 3), (4, 3, 3), (4, 4, 3), (5, 4, 3)],
        (2, 2, 2),
    ),
}

_SUGGESTION_CELLS = {
    **{name: (lattice, atoms) for name, (lattice, atoms, _s, _r) in _ORDINARY_CELLS.items()},
    "oblique": ([[5.0, 0.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]], 1),
    "sheared host": (_SHEARED_HOST, 1),
    "sheared small host": (_SHEARED_SMALL_HOST, 1),
    "sheared long host": (_SHEARED_LONG_HOST, 1),
    "sheared flat host": (_SHEARED_FLAT_HOST, 2),
    "sheared slow host": (_SHEARED_SLOW_HOST, 2),
    **{
        structure.label: (structure.lattice_vectors, len(structure.species))
        for structure in (
            bcc_host(),
            cof2_host(),
            diamond_si_host(),
            fcc_primitive_host(),
            gdru2si2_host(),
            hcp_host(),
            rutile_host(),
            si_primitive_host(),
            triclinic_host(),
            zno_host(),
            hexagonal_magnesium_oxide_host(),
            si_pair_host(),
        )
    },
}


def test_sheared_unit_cell_is_measured_by_its_shortest_translation() -> None:
    result = suggest_supercell(_SHEARED_HOST, primitive_atom_count=1)

    # The adjacent cells alone give 20 Å for the unit cell and recommend it.
    old = _adjacent_cell_suggestion(_SHEARED_HOST, 1)
    assert _recommended(old).scaling == (1, 1, 1)
    assert _recommended(old).min_image_distance == pytest.approx(20.0)

    assert all(c.scaling != (1, 1, 1) for c in result.candidates)
    rec = _recommended(result)
    assert rec.scaling == (1, 3, 1)
    assert rec.atom_count == 3
    assert rec.min_image_distance == pytest.approx(math.sqrt(180.0))
    for c in result.candidates:
        assert c.min_image_distance == pytest.approx(
            brute_force_shortest_translation(_scaled(_SHEARED_HOST, c.scaling)), rel=1e-12
        )


def test_sheared_cell_recommendation_grows_to_a_cell_that_truly_reaches_the_target() -> None:
    old = _adjacent_cell_suggestion(_SHEARED_SMALL_HOST, 1)
    result = suggest_supercell(_SHEARED_SMALL_HOST, primitive_atom_count=1)

    # (3, 3, 2) has 12, 28.5 and 10 Å edges, but 3 A2 - 6 A1 = (3, 9, 0) Å is only 9.49 Å.
    assert _recommended(old).scaling == (3, 3, 2)
    shortest = brute_force_shortest_translation(_scaled(_SHEARED_SMALL_HOST, (3, 3, 2)))
    assert shortest == pytest.approx(math.sqrt(90.0))
    rec = _recommended(result)
    assert rec.scaling == (3, 4, 2)
    assert rec.min_image_distance == pytest.approx(10.0)
    assert rec.satisfies_criterion


def test_sheared_host_recommends_the_cheapest_cell_that_reaches_the_target() -> None:
    result = suggest_supercell(_SHEARED_LONG_HOST, primitive_atom_count=1)

    rec = _recommended(result)
    assert rec.scaling == (13, 4, 3)
    assert rec.atom_count == 156
    assert rec.satisfies_criterion
    # Nothing cheaper reaches 10 Å, and nothing over the budget is listed.
    assert not any(c.satisfies_criterion for c in result.candidates[: result.recommended_index])
    assert max(c.atom_count for c in result.candidates) <= MAX_SUPERCELL_ATOMS
    for c in result.candidates:
        vectors = _scaled(_SHEARED_LONG_HOST, c.scaling)
        assert c.min_image_distance == pytest.approx(
            brute_force_shortest_translation(vectors, upper_bound=c.min_image_distance), rel=1e-12
        )


def test_suggestions_stay_within_the_editor_scaling_and_a_bounded_search() -> None:
    flat = suggest_supercell(_SHEARED_FLAT_HOST, primitive_atom_count=2)
    slow = suggest_supercell(_SHEARED_SLOW_HOST, primitive_atom_count=2)

    # The editor's scaling boxes stop at 64.
    assert all(max(c.scaling) <= 64 for c in (*flat.candidates, *slow.candidates))
    assert _recommended(flat).satisfies_criterion
    # The search past the budget stops at five times it.
    assert all(c.atom_count <= 5 * MAX_SUPERCELL_ATOMS for c in slow.candidates)


def test_best_affordable_cell_is_listed_when_it_is_closer_than_the_display_threshold() -> None:
    # 84 atoms: no cell under 200 atoms reaches 8 Å, let alone 10 Å.
    result = suggest_supercell(
        [[4.9, 0.0, 0.0], [0.0, 7.95, 0.0], [0.0, 0.0, 9.9]], primitive_atom_count=84
    )

    rec = _recommended(result)
    assert rec.scaling == (2, 1, 1)
    assert rec.atom_count == 168
    assert rec.min_image_distance == pytest.approx(7.95)
    assert not rec.satisfies_criterion


def test_a_cell_no_farther_apart_than_a_cheaper_one_but_for_roundoff_is_not_listed() -> None:
    # (1, 15, 5) and (1, 18, 5) are 9.8955 Å apart alike; roundoff made the larger
    # one 2e-15 Å farther, so it was listed and recommended at 180 atoms.
    result = suggest_supercell(
        [[2.4, 9.6, 0.0], [0.0, 2.4, 0.0], [0.0, 0.0, 2.4]], primitive_atom_count=2
    )

    assert all(c.scaling != (1, 18, 5) for c in result.candidates)
    assert _recommended(result).scaling == (1, 15, 5)


@pytest.mark.parametrize(
    ("lattice", "atoms"), _SUGGESTION_CELLS.values(), ids=_SUGGESTION_CELLS.keys()
)
def test_every_listed_cell_is_farther_apart_than_every_cheaper_one(lattice, atoms) -> None:
    result = suggest_supercell(lattice, primitive_atom_count=atoms)

    counts = [c.atom_count for c in result.candidates]
    distances = [c.min_image_distance for c in result.candidates]
    assert counts == sorted(counts)
    assert all(later > earlier * (1 + 1e-9) for earlier, later in zip(distances, distances[1:]))


@pytest.mark.parametrize(
    ("lattice", "atoms"), _SUGGESTION_CELLS.values(), ids=_SUGGESTION_CELLS.keys()
)
def test_image_distance_never_grows_and_the_recommendation_never_shrinks(lattice, atoms) -> None:
    for n1 in range(1, 6):
        for n2 in range(1, 6):
            for n3 in range(1, 6):
                vectors = _scaled(lattice, (n1, n2, n3))
                assert suggestion_module.shortest_translation_length(vectors) <= (
                    _adjacent_cell_search(vectors) * (1 + 1e-12)
                )

    old = _recommended(_adjacent_cell_suggestion(lattice, atoms))
    new = _recommended(suggest_supercell(lattice, primitive_atom_count=atoms))

    assert new.atom_count >= old.atom_count


@pytest.mark.parametrize(
    ("lattice", "atoms", "scalings", "recommended"),
    _ORDINARY_CELLS.values(),
    ids=_ORDINARY_CELLS.keys(),
)
def test_ordinary_cells_keep_their_candidates_and_recommendation(
    lattice, atoms, scalings, recommended
) -> None:
    old = _adjacent_cell_suggestion(lattice, atoms)
    new = suggest_supercell(lattice, primitive_atom_count=atoms)

    assert [c.scaling for c in new.candidates] == scalings
    assert _recommended(new).scaling == recommended
    assert [c.min_image_distance for c in new.candidates] == pytest.approx(
        [c.min_image_distance for c in old.candidates], rel=1e-12
    )

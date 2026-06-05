"""Tests for the supercell suggestion algorithm."""

from __future__ import annotations

import math

import pytest

from mufinder.science.supercell_suggestion import (
    DISPLAY_DISTANCE_ANGSTROM,
    MAX_SUPERCELL_ATOMS,
    MIN_IMAGE_DISTANCE_ANGSTROM,
    SupercellSuggestionError,
    _min_image_distance,
    suggest_supercell,
)

import numpy as np


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
# Unit test for the 27-point primitive
# ---------------------------------------------------------------------------


def test_min_image_distance_cubic_equals_axis() -> None:
    sv = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    assert _min_image_distance(sv) == pytest.approx(5.0)


def test_min_image_distance_diagonal_shorter_than_axes() -> None:
    # Oblique cell: a1=[5,0,0], a2=[4,3,0] → (1,-1,0) = [1,-3,0] length √10 ≈ 3.16
    sv = np.array([[5.0, 0, 0], [4.0, 3.0, 0], [0, 0, 5.0]])
    d = _min_image_distance(sv)
    axis_min = min(math.sqrt(sum(c**2 for c in row)) for row in sv)
    assert d < axis_min
    assert d == pytest.approx(math.sqrt(10), rel=1e-9)


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
    for c in result.candidates[idx + 1:]:
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

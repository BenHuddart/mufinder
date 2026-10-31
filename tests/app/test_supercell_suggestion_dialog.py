"""Tests for the supercell suggestion dialog."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel

from mufinder.app.supercell_suggestion_dialog import SupercellSuggestionDialog
from mufinder.science.supercell_suggestion import SupercellCandidate, SupercellSuggestion


def _cand(
    scaling: tuple[int, int, int],
    atom_count: int,
    min_image_distance: float,
    satisfies: bool,
) -> SupercellCandidate:
    return SupercellCandidate(
        scaling=scaling,
        atom_count=atom_count,
        min_image_distance=min_image_distance,
        satisfies_criterion=satisfies,
    )


def _suggestion(
    candidates: list[SupercellCandidate],
    recommended_index: int = 0,
    primitive_atom_count: int = 1,
    primitive_exceeds_budget: bool = False,
    distance_target: float = 10.0,
    display_threshold: float = 8.0,
    max_atoms: int = 200,
) -> SupercellSuggestion:
    return SupercellSuggestion(
        candidates=tuple(candidates),
        recommended_index=recommended_index,
        primitive_atom_count=primitive_atom_count,
        distance_target=distance_target,
        display_threshold=display_threshold,
        max_atoms=max_atoms,
        primitive_exceeds_budget=primitive_exceeds_budget,
    )


def _label_texts(dialog: SupercellSuggestionDialog) -> list[str]:
    """Return the texts of the labels directly in the dialog's layout, top to bottom."""
    layout = dialog.layout()
    widgets = [layout.itemAt(index).widget() for index in range(layout.count())]
    return [widget.text() for widget in widgets if isinstance(widget, QLabel)]


# Three-candidate suggestion: sub-criterion, recommended (criterion), expensive.
def _three_candidate_suggestion() -> SupercellSuggestion:
    return _suggestion(
        candidates=[
            _cand((3, 3, 3), 27, 9.0, False),
            _cand((4, 4, 4), 64, 12.0, True),  # recommended (satisfies criterion, < 200)
            _cand((5, 5, 5), 125, 15.0, True),
        ],
        recommended_index=1,
    )


# Two-candidate suggestion where criterion is over budget: affordable is recommended.
def _affordable_recommended_suggestion() -> SupercellSuggestion:
    return _suggestion(
        candidates=[
            _cand((3, 3, 3), 27, 9.0, False),  # recommended (best affordable, not criterion)
            _cand((10, 10, 10), 1000, 10.0, True),
        ],
        recommended_index=0,
    )


def _single_suggestion() -> SupercellSuggestion:
    return _suggestion(
        candidates=[_cand((1, 1, 1), 200, 4.0, False)],
        recommended_index=0,
        primitive_atom_count=200,
        primitive_exceeds_budget=True,
    )


# ---------------------------------------------------------------------------
# Table structure
# ---------------------------------------------------------------------------


def test_table_has_correct_row_count(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    assert dialog._table.rowCount() == 3
    dialog.close()


def test_table_has_three_columns_with_no_blank_header(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    headers = [
        dialog._table.horizontalHeaderItem(column).text()
        for column in range(dialog._table.columnCount())
    ]
    assert headers == ["Scaling", "Atoms", "Min image (Å)"]
    dialog.close()


def test_table_marks_recommended_row_in_scaling_cell(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    assert dialog._table.item(1, dialog._SCALING_COLUMN).text() == "4 × 4 × 4 (Recommended)"
    assert dialog._table.item(0, dialog._SCALING_COLUMN).text() == "3 × 3 × 3"
    assert dialog._table.item(2, dialog._SCALING_COLUMN).text() == "5 × 5 × 5"
    dialog.close()


def test_table_ticks_distances_that_reach_the_target(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    assert dialog._table.item(0, dialog._DISTANCE_COLUMN).text() == "9.00 Å"
    assert dialog._table.item(1, dialog._DISTANCE_COLUMN).text() == "12.00 Å ✓"
    assert dialog._table.item(2, dialog._DISTANCE_COLUMN).text() == "15.00 Å ✓"
    dialog.close()


def test_distance_header_explains_the_tick(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    tooltip = dialog._table.horizontalHeaderItem(dialog._DISTANCE_COLUMN).toolTip()
    assert "tick" in tooltip
    assert "10 Å" in tooltip
    dialog.close()


def test_recommended_row_preselected(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    selected = dialog._table.selectedItems()
    assert selected
    assert selected[0].row() == 1  # recommended_index == 1
    dialog.close()


# ---------------------------------------------------------------------------
# Accept: selected row determines chosen scaling
# ---------------------------------------------------------------------------


def test_accept_recommended_row_returns_its_scaling(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    dialog._table.selectRow(1)
    dialog._accept_if_valid()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.chosen_scaling() == (4, 4, 4)
    dialog.close()


def test_accept_first_row_returns_its_scaling(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    dialog._table.selectRow(0)
    dialog._accept_if_valid()
    assert dialog.chosen_scaling() == (3, 3, 3)
    dialog.close()


def test_accept_last_row_returns_its_scaling(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    dialog._table.selectRow(2)
    dialog._accept_if_valid()
    assert dialog.chosen_scaling() == (5, 5, 5)
    dialog.close()


def test_chosen_matrix_is_diagonal(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog._table is not None
    dialog._table.selectRow(1)
    dialog._accept_if_valid()
    assert dialog.chosen_matrix() == [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
    dialog.close()


def test_chosen_matrix_none_before_accept(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert dialog.chosen_matrix() is None
    dialog.close()


# ---------------------------------------------------------------------------
# Affordable-recommended case
# ---------------------------------------------------------------------------


def test_affordable_recommended_row_marked(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_affordable_recommended_suggestion())
    assert dialog._table is not None
    assert "(Recommended)" in dialog._table.item(0, dialog._SCALING_COLUMN).text()
    assert "(Recommended)" not in dialog._table.item(1, dialog._SCALING_COLUMN).text()
    dialog.close()


def test_affordable_recommended_row_preselected(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_affordable_recommended_suggestion())
    assert dialog._table is not None
    selected = dialog._table.selectedItems()
    assert selected[0].row() == 0
    dialog.close()


# ---------------------------------------------------------------------------
# Primitive-exceeds-budget case
# ---------------------------------------------------------------------------


def test_primitive_exceeds_budget_table_has_one_row(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_single_suggestion())
    assert dialog._table is not None
    assert dialog._table.rowCount() == 1
    dialog.close()


def test_primitive_exceeds_budget_shows_only_the_status_line(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_single_suggestion())
    assert _label_texts(dialog) == [
        "The primitive cell already has 200 atoms, so no search under 200 atoms was possible. "
        "This calculation will be expensive. It does not reach 10 Å."
    ]
    dialog.close()


def test_primitive_exceeds_budget_status_omits_target_when_reached(qapp) -> None:
    suggestion = _suggestion(
        candidates=[_cand((1, 1, 1), 240, 12.0, True)],
        primitive_atom_count=240,
        primitive_exceeds_budget=True,
    )
    dialog = SupercellSuggestionDialog(suggestion=suggestion)
    assert _label_texts(dialog) == [
        "The primitive cell already has 240 atoms, so no search under 200 atoms was possible. "
        "This calculation will be expensive."
    ]
    dialog.close()


# ---------------------------------------------------------------------------
# Status line under the table
# ---------------------------------------------------------------------------


def test_no_status_line_when_recommendation_meets_target_within_budget(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert len(_label_texts(dialog)) == 1  # the intro alone
    dialog.close()


def test_status_line_when_no_cell_under_budget_reaches_target(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_affordable_recommended_suggestion())
    assert _label_texts(dialog)[1] == (
        "No cell under 200 atoms reaches 10 Å. The recommended cell is the best one under "
        "that limit."
    )
    dialog.close()


def test_status_line_when_recommended_cell_is_over_budget(qapp) -> None:
    suggestion = _suggestion(
        candidates=[_cand((6, 6, 6), 432, 12.0, True)],
        recommended_index=0,
        primitive_atom_count=2,
    )
    dialog = SupercellSuggestionDialog(suggestion=suggestion)
    assert _label_texts(dialog)[1] == (
        "No cell under 200 atoms reaches 10 Å. The recommended cell is the smallest that does, "
        "and its 432 atoms may be expensive."
    )
    dialog.close()


# ---------------------------------------------------------------------------
# Reject / cancel
# ---------------------------------------------------------------------------


def test_reject_gives_rejected_result(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    dialog.reject()
    assert dialog.result() == QDialog.DialogCode.Rejected
    dialog.close()


def test_chosen_scaling_none_after_reject(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    dialog.reject()
    assert dialog.chosen_scaling() is None
    dialog.close()


# ---------------------------------------------------------------------------
# Intro text comes from the suggestion
# ---------------------------------------------------------------------------


def test_intro_is_two_sentences_with_the_default_thresholds(qapp) -> None:
    dialog = SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())
    assert _label_texts(dialog)[0] == (
        "Diagonal supercells whose periodic muon images are at least 8 Å apart. The recommended "
        "cell is the cheapest one that reaches 10 Å (✓) with fewer than 200 atoms."
    )
    dialog.close()


def test_intro_and_status_follow_the_suggestion_thresholds(qapp) -> None:
    suggestion = _suggestion(
        candidates=[_cand((3, 3, 3), 27, 9.0, False)],
        distance_target=12.5,
        display_threshold=6.0,
        max_atoms=500,
    )
    dialog = SupercellSuggestionDialog(suggestion=suggestion)
    intro, status = _label_texts(dialog)
    assert "at least 6 Å apart" in intro
    assert "reaches 12.5 Å (✓) with fewer than 500 atoms" in intro
    assert status.startswith("No cell under 500 atoms reaches 12.5 Å.")
    dialog.close()

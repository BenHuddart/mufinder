"""Tests for the parameter Info dialog built from keyword metadata."""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QLabel, QTextEdit

from mufinder.app.help_dialog import ParameterHelpDialog, exclusive_sentence
from mufinder.castep_io.schema import KeywordSpec, get_keyword_spec

# Catalog internals the Info dialog never shows.
_INTERNAL_PREFIXES = (
    "File:",
    "Type:",
    "CASTEP Type:",
    "Category:",
    "Exclusive group:",
    "Level:",
    "Aliases:",
    "Allowed tasks:",
)


def _summary(dialog: ParameterHelpDialog) -> list[str]:
    labels = dialog.findChildren(QLabel)
    # The first label is the title; the second holds the summary lines.
    return labels[1].text().splitlines()


def _details(dialog: ParameterHelpDialog) -> str:
    body = dialog.findChild(QTextEdit)
    assert body is not None
    return body.toPlainText()


def _assert_no_internals(lines: list[str]) -> None:
    for line in lines:
        assert not line.startswith(_INTERNAL_PREFIXES), line


def test_info_for_a_scalar_shows_user_fields_once(qapp) -> None:
    spec = KeywordSpec(
        name="smearing_width",
        file="param",
        kind="scalar",
        value_type="positive_float_with_optional_unit",
        option_type="Physical",
        level="Basic",
        unit="eV",
        example="SMEARING_WIDTH : 0.1 eV",
        notes="MuFinder checks only that the value is a single positive number.",
        label="Smearing Width",
        description="Electronic smearing width.",
        help_text="MuFinder checks only that the value is a single positive number.",
        category="scf",
        default_value=0.2,
        aliases=("smearing_widths",),
    )

    dialog = ParameterHelpDialog(spec)

    assert dialog.windowTitle() == "Smearing Width"
    lines = _summary(dialog)
    assert lines == [
        "Electronic smearing width.",
        "CASTEP keyword: SMEARING_WIDTH",
        "Unit: eV",
        "Default: 0.2",
    ]
    _assert_no_internals(lines)
    details = _details(dialog)
    assert details == (
        "Example:\nSMEARING_WIDTH : 0.1 eV\n\n"
        "Notes:\nMuFinder checks only that the value is a single positive number."
    )
    dialog.close()


def test_info_shows_castep_text_that_adds_to_the_description(qapp) -> None:
    spec = KeywordSpec(
        name="iprint",
        file="param",
        kind="scalar",
        value_type="int",
        label="Iprint",
        description="Controls the verbosity.",
        help_text="Controls the verbosity.\nAllowed values: 0, 1, 2 or 3",
        default_value=1,
    )

    dialog = ParameterHelpDialog(spec)

    # The first sentence is the one-line description; the details keep the rest.
    assert _details(dialog) == "Allowed values: 0, 1, 2 or 3"
    dialog.close()


def test_info_for_a_block_reads_plainly(qapp) -> None:
    spec = get_keyword_spec("cell", "hubbard_u")
    assert spec is not None

    dialog = ParameterHelpDialog(spec)

    lines = _summary(dialog)
    assert lines[0] == spec.description
    assert "CASTEP keyword: HUBBARD_U" in lines
    _assert_no_internals(lines)
    details = _details(dialog)
    assert "%BLOCK HUBBARD_U" in details
    assert details.count("Notes:") == 1
    # Backticks mark keywords in the catalog; Info shows plain text.
    assert "`" not in details
    dialog.close()


def test_info_for_an_exclusive_keyword_names_the_alternative(qapp) -> None:
    cutoff = get_keyword_spec("param", "cut_off_energy")
    precision = get_keyword_spec("param", "basis_precision")
    assert cutoff is not None and precision is not None

    cutoff_dialog = ParameterHelpDialog(cutoff)
    precision_dialog = ParameterHelpDialog(precision)

    cutoff_lines = _summary(cutoff_dialog)
    assert f"Use either this or {precision.label}." in cutoff_lines
    assert "Unit: eV" in cutoff_lines
    _assert_no_internals(cutoff_lines)
    assert f"Use either this or {cutoff.label}." in _summary(precision_dialog)
    # The catalog note says what happens in plain words, once.
    cutoff_details = _details(cutoff_dialog)
    assert cutoff_details.count("Overrides BASIS_PRECISION when both are set.") == 1
    assert "Direct CASTEP docs" not in cutoff_details
    cutoff_dialog.close()
    precision_dialog.close()


def test_info_for_a_three_way_choice_lists_every_alternative(qapp) -> None:
    grid = get_keyword_spec("cell", "kpoint_mp_grid")
    assert grid is not None

    dialog = ParameterHelpDialog(grid)

    lines = _summary(dialog)
    choice = [line for line in lines if line.startswith("Use only one of this, ")]
    assert len(choice) == 1
    assert get_keyword_spec("cell", "kpoint_mp_spacing").label in choice[0]
    assert get_keyword_spec("cell", "kpoints_list").label in choice[0]
    dialog.close()


@pytest.mark.parametrize(
    ("partners", "expected"),
    [
        ((), ""),
        (("Basis Precision",), "Use either this or Basis Precision."),
        (("A", "B"), "Use only one of this, A or B."),
        (("A", "B", "C"), "Use only one of this, A, B or C."),
    ],
)
def test_exclusive_sentence_wording(partners, expected) -> None:
    assert exclusive_sentence(partners) == expected


def test_info_without_details_hides_the_details_box(qapp) -> None:
    spec = KeywordSpec(
        name="charge",
        file="param",
        kind="scalar",
        value_type="int",
        label="Charge",
        description="Total system charge.",
        help_text="Total system charge.",
        notes="Total system charge.",
    )

    dialog = ParameterHelpDialog(spec)

    body = dialog.findChild(QTextEdit)
    assert body is not None
    assert body.toPlainText() == ""
    assert body.isHidden()
    dialog.close()


def test_info_does_not_repeat_the_description_in_the_notes(qapp) -> None:
    notes = "Fix the centre of mass.\nEnergy-barrier workflows emit FALSE."
    spec = KeywordSpec(
        name="fix_com",
        file="cell",
        kind="scalar",
        value_type="bool",
        label="Fix Com",
        description="Fix the centre of mass.",
        help_text=notes,
        notes=notes,
    )

    dialog = ParameterHelpDialog(spec)

    assert _details(dialog) == "Notes:\nEnergy-barrier workflows emit FALSE."
    dialog.close()


def test_info_drops_a_description_that_castep_wrapped_over_two_lines(qapp) -> None:
    spec = KeywordSpec(
        name="test_wrapped",
        file="param",
        kind="scalar",
        value_type="int",
        label="Test Wrapped",
        description="Determines the cut-off energy for the basis.",
        help_text="Determines the cut-off\nenergy for the basis.\nModifiable: never",
    )

    dialog = ParameterHelpDialog(spec)

    assert _details(dialog) == "Modifiable: never"
    dialog.close()

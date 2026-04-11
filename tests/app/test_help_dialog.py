"""Tests for parameter help dialog rendering from keyword metadata."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTextEdit

from mufinder.app.help_dialog import ParameterHelpDialog
from mufinder.castep_io.schema import KeywordSpec


def test_parameter_help_dialog_renders_metadata(qapp) -> None:
    spec = KeywordSpec(
        name="cut_off_energy",
        file="param",
        kind="scalar",
        value_type="positive_float_with_optional_unit",
        unit="eV",
        allowed_values=("340", "360"),
        example="CUT_OFF_ENERGY : 340 eV",
        notes="Direct CASTEP docs say this overrides BASIS_PRECISION.",
        label="Cutoff Energy",
        description="Plane-wave cutoff energy in eV.",
        help_text="Sets the plane-wave basis cutoff.",
        category="electronic",
        default_value=340.0,
        aliases=("basis_cutoff",),
        option_type="Physical",
        level="Basic",
        allowed_tasks=("SINGLEPOINT", "GEOMETRYOPTIMIZATION"),
        exclusive_group="basis_definition",
    )

    dialog = ParameterHelpDialog(spec)

    assert dialog.windowTitle() == "Cutoff Energy"

    labels = [item.text() for item in dialog.findChildren(QLabel)]
    combined = "\n".join(labels)
    assert "Cutoff Energy" in combined
    assert "Keyword: CUT_OFF_ENERGY" in combined
    assert "Aliases: BASIS_CUTOFF" in combined
    assert "File: param" in combined
    assert "Type: positive_float_with_optional_unit" in combined
    assert "CASTEP Type: Physical" in combined
    assert "Level: Basic" in combined
    assert "Unit: eV" in combined
    assert "Default: 340.0" in combined
    assert "Allowed values: 340, 360" in combined
    assert "Category: electronic" in combined
    assert "Allowed tasks: SINGLEPOINT, GEOMETRYOPTIMIZATION" in combined
    assert "Exclusive group: basis_definition" in combined

    body = dialog.findChild(QTextEdit)
    assert body is not None
    text = body.toPlainText()
    assert "Sets the plane-wave basis cutoff." in text
    assert "Example:" in text
    assert "CUT_OFF_ENERGY : 340 eV" in text
    assert "Notes:" in text

    dialog.close()

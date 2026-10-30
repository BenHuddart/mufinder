"""Tests for the validated decimal line edit and the vector input built from it."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QValidator

from mufinder.app.decimal_input import DecimalLineEdit, VectorInput


def test_decimal_line_edit_snaps_an_out_of_range_value_when_editing_finishes(qapp) -> None:
    """The field must show the value readers get, not a number below the minimum."""

    field = DecimalLineEdit()
    field.setRange(0.001, 1.0)
    field.setDecimals(6)
    field.setText("0.0001")

    assert field.value() == 0.001
    assert field.text() == "0.0001"

    field.editingFinished.emit()

    assert field.text() == "0.001"
    assert field.value() == 0.001


def test_decimal_line_edit_leaves_in_range_and_blank_text_alone(qapp) -> None:
    field = DecimalLineEdit()
    field.setRange(0.001, 1.0)
    field.setText("0.25")
    field.editingFinished.emit()
    assert field.text() == "0.25"

    field.setText("")
    field.editingFinished.emit()
    assert field.text() == ""


def test_decimal_line_edit_set_up_by_keywords_keeps_the_given_text(qapp) -> None:
    field = DecimalLineEdit(
        text="1.0", minimum=0.0, maximum=1000.0, decimals=6, placeholder="amplitude"
    )

    assert field.text() == "1.0"
    assert field.value() == 1.0
    assert field.placeholderText() == "amplitude"
    assert (field.minimum(), field.maximum()) == (0.0, 1000.0)
    validator = field.validator()
    assert (validator.bottom(), validator.top(), validator.decimals()) == (0.0, 1000.0, 6)
    assert DecimalLineEdit(text="").text() == ""


def test_decimal_line_edit_standard_notation_refuses_exponents(qapp) -> None:
    standard = DecimalLineEdit(standard_notation=True)
    scientific = DecimalLineEdit()

    assert standard.validator().validate("1e-3", 0)[0] == QValidator.State.Invalid
    assert scientific.validator().validate("1e-3", 0)[0] == QValidator.State.Acceptable


def test_decimal_line_edit_range_change_can_keep_the_typed_text(qapp) -> None:
    field = DecimalLineEdit(text="0.50", minimum=-1.0, maximum=1.0)

    field.setRange(-0.25, 0.25, keep_text=True)

    assert field.text() == "0.50"
    assert (field.minimum(), field.maximum()) == (-0.25, 0.25)
    assert field.validator().top() == 0.25
    field.setRange(-0.25, 0.25)
    assert field.text() == "0.25"


def test_vector_input_reads_empty_boxes_as_zero_and_names_a_bad_component(qapp) -> None:
    vector = VectorInput((0.0, 0.0, 1.0), name="Applied Field", tooltip="Field, T")
    changes: list[bool] = []
    vector.textChanged.connect(lambda: changes.append(True))

    assert [edit.text() for edit in vector.edits] == ["0", "0", "1"]
    assert [edit.accessibleName() for edit in vector.edits] == [
        "Applied Field x",
        "Applied Field y",
        "Applied Field z",
    ]
    assert vector.edits[1].toolTip() == "Field, T (y)"
    x, y, z = vector.edits
    x.setText("")
    y.setText("2.5")
    assert changes == [True, True]
    assert vector.value("applied field") == (0.0, 2.5, 1.0)

    z.setText("-")
    with pytest.raises(ValueError, match="applied field components must be decimal numbers"):
        vector.value("applied field")
    assert vector.value_or_zero() == (0.0, 2.5, 0.0)
    z.setText("inf")
    with pytest.raises(ValueError, match="applied field components must be finite"):
        vector.value("applied field")
    vector.close()


def test_vector_input_shows_zeros_for_a_malformed_saved_vector(qapp) -> None:
    vector = VectorInput(name="Moment", text="0.0", tooltip=lambda axis: f"{axis} part")

    assert [edit.text() for edit in vector.edits] == ["0.0", "0.0", "0.0"]
    assert vector.edits[2].toolTip() == "z part"
    vector.set_value([1, "2", 3.5])
    assert vector.value("moment") == (1.0, 2.0, 3.5)
    for malformed in (None, "abc", [1.0, 2.0], ["a", 1.0, 2.0], [float("nan"), 0.0, 0.0]):
        vector.set_value((1.0, 1.0, 1.0))
        vector.set_value(malformed)
        assert vector.value("moment") == (0.0, 0.0, 0.0)
    vector.close()


def test_vector_input_boxes_keep_the_range_they_are_given(qapp) -> None:
    vector = VectorInput(name="Field", minimum=-1_000.0, maximum=1_000.0, decimals=6, box_width=80)

    for edit in vector.edits:
        assert (edit.minimum(), edit.maximum()) == (-1_000.0, 1_000.0)
        assert edit.validator().decimals() == 6
        assert edit.maximumWidth() == 80
    vector.setEnabled(False)
    assert not any(edit.isEnabled() for edit in vector.edits)
    vector.close()

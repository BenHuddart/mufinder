"""Tests for the validated decimal line edit."""

from __future__ import annotations

from PySide6.QtGui import QValidator

from mufinder.app.decimal_input import DecimalLineEdit


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

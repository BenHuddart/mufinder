"""Tests for the validated decimal line edit."""

from __future__ import annotations

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

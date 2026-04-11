"""Tests for typed CASTEP parameter editor widgets."""

from __future__ import annotations

from mufinder.app.param_widgets import create_param_editor
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import BlockSection, ValueWithUnit


def test_create_param_editor_for_enum_and_roundtrip(qapp) -> None:
    spec = KeywordSpec(
        name="task",
        file="param",
        kind="scalar",
        value_type="enum",
        allowed_values=("SINGLEPOINT", "GEOMETRYOPTIMIZATION"),
        widget_hint="dropdown",
        default_value="SINGLEPOINT",
    )

    editor = create_param_editor(spec)
    assert editor.value() == "SINGLEPOINT"
    editor.set_value("GEOMETRYOPTIMIZATION")
    assert editor.value() == "GEOMETRYOPTIMIZATION"


def test_create_param_editor_for_unit_float(qapp) -> None:
    spec = KeywordSpec(
        name="cut_off_energy",
        file="param",
        kind="scalar",
        value_type="positive_float_with_optional_unit",
        unit="eV",
        widget_hint="unit_float",
        default_value=300.0,
    )

    editor = create_param_editor(spec)
    assert isinstance(editor.value(), ValueWithUnit)

    editor.set_value(ValueWithUnit(450.0, "eV"))
    value = editor.value()
    assert isinstance(value, ValueWithUnit)
    assert value.value == 450.0
    assert value.unit == "eV"


def test_create_param_editor_for_int_triplet(qapp) -> None:
    spec = KeywordSpec(
        name="kpoint_mp_grid",
        file="cell",
        kind="scalar",
        value_type="int_triplet",
        widget_hint="int_triplet",
        default_value=(4, 4, 4),
    )

    editor = create_param_editor(spec)
    assert editor.value() == (4, 4, 4)
    editor.set_value((6, 8, 10))
    assert editor.value() == (6, 8, 10)


def test_create_param_editor_for_block_roundtrip(qapp) -> None:
    spec = KeywordSpec(
        name="species_pot",
        file="cell",
        kind="block",
        value_type="block",
        widget_hint="block",
    )

    editor = create_param_editor(spec)
    value = BlockSection(name="species_pot", rows=(("Si", "Si_00PBE.usp"),))
    editor.set_value(value)
    assert editor.value() == value


def test_create_param_editor_ignores_malformed_real_vector_default(qapp) -> None:
    spec = KeywordSpec(
        name="quantization_axis",
        file="cell",
        kind="scalar",
        value_type="real_vector",
        widget_hint="real_vector",
        default_value="D",
    )

    editor = create_param_editor(spec)

    assert editor.value() == (0.0, 0.0, 0.0)

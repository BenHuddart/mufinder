"""Tests for typed CASTEP parameter editor widgets."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from mufinder.app.hubbard_u_editor import (
    HubbardUDialog,
    HubbardUEntry,
    hubbard_block_from_entries,
    hubbard_entries_from_block,
    summarize_hubbard_u,
)
from mufinder.app.param_widgets import HubbardUParamEditor, create_param_editor
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


def test_create_param_editor_for_hubbard_u_roundtrip(qapp) -> None:
    spec = KeywordSpec(
        name="hubbard_u",
        file="cell",
        kind="block",
        value_type="hubbard_u_block",
        widget_hint="hubbard_u",
    )

    editor = create_param_editor(spec, species_options=("La", "Mn", "O"))
    value = BlockSection(name="hubbard_u", rows=(("Ni", "d:", 2.4),), unit="eV")
    editor.set_value(value)

    assert isinstance(editor, HubbardUParamEditor)
    assert editor._species_options == ("La", "Mn", "O")
    assert editor.value() == value
    assert summarize_hubbard_u(value) == "1 entry (eV)"


def test_hubbard_u_helpers_write_species_orbital_rows(qapp) -> None:
    block = hubbard_block_from_entries(
        (
            HubbardUEntry(species="Sm", orbital="f", value=6.1),
            HubbardUEntry(species="Ni", orbital="d", value=2.4),
            HubbardUEntry(species="U", orbital="d", value=1.2),
            HubbardUEntry(species="U", orbital="f", value=2.1),
        ),
        unit="eV",
    )

    assert block == BlockSection(
        name="hubbard_u",
        rows=(
            ("Sm", "f:", 6.1),
            ("Ni", "d:", 2.4),
            ("U", "d:", 1.2),
            ("U", "f:", 2.1),
        ),
        unit="eV",
    )
    assert hubbard_entries_from_block(block) == (
        HubbardUEntry(species="Sm", orbital="f", value=6.1),
        HubbardUEntry(species="Ni", orbital="d", value=2.4),
        HubbardUEntry(species="U", orbital="d", value=1.2),
        HubbardUEntry(species="U", orbital="f", value=2.1),
    )


def test_hubbard_u_parser_expands_legacy_atom_specific_rows(qapp) -> None:
    legacy = BlockSection(name="hubbard_u", rows=(("U", 2, "d:", 1.2, "f:", 2.1),), unit="eV")

    assert hubbard_entries_from_block(legacy) == (
        HubbardUEntry(species="U", orbital="d", value=1.2),
        HubbardUEntry(species="U", orbital="f", value=2.1),
    )


def test_hubbard_u_dialog_uses_species_and_orbital_dropdowns(qapp) -> None:
    dialog = HubbardUDialog(species_options=("La", "Mn", "O"))

    species = dialog._table.cellWidget(0, dialog._SPECIES_COLUMN)
    orbital = dialog._table.cellWidget(0, dialog._ORBITAL_COLUMN)

    assert isinstance(species, QComboBox)
    assert [species.itemText(index) for index in range(species.count())] == ["La", "Mn", "O"]
    assert not species.isEditable()
    assert isinstance(orbital, QComboBox)
    assert [orbital.itemText(index) for index in range(orbital.count())] == ["s", "p", "d", "f"]
    assert orbital.currentText() == "d"


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

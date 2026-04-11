"""Focused tests for CASTEP validator and GUI helper branches."""

from __future__ import annotations

import pytest

import mufinder.castep_io.schema as schema_module
from mufinder.castep_io.gui_metadata import (
    default_assignments,
    gui_value_is_active,
    make_block_section,
    make_keyword_assignment,
    normalize_gui_value,
    validate_gui_values,
)
from mufinder.castep_io.schema import get_keyword_spec
from mufinder.castep_io.types import BlockSection, KeywordAssignment, ValueWithUnit
from mufinder.castep_io.validators import CastepValidationError, UnknownKeywordError, validate_keyword_value


@pytest.fixture(autouse=True)
def fallback_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    original_sources = schema_module._candidate_catalog_sources
    monkeypatch.setattr(schema_module, "_candidate_catalog_sources", lambda: ())
    schema_module.refresh_keyword_catalog(allow_fallback=True)
    yield
    monkeypatch.setattr(schema_module, "_candidate_catalog_sources", original_sources)
    schema_module.refresh_keyword_catalog(allow_fallback=True)


def test_validate_keyword_value_covers_scalar_and_error_branches() -> None:
    with pytest.raises(UnknownKeywordError):
        validate_keyword_value("param", "missing_keyword", 1)

    validate_keyword_value("cell", "symmetry_generate", False)

    with pytest.raises(CastepValidationError, match="expects TRUE or FALSE"):
        validate_keyword_value("param", "spin_polarized", "TRUE")

    with pytest.raises(CastepValidationError, match="Allowed values"):
        validate_keyword_value("param", "basis_precision", "INVALID")

    with pytest.raises(CastepValidationError, match="expects a tuple of three integers"):
        validate_keyword_value("cell", "kpoint_mp_grid", "1 1 1")

    with pytest.raises(CastepValidationError, match="greater than zero"):
        validate_keyword_value("param", "max_scf_cycles", 0)

    with pytest.raises(CastepValidationError, match="greater than or equal to zero"):
        validate_keyword_value("param", "geom_max_iter", -1)

    with pytest.raises(CastepValidationError, match="expects a numeric value"):
        validate_keyword_value("param", "spin", "not-a-number")

    with pytest.raises(CastepValidationError, match="expects a scalar numeric value"):
        validate_keyword_value("param", "cut_off_energy", (1, 2, 3))

    with pytest.raises(CastepValidationError, match="units may not be empty"):
        validate_keyword_value("param", "cut_off_energy", ValueWithUnit(300.0, ""))

    with pytest.raises(CastepValidationError, match="not per-atom units"):
        validate_keyword_value("param", "elec_energy_tol", ValueWithUnit(1e-6, "eV/atom"))

    with pytest.raises(CastepValidationError, match="expects exactly 3 components"):
        validate_keyword_value("cell", "kpoint_mp_offset", (0.0, 0.0))

    with pytest.raises(CastepValidationError, match="expects text"):
        validate_keyword_value("cell", "unit_cell", 1.0)

    validate_keyword_value("param", "max_scf_cycles", 100)
    validate_keyword_value("cell", "symmetry_generate", True)
    validate_keyword_value("param", "cut_off_energy", ValueWithUnit(450.0, "eV"))


def test_normalize_gui_value_and_assignment_helpers_cover_branches() -> None:
    block_spec = get_keyword_spec("cell", "species_pot")
    assert block_spec is not None
    block = BlockSection("not_canonical", (("Si", "Si.recpot"),))
    normalized_block = normalize_gui_value(block_spec, block)
    assert isinstance(normalized_block, BlockSection)
    assert normalized_block.name == "species_pot"

    physical_spec = get_keyword_spec("param", "cut_off_energy")
    assert physical_spec is not None
    assert normalize_gui_value(physical_spec, 450.0) == ValueWithUnit(450.0, "eV")
    assert normalize_gui_value(physical_spec, ValueWithUnit(400.0, "eV")) == ValueWithUnit(400.0, "eV")

    triplet_spec = get_keyword_spec("cell", "kpoint_mp_grid")
    assert triplet_spec is not None
    assert normalize_gui_value(triplet_spec, [2, 2, 2]) == (2, 2, 2)
    with pytest.raises(ValueError, match="requires exactly 3 integer components"):
        normalize_gui_value(triplet_spec, [2, 2])

    vector_spec = get_keyword_spec("cell", "kpoint_mp_offset")
    assert vector_spec is not None
    assert normalize_gui_value(vector_spec, [0, 0.5, 0]) == (0.0, 0.5, 0.0)
    with pytest.raises(ValueError, match="requires exactly 3 numeric components"):
        normalize_gui_value(vector_spec, [0, 0.5])

    int_spec = get_keyword_spec("param", "max_scf_cycles")
    number_spec = get_keyword_spec("param", "spin")
    bool_spec = get_keyword_spec("param", "spin_polarized")
    enum_spec = get_keyword_spec("param", "basis_precision")
    assert int_spec is not None and number_spec is not None and bool_spec is not None and enum_spec is not None
    assert normalize_gui_value(int_spec, "12") == 12
    assert normalize_gui_value(number_spec, "1.5") == pytest.approx(1.5)
    assert normalize_gui_value(bool_spec, 1) is True
    assert normalize_gui_value(enum_spec, "fine") == "FINE"

    flag_spec = get_keyword_spec("cell", "symmetry_generate")
    assert flag_spec is not None
    assert make_keyword_assignment(flag_spec, True) == KeywordAssignment(name="symmetry_generate", value=None)
    assert make_keyword_assignment(flag_spec, False) is None
    with pytest.raises(ValueError, match="block keyword"):
        make_keyword_assignment(block_spec, "bad")
    with pytest.raises(ValueError, match="is not a block keyword"):
        make_block_section(enum_spec, "FINE")
    assert make_block_section(block_spec, block) == normalized_block


def test_gui_value_helpers_cover_activity_validation_and_defaults() -> None:
    flag_spec = get_keyword_spec("cell", "symmetry_generate")
    bool_spec = get_keyword_spec("param", "spin_polarized")
    enum_spec = get_keyword_spec("param", "basis_precision")
    requires_spec = get_keyword_spec("param", "geom_force_tol")
    conflict_spec = get_keyword_spec("param", "cut_off_energy")
    assert flag_spec is not None
    assert bool_spec is not None
    assert enum_spec is not None
    assert requires_spec is not None
    assert conflict_spec is not None

    assert gui_value_is_active(flag_spec, True) is True
    assert gui_value_is_active(flag_spec, False) is False
    assert gui_value_is_active(bool_spec, False) is False
    assert gui_value_is_active(enum_spec, "") is False

    specs_by_name = {
        spec.name: spec
        for spec in (flag_spec, bool_spec, enum_spec, requires_spec, conflict_spec, get_keyword_spec("param", "basis_precision"))
        if spec is not None
    }
    errors = validate_gui_values(
        {
            "task": "SINGLEPOINT",
            "geom_force_tol": 0.05,
            "cut_off_energy": 340.0,
            "basis_precision": "FINE",
            "symmetry_generate": False,
        },
        specs_by_name,
        task="SINGLEPOINT",
    )
    assert any("only available for task(s)" in error for error in errors)
    assert any("conflicts with" in error for error in errors)

    default_specs = [
        get_keyword_spec("cell", "symmetry_generate"),
        get_keyword_spec("param", "max_scf_cycles"),
        get_keyword_spec("param", "cut_off_energy"),
    ]
    assignments = default_assignments(spec for spec in default_specs if spec is not None)
    names = {assignment.name for assignment in assignments}
    assert {"symmetry_generate", "max_scf_cycles", "cut_off_energy"} <= names

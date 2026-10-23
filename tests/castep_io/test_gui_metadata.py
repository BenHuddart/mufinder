"""Tests for GUI metadata helpers over CASTEP keyword schema."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from mufinder.castep_io.gui_metadata import (
    default_assignments,
    get_gui_keyword_specs,
    gui_value_is_active,
    infer_task,
    keyword_locked_in_workflow,
    keyword_visible_for_task,
    make_block_section,
    make_keyword_assignment,
    normalize_gui_value,
    validate_gui_values,
)
from mufinder.castep_io.schema import get_keyword_spec
from mufinder.castep_io.types import BlockSection, KeywordAssignment, ValueWithUnit
from tests.support.golden_inputs import pinned_starter_catalog


@pytest.fixture
def starter_catalog() -> Iterator[None]:
    """Read keyword specs from the bundled starter catalog, whatever is installed locally."""

    with pinned_starter_catalog():
        yield


def test_get_gui_keyword_specs_respects_basic_vs_advanced() -> None:
    basic_specs = get_gui_keyword_specs(
        file="param",
        workflow_stage="convergence_cutoff",
        include_advanced=False,
    )
    advanced_specs = get_gui_keyword_specs(
        file="param",
        workflow_stage="convergence_cutoff",
        include_advanced=True,
    )

    basic_names = {spec.name for spec in basic_specs}
    advanced_names = {spec.name for spec in advanced_specs}

    assert "task" in basic_names
    assert "max_scf_cycles" in basic_names
    assert "basis_precision" not in basic_names
    assert "basis_precision" in advanced_names


def test_make_keyword_assignment_unitized_float() -> None:
    spec = get_keyword_spec("param", "cut_off_energy")
    assert spec is not None

    assignment = make_keyword_assignment(spec, 340.0)

    assert assignment is not None
    assert assignment.name == "cut_off_energy"
    assert isinstance(assignment.value, ValueWithUnit)
    assert assignment.value.value == 340.0
    assert assignment.value.unit == "eV"


def test_default_assignments_include_declared_defaults() -> None:
    specs = [
        spec
        for spec in get_gui_keyword_specs(
            file="param",
            workflow_stage="convergence_cutoff",
            include_advanced=False,
        )
        if spec.default_value is not None
    ]

    assignments = default_assignments(specs)
    names = {item.name for item in assignments}

    assert "task" in names
    assert "xc_functional" in names


def test_infer_task_from_workflow_stage() -> None:
    assert infer_task("convergence_cutoff") == "SINGLEPOINT"
    assert infer_task("convergence_kpoint") == "SINGLEPOINT"
    assert infer_task("uep") == "SINGLEPOINT"
    assert infer_task("pristine_relaxation") == "GEOMETRYOPTIMIZATION"
    assert infer_task("muon_relaxation") == "GEOMETRYOPTIMIZATION"
    assert infer_task("energy_barriers") == "TRANSITIONSTATESEARCH"


def test_keyword_locked_in_workflow() -> None:
    cut_off = get_keyword_spec("param", "cut_off_energy")
    assert cut_off is not None
    assert keyword_locked_in_workflow(cut_off, workflow_stage="convergence_cutoff") is True
    assert keyword_locked_in_workflow(cut_off, workflow_stage="convergence_kpoint") is False

    formatted_density = get_keyword_spec("param", "write_formatted_density")
    assert formatted_density is not None
    assert keyword_locked_in_workflow(formatted_density, workflow_stage="uep") is True


def test_keyword_visible_for_task_filters_task_specific_keywords() -> None:
    geom_force_tol = get_keyword_spec("param", "geom_force_tol")
    assert geom_force_tol is not None

    assert keyword_visible_for_task(geom_force_tol, task="GEOMETRYOPTIMIZATION") is True
    assert keyword_visible_for_task(geom_force_tol, task="SINGLEPOINT") is False


def test_structure_controlled_blocks_are_not_exposed_in_gui_specs() -> None:
    names = {
        spec.name
        for spec in get_gui_keyword_specs(
            file="cell",
            workflow_stage="convergence_cutoff",
            include_advanced=True,
        )
    }

    assert "lattice_cart" not in names
    assert "lattice_abc" not in names
    assert "positions_frac" not in names
    assert "positions_abs" not in names


def test_xc_functional_spec_exposes_enum_allowed_values() -> None:
    spec = get_keyword_spec("param", "xc_functional")

    assert spec is not None
    assert spec.value_type == "enum"
    assert "PBE" in spec.allowed_values
    assert "LDA" in spec.allowed_values


@pytest.mark.usefixtures("starter_catalog")
def test_normalize_gui_value_coerces_each_value_type_and_builds_assignments() -> None:
    block_spec = get_keyword_spec("cell", "species_pot")
    assert block_spec is not None
    block = BlockSection("not_canonical", (("Si", "Si.recpot"),))
    normalized_block = normalize_gui_value(block_spec, block)
    assert isinstance(normalized_block, BlockSection)
    assert normalized_block.name == "species_pot"

    physical_spec = get_keyword_spec("param", "cut_off_energy")
    assert physical_spec is not None
    assert normalize_gui_value(physical_spec, 450.0) == ValueWithUnit(450.0, "eV")
    assert normalize_gui_value(physical_spec, ValueWithUnit(400.0, "eV")) == ValueWithUnit(
        400.0, "eV"
    )

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
    assert (
        int_spec is not None
        and number_spec is not None
        and bool_spec is not None
        and enum_spec is not None
    )
    assert normalize_gui_value(int_spec, "12") == 12
    assert normalize_gui_value(number_spec, "1.5") == pytest.approx(1.5)
    assert normalize_gui_value(bool_spec, 1) is True
    assert normalize_gui_value(enum_spec, "fine") == "FINE"

    flag_spec = get_keyword_spec("cell", "symmetry_generate")
    assert flag_spec is not None
    assert make_keyword_assignment(flag_spec, True) == KeywordAssignment(
        name="symmetry_generate", value=None
    )
    assert make_keyword_assignment(flag_spec, False) is None
    with pytest.raises(ValueError, match="block keyword"):
        make_keyword_assignment(block_spec, "bad")
    with pytest.raises(ValueError, match="is not a block keyword"):
        make_block_section(enum_spec, "FINE")
    assert make_block_section(block_spec, block) == normalized_block


@pytest.mark.usefixtures("starter_catalog")
def test_gui_values_report_activity_task_and_conflict_errors_and_defaults() -> None:
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
        spec.name: spec for spec in (flag_spec, bool_spec, enum_spec, requires_spec, conflict_spec)
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

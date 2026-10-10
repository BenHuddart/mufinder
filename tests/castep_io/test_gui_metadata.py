"""Tests for GUI metadata helpers over CASTEP keyword schema."""

from __future__ import annotations

from mufinder.castep_io.gui_metadata import (
    default_assignments,
    get_gui_keyword_specs,
    infer_task,
    keyword_locked_in_workflow,
    keyword_visible_for_task,
    make_keyword_assignment,
)
from mufinder.castep_io.schema import get_keyword_spec
from mufinder.castep_io.types import ValueWithUnit


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

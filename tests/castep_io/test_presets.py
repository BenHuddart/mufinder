"""Tests for CASTEP parameter presets."""

from __future__ import annotations

from mufinder.castep_io.presets import geometry_optimization, molecular_dynamics, single_point
from mufinder.castep_io.types import ValueWithUnit


def _keyword_map(param_file) -> dict[str, object]:
    return {assignment.name: assignment.value for assignment in param_file.keywords}


def test_single_point_preset_serializes_optional_inputs() -> None:
    preset = single_point(
        xc_functional="PBE0",
        cut_off_energy=340.0,
        basis_precision="FINE",
        spin_polarized=True,
        max_scf_cycles=120,
    )
    values = _keyword_map(preset)

    assert values["task"] == "SINGLEPOINT"
    assert values["xc_functional"] == "PBE0"
    assert values["basis_precision"] == "FINE"
    assert values["spin_polarized"] is True
    assert values["max_scf_cycles"] == 120
    assert values["cut_off_energy"] == ValueWithUnit(340.0, "eV")


def test_geometry_optimization_uses_default_tolerances_and_unitized_values() -> None:
    preset = geometry_optimization(
        cut_off_energy=ValueWithUnit(420.0, "eV"),
        max_scf_cycles=80,
        geom_max_iter=50,
    )
    values = _keyword_map(preset)

    assert values["task"] == "GEOMETRYOPTIMIZATION"
    assert values["xc_functional"] == "PBE"
    assert values["cut_off_energy"] == ValueWithUnit(420.0, "eV")
    assert values["max_scf_cycles"] == 80
    assert values["geom_energy_tol"] == ValueWithUnit(2e-5, "eV")
    assert values["geom_force_tol"] == ValueWithUnit(0.05, "eV/ang")
    assert values["geom_stress_tol"] == ValueWithUnit(0.1, "GPa")
    assert values["geom_max_iter"] == 50


def test_molecular_dynamics_allows_optional_keywords_to_be_disabled() -> None:
    preset = molecular_dynamics(
        cut_off_energy=500.0,
        md_delta_t=None,
        md_num_iter=None,
        md_temperature=None,
    )
    values = _keyword_map(preset)

    assert values["task"] == "MOLECULARDYNAMICS"
    assert values["xc_functional"] == "PBE"
    assert values["cut_off_energy"] == ValueWithUnit(500.0, "eV")
    assert "md_delta_t" not in values
    assert "md_num_iter" not in values
    assert "md_temperature" not in values


def test_molecular_dynamics_includes_default_md_keywords() -> None:
    preset = molecular_dynamics()
    values = _keyword_map(preset)

    assert values["md_delta_t"] == ValueWithUnit(1.0, "fs")
    assert values["md_num_iter"] == 100
    assert values["md_temperature"] == ValueWithUnit(300.0, "K")

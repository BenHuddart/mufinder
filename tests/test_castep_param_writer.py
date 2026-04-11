"""Tests for CASTEP ``.param`` serialization."""

from __future__ import annotations

from pathlib import Path

from mufinder.castep_io import KeywordAssignment, ParamFile, ValueWithUnit, geometry_optimization, write_param


def test_write_param_serializes_ordered_keywords() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("geom_force_tol", ValueWithUnit(0.05, "eV/ang")),
            KeywordAssignment("task", "GEOMETRYOPTIMIZATION"),
            KeywordAssignment("xc_functional", "PBE"),
            KeywordAssignment("cut_off_energy", ValueWithUnit(340.0, "eV")),
        )
    )

    rendered = write_param(param_file)
    lines = rendered.splitlines()

    assert lines[0] == "TASK : GEOMETRYOPTIMIZATION"
    assert lines[1] == "XC_FUNCTIONAL : PBE"
    assert "CUT_OFF_ENERGY : 340 eV" in rendered
    assert "GEOM_FORCE_TOL : 0.05 eV/ang" in rendered


def test_geometry_optimization_preset_matches_golden_file() -> None:
    param_file = geometry_optimization(
        cut_off_energy=340.0,
        max_scf_cycles=60,
        geom_energy_tol=ValueWithUnit(2e-5, "eV"),
        geom_force_tol=ValueWithUnit(0.05, "eV/ang"),
        geom_stress_tol=ValueWithUnit(0.1, "GPa"),
        geom_max_iter=30,
    )
    param_file.keywords += (KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-6, "eV")),)

    rendered = write_param(param_file)
    golden = Path("tests/golden/si_geom_opt.param").read_text(encoding="utf-8")

    assert rendered == golden


def test_write_param_supports_boolean_spin_keywords() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("task", "SINGLEPOINT"),
            KeywordAssignment("spin_polarized", True),
            KeywordAssignment("smearing_width", ValueWithUnit(0.1, "eV")),
        )
    )

    rendered = write_param(param_file)

    assert "SPIN_POLARIZED : TRUE" in rendered
    assert "SMEARING_WIDTH : 0.1 eV" in rendered


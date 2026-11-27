"""Small, explicit CASTEP parameter presets."""

from __future__ import annotations

from .types import KeywordAssignment, ParamFile, ValueWithUnit


def single_point(
    *,
    xc_functional: str = "PBE",
    cut_off_energy: float | ValueWithUnit | None = None,
    basis_precision: str | None = None,
    spin_polarized: bool | None = None,
    max_scf_cycles: int | None = None,
) -> ParamFile:
    """Build a conservative single-point parameter file preset."""

    keywords = [
        KeywordAssignment("task", "SINGLEPOINT"),
        KeywordAssignment("xc_functional", xc_functional),
    ]
    if basis_precision is not None:
        keywords.append(KeywordAssignment("basis_precision", basis_precision))
    if cut_off_energy is not None:
        keywords.append(_unitized("cut_off_energy", cut_off_energy, "eV"))
    if spin_polarized is not None:
        keywords.append(KeywordAssignment("spin_polarized", spin_polarized))
    if max_scf_cycles is not None:
        keywords.append(KeywordAssignment("max_scf_cycles", max_scf_cycles))
    return ParamFile(keywords=tuple(keywords))


def geometry_optimization(
    *,
    xc_functional: str = "PBE",
    cut_off_energy: float | ValueWithUnit | None = None,
    max_scf_cycles: int | None = None,
    geom_energy_tol: float | ValueWithUnit | None = ValueWithUnit(2e-5, "eV"),
    geom_force_tol: float | ValueWithUnit | None = ValueWithUnit(0.05, "eV/ang"),
    geom_stress_tol: float | ValueWithUnit | None = ValueWithUnit(0.1, "GPa"),
    geom_max_iter: int | None = None,
) -> ParamFile:
    """Build a conservative geometry-optimization parameter preset."""

    keywords = [
        KeywordAssignment("task", "GEOMETRYOPTIMIZATION"),
        KeywordAssignment("xc_functional", xc_functional),
    ]
    if cut_off_energy is not None:
        keywords.append(_unitized("cut_off_energy", cut_off_energy, "eV"))
    if max_scf_cycles is not None:
        keywords.append(KeywordAssignment("max_scf_cycles", max_scf_cycles))
    if geom_energy_tol is not None:
        keywords.append(_unitized("geom_energy_tol", geom_energy_tol, "eV"))
    if geom_force_tol is not None:
        keywords.append(_unitized("geom_force_tol", geom_force_tol, "eV/ang"))
    if geom_stress_tol is not None:
        keywords.append(_unitized("geom_stress_tol", geom_stress_tol, "GPa"))
    if geom_max_iter is not None:
        keywords.append(KeywordAssignment("geom_max_iter", geom_max_iter))
    return ParamFile(keywords=tuple(keywords))


def molecular_dynamics(
    *,
    xc_functional: str = "PBE",
    cut_off_energy: float | ValueWithUnit | None = None,
    md_delta_t: float | ValueWithUnit | None = ValueWithUnit(1.0, "fs"),
    md_num_iter: int | None = 100,
    md_temperature: float | ValueWithUnit | None = ValueWithUnit(300.0, "K"),
) -> ParamFile:
    """Build a conservative molecular-dynamics parameter preset."""

    keywords = [
        KeywordAssignment("task", "MOLECULARDYNAMICS"),
        KeywordAssignment("xc_functional", xc_functional),
    ]
    if cut_off_energy is not None:
        keywords.append(_unitized("cut_off_energy", cut_off_energy, "eV"))
    if md_delta_t is not None:
        keywords.append(_unitized("md_delta_t", md_delta_t, "fs"))
    if md_num_iter is not None:
        keywords.append(KeywordAssignment("md_num_iter", md_num_iter))
    if md_temperature is not None:
        keywords.append(_unitized("md_temperature", md_temperature, "K"))
    return ParamFile(keywords=tuple(keywords))


def _unitized(name: str, value: float | ValueWithUnit, default_unit: str) -> KeywordAssignment:
    if isinstance(value, ValueWithUnit):
        return KeywordAssignment(name, value)
    return KeywordAssignment(name, ValueWithUnit(value, default_unit))

"""Tests for JSON-backed MuFinder-native UEP optimization job helpers."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from mufinder.domain.constants import ATOMIC_MASS_KG, MUON_MASS_KG
from mufinder.science.uep import StationaryPointClassification, UepOptimizationResult, UepStructure
from mufinder.services.uep_optimize_job import (
    parse_result_file,
    result_file_name,
    run_manifest_file,
    write_manifest,
)


def test_run_manifest_file_writes_roundtrippable_result(tmp_path: Path) -> None:
    seed_name = "si2_trial"
    for fixture in ("Si2.den_fmt", "Si2.castep", "Si2.cell"):
        shutil.copy(Path("tests/fixtures/uep") / fixture, tmp_path / fixture)

    manifest = write_manifest(
        tmp_path,
        seed_name,
        {
            "seed_name": seed_name,
            "density_file": "Si2.den_fmt",
            "castep_file": "Si2.castep",
            "cell_file": "Si2.cell",
            "gw_factor": 5.0,
            "start_position_ang": [1.0, 1.0, 1.0],
            "max_steps": 10,
        },
    )

    target = run_manifest_file(manifest)

    assert target.name == result_file_name(seed_name)
    parsed = parse_result_file(tmp_path, seed_name)
    assert parsed.completed in {True, False}
    assert parsed.optimization is not None
    assert parsed.probe is not None
    assert parsed.probe.classification.label in {"minimum", "maximum", "saddle", "flat"}


@pytest.mark.parametrize(
    ("mass_fields", "expected_kg"),
    [
        ({}, MUON_MASS_KG),
        ({"particle_mass_amu": None}, MUON_MASS_KG),
        ({"particle_mass_amu": 1.0}, ATOMIC_MASS_KG),
        ({"particle_mass_kg": 2.0e-28, "particle_mass_amu": 1.0}, 2.0e-28),
    ],
    ids=["default", "null-amu", "amu", "kg-wins"],
)
def test_run_manifest_file_resolves_the_particle_mass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mass_fields: dict, expected_kg: float
) -> None:
    for fixture in ("Si2.den_fmt", "Si2.castep", "Si2.cell"):
        shutil.copy(Path("tests/fixtures/uep") / fixture, tmp_path / fixture)
    seen: list[float] = []

    def record_mass(field, **kwargs):
        seen.append(kwargs["particle_mass_kg"])
        raise RuntimeError("stop after the mass is resolved")

    monkeypatch.setattr("mufinder.services.uep_optimize_job.optimize_muon_position", record_mass)
    manifest = write_manifest(
        tmp_path,
        "si2_mass",
        {
            "seed_name": "si2_mass",
            "density_file": "Si2.den_fmt",
            "castep_file": "Si2.castep",
            "cell_file": "Si2.cell",
            **mass_fields,
        },
    )

    run_manifest_file(manifest)

    assert seen == [expected_kg]


def _host() -> UepStructure:
    return UepStructure(
        lattice_vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        species=("Si",),
        positions_ang=((0.0, 0.0, 0.0),),
        fractional_positions=((0.0, 0.0, 0.0),),
        atomic_spins=None,
    )


def _optimization(
    fractional: tuple[float, float, float],
    *,
    total: float,
) -> UepOptimizationResult:
    return UepOptimizationResult(
        start_position_ang=(0.0, 0.0, 0.0),
        optimized_position_ang=tuple(4.0 * value for value in fractional),
        optimized_fractional_position=fractional,
        classical_energy_ev=float(total),
        zero_point_energy_ev=0.0,
        total_energy_ev=float(total),
        success=True,
        message="",
        iterations=1,
        classification=StationaryPointClassification(
            label="minimum",
            eigenvalues=(1.0, 1.0, 1.0),
            positive_count=3,
            negative_count=0,
            near_zero_count=0,
        ),
    )

"""Tests for JSON-backed MuFinder-native UEP optimization job helpers."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from mufinder.science.uep import StationaryPointClassification, UepOptimizationResult, UepStructure
from mufinder.science.uep_clustering import UepClusteringConfig
from mufinder.services.uep_optimize_job import (
    parse_result_file,
    result_file_name,
    run_manifest_file,
    summarize_batch_results,
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


def test_summarize_batch_results_supports_soprano_backed_metadata() -> None:
    summary = summarize_batch_results(
        [
            _optimization((0.25, 0.25, 0.25), total=-1.0),
            _optimization((0.75, 0.75, 0.75), total=-1.0),
        ],
        source_structure=_host(),
        clustering_config=UepClusteringConfig(clustering_method="hier", clustering_hier_t=0.05),
    )

    assert summary["clustering_method"] == "hier"
    assert summary["feature_mode"] == "soprano_defect_asymmetric_fpos"
    assert len(summary["clusters"]) == 1
    assert summary["clusters"][0]["representative_asymmetric_fractional_position"] is not None


def test_summarize_batch_results_preserves_legacy_threshold_path() -> None:
    summary = summarize_batch_results([], threshold_ang=0.3)

    assert summary["threshold_ang"] == pytest.approx(0.3)
    assert summary["clustering_method"] == "legacy_cartesian_hier"

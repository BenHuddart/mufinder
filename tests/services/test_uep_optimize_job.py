"""Tests for JSON-backed MuFinder-native UEP optimization job helpers."""

from __future__ import annotations

from pathlib import Path
import shutil

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

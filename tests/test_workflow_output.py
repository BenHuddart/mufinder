"""Tests for rendering resolved jobs to local filesystem output."""

from __future__ import annotations

import json
from pathlib import Path

from mufinder.castep_io.types import KeywordAssignment
from mufinder.domain.models import Structure
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.convergence import ConvergenceWorkflow
from mufinder.workflow.output import write_job_set
from mufinder.workflow.types import InputTemplate, ParamTemplate


def _jobs() -> list[object]:
    structure = Structure(
        label="Si",
        lattice_vectors=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    workflow = ConvergenceWorkflow(
        structure=structure_to_spec(structure),
        sweep_parameter="cut_off_energy",
        sweep_values=[300.0, 340.0],
        name_prefix="conv",
        workflow_defaults=InputTemplate(
            param=ParamTemplate(
                keywords=(
                    KeywordAssignment("task", "SINGLEPOINT"),
                    KeywordAssignment("xc_functional", "PBE"),
                )
            )
        ),
    )
    return workflow.expand()


def test_write_job_set_creates_input_files_and_manifest(tmp_path: Path) -> None:
    jobs = _jobs()
    out_dir = tmp_path / "jobs"

    manifest_path = write_job_set(jobs, out_dir)

    assert manifest_path == out_dir / "manifest.json"
    assert manifest_path.exists()

    for job in jobs:
        job_dir = out_dir / job.name
        assert job_dir.exists()

        cell_path = job_dir / f"{job.name}.cell"
        param_path = job_dir / f"{job.name}.param"
        assert cell_path.exists()
        assert param_path.exists()

        assert "%BLOCK LATTICE_CART" in cell_path.read_text(encoding="utf-8")
        assert "TASK : SINGLEPOINT" in param_path.read_text(encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert len(manifest["jobs"]) == 2
    assert manifest["jobs"][0]["name"] == jobs[0].name

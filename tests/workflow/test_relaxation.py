"""Tests for pristine-relaxation workflow expansion."""

from __future__ import annotations

import pytest

from mufinder.domain.models import Structure
from mufinder.castep_io import write_cell, write_param
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.relaxation import RelaxationJobConfig, expand_relaxation_jobs


def _base_structure_spec() -> object:
    structure = Structure(
        label="Si",
        lattice_vectors=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    return structure_to_spec(structure)


def test_expand_relaxation_jobs_sets_geometry_task_and_row_controls() -> None:
    jobs = expand_relaxation_jobs(
        jobs=(
            RelaxationJobConfig(
                label="row1",
                structure=_base_structure_spec(),
                fix_all_cell=False,
                kpoint_grid=(2, 2, 2),
            ),
        ),
        name_prefix="pristine",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "pristine_row1"

    task_values = [keyword.value for keyword in job.param_file.keywords if keyword.name == "task"]
    assert task_values == ["GEOMETRYOPTIMIZATION"]
    param_values = {keyword.name: keyword.value for keyword in job.param_file.keywords}
    assert param_values["continuation"] == "DEFAULT"
    assert "CONTINUATION : DEFAULT" in write_param(job.param_file)

    scalar_map = {keyword.name: keyword.value for keyword in job.cell_file.scalars}
    assert scalar_map["fix_all_cell"] is False
    assert "FIX_ALL_CELL : FALSE" in write_cell(job.cell_file)
    assert job.cell_file.kpoint_mp_grid == (2, 2, 2)
    assert job.provenance["label"] == "row1"


def test_expand_relaxation_jobs_rejects_blank_labels() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        expand_relaxation_jobs(
            jobs=(
                RelaxationJobConfig(
                    label="   ",
                    structure=_base_structure_spec(),
                ),
            ),
            name_prefix="pristine",
        )

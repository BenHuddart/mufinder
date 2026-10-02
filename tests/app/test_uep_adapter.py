"""Tests for UEP source-job expansion logic."""

from __future__ import annotations

import pytest

from mufinder.app.uep_adapter import expand_uep_jobs
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.castep_io import write_cell
from mufinder.domain.models import Structure


def _host_structure() -> Structure:
    return Structure(
        label="Si",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def test_expand_uep_jobs_applies_project_defaults() -> None:
    config = UepConfig(
        name_prefix="uep",
        rows=(
            UepSourceRow(
                label="row_a",
                parameter_values={},
            ),
        ),
    )

    expansion = expand_uep_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=650.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    assert len(expansion.jobs) == 1
    job = expansion.jobs[0]
    param_values = {keyword.name: keyword.value for keyword in job.param_file.keywords}
    assert job.workflow_stage.value == "uep"
    assert param_values["cut_off_energy"].value == pytest.approx(650.0)
    assert param_values["task"] == "SINGLEPOINT"
    assert param_values["write_formatted_density"] is True
    assert job.cell_file.symmetry_generate is True
    assert "SYMMETRY_GENERATE" in write_cell(job.cell_file)
    assert job.cell_file.kpoint_mp_grid == (4, 4, 4)


def test_expand_uep_jobs_uses_exact_project_grid_for_unit_cell_with_reference_lattice() -> None:
    config = UepConfig(
        name_prefix="uep",
        rows=(
            UepSourceRow(
                label="row_unit",
                parameter_values={},
            ),
        ),
    )
    host = _host_structure()
    host.lattice_vectors = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]

    expansion = expand_uep_jobs(
        host_structure=host,
        config=config,
        project_cutoff_energy=650.0,
        project_kpoint_grid=(3, 3, 3),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
    )

    assert expansion.jobs[0].cell_file.kpoint_mp_grid == (3, 3, 3)
    assert expansion.kpoint_messages == ()


def test_expand_uep_jobs_forces_formatted_density_even_when_disabled_in_row() -> None:
    config = UepConfig(
        name_prefix="uep",
        rows=(
            UepSourceRow(
                label="row_locked",
                parameter_values={
                    "task": "GEOMETRYOPTIMIZATION",
                    "write_formatted_density": False,
                },
            ),
        ),
    )

    expansion = expand_uep_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=None,
        project_kpoint_reference_lattice=None,
    )

    param_values = {keyword.name: keyword.value for keyword in expansion.jobs[0].param_file.keywords}
    assert param_values["task"] == "SINGLEPOINT"
    assert param_values["write_formatted_density"] is True


def test_expand_uep_jobs_uses_row_label_verbatim_for_job_suffix() -> None:
    config = UepConfig(
        name_prefix="uep",
        rows=(
            UepSourceRow(
                label="source_spin_norelax",
                parameter_values={},
            ),
        ),
    )

    expansion = expand_uep_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=None,
        project_kpoint_reference_lattice=None,
    )

    job = expansion.jobs[0]
    assert job.name == "uep_source_spin_norelax"
    assert job.provenance["label"] == "source_spin_norelax"


def test_expand_uep_jobs_does_not_prepend_source_when_row_label_lacks_it() -> None:
    config = UepConfig(
        name_prefix="uep",
        rows=(
            UepSourceRow(
                label="spin_norelax",
                parameter_values={},
            ),
        ),
    )

    expansion = expand_uep_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=None,
        project_kpoint_reference_lattice=None,
    )

    assert expansion.jobs[0].name == "uep_spin_norelax"


def test_expand_uep_jobs_applies_project_xc_functional_unless_row_sets_one() -> None:
    config = UepConfig(
        name_prefix="uep",
        rows=(
            UepSourceRow(label="row_default", parameter_values={}),
            UepSourceRow(label="row_explicit", parameter_values={"xc_functional": "LDA"}),
        ),
    )

    expansion = expand_uep_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=650.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=None,
        project_xc_functional="PBE",
    )

    functionals = [
        {keyword.name: keyword.value for keyword in job.param_file.keywords}["xc_functional"]
        for job in expansion.jobs
    ]
    assert functionals == ["PBE", "LDA"]

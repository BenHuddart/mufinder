"""Tests for pristine-relaxation adapter expansion logic."""

from __future__ import annotations

import pytest

from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs, suggested_grid_for_row
from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
from mufinder.domain.models import Structure


def _host_structure() -> Structure:
    return Structure(
        label="Si",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def test_expand_pristine_relaxation_jobs_applies_project_defaults() -> None:
    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(
                label="row_a",
                fix_all_cell=True,
                supercell_matrix=None,
                parameter_values={"task": "GEOMETRYOPTIMIZATION"},
            ),
        ),
    )

    expansion = expand_pristine_relaxation_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    assert len(expansion.jobs) == 1
    job = expansion.jobs[0]
    param_values = {keyword.name: keyword.value for keyword in job.param_file.keywords}
    cell_values = {keyword.name: keyword.value for keyword in job.cell_file.scalars}

    assert job.name == "pristine_relax_row_a"
    assert param_values["cut_off_energy"].value == pytest.approx(600.0)
    assert param_values["write_cell_structure"] is True
    assert cell_values["fix_all_cell"] is True
    assert job.cell_file.symmetry_generate is True
    assert job.cell_file.kpoint_mp_grid == (4, 4, 4)
    assert expansion.kpoint_messages == ()


def test_expand_pristine_relaxation_jobs_forces_write_cell_structure_true() -> None:
    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(
                label="row_locked",
                fix_all_cell=False,
                supercell_matrix=None,
                parameter_values={
                    "task": "GEOMETRYOPTIMIZATION",
                    "write_cell_structure": False,
                },
            ),
        ),
    )

    expansion = expand_pristine_relaxation_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=None,
        project_kpoint_reference_lattice=None,
    )

    param_values = {keyword.name: keyword.value for keyword in expansion.jobs[0].param_file.keywords}
    assert param_values["write_cell_structure"] is True


def test_expand_pristine_relaxation_jobs_emits_supercell_kpoint_message() -> None:
    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(
                label="row_b",
                fix_all_cell=False,
                supercell_matrix=((2, 0, 0), (0, 1, 0), (0, 0, 1)),
                parameter_values={"task": "GEOMETRYOPTIMIZATION"},
            ),
        ),
    )

    expansion = expand_pristine_relaxation_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    assert len(expansion.kpoint_messages) == 1
    assert "4x4x4 to 2x4x4" in expansion.kpoint_messages[0]


def test_expand_pristine_relaxation_jobs_honors_explicit_row_kpoint_override() -> None:
    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(
                label="row_c",
                fix_all_cell=True,
                supercell_matrix=((2, 0, 0), (0, 1, 0), (0, 0, 1)),
                parameter_values={
                    "task": "GEOMETRYOPTIMIZATION",
                    "kpoint_mp_grid": (3, 3, 3),
                },
            ),
        ),
    )

    expansion = expand_pristine_relaxation_jobs(
        host_structure=_host_structure(),
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    assert expansion.jobs[0].cell_file.kpoint_mp_grid == (3, 3, 3)
    assert expansion.kpoint_messages == ()



def test_suggested_grid_for_row_handles_identity_and_supercell() -> None:
    host_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    identity = suggested_grid_for_row(
        host_lattice=host_lattice,
        row_matrix=None,
        reference_lattice=host_lattice,
        reference_grid=(4, 4, 4),
    )
    doubled_a = suggested_grid_for_row(
        host_lattice=host_lattice,
        row_matrix=((2, 0, 0), (0, 1, 0), (0, 0, 1)),
        reference_lattice=host_lattice,
        reference_grid=(4, 4, 4),
    )

    assert identity == (4, 4, 4)
    assert doubled_a == (2, 4, 4)

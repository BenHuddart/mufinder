"""Tests for pristine-relaxation adapter expansion logic."""

from __future__ import annotations

import pytest

from uuid import uuid4

from mufinder.app.relaxation_adapter import (
    expand_muon_relaxation_jobs,
    expand_pristine_relaxation_jobs,
    suggested_grid_for_row,
)
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
)
from mufinder.castep_io import write_cell
from tests.support.structures import si_pair_host


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
        host_structure=si_pair_host(),
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
    assert param_values["continuation"] == "DEFAULT"
    assert param_values["geom_max_iter"] == 300
    assert param_values["write_cell_structure"] is True
    assert cell_values["fix_all_cell"] is True
    rendered_cell = write_cell(job.cell_file)
    assert "FIX_ALL_CELL : TRUE" in rendered_cell
    assert "FIX_CELL" not in rendered_cell
    assert job.cell_file.symmetry_generate is True
    assert job.cell_file.kpoint_mp_grid == (4, 4, 4)
    assert expansion.kpoint_adjustments == ()


def test_expand_pristine_relaxation_jobs_uses_exact_project_grid_for_unit_cell_with_reference_lattice() -> None:
    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(
                label="row_unit",
                fix_all_cell=True,
                supercell_matrix=None,
                parameter_values={"task": "GEOMETRYOPTIMIZATION"},
            ),
        ),
    )
    host = si_pair_host()
    host.lattice_vectors = [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]]

    expansion = expand_pristine_relaxation_jobs(
        host_structure=host,
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(3, 3, 3),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
    )

    assert expansion.jobs[0].cell_file.kpoint_mp_grid == (3, 3, 3)
    assert expansion.kpoint_adjustments == ()


def test_expand_pristine_relaxation_jobs_uses_exact_project_grid_for_anisotropic_unit_cell() -> None:
    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(
                label="row_unit",
                fix_all_cell=True,
                supercell_matrix=None,
                parameter_values={"task": "GEOMETRYOPTIMIZATION"},
            ),
        ),
    )
    host = si_pair_host()
    host.lattice_vectors = [[3.0, 0.0, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 5.0]]

    expansion = expand_pristine_relaxation_jobs(
        host_structure=host,
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(3, 3, 3),
        project_kpoint_reference_lattice=None,
    )

    assert expansion.jobs[0].cell_file.kpoint_mp_grid == (3, 3, 3)
    assert expansion.kpoint_adjustments == ()


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
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=None,
        project_kpoint_reference_lattice=None,
    )

    param_values = {keyword.name: keyword.value for keyword in expansion.jobs[0].param_file.keywords}
    assert param_values["write_cell_structure"] is True


def test_expand_pristine_relaxation_jobs_records_supercell_kpoint_adjustment() -> None:
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
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    assert [
        (adjustment.row_label, adjustment.cell_label, adjustment.adjusted_grid)
        for adjustment in expansion.kpoint_adjustments
    ] == [("row_b", "2x1x1", (2, 4, 4))]


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
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    assert expansion.jobs[0].cell_file.kpoint_mp_grid == (3, 3, 3)
    assert expansion.kpoint_adjustments == ()



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


def test_suggested_grid_for_row_halves_grid_for_doubled_unequal_lattice() -> None:
    host_lattice = [[3.6, 0.0, 0.0], [0.0, 4.2, 0.0], [0.0, 0.0, 5.4]]

    grid = suggested_grid_for_row(
        host_lattice=host_lattice,
        row_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 2)),
        reference_lattice=host_lattice,
        reference_grid=(4, 4, 4),
    )

    assert grid == (2, 2, 2)


def test_suggested_grid_for_row_returns_exact_project_grid_for_identity_reference_mismatch() -> None:
    grid = suggested_grid_for_row(
        host_lattice=[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        row_matrix=None,
        reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        reference_grid=(3, 3, 3),
    )

    assert grid == (3, 3, 3)


def test_suggested_grid_for_row_rescales_a_supercell_converged_grid_for_the_unit_cell() -> None:
    """A grid converged on a 2x1x1 supercell is twice as dense along a on the unit cell."""

    grid = suggested_grid_for_row(
        host_lattice=[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        row_matrix=None,
        reference_lattice=[[6.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        reference_grid=(2, 3, 3),
    )

    assert grid == (4, 3, 3)


def test_expand_muon_relaxation_jobs_forces_fixed_cell_and_charge() -> None:
    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=uuid4(),
                label="site_a",
                initial_fractional_position=(0.2, 0.4, 0.6),
                charge_state="positive_muon",
                parameter_values={"fix_all_cell": False, "charge": 0, "write_cell_structure": False},
            ),
            MuonRelaxationRow(
                site_id=uuid4(),
                label="site_b",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="muonium",
            ),
        )
    )

    expansion = expand_muon_relaxation_jobs(
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    assert [job.name for job in expansion.jobs] == ["positivemuon_site_a", "muonium_site_b"]
    assert {job.workflow_stage.value for job in expansion.jobs} == {"muon_relaxation"}
    assert {job.provenance["workflow"] for job in expansion.jobs} == {"muon_relaxation"}
    positive_params = {keyword.name: keyword.value for keyword in expansion.jobs[0].param_file.keywords}
    muonium_params = {keyword.name: keyword.value for keyword in expansion.jobs[1].param_file.keywords}
    positive_cell = {keyword.name: keyword.value for keyword in expansion.jobs[0].cell_file.scalars}

    assert positive_params["continuation"] == "DEFAULT"
    assert muonium_params["continuation"] == "DEFAULT"
    assert positive_params["geom_max_iter"] == 300
    assert muonium_params["geom_max_iter"] == 300
    assert positive_params["charge"] == 1
    assert muonium_params["charge"] == 0
    # Muonium is open-shell: spin-polarised by default; the diamagnetic mu+ is not.
    assert muonium_params["spin_polarized"] is True
    assert "spin_polarized" not in positive_params
    assert positive_params["write_cell_structure"] is True
    assert expansion.jobs[0].cell_file.symmetry_generate is False
    assert positive_cell["fix_all_cell"] is True
    rendered_cell = write_cell(expansion.jobs[0].cell_file)
    assert "FIX_ALL_CELL : TRUE" in rendered_cell
    assert "FIX_CELL" not in rendered_cell
    assert "SYMMETRY_GENERATE" not in rendered_cell


def test_expand_muon_relaxation_jobs_seeds_muonium_spin_in_cell() -> None:
    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=uuid4(),
                label="dia",
                initial_fractional_position=(0.2, 0.4, 0.6),
                charge_state="positive_muon",
            ),
            MuonRelaxationRow(
                site_id=uuid4(),
                label="mu",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="muonium",
            ),
        )
    )

    expansion = expand_muon_relaxation_jobs(
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    positive_cell = write_cell(expansion.jobs[0].cell_file)
    muonium_cell = write_cell(expansion.jobs[1].cell_file)
    # Neutral muonium seeds +1 on the muon (one unpaired electron); mu+ carries no spin.
    assert "SPIN=1" in muonium_cell
    assert "SPIN=" not in positive_cell


def test_expand_muon_relaxation_jobs_muonium_spin_polarized_is_overridable() -> None:
    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=uuid4(),
                label="mu",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="muonium",
                parameter_values={"spin_polarized": False},
            ),
        )
    )

    expansion = expand_muon_relaxation_jobs(
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    params = {keyword.name: keyword.value for keyword in expansion.jobs[0].param_file.keywords}
    # The default is overridable from the param dialog, so an explicit False is honoured.
    assert params["spin_polarized"] is False


def test_expand_muon_relaxation_jobs_places_muon_in_origin_supercell_image() -> None:
    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=uuid4(),
                label="site_super",
                initial_fractional_position=(0.5, 0.25, 0.75),
                charge_state="muonium",
                supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
            ),
        )
    )

    expansion = expand_muon_relaxation_jobs(
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    )

    muon_position = expansion.jobs[0].cell_file.positions_frac.positions[-1]
    assert muon_position.species == "H:mu"
    assert muon_position.coordinates == pytest.approx((0.25, 0.125, 0.75))
    assert expansion.jobs[0].cell_file.kpoint_mp_grid == (2, 2, 4)


def test_expand_muon_relaxation_jobs_writes_species_mass_block() -> None:
    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=uuid4(),
                label="site_a",
                initial_fractional_position=(0.2, 0.4, 0.6),
                charge_state="positive_muon",
            ),
        )
    )

    expansion = expand_muon_relaxation_jobs(
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=None,
        project_kpoint_grid=None,
        project_kpoint_reference_lattice=None,
    )

    rendered_cell = write_cell(expansion.jobs[0].cell_file)
    assert "H:mu 0.11342892566" in rendered_cell


def test_expand_muon_relaxation_jobs_rejects_sanitized_seed_name_collisions() -> None:
    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=uuid4(),
                label="A B",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="positive_muon",
            ),
            MuonRelaxationRow(
                site_id=uuid4(),
                label="A_B",
                initial_fractional_position=(0.4, 0.5, 0.6),
                charge_state="positive_muon",
            ),
        )
    )

    with pytest.raises(ValueError, match="positivemuon_A_B"):
        expand_muon_relaxation_jobs(
            host_structure=si_pair_host(),
            config=config,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )


def test_expand_pristine_relaxation_jobs_applies_project_xc_functional_unless_row_sets_one() -> None:
    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(
                label="row_default",
                fix_all_cell=True,
                supercell_matrix=None,
                parameter_values={"task": "GEOMETRYOPTIMIZATION"},
            ),
            PristineRelaxationRow(
                label="row_explicit",
                fix_all_cell=True,
                supercell_matrix=None,
                parameter_values={"task": "GEOMETRYOPTIMIZATION", "xc_functional": "LDA"},
            ),
        ),
    )

    expansion = expand_pristine_relaxation_jobs(
        host_structure=si_pair_host(),
        config=config,
        project_cutoff_energy=600.0,
        project_kpoint_grid=(4, 4, 4),
        project_kpoint_reference_lattice=None,
        project_xc_functional="PBE",
    )

    functionals = [
        {keyword.name: keyword.value for keyword in job.param_file.keywords}["xc_functional"]
        for job in expansion.jobs
    ]
    assert functionals == ["PBE", "LDA"]

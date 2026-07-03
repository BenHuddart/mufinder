"""Tests for convergence GUI adapter into workflow job expansion."""

from __future__ import annotations

import pytest

import mufinder.app.convergence_adapter as adapter_module
from mufinder.app.convergence_adapter import expand_convergence_jobs
from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.castep_io.types import BlockSection, ValueWithUnit
from mufinder.domain.models import Structure
from mufinder.workflow.bridge import structure_to_spec


def _base_structure_spec():
    structure = Structure(
        label="Si",
        lattice_vectors=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    return structure_to_spec(structure)


def _keyword_value(job, name: str):
    for item in job.param_file.keywords:
        if item.name == name:
            return item.value
    raise AssertionError(f"Missing keyword: {name}")


def test_expand_convergence_jobs_shared_defaults_only() -> None:
    config = ConvergenceConfig(
        sweep_parameter="cut_off_energy",
        sweep_values=(300.0, 360.0),
        cutoff_energies=(300.0, 360.0),
        shared_params={
            "task": "SINGLEPOINT",
            "xc_functional": "PBE",
            "max_scf_cycles": 80,
        },
    )

    jobs = expand_convergence_jobs(structure=_base_structure_spec(), config=config).jobs

    assert [job.name for job in jobs] == ["convergence_ecut_300", "convergence_ecut_360"]
    assert _keyword_value(jobs[0], "max_scf_cycles") == 80
    cutoff = _keyword_value(jobs[0], "cut_off_energy")
    assert isinstance(cutoff, ValueWithUnit)
    assert cutoff.value == 300.0


def test_expand_convergence_jobs_sets_atomic_spins_only_when_spin_polarized() -> None:
    structure = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties={"atomic_spins": [2.0]},
    )
    structure_spec = structure_to_spec(structure)

    spin_jobs = expand_convergence_jobs(
        structure=structure_spec,
        config=ConvergenceConfig(
            sweep_parameter="cut_off_energy",
            sweep_values=(300.0,),
            cutoff_energies=(300.0,),
            shared_params={"task": "SINGLEPOINT", "spin_polarized": True},
        ),
    ).jobs
    non_spin_jobs = expand_convergence_jobs(
        structure=structure_spec,
        config=ConvergenceConfig(
            sweep_parameter="cut_off_energy",
            sweep_values=(300.0,),
            cutoff_energies=(300.0,),
            shared_params={"task": "SINGLEPOINT"},
        ),
    ).jobs

    assert spin_jobs[0].cell_file.atomic_spins == pytest.approx((2.0,))
    assert non_spin_jobs[0].cell_file.atomic_spins is None


def test_expand_convergence_jobs_applies_subset_override() -> None:
    config = ConvergenceConfig(
        sweep_parameter="cut_off_energy",
        sweep_values=(300.0, 360.0),
        cutoff_energies=(300.0, 360.0),
        shared_params={
            "task": "SINGLEPOINT",
            "xc_functional": "PBE",
            "max_scf_cycles": 60,
        },
        overrides={frozenset({360.0}): {"max_scf_cycles": 120}},
        name_prefix="conv",
    )

    jobs = expand_convergence_jobs(structure=_base_structure_spec(), config=config).jobs

    assert [job.name for job in jobs] == ["conv_ecut_300", "conv_ecut_360"]
    assert _keyword_value(jobs[0], "max_scf_cycles") == 60
    assert _keyword_value(jobs[1], "max_scf_cycles") == 120


def test_build_convergence_workflow_rejects_overrides() -> None:
    config = ConvergenceConfig(
        sweep_parameter="cut_off_energy",
        sweep_values=(300.0,),
        cutoff_energies=(300.0,),
        shared_params={"task": "SINGLEPOINT"},
        overrides={frozenset({300.0}): {"max_scf_cycles": 120}},
    )

    with pytest.raises(ValueError, match="Per-cutoff overrides"):
        adapter_module.build_convergence_workflow(structure=_base_structure_spec(), config=config)


def test_build_convergence_workflow_happy_path_matches_direct_expansion() -> None:
    # build_convergence_workflow is a deprecated public helper; its happy path is
    # orphaned in the GUI, so cover it here to guard the delegating shim. Its
    # expand() must produce the same jobs as calling the sweep expander directly.
    structure = _base_structure_spec()
    config = ConvergenceConfig(
        sweep_parameter="cut_off_energy",
        sweep_values=(300.0, 360.0),
        cutoff_energies=(300.0, 360.0),
        shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"},
    )

    workflow = adapter_module.build_convergence_workflow(structure=structure, config=config)
    shim_jobs = workflow.expand()
    direct_jobs = expand_convergence_jobs(structure=structure, config=config).jobs

    assert [job.name for job in shim_jobs] == [job.name for job in direct_jobs]
    assert [job.name for job in shim_jobs] == ["convergence_ecut_300", "convergence_ecut_360"]
    for shim_job, direct_job in zip(shim_jobs, direct_jobs):
        assert shim_job.param_file.keywords == direct_job.param_file.keywords
        assert shim_job.workflow_stage == direct_job.workflow_stage
        assert shim_job.tags == direct_job.tags


def test_expand_convergence_jobs_rejects_empty_and_duplicate_cutoffs() -> None:
    with pytest.raises(ValueError, match="At least one cutoff"):
        expand_convergence_jobs(
            structure=_base_structure_spec(),
            config=ConvergenceConfig(
                sweep_parameter="cut_off_energy",
                sweep_values=(),
                cutoff_energies=(),
                shared_params={"task": "SINGLEPOINT"},
            ),
        )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        expand_convergence_jobs(
            structure=_base_structure_spec(),
            config=ConvergenceConfig(
                sweep_parameter="cut_off_energy",
                sweep_values=(300.0, 300.0),
                cutoff_energies=(300.0, 300.0),
                shared_params={"task": "SINGLEPOINT"},
            ),
        )


def test_expand_convergence_jobs_rejects_override_outside_sweep() -> None:
    config = ConvergenceConfig(
        sweep_parameter="cut_off_energy",
        sweep_values=(300.0, 360.0),
        cutoff_energies=(300.0, 360.0),
        shared_params={"task": "SINGLEPOINT"},
        overrides={frozenset({420.0}): {"max_scf_cycles": 120}},
    )

    with pytest.raises(ValueError, match="not present in the sweep list"):
        expand_convergence_jobs(structure=_base_structure_spec(), config=config)


def test_expand_convergence_jobs_merges_multiple_override_layers_same_cutoff() -> None:
    config = ConvergenceConfig(
        sweep_parameter="cut_off_energy",
        sweep_values=(300.0, 360.0),
        cutoff_energies=(300.0, 360.0),
        shared_params={"task": "SINGLEPOINT", "max_scf_cycles": 40},
        overrides={
            frozenset({360.0, 300.0}): {"max_scf_cycles": 80},
            frozenset({360.0}): {"smearing_width": 0.2},
        },
    )

    jobs = expand_convergence_jobs(structure=_base_structure_spec(), config=config).jobs

    assert _keyword_value(jobs[0], "max_scf_cycles") == 80
    assert _keyword_value(jobs[1], "max_scf_cycles") == 80
    smearing = _keyword_value(jobs[1], "smearing_width")
    assert isinstance(smearing, ValueWithUnit)
    assert smearing.value == 0.2


def test_template_from_values_handles_cell_params_and_errors() -> None:
    template = adapter_module._template_from_values(
        {
            "task": "SINGLEPOINT",
            "kpoint_mp_grid": (4, 4, 4),
            "kpoint_mp_spacing": 0.08,
            "symmetry_generate": True,
        }
    )

    assert template.param is not None
    assert template.cell is not None
    assert template.cell.kpoint_mp_grid is None
    assert isinstance(template.cell.kpoint_mp_spacing, ValueWithUnit)
    assert template.cell.symmetry_generate is True

    with pytest.raises(ValueError, match="Unsupported CASTEP keyword"):
        adapter_module._template_from_values({"made_up_keyword": 1})

    with pytest.raises(ValueError, match="cut_off_energy is controlled"):
        adapter_module._template_from_values({"cut_off_energy": 340.0})


def test_expand_convergence_jobs_returns_empty_messages() -> None:
    # Convergence sweeps a fixed structure, so no supercell rescaling happens
    # and the shared AdapterExpansion messages must stay empty.
    config = ConvergenceConfig(
        sweep_parameter="cut_off_energy",
        sweep_values=(300.0, 360.0),
        cutoff_energies=(300.0, 360.0),
        shared_params={"task": "SINGLEPOINT"},
    )

    expansion = expand_convergence_jobs(structure=_base_structure_spec(), config=config)

    assert expansion.messages == ()
    assert expansion.kpoint_messages == ()
    assert len(expansion.jobs) == 2


def test_expand_convergence_jobs_kpoint_sweep() -> None:
    config = ConvergenceConfig(
        sweep_parameter="kpoint_mp_grid",
        sweep_values=((2, 2, 2), (3, 3, 2)),
        shared_params={
            "task": "SINGLEPOINT",
            "xc_functional": "PBE",
        },
    )

    jobs = expand_convergence_jobs(structure=_base_structure_spec(), config=config).jobs

    assert [job.name for job in jobs] == [
        "convergence_kgrid_2x2x2",
        "convergence_kgrid_3x3x2",
    ]
    assert jobs[0].workflow_stage.value == "convergence_kpoint"
    assert jobs[0].cell_file.kpoint_mp_grid == (2, 2, 2)
    assert jobs[1].cell_file.kpoint_mp_grid == (3, 3, 2)


def test_template_from_values_rejects_structure_controlled_blocks() -> None:
    with pytest.raises(ValueError, match="lattice_abc is controlled by the selected structure"):
        adapter_module._template_from_values(
            {
                "task": "SINGLEPOINT",
                "lattice_abc": BlockSection(
                    name="lattice_abc",
                    rows=((5.43, 5.43, 5.43), (90.0, 90.0, 90.0)),
                    unit="ang",
                ),
            }
        )

    with pytest.raises(ValueError, match="positions_abs is controlled by the selected structure"):
        adapter_module._template_from_values(
            {
                "task": "SINGLEPOINT",
                "positions_abs": BlockSection(
                    name="positions_abs",
                    rows=(("Si", 0.0, 0.0, 0.0),),
                    unit="ang",
                ),
            }
        )


def test_template_from_values_still_supports_non_structural_cell_blocks() -> None:
    template = adapter_module._template_from_values(
        {
            "task": "SINGLEPOINT",
            "species_pot": BlockSection(
                name="species_pot",
                rows=(("Si", "Si_00PBE.usp"),),
            ),
            "kpoints_list": BlockSection(
                name="kpoints_list",
                rows=((0.0, 0.0, 0.0, 1.0),),
            ),
            "hubbard_u": BlockSection(
                name="hubbard_u",
                rows=(("Ni", "d:", 2.4),),
                unit="eV",
            ),
        }
    )

    assert template.cell is not None
    assert template.cell.species_pot is not None
    assert template.cell.species_pot[0].species == "Si"
    assert template.cell.kpoints_list is not None
    assert template.cell.kpoints_list[0].weight == 1.0
    assert template.cell.extra_blocks == (
        BlockSection(name="hubbard_u", rows=(("Ni", "d:", 2.4),), unit="eV"),
    )

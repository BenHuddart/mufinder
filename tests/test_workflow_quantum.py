"""Tests for quantum-corrections CASTEP phonon workflow expansion."""

from __future__ import annotations

from uuid import uuid4

import pytest

from mufinder.castep_io import write_cell, write_param
from mufinder.castep_io.types import BlockSection, KeywordAssignment
from mufinder.domain.constants import MUON_GAMMA_RAD_S_T, MUON_MASS_AMU
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Structure
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.types import CellTemplate, ParamTemplate
from mufinder.workflow.quantum import (
    QUANTUM_SINGLEPOINT_CALC_TYPE,
    QUANTUM_PHONON_CALC_TYPE,
    QuantumPhononJobConfig,
    QuantumPhononSource,
    expand_quantum_singlepoint_jobs,
    expand_quantum_phonon_jobs,
    phonon_method_compatibility,
    quantum_source_eligibility,
)


def _relaxed_muon_structure() -> Structure:
    return Structure(
        label="relaxed site",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )


def _source() -> QuantumPhononSource:
    return QuantumPhononSource(
        label="site1",
        structure=structure_to_spec(_relaxed_muon_structure()),
        source_site_id=uuid4(),
        source_relaxation_run_id=uuid4(),
        source_relaxation_job_id=uuid4(),
        charge_state="positive_muon",
        relative_energy_eV=-10.0,
    )


def _muonium_source() -> QuantumPhononSource:
    return QuantumPhononSource(
        label="mu1",
        structure=structure_to_spec(_relaxed_muon_structure()),
        source_site_id=uuid4(),
        source_relaxation_run_id=uuid4(),
        source_relaxation_job_id=uuid4(),
        charge_state="muonium",
        relative_energy_eV=-10.0,
    )


def test_expand_quantum_phonon_jobs_seeds_muonium_spin_polarized() -> None:
    jobs = expand_quantum_phonon_jobs(
        jobs=(QuantumPhononJobConfig(source=_muonium_source()),),
        name_prefix="quantum_phonon",
    )
    job = jobs[0]
    param_text = write_param(job.param_file)
    cell_text = write_cell(job.cell_file)
    # Open-shell muonium: neutral, spin-polarised, with +1 seeded on the muon.
    assert "CHARGE : 0" in param_text
    assert "SPIN_POLARIZED : TRUE" in param_text
    assert "SPIN=1" in cell_text


def test_expand_quantum_singlepoint_jobs_seeds_muonium_spin_polarized() -> None:
    jobs = expand_quantum_singlepoint_jobs(
        jobs=(QuantumPhononJobConfig(source=_muonium_source()),),
        name_prefix="quantum_singlepoint",
    )
    job = jobs[0]
    param_text = write_param(job.param_file)
    cell_text = write_cell(job.cell_file)
    assert "CHARGE : 0" in param_text
    assert "SPIN_POLARIZED : TRUE" in param_text
    assert "SPIN=1" in cell_text


def test_expand_quantum_phonon_jobs_positive_muon_has_no_spin() -> None:
    jobs = expand_quantum_phonon_jobs(
        jobs=(QuantumPhononJobConfig(source=_source()),),
        name_prefix="quantum_phonon",
    )
    job = jobs[0]
    # Diamagnetic mu+ binds no electron: not spin-polarised, no per-atom spin.
    assert "SPIN_POLARIZED" not in write_param(job.param_file)
    assert "SPIN=" not in write_cell(job.cell_file)


def test_expand_quantum_phonon_jobs_muonium_rejects_missing_muon_species() -> None:
    config = QuantumPhononJobConfig(source=_muonium_source(), muon_species="H:other")
    with pytest.raises(ValueError, match="no atom matching muon species"):
        expand_quantum_phonon_jobs(jobs=(config,), name_prefix="quantum_phonon")


def test_expand_quantum_phonon_jobs_muonium_spin_polarized_is_overridable() -> None:
    overrides = ParamTemplate(keywords=(KeywordAssignment("spin_polarized", False),))
    jobs = expand_quantum_phonon_jobs(
        jobs=(QuantumPhononJobConfig(source=_muonium_source(), param_overrides=overrides),),
        name_prefix="quantum_phonon",
    )
    job = jobs[0]
    assert "SPIN_POLARIZED : FALSE" in write_param(job.param_file)


def test_expand_quantum_singlepoint_jobs_writes_singlepoint_inputs() -> None:
    jobs = expand_quantum_singlepoint_jobs(
        jobs=(QuantumPhononJobConfig(source=_source()),),
        name_prefix="quantum_singlepoint",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "quantum_singlepoint_site1"
    assert job.workflow_stage == WorkflowStage.QUANTUM_CORRECTIONS
    assert job.provenance["calc_type"] == QUANTUM_SINGLEPOINT_CALC_TYPE

    cell_text = write_cell(job.cell_file)
    param_text = write_param(job.param_file)
    assert "PHONON_KPOINT_MP_GRID 1 1 1" not in cell_text
    assert "TASK : SINGLEPOINT" in param_text
    assert "PHONON_METHOD" not in param_text
    assert "CONTINUATION" not in param_text
    assert "CHARGE : 1" in param_text


def test_expand_quantum_singlepoint_jobs_writes_muon_species_mass() -> None:
    """Stage 1 must carry SPECIES_MASS, not just stage 2.

    The stage-2 phonon job is a CONTINUATION and CASTEP restores species masses
    from the stage-1 checkpoint, so a mass written only in stage 2 is ignored and
    the phonons run with the default hydrogen mass.
    """

    jobs = expand_quantum_singlepoint_jobs(
        jobs=(QuantumPhononJobConfig(source=_source()),),
        name_prefix="quantum_singlepoint",
    )

    cell_text = write_cell(jobs[0].cell_file)
    assert "%BLOCK SPECIES_MASS" in cell_text
    assert f"H:mu {MUON_MASS_AMU}" in cell_text
    assert "%BLOCK SPECIES_GAMMA" in cell_text
    assert f"H:mu {MUON_GAMMA_RAD_S_T}" in cell_text
    assert jobs[0].provenance["muon_mass_amu"] == MUON_MASS_AMU


def test_quantum_stages_agree_on_the_muon_species_mass() -> None:
    """Both stages must declare the same mass for the same muon species."""

    config = QuantumPhononJobConfig(source=_source())
    singlepoint = expand_quantum_singlepoint_jobs(
        jobs=(config,), name_prefix="quantum_singlepoint"
    )[0]
    phonon = expand_quantum_phonon_jobs(jobs=(config,), name_prefix="quantum_phonon")[0]

    def _species_block(job, name: str):
        return next(block for block in job.cell_file.blocks if block.name == name)

    for name in ("species_mass", "species_gamma"):
        assert _species_block(singlepoint, name).rows == _species_block(phonon, name).rows


def test_expand_quantum_singlepoint_jobs_allows_overriding_species_mass() -> None:
    """A caller-supplied SPECIES_MASS block still wins over the stage-1 base."""

    override = BlockSection("species_mass", (("H:mu", 0.5),), unit="AMU")
    jobs = expand_quantum_singlepoint_jobs(
        jobs=(
            QuantumPhononJobConfig(
                source=_source(),
                cell_overrides=CellTemplate(extra_blocks=(override,)),
            ),
        ),
        name_prefix="quantum_singlepoint",
    )

    cell_text = write_cell(jobs[0].cell_file)
    assert "H:mu 0.5" in cell_text
    assert f"H:mu {MUON_MASS_AMU}" not in cell_text


def test_expand_quantum_phonon_jobs_writes_gamma_phonon_inputs() -> None:
    jobs = expand_quantum_phonon_jobs(
        jobs=(QuantumPhononJobConfig(source=_source(), num_farms=4),),
        name_prefix="quantum_phonon",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "quantum_phonon_site1"
    assert job.workflow_stage == WorkflowStage.QUANTUM_CORRECTIONS
    assert job.provenance["calc_type"] == QUANTUM_PHONON_CALC_TYPE

    cell_text = write_cell(job.cell_file)
    param_text = write_param(job.param_file)
    assert "PHONON_KPOINT_MP_GRID 1 1 1" in cell_text
    assert "%BLOCK SPECIES_MASS" in cell_text
    assert "H:mu 0.11342892566" in cell_text
    assert "%BLOCK SPECIES_GAMMA" in cell_text
    assert "TASK : PHONON" in param_text
    assert "PHONON_METHOD : FINITEDISPLACEMENT" in param_text
    assert "CONTINUATION : DEFAULT" in param_text
    assert "NONSCF_EIGENVALUE_TOL : 1e-09 eV" in param_text
    assert "NUM_FARMS : 4" in param_text
    assert "CHARGE : 1" in param_text


def test_quantum_phonon_method_compatibility_blocks_dfpt_with_ultrasoft() -> None:
    compatible, message = phonon_method_compatibility("dfpt", pseudopotential_family="ultrasoft")

    assert compatible is False
    assert message is not None
    assert "norm-conserving" in message


def test_expand_quantum_phonon_jobs_rejects_incompatible_method() -> None:
    with pytest.raises(ValueError, match="norm-conserving"):
        expand_quantum_phonon_jobs(
            jobs=(
                QuantumPhononJobConfig(
                    source=_source(),
                    method="dfpt",
                    pseudopotential_family="ultrasoft",
                ),
            ),
            name_prefix="quantum",
        )


def test_quantum_source_eligibility_requires_completed_muon_relaxation() -> None:
    run_id = uuid4()
    site = MuonSite(status="relaxed", relaxation_run_id=run_id)
    run = CalculationRun(id=run_id, status=JobStatus.COMPLETED)
    job = Job(workflow_stage=WorkflowStage.MUON_RELAXATION, status=JobStatus.COMPLETED)
    structure = _relaxed_muon_structure()

    verdict = quantum_source_eligibility(site=site, run=run, job=job, relaxed_structure=structure)

    assert verdict.eligible is True
    uep_job = Job(workflow_stage=WorkflowStage.UEP, status=JobStatus.COMPLETED)
    verdict = quantum_source_eligibility(site=site, run=run, job=uep_job, relaxed_structure=structure)
    assert verdict.eligible is False

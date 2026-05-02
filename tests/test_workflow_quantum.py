"""Tests for quantum-corrections CASTEP phonon workflow expansion."""

from __future__ import annotations

from uuid import uuid4

import pytest

from mufinder.castep_io import write_cell, write_param
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Structure
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.quantum import (
    QUANTUM_PHONON_CALC_TYPE,
    QuantumPhononJobConfig,
    QuantumPhononSource,
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


def test_expand_quantum_phonon_jobs_writes_gamma_phonon_inputs() -> None:
    jobs = expand_quantum_phonon_jobs(
        jobs=(QuantumPhononJobConfig(source=_source()),),
        name_prefix="quantum",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job.name == "quantum_site1"
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

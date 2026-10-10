"""Tests for quantum-corrections CASTEP phonon workflow expansion."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from mufinder.castep_io import write_cell, write_param
from mufinder.castep_io.types import BlockSection, KeywordAssignment, ValueWithUnit
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
    method_to_castep,
    phonon_method_compatibility,
    quantum_source_eligibility,
)
from tests.support.structures import QUARTER_DIAGONAL, host_structure

_RUN_ID = UUID("11111111-1111-1111-1111-111111111111")
_SITE_ID = UUID("22222222-2222-2222-2222-222222222222")
_JOB_ID = UUID("33333333-3333-3333-3333-333333333333")


def _relaxed_muon_structure(species: tuple[str, ...] = ("Si", "H:mu")) -> Structure:
    return host_structure(
        label="relaxed site",
        origin="muon_relaxation",
        species=species,
        positions=QUARTER_DIAGONAL,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )


def _source(label: str = "site1", charge_state: str | None = "positive_muon") -> QuantumPhononSource:
    return QuantumPhononSource(
        label=label,
        structure=structure_to_spec(_relaxed_muon_structure()),
        source_site_id=_SITE_ID,
        source_relaxation_run_id=_RUN_ID,
        source_relaxation_job_id=_JOB_ID,
        charge_state=charge_state,
        total_energy_eV=-10.0,
    )


def _muonium_source() -> QuantumPhononSource:
    return _source(label="mu1", charge_state="muonium")


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


def test_expand_quantum_singlepoint_jobs_omits_charge_without_a_charge_state() -> None:
    jobs = expand_quantum_singlepoint_jobs(
        jobs=(QuantumPhononJobConfig(source=_source(charge_state=None)),),
        name_prefix="quantum_singlepoint",
    )

    assert "CHARGE" not in write_param(jobs[0].param_file)


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


def test_expand_quantum_singlepoint_jobs_merges_param_overrides() -> None:
    overrides = ParamTemplate(keywords=(KeywordAssignment("cut_off_energy", ValueWithUnit(500.0, "eV")),))
    jobs = expand_quantum_singlepoint_jobs(
        jobs=(QuantumPhononJobConfig(source=_source(), param_overrides=overrides),),
        name_prefix="quantum_singlepoint",
    )

    param_text = write_param(jobs[0].param_file)
    assert "TASK : SINGLEPOINT" in param_text
    assert "CUT_OFF_ENERGY : 500 eV" in param_text


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


def test_expand_quantum_phonon_jobs_merges_cell_and_param_overrides() -> None:
    jobs = expand_quantum_phonon_jobs(
        jobs=(
            QuantumPhononJobConfig(
                source=_source(),
                cell_overrides=CellTemplate(kpoint_mp_grid=(2, 2, 2)),
                param_overrides=ParamTemplate(keywords=(KeywordAssignment("max_scf_cycles", 120),)),
            ),
        ),
        name_prefix="quantum_phonon",
    )

    cell_text = write_cell(jobs[0].cell_file)
    param_text = write_param(jobs[0].param_file)
    assert "KPOINT_MP_GRID 2 2 2" in cell_text
    # The base species blocks survive the override merge.
    assert "%BLOCK SPECIES_MASS" in cell_text
    assert "%BLOCK SPECIES_GAMMA" in cell_text
    assert "TASK : PHONON" in param_text
    assert "MAX_SCF_CYCLES : 120" in param_text


@pytest.mark.parametrize(
    ("elec_energy_tol", "expected"),
    [
        (ValueWithUnit(1e-12, "eV"), "NONSCF_EIGENVALUE_TOL : 1e-12 eV"),
        (ValueWithUnit(1e-3, "eV"), "NONSCF_EIGENVALUE_TOL : 1e-09 eV"),
        (ValueWithUnit(1e-12, "hartree"), "NONSCF_EIGENVALUE_TOL : 1e-09 eV"),
        (1e-11, "NONSCF_EIGENVALUE_TOL : 1e-11 eV"),
        (None, "NONSCF_EIGENVALUE_TOL : 1e-09 eV"),
    ],
    ids=["tighter-ev", "looser-ev-capped", "non-ev-unit", "bare-number", "no-elec-energy-tol"],
)
def test_expand_quantum_phonon_jobs_derives_nonscf_tolerance_from_elec_energy_tol(
    elec_energy_tol: object, expected: str
) -> None:
    keywords = [KeywordAssignment("max_scf_cycles", 120)]
    if elec_energy_tol is not None:
        keywords.append(KeywordAssignment("elec_energy_tol", elec_energy_tol))
    jobs = expand_quantum_phonon_jobs(
        jobs=(
            QuantumPhononJobConfig(source=_source(), param_overrides=ParamTemplate(keywords=tuple(keywords))),
        ),
        name_prefix="quantum_phonon",
    )

    assert expected in write_param(jobs[0].param_file)
    assert "PHONON_KPOINT_MP_GRID 1 1 1" in write_cell(jobs[0].cell_file)


def test_expand_quantum_phonon_jobs_ignores_boolean_elec_energy_tol() -> None:
    overrides = ParamTemplate(keywords=(KeywordAssignment("elec_energy_tol", True),))
    jobs = expand_quantum_phonon_jobs(
        jobs=(QuantumPhononJobConfig(source=_source(), param_overrides=overrides),),
        name_prefix="quantum_phonon",
    )

    nonscf = {kw.name: kw.value for kw in jobs[0].param_file.keywords}["nonscf_eigenvalue_tol"]
    assert nonscf == ValueWithUnit(1e-9, "eV")


def test_method_to_castep_maps_known_methods_and_rejects_unknown() -> None:
    assert method_to_castep("dfpt") == "DFPT"
    with pytest.raises(ValueError, match="Unsupported phonon method"):
        method_to_castep("not_a_method")  # type: ignore[arg-type]


def test_quantum_phonon_method_compatibility_blocks_dfpt_with_ultrasoft() -> None:
    compatible, message = phonon_method_compatibility("dfpt", pseudopotential_family="ultrasoft")

    assert compatible is False
    assert message is not None
    assert "norm-conserving" in message


@pytest.mark.parametrize(
    ("method", "options", "fragment"),
    [
        ("dfpt", {"uses_hubbard_u": True}, "Hubbard U"),
        ("linear_response", {"uses_exact_exchange": True}, "exact exchange"),
        ("dfpt", {"uses_spin_orbit": True}, "spin-orbit"),
    ],
    ids=["hubbard-u", "exact-exchange", "spin-orbit"],
)
def test_quantum_phonon_method_compatibility_blocks_dfpt_for_unsupported_physics(
    method: str, options: dict[str, bool], fragment: str
) -> None:
    compatible, message = phonon_method_compatibility(method, **options)  # type: ignore[arg-type]

    assert compatible is False
    assert message is not None
    assert fragment in message


def test_quantum_phonon_method_compatibility_warns_for_unknown_pseudopotential_family() -> None:
    compatible, message = phonon_method_compatibility("dfpt", pseudopotential_family="unknown")

    assert compatible is True
    assert message is not None
    assert "unknown" in message


def test_quantum_phonon_method_compatibility_allows_dfpt_with_norm_conserving() -> None:
    assert phonon_method_compatibility("dfpt", pseudopotential_family="norm_conserving") == (True, None)


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


@pytest.mark.parametrize("expand", [expand_quantum_singlepoint_jobs, expand_quantum_phonon_jobs])
def test_quantum_expansion_requires_at_least_one_job(expand) -> None:
    with pytest.raises(ValueError, match="At least one quantum"):
        expand(jobs=(), name_prefix="q")


@pytest.mark.parametrize("expand", [expand_quantum_singlepoint_jobs, expand_quantum_phonon_jobs])
def test_quantum_expansion_rejects_blank_labels(expand) -> None:
    with pytest.raises(ValueError, match="labels must not be empty"):
        expand(jobs=(QuantumPhononJobConfig(source=_source(label="   ")),), name_prefix="q")


def _eligibility_inputs(**overrides: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "site": MuonSite(status="relaxed", relaxation_run_id=_RUN_ID),
        "run": CalculationRun(id=_RUN_ID, status=JobStatus.COMPLETED),
        "job": Job(workflow_stage=WorkflowStage.MUON_RELAXATION, status=JobStatus.COMPLETED),
        "relaxed_structure": _relaxed_muon_structure(),
    }
    inputs.update(overrides)
    return inputs


def test_quantum_source_eligibility_requires_completed_muon_relaxation() -> None:
    verdict = quantum_source_eligibility(**_eligibility_inputs())

    assert verdict.eligible is True
    uep_job = Job(workflow_stage=WorkflowStage.UEP, status=JobStatus.COMPLETED)
    verdict = quantum_source_eligibility(**_eligibility_inputs(job=uep_job))
    assert verdict.eligible is False


def _imported_structure() -> Structure:
    structure = _relaxed_muon_structure()
    structure.origin = "imported"
    return structure


def _structure_without_stage_provenance() -> Structure:
    structure = _relaxed_muon_structure()
    structure.properties.clear()
    return structure


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"site": MuonSite(status="candidate", relaxation_run_id=_RUN_ID)},
            "structurally relaxed",
        ),
        (
            {"site": MuonSite(status="relaxed", relaxation_run_id=None)},
            "no structural-relaxation run provenance",
        ),
        ({"run": None}, "recorded relaxation run is missing"),
        (
            {"run": CalculationRun(id=uuid4(), status=JobStatus.COMPLETED)},
            "recorded relaxation run is missing",
        ),
        (
            {"run": CalculationRun(id=_RUN_ID, status=JobStatus.RUNNING)},
            "relaxation run is not complete",
        ),
        ({"job": None}, "not a MuFinder muon structural relaxation"),
        (
            {"job": Job(workflow_stage=WorkflowStage.MUON_RELAXATION, status=JobStatus.RUNNING)},
            "muon relaxation job is not complete",
        ),
        ({"relaxed_structure": None}, "no parsed relaxed structure"),
        ({"relaxed_structure": _imported_structure()}, "not produced by muon structural relaxation"),
        (
            {"relaxed_structure": _structure_without_stage_provenance()},
            "does not carry muon-relaxation provenance",
        ),
        (
            {"relaxed_structure": _relaxed_muon_structure(species=("Si", "Si"))},
            "does not contain an H:mu atom",
        ),
    ],
    ids=[
        "unrelaxed-site",
        "no-run-provenance",
        "run-missing",
        "run-mismatch",
        "run-incomplete",
        "job-missing",
        "job-incomplete",
        "no-relaxed-structure",
        "wrong-origin",
        "no-stage-provenance",
        "no-muon",
    ],
)
def test_quantum_source_eligibility_names_the_rejection_reason(
    overrides: dict[str, Any], reason: str
) -> None:
    verdict = quantum_source_eligibility(**_eligibility_inputs(**overrides))

    assert verdict.eligible is False
    assert reason in verdict.reason

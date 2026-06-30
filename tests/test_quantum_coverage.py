"""Branch coverage for quantum-corrections workflow expansion.

These tests target the uncovered error paths, compatibility branches,
eligibility verdicts, override-merging paths, and the non-SCF eigenvalue
tolerance helper in :mod:`mufinder.workflow.quantum`. They are fully
deterministic: no network, GUI, filesystem, sleep, or randomness.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from mufinder.castep_io import write_cell, write_param
from mufinder.castep_io.types import KeywordAssignment, ValueWithUnit
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Structure
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.quantum import (
    QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL,
    QuantumPhononJobConfig,
    QuantumPhononSource,
    _cell_template,
    _param_keyword_value,
    _param_template,
    _phonon_nonscf_eigenvalue_tol,
    _singlepoint_param_template,
    expand_quantum_phonon_jobs,
    expand_quantum_singlepoint_jobs,
    method_to_castep,
    phonon_method_compatibility,
    quantum_source_eligibility,
)
from mufinder.workflow.types import CellTemplate, ParamTemplate

# Fixed UUIDs keep the tests deterministic.
_RUN_ID = UUID("11111111-1111-1111-1111-111111111111")
_SITE_ID = UUID("22222222-2222-2222-2222-222222222222")
_JOB_ID = UUID("33333333-3333-3333-3333-333333333333")


def _relaxed_muon_structure() -> Structure:
    return Structure(
        label="relaxed site",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
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
        relative_energy_eV=-10.0,
    )


# --- method_to_castep error path (lines 74-75) --------------------------------


def test_method_to_castep_known_value() -> None:
    assert method_to_castep("dfpt") == "DFPT"


def test_method_to_castep_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported phonon method"):
        method_to_castep("not_a_method")  # type: ignore[arg-type]


# --- phonon_method_compatibility branches (lines 92-100) ----------------------


def test_compatibility_hubbard_u_blocks_dfpt() -> None:
    compatible, message = phonon_method_compatibility("dfpt", uses_hubbard_u=True)
    assert compatible is False
    assert message is not None
    assert "Hubbard U" in message


def test_compatibility_exact_exchange_blocks_dfpt() -> None:
    compatible, message = phonon_method_compatibility("linear_response", uses_exact_exchange=True)
    assert compatible is False
    assert message is not None
    assert "exact exchange" in message


def test_compatibility_spin_orbit_blocks_dfpt() -> None:
    compatible, message = phonon_method_compatibility("dfpt", uses_spin_orbit=True)
    assert compatible is False
    assert message is not None
    assert "spin-orbit" in message


def test_compatibility_unknown_family_warns_but_allows() -> None:
    compatible, message = phonon_method_compatibility("dfpt", pseudopotential_family="unknown")
    assert compatible is True
    assert message is not None
    assert "unknown" in message


def test_compatibility_norm_conserving_dfpt_is_clean() -> None:
    compatible, message = phonon_method_compatibility(
        "dfpt", pseudopotential_family="norm_conserving"
    )
    assert compatible is True
    assert message is None


# --- quantum_source_eligibility rejection reasons (lines 113-131) -------------


def _ok_args() -> dict[str, object]:
    site = MuonSite(status="relaxed", relaxation_run_id=_RUN_ID)
    run = CalculationRun(id=_RUN_ID, status=JobStatus.COMPLETED)
    job = Job(workflow_stage=WorkflowStage.MUON_RELAXATION, status=JobStatus.COMPLETED)
    structure = _relaxed_muon_structure()
    return {"site": site, "run": run, "job": job, "relaxed_structure": structure}


def test_eligibility_rejects_unrelaxed_site() -> None:
    args = _ok_args()
    args["site"] = MuonSite(status="candidate", relaxation_run_id=_RUN_ID)
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "structurally relaxed" in verdict.reason


def test_eligibility_rejects_missing_run_provenance() -> None:
    args = _ok_args()
    args["site"] = MuonSite(status="relaxed", relaxation_run_id=None)
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "no structural-relaxation run provenance" in verdict.reason


def test_eligibility_rejects_when_run_missing() -> None:
    args = _ok_args()
    args["run"] = None
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "recorded relaxation run is missing" in verdict.reason


def test_eligibility_rejects_run_mismatch() -> None:
    args = _ok_args()
    args["run"] = CalculationRun(id=uuid4(), status=JobStatus.COMPLETED)
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "recorded relaxation run is missing" in verdict.reason


def test_eligibility_rejects_incomplete_run() -> None:
    args = _ok_args()
    args["run"] = CalculationRun(id=_RUN_ID, status=JobStatus.RUNNING)
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "relaxation run is not complete" in verdict.reason


def test_eligibility_rejects_missing_job() -> None:
    args = _ok_args()
    args["job"] = None
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "not a MuFinder muon structural relaxation" in verdict.reason


def test_eligibility_rejects_incomplete_job() -> None:
    args = _ok_args()
    args["job"] = Job(workflow_stage=WorkflowStage.MUON_RELAXATION, status=JobStatus.RUNNING)
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "muon relaxation job is not complete" in verdict.reason


def test_eligibility_rejects_missing_relaxed_structure() -> None:
    args = _ok_args()
    args["relaxed_structure"] = None
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "no parsed relaxed structure" in verdict.reason


def test_eligibility_rejects_wrong_structure_origin() -> None:
    args = _ok_args()
    structure = _relaxed_muon_structure()
    object.__setattr__(structure, "origin", "imported")
    args["relaxed_structure"] = structure
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "not produced by muon structural relaxation" in verdict.reason


def test_eligibility_rejects_missing_stage_provenance() -> None:
    args = _ok_args()
    structure = _relaxed_muon_structure()
    structure.properties.clear()
    args["relaxed_structure"] = structure
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "does not carry muon-relaxation provenance" in verdict.reason


def test_eligibility_rejects_structure_without_muon() -> None:
    args = _ok_args()
    structure = Structure(
        label="no muon",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    args["relaxed_structure"] = structure
    verdict = quantum_source_eligibility(**args)  # type: ignore[arg-type]
    assert verdict.eligible is False
    assert "does not contain an H:mu atom" in verdict.reason


# --- empty-collection / empty-label guards (lines 147, 153, 202, 208) ---------


def test_expand_singlepoint_requires_at_least_one_job() -> None:
    with pytest.raises(ValueError, match="At least one quantum singlepoint job"):
        expand_quantum_singlepoint_jobs(jobs=(), name_prefix="q")


def test_expand_singlepoint_rejects_blank_label() -> None:
    with pytest.raises(ValueError, match="labels must not be empty"):
        expand_quantum_singlepoint_jobs(
            jobs=(QuantumPhononJobConfig(source=_source(label="   ")),),
            name_prefix="q",
        )


def test_expand_phonon_requires_at_least_one_job() -> None:
    with pytest.raises(ValueError, match="At least one quantum phonon job"):
        expand_quantum_phonon_jobs(jobs=(), name_prefix="q")


def test_expand_phonon_rejects_blank_label() -> None:
    with pytest.raises(ValueError, match="labels must not be empty"):
        expand_quantum_phonon_jobs(
            jobs=(QuantumPhononJobConfig(source=_source(label="\t")),),
            name_prefix="q",
        )


# --- override-merge paths (lines 268-269, 287-288, 299-300) -------------------


def test_cell_template_merges_cell_overrides() -> None:
    overrides = CellTemplate(kpoint_mp_grid=(2, 2, 2))
    config = QuantumPhononJobConfig(source=_source(), cell_overrides=overrides)
    template = _cell_template(config)
    assert template.kpoint_mp_grid == (2, 2, 2)
    # Base extra blocks (species mass/gamma) survive the merge.
    block_names = {block.name for block in template.extra_blocks}
    assert "species_mass" in block_names
    assert "species_gamma" in block_names


def test_param_template_merges_param_overrides_into_phonon_job() -> None:
    overrides = ParamTemplate(keywords=(KeywordAssignment("smearing_width", "0.05 eV"),))
    config = QuantumPhononJobConfig(source=_source(), param_overrides=overrides)
    template = _param_template(config)
    names = {kw.name for kw in template.keywords}
    assert "task" in names
    assert "smearing_width" in names


def test_singlepoint_param_template_merges_param_overrides() -> None:
    overrides = ParamTemplate(keywords=(KeywordAssignment("cut_off_energy", "500 eV"),))
    config = QuantumPhononJobConfig(source=_source(), param_overrides=overrides)
    template = _singlepoint_param_template(config)
    names = {kw.name for kw in template.keywords}
    assert "task" in names
    assert "cut_off_energy" in names


def test_singlepoint_param_template_no_charge_when_neutral() -> None:
    config = QuantumPhononJobConfig(source=_source(charge_state=None))
    template = _singlepoint_param_template(config)
    names = {kw.name for kw in template.keywords}
    assert "charge" not in names


def test_param_template_charge_zero_for_muonium() -> None:
    config = QuantumPhononJobConfig(source=_source(charge_state="muonium"))
    template = _param_template(config)
    charge = {kw.name: kw.value for kw in template.keywords}["charge"]
    assert charge == 0


# --- _phonon_nonscf_eigenvalue_tol branches (lines 306-317) -------------------


def test_nonscf_tol_uses_smaller_value_with_unit_in_ev() -> None:
    overrides = ParamTemplate(
        keywords=(KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-12, "eV")),)
    )
    config = QuantumPhononJobConfig(source=_source(), param_overrides=overrides)
    tol = _phonon_nonscf_eigenvalue_tol(config)
    assert tol.unit == "eV"
    assert tol.value == 1e-12


def test_nonscf_tol_caps_loose_value_with_unit_at_default() -> None:
    overrides = ParamTemplate(
        keywords=(KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-3, "eV")),)
    )
    config = QuantumPhononJobConfig(source=_source(), param_overrides=overrides)
    tol = _phonon_nonscf_eigenvalue_tol(config)
    assert tol.value == QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL.value


def test_nonscf_tol_ignores_non_ev_unit() -> None:
    overrides = ParamTemplate(
        keywords=(KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-12, "hartree")),)
    )
    config = QuantumPhononJobConfig(source=_source(), param_overrides=overrides)
    tol = _phonon_nonscf_eigenvalue_tol(config)
    assert tol == QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL


def test_nonscf_tol_accepts_bare_numeric() -> None:
    overrides = ParamTemplate(keywords=(KeywordAssignment("elec_energy_tol", 1e-11),))
    config = QuantumPhononJobConfig(source=_source(), param_overrides=overrides)
    tol = _phonon_nonscf_eigenvalue_tol(config)
    assert tol.unit == "eV"
    assert tol.value == 1e-11


def test_nonscf_tol_rejects_bool_numeric() -> None:
    overrides = ParamTemplate(keywords=(KeywordAssignment("elec_energy_tol", True),))
    config = QuantumPhononJobConfig(source=_source(), param_overrides=overrides)
    tol = _phonon_nonscf_eigenvalue_tol(config)
    assert tol == QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL


def test_nonscf_tol_defaults_without_override() -> None:
    config = QuantumPhononJobConfig(source=_source())
    tol = _phonon_nonscf_eigenvalue_tol(config)
    assert tol == QUANTUM_DEFAULT_NONSCF_EIGENVALUE_TOL


# --- _param_keyword_value (lines 323-327) -------------------------------------


def test_param_keyword_value_returns_none_for_no_template() -> None:
    assert _param_keyword_value(None, "elec_energy_tol") is None


def test_param_keyword_value_returns_none_when_absent() -> None:
    template = ParamTemplate(keywords=(KeywordAssignment("cut_off_energy", "500 eV"),))
    assert _param_keyword_value(template, "elec_energy_tol") is None


def test_param_keyword_value_matches_by_canonical_name() -> None:
    template = ParamTemplate(keywords=(KeywordAssignment("elec_energy_tol", "1e-12 eV"),))
    assert _param_keyword_value(template, "elec_energy_tol") == "1e-12 eV"


# --- integration: phonon override flows into rendered tolerance ---------------


def test_expand_phonon_threads_tightened_tolerance_into_param_file() -> None:
    overrides = ParamTemplate(
        keywords=(KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-12, "eV")),)
    )
    jobs = expand_quantum_phonon_jobs(
        jobs=(QuantumPhononJobConfig(source=_source(), param_overrides=overrides),),
        name_prefix="quantum_phonon",
    )
    param_text = write_param(jobs[0].param_file)
    assert "NONSCF_EIGENVALUE_TOL : 1e-12 eV" in param_text
    cell_text = write_cell(jobs[0].cell_file)
    assert "PHONON_KPOINT_MP_GRID 1 1 1" in cell_text

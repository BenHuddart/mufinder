"""Tests for resolving a relaxation's relaxed structure for export."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, muon_relaxation_calc_type
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, Result, Structure
from mufinder.services.project_document import ProjectDocument
from mufinder.services.structure_export import (
    NO_STRUCTURE_REASON,
    NOT_A_RELAXATION_REASON,
    NOT_COMPLETED_REASON,
    relaxed_structure_export_for_job,
    relaxed_structure_export_for_job_id,
    relaxed_structure_export_for_structure,
)
from mufinder.domain.models import Project

_NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def _document(tmp_path: Path) -> ProjectDocument:
    return ProjectDocument(
        project_root=tmp_path / "Iron.mufinder", project=Project(name="Iron study")
    )


def _completed_job(
    document: ProjectDocument,
    *,
    label: str,
    stage: WorkflowStage,
    calc_type: str,
    origin: str,
    energy: float | None = -1234.5,
    with_structure: bool = True,
    structure_properties: dict[str, object] | None = None,
) -> tuple[Job, Structure | None]:
    job = Job(label=label, workflow_stage=stage, calc_type=calc_type, status=JobStatus.COMPLETED)
    document.jobs[job.id] = job
    structure = None
    derived: list = []
    if with_structure:
        structure = Structure(
            label=f"{label} relaxed",
            origin=origin,
            lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            species=["Fe", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            properties={"source_job_id": str(job.id), **(structure_properties or {})},
        )
        document.structures[structure.id] = structure
        derived = [structure.id]
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.COMPLETED)
    data = {} if energy is None else {"total_energy": energy}
    result = Result(calculation_run_id=run.id, data=data, derived_structure_ids=derived)
    run.result_id = result.id
    document.runs[run.id] = run
    document.results[result.id] = result
    return job, structure


def test_muon_relaxation_export_names_charge_state_energy_and_job(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job, structure = _completed_job(
        document,
        label="Mu1",
        stage=WorkflowStage.MUON_RELAXATION,
        calc_type=muon_relaxation_calc_type(MUONIUM_CHARGE_STATE),
        origin="muon_relaxation",
    )

    export = relaxed_structure_export_for_job(document, job, now=_NOW)

    assert export.structure is structure
    assert export.unavailable_reason is None
    assert export.default_stem.startswith("mu1_")
    assert export.default_stem.endswith("_relaxed")
    assert export.default_directory == tmp_path
    assert export.header_lines[0] == "Relaxed structure exported by MuFinder"
    assert "Project: Iron study" in export.header_lines
    assert "Structure: Mu1 relaxed" in export.header_lines
    assert f"Source job: Mu1 (Muon relaxation, id {job.id})" in export.header_lines
    assert "Charge state: muonium" in export.header_lines
    assert "Final energy: -1234.500000 eV" in export.header_lines
    assert export.header_lines[-1].startswith("Exported: 2026-10-23")


def test_pristine_relaxation_export_has_no_charge_state(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job, structure = _completed_job(
        document,
        label="pristine_relax_row1",
        stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
        origin="pristine_relaxation",
        energy=None,
    )

    export = relaxed_structure_export_for_job_id(document, job.id, now=_NOW)

    assert export.structure is structure
    assert export.default_stem == "pristine_relax_row1_relaxed"
    assert not any(line.startswith("Charge state") for line in export.header_lines)
    assert not any(line.startswith("Final energy") for line in export.header_lines)


def test_imported_legacy_relaxation_exports_and_says_so(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job, structure = _completed_job(
        document,
        label="site_3",
        stage=WorkflowStage.MUON_RELAXATION,
        calc_type=muon_relaxation_calc_type("positive_muon"),
        origin="muon_relaxation",
        structure_properties={"import_source": {"kind": "legacy_mufinder"}},
    )

    export = relaxed_structure_export_for_job(document, job, now=_NOW)

    assert export.structure is structure
    assert "Imported from a legacy MuFinder project" in export.header_lines


def test_export_is_unavailable_until_the_relaxation_completes(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job, _structure = _completed_job(
        document,
        label="Mu1",
        stage=WorkflowStage.MUON_RELAXATION,
        calc_type=muon_relaxation_calc_type("positive_muon"),
        origin="muon_relaxation",
    )
    job.status = JobStatus.RUNNING

    export = relaxed_structure_export_for_job(document, job)

    assert export.structure is None
    assert export.unavailable_reason == NOT_COMPLETED_REASON
    assert relaxed_structure_export_for_job(document, None).unavailable_reason == (
        NOT_COMPLETED_REASON
    )


def test_completed_relaxation_without_a_parsed_cell_says_why(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job, _structure = _completed_job(
        document,
        label="Mu1",
        stage=WorkflowStage.MUON_RELAXATION,
        calc_type=muon_relaxation_calc_type("positive_muon"),
        origin="muon_relaxation",
        with_structure=False,
    )

    export = relaxed_structure_export_for_job(document, job)

    assert export.structure is None
    assert export.unavailable_reason == NO_STRUCTURE_REASON


def test_non_relaxation_job_has_no_relaxed_structure(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job = Job(
        label="uep",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
        status=JobStatus.COMPLETED,
    )
    document.jobs[job.id] = job

    export = relaxed_structure_export_for_job(document, job)

    assert export.structure is None
    assert export.unavailable_reason == NOT_A_RELAXATION_REASON


def test_export_for_structure_finds_its_job_through_source_job_id(tmp_path: Path) -> None:
    document = _document(tmp_path)
    job, structure = _completed_job(
        document,
        label="Mu2",
        stage=WorkflowStage.MUON_RELAXATION,
        calc_type=muon_relaxation_calc_type("positive_muon"),
        origin="muon_relaxation",
    )
    orphan = Structure(label="Loose cell", species=["Fe"], fractional_positions=[[0, 0, 0]])

    linked = relaxed_structure_export_for_structure(document, structure, now=_NOW)
    unlinked = relaxed_structure_export_for_structure(document, orphan, now=_NOW)

    assert f"Source job: Mu2 (Muon relaxation, id {job.id})" in linked.header_lines
    assert unlinked.structure is orphan
    assert unlinked.default_stem == "loose_cell"
    assert not any(line.startswith("Source job") for line in unlinked.header_lines)

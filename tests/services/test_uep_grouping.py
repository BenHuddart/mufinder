"""Tests for placing a UEP search: its source structure."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Job, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.uep_grouping import uep_source_structure_id

_A = 4.695
_C = 3.180
_X = 0.306
_LATTICE = ((_A, 0.0, 0.0), (0.0, _A, 0.0), (0.0, 0.0, _C))


def _cof2() -> Structure:
    return Structure(
        label="CoF2",
        lattice_vectors=[list(row) for row in _LATTICE],
        species=["Co", "Co", "F", "F", "F", "F"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [_X, _X, 0.0],
            [1.0 - _X, 1.0 - _X, 0.0],
            [0.5 + _X, 0.5 - _X, 0.5],
            [0.5 - _X, 0.5 + _X, 0.5],
        ],
    )


def _project(tmp_path: Path) -> tuple[ProjectService, ProjectDocument, Job]:
    service = ProjectService()
    document = service.create_project(tmp_path / "old.mufinder", "Old UEP")
    host = _cof2()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    definition = service.register_uep_definition(
        document,
        input_structure_id=host.id,
        parameters={"rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}]},
        output_directory=document.project_root / "files" / "uep",
    )
    job = service.add_job(
        document, label="uep_source", workflow_stage=WorkflowStage.UEP, calc_type="uep"
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    service.save_project(document)
    return service, document, job


def test_a_job_finds_its_source_structure_through_its_definition(tmp_path: Path) -> None:
    _service, document, job = _project(tmp_path)

    assert uep_source_structure_id(document, job.id) == document.project.host_structure_id
    assert uep_source_structure_id(document, uuid4()) is None

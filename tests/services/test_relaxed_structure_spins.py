"""Relaxed structures keep their run's SPIN= values, and old projects get them back."""

from __future__ import annotations

import json
from pathlib import Path

from mufinder.castep_io.cell_parser import parse_cell_text
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.relaxed_structure_spins import (
    apply_relaxed_cell_spins,
    relaxed_cell_spins,
)

_CELL_TEMPLATE = """
%BLOCK LATTICE_CART
ang
2.87 0.0 0.0
0.0 2.87 0.0
0.0 0.0 2.87
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Fe 0.0 0.0 0.0{spin}
Fe 0.5 0.5 0.5{spin}
%ENDBLOCK POSITIONS_FRAC
"""

_SPIN_CELL = _CELL_TEMPLATE.format(spin=" SPIN= 2.200")
_PLAIN_CELL = _CELL_TEMPLATE.format(spin="")


def _relaxed_project(
    tmp_path: Path, *, stage: WorkflowStage = WorkflowStage.PRISTINE_RELAXATION
) -> tuple[ProjectService, ProjectDocument, Structure, Path]:
    """A saved project with one relaxation job and its spin-less relaxed structure."""

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = service.add_job(
        document,
        label="relax_fe",
        workflow_stage=stage,
        calc_type="castep_pristine_relaxation",
    )
    structure = Structure(
        label="relax_fe relaxed",
        origin=stage.value,
        lattice_vectors=[[2.87, 0.0, 0.0], [0.0, 2.87, 0.0], [0.0, 0.0, 2.87]],
        species=["Fe", "Fe"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        properties={"source_job_id": str(job.id), "source_workflow_stage": stage.value},
    )
    service.add_structure(document, structure)
    service.save_project(document)
    job_dir = document.project_root / "files" / stage.value / job.label
    job_dir.mkdir(parents=True)
    return service, document, structure, job_dir


def _saved_structure_payload(document: ProjectDocument, structure: Structure) -> dict:
    path = document.project_root / "entities" / "structures" / f"{structure.id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_relaxed_cell_spins_reads_nonzero_spins_only() -> None:
    assert relaxed_cell_spins(parse_cell_text(_SPIN_CELL)) == [2.2, 2.2]
    assert relaxed_cell_spins(parse_cell_text(_PLAIN_CELL)) is None
    assert relaxed_cell_spins(parse_cell_text(_CELL_TEMPLATE.format(spin=" SPIN= 0.0"))) is None


def test_apply_relaxed_cell_spins_replaces_and_clears() -> None:
    structure = Structure(label="s", properties={"atomic_spins": [1.0, 1.0]})

    apply_relaxed_cell_spins(structure, parse_cell_text(_SPIN_CELL))
    assert structure.properties["atomic_spins"] == [2.2, 2.2]

    apply_relaxed_cell_spins(structure, parse_cell_text(_PLAIN_CELL))
    assert "atomic_spins" not in structure.properties


def test_opening_a_project_backfills_spins_from_the_out_cell(tmp_path: Path) -> None:
    _service, document, structure, job_dir = _relaxed_project(tmp_path)
    (job_dir / "relax_fe-out.cell").write_text(_SPIN_CELL, encoding="utf-8")
    assert "atomic_spins" not in _saved_structure_payload(document, structure)

    loader = ProjectService()
    reopened = loader.load_project(document.project_root)
    assert reopened.structures[structure.id].properties["atomic_spins"] == [2.2, 2.2]

    # The next save by the same service persists the recovered spins.
    loader.save_project(reopened)
    assert _saved_structure_payload(reopened, structure)["properties"]["atomic_spins"] == [
        2.2,
        2.2,
    ]


def test_backfill_falls_back_to_the_input_cell(tmp_path: Path) -> None:
    _service, document, structure, job_dir = _relaxed_project(
        tmp_path, stage=WorkflowStage.MUON_RELAXATION
    )
    (job_dir / "relax_fe-out.cell").write_text(_PLAIN_CELL, encoding="utf-8")
    (job_dir / "relax_fe.cell").write_text(_SPIN_CELL, encoding="utf-8")

    reopened = ProjectService().load_project(document.project_root)

    assert reopened.structures[structure.id].properties["atomic_spins"] == [2.2, 2.2]


def test_backfill_leaves_structures_without_recoverable_spins_alone(tmp_path: Path) -> None:
    _service, document, structure, job_dir = _relaxed_project(tmp_path)
    # A non-spin run, and an input cell for a different atom count.
    (job_dir / "relax_fe-out.cell").write_text(_PLAIN_CELL, encoding="utf-8")
    (job_dir / "relax_fe.cell").write_text(
        _SPIN_CELL.replace("Fe 0.5 0.5 0.5 SPIN= 2.200\n", ""), encoding="utf-8"
    )

    reopened = ProjectService().load_project(document.project_root)

    assert "atomic_spins" not in reopened.structures[structure.id].properties


def test_backfill_keeps_spins_a_structure_already_has(tmp_path: Path) -> None:
    service, document, structure, job_dir = _relaxed_project(tmp_path)
    structure.properties["atomic_spins"] = [0.0, 0.0]
    service.save_project(document)
    (job_dir / "relax_fe-out.cell").write_text(_SPIN_CELL, encoding="utf-8")

    reopened = ProjectService().load_project(document.project_root)

    assert reopened.structures[structure.id].properties["atomic_spins"] == [0.0, 0.0]


def test_backfill_skips_structures_no_relaxation_derived(tmp_path: Path) -> None:
    _service, document, structure, job_dir = _relaxed_project(tmp_path)
    structure.properties["source_workflow_stage"] = WorkflowStage.QUANTUM_CORRECTIONS.value
    ProjectService().save_project(document)
    (job_dir / "relax_fe-out.cell").write_text(_SPIN_CELL, encoding="utf-8")

    reopened = ProjectService().load_project(document.project_root)

    assert "atomic_spins" not in reopened.structures[structure.id].properties

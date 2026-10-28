"""Relaxed structures keep their run's SPIN= values, and old projects get them back."""

from __future__ import annotations

import json
from pathlib import Path

from mufinder.castep_io.cell_parser import parse_cell_text
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationRun, Result, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.relaxed_structure_spins import (
    SPINS_KEPT_SCHEMA_VERSION,
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
    tmp_path: Path,
    *,
    stage: WorkflowStage = WorkflowStage.PRISTINE_RELAXATION,
    link: str = "source_job_id",
) -> tuple[ProjectService, ProjectDocument, Structure, Path]:
    """A saved project with one ``stage`` job and its spin-less derived structure.

    ``link`` is how the structure names its job: ``"source_job_id"``, or
    ``"result"`` for a structure only a completed run's result lists.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = service.add_job(
        document,
        label="relax_fe",
        workflow_stage=stage,
        calc_type="castep_pristine_relaxation",
    )
    properties = (
        {"source_job_id": str(job.id), "source_workflow_stage": stage.value}
        if link == "source_job_id"
        else {}
    )
    structure = Structure(
        label="relax_fe relaxed",
        origin=stage.value,
        lattice_vectors=[[2.87, 0.0, 0.0], [0.0, 2.87, 0.0], [0.0, 0.0, 2.87]],
        species=["Fe", "Fe"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        properties=properties,
    )
    service.add_structure(document, structure)
    if link == "result":
        run = CalculationRun(job_id=job.id, run_index=1)
        result = Result(calculation_run_id=run.id, derived_structure_ids=[structure.id])
        run.result_id = result.id
        document.runs[run.id] = run
        document.results[result.id] = result
    service.save_project(document)
    job_dir = document.project_root / "files" / stage.value / job.label
    job_dir.mkdir(parents=True)
    return service, document, structure, job_dir


def _store_at_schema(document: ProjectDocument, version: int) -> None:
    """Mark the saved project as stored at schema ``version``."""

    manifest_path = document.project_root / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _reopen_schema_5(document: ProjectDocument, loader: ProjectService | None = None):
    """Reopen ``document`` as a project saved before relaxed structures kept spins."""

    _store_at_schema(document, 5)
    return (loader or ProjectService()).load_project(document.project_root)


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
    reopened = _reopen_schema_5(document, loader)
    assert reopened.structures[structure.id].properties["atomic_spins"] == [2.2, 2.2]

    # The next save by the same service persists the recovered spins at schema 6.
    loader.save_project(reopened)
    assert _saved_structure_payload(reopened, structure)["properties"]["atomic_spins"] == [
        2.2,
        2.2,
    ]
    manifest = json.loads((document.project_root / "project.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SPINS_KEPT_SCHEMA_VERSION


def test_a_project_saved_at_schema_6_is_not_backfilled(tmp_path: Path) -> None:
    # A schema 6 structure without spins came from a run that had none.
    _service, document, structure, job_dir = _relaxed_project(tmp_path)
    (job_dir / "relax_fe-out.cell").write_text(_SPIN_CELL, encoding="utf-8")
    _store_at_schema(document, SPINS_KEPT_SCHEMA_VERSION)

    reopened = ProjectService().load_project(document.project_root)

    assert "atomic_spins" not in reopened.structures[structure.id].properties


def test_backfill_finds_the_job_through_its_runs_result(tmp_path: Path) -> None:
    _service, document, structure, job_dir = _relaxed_project(tmp_path, link="result")
    (job_dir / "relax_fe-out.cell").write_text(_SPIN_CELL, encoding="utf-8")

    reopened = _reopen_schema_5(document)

    assert reopened.structures[structure.id].properties["atomic_spins"] == [2.2, 2.2]


def test_backfill_reads_an_out_cell_that_echoes_uncatalogued_settings(tmp_path: Path) -> None:
    # CASTEP echoes run settings into -out.cell; a keyword outside the loaded
    # catalog must not hide the SPIN= values.
    _service, document, structure, job_dir = _relaxed_project(tmp_path)
    echoed = (
        _SPIN_CELL + "\n%BLOCK EXTERNAL_PRESSURE\nGPa\n0 0 0\n0 0\n0\n%ENDBLOCK EXTERNAL_PRESSURE\n"
    )
    (job_dir / "relax_fe-out.cell").write_text(echoed, encoding="utf-8")

    reopened = _reopen_schema_5(document)

    assert reopened.structures[structure.id].properties["atomic_spins"] == [2.2, 2.2]


def test_backfill_falls_back_to_the_input_cell(tmp_path: Path) -> None:
    _service, document, structure, job_dir = _relaxed_project(
        tmp_path, stage=WorkflowStage.MUON_RELAXATION
    )
    (job_dir / "relax_fe-out.cell").write_text(_PLAIN_CELL, encoding="utf-8")
    (job_dir / "relax_fe.cell").write_text(_SPIN_CELL, encoding="utf-8")

    reopened = _reopen_schema_5(document)

    assert reopened.structures[structure.id].properties["atomic_spins"] == [2.2, 2.2]


def test_backfill_leaves_structures_without_recoverable_spins_alone(tmp_path: Path) -> None:
    _service, document, structure, job_dir = _relaxed_project(tmp_path)
    # A non-spin run, and an input cell for a different atom count.
    (job_dir / "relax_fe-out.cell").write_text(_PLAIN_CELL, encoding="utf-8")
    (job_dir / "relax_fe.cell").write_text(
        _SPIN_CELL.replace("Fe 0.5 0.5 0.5 SPIN= 2.200\n", ""), encoding="utf-8"
    )

    reopened = _reopen_schema_5(document)

    assert "atomic_spins" not in reopened.structures[structure.id].properties


def test_backfill_keeps_spins_a_structure_already_has(tmp_path: Path) -> None:
    service, document, structure, job_dir = _relaxed_project(tmp_path)
    structure.properties["atomic_spins"] = [0.0, 0.0]
    service.save_project(document)
    (job_dir / "relax_fe-out.cell").write_text(_SPIN_CELL, encoding="utf-8")

    reopened = _reopen_schema_5(document)

    assert reopened.structures[structure.id].properties["atomic_spins"] == [0.0, 0.0]


def test_backfill_skips_structures_no_relaxation_derived(tmp_path: Path) -> None:
    _service, document, structure, job_dir = _relaxed_project(
        tmp_path, stage=WorkflowStage.QUANTUM_CORRECTIONS
    )
    (job_dir / "relax_fe-out.cell").write_text(_SPIN_CELL, encoding="utf-8")

    reopened = _reopen_schema_5(document)

    assert "atomic_spins" not in reopened.structures[structure.id].properties

"""A project with one completed UEP source job on the Si2 fixture outputs, for the window tests.

``completed_uep_job_with_grid`` attaches the project to a ``MainWindow`` with the
job's ``.castep``, ``.den_fmt`` and ``.cell`` written from ``tests/fixtures/uep``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun
from mufinder.science.uep import UepField, load_uep_source, potential_grid
from mufinder.services.project_service import ProjectService
from tests.support.structures import QUARTER_DIAGONAL, host_structure

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow


uep_host_structure = partial(
    host_structure, positions=QUARTER_DIAGONAL, properties={"atomic_spins": [2.0, -2.0]}
)


def uep_fixture_text(name: str) -> str:
    return (Path(__file__).parent.parent / "fixtures" / "uep" / name).read_text(encoding="utf-8")


def completed_uep_job_with_grid(window: MainWindow, tmp_path: Path):
    """Attach a project with one completed UEP source job and return (doc, job, run, grid)."""

    service = ProjectService()
    doc = service.create_project(tmp_path / "uep_grid_launch.mufinder", "UEPGridLaunch")
    host = uep_host_structure()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    definition = service.register_uep_definition(
        doc,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "uep",
            "rows": [{"label": "row_1", "parameter_values": {}, "added_advanced": []}],
        },
        output_directory=doc.project_root / "files" / "uep",
    )
    doc.ui_state[window._ACTIVE_UEP_DEFINITION_KEY] = str(definition.id)

    job = service.add_job(
        doc,
        label="uep_row_1",
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.COMPLETED)
    doc.runs[run.id] = run
    window._document = doc

    output_dir = doc.project_root / "files" / "uep" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.castep").write_text(
        uep_fixture_text("Si2.castep"), encoding="utf-8"
    )
    (output_dir / f"{job.label}.den_fmt").write_text(
        uep_fixture_text("Si2.den_fmt"), encoding="utf-8"
    )
    (output_dir / f"{job.label}.cell").write_text(uep_fixture_text("Si2.cell"), encoding="utf-8")

    grid = potential_grid(
        UepField(
            load_uep_source(
                den_fmt_path=output_dir / f"{job.label}.den_fmt",
                castep_path=output_dir / f"{job.label}.castep",
                cell_path=output_dir / f"{job.label}.cell",
            )
        )
    )
    return doc, job, run, grid

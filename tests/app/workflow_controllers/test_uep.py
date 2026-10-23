"""Tests for :class:`UepController`.

Covers submission, the ``density_available`` result flag, ``validate_artifacts``
and definition maintenance, including how UEP ranks a defined job against a
failed one compared with Pristine Relaxation.
"""

from __future__ import annotations

from pathlib import Path

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.pristine_relaxation import PristineRelaxationController
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.app.workflow_controllers.uep import UepController, UepDeps
from mufinder.app.workflow_config_codecs import (
    deserialize_pristine_relaxation_config,
    deserialize_uep_config,
    serialize_uep_config,
)

from tests.support.controllers import (
    FakeRelaxationView,
    FakeRowState,
    RecordingBackend,
    job_with_status,
    make_context,
    muon_host_document,
    no_artifacts,
    pristine_sync_definition,
    stub_pristine_deps,
)


# --------------------------------------------------------------------------- #
# UepController
# --------------------------------------------------------------------------- #
#
# One of the simpler migrations: no pre_submit_check, no submission_groups
# override (single group, normalize=str, base-default is_ready), and a small
# build_result contributing only the ``density_available`` boolean flag — no
# derived structures and no document mutation. The tests drive it with narrow
# fakes (no MainWindow) and a real ``.den_fmt`` file on disk to exercise the
# present-vs-absent flag branches.


class _FakeUepView:
    """Minimal UEPWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.runtime_states: list[dict] = []

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


def _stub_uep_deps(**overrides) -> UepDeps:
    base = dict(
        deserialize_uep_config=lambda _payload: None,
        serialize_uep_config=serialize_uep_config,
        uep_row_signature=lambda row: (row.label,),
        uep_potential_grid_result_for_job=lambda _job_id: None,
    )
    base.update(overrides)
    return UepDeps(**base)


def _uep_job(service: ProjectService, document: ProjectDocument, label: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )


def test_uep_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    controller = UepController(make_context(document=document), _stub_uep_deps(), _FakeUepView())
    handled = _uep_job(service, document, "uep_row1")
    other = service.add_job(
        document,
        label="row1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False


def test_uep_controller_uses_workflow_row_ready_predicate() -> None:
    controller = UepController(make_context(), _stub_uep_deps(), _FakeUepView())
    # UEP uses the base-default (wider) workflow-row predicate: "not generated"
    # / "defined" / "ready" rows are ready, only "failed" is excluded.
    assert controller.is_ready({"r": FakeRowState("not generated")}, "r") is True
    assert controller.is_ready({"r": FakeRowState("ready")}, "r") is True
    assert controller.is_ready({"r": FakeRowState("failed")}, "r") is False


def test_uep_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeUepView()
    controller = UepController(make_context(document=None), _stub_uep_deps(), view)
    controller.refresh_view()
    assert view.runtime_states == [{}]


def test_uep_controller_build_result_flag_true_when_den_fmt_present(tmp_path: Path) -> None:
    """build_result reports ``density_available: True`` when the completed job's
    ``{seed}.den_fmt`` output exists on disk, as it always does on the success
    path (``validate_artifacts`` has already guaranteed the file)."""

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _uep_job(service, document, "uep_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}.den_fmt").write_text("density", encoding="utf-8")

    deps = _stub_uep_deps()
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = UepController(context, deps, _FakeUepView())

    parsed = CastepResult(completed=True, total_energy=-20.5, warnings=())
    result = controller.build_result(job, parsed, run, no_artifacts())

    assert result.payload == {"density_available": True}
    # No output_files contributed — the call site already passes them generically.
    assert result.output_files == []
    assert result.derived_structure_ids == []


def test_uep_controller_build_result_flag_false_when_den_fmt_absent(tmp_path: Path) -> None:
    """With no ``{seed}.den_fmt`` on disk, build_result reports
    ``density_available: False``. In production ``validate_artifacts`` fails the
    job before build_result is ever reached, so this branch is only exercised by a
    direct unit call — mirroring the pristine missing-out.cell unit test."""

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _uep_job(service, document, "uep_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()  # deliberately empty — no den_fmt written

    deps = _stub_uep_deps()
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = UepController(context, deps, _FakeUepView())

    parsed = CastepResult(completed=True, total_energy=-20.5, warnings=())
    result = controller.build_result(job, parsed, run, no_artifacts())

    assert result.payload == {"density_available": False}


def test_uep_controller_materialize_writes_job_ids_by_label(tmp_path: Path) -> None:
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = UepConfig(name_prefix="uep", rows=(UepSourceRow(label="row1"),))
    output_root = document.project_root / "files" / "uep"
    definition = service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps(
        deserialize_uep_config=lambda _payload: config,
    )
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = UepController(context, deps, _FakeUepView())

    jobs = controller.materialize(["row1"])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.calc_type == "castep_uep_source"
    mapping = definition.parameters["job_ids_by_label"]
    assert mapping["row1"] == str(job.id)
    # Rendered inputs were written under the output root.
    job_dir = output_root / job.label
    assert (job_dir / f"{job.label}.cell").exists()
    assert (job_dir / f"{job.label}.param").exists()

    # Re-materializing an existing DEFINED job returns the same job (no dup).
    again = controller.materialize(["row1"])
    assert len(again) == 1
    assert again[0].id == job.id


def test_uep_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = UepConfig(name_prefix="uep", rows=(UepSourceRow(label="row1"),))
    output_root = document.project_root / "files" / "uep"
    service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps(
        deserialize_uep_config=lambda _payload: config,
    )

    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = UepController(context, deps, _FakeUepView())

    controller.submit(["row1"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_uep_source"


def test_uep_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = UepConfig(name_prefix="uep", rows=(UepSourceRow(label="row1"),))
    output_root = document.project_root / "files" / "uep"
    service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps(
        deserialize_uep_config=lambda _payload: config,
    )

    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = UepController(context, deps, _FakeUepView())

    controller.submit(["row1"], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    submitted, extras = cluster.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_uep_source"


def test_uep_controller_validate_artifacts_requires_den_fmt(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _uep_job(service, document, "uep_row1")
    run = CalculationRun(job_id=job.id)
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    parsed = CastepResult(completed=True, total_energy=-1.0)
    missing = "Expected formatted density output was not found: uep_row1.den_fmt"

    def _controller(directory):
        context = make_context(document=document, job_output_directory=lambda _job: directory)
        return UepController(context, _stub_uep_deps(), _FakeUepView())

    assert _controller(output_dir).validate_artifacts(job, parsed, run, no_artifacts()) == missing
    assert _controller(None).validate_artifacts(job, parsed, run, no_artifacts()) == missing
    (output_dir / "uep_row1.den_fmt").write_text("density", encoding="utf-8")
    assert _controller(output_dir).validate_artifacts(job, parsed, run, no_artifacts()) is None


def _uep_sync_definition(service, document, host, job_ids_by_label=None):
    return service.register_uep_definition(
        document,
        input_structure_id=host.id,
        parameters=serialize_uep_config(UepConfig(rows=(UepSourceRow(label="row_1"),))),
        output_directory=document.project_root / "files" / "uep",
        job_ids_by_label=job_ids_by_label,
    )


def test_uep_and_pristine_sync_rank_a_defined_job_against_a_failed_one(tmp_path: Path) -> None:
    """UEP keeps the convergence order (failed beats defined); the relaxations prefer defined."""

    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)

    def jobs(stage: WorkflowStage, calc_type: str) -> tuple[str, str]:
        return (
            job_with_status(service, document, JobStatus.FAILED, stage=stage, calc_type=calc_type),
            job_with_status(service, document, JobStatus.DEFINED, stage=stage, calc_type=calc_type),
        )

    uep_failed, uep_defined = jobs(WorkflowStage.UEP, "castep_uep_source")
    _uep_sync_definition(service, document, host, {"row_1": uep_failed})
    _uep_sync_definition(service, document, host, {"row_1": uep_defined})
    uep_target = _uep_sync_definition(service, document, host)
    relax_failed, relax_defined = jobs(
        WorkflowStage.PRISTINE_RELAXATION, "castep_pristine_relaxation"
    )
    pristine_sync_definition(service, document, host, job_ids_by_label={"row_1": relax_failed})
    pristine_sync_definition(service, document, host, job_ids_by_label={"row_1": relax_defined})
    pristine_target = pristine_sync_definition(service, document, host)
    context = make_context(document=document)
    uep = UepController(
        context, _stub_uep_deps(deserialize_uep_config=deserialize_uep_config), _FakeUepView()
    )
    pristine = PristineRelaxationController(
        context,
        stub_pristine_deps(
            deserialize_pristine_relaxation_config=deserialize_pristine_relaxation_config
        ),
        FakeRelaxationView(),
    )

    uep.sync_mappings(uep_target)
    pristine.sync_mappings(pristine_target)

    assert uep_target.parameters["job_ids_by_label"] == {"row_1": uep_failed}
    assert pristine_target.parameters["job_ids_by_label"] == {"row_1": relax_defined}


def test_uep_controller_remove_rows_drops_the_row_binding(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    definition = _uep_sync_definition(service, document, host, {"row_1": "1"})
    controller = UepController(
        make_context(document=document),
        _stub_uep_deps(deserialize_uep_config=deserialize_uep_config),
        _FakeUepView(),
    )

    assert controller.remove_rows(["row_1"]) == (definition.id,)
    assert definition.id not in document.calculations

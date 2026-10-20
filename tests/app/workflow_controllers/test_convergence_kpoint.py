"""Tests for :class:`ConvergenceKpointController`.

The controller is tested against a real ``ProjectService`` document with fake
submit backends, pinning the ``job_ids_by_grid`` mapping, its legacy lookups and
its definition maintenance.
"""

from __future__ import annotations

from pathlib import Path

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.base import WorkflowControllerContext
from mufinder.app.workflow_controllers.convergence_kpoint import (
    ConvergenceKpointController,
    ConvergenceKpointDeps,
)
from mufinder.app.workflow_config_codecs import deserialize_convergence_config
from mufinder.app.workflow_submission import WorkflowSubmissionCoordinator

from tests.support.jobs import make_job
from tests.support.controllers import (
    RecordingBackend,
    job_with_status,
    make_context,
    muon_host_document,
)


# --------------------------------------------------------------------------- #
# ConvergenceKpointController
# --------------------------------------------------------------------------- #


class _FakeKpointView:
    """Minimal KPointConvergenceWidget stand-in for controller tests."""

    def __init__(self, form_state: dict | None = None) -> None:
        self._form_state = form_state or {}
        self.runtime_states: list[dict] = []

    def collect_state(self) -> dict:
        return self._form_state

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


def _grid_key(grid: tuple[int, int, int]) -> str:
    return f"{int(grid[0])}x{int(grid[1])}x{int(grid[2])}"


def _stub_kpoint_deps() -> ConvergenceKpointDeps:
    """Deps whose helpers all report 'nothing found' — for tests that never
    reach the definition-backed paths."""

    return ConvergenceKpointDeps(
        deserialize_convergence_config=lambda _payload: None,
        grid_key=_grid_key,
        max_stress_for_job=lambda _job_id: None,
    )


def _kpoint_context_and_deps(
    service: ProjectService,
    document: ProjectDocument,
    *,
    local_backend: RecordingBackend,
    cluster_backend: RecordingBackend,
) -> tuple[WorkflowControllerContext, ConvergenceKpointDeps]:
    """Shared context + kpoint deps wired with fake helpers.

    The kpoint-specific helpers below mirror the (unchanged) MainWindow versions
    closely enough to drive materialize/build_states without a MainWindow.
    """

    next(iter(document.calculations.values()), None)

    def deserialize(payload):
        return document.ui_state.get("_test_config")

    context = WorkflowControllerContext(
        document=lambda: document,
        project_service=service,
        job_execution=None,  # type: ignore[arg-type]
        submission_coordinator=WorkflowSubmissionCoordinator(),
        submit_jobs_locally=local_backend,
        submit_jobs_on_cluster=cluster_backend,
        prepare_jobs_for_resubmission=lambda _jobs: True,
        show_info=lambda *_: None,
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=lambda **_: None,
        status_message=lambda _m: None,
        definition_output_directory=lambda d: document.project_root / "files" / "convergence_kpoint",
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
        job_file_cleanup=None,  # type: ignore[arg-type]
        run_for_job=lambda _job_id: None,
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
    )
    deps = ConvergenceKpointDeps(
        deserialize_convergence_config=deserialize,
        grid_key=_grid_key,
        max_stress_for_job=lambda _job_id: None,
    )
    return context, deps


def _seed_kpoint_document(service: ProjectService, document: ProjectDocument):
    from mufinder.app.convergence_types import ConvergenceConfig
    from mufinder.domain.models import Structure

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = ConvergenceConfig(
        name_prefix="kconv",
        sweep_parameter="kpoint_mp_grid",
        sweep_values=((2, 2, 2),),
        shared_params={"task": "SINGLEPOINT"},
    )
    definition = service.register_convergence_kpoint_definition(
        document,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=document.project_root / "files" / "convergence_kpoint",
    )
    document.ui_state["_test_config"] = config
    # Persist the form so the controller's form_snapshot agrees with the active definition.
    document.ui_state["kpoint_convergence_form"] = {
        "name_prefix": "kconv",
        "kpoint_grids": [[2, 2, 2]],
    }
    return definition, config


def test_kpoint_controller_materialize_writes_job_ids_by_grid(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    definition, _config = _seed_kpoint_document(service, document)

    local = RecordingBackend()
    cluster = RecordingBackend()
    view = _FakeKpointView()
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceKpointController(context, deps, view)

    jobs = controller.materialize([(2, 2, 2)])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.calc_type == "castep_convergence_kpoint"
    mapping = definition.parameters["job_ids_by_grid"]
    assert mapping[_grid_key((2, 2, 2))] == str(job.id)
    # Rendered inputs were written.
    root = document.project_root / "files" / "convergence_kpoint" / job.label
    assert (root / f"{job.label}.cell").exists()
    assert (root / f"{job.label}.param").exists()


def test_kpoint_controller_materialize_rewrites_inputs_for_defined_job(
    tmp_path: Path,
) -> None:
    """Same invariant as the cutoff variant: defined jobs get fresh inputs."""

    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())

    (job,) = controller.materialize([(2, 2, 2)])
    param_path = (
        document.project_root / "files" / "convergence_kpoint" / job.label / f"{job.label}.param"
    )
    param_path.write_text("STALE")

    again = controller.materialize([(2, 2, 2)])

    assert [j.id for j in again] == [job.id]
    assert len(document.jobs) == 1
    assert param_path.read_text() != "STALE"


def test_kpoint_controller_materialize_keeps_non_defined_job_inputs(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())

    (job,) = controller.materialize([(2, 2, 2)])
    job.status = JobStatus.RUNNING
    param_path = (
        document.project_root / "files" / "convergence_kpoint" / job.label / f"{job.label}.param"
    )
    param_path.write_text("STALE")

    again = controller.materialize([(2, 2, 2)])

    assert [j.id for j in again] == [job.id]
    assert param_path.read_text() == "STALE"


def test_kpoint_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    local = RecordingBackend()
    cluster = RecordingBackend()
    view = _FakeKpointView()
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceKpointController(context, deps, view)

    controller.submit([(2, 2, 2)], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_convergence_kpoint"


def test_kpoint_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    local = RecordingBackend()
    cluster = RecordingBackend()
    view = _FakeKpointView()
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceKpointController(context, deps, view)

    controller.submit([(2, 2, 2)], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == "castep_convergence_kpoint"


def test_kpoint_controller_uses_shared_ready_predicate() -> None:
    controller = ConvergenceKpointController(make_context(), _stub_kpoint_deps(), _FakeKpointView())
    from mufinder.app.convergence_types import SweepPointRunState

    for ready_status in ("ready", "defined", "not generated"):
        states = {(2, 2, 2): SweepPointRunState(status=ready_status)}
        assert controller.is_ready(states, (2, 2, 2)) is True
    states = {(2, 2, 2): SweepPointRunState(status="running")}
    assert controller.is_ready(states, (2, 2, 2)) is False


def test_kpoint_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeKpointView()
    controller = ConvergenceKpointController(
        make_context(document=None), _stub_kpoint_deps(), view
    )

    controller.refresh_view()

    assert view.runtime_states == [{}]


def test_kpoint_controller_build_states_reflects_form_ready_grids(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())

    states = controller.build_states()

    assert (2, 2, 2) in states
    assert states[(2, 2, 2)].status == "ready"


def test_kpoint_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    controller = ConvergenceKpointController(
        make_context(document=document), _stub_kpoint_deps(), _FakeKpointView()
    )
    handled = make_job("j", service=service, document=document, calc_type="castep_convergence_kpoint")
    other = make_job("k", service=service, document=document, calc_type="castep_convergence_cutoff")
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False


def test_kpoint_controller_form_snapshot_reads_the_live_widget_first(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    # The persisted form lists (2, 2, 2) only; the widget has since gained a grid.
    view = _FakeKpointView(form_state={"name_prefix": "kconv", "kpoint_grids": [[2, 2, 2], [3, 3, 3]]})
    controller = ConvergenceKpointController(context, deps, view)

    assert controller.form_snapshot() == ("kconv", ((2, 2, 2), (3, 3, 3)))
    assert set(controller.build_states()) == {(2, 2, 2), (3, 3, 3)}

    # An empty widget falls back to the persisted form.
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())
    assert controller.form_snapshot() == ("kconv", ((2, 2, 2),))


def test_kpoint_controller_finds_a_legacy_definitions_job_by_label(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    definition, _config = _seed_kpoint_document(service, document)
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())
    (job,) = controller.materialize([(2, 2, 2)])

    # A definition written before job_ids_by_grid existed.
    del definition.parameters["job_ids_by_grid"]

    assert controller.job_for_sweep_key(definition, (2, 2, 2)) is job
    assert controller.job_for_sweep_key(definition, (3, 3, 3)) is None


def _kpoint_sync_definition(service, document, host, *, overrides, job_ids_by_grid=None):
    return service.register_convergence_kpoint_definition(
        document,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2], [3, 3, 3]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": overrides,
        },
        output_directory=document.project_root / "files" / "convergence_kpoint",
        job_ids_by_grid=job_ids_by_grid,
    )


def test_kpoint_controller_sync_mappings_respells_keys_and_compares_effective_params(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)

    def job() -> str:
        return job_with_status(
            service,
            document,
            JobStatus.COMPLETED,
            stage=WorkflowStage.CONVERGENCE_KPOINT,
            calc_type="castep_convergence_kpoint",
        )

    completed = job()
    overridden = job()
    _kpoint_sync_definition(
        service,
        document,
        host,
        overrides=[{"sweep_values": [[3, 3, 3]], "params": {"max_scf_cycles": 50}}],
        job_ids_by_grid={"02x2x2": completed, "3x3x3": overridden, "bad": completed},
    )
    target = _kpoint_sync_definition(service, document, host, overrides=[])
    controller = ConvergenceKpointController(
        make_context(document=document),
        ConvergenceKpointDeps(
            deserialize_convergence_config=deserialize_convergence_config,
            grid_key=_grid_key,
            max_stress_for_job=lambda _job_id: None,
        ),
        _FakeKpointView(),
    )

    controller.sync_mappings(target)

    # The source's 3x3x3 point ran with an override the target does not have.
    assert target.parameters["job_ids_by_grid"] == {"2x2x2": completed}


def test_kpoint_controller_remove_rows_keeps_the_definition(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    definition = _kpoint_sync_definition(
        service, document, host, overrides=[], job_ids_by_grid={"2x2x2": "1", "3x3x3": "2"}
    )
    controller = ConvergenceKpointController(
        make_context(document=document), _stub_kpoint_deps(), _FakeKpointView()
    )

    assert controller.remove_rows(["2x2x2", "3x3x3"]) == ()

    assert definition.parameters["configuration"]["sweep_values"] == []
    assert definition.parameters["job_ids_by_grid"] == {}
    assert definition.id in document.calculations

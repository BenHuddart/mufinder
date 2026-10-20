"""Tests for :class:`ConvergenceCutoffController`.

The controller is tested against a real ``ProjectService`` document with fake
submit backends, pinning that it reaches the right backend with the right jobs
and writes the ``job_ids_by_cutoff`` mapping. The shared sweep behaviour of
:class:`SweepConvergenceController` is exercised through this controller.
"""

from __future__ import annotations

from pathlib import Path

from mufinder.domain.enums import JobStatus
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.base import WorkflowControllerContext
from mufinder.app.workflow_controllers.convergence_cutoff import (
    ConvergenceCutoffController,
    ConvergenceCutoffDeps,
)
from mufinder.app.workflow_submission import WorkflowSubmissionCoordinator

from tests.support.jobs import make_job
from tests.support.controllers import RecordingBackend, make_context


# --------------------------------------------------------------------------- #
# ConvergenceCutoffController
# --------------------------------------------------------------------------- #


class _FakeConvergenceView:
    """Minimal CutoffConvergenceWidget stand-in for controller tests."""

    def __init__(self, form_state: dict | None = None) -> None:
        self._form_state = form_state or {}
        self.runtime_states: list[dict] = []

    def collect_state(self) -> dict:
        return self._form_state

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


def _stub_deps() -> ConvergenceCutoffDeps:
    """Deps whose helpers all report 'nothing found' — for tests that never
    reach the definition-backed paths."""

    return ConvergenceCutoffDeps(
        deserialize_convergence_config=lambda _payload: None,
        cutoff_key=lambda value: f"{float(value):.12g}",
        max_stress_for_job=lambda _job_id: None,
    )


def _cutoff_context_and_deps(
    service: ProjectService,
    document: ProjectDocument,
    *,
    local_backend: RecordingBackend,
    cluster_backend: RecordingBackend,
) -> tuple[WorkflowControllerContext, ConvergenceCutoffDeps]:
    """Shared context + cutoff deps wired with fake helpers.

    The cutoff-specific helpers below mirror the (unchanged) MainWindow versions
    closely enough to drive materialize/build_states without a MainWindow.
    """

    def deserialize(payload):
        # The test seeds a ConvergenceConfig directly on the definition.
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
        definition_output_directory=lambda d: document.project_root / "files" / "convergence_cutoff",
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
        job_file_cleanup=None,  # type: ignore[arg-type]
        run_for_job=lambda _job_id: None,
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
    )
    deps = ConvergenceCutoffDeps(
        deserialize_convergence_config=deserialize,
        cutoff_key=lambda value: f"{float(value):.12g}",
        max_stress_for_job=lambda _job_id: None,
    )
    return context, deps


def _seed_convergence_document(service: ProjectService, document: ProjectDocument):
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

    definition = service.register_convergence_cutoff_definition(
        document,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_values": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=document.project_root / "files" / "convergence_cutoff",
    )
    config = ConvergenceConfig(
        sweep_values=(300.0,),
        name_prefix="conv",
        shared_params={"task": "SINGLEPOINT"},
    )
    document.ui_state["_test_config"] = config
    return definition, config


def test_convergence_controller_materialize_writes_job_ids_by_cutoff(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    definition, _config = _seed_convergence_document(service, document)

    local = RecordingBackend()
    cluster = RecordingBackend()
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceCutoffController(context, deps, view)

    jobs = controller.materialize([300.0])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.label == "conv_ecut_300"
    assert job.calc_type == "castep_convergence_cutoff"
    mapping = definition.parameters["job_ids_by_cutoff"]
    assert mapping[f"{300.0:.12g}"] == str(job.id)
    # Rendered inputs were written.
    root = document.project_root / "files" / "convergence_cutoff" / "conv_ecut_300"
    assert (root / "conv_ecut_300.cell").exists()
    assert (root / "conv_ecut_300.param").exists()


def test_convergence_controller_materialize_rewrites_inputs_for_defined_job(
    tmp_path: Path,
) -> None:
    """A defined job's inputs are regenerated on re-materialize.

    This is what makes the post-reset ``materialize`` pass in
    ``WorkflowJobController.submit`` effective for convergence rows: a job reset
    back to ``defined`` after dashboard dismissal must not resubmit against the
    stale ``.cell``/``.param`` files its failed run left behind.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    controller = ConvergenceCutoffController(context, deps, view)

    (job,) = controller.materialize([300.0])
    param_path = (
        document.project_root
        / "files"
        / "convergence_cutoff"
        / "conv_ecut_300"
        / "conv_ecut_300.param"
    )
    param_path.write_text("STALE")

    again = controller.materialize([300.0])

    assert [j.id for j in again] == [job.id]
    assert len(document.jobs) == 1
    assert param_path.read_text() != "STALE"


def test_convergence_controller_materialize_keeps_non_defined_job_inputs(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    controller = ConvergenceCutoffController(context, deps, view)

    (job,) = controller.materialize([300.0])
    job.status = JobStatus.RUNNING
    param_path = (
        document.project_root
        / "files"
        / "convergence_cutoff"
        / "conv_ecut_300"
        / "conv_ecut_300.param"
    )
    param_path.write_text("STALE")

    again = controller.materialize([300.0])

    assert [j.id for j in again] == [job.id]
    assert param_path.read_text() == "STALE"


def test_convergence_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    local = RecordingBackend()
    cluster = RecordingBackend()
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceCutoffController(context, deps, view)

    controller.submit([300.0], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert [j.label for j in submitted] == ["conv_ecut_300"]


def test_convergence_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    local = RecordingBackend()
    cluster = RecordingBackend()
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceCutoffController(context, deps, view)

    controller.submit([300.0], target="cluster")

    assert local.calls == []
    assert [j.label for j in cluster.calls[0][0]] == ["conv_ecut_300"]


def test_convergence_controller_uses_shared_ready_predicate() -> None:
    controller = ConvergenceCutoffController(make_context(), _stub_deps(), _FakeConvergenceView())
    # Cutoff readiness now uses the shared workflow-row predicate, so the same
    # states that are submittable for every other workflow ("ready", "defined",
    # "not generated") are submittable here too; active states are not.
    from mufinder.app.convergence_types import SweepPointRunState

    for ready_status in ("ready", "defined", "not generated"):
        states = {300.0: SweepPointRunState(status=ready_status)}
        assert controller.is_ready(states, 300.0) is True
    states = {300.0: SweepPointRunState(status="running")}
    assert controller.is_ready(states, 300.0) is False


def test_convergence_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeConvergenceView()
    controller = ConvergenceCutoffController(make_context(document=None), _stub_deps(), view)

    controller.refresh_view()

    assert view.runtime_states == [{}]


def test_convergence_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    controller = ConvergenceCutoffController(
        make_context(document=document), _stub_deps(), _FakeConvergenceView()
    )
    handled = make_job("j", service=service, document=document, calc_type="castep_convergence_cutoff")
    other = make_job("k", service=service, document=document, calc_type="castep_uep_source")
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False


def test_cutoff_controller_finds_a_legacy_definitions_job_through_job_labels(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    definition, _config = _seed_convergence_document(service, document)
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    controller = ConvergenceCutoffController(context, deps, view)
    (job,) = controller.materialize([300.0])

    del definition.parameters["job_ids_by_cutoff"]
    definition.parameters["job_labels"] = [job.label]

    assert controller.job_for_cutoff(definition, 300.0) is job
    assert controller.job_for_cutoff(definition, 400.0) is None
    del definition.parameters["job_labels"]
    assert controller.job_for_cutoff(definition, 300.0) is None


def test_sweep_controllers_resolve_each_job_once(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=RecordingBackend(), cluster_backend=RecordingBackend()
    )
    controller = ConvergenceCutoffController(context, deps, view)
    (job,) = controller.materialize([300.0, 300.0])

    assert controller.jobs_for_keys([300.0, "300", 450.0, "not a cutoff"]) == [job]


def test_cutoff_controller_remove_cutoff_deletes_an_emptied_definition(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "c.mufinder", "C")
    definition, _config = _seed_convergence_document(service, document)
    definition.parameters["job_ids_by_cutoff"] = {"300": "1"}
    controller = ConvergenceCutoffController(
        make_context(document=document), _stub_deps(), _FakeConvergenceView()
    )
    document.ui_state[controller.active_key] = str(definition.id)

    assert controller.remove_cutoff_from_definition(definition, 300.0) is True

    assert definition.parameters["job_ids_by_cutoff"] == {}
    assert definition.id not in document.calculations
    assert controller.active_key not in document.ui_state

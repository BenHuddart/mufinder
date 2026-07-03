"""Tests for the per-workflow controller base class and cutoff migration.

The base :class:`WorkflowJobController` is exercised through a *fake* controller
so that the template-method ``submit`` flow, the no-ready path, the grouped
submission override, and the ``merge_controller_result`` result-dispatch hook are
proven independently of any concrete workflow (the six future migrations depend
on these). :class:`ConvergenceCutoffController` is then tested against a real
``ProjectService`` document with fake submit backends, pinning that it reaches
the right backend with the right jobs and writes the ``job_ids_by_cutoff``
mapping identically to the former ``MainWindow`` behavior.

Both use narrow fakes for the :class:`WorkflowControllerContext` callables — no
``MainWindow`` is constructed — which is exactly the decoupling the controller
layer is meant to provide.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_execution_controller import JobExecutionController
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    merge_controller_result,
)
from mufinder.app.workflow_controllers.convergence_cutoff import (
    ConvergenceCutoffController,
    ConvergenceCutoffDeps,
)
from mufinder.app.workflow_controllers.convergence_kpoint import (
    ConvergenceKpointController,
    ConvergenceKpointDeps,
)
from mufinder.app.workflow_submission import WorkflowSubmissionCoordinator


# --------------------------------------------------------------------------- #
# Shared fakes
# --------------------------------------------------------------------------- #


class _RecordingBackend:
    """Records ``(jobs, kwargs)`` for each submit call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Job], dict[str, object]]] = []

    def __call__(self, jobs: list[Job], **kwargs: object) -> None:
        self.calls.append((list(jobs), dict(kwargs)))


def _make_context(
    *,
    document: ProjectDocument | None = None,
    project_service: ProjectService | None = None,
    local_backend: _RecordingBackend | None = None,
    cluster_backend: _RecordingBackend | None = None,
    infos: list[tuple[str, str]] | None = None,
) -> WorkflowControllerContext:
    service = project_service or ProjectService()
    return WorkflowControllerContext(
        document=lambda: document,
        project_service=service,
        job_execution=None,  # type: ignore[arg-type]  # unused by these tests
        submission_coordinator=WorkflowSubmissionCoordinator(),
        submit_jobs_locally=local_backend or _RecordingBackend(),
        submit_jobs_on_cluster=cluster_backend or _RecordingBackend(),
        show_info=(lambda title, text: infos.append((title, text))) if infos is not None else (lambda *_: None),
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=lambda **_: None,
        status_message=lambda _message: None,
    )


class _State:
    """Minimal state object with a ``status`` and optional ``job_id``."""

    def __init__(self, status: str, job_id: object | None = None) -> None:
        self.status = status
        self.job_id = job_id


class _FakeController(WorkflowJobController):
    """Fake controller exercising the base template method + result hook."""

    calc_types = frozenset({"fake_calc"})
    no_ready_message = "No ready fake rows."

    def __init__(
        self,
        context: WorkflowControllerContext,
        *,
        states: dict[str, _State],
        existing: dict[str, Job] | None = None,
        materialized: dict[str, Job] | None = None,
        group_by_size: bool = False,
        pre_submit_ok: bool = True,
        build_result_payload: dict[str, object] | None = None,
    ) -> None:
        super().__init__(context)
        self._states = states
        self._existing = existing or {}
        self._materialized = materialized or {}
        self._group_by_size = group_by_size
        self._pre_submit_ok = pre_submit_ok
        self._build_result_payload = build_result_payload
        self.refreshed = 0

    def normalize(self, value: object) -> str:
        return str(value)

    def build_states(self) -> dict[str, _State]:
        return self._states

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        return [self._existing[k] for k in keys if k in self._existing]

    def materialize(self, keys: list[object]) -> list[Job]:
        return [self._materialized[k] for k in keys if k in self._materialized]

    def pre_submit_check(self, keys: list[object]) -> bool:
        return self._pre_submit_ok

    def submission_groups(self, jobs: list[Job]):
        if not self._group_by_size:
            return super().submission_groups(jobs)
        # Split into per-job groups, each carrying an index kwarg, to prove the
        # override path forwards extra kwargs to the backend.
        return [([job], {"index": i}) for i, job in enumerate(jobs)]

    def build_result(self, job: Job, parsed: CastepResult, run: CalculationRun) -> ControllerResult:
        if self._build_result_payload is None:
            return super().build_result(job, parsed, run)
        return ControllerResult(
            payload=dict(self._build_result_payload),
            output_files=["extra.out"],
            warnings=["controller warning"],
        )

    def refresh_view(self) -> None:
        self.refreshed += 1


def _job(service: ProjectService, document: ProjectDocument, label: str, calc_type: str = "fake_calc") -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type=calc_type,
    )


# --------------------------------------------------------------------------- #
# Base template-method: submit()
# --------------------------------------------------------------------------- #


def test_submit_local_dispatches_planned_jobs_to_local_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    existing = _job(service, document, "existing")
    fresh = _job(service, document, "fresh")

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(local_backend=local, cluster_backend=cluster)
    controller = _FakeController(
        context,
        states={"a": _State("ready"), "b": _State("ready")},
        existing={"a": existing},
        materialized={"b": fresh},
    )

    controller.submit(["a", "b"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted_jobs, extras = local.calls[0]
    assert [j.id for j in submitted_jobs] == [existing.id, fresh.id]
    assert extras == {}


def test_submit_cluster_dispatches_to_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    fresh = _job(service, document, "fresh")

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(local_backend=local, cluster_backend=cluster)
    controller = _FakeController(
        context,
        states={"b": _State("ready")},
        materialized={"b": fresh},
    )

    controller.submit(["b"], target="cluster")

    assert local.calls == []
    assert [j.id for j in cluster.calls[0][0]] == [fresh.id]


def test_submit_shows_info_when_no_ready_jobs(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(local_backend=local, infos=infos)
    controller = _FakeController(
        context,
        # No 'ready' rows and no jobs → empty plan.
        states={"a": _State("failed")},
    )

    controller.submit(["a"], target="local")

    assert local.calls == []
    assert infos == [("No ready jobs", "No ready fake rows.")]


def test_submit_pre_submit_check_short_circuits(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    fresh = _job(service, document, "fresh")
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(local_backend=local, infos=infos)
    controller = _FakeController(
        context,
        states={"b": _State("ready")},
        materialized={"b": fresh},
        pre_submit_ok=False,
    )

    controller.submit(["b"], target="local")

    assert local.calls == []
    assert infos == []  # pre-submit refusal is silent (no "no ready jobs")


def test_submit_grouped_override_forwards_extra_kwargs(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job_a = _job(service, document, "a")
    job_b = _job(service, document, "b")
    local = _RecordingBackend()
    context = _make_context(local_backend=local)
    controller = _FakeController(
        context,
        states={"a": _State("ready"), "b": _State("ready")},
        materialized={"a": job_a, "b": job_b},
        group_by_size=True,
    )

    controller.submit(["a", "b"], target="local")

    assert [[j.id for j in jobs] for jobs, _ in local.calls] == [[job_a.id], [job_b.id]]
    assert [extras for _, extras in local.calls] == [{"index": 0}, {"index": 1}]


def test_default_submission_groups_is_single_group() -> None:
    context = _make_context()
    controller = _FakeController(context, states={})
    jobs: list[Job] = []
    assert controller.submission_groups(jobs) == [([], {})]


def test_default_is_ready_uses_workflow_row_predicate() -> None:
    context = _make_context()
    controller = _FakeController(context, states={})
    # Base default accepts the wider workflow-row status set.
    assert controller.is_ready({"k": _State("not generated")}, "k") is True
    assert controller.is_ready({"k": _State("failed")}, "k") is False


# --------------------------------------------------------------------------- #
# Base result dispatch: merge_controller_result
# --------------------------------------------------------------------------- #


def _parsed() -> CastepResult:
    return CastepResult(completed=True, total_energy=-1.0, warnings=("parse warning",))


def test_merge_controller_result_no_matching_controller_is_noop(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _job(service, document, "j", calc_type="unhandled_calc")
    run = CalculationRun(job_id=job.id)
    controllers = [_FakeController(_make_context(), states={})]

    payload = {"total_energy": -1.0}
    merged = merge_controller_result(
        controllers,
        job,
        _parsed(),
        run,
        payload=payload,
        output_files=["seed.castep"],
        derived_structure_ids=[],
        warnings=["parse warning"],
    )

    assert merged == ({"total_energy": -1.0}, ["seed.castep"], [], ["parse warning"])
    # The caller's inputs were not mutated.
    assert payload == {"total_energy": -1.0}


def test_merge_controller_result_empty_result_is_noop(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _job(service, document, "j", calc_type="fake_calc")
    run = CalculationRun(job_id=job.id)
    # Fake controller with no build_result payload returns the empty default.
    controllers = [_FakeController(_make_context(), states={})]

    merged_payload, files, structures, warnings = merge_controller_result(
        controllers,
        job,
        _parsed(),
        run,
        payload={"total_energy": -1.0},
        output_files=["seed.castep"],
        derived_structure_ids=[],
        warnings=["parse warning"],
    )

    assert merged_payload == {"total_energy": -1.0}
    assert files == ["seed.castep"]
    assert structures == []
    assert warnings == ["parse warning"]


def test_merge_controller_result_additively_merges_matching_controller(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _job(service, document, "j", calc_type="fake_calc")
    run = CalculationRun(job_id=job.id)
    controller = _FakeController(
        _make_context(),
        states={},
        build_result_payload={"custom_metric": 42, "total_energy": -2.0},
    )

    merged_payload, files, structures, warnings = merge_controller_result(
        [controller],
        job,
        _parsed(),
        run,
        payload={"total_energy": -1.0, "completed": True},
        output_files=["seed.castep"],
        derived_structure_ids=[],
        warnings=["parse warning"],
    )

    # Controller payload updates over the base; files/warnings are appended.
    assert merged_payload == {"total_energy": -2.0, "completed": True, "custom_metric": 42}
    assert files == ["seed.castep", "extra.out"]
    assert structures == []
    assert warnings == ["parse warning", "controller warning"]


def test_merge_controller_result_selects_first_handling_controller(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _job(service, document, "j", calc_type="fake_calc")
    run = CalculationRun(job_id=job.id)

    class _OtherController(_FakeController):
        calc_types = frozenset({"other_calc"})

    other = _OtherController(_make_context(), states={}, build_result_payload={"x": 1})
    handler = _FakeController(_make_context(), states={}, build_result_payload={"custom_metric": 7})

    merged_payload, *_ = merge_controller_result(
        [other, handler],
        job,
        _parsed(),
        run,
        payload={},
        output_files=[],
        derived_structure_ids=[],
        warnings=[],
    )

    assert merged_payload == {"custom_metric": 7}


# --------------------------------------------------------------------------- #
# ConvergenceCutoffController
# --------------------------------------------------------------------------- #


class _FakeConvergenceView:
    """Minimal ConvergenceWidget stand-in for controller tests."""

    def __init__(self, form_state: dict | None = None) -> None:
        self._form_state = form_state or {}
        self.runtime_states: list[dict] = []

    def collect_state(self) -> dict:
        return self._form_state

    def set_cutoff_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


def _stub_deps() -> ConvergenceCutoffDeps:
    """Deps whose helpers all report 'nothing found' — for tests that never
    reach the definition-backed paths."""

    return ConvergenceCutoffDeps(
        active_convergence_definition=lambda: None,
        deserialize_convergence_config=lambda _payload: None,
        sync_equivalent_convergence_cutoff_mappings=lambda _definition: None,
        definition_for_convergence_cutoff=lambda _cutoff: (None, None),
        job_for_definition_cutoff=lambda _definition, _cutoff: None,
        definition_output_directory=lambda _definition: None,
        cutoff_key=lambda value: f"{float(value):.12g}",
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
        max_stress_for_job=lambda _job_id: None,
    )


def _cutoff_context_and_deps(
    service: ProjectService,
    document: ProjectDocument,
    *,
    local_backend: _RecordingBackend,
    cluster_backend: _RecordingBackend,
) -> tuple[WorkflowControllerContext, ConvergenceCutoffDeps]:
    """Shared context + cutoff deps wired with fake helpers.

    The cutoff-specific helpers below mirror the (unchanged) MainWindow versions
    closely enough to drive materialize/build_states without a MainWindow.
    """

    active_definition = next(iter(document.calculations.values()), None)

    def deserialize(payload):
        # The test seeds a ConvergenceConfig directly on the definition.
        return document.ui_state.get("_test_config")

    def definition_for_cutoff(cutoff):
        return active_definition, document.ui_state.get("_test_config")

    def job_for_definition_cutoff(definition, cutoff):
        mapping = definition.parameters.get("job_ids_by_cutoff", {}) if definition else {}
        job_id = mapping.get(f"{float(cutoff):.12g}")
        return document.jobs.get(job_id) if job_id else None

    context = WorkflowControllerContext(
        document=lambda: document,
        project_service=service,
        job_execution=None,  # type: ignore[arg-type]
        submission_coordinator=WorkflowSubmissionCoordinator(),
        submit_jobs_locally=local_backend,
        submit_jobs_on_cluster=cluster_backend,
        show_info=lambda *_: None,
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=lambda **_: None,
        status_message=lambda _m: None,
    )
    deps = ConvergenceCutoffDeps(
        active_convergence_definition=lambda: active_definition,
        deserialize_convergence_config=deserialize,
        sync_equivalent_convergence_cutoff_mappings=lambda _d: None,
        definition_for_convergence_cutoff=definition_for_cutoff,
        job_for_definition_cutoff=job_for_definition_cutoff,
        definition_output_directory=lambda d: document.project_root / "files" / "convergence_cutoff",
        cutoff_key=lambda value: f"{float(value):.12g}",
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
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
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=document.project_root / "files" / "convergence_cutoff",
    )
    config = ConvergenceConfig(
        name_prefix="conv",
        cutoff_energies=(300.0,),
        shared_params={"task": "SINGLEPOINT"},
    )
    document.ui_state["_test_config"] = config
    return definition, config


def test_convergence_controller_materialize_writes_job_ids_by_cutoff(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    definition, _config = _seed_convergence_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
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


def test_convergence_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
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

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceCutoffController(context, deps, view)

    controller.submit([300.0], target="cluster")

    assert local.calls == []
    assert [j.label for j in cluster.calls[0][0]] == ["conv_ecut_300"]


def test_convergence_controller_uses_sweep_point_ready_predicate() -> None:
    controller = ConvergenceCutoffController(_make_context(), _stub_deps(), _FakeConvergenceView())
    # Cutoff readiness is the strict "ready"-only predicate, not the wider
    # workflow-row set — a "defined" row must not be treated as ready.
    from mufinder.app.convergence_types import SweepPointRunState

    states = {300.0: SweepPointRunState(status="defined")}
    assert controller.is_ready(states, 300.0) is False
    states = {300.0: SweepPointRunState(status="ready")}
    assert controller.is_ready(states, 300.0) is True


def test_convergence_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeConvergenceView()
    controller = ConvergenceCutoffController(_make_context(document=None), _stub_deps(), view)

    controller.refresh_view()

    assert view.runtime_states == [{}]


def test_convergence_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    controller = ConvergenceCutoffController(
        _make_context(document=document), _stub_deps(), _FakeConvergenceView()
    )
    handled = _job(service, document, "j", calc_type="castep_convergence_cutoff")
    other = _job(service, document, "k", calc_type="castep_uep_source")
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False


# --------------------------------------------------------------------------- #
# ConvergenceKpointController
# --------------------------------------------------------------------------- #


class _FakeKpointView:
    """Minimal KPointConvergenceWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.runtime_states: list[dict] = []

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


def _grid_key(grid: tuple[int, int, int]) -> str:
    return f"{int(grid[0])}x{int(grid[1])}x{int(grid[2])}"


def _stub_kpoint_deps() -> ConvergenceKpointDeps:
    """Deps whose helpers all report 'nothing found' — for tests that never
    reach the definition-backed paths."""

    return ConvergenceKpointDeps(
        active_kpoint_definition=lambda: None,
        deserialize_convergence_config=lambda _payload: None,
        sync_equivalent_convergence_kpoint_mappings=lambda _definition: None,
        definition_for_kpoint_grid=lambda _grid: (None, None),
        job_for_definition_grid=lambda _definition, _grid: None,
        definition_output_directory=lambda _definition: None,
        grid_key=_grid_key,
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
        max_stress_for_job=lambda _job_id: None,
    )


def _kpoint_context_and_deps(
    service: ProjectService,
    document: ProjectDocument,
    *,
    local_backend: _RecordingBackend,
    cluster_backend: _RecordingBackend,
) -> tuple[WorkflowControllerContext, ConvergenceKpointDeps]:
    """Shared context + kpoint deps wired with fake helpers.

    The kpoint-specific helpers below mirror the (unchanged) MainWindow versions
    closely enough to drive materialize/build_states without a MainWindow.
    """

    active_definition = next(iter(document.calculations.values()), None)

    def deserialize(payload):
        return document.ui_state.get("_test_config")

    def definition_for_grid(grid):
        return active_definition, document.ui_state.get("_test_config")

    def job_for_definition_grid(definition, grid):
        mapping = definition.parameters.get("job_ids_by_grid", {}) if definition else {}
        job_id = mapping.get(_grid_key(grid))
        return document.jobs.get(job_id) if job_id else None

    context = WorkflowControllerContext(
        document=lambda: document,
        project_service=service,
        job_execution=None,  # type: ignore[arg-type]
        submission_coordinator=WorkflowSubmissionCoordinator(),
        submit_jobs_locally=local_backend,
        submit_jobs_on_cluster=cluster_backend,
        show_info=lambda *_: None,
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=lambda **_: None,
        status_message=lambda _m: None,
    )
    deps = ConvergenceKpointDeps(
        active_kpoint_definition=lambda: active_definition,
        deserialize_convergence_config=deserialize,
        sync_equivalent_convergence_kpoint_mappings=lambda _d: None,
        definition_for_kpoint_grid=definition_for_grid,
        job_for_definition_grid=job_for_definition_grid,
        definition_output_directory=lambda d: document.project_root / "files" / "convergence_kpoint",
        grid_key=_grid_key,
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
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
    # Persist the form so _kpoint_form_snapshot agrees with the active definition.
    document.ui_state["kpoint_convergence_form"] = {
        "name_prefix": "kconv",
        "kpoint_grids": [[2, 2, 2]],
    }
    return definition, config


def test_kpoint_controller_materialize_writes_job_ids_by_grid(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    definition, _config = _seed_kpoint_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
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


def test_kpoint_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
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

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    view = _FakeKpointView()
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceKpointController(context, deps, view)

    controller.submit([(2, 2, 2)], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == "castep_convergence_kpoint"


def test_kpoint_controller_uses_sweep_point_ready_predicate() -> None:
    controller = ConvergenceKpointController(_make_context(), _stub_kpoint_deps(), _FakeKpointView())
    from mufinder.app.convergence_types import SweepPointRunState

    states = {(2, 2, 2): SweepPointRunState(status="defined")}
    assert controller.is_ready(states, (2, 2, 2)) is False
    states = {(2, 2, 2): SweepPointRunState(status="ready")}
    assert controller.is_ready(states, (2, 2, 2)) is True


def test_kpoint_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeKpointView()
    controller = ConvergenceKpointController(
        _make_context(document=None), _stub_kpoint_deps(), view
    )

    controller.refresh_view()

    assert view.runtime_states == [{}]


def test_kpoint_controller_build_states_reflects_form_ready_grids(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=_RecordingBackend(), cluster_backend=_RecordingBackend()
    )
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())

    states = controller.build_states()

    assert (2, 2, 2) in states
    assert states[(2, 2, 2)].status == "ready"


def test_kpoint_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    controller = ConvergenceKpointController(
        _make_context(document=document), _stub_kpoint_deps(), _FakeKpointView()
    )
    handled = _job(service, document, "j", calc_type="castep_convergence_kpoint")
    other = _job(service, document, "k", calc_type="castep_convergence_cutoff")
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False

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
from mufinder.app.workflow_controllers.muon_relaxation import (
    MuonRelaxationController,
    MuonRelaxationDeps,
)
from mufinder.app.workflow_controllers.pristine_relaxation import (
    PristineRelaxationController,
    PristineRelaxationDeps,
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


# --------------------------------------------------------------------------- #
# PristineRelaxationController
# --------------------------------------------------------------------------- #
#
# This is the first migration whose ``build_result`` does real work: it parses
# the relaxed ``-out.cell``, contributes the ``lattice_a/b/c`` payload, and
# creates the derived relaxed Structure on the document. The tests below drive
# it with narrow fakes (no MainWindow) and a REAL ``-out.cell`` fixture.


class _FakeRelaxationView:
    """Minimal PristineRelaxationWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.runtime_states: list[dict] = []

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


_OUT_CELL_TEXT = """
%BLOCK LATTICE_CART
ang
5.1 0.0 0.0
0.0 5.2 0.0
0.0 0.0 5.3
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
"""


def _real_parse_relaxed_out_cell(path: Path):
    """Parse an ``-out.cell`` exactly as MainWindow._parse_relaxed_out_cell does."""

    from mufinder.castep_io.cell_parser import parse_cell_file
    from mufinder.science.kpoint_grids import lattice_parameters

    relaxed = parse_cell_file(path)
    a, b, c = lattice_parameters(relaxed.lattice_vectors)[:3]
    return relaxed, a, b, c


def _stub_pristine_deps(**overrides) -> PristineRelaxationDeps:
    base = dict(
        active_pristine_relaxation_definition=lambda: None,
        deserialize_pristine_relaxation_config=lambda _payload: None,
        sync_equivalent_pristine_relaxation_mappings=lambda _definition: None,
        pristine_relaxation_form_config=lambda: None,
        pristine_relaxation_row_signature=lambda _row: (),
        job_for_pristine_relaxation_label=lambda _definition, _label: None,
        definition_output_directory=lambda _definition: None,
        expand_pristine_relaxation_jobs=lambda _definition, _config: None,
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
        parse_relaxed_out_cell=_real_parse_relaxed_out_cell,
        run_for_job=lambda _job_id: None,
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
    )
    base.update(overrides)
    return PristineRelaxationDeps(**base)


def _pristine_job(service: ProjectService, document: ProjectDocument, label: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )


def test_pristine_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    controller = PristineRelaxationController(
        _make_context(document=document), _stub_pristine_deps(), _FakeRelaxationView()
    )
    handled = _pristine_job(service, document, "pristine_relax_row1")
    muon = service.add_job(
        document,
        label="site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muonium_relaxation",
    )
    # The still-inline muon-relaxation branch must NOT be claimed by this
    # controller — it is dispatched by the inline code, not build_result.
    assert controller.handles_job(handled) is True
    assert controller.handles_job(muon) is False


def test_pristine_controller_uses_workflow_row_ready_predicate() -> None:
    controller = PristineRelaxationController(
        _make_context(), _stub_pristine_deps(), _FakeRelaxationView()
    )
    # Pristine relaxation uses the base-default (wider) workflow-row predicate:
    # "not generated" / "defined" rows are ready, only "failed" is excluded.
    assert controller.is_ready({"r": _State("not generated")}, "r") is True
    assert controller.is_ready({"r": _State("ready")}, "r") is True
    assert controller.is_ready({"r": _State("failed")}, "r") is False


def test_pristine_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeRelaxationView()
    controller = PristineRelaxationController(
        _make_context(document=None), _stub_pristine_deps(), view
    )
    controller.refresh_view()
    assert view.runtime_states == [{}]


def test_pristine_controller_build_result_parses_lattice_and_creates_structure(
    tmp_path: Path,
) -> None:
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

    definition = service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "pristine_relax", "rows": []},
        output_directory=document.project_root / "files" / "pristine_relaxation",
    )

    job = _pristine_job(service, document, "pristine_relax_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # A REAL -out.cell fixture on disk, fed through the controller's parse helper.
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(_OUT_CELL_TEXT, encoding="utf-8")

    deps = _stub_pristine_deps(
        active_pristine_relaxation_definition=lambda: definition,
        job_output_directory=lambda _job: output_dir,
    )
    context = _make_context(document=document, project_service=service)
    controller = PristineRelaxationController(context, deps, _FakeRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())
    result = controller.build_result(job, parsed, run)

    # Payload gains the first three lattice parameters from the relaxed cell.
    assert result.payload["lattice_a"] == pytest.approx(5.1)
    assert result.payload["lattice_b"] == pytest.approx(5.2)
    assert result.payload["lattice_c"] == pytest.approx(5.3)

    # No output_files contributed — the call site already passes them generically.
    assert result.output_files == []

    # Exactly one derived structure created on the document, with the right
    # origin and parentage.
    assert len(result.derived_structure_ids) == 1
    derived_id = result.derived_structure_ids[0]
    derived = document.structures[derived_id]
    assert derived.origin == "pristine_relaxation"
    assert derived.parent_structure_id == host.id
    assert derived.properties["source_job_id"] == str(job.id)
    assert derived.properties["source_workflow_stage"] == WorkflowStage.PRISTINE_RELAXATION.value
    assert derived.label == "pristine_relax_row1 relaxed"


def test_pristine_controller_build_result_missing_out_cell_raises(tmp_path: Path) -> None:
    """build_result surfaces the parse error rather than silently fabricating a
    structure when the -out.cell is absent. In production the inline prologue in
    ``_on_local_job_finished`` guards this case (see the FAILED-job regression in
    tests/app/test_main_window_actions.py) so build_result is never reached with a
    missing file; this test pins only build_result's own parse contract."""

    from mufinder.castep_io.cell_parser import CellParseError

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _pristine_job(service, document, "pristine_relax_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    deps = _stub_pristine_deps(job_output_directory=lambda _job: empty_dir)
    context = _make_context(document=document, project_service=service)
    controller = PristineRelaxationController(context, deps, _FakeRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())
    with pytest.raises((CellParseError, FileNotFoundError, OSError)):
        controller.build_result(job, parsed, run)
    # No derived structure was created.
    assert document.structures == {} or all(
        s.origin != "pristine_relaxation" for s in document.structures.values()
    )


def test_pristine_controller_materialize_writes_job_ids_by_label(tmp_path: Path) -> None:
    from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
    from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
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

    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(PristineRelaxationRow(label="row1"),),
    )
    output_root = document.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "pristine_relax",
            "rows": [{"label": "row1", "parameter_values": {}}],
        },
        output_directory=output_root,
    )

    def job_for_label(defn, label):
        mapping = defn.parameters.get("job_ids_by_label", {}) if defn else {}
        job_id = mapping.get(label)
        return document.jobs.get(UUID(job_id)) if isinstance(job_id, str) else None

    def expand(defn, cfg):
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=cfg,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )

    deps = _stub_pristine_deps(
        active_pristine_relaxation_definition=lambda: definition,
        deserialize_pristine_relaxation_config=lambda _payload: config,
        job_for_pristine_relaxation_label=job_for_label,
        definition_output_directory=lambda _defn: output_root,
        expand_pristine_relaxation_jobs=expand,
    )
    context = _make_context(document=document, project_service=service)
    controller = PristineRelaxationController(context, deps, _FakeRelaxationView())

    jobs = controller.materialize(["row1"])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.calc_type == "castep_pristine_relaxation"
    mapping = definition.parameters["job_ids_by_label"]
    assert mapping["row1"] == str(job.id)
    # Rendered inputs were written under the output root.
    job_dir = output_root / job.label
    assert (job_dir / f"{job.label}.cell").exists()
    assert (job_dir / f"{job.label}.param").exists()


def test_pristine_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
    from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
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

    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(PristineRelaxationRow(label="row1"),),
    )
    output_root = document.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "pristine_relax", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    def job_for_label(defn, label):
        mapping = defn.parameters.get("job_ids_by_label", {}) if defn else {}
        job_id = mapping.get(label)
        return document.jobs.get(UUID(job_id)) if isinstance(job_id, str) else None

    def expand(defn, cfg):
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=cfg,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )

    deps = _stub_pristine_deps(
        active_pristine_relaxation_definition=lambda: definition,
        deserialize_pristine_relaxation_config=lambda _payload: config,
        pristine_relaxation_form_config=lambda: config,
        pristine_relaxation_row_signature=lambda row: (row.label,),
        job_for_pristine_relaxation_label=job_for_label,
        definition_output_directory=lambda _defn: output_root,
        expand_pristine_relaxation_jobs=expand,
    )

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
    )
    controller = PristineRelaxationController(context, deps, _FakeRelaxationView())

    controller.submit(["row1"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_pristine_relaxation"


# --------------------------------------------------------------------------- #
# MuonRelaxationController
# --------------------------------------------------------------------------- #
#
# The first migration exercising the two base hooks beyond their defaults:
# ``pre_submit_check`` (staged-site/host validation runs BEFORE any submission
# plan is built) and ``submission_groups`` (a mixed job list is split into
# per-charge-state groups, each dispatched separately through the base template
# loop). ``build_result`` carries the richest result logic so far — muon
# position payload (supercell + folded host coordinates), derived structure,
# and the sanctioned source-site mutation — driven below with a REAL
# ``-out.cell`` fixture containing an ``H:mu`` muon.


class _FakeMuonRelaxationView:
    """Minimal MuonRelaxationWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.runtime_states: list[dict] = []

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


_MUON_OUT_CELL_TEXT = """
%BLOCK LATTICE_CART
ang
8.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
H:mu 0.3 0.4 0.5
%ENDBLOCK POSITIONS_FRAC
"""


def _real_muon_position_from_cell(cell):
    """Extract the H:mu position exactly as MainWindow._muon_fractional_position_from_cell does."""

    species = getattr(cell, "species", ())
    positions = getattr(cell, "fractional_positions", ())
    for atom_species, position in reversed(list(zip(species, positions, strict=False))):
        if str(atom_species).strip().casefold() != "h:mu":
            continue
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            return None
        return (float(position[0]), float(position[1]), float(position[2]))
    return None


def _real_coerce_fractional_triplet(value):
    """Coerce exactly as MainWindow._coerce_fractional_triplet does."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _stub_muon_deps(**overrides) -> MuonRelaxationDeps:
    from mufinder.app.relaxation_adapter import muon_charge_state_prefix

    base = dict(
        active_muon_relaxation_definition=lambda: None,
        deserialize_muon_relaxation_config=lambda _payload: None,
        sync_equivalent_muon_relaxation_mappings=lambda _definition: None,
        sync_active_muon_relaxation_dashboard_jobs=lambda _definition: None,
        muon_relaxation_form_config=lambda: None,
        muon_relaxation_row_signature=lambda row: (row.label, row.charge_state),
        muon_relaxation_row_key=(
            lambda row: f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"
        ),
        job_for_muon_relaxation_label=lambda _definition, _label, _charge_state: None,
        job_is_active_local=lambda _job: False,
        definition_output_directory=lambda _definition: None,
        handle_site_host_mismatch=lambda _mismatch: False,
        muon_relaxation_definition_and_row_for_job=lambda _job: (None, None),
        muon_fractional_position_from_cell=_real_muon_position_from_cell,
        coerce_fractional_triplet=_real_coerce_fractional_triplet,
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
        parse_relaxed_out_cell=_real_parse_relaxed_out_cell,
        run_for_job=lambda _job_id: None,
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
    )
    base.update(overrides)
    return MuonRelaxationDeps(**base)


def _muon_job(
    service: ProjectService, document: ProjectDocument, label: str, calc_type: str
) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=calc_type,
    )


def _muon_host_document(service: ProjectService, tmp_path: Path):
    from mufinder.domain.models import Structure

    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    return document, host


def test_muon_controller_handles_both_charge_state_calc_types(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = _muon_host_document(service, tmp_path)
    controller = MuonRelaxationController(
        _make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    positive = _muon_job(service, document, "s1", "castep_positivemuon_relaxation")
    muonium = _muon_job(service, document, "s2", "castep_muonium_relaxation")
    pristine = service.add_job(
        document,
        label="p",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    assert controller.handles_job(positive) is True
    assert controller.handles_job(muonium) is True
    # Pristine relaxation is owned by its own controller, not this one.
    assert controller.handles_job(pristine) is False


def test_muon_controller_ready_predicate_matches_bare_and_qualified_labels() -> None:
    controller = MuonRelaxationController(
        _make_context(), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    states = {
        "muonium:Site 1": _State("ready"),
        "positivemuon:Site 2": _State("failed"),
    }
    # Charge-qualified keys match their own state only.
    assert controller.is_ready(states, "muonium:Site 1") is True
    assert controller.is_ready(states, "positivemuon:Site 2") is False
    # A bare label matches any charge-qualified key ending in ":{label}".
    assert controller.is_ready(states, "Site 1") is True
    assert controller.is_ready(states, "Site 2") is False
    # "defined" and "not generated" also count as ready (unlike the sweep predicate).
    assert controller.is_ready({"s": _State("defined")}, "s") is True
    assert controller.is_ready({"s": _State("not generated")}, "s") is True


def test_muon_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeMuonRelaxationView()
    controller = MuonRelaxationController(
        _make_context(document=None), _stub_muon_deps(), view
    )
    controller.refresh_view()
    assert view.runtime_states == [{}]


def test_muon_controller_pre_submit_check_blocks_submission(tmp_path: Path) -> None:
    """An unresolved staged-site/host mismatch silently refuses the submission:
    no submission plan is built, no jobs are materialized or dispatched, and no
    'no ready jobs' dialog appears."""

    from mufinder.domain.models import CalculationDefinition, MuonSite, Structure

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    other = Structure(
        label="Other",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, other)
    # The staged site was generated for a different host than the active one.
    site = MuonSite(
        label="Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=other.id,
    )
    document.sites[site.id] = site
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    mismatches: list[object] = []
    form_config_calls: list[bool] = []
    deps = _stub_muon_deps(
        active_muon_relaxation_definition=lambda: definition,
        handle_site_host_mismatch=lambda mismatch: (mismatches.append(mismatch), False)[1],
        muon_relaxation_form_config=lambda: (form_config_calls.append(True), None)[1],
    )
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1"], target="local")

    assert len(mismatches) == 1  # the host check ran and surfaced the mismatch
    assert local.calls == []  # nothing dispatched
    assert infos == []  # the refusal is silent (no "no ready jobs" dialog)
    assert form_config_calls == []  # no submission plan was even built
    assert document.jobs == {}  # nothing materialized


def test_muon_controller_pre_submit_check_passes_when_sites_match_host(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    mismatches: list[object] = []
    deps = _stub_muon_deps(
        active_muon_relaxation_definition=lambda: definition,
        handle_site_host_mismatch=lambda mismatch: (mismatches.append(mismatch), False)[1],
    )
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1"], target="local")

    # No mismatch dialog; the check passed and the plan ran (and found no
    # configured rows, hence the standard no-ready dialog).
    assert mismatches == []
    assert infos == [("No ready jobs", MuonRelaxationController.no_ready_message)]
    assert local.calls == []


def test_muon_controller_submission_groups_split_by_charge_state(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = _muon_host_document(service, tmp_path)
    controller = MuonRelaxationController(
        _make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    pos_a = _muon_job(service, document, "a", "castep_positivemuon_relaxation")
    mu_b = _muon_job(service, document, "b", "castep_muonium_relaxation")
    pos_c = _muon_job(service, document, "c", "castep_positivemuon_relaxation")

    # Mixed list splits into positive-muon group first, then muonium, no extra
    # kwargs; a duplicate job id is deduped.
    groups = controller.submission_groups([pos_a, mu_b, pos_c, pos_a])
    assert groups == [([pos_a, pos_c], {}), ([mu_b], {})]

    # A single-charge-state list yields one group (no empty groups).
    assert controller.submission_groups([mu_b]) == [([mu_b], {})]


def _muon_two_row_submission_fixture(service: ProjectService, tmp_path: Path):
    """Document + config with one positive-muon and one muonium row, wired for
    real materialization through expand_muon_relaxation_jobs."""

    from uuid import UUID

    from mufinder.app.relaxation_adapter import muon_charge_state_prefix
    from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow
    from mufinder.domain.models import MuonSite

    document, host = _muon_host_document(service, tmp_path)
    site1 = MuonSite(
        label="Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    site2 = MuonSite(
        label="Site 2",
        status="staged",
        fractional_position=[0.4, 0.1, 0.2],
        source_structure_id=host.id,
    )
    document.sites[site1.id] = site1
    document.sites[site2.id] = site2

    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=site1.id,
                label="Site 1",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="positive_muon",
            ),
            MuonRelaxationRow(
                site_id=site2.id,
                label="Site 2",
                initial_fractional_position=(0.4, 0.1, 0.2),
                charge_state="muonium",
            ),
        )
    )
    output_root = document.project_root / "files" / "muon_relaxation"
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters={"configuration": {}},
        output_directory=output_root,
    )

    def job_for_label(defn, label, charge_state):
        mapping = defn.parameters.get("job_ids_by_label", {}) if defn else {}
        keys = (
            [f"{muon_charge_state_prefix(charge_state)}:{label}"] if charge_state else []
        )
        keys.append(label)
        for key in keys:
            job_id = mapping.get(key)
            if isinstance(job_id, str):
                return document.jobs.get(UUID(job_id))
        return None

    deps = _stub_muon_deps(
        active_muon_relaxation_definition=lambda: definition,
        deserialize_muon_relaxation_config=lambda _payload: config,
        muon_relaxation_form_config=lambda: config,
        job_for_muon_relaxation_label=job_for_label,
        definition_output_directory=lambda _defn: output_root,
    )
    return document, definition, deps, output_root


def test_muon_controller_materialize_writes_charge_prefixed_mappings_and_persists(
    tmp_path: Path,
) -> None:
    import dataclasses

    service = ProjectService()
    document, definition, deps, output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    context = dataclasses.replace(
        _make_context(document=document, project_service=service),
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    jobs = controller.materialize(["muonium:Site 2"])

    assert [job.calc_type for job in jobs] == ["castep_muonium_relaxation"]
    assert [job.label for job in jobs] == ["Site 2"]
    mapping = definition.parameters["job_ids_by_label"]
    seed_mapping = definition.parameters["job_seed_names_by_label"]
    assert mapping == {"muonium:Site 2": str(jobs[0].id)}
    assert set(seed_mapping) == {"muonium:Site 2"}
    # Rendered inputs were written under the charge-state root.
    seed = seed_mapping["muonium:Site 2"]
    job_dir = output_root / "muonium" / seed
    assert (job_dir / f"{seed}.cell").exists()
    assert (job_dir / f"{seed}.param").exists()
    # The document was persisted with the muon-specific failure message.
    assert persists == [
        {"failure_message": "Muon relaxation jobs were generated, but autosave failed."}
    ]


def test_muon_controller_submit_local_splits_groups_across_backend_calls(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, deps, _output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1", "muonium:Site 2"], target="local")

    assert cluster.calls == []
    # One backend call per charge-state group: positive muon first, then muonium.
    assert len(local.calls) == 2
    first_group, first_extras = local.calls[0]
    second_group, second_extras = local.calls[1]
    assert first_extras == {} and second_extras == {}
    assert [job.calc_type for job in first_group] == ["castep_positivemuon_relaxation"]
    assert [job.label for job in first_group] == ["Site 1"]
    assert [job.calc_type for job in second_group] == ["castep_muonium_relaxation"]
    assert [job.label for job in second_group] == ["Site 2"]
    # Both jobs were recorded under charge-prefixed mapping keys.
    mapping = definition.parameters["job_ids_by_label"]
    assert set(mapping) == {"positivemuon:Site 1", "muonium:Site 2"}


def test_muon_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, deps, _output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1", "muonium:Site 2"], target="cluster")

    assert local.calls == []
    assert [
        [job.calc_type for job in group] for group, _extras in cluster.calls
    ] == [["castep_positivemuon_relaxation"], ["castep_muonium_relaxation"]]


def test_muon_controller_build_result_records_position_structure_and_site_mutation(
    tmp_path: Path,
) -> None:
    """build_result with a REAL -out.cell containing an H:mu muon contributes
    the lattice + folded-position payload, creates the derived structure, and
    mutates the source site (the sanctioned document side effect)."""

    from mufinder.app.relaxation_types import MuonRelaxationRow
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site

    row = MuonRelaxationRow(
        site_id=site.id,
        label="Shared Site",
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state="muonium",
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # A REAL -out.cell fixture (2x2x1 supercell of the 4 A host) on disk, fed
    # through the controller's parse + position-extraction helpers.
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(_MUON_OUT_CELL_TEXT, encoding="utf-8")

    deps = _stub_muon_deps(
        active_muon_relaxation_definition=lambda: definition,
        muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
        job_output_directory=lambda _job: output_dir,
    )
    context = _make_context(document=document, project_service=service)
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-12.0, warnings=())
    result = controller.build_result(job, parsed, run)

    # Relaxed supercell lattice payload.
    assert result.payload["lattice_a"] == pytest.approx(8.0)
    assert result.payload["lattice_b"] == pytest.approx(8.0)
    assert result.payload["lattice_c"] == pytest.approx(4.0)
    # Muon position: raw supercell fraction, and folded back to host coordinates
    # through the row's 2x2x1 supercell matrix.
    assert result.payload["muon_final_fractional_position_supercell"] == pytest.approx(
        [0.3, 0.4, 0.5]
    )
    assert result.payload["muon_final_fractional_position_host"] == pytest.approx(
        [0.6, 0.8, 0.5]
    )
    # No output_files contributed — the call site already passes them generically.
    assert result.output_files == []

    # Exactly one derived structure created on the document, with the right
    # origin, parentage, and the muon present.
    assert len(result.derived_structure_ids) == 1
    derived = document.structures[result.derived_structure_ids[0]]
    assert derived.origin == "muon_relaxation"
    assert derived.parent_structure_id == host.id
    assert derived.label == "Shared Site relaxed"
    assert derived.properties["source_job_id"] == str(job.id)
    assert derived.properties["source_workflow_stage"] == WorkflowStage.MUON_RELAXATION.value
    assert any(str(s).strip().casefold() == "h:mu" for s in derived.species)

    # Sanctioned document side effect: the source site is marked relaxed and
    # linked to this run with the job's total energy.
    assert site.status == "relaxed"
    assert site.relaxation_run_id == run.id
    assert site.relative_energy_eV == pytest.approx(-12.0)


def test_muon_controller_build_result_fails_loudly_on_muonless_cell(tmp_path: Path) -> None:
    """The success-path invariant is self-contained: if build_result is ever
    reached with a relaxed cell that lacks the muon (the inline guard's job to
    prevent), it raises a descriptive error instead of a deep TypeError."""

    from mufinder.app.relaxation_types import MuonRelaxationRow
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site
    row = MuonRelaxationRow(
        site_id=site.id,
        label="Shared Site",
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state="muonium",
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition
    job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # Same fixture with the muon line removed: extraction returns None.
    muonless_text = "\n".join(
        line for line in _MUON_OUT_CELL_TEXT.splitlines() if "h:mu" not in line.casefold()
    )
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(muonless_text, encoding="utf-8")

    deps = _stub_muon_deps(
        active_muon_relaxation_definition=lambda: definition,
        muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
        job_output_directory=lambda _job: output_dir,
    )
    context = _make_context(document=document, project_service=service)
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-12.0, warnings=())
    with pytest.raises(ValueError, match="relaxed cell containing"):
        controller.build_result(job, parsed, run)
    # No partial side effects: site untouched, no derived structure added.
    assert site.status == "staged"
    assert site.relaxation_run_id is None

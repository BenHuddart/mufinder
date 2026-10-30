"""Tests for :class:`ConvergenceCutoffController`.

The controller is tested against a real ``ProjectService`` document with fake
submit backends, pinning that it reaches the right backend with the right jobs
and writes the ``job_ids_by_cutoff`` mapping. The shared sweep behaviour of
:class:`SweepConvergenceController` is exercised through this controller.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Structure
from mufinder.persistence.migrations import RETIRED_UI_STATE_KEYS
from mufinder.services.convergence_defaults import (
    CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY,
    CUTOFF_EXAMINER_STATE_KEY,
)
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.convergence_types import SweepPointRunState
from mufinder.app.workflow_config_codecs import deserialize_convergence_config
from mufinder.app.workflow_controllers.base import WorkflowControllerContext
from mufinder.app.workflow_controllers.convergence_cutoff import (
    ConvergenceCutoffController,
    ConvergenceCutoffDeps,
)
from mufinder.app.workflow_submission import WorkflowSubmissionCoordinator

from tests.support.jobs import make_job
from tests.support.controllers import RecordingBackend, RecordingSweepWindow, make_context


# --------------------------------------------------------------------------- #
# ConvergenceCutoffController
# --------------------------------------------------------------------------- #


class _FakeConvergenceView:
    """Minimal CutoffConvergenceWidget stand-in for controller tests."""

    def __init__(self, form_state: dict | None = None) -> None:
        self._form_state = form_state or {}
        self.runtime_states: list[dict] = []
        self.simulation_cell_options: list[dict] = []
        self.species_options: list[tuple[str, ...]] = []

    def collect_state(self) -> dict:
        return self._form_state

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))

    def set_simulation_cell_options(self, **options) -> None:
        self.simulation_cell_options.append(options)

    def set_parameter_species_options(self, species_options) -> None:
        self.species_options.append(tuple(species_options))


def _stub_deps() -> ConvergenceCutoffDeps:
    """Deps whose helpers all report 'nothing found' — for tests that never
    reach the definition-backed paths."""

    return ConvergenceCutoffDeps(
        deserialize_convergence_config=lambda _payload: None,
        cutoff_key=lambda value: f"{float(value):.12g}",
        max_stress_for_job=lambda _job_id: None,
        **RecordingSweepWindow().deps(),
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
        definition_output_directory=lambda d: (
            document.project_root / "files" / "convergence_cutoff"
        ),
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
        **RecordingSweepWindow().deps(),
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
    handled = make_job(
        "j", service=service, document=document, calc_type="castep_convergence_cutoff"
    )
    other = make_job("k", service=service, document=document, calc_type="castep_uep_source")
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False


def test_cutoff_controller_finds_a_legacy_definitions_job_through_job_labels(
    tmp_path: Path,
) -> None:
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


# --------------------------------------------------------------------------- #
# Examine and adopt (SweepConvergenceController, through the cutoff sweep)
# --------------------------------------------------------------------------- #


def _examine_controller(
    document: ProjectDocument | None,
    window: RecordingSweepWindow,
    *,
    service: ProjectService | None = None,
    infos: list[tuple[str, str]] | None = None,
    statuses: list[str] | None = None,
    persisted: list[str] | None = None,
) -> ConvergenceCutoffController:
    deps = ConvergenceCutoffDeps(
        deserialize_convergence_config=deserialize_convergence_config,
        cutoff_key=lambda value: f"{float(value):.12g}",
        max_stress_for_job=lambda _job_id: None,
        **window.deps(),
    )
    context = make_context(
        document=document,
        project_service=service,
        infos=infos,
        statuses=statuses,
        persisted=persisted,
    )
    return ConvergenceCutoffController(context, deps, _FakeConvergenceView())


def _adopt_document(
    service: ProjectService, tmp_path: Path, *, shared_params: dict[str, object] | None = None
) -> tuple[ProjectDocument, Structure]:
    document = service.create_project(tmp_path / "adopt.mufinder", "Adopt")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    definition = service.register_convergence_cutoff_definition(
        document,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_parameter": "cut_off_energy",
            "sweep_values": [300.0, 400.0, 550.0],
            "shared_params": shared_params or {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=document.project_root / "files" / "convergence_cutoff",
    )
    document.ui_state[ConvergenceCutoffController.active_key] = str(definition.id)
    return document, host


def test_examine_without_a_project_asks_for_one() -> None:
    window = RecordingSweepWindow()
    infos: list[tuple[str, str]] = []
    controller = _examine_controller(None, window, infos=infos)

    controller.examine()

    assert infos == [("No project", "Create or open a project first.")]
    assert window.examiner_calls == []


def test_examine_without_a_host_structure_warns(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "no_host.mufinder", "NoHost")
    window = RecordingSweepWindow()
    controller = _examine_controller(document, window, service=service)

    controller.examine()

    assert window.warnings == [
        ("No host structure", "Load a host structure before examining convergence.")
    ]
    assert window.examiner_calls == []


def test_examine_opens_the_examiner_on_the_sweep_and_keeps_its_final_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProjectService()
    document, _host = _adopt_document(service, tmp_path)
    document.ui_state[CUTOFF_EXAMINER_STATE_KEY] = "invalid"
    window = RecordingSweepWindow(examiner_state={"restored": True})
    controller = _examine_controller(document, window, service=service)
    states = {
        300.0: SweepPointRunState(status="completed", total_energy=-10.0, max_force=0.2),
        400.0: SweepPointRunState(status="completed", total_energy=-10.02, max_force=0.15),
    }
    monkeypatch.setattr(controller, "build_states", lambda: states)

    controller.examine()

    ((points, options),) = window.examiner_calls
    assert points == states
    # The cutoff examiner keeps the dialog's cutoff defaults.
    assert options == {"num_atoms": 2, "initial_state": {}}
    assert document.ui_state[CUTOFF_EXAMINER_STATE_KEY] == {"restored": True}


def test_examiner_state_is_kept_in_ui_state_without_saving(tmp_path: Path) -> None:
    service = ProjectService()
    root = tmp_path / "examiner_state.mufinder"
    document = service.create_project(root, "ExaminerState")
    initial_state_text = document.project_root.joinpath("state.json").read_text(encoding="utf-8")
    persisted: list[str] = []
    controller = _examine_controller(
        document, RecordingSweepWindow(), service=service, persisted=persisted
    )

    controller.store_examiner_state(
        {
            "results_text": "Recommended cutoff: 500 eV",
            "plot_data": {"cutoffs": [300.0, 400.0, 500.0], "energies": [-10.0, -10.01, -10.0105]},
        }
    )

    assert document.ui_state[CUTOFF_EXAMINER_STATE_KEY]["results_text"] == (
        "Recommended cutoff: 500 eV"
    )
    assert persisted == []
    assert (
        document.project_root.joinpath("state.json").read_text(encoding="utf-8")
        == initial_state_text
    )
    # An explicit save keeps it across a reload.
    service.save_project(document)
    loaded = service.load_project(root)
    assert loaded.ui_state[CUTOFF_EXAMINER_STATE_KEY]["plot_data"]["cutoffs"] == [
        300.0,
        400.0,
        500.0,
    ]


def test_adopting_from_the_examiner_sets_the_default_and_pins_the_snapshot(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, host = _adopt_document(service, tmp_path)
    examiner_state = {
        "plot_data": {
            "cutoffs": [300.0, 400.0, 550.0],
            "energies": [-10.0, -10.1, -10.15],
            "forces": [0.3, 0.18, 0.05],
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
        }
    }
    document.ui_state[CUTOFF_EXAMINER_STATE_KEY] = examiner_state
    window = RecordingSweepWindow(examiner_state=examiner_state)
    statuses: list[str] = []
    persisted: list[str] = []
    controller = _examine_controller(
        document, window, service=service, statuses=statuses, persisted=persisted
    )
    controller.examine()
    assert window.on_adopted is not None

    window.on_adopted(550.0, "converged", 0.001, 0.05, None)

    assert document.project.recommended_cutoff_energy == pytest.approx(550.0)
    assert document.project.recommended_cutoff_method == "converged"
    assert document.project.recommended_cutoff_energy_threshold == pytest.approx(0.001)
    assert document.project.recommended_cutoff_force_threshold == pytest.approx(0.05)
    assert document.project.recommended_cutoff_cell_id == str(host.id)
    assert RETIRED_UI_STATE_KEYS.isdisjoint(document.ui_state)
    snapshot = document.ui_state[CONVERGENCE_OVERVIEW_SNAPSHOTS_KEY]["cutoff"]
    assert "method" not in snapshot
    assert snapshot["host_structure_id"] == str(host.id)
    assert snapshot["simulation_cell_structure_id"] == str(host.id)
    assert snapshot["default_value"] == pytest.approx(550.0)
    assert snapshot["plot_data"]["cutoffs"] == [300.0, 400.0, 550.0]
    assert document.provenance[-1].action == "set_recommended_cutoff"
    assert window.refreshes == 1
    assert persisted == ["Could not persist convergence overview snapshot to disk"]
    assert statuses == ["Project default cutoff set to 550 eV"]


def test_adopting_a_result_from_another_functional_is_confirmed(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = _adopt_document(
        service, tmp_path, shared_params={"task": "SINGLEPOINT", "xc_functional": "PBE"}
    )
    document.project.recommended_xc_functional = "LDA"
    window = RecordingSweepWindow(confirm_answer="cancel")
    statuses: list[str] = []
    controller = _examine_controller(document, window, service=service, statuses=statuses)

    controller.adopt(400.0, "converged", 0.001, 0.05, point_labels={})

    ((title, text, choices, options),) = window.confirmations
    assert title == "Change project functional?"
    assert "PBE" in text and "LDA" in text
    assert choices == [("change", "Change functional"), ("cancel", "Cancel")]
    assert options == {"default": "cancel", "escape": "cancel"}
    assert document.project.recommended_cutoff_energy is None
    assert window.refreshes == 0
    assert statuses == []

    window.confirm_answer = "change"
    controller.adopt(400.0, "converged", 0.001, 0.05, point_labels={})

    assert document.project.recommended_cutoff_energy == pytest.approx(400.0)
    assert document.project.recommended_xc_functional == "PBE"


def test_adopting_without_a_project_does_nothing() -> None:
    window = RecordingSweepWindow()
    controller = _examine_controller(None, window)

    controller.adopt(400.0, "converged", 0.001, 0.05, point_labels={})

    assert window.refreshes == 0
    assert controller.definition_xc_functional(None) is None
    assert controller.confirm_adopted_xc_functional("PBE") is True


# --------------------------------------------------------------------------- #
# Simulation cell (SweepConvergenceController, through the cutoff sweep)
# --------------------------------------------------------------------------- #

_CUTOFF = WorkflowStage.CONVERGENCE_CUTOFF


def _cell_controller(
    document: ProjectDocument | None,
    window: RecordingSweepWindow,
    *,
    service: ProjectService | None = None,
    statuses: list[str] | None = None,
) -> tuple[ConvergenceCutoffController, _FakeConvergenceView]:
    view = _FakeConvergenceView()
    deps = ConvergenceCutoffDeps(
        deserialize_convergence_config=deserialize_convergence_config,
        cutoff_key=lambda value: f"{float(value):.12g}",
        max_stress_for_job=lambda _job_id: None,
        **window.deps(),
    )
    context = make_context(document=document, project_service=service, statuses=statuses)
    return ConvergenceCutoffController(context, deps, view), view


def _cell_document(
    service: ProjectService,
    tmp_path: Path,
    *,
    stage: WorkflowStage = _CUTOFF,
    matrix: list[list[int]] | None = None,
) -> tuple[ProjectDocument, Structure]:
    """A one-atom Si host and one saved supercell of it owned by ``stage``."""

    document = service.create_project(tmp_path / "cell.mufinder", "Cell")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    asset = service.create_workflow_supercell_asset(
        document,
        workflow_stage=stage,
        host_structure_id=host.id,
        matrix=matrix or [[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        label="Saved cell",
        reuse_existing=False,
    )
    return document, asset


def _use_saved_cell(service: ProjectService, document: ProjectDocument, asset: Structure) -> None:
    service.use_workflow_supercell_asset(
        document, workflow_stage=_CUTOFF, asset_structure_id=asset.id
    )


def test_refresh_simulation_cell_without_a_project_disables_the_selector() -> None:
    controller, view = _cell_controller(None, RecordingSweepWindow())

    controller.refresh_simulation_cell()

    assert view.simulation_cell_options == [
        {
            "summary": "Load a host structure to choose a simulation cell.",
            "assets": [],
            "selected_structure_id": None,
            "enabled": False,
        }
    ]
    assert view.species_options == [()]


def test_refresh_simulation_cell_lists_saved_cells_and_the_cell_species(tmp_path: Path) -> None:
    service = ProjectService()
    document, asset = _cell_document(service, tmp_path)
    _use_saved_cell(service, document, asset)
    controller, view = _cell_controller(document, RecordingSweepWindow(), service=service)

    controller.refresh_simulation_cell()

    (options,) = view.simulation_cell_options
    assert options["enabled"] is True
    assert options["selected_structure_id"] == str(asset.id)
    assert {"structure_id": str(asset.id), "label": "Saved cell"} in options["assets"]
    assert view.species_options == [("Si",)]


def test_supercell_initial_matrix_prefers_the_selected_saved_cell(tmp_path: Path) -> None:
    service = ProjectService()
    document, asset = _cell_document(service, tmp_path, matrix=[[3, 0, 0], [0, 1, 0], [0, 0, 1]])
    document.project.supercell_matrix = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    controller, _view = _cell_controller(document, RecordingSweepWindow(), service=service)

    assert controller.supercell_initial_matrix() == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]

    _use_saved_cell(service, document, asset)

    assert controller.supercell_initial_matrix() == [[3, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_supercell_initial_matrix_without_a_project_is_the_identity() -> None:
    controller, _view = _cell_controller(None, RecordingSweepWindow())

    assert controller.supercell_initial_matrix() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def test_create_simulation_cell_uses_the_cell_picked_in_the_editor(tmp_path: Path) -> None:
    service = ProjectService()
    document, asset = _cell_document(service, tmp_path)
    window = RecordingSweepWindow(editor_selection=asset.id)
    statuses: list[str] = []
    controller, view = _cell_controller(document, window, service=service, statuses=statuses)

    controller.create_simulation_cell()

    assert window.editor_calls == [(_CUTOFF, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])]
    selection = document.ui_state["workflow_supercells"]["convergence_cutoff"]
    assert selection["structure_id"] == str(asset.id)
    assert view.simulation_cell_options[-1]["selected_structure_id"] == str(asset.id)
    assert statuses == ["Selected convergence simulation cell 'Saved cell'"]


def test_create_simulation_cell_changes_nothing_when_the_editor_picks_nothing(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, _asset = _cell_document(service, tmp_path)
    window = RecordingSweepWindow()
    statuses: list[str] = []
    controller, view = _cell_controller(document, window, service=service, statuses=statuses)

    controller.create_simulation_cell()

    assert len(window.editor_calls) == 1
    assert view.simulation_cell_options == []
    assert statuses == []


def test_select_simulation_cell_clones_another_workflows_saved_cell(tmp_path: Path) -> None:
    service = ProjectService()
    document, source = _cell_document(service, tmp_path, stage=WorkflowStage.MUON_RELAXATION)
    statuses: list[str] = []
    controller, view = _cell_controller(
        document, RecordingSweepWindow(), service=service, statuses=statuses
    )

    controller.select_simulation_cell(str(source.id))

    selection = document.ui_state["workflow_supercells"]["convergence_cutoff"]
    cloned = document.structures[UUID(selection["structure_id"])]
    assert selection["mode"] == "saved_supercell"
    assert cloned.id != source.id
    assert cloned.properties["supercell_asset"]["owner_workflow_stage"] == "convergence_cutoff"
    assert cloned.properties["supercell_asset"]["source_asset_structure_id"] == str(source.id)
    assert view.simulation_cell_options[-1]["selected_structure_id"] == str(cloned.id)
    assert statuses == [f"Selected convergence simulation cell '{cloned.label}'"]


@pytest.mark.parametrize("payload", [None, "not-a-uuid"])
def test_select_simulation_cell_without_an_id_goes_back_to_the_unit_cell(
    tmp_path: Path, payload: object
) -> None:
    service = ProjectService()
    document, asset = _cell_document(service, tmp_path)
    _use_saved_cell(service, document, asset)
    controller, view = _cell_controller(document, RecordingSweepWindow(), service=service)

    controller.select_simulation_cell(payload)

    assert document.ui_state["workflow_supercells"]["convergence_cutoff"]["mode"] == "unit_cell"
    assert view.simulation_cell_options[-1]["selected_structure_id"] is None


def test_select_simulation_cell_warns_when_the_cell_is_unavailable(tmp_path: Path) -> None:
    service = ProjectService()
    document, _asset = _cell_document(service, tmp_path)
    window = RecordingSweepWindow()
    statuses: list[str] = []
    controller, view = _cell_controller(document, window, service=service, statuses=statuses)

    controller.select_simulation_cell(str(UUID(int=0)))

    assert window.warnings == [
        ("Simulation cell unavailable", "Selected supercell does not exist in this project.")
    ]
    assert len(view.simulation_cell_options) == 1
    assert statuses == []


def test_use_unit_cell_clears_the_saved_cell_selection(tmp_path: Path) -> None:
    service = ProjectService()
    document, asset = _cell_document(service, tmp_path)
    _use_saved_cell(service, document, asset)
    controller, view = _cell_controller(document, RecordingSweepWindow(), service=service)

    controller.use_unit_cell()

    assert document.ui_state["workflow_supercells"]["convergence_cutoff"]["mode"] == "unit_cell"
    assert view.simulation_cell_options[-1]["selected_structure_id"] is None


def test_simulation_cell_requests_without_a_project_change_nothing() -> None:
    window = RecordingSweepWindow(editor_selection=UUID(int=1))
    statuses: list[str] = []
    controller, view = _cell_controller(None, window, statuses=statuses)

    controller.select_simulation_cell(str(UUID(int=1)))
    controller.use_unit_cell()
    controller.create_simulation_cell()

    assert view.simulation_cell_options == []
    assert window.warnings == []
    assert statuses == []

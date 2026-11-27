"""Tests for :class:`UepController`.

Covers submission, the ``density_available`` result flag, ``validate_artifacts``
and definition maintenance, including how UEP ranks a defined job against a
failed one compared with Pristine Relaxation; the UEP page's source structures,
form state, Configure and row removal; and the analysis windows' guards,
flushing and closing.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from PySide6.QtWidgets import QDialog, QWidget

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.uep_results import UEP_ANALYSIS_STATES_KEY

from mufinder.app.workflow_controllers import uep as uep_module
from mufinder.app.workflow_controllers.pristine_relaxation import PristineRelaxationController
from mufinder.app.uep_types import UepConfig, UepRunState, UepSourceRow
from mufinder.app.workflow_controllers.uep import UepController, UepDeps
from mufinder.app.workflow_config_codecs import (
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
)
from tests.support.jobs import set_job_state


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
        self.source_options: list[tuple[list[dict], str | None, str]] = []
        self.species_options: list[tuple[str, ...]] = []

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))

    def set_source_structure_options(self, options, *, selected_id, summary) -> None:
        self.source_options.append((list(options), selected_id, summary))

    def set_parameter_species_options(self, species) -> None:
        self.species_options.append(tuple(species))


def _stub_uep_deps(**overrides) -> UepDeps:
    base = dict(
        dialog_parent=lambda: None,
        refresh_workflow_defaults=lambda: None,
        note_kpoint_adjustments=lambda _expansion: None,
        refresh_journal_view=lambda: None,
        refresh_generation_table=lambda: None,
        shutting_down=lambda: False,
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
    # The base-default workflow-row predicate: rows with nothing run yet and
    # rows whose job ended are ready; active and completed rows are not.
    assert controller.is_ready({"r": FakeRowState("not generated")}, "r") is True
    assert controller.is_ready({"r": FakeRowState("ready")}, "r") is True
    assert controller.is_ready({"r": FakeRowState("failed")}, "r") is True
    assert controller.is_ready({"r": FakeRowState("running")}, "r") is False


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

    output_root = document.project_root / "files" / "uep"
    definition = service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps()
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

    output_root = document.project_root / "files" / "uep"
    service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps()

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
    # The new rows' jobs are created once the backend accepts the submission.
    assert list(extras) == ["on_accepted"]
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

    output_root = document.project_root / "files" / "uep"
    service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps()

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
    # The new rows' jobs are created once the backend accepts the submission.
    assert list(extras) == ["on_accepted"]
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
            job_with_status(service, document, "failed", stage=stage, calc_type=calc_type),
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
    uep = UepController(context, _stub_uep_deps(), _FakeUepView())
    pristine = PristineRelaxationController(
        context,
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
        _stub_uep_deps(),
        _FakeUepView(),
    )

    assert controller.remove_rows(["row_1"]) == (definition.id,)
    assert definition.id not in document.calculations


def test_uep_controller_materialize_gives_a_row_its_own_folder_when_another_job_has_that_name(
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
    document.project.host_structure_id = host.id
    output_root = document.project_root / "files" / "uep"
    service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )
    other = _uep_job(service, document, "uep_row1")
    taken = output_root / other.label
    taken.mkdir(parents=True)
    (taken / "uep_row1.cell").write_text("another job's cell", encoding="utf-8")
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _defn: output_root,
        job_output_directory=lambda job: output_root / job.label,
    )
    deps = _stub_uep_deps()
    controller = UepController(context, deps, _FakeUepView())

    (job,) = controller.materialize(["row1"])

    assert job.label == "uep_row1_2"
    assert (output_root / job.label / f"{job.label}.cell").is_file()
    assert (taken / "uep_row1.cell").read_text(encoding="utf-8") == "another job's cell"


# --------------------------------------------------------------------------- #
# The UEP page: source structures, form state, Configure and row removal
# --------------------------------------------------------------------------- #


def _messages(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    shown: list[tuple[str, str, str]] = []
    for kind in ("information", "warning"):
        monkeypatch.setattr(
            uep_module.QMessageBox,
            kind,
            lambda _parent, title, text, *_args, kind=kind: shown.append((kind, title, text)),
        )
    return shown


def test_refresh_source_structures_offers_the_host_and_its_relaxations(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    relaxed = Structure(
        label="Relaxed",
        lattice_vectors=host.lattice_vectors,
        species=["Ge"],
        fractional_positions=host.fractional_positions,
        origin="pristine_relaxation",
        parent_structure_id=host.id,
    )
    service.add_structure(document, relaxed)
    view = _FakeUepView()
    controller = UepController(make_context(document=document), _stub_uep_deps(), view)

    controller.refresh_source_structures()
    controller.select_source_structure(f"  {relaxed.id}  ")

    host_option = {"id": str(host.id), "label": "Host"}
    relaxed_option = {"id": str(relaxed.id), "label": "Relaxed"}
    assert view.source_options == [
        ([host_option, relaxed_option], str(host.id), "Using source structure: Host"),
        ([host_option, relaxed_option], str(relaxed.id), "Using source structure: Relaxed"),
    ]
    assert view.species_options == [("Si",), ("Ge",)]
    assert controller.selected_source_structure() is relaxed
    assert document.ui_state["uep_form"] == {"selected_source_structure_id": str(relaxed.id)}


def test_refresh_source_structures_without_a_project_or_host(tmp_path: Path) -> None:
    view = _FakeUepView()
    UepController(make_context(document=None), _stub_uep_deps(), view).refresh_source_structures()

    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)
    document.project.host_structure_id = None
    UepController(
        make_context(document=document), _stub_uep_deps(), view
    ).refresh_source_structures()

    assert view.source_options == [
        ([], None, "Create or open a project first."),
        ([], None, "Load a host structure before configuring the UEP workflow."),
    ]
    assert view.species_options == [(), ()]


def test_select_source_structure_refreshes_the_page_defaults(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    document.ui_state["uep_form"] = {"selected_source_structure_id": str(host.id)}
    refreshes: list[str] = []
    controller = UepController(
        make_context(document=document),
        _stub_uep_deps(refresh_workflow_defaults=lambda: refreshes.append("defaults")),
        _FakeUepView(),
    )

    controller.select_source_structure("")

    assert document.ui_state["uep_form"] == {}
    assert refreshes == ["defaults"]
    # Without a project nothing is selected or refreshed.
    UepController(
        make_context(document=None),
        _stub_uep_deps(refresh_workflow_defaults=lambda: refreshes.append("defaults")),
        _FakeUepView(),
    ).select_source_structure(str(host.id))
    assert refreshes == ["defaults"]


def test_store_form_state_refreshes_the_defaults_once_when_they_change_the_form(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)
    refreshes: list[str] = []
    controller: UepController

    def _refresh() -> None:
        refreshes.append("defaults")
        # Pushing the defaults changes the form, which stores it again.
        controller.store_form_state({"name_prefix": "again"})

    controller = UepController(
        make_context(document=document),
        _stub_uep_deps(refresh_workflow_defaults=_refresh),
        _FakeUepView(),
    )

    controller.store_form_state({"name_prefix": "uep"})

    assert refreshes == ["defaults"]
    assert document.ui_state["uep_form"] == {"name_prefix": "again"}


def test_store_form_state_does_nothing_while_the_window_closes(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)
    refreshes: list[str] = []
    controller = UepController(
        make_context(document=document),
        _stub_uep_deps(
            refresh_workflow_defaults=lambda: refreshes.append("defaults"),
            shutting_down=lambda: True,
        ),
        _FakeUepView(),
    )

    controller.store_form_state({"name_prefix": "uep"})

    assert "uep_form" not in document.ui_state
    assert refreshes == []


def test_configure_registers_the_active_definition_on_the_selected_structure(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    calls: list[str] = []
    statuses: list[str] = []
    view = _FakeUepView()
    controller = UepController(
        make_context(document=document, project_service=service, statuses=statuses),
        _stub_uep_deps(
            note_kpoint_adjustments=lambda _expansion: calls.append("kpoints"),
            refresh_journal_view=lambda: calls.append("journal"),
        ),
        view,
    )

    controller.configure(
        UepConfig(name_prefix="uep", rows=(UepSourceRow(label="a"), UepSourceRow(label="b")))
    )

    definition = controller.active_definition()
    assert definition is not None
    assert definition.input_structure_ids == [host.id]
    assert [row["label"] for row in definition.parameters["configuration"]["rows"]] == ["a", "b"]
    assert document.ui_state[UepController.active_key] == str(definition.id)
    assert calls == ["kpoints", "journal"]
    assert view.runtime_states[-1] == {
        "a": UepRunState(status="ready"),
        "b": UepRunState(status="ready"),
    }
    assert statuses == ["Configured 2 UEP source jobs; input files will be generated on submission"]


def test_configure_refuses_without_a_project_or_a_source_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = _messages(monkeypatch)
    config = UepConfig(rows=(UepSourceRow(label="a"),))
    UepController(make_context(document=None), _stub_uep_deps(), _FakeUepView()).configure(config)

    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)
    document.project.host_structure_id = None
    UepController(
        make_context(document=document, project_service=service), _stub_uep_deps(), _FakeUepView()
    ).configure(config)

    assert shown == [
        ("information", "No project", "Create or open a project first."),
        (
            "warning",
            "No source structure",
            "Load a host structure or select a pristine-relaxed structure first.",
        ),
    ]
    assert not document.calculations


def test_configure_reports_a_configuration_the_adapter_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = _messages(monkeypatch)
    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)

    def _refuse(**_kwargs):
        raise ValueError("bad row")

    monkeypatch.setattr(uep_module, "expand_uep_jobs", _refuse)
    UepController(
        make_context(document=document, project_service=service), _stub_uep_deps(), _FakeUepView()
    ).configure(UepConfig(rows=(UepSourceRow(label="a"),)))

    assert shown == [("warning", "UEP setup invalid", "bad row")]
    assert not document.calculations


def test_remove_labelled_rows_drops_the_rows_and_refreshes(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    definition = _uep_sync_definition(service, document, host, {"row_1": "1"})
    view = _FakeUepView()
    controller = UepController(make_context(document=document), _stub_uep_deps(), view)

    controller.remove_labelled_rows(set())
    assert definition.id in document.calculations
    assert view.runtime_states == []

    controller.remove_labelled_rows({"row_1"})
    assert definition.id not in document.calculations
    assert view.runtime_states == [{}]


def test_active_job_ids_reads_the_active_definitions_bindings(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    controller = UepController(make_context(document=document), _stub_uep_deps(), _FakeUepView())
    assert controller.active_job_ids() == set()

    job_id = uuid4()
    definition = _uep_sync_definition(service, document, host, {"row_1": str(job_id)})
    definition.parameters["job_ids_by_label"]["row_2"] = "not an id"
    document.ui_state[UepController.active_key] = str(definition.id)

    assert controller.active_job_ids() == {job_id}


# --------------------------------------------------------------------------- #
# The analysis windows
# --------------------------------------------------------------------------- #


def _completed_uep_row(service, document, host, *, status=JobStatus.COMPLETED):
    job = _uep_job(service, document, "uep_row_1")
    set_job_state(job, status)
    definition = _uep_sync_definition(service, document, host, {"row_1": str(job.id)})
    document.ui_state[UepController.active_key] = str(definition.id)
    return job


def test_open_analysis_refuses_rows_without_a_completed_job_or_its_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown = _messages(monkeypatch)
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    def _controller(directory):
        context = make_context(document=document, job_output_directory=lambda _job: directory)
        return UepController(context, _stub_uep_deps(), _FakeUepView())

    _controller(output_dir).open_analysis("row_1")
    _completed_uep_row(service, document, host, status="failed")
    _controller(output_dir).open_analysis("row_1")
    _completed_uep_row(service, document, host)
    _controller(None).open_analysis("row_1")
    _controller(output_dir).open_analysis("row_1")
    _controller(output_dir).open_analysis("   ")

    missing_files = (
        "UEP analysis requires both `.den_fmt` and `.castep` files from the completed source job."
    )
    assert shown == [
        ("information", "No UEP workflow", "Configure and complete a UEP source job first."),
        ("information", "UEP source incomplete", "Only completed UEP source jobs can be analysed."),
        ("warning", "UEP files unavailable", "Could not locate the UEP job output directory."),
        ("warning", "UEP files unavailable", missing_files),
    ]


def test_analysis_state_and_search_are_not_kept_while_the_window_closes(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    job = _completed_uep_row(service, document, host)
    calls: list[str] = []
    controller = UepController(
        make_context(document=document),
        _stub_uep_deps(
            refresh_journal_view=lambda: calls.append("journal"),
            shutting_down=lambda: True,
        ),
        _FakeUepView(),
    )

    controller.analysis_state_changed(job.id, {"current_tab": 1})
    controller.search_changed(job.id, None)

    assert UEP_ANALYSIS_STATES_KEY not in document.ui_state
    assert calls == []


def test_flush_and_close_reach_every_analysis_window(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Dialog:
        def __init__(self, *, deleted: bool = False) -> None:
            self.deleted = deleted
            self.calls: list[str] = []

        def flush_pending_state(self) -> None:
            if self.deleted:
                raise RuntimeError("Internal C++ object already deleted.")
            self.calls.append("flush")

        def close(self) -> None:
            if self.deleted:
                raise RuntimeError("Internal C++ object already deleted.")
            self.calls.append("close")

    class _ChildDialog(QDialog):
        closes = 0

        def close(self) -> bool:
            type(self).closes += 1
            return super().close()

    monkeypatch.setattr(uep_module, "UepAnalysisDialog", _ChildDialog)
    parent = QWidget()
    child = _ChildDialog(parent)
    controller = UepController(
        make_context(), _stub_uep_deps(dialog_parent=lambda: parent), _FakeUepView()
    )
    live, deleted = _Dialog(), _Dialog(deleted=True)
    controller.analysis_dialogs.update({uuid4(): deleted, uuid4(): live})

    controller.flush_analysis_dialogs()
    controller.close_analysis_dialogs()

    assert live.calls == ["flush", "close"]
    assert _ChildDialog.closes == 1
    assert controller.analysis_dialogs == {}
    child.deleteLater()
    parent.deleteLater()

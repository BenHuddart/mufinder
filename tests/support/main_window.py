"""Set up a ``MainWindow`` with a project open, for the app integration tests.

The ``main_window`` and ``project_with_host`` fixtures in ``tests/conftest.py``
build on these helpers; a module that needs its own project name or host binds
them with ``functools.partial``. ``MainWindow`` is imported inside each helper so
importing this module stays cheap.

The rest of the module holds the window helpers that several
``tests/app/main_window`` modules share: selecting a project-tree route, cluster
submission doubles for the continuation and fresh-start flows, a staged
Quantum Corrections row and a recording stand-in for the failed-job dialog.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Result, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.quantum import QUANTUM_SINGLEPOINT_CALC_TYPE
from tests.support.jobs import set_job_state
from tests.support.structures import host_structure

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow


def empty_project(tmp_path: Path, name: str = "Alpha") -> Path:
    """Create an empty project ``<name>.mufinder`` under ``tmp_path`` and return its folder."""

    return ProjectService().create_project(tmp_path / f"{name}.mufinder", name).project_root


def answer_new_project(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    accept: bool = True,
    location: Path | None = None,
) -> list[str]:
    """Answer the New Project dialog: type ``name`` and press Create or Cancel.

    Returns a list that collects the location each dialog opened with.
    """

    from PySide6.QtWidgets import QDialog

    from mufinder.app.new_project_dialog import NewProjectDialog

    starts: list[str] = []

    def _exec(dialog: NewProjectDialog) -> int:
        starts.append(dialog._location_edit.text())
        if location is not None:
            dialog._location_edit.setText(str(location))
        dialog._name_edit.setText(name)
        return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected

    monkeypatch.setattr(NewProjectDialog, "exec", _exec)
    return starts


def open_project_with_host(
    window: MainWindow,
    tmp_path: Path,
    *,
    filename: str = "project.mufinder",
    name: str = "Project",
    host: Structure | None = None,
) -> tuple[ProjectService, ProjectDocument, Structure]:
    """Create a project under ``tmp_path``, load ``host`` as its host structure, and show it in ``window``.

    ``host`` defaults to :func:`tests.support.structures.host_structure`.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / filename, name)
    host = host_structure() if host is None else host
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    window._document = document
    window._refresh_ui_after_project_change()
    return service, document, host


def window_with_host(
    tmp_path: Path,
    **project: object,
) -> tuple[MainWindow, ProjectService, ProjectDocument, Structure]:
    """A new ``MainWindow`` with a host-structure project open; see :func:`open_project_with_host`."""

    from mufinder.app.main_window import MainWindow

    window = MainWindow()
    service, document, host = open_project_with_host(window, tmp_path, **project)  # type: ignore[arg-type]
    return window, service, document, host


def select_route(window: MainWindow, route: str) -> None:
    """Pick ``route``'s navigator row as the user does and let the window react to it.

    A row already current is left alone: its page is the one showing.
    """

    navigator = window._project_navigator
    index = navigator.index_of(route)
    assert index.isValid(), route
    navigator.view.setCurrentIndex(index)


def flush_dashboard(window: MainWindow) -> None:
    """Run the dashboard refresh and autosave a job event schedules, now."""

    window._dashboard_flush_pending = True
    window._flush_dashboard()


CONTINUATION_CLUSTER = ClusterConfig(
    name="alpha",
    hostname="alpha.example",
    username="alice",
    auth_method="key",
    key_path="/tmp/key",
    default_working_directory="/remote/work",
    queue_system="slurm",
    cores_per_node=64,
    default_walltime="01:00:00",
    template_script_path="/templates/slurm.sh",
)


def patch_continuation_cluster_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept the cluster submission dialog with a fixed /remote/work config."""

    import mufinder.app.jobs_controller as jobs_controller_module
    import mufinder.app.main_window as main_window_module

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(
        main_window_module, "load_cluster_configs", lambda _settings: [CONTINUATION_CLUSTER]
    )
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_configs", lambda _settings: [CONTINUATION_CLUSTER]
    )
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: CONTINUATION_CLUSTER
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None
    )


def add_cluster_input_job(service: ProjectService, doc, label: str) -> Job:
    """Create a convergence job with its .cell/.param inputs on disk.

    The job directory basename equals the label, exactly as a real submission
    produces it, so ``_job_seed_name`` resolves the same seed from either the
    local directory or the remote basename.
    """

    job = service.add_job(
        doc,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job_dir = doc.project_root / "files" / "convergence_cutoff" / label
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{label}.cell").write_text(
        "%BLOCK LATTICE_CART\n%ENDBLOCK LATTICE_CART\n", encoding="utf-8"
    )
    (job_dir / f"{label}.param").write_text("TASK : GEOMETRYOPTIMIZATION\n", encoding="utf-8")
    return job


def patch_continuation_cluster_runner(
    monkeypatch: pytest.MonkeyPatch,
    window: MainWindow,
    *,
    directory_exists: bool,
) -> dict[str, list]:
    """Patch the SSH-facing runner calls and record what the flow asked for."""

    recorded: dict[str, list] = {"checked": [], "deleted": [], "submitted": []}
    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda path: recorded["checked"].append(path) or directory_exists,
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda path: recorded["deleted"].append(path),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: recorded["submitted"].append(kwargs) or "99123",
    )
    return recorded


def stage_quantum_row(
    window: MainWindow,
    service: ProjectService,
    doc,
    *,
    singlepoint_status: JobStatus | None = None,
    label: str = "siteA",
):
    """Stage one eligible Quantum Corrections row, its site named ``label``, on ``window``'s active definition.

    With ``singlepoint_status``, the row also holds the singlepoint job an older
    MuFinder version ran before the phonon job, bound through
    ``singlepoint_job_ids_by_label`` in that status; the third return value is
    that job, else ``None``.
    """

    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure

    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(
        job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label=label, status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site

    definition = window._quantum_corrections_controller.ensure_definition([str(site.id)])
    assert definition is not None

    singlepoint_job = None
    if singlepoint_status is not None:
        singlepoint_job = service.add_job(
            doc,
            label=f"quantum_singlepoint_{label}",
            workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
        )
        set_job_state(singlepoint_job, singlepoint_status)
        definition.parameters["singlepoint_job_ids_by_label"] = {
            str(site.id): str(singlepoint_job.id)
        }

    candidate = window._quantum_corrections_controller.candidate_by_key(str(site.id))
    assert candidate is not None
    return definition, candidate, singlepoint_job


#: The Muon Relaxations row key :func:`stage_muon_relaxation_row` stages.
STAGED_MUON_RELAXATION_KEY = "positivemuon:Site 1"


def stage_muon_relaxation_row(
    window: MainWindow,
    service: ProjectService,
    doc: ProjectDocument,
    *,
    rows: Sequence[tuple[str, str]] = (("Site 1", "positive_muon"),),
) -> None:
    """Stage Muon Relaxations rows on a one-atom Si host as ``window``'s active definition.

    ``rows`` names each row's site label and charge state; the default is one
    row, ``Site 1`` as a positive muon. The page's stored form is the
    definition's configuration, as a configured page leaves it.
    """

    from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow
    from mufinder.app.workflow_config_codecs import serialize_muon_relaxation_config

    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    staged = []
    for index, (label, charge_state) in enumerate(rows):
        position = (0.1 + 0.1 * index, 0.2, 0.3)
        site = MuonSite(
            label=label,
            initial_fractional_position=list(position),
            status="staged",
            source_structure_id=host.id,
        )
        doc.sites[site.id] = site
        staged.append(
            MuonRelaxationRow(
                site_id=site.id,
                label=label,
                initial_fractional_position=position,
                charge_state=charge_state,
                parameter_values={"geom_max_iter": 120},
            )
        )
    config = MuonRelaxationConfig(rows=tuple(staged))
    definition = service.register_muon_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(config),
        output_directory=doc.project_root / "files" / "muon_relaxation",
    )
    doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["muon_relaxation_form"] = serialize_muon_relaxation_config(config)


def capture_failed_job_dialogs(
    monkeypatch: pytest.MonkeyPatch, *, action: object = None
) -> list[object]:
    """Replace the window's failed-job dialog with one that records itself and answers.

    Each dialog the window opens is appended to the returned list without being
    shown; its ``exec`` answers with ``action`` (a ``FailedJobAction``, or
    ``None`` for Close).
    """

    import mufinder.app.jobs_controller as jobs_controller_module
    from mufinder.app.failed_job_dialog import FailedJobDialog

    dialogs: list[object] = []

    class _RecordingDialog(FailedJobDialog):
        def exec(self) -> int:
            dialogs.append(self)
            self._action = action
            return 1 if action is not None else 0

    monkeypatch.setattr(jobs_controller_module, "FailedJobDialog", _RecordingDialog)
    return dialogs


class _FakeLocalSubmissionDialog:
    """The local submission dialog, answered without being shown."""

    accept = True
    opened: list[dict[str, object]] = []

    class DialogCode:
        Accepted = 1

    def __init__(self, **kwargs) -> None:
        type(self).opened.append(kwargs)

    def exec(self) -> int:
        return 1 if type(self).accept else 0

    def config(self):
        return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)


def fake_local_submission(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, *, accept: bool = True
) -> list[list[Job]]:
    """Answer the local submission dialog with ``accept``, and record what it submits.

    Everything up to the runner runs for real: the dialog's answer, the
    previous output cleared for a start-again, the run-again plan applied and
    the inputs rebuilt. The local runner is replaced by a recorder, which
    returns the jobs each submission handed it.
    """

    from mufinder.app import jobs_controller as jobs_controller_module
    from mufinder.castep_setup import CastepSetup
    from mufinder.services.job_execution_controller import SubmissionOutcome

    _FakeLocalSubmissionDialog.accept = accept
    _FakeLocalSubmissionDialog.opened = []
    monkeypatch.setattr(jobs_controller_module, "LocalSubmissionDialog", _FakeLocalSubmissionDialog)
    monkeypatch.setattr(
        jobs_controller_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/opt/castep", mpirun_path=""),
    )
    monkeypatch.setattr(window._keyword_catalog_guard, "confirm_submission", lambda: True)
    submitted: list[list[Job]] = []

    def submit_local_jobs(jobs, **_kwargs):
        submitted.append(list(jobs))
        return SubmissionOutcome(submitted_jobs=tuple(jobs))

    monkeypatch.setattr(window._job_execution, "submit_local_jobs", submit_local_jobs)
    return submitted

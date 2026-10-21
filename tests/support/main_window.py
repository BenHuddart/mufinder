"""Set up a ``MainWindow`` with a project open, for the app integration tests.

The ``main_window`` and ``project_with_host`` fixtures in ``tests/conftest.py``
build on these helpers; a module that needs its own project name or host binds
them with ``functools.partial``. ``MainWindow`` is imported inside each helper so
importing this module stays cheap.

The rest of the module holds the window helpers that several
``tests/app/main_window`` modules share: selecting a project-tree route, cluster
submission doubles for the continuation and fresh-start flows, and a staged
Quantum Corrections row.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Result, Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.quantum import QUANTUM_SINGLEPOINT_CALC_TYPE
from tests.support.qt import find_tree_item_by_route
from tests.support.structures import host_structure

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow


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
    """Select the project-tree item for ``route`` and let the window react to it."""

    item = find_tree_item_by_route(window._project_tree, route)
    assert item is not None
    window._project_tree.setCurrentItem(item)
    window._on_tree_selection_changed()


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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [CONTINUATION_CLUSTER])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: CONTINUATION_CLUSTER)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)


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
    (job_dir / f"{label}.cell").write_text("%BLOCK LATTICE_CART\n%ENDBLOCK LATTICE_CART\n", encoding="utf-8")
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
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
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
):
    """Stage one eligible Quantum Corrections row on ``window``'s active definition.

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
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site

    definition = window._quantum_corrections_controller.ensure_definition([str(site.id)])
    assert definition is not None

    singlepoint_job = None
    if singlepoint_status is not None:
        singlepoint_job = service.add_job(
            doc,
            label="quantum_singlepoint_siteA",
            workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
        )
        singlepoint_job.status = singlepoint_status
        definition.parameters["singlepoint_job_ids_by_label"] = {str(site.id): str(singlepoint_job.id)}

    candidate = window._quantum_corrections_controller.candidate_by_key(str(site.id))
    assert candidate is not None
    return definition, candidate, singlepoint_job

"""Fakes shared by the workflow-controller tests.

The tests in ``tests/app/workflow_controllers`` drive each controller with
narrow fakes for the :class:`WorkflowControllerContext` callables, so no
``MainWindow`` is constructed. The fakes more than one of those modules needs
live here:

- ``RecordingBackend`` stands in for the local and cluster submit backends and
  records ``(jobs, kwargs)`` for each call.
- ``make_context`` builds a :class:`WorkflowControllerContext` from a document,
  a ``ProjectService`` and recording backends; every other callable is a no-op.
- ``FakeRowState`` stands in for a workflow's per-row run state (a ``status``
  and an optional ``job_id``).
- ``no_artifacts`` and ``completed_castep_result`` are the parsed-output
  carriers a completion hands ``build_result`` and ``validate_artifacts``.
- ``muon_host_document`` creates a project with a one-atom Si host structure.
- ``job_with_status`` registers a job with a given status and returns its id.
- ``FakeRelaxationView``, ``stub_pristine_deps`` and ``pristine_sync_definition``
  stand in for the Pristine Relaxation widget, its dependency bundle and a
  registered definition; the UEP tests compare their sync ranking with it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.base import WorkflowControllerContext
from mufinder.app.workflow_controllers.pristine_relaxation import PristineRelaxationDeps
from mufinder.app.workflow_config_codecs import serialize_pristine_relaxation_config
from mufinder.app.workflow_submission import WorkflowSubmissionCoordinator


def no_artifacts() -> ParsedJobArtifacts:
    """An empty artifacts carrier for controllers that consume no secondary output."""

    return ParsedJobArtifacts()


class RecordingBackend:
    """Records ``(jobs, kwargs)`` for each submit call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Job], dict[str, object]]] = []

    def __call__(self, jobs: list[Job], **kwargs: object) -> None:
        self.calls.append((list(jobs), dict(kwargs)))


def make_context(
    *,
    document: ProjectDocument | None = None,
    project_service: ProjectService | None = None,
    local_backend: RecordingBackend | None = None,
    cluster_backend: RecordingBackend | None = None,
    infos: list[tuple[str, str]] | None = None,
    prepare_jobs_for_resubmission: Callable[[object], bool] | None = None,
    definition_output_directory: Callable[[object], object] = lambda _d: None,
    job_output_directory: Callable[[object], object] = lambda _j: None,
    job_seed_name: Callable[[object], str] = lambda j: j.label,
    run_for_job: Callable[[object], object] = lambda _r: None,
    total_energy_for_job: Callable[[object], object] = lambda _j: None,
    max_force_for_job: Callable[[object], object] = lambda _j: None,
) -> WorkflowControllerContext:
    service = project_service or ProjectService()
    return WorkflowControllerContext(
        document=lambda: document,
        project_service=service,
        job_execution=None,  # type: ignore[arg-type]  # unused by these tests
        submission_coordinator=WorkflowSubmissionCoordinator(),
        submit_jobs_locally=local_backend or RecordingBackend(),
        submit_jobs_on_cluster=cluster_backend or RecordingBackend(),
        prepare_jobs_for_resubmission=prepare_jobs_for_resubmission or (lambda _jobs: True),
        show_info=(lambda title, text: infos.append((title, text))) if infos is not None else (lambda *_: None),
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=lambda **_: None,
        status_message=lambda _message: None,
        definition_output_directory=definition_output_directory,
        job_output_directory=job_output_directory,
        job_seed_name=job_seed_name,
        run_for_job=run_for_job,
        total_energy_for_job=total_energy_for_job,
        max_force_for_job=max_force_for_job,
    )


class FakeRowState:
    """Minimal state object with a ``status`` and optional ``job_id``."""

    def __init__(self, status: str, job_id: object | None = None) -> None:
        self.status = status
        self.job_id = job_id


def completed_castep_result() -> CastepResult:
    return CastepResult(completed=True, total_energy=-1.0, warnings=("parse warning",))


class FakeRelaxationView:
    """Minimal PristineRelaxationWidget stand-in for controller tests.

    ``collect_state``/``set_form_state`` store the form state verbatim: the real
    widget also normalizes it (stripping implicit project defaults), and that
    round trip is pinned against the real widget in
    ``tests/app/main_window/test_relaxation_actions.py``.
    """

    def __init__(self, form_state: dict | None = None) -> None:
        self.runtime_states: list[dict] = []
        self.form_state: dict = form_state if form_state is not None else {"rows": []}

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))

    def collect_state(self) -> dict:
        return copy.deepcopy(self.form_state)

    def set_form_state(self, state: dict) -> None:
        self.form_state = copy.deepcopy(state)


def stub_pristine_deps(**overrides) -> PristineRelaxationDeps:
    base = dict(
        deserialize_pristine_relaxation_config=lambda _payload: None,
        serialize_pristine_relaxation_config=serialize_pristine_relaxation_config,
        pristine_relaxation_form_config=lambda: None,
        pristine_relaxation_row_signature=lambda _row: (),
        expand_pristine_relaxation_jobs=lambda _definition, _config: None,
    )
    base.update(overrides)
    return PristineRelaxationDeps(**base)


def muon_host_document(service: ProjectService, tmp_path: Path):
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


def job_with_status(
    service: ProjectService,
    document: ProjectDocument,
    status: JobStatus,
    *,
    stage: WorkflowStage,
    calc_type: str,
) -> str:
    job = service.add_job(document, label="job", workflow_stage=stage, calc_type=calc_type)
    job.status = status
    return str(job.id)


def pristine_sync_definition(service, document, host, rows=("row_1",), job_ids_by_label=None):
    from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow

    return service.register_pristine_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters=serialize_pristine_relaxation_config(
            PristineRelaxationConfig(rows=tuple(PristineRelaxationRow(label=label) for label in rows))
        ),
        output_directory=document.project_root / "files" / "pristine_relaxation",
        job_ids_by_label=job_ids_by_label,
    )

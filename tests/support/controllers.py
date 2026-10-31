"""Fakes shared by the workflow-controller tests.

The tests in ``tests/app/workflow_controllers`` drive each controller with
narrow fakes for the :class:`WorkflowControllerContext` callables, so no
``MainWindow`` is constructed. The fakes more than one of those modules needs
live here:

- ``RecordingBackend`` stands in for the local and cluster submit backends and
  records ``(jobs, kwargs)`` for each call.
- ``make_context`` builds a :class:`WorkflowControllerContext` from a document,
  a ``ProjectService`` and recording backends, with a real ``JobFileCleanup``
  over the given job readers; ``infos``, ``statuses`` and ``persisted`` record
  the information boxes, status messages and ``persist_ui_state`` failure
  messages; every other callable is a no-op.
- ``RecordingSweepWindow`` stands in for the window side of a convergence
  sweep's deps (the examiner dialog, warnings, confirmations, refreshes and
  the Supercell Editor).
- ``FakeRowState`` stands in for a workflow's per-row run state (a ``status``
  and an optional ``job_id``).
- ``no_artifacts`` and ``completed_castep_result`` are the parsed-output
  carriers a completion hands ``build_result`` and ``validate_artifacts``.
- ``muon_host_document`` creates a project with a one-atom Si host structure.
- ``job_with_status`` registers a job with a given status and returns its id.
- ``FakeRelaxationView`` and ``pristine_sync_definition`` stand in for the
  Pristine Relaxation widget and a registered definition; the UEP tests compare
  their sync ranking with it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.job_file_cleanup import JobFileCleanup
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.base import WorkflowControllerContext
from mufinder.app.workflow_config_codecs import serialize_pristine_relaxation_config
from mufinder.app.workflow_dialogs import SourceConfigureResult
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
    statuses: list[str] | None = None,
    persisted: list[str] | None = None,
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
        show_info=(lambda title, text: infos.append((title, text)))
        if infos is not None
        else (lambda *_: None),
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=(lambda *, failure_message: persisted.append(failure_message))
        if persisted is not None
        else (lambda **_: None),
        status_message=statuses.append if statuses is not None else (lambda _message: None),
        set_submitted_rows=lambda _keys: None,
        definition_output_directory=definition_output_directory,
        job_output_directory=job_output_directory,
        job_seed_name=job_seed_name,
        job_file_cleanup=JobFileCleanup(
            document_getter=lambda: document,
            job_output_directory=job_output_directory,  # type: ignore[arg-type]
            job_seed_name=job_seed_name,  # type: ignore[arg-type]
        ),
        run_for_job=run_for_job,
        total_energy_for_job=total_energy_for_job,
        max_force_for_job=max_force_for_job,
    )


class RecordingSweepWindow:
    """The window side of a convergence sweep's deps, recording every call.

    ``run_examiner`` records the points and dialog options, keeps the adopt and
    state callbacks for the test to fire, and returns ``examiner_state`` as the
    dialog's final state. ``confirm_choice`` answers ``confirm_answer``.
    ``select_simulation_cell_in_editor`` records the stage and the editor's
    initial matrix, then picks ``editor_selection`` when it is set.
    ``refresh_jobs_table`` and ``refresh_journal_view`` append their names to
    ``window_refreshes``.
    """

    def __init__(
        self,
        *,
        examiner_state: dict[str, object] | None = None,
        confirm_answer: str = "change",
        editor_selection: UUID | None = None,
    ) -> None:
        self.examiner_state = examiner_state if examiner_state is not None else {}
        self.confirm_answer = confirm_answer
        self.examiner_calls: list[tuple[dict[float, object], dict[str, object]]] = []
        self.on_adopted: Callable[..., None] | None = None
        self.on_state_changed: Callable[[dict], None] | None = None
        self.warnings: list[tuple[str, str]] = []
        self.confirmations: list[tuple[str, str, list[tuple[str, str]], dict[str, object]]] = []
        self.refreshes = 0
        self.editor_selection = editor_selection
        self.editor_calls: list[tuple[WorkflowStage, list[list[int]]]] = []
        self.window_refreshes: list[str] = []

    def run_examiner(
        self,
        points: dict[float, object],
        *,
        on_adopted: Callable[..., None],
        on_state_changed: Callable[[dict], None],
        **options: object,
    ) -> dict[str, object]:
        self.examiner_calls.append((dict(points), dict(options)))
        self.on_adopted = on_adopted
        self.on_state_changed = on_state_changed
        return dict(self.examiner_state)

    def show_warning(self, title: str, text: str) -> None:
        self.warnings.append((title, text))

    def confirm_choice(
        self, title: str, text: str, choices: list[tuple[str, str]], **kwargs: object
    ) -> str:
        self.confirmations.append((title, text, list(choices), dict(kwargs)))
        return self.confirm_answer

    def refresh_after_default_adopted(self) -> None:
        self.refreshes += 1

    def select_simulation_cell_in_editor(
        self,
        workflow_stage: WorkflowStage,
        *,
        initial_matrix: Callable[[], list[list[int]]],
        on_selected: Callable[[UUID], None],
    ) -> None:
        self.editor_calls.append((workflow_stage, initial_matrix()))
        if self.editor_selection is not None:
            on_selected(self.editor_selection)

    def refresh_jobs_table(self) -> None:
        self.window_refreshes.append("jobs_table")

    def refresh_journal_view(self) -> None:
        self.window_refreshes.append("journal_view")

    def deps(self) -> dict[str, Callable[..., object]]:
        """Return the deps fields both sweeps' deps dataclasses take."""

        return {
            "run_examiner": self.run_examiner,
            "show_warning": self.show_warning,
            "confirm_choice": self.confirm_choice,
            "refresh_after_default_adopted": self.refresh_after_default_adopted,
            "select_simulation_cell_in_editor": self.select_simulation_cell_in_editor,
            "refresh_jobs_table": self.refresh_jobs_table,
            "refresh_journal_view": self.refresh_journal_view,
        }


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
            PristineRelaxationConfig(
                rows=tuple(PristineRelaxationRow(label=label) for label in rows)
            )
        ),
        output_directory=document.project_root / "files" / "pristine_relaxation",
        job_ids_by_label=job_ids_by_label,
    )


def configure_result(values=None, advanced=(), untouched=()) -> SourceConfigureResult:
    """A source-configure dialog result with the given values and flags."""

    return SourceConfigureResult(
        values=dict(values or {}),
        advanced_names=frozenset(advanced),
        untouched_mixed=frozenset(untouched),
    )

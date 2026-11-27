"""What a Run or ``Resubmit…`` saves, from the window's own project, with the dialog answered.

Muon Relaxations and Quantum Corrections create their jobs only once a
submission is accepted, and save then when anything is unsaved. A cancelled
Run or ``Resubmit…`` therefore writes nothing, a new row's included, while
every accepted submission saves what it creates. Each project is saved and
reopened through the window first, as File > Open leaves it, so the window's
project service knows what is on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService, UnsavedChanges
from tests.support.jobs import set_job_state
from tests.support.main_window import (
    STAGED_MUON_RELAXATION_KEY as _MUON_KEY,
)
from tests.support.main_window import (
    fake_local_submission,
    stage_muon_relaxation_row,
    stage_quantum_row,
)
from tests.support.submission_snapshot import cancel_submission as _cancel_submission
from tests.support.submission_snapshot import project_files as _project_files
from tests.support.submission_snapshot import reopen_saved as _reopen


def _changed(before: dict[str, bytes], after: dict[str, bytes]) -> set[str]:
    """Return the files present before that differ or are gone after, ids shown as ``<id>``."""

    return {_without_id(name) for name, data in before.items() if after.get(name) != data}


def _without_id(name: str) -> str:
    stem = Path(name).stem
    return name.replace(stem, "<id>") if len(stem) == 36 and stem.count("-") == 4 else name


#: What a new row's Run rewrites before its dialog, besides the job and inputs it adds.
_NEW_JOB_CHANGES = {"entities/calculations/<id>.json", "project.json"}
#: What an accepted run again of an ended job rewrites.
_RUN_AGAIN_CHANGES = {"entities/jobs/<id>.json", "project.json", "provenance.jsonl"}


def _modified_at(root: Path) -> str:
    return json.loads((root / "project.json").read_text(encoding="utf-8"))["modified_at"]


def _saved_document(window: MainWindow):
    """Load what is on disk for the window's project, through a service of its own."""

    return ProjectService().load_project(window._document.project_root)


def _muon_project(window: MainWindow, tmp_path: Path, *, status: str | None, target: str):
    """A Muon Relaxations page with one row; with ``status``, its job has run and ended so."""

    service = ProjectService()
    document = service.create_project(tmp_path / "muon.mufinder", "Muon")
    stage_muon_relaxation_row(window, service, document)
    window._document = document
    window._refresh_ui_after_project_change()
    if status is not None:
        (job,) = window._muon_relaxation_controller.materialize([_MUON_KEY])
        set_job_state(job, status)
        job.execution_target = target
    return _reopen(window, document)


def _quantum_project(window: MainWindow, tmp_path: Path, *, status: str | None, target: str):
    """A Quantum Corrections page with one row; with ``status``, its job has run and ended so."""

    service = ProjectService()
    document = service.create_project(tmp_path / "quantum.mufinder", "Quantum")
    window._document = document
    _definition, candidate, _singlepoint = stage_quantum_row(window, service, document)
    if status is not None:
        (job,) = window._quantum_corrections_controller.materialize([candidate.key])
        set_job_state(job, status)
        job.execution_target = target
    return _reopen(window, document), candidate.key


def _hyperfine_project(window: MainWindow, tmp_path: Path, *, target: str):
    """A Hyperfine Coupling row whose job has run and failed."""

    service = ProjectService()
    document = service.create_project(tmp_path / "hyperfine.mufinder", "Hyperfine")
    window._document = document
    _definition, candidate, _singlepoint = stage_quantum_row(window, service, document)
    window._hyperfine_coupling_controller.ensure_definition([candidate.key])
    (job,) = window._hyperfine_coupling_controller.materialize([candidate.key])
    set_job_state(job, "failed")
    job.execution_target = target
    return _reopen(window, document), candidate.key


def _only_job(document, stage: WorkflowStage = WorkflowStage.MUON_RELAXATION):
    """Return the one job of ``stage``; the quantum rows' relaxation job is the other."""

    (job,) = [job for job in document.jobs.values() if job.workflow_stage == stage]
    return job


# ---------------------------------------------------------------------------
# An ended job run again, cancelled: nothing is written
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["local", "alpha"])
def test_a_cancelled_resubmit_of_a_failed_muon_relaxation_writes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    window = main_window
    document = _muon_project(window, tmp_path, status="failed", target=target)
    job = _only_job(document)
    before = _project_files(document.project_root)
    opened = _cancel_submission(window, monkeypatch, target)

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert opened == ["local" if target == "local" else "cluster"]
    assert _project_files(document.project_root) == before
    assert window._project_service.unsaved_changes(document) is UnsavedChanges.NONE
    assert job.status == JobStatus.ENDED


@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_a_cancelled_run_of_a_failed_quantum_corrections_row_writes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    window = main_window
    document, key = _quantum_project(window, tmp_path, status="failed", target="local")
    job = _only_job(document, WorkflowStage.QUANTUM_CORRECTIONS)
    before = _project_files(document.project_root)
    opened = _cancel_submission(window, monkeypatch, "local")

    if entry == "row":
        window._quantum_corrections_view.submit_local_requested.emit([key])
    else:
        window._jobs._on_resubmit_from_dashboard([job.id])

    assert opened == ["local"]
    assert _project_files(document.project_root) == before
    assert window._project_service.unsaved_changes(document) is UnsavedChanges.NONE
    assert job.status == JobStatus.ENDED


def test_a_cancelled_resubmit_of_a_failed_hyperfine_coupling_writes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    document, _key = _hyperfine_project(window, tmp_path, target="local")
    job = _only_job(document, WorkflowStage.HYPERFINE_COUPLING)
    before = _project_files(document.project_root)
    opened = _cancel_submission(window, monkeypatch, "local")

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert opened == ["local"]
    assert _project_files(document.project_root) == before
    assert window._project_service.unsaved_changes(document) is UnsavedChanges.NONE


def test_a_cancelled_resubmit_leaves_what_was_unsaved_before_it_unsaved(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cancelled submission saves nothing, not even changes made before it."""

    window = main_window
    document = _muon_project(window, tmp_path, status="failed", target="local")
    job = _only_job(document)
    job.label = "Site 1 (renamed)"
    before = _project_files(document.project_root)
    _cancel_submission(window, monkeypatch, "local")

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert _project_files(document.project_root) == before
    assert window._project_service.unsaved_changes(document) is UnsavedChanges.ROUTINE


# ---------------------------------------------------------------------------
# A new row's cancelled Run creates nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["local", "alpha"])
def test_a_cancelled_run_of_a_new_muon_relaxation_row_creates_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    window = main_window
    document = _muon_project(window, tmp_path, status=None, target=target)
    root = document.project_root
    before = _project_files(root)
    opened = _cancel_submission(window, monkeypatch, target)

    window._muon_relaxation_controller.submit(
        [_MUON_KEY], target="local" if target == "local" else "cluster"
    )

    assert opened == ["local" if target == "local" else "cluster"]
    assert not [
        job for job in document.jobs.values() if job.workflow_stage == WorkflowStage.MUON_RELAXATION
    ]
    assert _project_files(root) == before
    assert window._project_service.unsaved_changes(document) is UnsavedChanges.NONE


def test_a_cancelled_run_of_a_new_quantum_corrections_row_creates_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    document, key = _quantum_project(window, tmp_path, status=None, target="local")
    root = document.project_root
    before = _project_files(root)
    jobs_before = set(document.jobs)
    opened = _cancel_submission(window, monkeypatch, "local")

    window._quantum_corrections_view.submit_local_requested.emit([key])

    assert opened == ["local"]
    assert set(document.jobs) == jobs_before
    assert _project_files(root) == before
    assert window._project_service.unsaved_changes(document) is UnsavedChanges.NONE


# ---------------------------------------------------------------------------
# Accepted submissions save what they change
# ---------------------------------------------------------------------------


def _assert_saved_as_defined(
    window: MainWindow,
    job_id,
    mapping_key: str,
    before: dict[str, bytes],
    *,
    changed: set[str],
    added: int,
) -> None:
    """The job is saved ``defined`` on its row, and the folder changed only as pinned."""

    after = _project_files(window._document.project_root)
    assert _changed(before, after) == changed
    assert len(set(after) - set(before)) == added
    saved = _saved_document(window)
    assert saved.jobs[job_id].status == JobStatus.DEFINED
    definitions = [
        definition
        for definition in saved.calculations.values()
        if definition.parameters.get("job_ids_by_label", {}).get(mapping_key) == str(job_id)
    ]
    assert len(definitions) == 1
    assert window._project_service.unsaved_changes(window._document) is UnsavedChanges.NONE


def test_an_accepted_resubmit_of_a_failed_muon_relaxation_saves_its_return_to_defined(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    document = _muon_project(window, tmp_path, status="failed", target="local")
    job = _only_job(document)
    before = _project_files(document.project_root)
    submitted = fake_local_submission(window, monkeypatch)

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert submitted == [[job]]
    _assert_saved_as_defined(window, job.id, _MUON_KEY, before, changed=_RUN_AGAIN_CHANGES, added=0)


@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_an_accepted_run_of_a_failed_quantum_corrections_row_saves_its_return_to_defined(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    window = main_window
    document, key = _quantum_project(window, tmp_path, status="failed", target="local")
    job = _only_job(document, WorkflowStage.QUANTUM_CORRECTIONS)
    before = _project_files(document.project_root)
    submitted = fake_local_submission(window, monkeypatch)

    if entry == "row":
        window._quantum_corrections_view.submit_local_requested.emit([key])
    else:
        window._jobs._on_resubmit_from_dashboard([job.id])

    assert submitted == [[job]]
    _assert_saved_as_defined(window, job.id, key, before, changed=_RUN_AGAIN_CHANGES, added=0)


def test_an_accepted_run_of_a_new_muon_relaxation_row_saves_its_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    document = _muon_project(window, tmp_path, status=None, target="local")
    before = _project_files(document.project_root)
    submitted = fake_local_submission(window, monkeypatch)

    window._muon_relaxation_controller.submit([_MUON_KEY], target="local")

    job = _only_job(document)
    assert submitted == [[job]]
    _assert_saved_as_defined(window, job.id, _MUON_KEY, before, changed=_NEW_JOB_CHANGES, added=3)


def test_an_accepted_run_of_a_new_quantum_corrections_row_saves_its_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    document, key = _quantum_project(window, tmp_path, status=None, target="local")
    jobs_before = set(document.jobs)
    before = _project_files(document.project_root)
    submitted = fake_local_submission(window, monkeypatch)

    window._quantum_corrections_view.submit_local_requested.emit([key])

    (job_id,) = set(document.jobs) - jobs_before
    assert [[job.id for job in jobs] for jobs in submitted] == [[job_id]]
    _assert_saved_as_defined(window, job_id, key, before, changed=_NEW_JOB_CHANGES, added=3)


@pytest.mark.parametrize("workflow", ["muon", "quantum"])
def test_an_accepted_run_again_survives_closing_without_a_save(
    main_window, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    """The run-again's own save keeps the row's job and mapping when closing says Don't save."""

    window = main_window
    if workflow == "muon":
        document = _muon_project(window, tmp_path, status="failed", target="local")
        key = _MUON_KEY
    else:
        document, key = _quantum_project(window, tmp_path, status="failed", target="local")
    stage = (
        WorkflowStage.MUON_RELAXATION if workflow == "muon" else WorkflowStage.QUANTUM_CORRECTIONS
    )
    job = _only_job(document, stage)
    fake_local_submission(window, monkeypatch)
    window._jobs._on_resubmit_from_dashboard([job.id])
    monkeypatch.setattr(
        main_window_module.MainWindow,
        "_answer_for_saving_on_close",
        lambda _self: "discard",
    )
    window.show()
    window.close()

    reopened = MainWindow()
    assert reopened.open_project(document.project_root)
    try:
        saved_job = reopened._document.jobs[job.id]
        assert saved_job.status == JobStatus.DEFINED
        if workflow == "muon":
            controller = reopened._muon_relaxation_controller
            definition = controller.active_definition()
            assert controller.job_for_label(definition, "Site 1", "positive_muon") is saved_job
        else:
            controller = reopened._quantum_corrections_controller
            definition = controller.active_definition()
            assert controller.phonon_job_for_key(definition, key) is saved_job
        assert controller.build_states()[key].status == "ready"
    finally:
        reopened.close()

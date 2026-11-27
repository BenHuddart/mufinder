"""Muon Relaxations and Quantum Corrections create their jobs only once a submission is accepted.

A Run, or the Job Dashboard's ``Resubmit…``, plans the new rows' jobs without
adding them to the project or writing their folders. A submission cancelled at
its dialog, or refused at any later point before the batch starts, leaves the
project folder byte for byte, the in-memory project and the Journal as they
were. An accepted submission writes the same files, project and cluster
targets as creating the jobs before the dialog did. Each project is staged,
saved and reopened as File > Open leaves it first.

The last tests hold for Pristine Relaxation too: a write that fails once a
submission is accepted keeps a start-again job's previous output.
"""

from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace

import pytest

import mufinder.app.jobs_controller as jobs_controller_module
from mufinder.app.main_window import MainWindow
from mufinder.app.workflow_controllers import (
    QuantumCorrectionsController,
)
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.services.project_service import ProjectService, UnsavedChanges
from tests.support.jobs import set_job_state
from tests.support.main_window import (
    fake_local_submission,
    open_project_with_host,
    stage_muon_relaxation_row,
    stage_quantum_row,
)
from tests.support.submission_snapshot import (
    accepted_summary,
    expected_accepted,
    ProjectSnapshot,
    accept_cluster_dialog,
    cancel_submission,
    quiet_messages,
    reach_the_remote_check,
    reopen_saved,
)

_STAGES = {
    "muon": WorkflowStage.MUON_RELAXATION,
    "quantum": WorkflowStage.QUANTUM_CORRECTIONS,
    "pristine": WorkflowStage.PRISTINE_RELAXATION,
}


def _controller(window: MainWindow, workflow: str):
    return {
        "muon": window._muon_relaxation_controller,
        "quantum": window._quantum_corrections_controller,
        "pristine": window._pristine_relaxation_controller,
    }[workflow]


def _view(window: MainWindow, workflow: str):
    return {
        "muon": window._muon_relaxation_view,
        "quantum": window._quantum_corrections_view,
        "pristine": window._pristine_relaxation_view,
    }[workflow]


def _jobs(window: MainWindow, workflow: str) -> list:
    return [
        job for job in window._document.jobs.values() if job.workflow_stage == _STAGES[workflow]
    ]


def _staged(window: MainWindow, tmp_path: Path, workflow: str, *, rows: int = 2) -> list[str]:
    """Stage ``rows`` ready rows of ``workflow`` on a new project in ``window``; return their keys."""

    service = ProjectService()
    document = service.create_project(tmp_path / f"{workflow}.mufinder", workflow)
    window._document = document
    if workflow == "muon":
        labels = [f"Site {index + 1}" for index in range(rows)]
        stage_muon_relaxation_row(
            window, service, document, rows=[(label, "positive_muon") for label in labels]
        )
        window._refresh_ui_after_project_change()
        return [f"positivemuon:{label}" for label in labels]
    if workflow == "quantum":
        keys = [
            stage_quantum_row(window, service, document, label=f"site{chr(65 + index)}")[1].key
            for index in range(rows)
        ]
        window._refresh_ui_after_project_change()
        return keys
    open_project_with_host(window, tmp_path)
    view = window._pristine_relaxation_view
    for _ in range(rows):
        view._add_row()
    with pytest.MonkeyPatch.context() as patch:
        cancel_submission(window, patch, "local")
        view._run_local_button.click()
    return [f"relax_{index + 1}" for index in range(rows)]


def _project(window: MainWindow, tmp_path: Path, workflow: str, *, ended_target: str = "local"):
    """A saved, reopened project: the first row ready, the second's job failed."""

    ready, ended = _staged(window, tmp_path, workflow)
    (job,) = _controller(window, workflow).materialize([ended])
    set_job_state(job, "failed")
    job.execution_target = ended_target
    reopen_saved(window, window._document)
    return ready, ended


def _run(window: MainWindow, workflow: str, keys: list[str], target: str) -> None:
    """Press Run on this computer or on a cluster for ``keys``, as the page does."""

    view = _view(window, workflow)
    signal = view.submit_local_requested if target == "local" else view.submit_cluster_requested
    signal.emit(list(keys))


def _assert_unchanged(before: ProjectSnapshot, window: MainWindow) -> None:
    after = ProjectSnapshot.take(window)
    assert after.files == before.files
    assert after.model == before.model
    assert after.journal == before.journal
    assert after.unsaved is before.unsaved is UnsavedChanges.NONE


_ROWS = {"ready": [0], "ended": [1], "mixed": [0, 1]}
_WORKFLOWS = ["muon", "quantum"]


# ---------------------------------------------------------------------------
# Cancelled at the dialog: nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", list(_ROWS))
@pytest.mark.parametrize("target", ["local", "alpha"])
@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_a_run_cancelled_at_its_dialog_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow, target, rows
) -> None:
    window = main_window
    keys = _project(window, tmp_path, workflow)
    before = ProjectSnapshot.take(window)
    opened = cancel_submission(window, monkeypatch, target)

    _run(window, workflow, [keys[index] for index in _ROWS[rows]], target)

    assert opened == ["local" if target == "local" else "cluster"]
    _assert_unchanged(before, window)
    assert _controller(window, workflow).planned_jobs() == ()


@pytest.mark.parametrize("target", ["local", "alpha"])
@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_a_resubmit_cancelled_at_its_dialog_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow, target
) -> None:
    window = main_window
    _project(window, tmp_path, workflow, ended_target=target)
    (job,) = _jobs(window, workflow)
    before = ProjectSnapshot.take(window)
    opened = cancel_submission(window, monkeypatch, target)

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert opened == ["local" if target == "local" else "cluster"]
    _assert_unchanged(before, window)


@pytest.mark.parametrize("target", ["local", "alpha"])
def test_an_edited_quantum_corrections_row_cancelled_at_its_dialog_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """A completed row whose parameters changed runs again through the run-again plan."""

    window = main_window
    ready, completed_key = _project(window, tmp_path, "quantum")
    controller = window._quantum_corrections_controller
    (job,) = _jobs(window, "quantum")
    set_job_state(job, "completed")
    reopen_saved(window, window._document)
    monkeypatch.setattr(
        QuantumCorrectionsController,
        "completed_row_is_edited",
        lambda _self, _definition, key, _job: key == completed_key,
    )
    assert controller.build_states()[completed_key].status == "edited"
    before = ProjectSnapshot.take(window)
    opened = cancel_submission(window, monkeypatch, target)

    _run(window, "quantum", [ready, completed_key], target)

    assert opened == ["local" if target == "local" else "cluster"]
    _assert_unchanged(before, window)
    assert job.status == JobStatus.COMPLETED


# ---------------------------------------------------------------------------
# Refused after the dialog: nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", list(_ROWS))
@pytest.mark.parametrize("refusal", ["no credentials", "connection failed"])
@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_a_cluster_run_refused_before_its_batch_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow, refusal, rows
) -> None:
    window = main_window
    keys = _project(window, tmp_path, workflow, ended_target="alpha")
    before = ProjectSnapshot.take(window)
    shown = quiet_messages(monkeypatch)
    accept_cluster_dialog(window, monkeypatch, credentials=refusal != "no credentials")
    if refusal == "connection failed":

        def refuse(*_args, **_kwargs):
            raise ConnectionError("host unreachable")

        monkeypatch.setattr(window._jobs, "_connect_to_cluster_with_trust", refuse)

    _run(window, workflow, [keys[index] for index in _ROWS[rows]], "alpha")

    assert (
        "Missing credentials" if refusal == "no credentials" else "Cluster connection failed"
    ) in shown
    _assert_unchanged(before, window)


@pytest.mark.parametrize("answer", ["no", "no_to_all"])
@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_a_cluster_run_whose_overwrite_prompts_are_declined_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow, answer
) -> None:
    window = main_window
    keys = _project(window, tmp_path, workflow, ended_target="beta")
    before = ProjectSnapshot.take(window)
    quiet_messages(monkeypatch)
    batches = reach_the_remote_check(window, monkeypatch, answer=answer)

    _run(window, workflow, keys, "alpha")

    assert batches == []
    _assert_unchanged(before, window)


@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_a_cluster_run_with_no_output_directory_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow
) -> None:
    window = main_window
    keys = _project(window, tmp_path, workflow, ended_target="beta")
    before = ProjectSnapshot.take(window)
    shown = quiet_messages(monkeypatch)
    batches = reach_the_remote_check(window, monkeypatch, answer="yes")
    planner = jobs_controller_module.plan_cluster_submission
    monkeypatch.setattr(
        jobs_controller_module,
        "plan_cluster_submission",
        lambda jobs, **kwargs: planner(jobs, **{**kwargs, "output_directory": lambda _job: None}),
    )

    _run(window, workflow, keys, "alpha")

    assert "Missing job directories" in shown
    assert batches == []
    _assert_unchanged(before, window)


@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_a_resubmit_whose_previous_output_cannot_be_cleared_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow
) -> None:
    window = main_window
    _project(window, tmp_path, workflow)
    (job,) = _jobs(window, workflow)
    before = ProjectSnapshot.take(window)
    shown = quiet_messages(monkeypatch)
    fake_local_submission(window, monkeypatch)
    monkeypatch.setattr(
        window._jobs,
        "_clear_outputs_for_fresh_start",
        lambda jobs: ([], [f"- {jobs[0].label}: could not clear its previous output"]),
    )

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert "Previous output not cleared" in shown
    _assert_unchanged(before, window)


# ---------------------------------------------------------------------------
# Several groups: a cancelled group creates nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("answers", [(True, False), (False, True)], ids=["second", "first"])
def test_a_cancelled_charge_state_group_creates_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answers
) -> None:
    """Each charge state is its own submission; only the accepted one creates its job."""

    window = main_window
    service = ProjectService()
    document = service.create_project(tmp_path / "groups.mufinder", "Groups")
    window._document = document
    stage_muon_relaxation_row(
        window, service, document, rows=[("Site 1", "positive_muon"), ("Site 2", "muonium")]
    )
    window._refresh_ui_after_project_change()
    reopen_saved(window, document)
    fake_local_submission(window, monkeypatch)
    pending = list(answers)

    class _Dialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1 if pending.pop(0) else 0

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(jobs_controller_module, "LocalSubmissionDialog", _Dialog)
    root = window._document.project_root / "files" / "muon_relaxation"

    _run(window, "muon", ["positivemuon:Site 1", "muonium:Site 2"], "local")

    assert pending == []
    created = {job.label for job in _jobs(window, "muon")}
    assert created == ({"Site 1"} if answers[0] else {"Site 2"})
    assert (root / "positivemuon").exists() is answers[0]
    assert (root / "muonium").exists() is answers[1]


# ---------------------------------------------------------------------------
# Accepted: the same as creating the jobs before the dialog
# ---------------------------------------------------------------------------


def _accepted_run(window: MainWindow, tmp_path: Path, monkeypatch, workflow: str, target: str):
    """Run ``workflow``'s rows on ``target``, accepted: what the project holds and sends then."""

    keys = _staged(window, tmp_path, workflow)
    reopen_saved(window, window._document)
    batches: list[list[dict]] = []
    if target == "local":
        fake_local_submission(window, monkeypatch)
    else:
        quiet_messages(monkeypatch)
        batches = reach_the_remote_check(window, monkeypatch, answer="yes", existing=False)
    _run(window, workflow, keys, target)
    window._project_service.save_project(window._document)
    return accepted_summary(window, batches)


@pytest.mark.parametrize("target", ["local", "alpha"])
@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_an_accepted_run_writes_its_jobs_files_journal_and_cluster_targets(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, target: str
) -> None:
    """The same as creating the jobs before the dialog wrote (stored when that path was removed)."""

    summary = _accepted_run(main_window, tmp_path, monkeypatch, workflow, target)

    assert summary == expected_accepted(f"{workflow}-{target}")
    assert len(summary["sent"]) == (0 if target == "local" else 2)


@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_two_new_rows_in_one_submission_get_folders_of_their_own(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    window = main_window
    keys = _staged(window, tmp_path, workflow)
    submitted = fake_local_submission(window, monkeypatch)

    _run(window, workflow, keys, "local")

    (jobs,) = submitted
    folders = {window._job_output_directory(job) for job in jobs}
    assert len(folders) == 2
    assert all((folder / f"{folder.name}.cell").is_file() for folder in folders)
    assert {job.id for job in jobs} <= set(window._document.jobs)


def test_a_new_quantum_corrections_rows_farm_count_reaches_the_dialog(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    (key,) = _staged(window, tmp_path, "quantum", rows=1)
    definition = window._quantum_corrections_controller.active_definition()
    definition.parameters["configuration"].setdefault("row_settings", {})[key] = {"num_farms": 2}
    fake_local_submission(window, monkeypatch, accept=False)
    shown: list[dict] = []
    dialog = jobs_controller_module.LocalSubmissionDialog
    monkeypatch.setattr(
        jobs_controller_module,
        "LocalSubmissionDialog",
        lambda **kwargs: shown.append(kwargs) or dialog(**kwargs),
    )

    _run(window, "quantum", [key], "local")

    ((label,),) = [kwargs["job_labels"] for kwargs in shown]
    assert shown[0]["farm_counts"] == {label: 2}
    assert _jobs(window, "quantum") == []


# ---------------------------------------------------------------------------
# A write that fails once accepted keeps a start-again job's previous output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", ["pristine", *_WORKFLOWS])
def test_a_write_that_fails_once_accepted_keeps_the_previous_output(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str
) -> None:
    window = main_window
    keys = _project(window, tmp_path, workflow)
    (ended,) = _jobs(window, workflow)
    folder = window._job_output_directory(ended)
    output = folder / f"{folder.name}.castep"
    output.write_text("the previous run's output\n", encoding="utf-8")
    window._project_service.save_project(window._document)
    before = ProjectSnapshot.take(window)
    shown = quiet_messages(monkeypatch)
    submitted = fake_local_submission(window, monkeypatch)
    write_text = Path.write_text

    def full_disk(path: Path, *args, **kwargs):
        if path.suffix == ".param" and path.parent != folder:
            raise OSError(errno.ENOSPC, "No space left on device")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full_disk)

    _run(window, workflow, keys, "local")

    assert submitted == []
    assert "Jobs not started" in shown
    assert output.read_text(encoding="utf-8") == "the previous run's output\n"
    _assert_unchanged(before, window)


def test_an_accepted_start_again_clears_the_previous_output_after_the_new_inputs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _project(window, tmp_path, "muon")
    (ended,) = _jobs(window, "muon")
    folder = window._job_output_directory(ended)
    (folder / f"{folder.name}.castep").write_text("old\n", encoding="utf-8")
    submitted = fake_local_submission(window, monkeypatch)

    window._jobs._on_resubmit_from_dashboard([ended.id])

    assert submitted == [[ended]]
    assert sorted(path.name for path in folder.iterdir()) == [
        f"{folder.name}.cell",
        f"{folder.name}.param",
    ]
    assert ended.status == JobStatus.DEFINED


def test_a_start_again_job_left_out_for_its_output_keeps_its_row_as_it_was(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Its inputs go back, and so does its label: the row's bookkeeping waits for the job to go."""

    window = main_window
    _project(window, tmp_path, "quantum")
    (job,) = _jobs(window, "quantum")
    job.label = "an earlier name"
    reopen_saved(window, window._document)
    (job,) = _jobs(window, "quantum")
    before = ProjectSnapshot.take(window)
    quiet_messages(monkeypatch)
    fake_local_submission(window, monkeypatch)
    monkeypatch.setattr(
        window._jobs,
        "_clear_outputs_for_fresh_start",
        lambda jobs: ([], [f"- {jobs[0].label}: could not clear its previous output"]),
    )

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert job.label == "an earlier name"
    _assert_unchanged(before, window)


def test_a_failure_after_the_inputs_are_written_takes_them_back(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A submission whose definition cannot take its sources writes nothing."""

    window = main_window
    keys = _staged(window, tmp_path, "quantum")
    reopen_saved(window, window._document)
    before = ProjectSnapshot.take(window)
    shown = quiet_messages(monkeypatch)
    submitted = fake_local_submission(window, monkeypatch)

    def refuse(*_args, **_kwargs):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(QuantumCorrectionsController, "ensure_definition", refuse)

    _run(window, "quantum", keys, "local")

    assert submitted == []
    assert "Jobs not started" in shown
    _assert_unchanged(before, window)

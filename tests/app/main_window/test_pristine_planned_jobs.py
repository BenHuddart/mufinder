"""Pristine Relaxation creates its jobs only once a submission is accepted.

A Run, or the Job Dashboard's ``Resubmit…``, plans the new rows' jobs without
adding them to the project or writing their folders. A submission cancelled at
its dialog, or refused at any later point before the batch starts, leaves the
project folder byte for byte, the in-memory project and the Journal as they
were. An accepted submission writes the same files and project as creating the
jobs before the dialog did, and a write that fails once it is accepted creates
nothing. Each project is configured on the page, saved and reopened as File >
Open leaves it first.
"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

import mufinder.app.jobs_controller as jobs_controller_module
import mufinder.app.workflow_controllers.pristine_relaxation as pristine_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.services.project_service import UnsavedChanges
from mufinder.workflow import StructureBridgeError
from tests.support.jobs import set_job_state
from tests.support.main_window import (
    fake_local_submission,
    open_project_with_host,
)
from tests.support.submission_snapshot import (
    accepted_summary,
    expected_accepted,
    accept_cluster_dialog,
    quiet_messages,
    reach_the_remote_check,
    ProjectSnapshot,
    cancel_submission,
    reopen_saved,
)

_READY = "relax_1"
_ENDED = "relax_2"


def _pristine_jobs(window: MainWindow) -> list:
    return [
        job
        for job in window._document.jobs.values()
        if job.workflow_stage == WorkflowStage.PRISTINE_RELAXATION
    ]


def _project(window: MainWindow, tmp_path: Path, monkeypatch, *, ended_target: str = "local"):
    """Two rows on a saved, reopened project: ``relax_1`` ready, ``relax_2``'s job failed."""

    open_project_with_host(window, tmp_path)
    view = window._pristine_relaxation_view
    view._add_row()
    view._add_row()
    with monkeypatch.context() as patch:
        cancel_submission(window, patch, "local")
        view._run_local_button.click()
    (job,) = window._pristine_relaxation_controller.materialize([_ENDED])
    set_job_state(job, "failed")
    job.execution_target = ended_target
    reopen_saved(window, window._document)
    return view


def _select(view, keys: list[str]) -> None:
    """Select the rows ``keys`` names; both rows is no selection, which runs them all."""

    view._table.clearSelection()
    if len(keys) == 1:
        view._table.selectRow([_READY, _ENDED].index(keys[0]))


def _assert_unchanged(before: ProjectSnapshot, window: MainWindow) -> None:
    after = ProjectSnapshot.take(window)
    assert after.files == before.files
    assert after.model == before.model
    assert after.journal == before.journal
    assert after.unsaved is before.unsaved is UnsavedChanges.NONE


# ---------------------------------------------------------------------------
# Cancelled at the dialog: nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [[_READY], [_ENDED], [_READY, _ENDED]], ids=str)
@pytest.mark.parametrize("target", ["local", "alpha"])
def test_a_run_cancelled_at_its_dialog_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[str], target: str
) -> None:
    window = main_window
    view = _project(window, tmp_path, monkeypatch)
    _select(view, rows)
    before = ProjectSnapshot.take(window)
    opened = cancel_submission(window, monkeypatch, target)

    (view._run_local_button if target == "local" else view._run_cluster_button).click()

    assert opened == ["local" if target == "local" else "cluster"]
    _assert_unchanged(before, window)
    assert window._pristine_relaxation_controller.planned_jobs() == ()


@pytest.mark.parametrize("target", ["local", "alpha"])
def test_a_resubmit_cancelled_at_its_dialog_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    window = main_window
    _project(window, tmp_path, monkeypatch, ended_target=target)
    (job,) = _pristine_jobs(window)
    before = ProjectSnapshot.take(window)
    opened = cancel_submission(window, monkeypatch, target)

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert opened == ["local" if target == "local" else "cluster"]
    _assert_unchanged(before, window)


def test_the_dialog_lists_the_planned_job_under_its_folder_name(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    view = _project(window, tmp_path, monkeypatch)
    _select(view, [_READY])
    fake_local_submission(window, monkeypatch, accept=False)
    shown: list[dict] = []
    dialog = jobs_controller_module.LocalSubmissionDialog
    monkeypatch.setattr(
        jobs_controller_module,
        "LocalSubmissionDialog",
        lambda **kwargs: shown.append(kwargs) or dialog(**kwargs),
    )

    view._run_local_button.click()

    (kwargs,) = shown
    assert kwargs["job_labels"] == ["pristine_relax_relax_1"]


# ---------------------------------------------------------------------------
# Refused after the dialog: nothing changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [[_READY], [_ENDED], [_READY, _ENDED]], ids=str)
@pytest.mark.parametrize("refusal", ["no credentials", "connection failed"])
def test_a_cluster_run_refused_before_its_batch_changes_nothing(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[str],
    refusal: str,
) -> None:
    window = main_window
    view = _project(window, tmp_path, monkeypatch, ended_target="alpha")
    _select(view, rows)
    before = ProjectSnapshot.take(window)
    shown = quiet_messages(monkeypatch)
    accept_cluster_dialog(window, monkeypatch, credentials=refusal != "no credentials")
    if refusal == "connection failed":

        def refuse(*_args, **_kwargs):
            raise ConnectionError("host unreachable")

        monkeypatch.setattr(window._jobs, "_connect_to_cluster_with_trust", refuse)

    view._run_cluster_button.click()

    assert shown == [
        "Missing credentials" if refusal == "no credentials" else "Cluster connection failed"
    ]
    _assert_unchanged(before, window)


@pytest.mark.parametrize("rows", [[_READY], [_ENDED], [_READY, _ENDED]], ids=str)
@pytest.mark.parametrize("answer", ["no", "no_to_all"])
def test_a_cluster_run_whose_overwrite_prompts_are_declined_changes_nothing(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[str],
    answer: str,
) -> None:
    window = main_window
    view = _project(window, tmp_path, monkeypatch, ended_target="beta")
    _select(view, rows)
    before = ProjectSnapshot.take(window)
    quiet_messages(monkeypatch)
    batches = reach_the_remote_check(window, monkeypatch, answer=answer)

    view._run_cluster_button.click()

    assert batches == []
    _assert_unchanged(before, window)


@pytest.mark.parametrize("rows", [[_READY], [_READY, _ENDED]], ids=str)
def test_a_cluster_run_with_no_output_directory_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: list[str]
) -> None:
    window = main_window
    view = _project(window, tmp_path, monkeypatch, ended_target="beta")
    _select(view, rows)
    before = ProjectSnapshot.take(window)
    shown = quiet_messages(monkeypatch)
    batches = reach_the_remote_check(window, monkeypatch, answer="yes")
    planner = jobs_controller_module.plan_cluster_submission
    monkeypatch.setattr(
        jobs_controller_module,
        "plan_cluster_submission",
        lambda jobs, **kwargs: planner(jobs, **{**kwargs, "output_directory": lambda _job: None}),
    )

    view._run_cluster_button.click()

    assert "Missing job directories" in shown
    assert batches == []
    _assert_unchanged(before, window)


def test_a_local_resubmit_whose_previous_output_cannot_be_cleared_changes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _project(window, tmp_path, monkeypatch)
    (job,) = _pristine_jobs(window)
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
# Accepted: the jobs are created, as before
# ---------------------------------------------------------------------------


def test_two_new_rows_in_one_submission_get_folders_of_their_own(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    open_project_with_host(window, tmp_path)
    view = window._pristine_relaxation_view
    view._add_row()
    view._add_row()
    submitted = fake_local_submission(window, monkeypatch)

    view._run_local_button.click()

    (jobs,) = submitted
    folders = {window._job_output_directory(job) for job in jobs}
    assert len(folders) == 2
    assert all((folder / f"{folder.name}.cell").is_file() for folder in folders)
    assert {job.id for job in jobs} <= set(window._document.jobs)


def test_planned_rows_wanting_one_name_are_given_distinct_folders(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planner reserves each folder it hands out, though none is on disk yet."""

    window = main_window
    _project(window, tmp_path, monkeypatch)
    controller = window._pristine_relaxation_controller
    definition = controller.active_definition()
    spec = controller.describe_rows([_READY])[2][0][1]
    root = window._document.project_root / "files" / "pristine_relaxation"

    first = controller.plan_new_job(
        spec, root, "a", calc_type="castep_pristine_relaxation", definition=definition
    )
    second = controller.plan_new_job(
        spec, root, "b", calc_type="castep_pristine_relaxation", definition=definition
    )

    assert controller.planned_job(first.id).folder != controller.planned_job(second.id).folder
    assert window._job_output_directory(second) == root / second.label
    assert window._job_file_cleanup.folder_used_by_another_job(root / first.label)
    controller.clear_planned()
    assert not (root / first.label).exists()


def _accepted_run(window: MainWindow, tmp_path: Path, monkeypatch) -> dict:
    """Two new rows, Run on this computer and accepted: what the project holds then."""

    open_project_with_host(window, tmp_path)
    view = window._pristine_relaxation_view
    view._add_row()
    view._add_row()
    fake_local_submission(window, monkeypatch)
    view._run_local_button.click()
    window._project_service.save_project(window._document)
    return accepted_summary(window)


def test_an_accepted_run_writes_its_jobs_files_and_journal(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same as creating the jobs before the dialog wrote (stored when that path was removed)."""

    assert _accepted_run(main_window, tmp_path, monkeypatch) == expected_accepted("pristine-local")


def test_an_edited_row_is_recorded_in_the_journal_only_once_its_run_is_accepted(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    view = _project(window, tmp_path, monkeypatch)
    host = window._document.structures[window._document.project.host_structure_id]
    view.set_host_lattice(host.lattice_vectors)
    view.set_row_supercell(view._row_id_at(0), ((2, 0, 0), (0, 1, 0), (0, 0, 1)))
    _select(view, [_READY])
    journal = ProjectSnapshot.take(window).journal
    cancel_submission(window, monkeypatch, "local")
    view._run_local_button.click()

    assert ProjectSnapshot.take(window).journal == journal
    assert window._project_service.unsaved_changes(window._document) is not UnsavedChanges.NONE

    # Changed again before any Run is accepted: the first configuration never ran.
    view.set_row_supercell(view._row_id_at(0), ((1, 0, 0), (0, 2, 0), (0, 0, 1)))
    view._run_local_button.click()
    fake_local_submission(window, monkeypatch)
    view._run_local_button.click()

    after = ProjectSnapshot.take(window)
    configured = [
        action for action, _text in after.journal[len(journal) :] if action.startswith("configure")
    ]
    assert configured == ["configure_pristine_relaxation"]
    active = window._pristine_relaxation_controller.active_definition()
    (record,) = [r for r in window._document.provenance if r.action.startswith("configure")][-1:]
    assert record.output_entity_ids == [active.id]


# ---------------------------------------------------------------------------
# Accepted, but the jobs cannot be written: nothing is created
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failure", ["disk full", "unreadable structure"])
def test_a_write_that_fails_once_accepted_creates_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    window = main_window
    view = _project(window, tmp_path, monkeypatch)
    view._add_row()
    with monkeypatch.context() as patch:
        cancel_submission(window, patch, "local")
        view._run_local_button.click()
    reopen_saved(window, window._document)
    view._table.clearSelection()
    before = ProjectSnapshot.take(window)
    shown = quiet_messages(monkeypatch)
    submitted = fake_local_submission(window, monkeypatch)
    dialog = jobs_controller_module.LocalSubmissionDialog

    def fail_after_planning(**kwargs):
        if failure == "disk full":
            write_text = Path.write_text

            def full(path, *args, **kw):
                if path.name.startswith("pristine_relax_relax_3"):
                    raise OSError(errno.ENOSPC, "No space left on device")
                return write_text(path, *args, **kw)

            monkeypatch.setattr(Path, "write_text", full)
        else:

            def unreadable(**_kw):
                raise StructureBridgeError("the host structure cannot be read")

            monkeypatch.setattr(pristine_module, "expand_pristine_relaxation_jobs", unreadable)
        return dialog(**kwargs)

    monkeypatch.setattr(jobs_controller_module, "LocalSubmissionDialog", fail_after_planning)

    view._run_local_button.click()

    assert submitted == []
    assert "Jobs not started" in shown
    _assert_unchanged(before, window)
    assert {job.status for job in _pristine_jobs(window)} == {JobStatus.ENDED}


def test_an_edited_ended_row_runs_in_its_earlier_jobs_folder(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The planned job of a reconfigured row shares its earlier job's folder, as before."""

    window = main_window
    view = _project(window, tmp_path, monkeypatch)
    (ended,) = _pristine_jobs(window)
    host = window._document.structures[window._document.project.host_structure_id]
    view.set_host_lattice(host.lattice_vectors)
    view.set_row_supercell(view._row_id_at(1), ((2, 0, 0), (0, 1, 0), (0, 0, 1)))
    _select(view, [_ENDED])
    shown = quiet_messages(monkeypatch)
    submitted = fake_local_submission(window, monkeypatch)

    view._run_local_button.click()

    assert "Jobs not started" not in shown
    (jobs,) = submitted
    (job,) = jobs
    assert job.id != ended.id and job.id in window._document.jobs
    assert window._job_output_directory(job) == window._job_output_directory(ended)


def test_a_configuration_stored_before_a_reopen_is_recorded_when_its_run_is_accepted(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The held record goes with the session; the accepted Run after a reopen rebuilds it."""

    first = MainWindow()
    _project(first, tmp_path, monkeypatch)
    root = first._document.project_root
    assert not any(r.action.startswith("configure") for r in first._document.provenance)

    second = MainWindow()
    assert second.open_project(root)
    fake_local_submission(second, monkeypatch)
    second._pristine_relaxation_view._run_local_button.click()

    configured = [r for r in second._document.provenance if r.action.startswith("configure")]
    active = second._pristine_relaxation_controller.active_definition()
    assert [record.output_entity_ids for record in configured] == [[active.id]]


def test_the_record_written_on_acceptance_describes_the_rows_as_they_are_then(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row removed after the Run that held the record is not in the record written."""

    window = main_window
    open_project_with_host(window, tmp_path)
    view = window._pristine_relaxation_view
    view._add_row()
    view._add_row()
    with monkeypatch.context() as patch:
        cancel_submission(window, patch, "local")
        view._run_local_button.click()
    # Removing a row edits the active definition in place.
    window._pristine_relaxation_controller.remove_rows([_ENDED])
    fake_local_submission(window, monkeypatch)

    window._pristine_relaxation_controller.submit([_READY], target="local")

    (record,) = [r for r in window._document.provenance if r.action.startswith("configure")]
    assert record.description == "Configured pristine relaxation with 1 job row"

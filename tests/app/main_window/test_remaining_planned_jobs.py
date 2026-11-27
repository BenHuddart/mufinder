"""Hyperfine Coupling, Energy Barriers, the convergence sweeps and UEP create their jobs once accepted.

A Run, or the Job Dashboard's ``Resubmit…``, plans the new rows' jobs without
adding them to the project or writing their folders. A submission cancelled at
its dialog, or refused at any later point before the batch starts, leaves the
project folder byte for byte, the in-memory project and the Journal as they
were. An accepted submission writes the same files, project and cluster
targets as creating the jobs before the dialog did, and a write that fails once
accepted keeps a start-again job's previous output. Each project is configured
or staged, saved and reopened as File > Open leaves it first.
"""

from __future__ import annotations

import errno
from pathlib import Path

import pytest

import mufinder.app.jobs_controller as jobs_controller_module
from mufinder.app.main_window import MainWindow
from mufinder.app.workflow_controllers import (
    ConvergenceCutoffController,
    ConvergenceKpointController,
    EnergyBarriersController,
    HyperfineCouplingController,
    UepController,
)
from mufinder.domain.enums import WorkflowStage
from mufinder.services.project_service import ProjectService
from tests.support.energy_barriers import seed_energy_barriers_document
from tests.support.jobs import set_job_state
from tests.support.main_window import (
    fake_local_submission,
    open_project_with_host,
    stage_quantum_row,
)
from tests.support.submission_snapshot import (
    accepted_summary,
    expected_accepted,
    ProjectSnapshot,
    accept_cluster_dialog,
    assert_unchanged,
    cancel_submission,
    quiet_messages,
    reach_the_remote_check,
    reopen_saved,
)

_WORKFLOWS = ["hyperfine", "barriers", "cutoff", "kpoint", "uep"]
_CONTROLLERS = {
    "hyperfine": HyperfineCouplingController,
    "barriers": EnergyBarriersController,
    "cutoff": ConvergenceCutoffController,
    "kpoint": ConvergenceKpointController,
    "uep": UepController,
}
_STAGES = {
    "hyperfine": WorkflowStage.HYPERFINE_COUPLING,
    "barriers": WorkflowStage.ENERGY_BARRIERS,
    "cutoff": WorkflowStage.CONVERGENCE_CUTOFF,
    "kpoint": WorkflowStage.CONVERGENCE_KPOINT,
    "uep": WorkflowStage.UEP,
}


def _controller(window: MainWindow, workflow: str):
    return {
        "hyperfine": window._hyperfine_coupling_controller,
        "barriers": window._energy_barriers_controller,
        "cutoff": window._convergence_controller,
        "kpoint": window._kpoint_convergence_controller,
        "uep": window._uep_controller,
    }[workflow]


def _jobs(window: MainWindow, workflow: str) -> list:
    return [
        job for job in window._document.jobs.values() if job.workflow_stage == _STAGES[workflow]
    ]


def _run(window: MainWindow, workflow: str, keys: list, target: str) -> None:
    """Press Run on this computer or on a cluster for ``keys``, as the page does."""

    if workflow == "barriers":
        view = window._ensure_energy_barriers_view()
        signal = (
            view.submit_neb_local_requested
            if target == "local"
            else view.submit_neb_cluster_requested
        )
    else:
        view = {
            "hyperfine": window._hyperfine_coupling_view,
            "cutoff": window._convergence_view,
            "kpoint": window._kpoint_convergence_view,
            "uep": window._uep_view,
        }[workflow]
        signal = view.submit_local_requested if target == "local" else view.submit_cluster_requested
    signal.emit(list(keys))


def _staged(window: MainWindow, tmp_path: Path, workflow: str) -> list:
    """Stage two ready rows of ``workflow`` on a new project in ``window``; return their keys."""

    if workflow == "hyperfine":
        service = ProjectService()
        document = service.create_project(tmp_path / "hyperfine.mufinder", "Hyperfine")
        window._document = document
        keys = [
            stage_quantum_row(window, service, document, label=label)[1].key
            for label in ("siteA", "siteB")
        ]
        window._refresh_ui_after_project_change()
        return keys
    if workflow == "barriers":
        document, _definition, _source, _root = seed_energy_barriers_document(
            ProjectService(), tmp_path, paths=2
        )
        window._document = document
        window._refresh_ui_after_project_change()
        window._ensure_energy_barriers_view()
        return ["source-key", "source-key-2"]
    open_project_with_host(window, tmp_path)
    view, keys = {
        "cutoff": (window._convergence_view, [300.0, 360.0]),
        "kpoint": (window._kpoint_convergence_view, [(2, 2, 2), (3, 3, 3)]),
        "uep": (window._uep_view, None),
    }[workflow]
    if keys is None:
        view._add_row()
        view._add_row()
    else:
        for key in keys:
            view._add_row(key)
    # A Run cancelled at its dialog stores the page's configuration.
    with pytest.MonkeyPatch.context() as patch:
        cancel_submission(window, patch, "local")
        view._run_local_button.click()
    if keys is None:
        rows = _controller(window, workflow).active_definition().parameters["configuration"]["rows"]
        keys = [row["label"] for row in rows]
    return keys


def _project(window: MainWindow, tmp_path: Path, workflow: str, *, ended_target: str = "local"):
    """A saved, reopened project: the first row ready, the second's job failed."""

    ready, ended = _staged(window, tmp_path, workflow)
    (job,) = _controller(window, workflow).materialize([ended])
    set_job_state(job, "failed")
    job.execution_target = ended_target
    reopen_saved(window, window._document)
    return ready, ended


_ROWS = {"ready": [0], "ended": [1], "mixed": [0, 1]}


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
    assert_unchanged(before, window)
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
    assert_unchanged(before, window)


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
    assert_unchanged(before, window)


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
    assert_unchanged(before, window)


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
    assert_unchanged(before, window)


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
    assert_unchanged(before, window)


# ---------------------------------------------------------------------------
# Accepted: the same as creating the jobs before the dialog
# ---------------------------------------------------------------------------


def _accepted_run(window: MainWindow, tmp_path: Path, monkeypatch, workflow: str, target: str):
    """Run ``workflow``'s rows on ``target``, accepted: what the project holds and sends then."""

    keys = _project(window, tmp_path, workflow)
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
    reopen_saved(window, window._document)
    submitted = fake_local_submission(window, monkeypatch)

    _run(window, workflow, keys, "local")

    (jobs,) = submitted
    folders = {window._job_output_directory(job) for job in jobs}
    assert len(folders) == 2
    assert all((folder / f"{folder.name}.cell").is_file() for folder in folders)
    assert {job.id for job in jobs} <= set(window._document.jobs)


@pytest.mark.parametrize("target", ["local", "alpha"])
def test_a_new_energy_barriers_paths_farm_count_reaches_the_dialog(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    window = main_window
    keys = _staged(window, tmp_path, "barriers")
    definition = window._energy_barriers_controller.active_definition()
    definition.parameters["configuration"]["row_settings"] = {keys[0]: {"num_farms": 3}}
    reopen_saved(window, window._document)
    shown: list[dict] = []
    if target == "local":
        fake_local_submission(window, monkeypatch, accept=False)
        dialog = jobs_controller_module.LocalSubmissionDialog
        monkeypatch.setattr(
            jobs_controller_module,
            "LocalSubmissionDialog",
            lambda **kwargs: shown.append(kwargs) or dialog(**kwargs),
        )
    else:
        cancel_submission(window, monkeypatch, "alpha")
        dialog = jobs_controller_module.ClusterSubmissionDialog
        monkeypatch.setattr(
            jobs_controller_module,
            "ClusterSubmissionDialog",
            lambda **kwargs: shown.append(kwargs) or dialog(**kwargs),
        )

    _run(window, "barriers", [keys[0]], target)

    ((label,),) = [kwargs["job_labels"] for kwargs in shown]
    assert shown[0]["farm_counts"] == {label: 3}
    assert _jobs(window, "barriers") == []


# ---------------------------------------------------------------------------
# A write that fails once accepted keeps a start-again job's previous output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", _WORKFLOWS)
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
    assert_unchanged(before, window)


def _renamed_hyperfine_run(window: MainWindow, tmp_path: Path, monkeypatch, target: str):
    """A defined Hyperfine job whose source's label changed: Run it, accepted."""

    ready, _other = _staged(window, tmp_path, "hyperfine")
    (job,) = window._hyperfine_coupling_controller.materialize([ready])
    old = window._job_output_directory(job)
    stale = old.parent / "hyperfine_stale"
    old.rename(stale)
    for path in list(stale.iterdir()):
        path.rename(stale / f"hyperfine_stale{path.suffix}")
    job.label = "hyperfine_stale"
    reopen_saved(window, window._document)
    batches: list[list[dict]] = []
    submitted: list = []
    if target == "local":
        submitted = fake_local_submission(window, monkeypatch)
    else:
        quiet_messages(monkeypatch)
        batches = reach_the_remote_check(window, monkeypatch, answer="yes", existing=False)
    _run(window, "hyperfine", [ready], target)
    seeds = [window._job_seed_name(job) for jobs in submitted for job in jobs]
    return {**accepted_summary(window, batches), "local_seeds": seeds}


@pytest.mark.parametrize("target", ["local", "alpha"])
def test_a_defined_hyperfine_job_follows_its_sources_label_to_the_folder_it_runs_in(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """The dialogs and the cluster plan see the folder the job runs in (stored expectation)."""

    summary = _renamed_hyperfine_run(main_window, tmp_path, monkeypatch, target)

    assert summary == expected_accepted(f"hyperfine-renamed-{target}")
    assert "hyperfine_stale" not in {label for label, _status in summary["jobs"]}


def test_a_planned_move_reserves_the_folder_it_moves_to(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new row whose seed is a moving job's destination gets a folder of its own."""

    window = main_window
    keys = _staged(window, tmp_path, "hyperfine")
    controller = window._hyperfine_coupling_controller
    (job,) = controller.materialize([keys[0]])
    _definition, root, rows = controller.describe_rows([keys[1]])
    ((key, spec, _existing),) = rows

    controller.plan_move(job, root / spec.name)
    planned = controller.plan_new_job(
        spec, root, key, calc_type=controller.planned_calc_type, definition=None
    )

    assert controller.planned_job(planned.id).folder != root / spec.name
    assert window._job_output_directory(job) == root / spec.name
    controller.clear_planned()
    assert window._job_output_directory(job) != root / spec.name
